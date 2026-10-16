"""Parity-gated comparison of sequential/static/hybrid decoder paths."""

import argparse
import gc
import json
import os
import platform
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import transformers

from scripts.evaluate_real_speech import RealSpeechBenchmark, generate_burst_traffic, generate_poisson_traffic
from scripts.gpu_telemetry import (
    CpuTelemetryMonitor,
    GpuTelemetryMonitor,
    evaluate_run_gate,
    evaluate_session_gate,
    get_max_boost_clock,
)
from scripts.run_interleaved_benchmark_graph import cooldown_gpu
from scripts.test_parity_graph import hf_reference_wer

try:
    import triton
    TRITON_VERSION = triton.__version__
except Exception:
    TRITON_VERSION = "not installed"


SYSTEMS: List[Tuple[str, Dict[str, Any]]] = [
    ("sequential", {"kind": "sequential", "decode": "eager", "readback_k": 1}),
    ("static_batching", {"kind": "static", "decode": "eager", "graph_decode": False, "readback_k": 1}),
    ("static_graph", {"kind": "static", "decode": "cuda_graph", "graph_decode": True, "readback_k": 4}),
    ("hybrid_before", {
        "kind": "engine", "decode": "eager", "cache_cross_kv": False,
        "graph_decode": False, "readback_k": 1,
    }),
    ("hybrid_after", {
        "kind": "engine", "decode": "cuda_graph", "cache_cross_kv": True,
        "graph_decode": True, "readback_k": 4,
    }),
]


def run_system(bench: RealSpeechBenchmark, traffic: List[Dict[str, Any]], cfg: Dict[str, Any]) -> Dict[str, Any]:
    if cfg["kind"] == "sequential":
        return bench.run_sequential(traffic)
    if cfg["kind"] == "static":
        return bench.run_static_batching(
            traffic, batch_size=4, graph_decode=cfg["graph_decode"],
            graph_readback_k=cfg["readback_k"],
        )
    return bench.run_engine(
        traffic,
        max_batch_size=4,
        retain_encoder_cache=False,
        cache_cross_kv=cfg["cache_cross_kv"],
        batch_encoder=False,
        no_sync_decode=False,
        avoid_redundant_cross_kv=False,
        graph_decode=cfg["graph_decode"],
        graph_readback_k=cfg["readback_k"],
        graph_non_blocking_copies=True,
        graph_zero_slot_buffers_on_release=False,
        graph_force_bucket=None,
    )


def runtime_metadata() -> Dict[str, str]:
    try:
        driver = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        driver = "unknown"
    return {
        "os": platform.platform(), "driver": driver, "python": sys.version.split()[0],
        "torch": torch.__version__, "transformers": transformers.__version__, "triton": TRITON_VERSION,
    }


def summarize(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    def med(key: str) -> float:
        return float(np.median([r[key] for r in runs]))

    exact = med("exact_match_pct")
    return {
        "status": "VALID" if exact >= 99.0 else "INVALID",
        "wall_time_s_median": med("wall_time_s"),
        "service_capacity_req_per_sec_median": med("service_capacity_req_per_sec"),
        "throughput_audio_per_sec_median": med("throughput_audio_per_sec"),
        "ttft_p50_s_median": med("ttft_p50_s"),
        "ttft_p95_s_median": med("ttft_p95_s"),
        "latency_p50_s_median": med("latency_p50_s"),
        "latency_p95_s_median": med("latency_p95_s"),
        "peak_vram_allocated_mb_median": med("peak_vram_allocated_mb"),
        "wer_pct": med("wer_pct"),
        "exact_match_pct": exact,
        "busy_median_clock_mhz": float(np.median([r["telemetry_gpu"]["busy_median_clock_mhz"] for r in runs])),
        "busy_throttle_fraction": float(max(r["telemetry_gpu"]["busy_throttle_sample_pct"] for r in runs) / 100.0),
        "runs": runs,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "data" / "hybrid_before_after_burst.json")
    parser.add_argument("--matched-only", action="store_true", help="Measure only static eager/graph and hybrid before/after")
    parser.add_argument("--include-poisson-high", action="store_true", help="Also run 100 requests at Poisson rate 4.0 req/s")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    session_id = f"burst_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    systems = [(name, cfg) for name, cfg in SYSTEMS if not args.matched_only or name != "sequential"]
    bench = RealSpeechBenchmark(device="cuda", dtype=torch.float16)
    traffic_burst = generate_burst_traffic(100)
    profiles = {"burst": traffic_burst}
    if args.include_poisson_high:
        profiles["poisson_high_4_0"] = generate_poisson_traffic(100, rate=4.0, seed=42)
    canary_n = min(20, len(bench.samples))
    canary_traffic = generate_burst_traffic(canary_n)
    reference_wer = hf_reference_wer(bench)

    cooldown_gpu(target_temp=62.0, max_wait_s=45)
    print(f"\n=== Sequential canary START ({canary_n}) ===")
    canary_start = bench.run_sequential(canary_traffic)

    print("\n=== 100-utterance parity gate ===")
    parity: Dict[str, Any] = {}
    for name, cfg in systems:
        result = run_system(bench, traffic_burst, cfg)
        exact = result["token_match_rate_pct"]
        delta = result["wer_pct"] - reference_wer
        passed = exact >= 99.0 and abs(delta) <= 0.2
        parity[name] = {
            "status": "VALID" if passed else "INVALID", "passed": passed,
            "exact_match_pct": exact, "wer_pct": result["wer_pct"],
            "reference_wer_pct": reference_wer, "wer_delta_points": delta,
        }
        print(f"{name}: exact={exact:.1f}% | WER={result['wer_pct']:.3f}% | delta={delta:+.3f} | {parity[name]['status']}")

    invalid = [name for name, gate in parity.items() if not gate["passed"]]
    if invalid:
        payload = {
            "session_id": session_id, "status": "PARITY_FAILED", "invalid_systems": invalid,
            "systems": dict(systems), "parity_gate": parity, "speed_rows": {},
        }
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8", newline="\n")
        raise SystemExit("Refusing benchmark; parity-invalid: " + ", ".join(invalid))

    all_gpu_summaries: List[Dict[str, Any]] = []
    summaries_by_profile: Dict[str, Any] = {}
    for profile_index, (profile_name, traffic) in enumerate(profiles.items()):
        print(f"\n=== {profile_name}: discarded interleaved warm-up ===")
        for name, cfg in systems:
            print("warming", name)
            run_system(bench, traffic, cfg)

        runs_by_system: Dict[str, List[Dict[str, Any]]] = {name: [] for name, _ in systems}
        for repeat in range(args.repeats):
            offset = (repeat + profile_index * 2) % len(systems)
            order = systems[offset:] + systems[:offset]
            print(f"\n=== {profile_name} repeat {repeat + 1}/{args.repeats}: {[n for n, _ in order]} ===")
            for order_index, (name, cfg) in enumerate(order):
                gc.collect()
                torch.cuda.empty_cache()
                cooldown_gpu(target_temp=62.0, max_wait_s=45)
                gpu_mon = GpuTelemetryMonitor(interval_sec=0.5)
                cpu_mon = CpuTelemetryMonitor(interval_sec=1)
                gpu_mon.start()
                cpu_mon.start()
                result = run_system(bench, traffic, cfg)
                gpu = gpu_mon.stop()
                cpu = cpu_mon.stop()
                run_gate, failures = evaluate_run_gate(gpu, max_temp_threshold_c=88.0)
                all_gpu_summaries.append(gpu)
                exact = result["token_match_rate_pct"]
                status = "VALID" if exact >= 99.0 else "INVALID"
                speed_row = {
                    "profile": profile_name, "repeat": repeat + 1, "order_index": order_index + 1, "status": status,
                    "wall_time_s": result["wall_time_s"],
                    "service_capacity_req_per_sec": 100.0 / result["wall_time_s"],
                    "throughput_audio_per_sec": result["throughput_audio_per_sec"],
                    "ttft_p50_s": result["ttft_p50_s"], "ttft_p95_s": result["ttft_p95_s"],
                    "latency_p50_s": result["latency_p50_s"], "latency_p95_s": result["latency_p95_s"],
                    "peak_vram_allocated_mb": result["peak_vram_allocated_mb"],
                    "exact_match_pct": exact, "wer_pct": result["wer_pct"],
                    "telemetry_gpu": gpu, "telemetry_cpu": cpu,
                    "run_gate_passed": run_gate, "run_gate_failures": failures,
                }
                runs_by_system[name].append(speed_row)
                print(
                    f"{name}: capacity={speed_row['service_capacity_req_per_sec']:.3f} req/s | "
                    f"p95={speed_row['latency_p95_s']:.3f}s | exact={exact:.1f}% | "
                    f"WER={result['wer_pct']:.3f}% | {status}"
                )
        summaries_by_profile[profile_name] = {
            "systems": {name: summarize(runs_by_system[name]) for name, _ in systems}
        }

    cooldown_gpu(target_temp=62.0, max_wait_s=45)
    print(f"\n=== Sequential canary END ({canary_n}) ===")
    canary_end = bench.run_sequential(canary_traffic)
    start_tp = canary_start["throughput_audio_per_sec"]
    end_tp = canary_end["throughput_audio_per_sec"]
    canary_drift = abs(end_tp - start_tp) / start_tp * 100.0 if start_tp else 0.0

    system_clocks = {
        name: float(np.median([
            profile["systems"][name]["busy_median_clock_mhz"]
            for profile in summaries_by_profile.values()
        ]))
        for name, _ in systems
    }
    gate_passed, gate_details = evaluate_session_gate(
        system_median_busy_clocks=system_clocks,
        all_runs_summaries=all_gpu_summaries,
        canary_drift_pct=canary_drift,
        max_temp_threshold_c=88.0,
    )
    payload = {
        "session_id": session_id, "timestamp": datetime.now().isoformat(),
        "status": "VALID" if gate_passed else "TELEMETRY_INVALID",
        "environment": runtime_metadata(),
        "methodology": {
            "profiles": list(profiles), "poisson_rate_req_per_s": 4.0 if args.include_poisson_high else None,
            "utterances_per_profile": 100, "repeats": args.repeats,
            "batch_size": 4, "order": "cyclic rotation", "discarded_warmup_cycles_per_profile": 1,
            "graph_buckets": [64, 128, 256, 448], "graph_readback_k": 4,
        },
        "system_configs": dict(systems), "parity_gate": parity,
        "canary": {
            "utterances": canary_n, "start_throughput_audio_per_sec": start_tp,
            "end_throughput_audio_per_sec": end_tp, "drift_pct": canary_drift,
            "passed": canary_drift <= 15.0,
        },
        "session_gate": gate_details,
        "profiles": summaries_by_profile,
        "speed_rows": summaries_by_profile["burst"]["systems"],
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8", newline="\n")

    for profile_name, profile in summaries_by_profile.items():
        print(f"\n=== {profile_name} median rows ===")
        for name, row in profile["systems"].items():
            print(
                f"{name}: {row['service_capacity_req_per_sec_median']:.3f} req/s | "
                f"p95={row['latency_p95_s_median']:.3f}s | exact={row['exact_match_pct']:.1f}% | "
                f"WER={row['wer_pct']:.3f}% | {row['status']}"
            )
    print(
        f"Session gate: {'PASS' if gate_passed else 'FAIL'} | "
        f"busy throttle={gate_details['max_busy_throttle_pct']:.1f}% | "
        f"clock spread={gate_details['clock_spread_pct']:.1f}% | canary drift={canary_drift:.1f}%"
    )
    print("Saved", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
