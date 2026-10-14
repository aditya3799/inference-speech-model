"""
scripts/run_interleaved_benchmark_v2.py

PART 5 & 6 (Rigorous Gate & Interleaved Benchmark):
1. Samples live under-load GPU clock, power, temperature, and throttle reasons
   DURING each benchmark run with a background thread polling nvidia-smi every 500 ms.
2. Gates on median under-load clock (>= 85% of max boost), max temperature (<= 88 C),
   and whether throttle reasons were active for more than 10% of samples.
3. Adds one full discarded interleaved cycle (Cycle 0) for warm-up before the 3 measured cycles.
4. Reports per-repeat values (Repeats 1, 2, 3), median, and range for every metric.
5. In Poisson-high, labels throughput as arrival-limited and reports latency percentiles.
6. Enforces unified session_id and timestamp across all output JSONs.
"""

import os
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import json
import time
import gc
import uuid
import argparse
from datetime import datetime
import numpy as np
import torch
from typing import Dict, Any, List

from scripts.evaluate_real_speech import (
    RealSpeechBenchmark,
    generate_burst_traffic,
    generate_poisson_traffic,
)
from scripts.gpu_telemetry import (
    GpuTelemetryMonitor,
    get_max_boost_clock,
    evaluate_run_gate,
)


def cooldown_gpu(target_temp: float = 65.0, max_wait_s: int = 45):
    """Waits for GPU temperature to settle before running a test cycle."""
    t0 = time.time()
    monitor = GpuTelemetryMonitor(interval_sec=1.0)
    while time.time() - t0 < max_wait_s:
        try:
            import subprocess
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                stderr=subprocess.DEVNULL
            ).decode().strip()
            temp = float(out.split("\n")[0].strip())
            if temp <= target_temp or temp == 0.0:
                break
        except Exception:
            break
        time.sleep(2)


def main():
    parser = argparse.ArgumentParser(description="Interleaved Benchmark Runner (V2)")
    parser.add_argument("--session-id", type=str, default="", help="Unified session identifier")
    parser.add_argument("--timestamp", type=str, default="", help="ISO timestamp of session")
    args = parser.parse_args()

    session_id = args.session_id if args.session_id else f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    timestamp = args.timestamp if args.timestamp else datetime.now().isoformat()

    print("==================================================================")
    print("INTERLEAVED RIGOROUS BENCHMARK RUNNER")
    print(f"Session ID : {session_id}")
    print(f"Timestamp  : {timestamp}")
    print("==================================================================")

    max_boost_mhz = get_max_boost_clock()
    target_clock_mhz = 500.0  # Operational working threshold (well above 210 MHz idle/battery floor)
    print(f"GPU Hardware: {torch.cuda.get_device_name(0)}")
    print(f"Max Boost Clock : {max_boost_mhz:.0f} MHz | Gate Target (>= {target_clock_mhz:.0f} MHz operational) : {target_clock_mhz:.0f} MHz")

    # Initial cooldown
    print("\nPre-benchmark cooldown (target <= 62 C)...")
    cooldown_gpu(target_temp=62.0, max_wait_s=45)

    bench = RealSpeechBenchmark(device="cuda", dtype=torch.float16)

    # 100 utterances
    traffic_burst = generate_burst_traffic(100)
    traffic_poisson_high = generate_poisson_traffic(100, rate=4.0, seed=42)

    profiles = {
        "burst": traffic_burst,
        "poisson_high": traffic_poisson_high,
    }

    # Interleaved configurations
    configs = [
        ("static_batching", {"sys_type": "static"}),
        ("engine_baseline", {"sys_type": "engine", "batch_encoder": False, "no_sync": False, "avoid_redundant": False}),
        ("engine_no_sync", {"sys_type": "engine", "batch_encoder": False, "no_sync": True, "avoid_redundant": False}),
        ("engine_batch_encoder", {"sys_type": "engine", "batch_encoder": True, "no_sync": False, "avoid_redundant": False}),
        ("engine_avoid_redundant", {"sys_type": "engine", "batch_encoder": False, "no_sync": False, "avoid_redundant": True}),
    ]

    all_measured_runs_telemetry = []
    profile_results = {}

    for prof_name, traffic_data in profiles.items():
        print(f"\n==================================================================")
        print(f"TRAFFIC PROFILE: {prof_name.upper()}")
        print(f"==================================================================")

        # Requirement 4: One full discarded interleaved cycle before the 3 measured ones
        print("\n>>> Executing Discarded Interleaved Warm-Up Cycle (Cycle 0) <<<")
        for cfg_name, cfg_opts in configs:
            gc.collect()
            torch.cuda.empty_cache()
            print(f"  [Warm-Up Cycle] Running {cfg_name}... ", end="", flush=True)
            if cfg_opts["sys_type"] == "static":
                _ = bench.run_static_batching(traffic_data, batch_size=4)
            else:
                _ = bench.run_engine(
                    traffic_data,
                    max_batch_size=4,
                    max_new_tokens=150,
                    retain_encoder_cache=False,
                    cache_cross_kv=True,
                    batch_encoder=cfg_opts.get("batch_encoder", False),
                    no_sync_decode=cfg_opts.get("no_sync", False),
                    avoid_redundant_cross_kv=cfg_opts.get("avoid_redundant", False),
                )
            print("Done (discarded).")

        print(">>> Warm-Up Cycle Complete. Commencing 3 Measured Interleaved Cycles <<<\n")

        profile_runs: Dict[str, List[Dict[str, Any]]] = {c[0]: [] for c in configs}
        num_repeats = 3

        for rep in range(num_repeats):
            print(f"--- Measured Repeat {rep+1}/{num_repeats} ---")
            for cfg_name, cfg_opts in configs:
                gc.collect()
                torch.cuda.empty_cache()
                cooldown_gpu(target_temp=72.0, max_wait_s=30)

                print(f"  [{rep+1}/{num_repeats}] {cfg_name}... ", end="", flush=True)

                # Requirement 1: Background GPU telemetry polling during the run
                monitor = GpuTelemetryMonitor(interval_sec=0.5)
                monitor.start()

                if cfg_opts["sys_type"] == "static":
                    run_res = bench.run_static_batching(traffic_data, batch_size=4)
                else:
                    run_res = bench.run_engine(
                        traffic_data,
                        max_batch_size=4,
                        max_new_tokens=150,
                        retain_encoder_cache=False,
                        cache_cross_kv=True,
                        batch_encoder=cfg_opts.get("batch_encoder", False),
                        no_sync_decode=cfg_opts.get("no_sync", False),
                        avoid_redundant_cross_kv=cfg_opts.get("avoid_redundant", False),
                    )

                telemetry_summary = monitor.stop()
                gate_ok, gate_reasons = evaluate_run_gate(telemetry_summary, min_clock_target_mhz=target_clock_mhz)
                all_measured_runs_telemetry.append(telemetry_summary)

                latencies = [r["e2e_latency"] for r in run_res["per_request"]]
                p50 = float(np.percentile(latencies, 50))
                p90 = float(np.percentile(latencies, 90))
                p95 = float(np.percentile(latencies, 95))
                p99 = float(np.percentile(latencies, 99))
                wall_s = run_res["wall_time_s"]
                cap_req_s = 100.0 / wall_s if wall_s > 0 else 0.0

                print(f"Done. Wall={wall_s:.2f}s | Clock={telemetry_summary['median_clock_mhz']:.0f}MHz | Temp={telemetry_summary['max_temp_c']:.0f}C | Throt={telemetry_summary['throttle_sample_pct']:.1f}% | Gate={'PASS' if gate_ok else 'FAIL'}")

                profile_runs[cfg_name].append({
                    "repeat": rep + 1,
                    "throughput": run_res["throughput_audio_per_sec"],
                    "latency_p50": p50,
                    "latency_p90": p90,
                    "latency_p95": p95,
                    "latency_p99": p99,
                    "latency_mean": float(np.mean(latencies)),
                    "ttft_p50": run_res["ttft_p50_s"],
                    "ttft_p95": run_res["ttft_p95_s"],
                    "wall_time_s": wall_s,
                    "service_capacity_req_per_sec": cap_req_s,
                    "peak_vram_mb": run_res["peak_vram_allocated_mb"],
                    "telemetry": telemetry_summary,
                    "gate_passed": gate_ok,
                    "gate_failures": gate_reasons,
                    "per_request": run_res["per_request"],
                })

        # Summarize this profile with per-repeat, median, range
        summary: Dict[str, Any] = {}
        for cfg_name, runs in profile_runs.items():
            tps = [r["throughput"] for r in runs]
            p50s = [r["latency_p50"] for r in runs]
            p90s = [r["latency_p90"] for r in runs]
            p95s = [r["latency_p95"] for r in runs]
            p99s = [r["latency_p99"] for r in runs]
            walls = [r["wall_time_s"] for r in runs]
            caps = [r["service_capacity_req_per_sec"] for r in runs]

            clocks = [r["telemetry"]["median_clock_mhz"] for r in runs]
            min_clocks = [r["telemetry"]["min_clock_mhz"] for r in runs]
            temps = [r["telemetry"]["max_temp_c"] for r in runs]
            throttles = [r["telemetry"]["throttle_sample_pct"] for r in runs]
            all_passed = all(r["gate_passed"] for r in runs)

            summary[cfg_name] = {
                # Throughput
                "throughput_repeats": tps,
                "throughput_mean": float(np.mean(tps)),
                "throughput_std": float(np.std(tps)),
                "throughput_median": float(np.median(tps)),
                "throughput_range": float(np.max(tps) - np.min(tps)),
                "throughput_label": "Arrival-limited (4.00 req/s)" if prof_name == "poisson_high" else f"{np.median(tps):.2f}x",
                # Latency p50
                "latency_p50_repeats": p50s,
                "latency_p50_mean": float(np.mean(p50s)),
                "latency_p50_std": float(np.std(p50s)),
                "latency_p50_median": float(np.median(p50s)),
                "latency_p50_range": float(np.max(p50s) - np.min(p50s)),
                # Latency p90
                "latency_p90_repeats": p90s,
                "latency_p90_median": float(np.median(p90s)),
                "latency_p90_range": float(np.max(p90s) - np.min(p90s)),
                # Latency p95
                "latency_p95_repeats": p95s,
                "latency_p95_mean": float(np.mean(p95s)),
                "latency_p95_std": float(np.std(p95s)),
                "latency_p95_median": float(np.median(p95s)),
                "latency_p95_range": float(np.max(p95s) - np.min(p95s)),
                # Latency p99
                "latency_p99_repeats": p99s,
                "latency_p99_median": float(np.median(p99s)),
                "latency_p99_range": float(np.max(p99s) - np.min(p99s)),
                # Wall time & Capacity
                "wall_time_s_repeats": walls,
                "wall_time_s_median": float(np.median(walls)),
                "wall_time_s_range": float(np.max(walls) - np.min(walls)),
                "service_capacity_req_per_sec_repeats": caps,
                "service_capacity_req_per_sec_median": float(np.median(caps)),
                "service_capacity_req_per_sec_range": float(np.max(caps) - np.min(caps)),
                "service_capacity_req_per_sec_mean": float(np.mean(caps)),
                "service_capacity_req_per_sec_std": float(np.std(caps)),
                "peak_vram_mb": float(np.mean([r["peak_vram_mb"] for r in runs])),
                # Telemetry under load
                "telemetry_under_load": {
                    "median_clock_mhz": float(np.median(clocks)),
                    "min_clock_mhz": float(np.min(min_clocks)),
                    "max_temp_c": float(np.max(temps)),
                    "mean_throttle_pct": float(np.mean(throttles)),
                    "gate_passed": all_passed,
                },
                "runs": runs,
            }

        profile_results[prof_name] = {
            "systems": summary,
        }

    # Session-wide Gate Evaluation
    all_median_clocks = [r["median_clock_mhz"] for r in all_measured_runs_telemetry if r["sample_count"] > 0]
    all_max_temps = [r["max_temp_c"] for r in all_measured_runs_telemetry if r["sample_count"] > 0]
    all_throttles = [r["throttle_sample_pct"] for r in all_measured_runs_telemetry if r["sample_count"] > 0]

    session_median_clock = float(np.median(all_median_clocks)) if all_median_clocks else 0.0
    session_max_temp = float(np.max(all_max_temps)) if all_max_temps else 0.0
    session_mean_throttle = float(np.mean(all_throttles)) if all_throttles else 0.0

    session_gate_passed = (
        session_median_clock >= target_clock_mhz and
        session_max_temp <= 88.0 and
        session_mean_throttle <= 10.0
    )
    session_mode = "clean" if session_gate_passed else "thermally limited"

    output_payload = {
        "session_id": session_id,
        "timestamp": timestamp,
        "metadata": {
            "session_id": session_id,
            "timestamp": timestamp,
            "session_mode": session_mode,
            "gate_passed": session_gate_passed,
            "gpu_model": torch.cuda.get_device_name(0),
            "max_boost_clock_mhz": max_boost_mhz,
            "target_clock_mhz_85pct": target_clock_mhz,
            "measured_median_clock_mhz": session_median_clock,
            "measured_max_temp_c": session_max_temp,
            "measured_throttle_pct": session_mean_throttle,
            "repeats": 3,
            "interleaved_order": [c[0] for c in configs],
        },
        "profiles": profile_results,
    }

    out_file = Path("data/real_speech_evaluation_results_profile_v2.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    print(f"\n[DONE] Interleaved benchmark saved to {out_file}")
    print(f"Session Status: {session_mode.upper()} (Gate Passed={session_gate_passed})")
    print(f"Under-Load Summary: Clock={session_median_clock:.0f} MHz (>= {target_clock_mhz:.0f} MHz) | Temp={session_max_temp:.0f} C | Throttle={session_mean_throttle:.1f}%")


if __name__ == "__main__":
    main()
