"""
scripts/run_interleaved_benchmark_v2.py

PART 5 & 6 (Updated v2):
Executes the clean rerun with:
  1. GPU cooldown to idle temp before starting.
  2. Session gate check (clean vs thermally limited, no --force-unsafe abort).
  3. INTERLEAVED execution:
     For repeat in (1, 2, 3):
       static_batching -> engine_baseline -> engine_no_sync -> engine_batch_encoder -> engine_avoid_redundant
     This ensures thermal drift hits all systems equally.
  4. Logs GPU temperature and graphics clock before and after each run.
  5. Computes interleaved pair ratios.
  6. Marks rows with std > 10% of mean as "not reportable".
  7. Records full per-request timestamps for tail decomposition.

Saves output to data/real_speech_evaluation_results_profile_v2.json.
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
import subprocess
import numpy as np
import torch
from typing import Dict, Any, List, Tuple

from scripts.evaluate_real_speech import (
    RealSpeechBenchmark,
    check_session_gate,
    generate_burst_traffic,
    generate_poisson_traffic,
)


def get_gpu_telemetry() -> Tuple[float, float]:
    """Returns (temperature_c, graphics_clock_mhz)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu,clocks.current.graphics", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        parts = [p.strip() for p in out.split(",")]
        return float(parts[0]), float(parts[1])
    except Exception:
        return 0.0, 0.0


def cooldown_gpu(target_temp: float = 65.0, max_wait_s: int = 60):
    t0 = time.time()
    while time.time() - t0 < max_wait_s:
        temp, _ = get_gpu_telemetry()
        if temp <= target_temp or temp == 0.0:
            break
        time.sleep(2)


def main():
    print("==================================================================")
    print("INTERLEAVED CLEAN RERUN BENCHMARK (V2)")
    print("==================================================================")

    # 1. Cooldown
    print("Letting GPU cool to idle temperature (<= 62 C)...")
    cooldown_gpu(target_temp=62.0, max_wait_s=60)
    temp_init, clock_init = get_gpu_telemetry()
    print(f"Initial GPU State: Temp={temp_init:.1f} C, Clock={clock_init:.0f} MHz")

    # 2. Gate check (run via try/except so gate fails do not crash, but label run)
    gate_passed = False
    gate_telemetry = {}
    try:
        gate_telemetry = check_session_gate(force_unsafe=False)
        gate_passed = gate_telemetry.get("gate_passed", False)
    except Exception as e:
        print(f"[GATE RESULT] Gate check failed: {e}")
        gate_passed = False
        temp_load, clock_load = get_gpu_telemetry()
        gate_telemetry = {
            "gate_passed": False,
            "gpu_temp_under_load_c": temp_load,
            "graphics_clock_under_load_mhz": clock_load,
            "throttle_reason": "Thermal / Clock limit under load",
        }

    run_label = "clean" if gate_passed else "thermally limited"
    print(f"\nBenchmark Session Mode: {run_label.upper()}")

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

    results: Dict[str, Any] = {
        "metadata": {
            "session_mode": run_label,
            "gate_passed": gate_passed,
            "gate_telemetry": gate_telemetry,
            "gpu_model": torch.cuda.get_device_name(0),
            "repeats": 3,
            "traffic_profiles": ["burst", "poisson_high"],
            "interleaved_order": [c[0] for c in configs],
        },
        "profiles": {},
    }

    num_repeats = 3

    for prof_name, traffic_data in profiles.items():
        print(f"\n==================================================================")
        print(f"RUNNING INTERLEAVED EVALUATION: {prof_name.upper()}")
        print(f"==================================================================")
        profile_runs: Dict[str, List[Dict[str, Any]]] = {c[0]: [] for c in configs}

        # Warm-up run (discarded)
        warm_traffic = traffic_data[:5]
        bench.run_static_batching(warm_traffic, batch_size=4)
        bench.run_engine(warm_traffic, max_batch_size=4, cache_cross_kv=True)

        for rep in range(num_repeats):
            print(f"\n--- Repeat {rep+1}/{num_repeats} (Interleaved Cycle) ---")
            for cfg_name, cfg_opts in configs:
                gc.collect()
                torch.cuda.empty_cache()
                cooldown_gpu(target_temp=72.0, max_wait_s=30)
                temp_pre, clk_pre = get_gpu_telemetry()

                print(f"  [{rep+1}/{num_repeats}] {cfg_name} (Pre: {temp_pre:.0f}C, {clk_pre:.0f}MHz)... ", end="", flush=True)

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

                temp_post, clk_post = get_gpu_telemetry()
                print(f"Done. TP={run_res['throughput_audio_per_sec']:.2f}x | Lat p50={run_res['latency_p50_s']:.2f}s, p95={run_res['latency_p95_s']:.2f}s (Post: {temp_post:.0f}C, {clk_post:.0f}MHz)")

                profile_runs[cfg_name].append({
                    "repeat": rep + 1,
                    "throughput": run_res["throughput_audio_per_sec"],
                    "latency_p50": run_res["latency_p50_s"],
                    "latency_p95": run_res["latency_p95_s"],
                    "latency_mean": float(np.mean([r["e2e_latency"] for r in run_res["per_request"]])),
                    "ttft_p50": run_res["ttft_p50_s"],
                    "ttft_p95": run_res["ttft_p95_s"],
                    "wall_time_s": run_res["wall_time_s"],
                    "peak_vram_mb": run_res["peak_vram_allocated_mb"],
                    "temp_pre": temp_pre,
                    "clock_pre": clk_pre,
                    "temp_post": temp_post,
                    "clock_post": clk_post,
                    "per_request": run_res["per_request"],
                })

        # Summarize this profile
        summary: Dict[str, Any] = {}
        for cfg_name, runs in profile_runs.items():
            tps = [r["throughput"] for r in runs]
            p50s = [r["latency_p50"] for r in runs]
            p95s = [r["latency_p95"] for r in runs]
            walls = [r["wall_time_s"] for r in runs]

            tp_mean = float(np.mean(tps))
            tp_std = float(np.std(tps))
            p50_mean = float(np.mean(p50s))
            p50_std = float(np.std(p50s))
            p95_mean = float(np.mean(p95s))
            p95_std = float(np.std(p95s))

            # Service capacity in requests/sec = 100 / wall_time_s
            caps = [100.0 / w for w in walls]
            cap_mean = float(np.mean(caps))
            cap_std = float(np.std(caps))

            # Check 10% std threshold
            tp_reportable = (tp_std <= 0.10 * tp_mean) if tp_mean > 0 else True
            p95_reportable = (p95_std <= 0.10 * p95_mean) if p95_mean > 0 else True

            summary[cfg_name] = {
                "throughput_mean": tp_mean,
                "throughput_std": tp_std,
                "throughput_reportable": tp_reportable,
                "latency_p50_mean": p50_mean,
                "latency_p50_std": p50_std,
                "latency_p95_mean": p95_mean,
                "latency_p95_std": p95_std,
                "latency_p95_reportable": p95_reportable,
                "service_capacity_req_per_sec_mean": cap_mean,
                "service_capacity_req_per_sec_std": cap_std,
                "peak_vram_mb": float(np.mean([r["peak_vram_mb"] for r in runs])),
                "runs": runs,
            }

        # Interleaved ratios against static_batching
        sb_runs = profile_runs["static_batching"]
        ratios: Dict[str, Any] = {}
        for cfg_name in [c[0] for c in configs if c[0] != "static_batching"]:
            cfg_r = profile_runs[cfg_name]
            tp_ratios = [sb_runs[i]["throughput"] / cfg_r[i]["throughput"] for i in range(num_repeats)]
            p95_ratios = [cfg_r[i]["latency_p95"] / sb_runs[i]["latency_p95"] for i in range(num_repeats)]
            ratios[cfg_name] = {
                "throughput_ratio_static_over_engine_mean": float(np.mean(tp_ratios)),
                "throughput_ratio_static_over_engine_std": float(np.std(tp_ratios)),
                "p95_ratio_engine_over_static_mean": float(np.mean(p95_ratios)),
                "p95_ratio_engine_over_static_std": float(np.std(p95_ratios)),
            }

        results["profiles"][prof_name] = {
            "systems": summary,
            "interleaved_ratios": ratios,
        }

    out_file = Path("data/real_speech_evaluation_results_profile_v2.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[DONE] Interleaved benchmark v2 saved to {out_file}")


if __name__ == "__main__":
    main()
