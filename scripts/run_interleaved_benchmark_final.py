"""
scripts/run_interleaved_benchmark_final.py

ONE-PASS Interleaved Benchmark Runner (Section B1):
1. Runs sequential canary at start (20 utterances).
2. Interleaved benchmark across 5 systems:
   - static_batching
   - engine_baseline
   - engine_no_sync
   - engine_batch_encoder
   - engine_avoid_redundant
3. Across 3 traffic profiles (100 utterances each):
   - burst: capacity = 100 / wall_time
   - poisson_high (lambda = 4.0 req/s): arrival-limited throughput, latency percentiles
   - poisson_mid (lambda = 3.0 req/s): arrival-limited throughput, latency percentiles
4. Warm-up policy: 1 full discarded interleaved cycle (Cycle 0) before 3 measured cycles.
5. Live under-load telemetry:
   - GPU: polls nvidia-smi every 500 ms (clock, power, temp, throttle, utilization).
     Gated over busy samples (utilization >= 80%).
   - CPU: polls typeperf '% Processor Performance' every 1s (median and min per system).
6. Runs sequential canary at end (20 utterances), failing if drift > 15%.
7. Saves output to data/real_speech_evaluation_results_profile_final.json.
"""

import os
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

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
    CpuTelemetryMonitor,
    get_max_boost_clock,
    evaluate_run_gate,
    evaluate_session_gate,
)


def cooldown_gpu(target_temp: float = 65.0, max_wait_s: int = 45):
    """Waits for GPU temperature to settle before running a test cycle."""
    t0 = time.time()
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
    parser = argparse.ArgumentParser(description="Final Interleaved Benchmark Runner")
    parser.add_argument("--session-id", type=str, default="", help="Unified session identifier")
    parser.add_argument("--timestamp", type=str, default="", help="ISO timestamp of session")
    args = parser.parse_args()

    session_id = args.session_id if args.session_id else f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    timestamp = args.timestamp if args.timestamp else datetime.now().isoformat()

    print("==================================================================")
    print("FINAL INTERLEAVED BENCHMARK RUNNER (Section B1)")
    print(f"Session ID : {session_id}")
    print(f"Timestamp  : {timestamp}")
    print("==================================================================")

    max_boost_mhz = get_max_boost_clock()
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"GPU Hardware: {device_name}")
    print(f"Max Boost Clock : {max_boost_mhz:.0f} MHz")

    # Cooldown before starting session
    print("\nPre-benchmark cooldown (target <= 60 C)...")
    cooldown_gpu(target_temp=60.0, max_wait_s=45)

    bench = RealSpeechBenchmark(device="cuda", dtype=torch.float16)

    # 1. Sequential Canary at START (20 utterances)
    canary_samples_count = min(20, len(bench.samples))
    canary_traffic = [{"req_id": i, "sample_idx": i, "arrival_time": 0.0} for i in range(canary_samples_count)]
    print(f"\n--- Running Sequential Canary START ({canary_samples_count} utterances) ---")
    canary_start_res = bench.run_sequential(canary_traffic)
    canary_start_tp = canary_start_res["throughput_audio_per_sec"]
    print(f"Canary START throughput: {canary_start_tp:.2f}x audio-time")

    # Traffic profiles: Burst, Poisson-high (4.0 req/s), Poisson-mid (3.0 req/s)
    traffic_burst = generate_burst_traffic(100)
    traffic_poisson_high = generate_poisson_traffic(100, rate=4.0, seed=42)
    traffic_poisson_mid = generate_poisson_traffic(100, rate=3.0, seed=42)

    profiles = {
        "burst": traffic_burst,
        "poisson_high": traffic_poisson_high,
        "poisson_mid": traffic_poisson_mid,
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
        print(f"TRAFFIC PROFILE: {prof_name.upper()} (100 utterances)")
        print(f"==================================================================")

        # Requirement B1: 1 full discarded interleaved cycle (Cycle 0)
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
                cooldown_gpu(target_temp=70.0, max_wait_s=30)

                print(f"  [{rep+1}/{num_repeats}] {cfg_name}... ", end="", flush=True)

                # Active GPU & CPU telemetry polling
                gpu_mon = GpuTelemetryMonitor(interval_sec=0.5)
                cpu_mon = CpuTelemetryMonitor(interval_sec=1)
                gpu_mon.start()
                cpu_mon.start()

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

                gpu_summary = gpu_mon.stop()
                cpu_summary = cpu_mon.stop()

                gate_ok, gate_reasons = evaluate_run_gate(gpu_summary, max_temp_threshold_c=88.0)
                all_measured_runs_telemetry.append(gpu_summary)

                latencies = [r["e2e_latency"] for r in run_res["per_request"]]
                p50 = float(np.percentile(latencies, 50))
                p90 = float(np.percentile(latencies, 90))
                p95 = float(np.percentile(latencies, 95))
                p99 = float(np.percentile(latencies, 99))
                wall_s = run_res["wall_time_s"]
                cap_req_s = 100.0 / wall_s if wall_s > 0 else 0.0

                print(
                    f"Done. Wall={wall_s:.2f}s | "
                    f"BusyClock={gpu_summary['busy_median_clock_mhz']:.0f}MHz | "
                    f"Temp={gpu_summary['busy_max_temp_c']:.0f}C | "
                    f"BusyThrot={gpu_summary['busy_throttle_sample_pct']:.1f}% | "
                    f"CPU Perf(med/min)={cpu_summary['median_cpu_perf_pct']:.0f}%/{cpu_summary['min_cpu_perf_pct']:.0f}%"
                )

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
                    "telemetry_gpu": gpu_summary,
                    "telemetry_cpu": cpu_summary,
                    "gate_passed": gate_ok,
                    "gate_failures": gate_reasons,
                    "per_request": run_res["per_request"],
                })

        # Summarize profile
        summary: Dict[str, Any] = {}
        for cfg_name, runs in profile_runs.items():
            tps = [r["throughput"] for r in runs]
            p50s = [r["latency_p50"] for r in runs]
            p90s = [r["latency_p90"] for r in runs]
            p95s = [r["latency_p95"] for r in runs]
            p99s = [r["latency_p99"] for r in runs]
            walls = [r["wall_time_s"] for r in runs]
            caps = [r["service_capacity_req_per_sec"] for r in runs]

            busy_clocks = [r["telemetry_gpu"]["busy_median_clock_mhz"] for r in runs]
            busy_min_clocks = [r["telemetry_gpu"]["busy_min_clock_mhz"] for r in runs]
            busy_temps = [r["telemetry_gpu"]["busy_max_temp_c"] for r in runs]
            busy_throttles = [r["telemetry_gpu"]["busy_throttle_sample_pct"] for r in runs]

            cpu_meds = [r["telemetry_cpu"]["median_cpu_perf_pct"] for r in runs]
            cpu_mins = [r["telemetry_cpu"]["min_cpu_perf_pct"] for r in runs]

            if prof_name.startswith("poisson"):
                rate_val = "4.00" if "high" in prof_name else "3.00"
                tp_label = f"Arrival-limited ({rate_val} req/s)"
            else:
                tp_label = f"{np.median(caps):.2f} req/s (capacity)"

            summary[cfg_name] = {
                # Throughput
                "throughput_repeats": tps,
                "throughput_median": float(np.median(tps)),
                "throughput_range": float(np.max(tps) - np.min(tps)),
                "throughput_label": tp_label,
                # Latency p50
                "latency_p50_repeats": p50s,
                "latency_p50_median": float(np.median(p50s)),
                "latency_p50_range": float(np.max(p50s) - np.min(p50s)),
                # Latency p90
                "latency_p90_repeats": p90s,
                "latency_p90_median": float(np.median(p90s)),
                "latency_p90_range": float(np.max(p90s) - np.min(p90s)),
                # Latency p95
                "latency_p95_repeats": p95s,
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
                "peak_vram_mb": float(np.mean([r["peak_vram_mb"] for r in runs])),
                # Telemetry under load
                "telemetry_under_load": {
                    "busy_median_clock_mhz": float(np.median(busy_clocks)),
                    "busy_min_clock_mhz": float(np.min(busy_min_clocks)),
                    "busy_max_temp_c": float(np.max(busy_temps)),
                    "busy_throttle_sample_pct": float(np.mean(busy_throttles)),
                    "cpu_median_perf_pct": float(np.median(cpu_meds)),
                    "cpu_min_perf_pct": float(np.min(cpu_mins)),
                },
                "runs": runs,
            }

        profile_results[prof_name] = {
            "systems": summary,
        }

    # 2. Sequential Canary at END (20 utterances)
    print(f"\n--- Running Sequential Canary END ({canary_samples_count} utterances) ---")
    canary_end_res = bench.run_sequential(canary_traffic)
    canary_end_tp = canary_end_res["throughput_audio_per_sec"]
    canary_drift_pct = abs(canary_end_tp - canary_start_tp) / canary_start_tp * 100.0 if canary_start_tp > 0 else 0.0
    print(f"Canary END throughput: {canary_end_tp:.2f}x (Start: {canary_start_tp:.2f}x, Drift: {canary_drift_pct:.1f}%)")

    # Evaluate System Median Busy Clocks across all runs
    system_median_busy_clocks = {}
    for cfg_name, _ in configs:
        sys_clks = []
        for prof in profile_results.values():
            if cfg_name in prof["systems"]:
                sys_clks.append(prof["systems"][cfg_name]["telemetry_under_load"]["busy_median_clock_mhz"])
        if sys_clks:
            system_median_busy_clocks[cfg_name] = float(np.median(sys_clks))

    gate_passed, gate_details = evaluate_session_gate(
        system_median_busy_clocks=system_median_busy_clocks,
        all_runs_summaries=all_measured_runs_telemetry,
        canary_drift_pct=canary_drift_pct,
        max_temp_threshold_c=88.0,
    )

    session_mode = "clean" if gate_passed else "thermally limited"

    output_payload = {
        "session_id": session_id,
        "timestamp": timestamp,
        "metadata": {
            "session_id": session_id,
            "timestamp": timestamp,
            "session_mode": session_mode,
            "gate_passed": gate_passed,
            "gate_details": gate_details,
            "gpu_model": device_name,
            "max_boost_clock_mhz": max_boost_mhz,
            "canary": {
                "start_throughput": canary_start_tp,
                "end_throughput": canary_end_tp,
                "drift_pct": canary_drift_pct,
                "passed": canary_drift_pct <= 15.0,
            },
            "repeats": 3,
            "interleaved_order": [c[0] for c in configs],
        },
        "profiles": profile_results,
    }

    out_file = Path("data/real_speech_evaluation_results_profile_final.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    print(f"\n[DONE] Final interleaved benchmark saved to {out_file}")
    print(f"Session Status: {session_mode.upper()} (Gate Passed={gate_passed})")
    print(f"Gate Summary: Max Temp={gate_details['max_temperature_c']:.1f} C | Max Throttle={gate_details['max_busy_throttle_pct']:.1f}% | Clock Spread={gate_details['clock_spread_pct']:.1f}% | Canary Drift={canary_drift_pct:.1f}%")
    if gate_details["failures"]:
        print("Gate Failures:")
        for fail in gate_details["failures"]:
            print(f"  - {fail}")


if __name__ == "__main__":
    main()
