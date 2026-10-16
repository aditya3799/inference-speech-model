"""Default-off phase-timing diagnosis for static_graph versus hybrid_after."""

import argparse
import ctypes
import gc
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import jiwer
from torch.profiler import ProfilerActivity, profile

from scripts.evaluate_real_speech import RealSpeechBenchmark, generate_burst_traffic
from scripts.gpu_telemetry import CpuTelemetryMonitor, GpuTelemetryMonitor, evaluate_run_gate, evaluate_session_gate
from scripts.run_interleaved_benchmark_graph import cooldown_gpu, get_runtime_metadata
from speech_serving.phase_timing import PHASES


SYSTEMS: List[Tuple[str, Dict[str, Any]]] = [
    ("static_graph", {"kind": "static", "graph_decode": True, "readback_k": 4, "max_batch_size": 4}),
    ("hybrid_after", {
        "kind": "engine", "graph_decode": True, "readback_k": 4, "max_batch_size": 4,
        "cache_cross_kv": True, "retain_encoder_cache": False, "batch_encoder": False,
        "no_sync_decode": False, "avoid_redundant_cross_kv": False,
        "graph_non_blocking_copies": True, "graph_zero_slot_buffers_on_release": False,
        "graph_force_bucket": None,
    }),
]


def get_power_state() -> Dict[str, Any]:
    if os.name != "nt":
        return {"status": "not measured", "reason": "AC status probe implemented for Windows only"}
    class SystemPowerStatus(ctypes.Structure):
        _fields_ = [
            ("ACLineStatus", ctypes.c_ubyte), ("BatteryFlag", ctypes.c_ubyte),
            ("BatteryLifePercent", ctypes.c_ubyte), ("SystemStatusFlag", ctypes.c_ubyte),
            ("BatteryLifeTime", ctypes.c_ulong), ("BatteryFullLifeTime", ctypes.c_ulong),
        ]
    state = SystemPowerStatus()
    if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(state)):
        return {"status": "not measured", "reason": "GetSystemPowerStatus failed"}
    return {
        "status": "measured", "ac_line_status": int(state.ACLineStatus),
        "plugged_in": state.ACLineStatus == 1, "battery_percent": int(state.BatteryLifePercent),
    }


def run_system(bench, traffic, cfg, phase_timing):
    if cfg["kind"] == "static":
        return bench.run_static_batching(
            traffic, batch_size=4, graph_decode=True, graph_readback_k=4,
            phase_timing=phase_timing,
        )
    return bench.run_engine(
        traffic, max_batch_size=4, retain_encoder_cache=False, cache_cross_kv=True,
        batch_encoder=False, no_sync_decode=False, avoid_redundant_cross_kv=False,
        graph_decode=True, graph_readback_k=4, graph_non_blocking_copies=True,
        graph_zero_slot_buffers_on_release=False, graph_force_bucket=None,
        phase_timing=phase_timing,
    )


def reference_wer_for_traffic(bench, traffic) -> float:
    samples = [bench.samples[item["sample_idx"]] for item in traffic]
    refs = [sample["norm_ref"] for sample in samples]
    hyps = [bench.normalizer(bench.processor.tokenizer.decode(sample["hf_tokens"])) for sample in samples]
    return jiwer.wer(refs, hyps) * 100.0


def profiler_gpu_union(prof, wall_ms: float) -> Dict[str, Any]:
    intervals = []
    for event in prof.events():
        device = str(getattr(event, "device_type", "")).lower()
        interval = getattr(event, "time_range", None)
        if "cuda" not in device or interval is None:
            continue
        start = float(interval.start)
        end = float(interval.end)
        if end > start:
            intervals.append((start, end))
    if not intervals:
        return {"status": "not measured", "reason": "profiler returned no CUDA kernel intervals"}
    intervals.sort()
    union_us = 0.0
    start, end = intervals[0]
    for next_start, next_end in intervals[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            union_us += end - start
            start, end = next_start, next_end
    union_us += end - start
    union_ms = union_us / 1000.0
    return {
        "status": "measured", "kernel_interval_count": len(intervals),
        "gpu_kernel_union_ms": union_ms,
        "profiled_wall_ms": wall_ms,
        "gpu_kernel_coverage_pct": min(100.0, 100.0 * union_ms / wall_ms) if wall_ms else 0.0,
        "idle_gpu_time_ms": max(0.0, wall_ms - union_ms),
        "idle_gpu_fraction_pct": max(0.0, 100.0 * (wall_ms - union_ms) / wall_ms) if wall_ms else 0.0,
        "definition": "idle is profiled run wall time minus the union of CUDA kernel intervals",
    }


def cuda_profiler_intervals_supported() -> bool:
    """Probe whether this PyTorch/Kineto build exposes CUDA kernel intervals."""
    probe = torch.ones(1, device="cuda")
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        probe.add_(1)
        torch.cuda.synchronize()
    return any("cuda" in str(getattr(event, "device_type", "")).lower() for event in prof.events())


def aggregate_phases(per_request: List[Dict[str, Any]]) -> Dict[str, Any]:
    request_wall_ms = [r["e2e_latency"] * 1000.0 for r in per_request]
    total_request_wall_ms = float(sum(request_wall_ms))
    phases = {}
    for phase_name in PHASES:
        values = []
        cuda_values = []
        for request in per_request:
            value = request.get("phase_timing", {}).get("phases", {}).get(phase_name, "not measured")
            if isinstance(value, dict):
                values.append(float(value["wall_ms"]))
                cuda_values.append(float(value["cuda_ms"]))
        if not values:
            phases[phase_name] = "not measured"
        else:
            total_ms = float(sum(values))
            phases[phase_name] = {
                "request_count_measured": len(values), "total_ms": total_ms,
                "mean_ms_per_request": float(np.mean(values)),
                "total_cuda_event_ms": float(sum(cuda_values)),
                "mean_cuda_event_ms_per_request": float(np.mean(cuda_values)),
                "pct_of_request_wall_time": 100.0 * total_ms / total_request_wall_ms if total_request_wall_ms else 0.0,
            }
    slot_holds = [r.get("phase_timing", {}).get("slot_hold_steps_after_eos", 0) for r in per_request]
    return {
        "request_count": len(per_request), "total_request_wall_ms": total_request_wall_ms,
        "mean_request_wall_ms": float(np.mean(request_wall_ms)), "phases": phases,
        "slot_hold_steps_after_eos": {
            "total": int(sum(slot_holds)), "mean_per_request": float(np.mean(slot_holds)), "values": slot_holds,
        },
    }


def combine_repeats(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    combined_requests = [request for run in runs for request in run["per_request"]]
    requests_per_run = int(runs[0]["num_requests"])
    result = aggregate_phases(combined_requests)
    result.update({
        "wall_time_s_repeats": [run["wall_time_s"] for run in runs],
        "wall_time_s_median": float(np.median([run["wall_time_s"] for run in runs])),
        "capacity_req_per_s_median": float(np.median([requests_per_run / run["wall_time_s"] for run in runs])),
        "exact_match_pct": float(np.median([run["token_match_rate_pct"] for run in runs])),
        "wer_pct": float(np.median([run["wer_pct"] for run in runs])),
        "actual_batch4_wall_ms": (
            {"total_ms": float(sum(sum(run["batch_wall_times_ms"]) for run in runs)),
             "mean_ms": float(np.mean([v for run in runs for v in run["batch_wall_times_ms"]]))}
            if all(isinstance(run["batch_wall_times_ms"], list) for run in runs) else "not measured"
        ),
        "equivalent_batch4_wall_ms": float(np.median([run["wall_time_s"] * 1000.0 * 4.0 / requests_per_run for run in runs])),
        "profiler_gpu": {
            "gpu_kernel_union_ms_median": float(np.median([run["profiler_gpu"]["gpu_kernel_union_ms"] for run in runs if run["profiler_gpu"]["status"] == "measured"])),
            "idle_gpu_time_ms_median": float(np.median([run["profiler_gpu"]["idle_gpu_time_ms"] for run in runs if run["profiler_gpu"]["status"] == "measured"])),
            "idle_gpu_fraction_pct_median": float(np.median([run["profiler_gpu"]["idle_gpu_fraction_pct"] for run in runs if run["profiler_gpu"]["status"] == "measured"])),
        } if any(run["profiler_gpu"]["status"] == "measured" for run in runs) else "not measured",
        "runs": runs,
    })
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-timing", action="store_true", help="Enable boundary-sync CUDA-event phase instrumentation")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--utterances", type=int, default=100)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "data" / "phase_timing_diagnosis.json")
    args = parser.parse_args()
    if not args.phase_timing:
        raise SystemExit("Diagnosis requires explicit --phase-timing; normal engine behavior remains unchanged by default")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    bench = RealSpeechBenchmark(device="cuda", dtype=torch.float16)
    traffic = generate_burst_traffic(args.utterances)
    canary_traffic = generate_burst_traffic(min(20, args.utterances))
    reference_wer = reference_wer_for_traffic(bench, traffic)
    profiler_supported = cuda_profiler_intervals_supported()
    print("CUDA kernel intervals available from profiler:", profiler_supported)

    cooldown_gpu(target_temp=62.0, max_wait_s=60)
    print("\n=== Sequential canary START ===")
    canary_start = bench.run_sequential(canary_traffic)

    parity = {}
    print("\n=== Correctness gate ===")
    for name, cfg in SYSTEMS:
        result = run_system(bench, traffic, cfg, phase_timing=False)
        delta = result["wer_pct"] - reference_wer
        passed = result["token_match_rate_pct"] >= 99.0 and abs(delta) <= 0.2
        parity[name] = {
            "passed": passed, "status": "VALID" if passed else "INVALID",
            "exact_match_pct": result["token_match_rate_pct"], "wer_pct": result["wer_pct"],
            "reference_wer_pct": reference_wer, "wer_delta_points": delta,
        }
        print(name, parity[name])
    invalid = [name for name, gate in parity.items() if not gate["passed"]]
    if invalid:
        raise SystemExit("Parity-invalid systems: " + ", ".join(invalid))

    print("\n=== Discarded interleaved warm-up ===")
    for name, cfg in SYSTEMS:
        print("warming", name)
        run_system(bench, traffic, cfg, phase_timing=True)

    runs_by_system = {name: [] for name, _ in SYSTEMS}
    all_gpu = []
    for repeat in range(args.repeats):
        order = SYSTEMS[repeat % 2:] + SYSTEMS[:repeat % 2]
        print(f"\n=== Measured repeat {repeat + 1}: {[name for name, _ in order]} ===")
        for order_index, (name, cfg) in enumerate(order):
            gc.collect()
            torch.cuda.empty_cache()
            cooldown_gpu(target_temp=62.0, max_wait_s=60)
            gpu_mon, cpu_mon = GpuTelemetryMonitor(0.5), CpuTelemetryMonitor(1)
            gpu_mon.start(); cpu_mon.start()
            started = time.perf_counter()
            if profiler_supported:
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                    result = run_system(bench, traffic, cfg, phase_timing=True)
                profiler_result = profiler_gpu_union(prof, (time.perf_counter() - started) * 1000.0)
                del prof
            else:
                result = run_system(bench, traffic, cfg, phase_timing=True)
                profiler_result = {
                    "status": "not measured",
                    "reason": "installed PyTorch/Kineto build exposes no CUDA kernel intervals; Nsight Systems is unavailable",
                }
            gpu, cpu = gpu_mon.stop(), cpu_mon.stop()
            run_gate, failures = evaluate_run_gate(gpu, 88.0)
            all_gpu.append(gpu)
            result["repeat"] = repeat + 1
            result["order_index"] = order_index + 1
            result["profiler_gpu"] = profiler_result
            result["telemetry_gpu"] = gpu
            result["telemetry_cpu"] = cpu
            result["run_gate_passed"] = run_gate
            result["run_gate_failures"] = failures
            runs_by_system[name].append(result)
            print(f"{name}: wall={result['wall_time_s']:.3f}s exact={result['token_match_rate_pct']:.1f}% WER={result['wer_pct']:.3f}%")

    print("\n=== Instrumentation overhead audit (profiler disabled) ===")
    overhead = {}
    for name, cfg in SYSTEMS:
        measurements = {}
        for enabled in (False, True):
            gc.collect(); torch.cuda.empty_cache(); cooldown_gpu(target_temp=62.0, max_wait_s=60)
            result = run_system(bench, traffic, cfg, phase_timing=enabled)
            measurements["on" if enabled else "off"] = result["wall_time_s"]
        off_s, on_s = measurements["off"], measurements["on"]
        overhead[name] = {
            "flag_off_wall_time_s": off_s, "flag_on_wall_time_s": on_s,
            "overhead_s": on_s - off_s, "overhead_pct": 100.0 * (on_s / off_s - 1.0),
        }
        print(name, overhead[name])

    cooldown_gpu(target_temp=62.0, max_wait_s=60)
    print("\n=== Sequential canary END ===")
    canary_end = bench.run_sequential(canary_traffic)
    start_tp, end_tp = canary_start["throughput_audio_per_sec"], canary_end["throughput_audio_per_sec"]
    drift = abs(end_tp - start_tp) / start_tp * 100.0

    systems = {name: combine_repeats(runs_by_system[name]) for name, _ in SYSTEMS}
    ranked = []
    for phase_name in PHASES:
        static_phase = systems["static_graph"]["phases"][phase_name]
        hybrid_phase = systems["hybrid_after"]["phases"][phase_name]
        if isinstance(static_phase, dict) and isinstance(hybrid_phase, dict):
            difference = hybrid_phase["mean_ms_per_request"] - static_phase["mean_ms_per_request"]
            ranked.append({
                "phase": phase_name,
                "static_mean_ms": static_phase["mean_ms_per_request"],
                "hybrid_mean_ms": hybrid_phase["mean_ms_per_request"],
                "hybrid_minus_static_mean_ms": difference,
            })
        else:
            ranked.append({"phase": phase_name, "status": "not measured"})
    ranked.sort(key=lambda row: row.get("hybrid_minus_static_mean_ms", float("-inf")), reverse=True)

    clocks = {name: systems[name]["runs"][0]["telemetry_gpu"]["busy_median_clock_mhz"] for name, _ in SYSTEMS}
    for name in clocks:
        clocks[name] = float(np.median([r["telemetry_gpu"]["busy_median_clock_mhz"] for r in systems[name]["runs"]]))
    gate_passed, gate = evaluate_session_gate(clocks, all_gpu, drift, 88.0)
    payload = {
        "timestamp": datetime.now().isoformat(), "status": "VALID" if gate_passed else "TELEMETRY_INVALID",
        "environment": {**get_runtime_metadata(), "power_state": get_power_state()},
        "methodology": {
            "traffic": "burst", "utterances": args.utterances, "systems": [name for name, _ in SYSTEMS],
            "repeats": args.repeats, "discarded_warmup_cycles": 1, "interleaving": "alternating order",
            "max_batch_size": 4, "graph_decode": True, "readback_k": 4,
            "graph_buckets": [64, 128, 256, 448], "phase_timing": True,
            "profiler_cuda_intervals_supported": profiler_supported,
            "busy_sample_definition": "GPU utilization >= 80%",
            "request_phase_semantics": "shared batch work is request-experienced time and is recorded for every participating request",
            "reported_phases": list(PHASES),
        },
        "system_configs": dict(SYSTEMS), "parity_gate": parity, "systems": systems,
        "phase_difference_ranked": ranked, "instrumentation_overhead": overhead,
        "canary": {"start_audio_x": start_tp, "end_audio_x": end_tp, "drift_pct": drift, "passed": drift <= 15.0},
        "session_gate": gate,
        "gate_criteria": {
            "busy_throttle_pct": "<= 10%", "clock_spread_pct": "<= 10%",
            "canary_drift_pct": "<= 15%", "max_temperature_c": "<= 88 C",
        },
    }
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8", newline="\n")
    print("Saved", args.output)
    print("Session", payload["status"], gate["failures"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
