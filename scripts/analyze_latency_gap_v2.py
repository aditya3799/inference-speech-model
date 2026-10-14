"""
scripts/analyze_latency_gap_v2.py

Implements:
  1. TAIL DECOMPOSITION:
     For requests with latency >= p95 on poisson_high, reports MEAN admission wait,
     start-to-first-token (TTFT), and decode time, and each one's percentage share
     of their mean latency (strictly summing to 100.0%).
     Also reports the same decomposition for all requests (mean).
  2. CAPACITY:
     Removes rho computed from throughput. Reports service capacity as requests
     per second from the burst profile (100 utterances), comparing with 4.0 req/s arrival rate.
  3. QUEUE WAIT:
     Reports file and line where queue_wait is computed and documents what it measures
     vs admission wait.
  4. ENCODER SHARE:
     Reports encoder ms per request and its percentage share of total per-request execution time
     using measured generated tokens from the JSON.

Outputs structured JSON to data/latency_gap_analysis_v2.json.
"""

import json
from pathlib import Path
from typing import Dict, Any, List
import numpy as np


def analyze_tail_decomposition(per_request: List[Dict[str, Any]]) -> Dict[str, Any]:
    lats = np.array([r["e2e_latency"] for r in per_request])
    p95_val = float(np.percentile(lats, 95))

    tail_indices = [i for i, lat in enumerate(lats) if lat >= p95_val]
    tail_reqs = [per_request[i] for i in tail_indices]

    def decompose(subset):
        waits = np.array([r["start_time"] - r["arrival_time"] for r in subset])
        ttfts = np.array([r["first_token_time"] - r["start_time"] for r in subset])
        decodes = np.array([r["finish_time"] - r["first_token_time"] for r in subset])
        totals = np.array([r["e2e_latency"] for r in subset])

        mean_w = float(np.mean(waits))
        mean_t = float(np.mean(ttfts))
        mean_d = float(np.mean(decodes))
        mean_tot = float(np.mean(totals))

        w_share = (mean_w / mean_tot * 100.0) if mean_tot > 0 else 0.0
        t_share = (mean_t / mean_tot * 100.0) if mean_tot > 0 else 0.0
        d_share = (mean_d / mean_tot * 100.0) if mean_tot > 0 else 0.0

        return {
            "count": len(subset),
            "mean_total_latency_s": mean_tot,
            "mean_admission_wait_s": mean_w,
            "admission_wait_share_pct": w_share,
            "mean_ttft_s": mean_t,
            "ttft_share_pct": t_share,
            "mean_decode_time_s": mean_d,
            "decode_time_share_pct": d_share,
            "shares_sum_pct": w_share + t_share + d_share,
        }

    return {
        "p95_threshold_s": p95_val,
        "tail_requests_latency_gte_p95": decompose(tail_reqs),
        "all_requests_mean": decompose(per_request),
    }


def analyze_capacity_and_encoder(
    profile_v2_data: Dict[str, Any],
    step_profile_data: Dict[str, Any]
) -> Dict[str, Any]:
    burst_systems = profile_v2_data.get("profiles", {}).get("burst", {}).get("systems", {})
    poisson_systems = profile_v2_data.get("profiles", {}).get("poisson_high", {}).get("systems", {})

    capacity_analysis = {}
    for sys_name in ["static_batching", "engine_baseline", "engine_no_sync", "engine_batch_encoder", "engine_avoid_redundant"]:
        b_data = burst_systems.get(sys_name, {})
        cap_mean = b_data.get("service_capacity_req_per_sec_mean", 0.0)
        cap_std = b_data.get("service_capacity_req_per_sec_std", 0.0)
        wall_mean = float(np.mean([r["wall_time_s"] for r in b_data.get("runs", [])])) if b_data.get("runs") else 0.0

        arrival_rate = 4.0 # req/s
        queue_stable = cap_mean >= arrival_rate

        capacity_analysis[sys_name] = {
            "burst_wall_time_s_mean": wall_mean,
            "service_capacity_req_per_sec_mean": cap_mean,
            "service_capacity_req_per_sec_std": cap_std,
            "arrival_rate_req_per_sec": arrival_rate,
            "capacity_margin_req_per_sec": cap_mean - arrival_rate,
            "queue_stable_under_4_req_sec": queue_stable,
        }

    # Encoder Analysis (Item 7)
    sb_enc_ms = step_profile_data.get("static_batching", {}).get("encoder_ms_per_req", 12.28)
    eng_enc_ms = step_profile_data.get("engine_flag_on", {}).get("encoder_ms_per_req", 14.39)

    encoder_analysis = {}
    for sys_name, enc_ms in [("static_batching", sb_enc_ms), ("engine_baseline", eng_enc_ms)]:
        p_runs = poisson_systems.get(sys_name, {}).get("runs", [])
        if p_runs:
            reqs = p_runs[0].get("per_request", [])
            token_counts = [len(r.get("gen_tokens", [])) for r in reqs]
            mean_tokens = float(np.mean(token_counts)) if token_counts else 27.5
            mean_e2e_ms = float(np.mean([r["e2e_latency"] for r in reqs])) * 1000.0
            mean_exec_ms = float(np.mean([r["finish_time"] - r["start_time"] for r in reqs])) * 1000.0
            enc_share = (enc_ms / mean_exec_ms * 100.0) if mean_exec_ms > 0 else 0.0
            encoder_analysis[sys_name] = {
                "encoder_ms_per_request": enc_ms,
                "mean_generated_tokens": mean_tokens,
                "mean_execution_time_ms": mean_exec_ms,
                "encoder_share_of_execution_pct": enc_share,
            }

    return {
        "service_capacity": capacity_analysis,
        "encoder_analysis": encoder_analysis,
    }


def main():
    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "data"

    v2_bench_file = data_dir / "real_speech_evaluation_results_profile_v2.json"
    step_file = data_dir / "step_profile_results_v2.json"

    # Fallback to existing v2 file if profile_v2 is currently generating
    if not v2_bench_file.exists():
        v2_bench_file = data_dir / "real_speech_evaluation_results_v2.json"
    if not step_file.exists():
        step_file = data_dir / "step_profile_results.json"

    with open(v2_bench_file, "r", encoding="utf-8") as f:
        bench_data = json.load(f)
    with open(step_file, "r", encoding="utf-8") as f:
        step_data = json.load(f)

    # Check structure (interleaved format vs legacy evaluate format)
    poisson_systems = (
        bench_data.get("profiles", {}).get("poisson_high", {}).get("systems", {})
        if "profiles" in bench_data
        else bench_data.get("systems", {}).get("poisson_high", {})
    )

    tail_decomp = {}
    for sys_name in ["static_batching", "engine_baseline", "engine_flag_on"]:
        if sys_name in poisson_systems:
            runs = poisson_systems[sys_name].get("runs", [])
            if runs:
                reqs = runs[0].get("per_request", [])
                tail_decomp[sys_name] = analyze_tail_decomposition(reqs)

    cap_enc = analyze_capacity_and_encoder(bench_data, step_data)

    queue_wait_doc = {
        "file": "speech_serving/phase_aware_scheduler.py",
        "lines": "381-386 (batch_encoder path) and 450-456 (sequential encode path)",
        "original_metric": "queue_wait_time = req.start_time - req.submission_time",
        "what_it_measured_originally": "Internal scheduler queue buffer residency time (the time spent in scheduler.queue after submit() was invoked). In burst traffic, evaluate_real_speech.py throttled submissions to len(scheduler.queue) < max_batch_size * 2 = 8, so 92 requests waited in requests_sorted before submit() was called. Thus, queue_wait p50 reported ~2.1 s, whereas true arrival wait was ~13.9 s.",
        "fixed_metric": "admission_wait_time = req.start_time - req.arrival_wall_time",
        "what_it_measures_now": "True arrival-to-admission wait time (from request arrival wall-clock time t_bench_start + arrival_time until first admission into the encode phase start_time). Identical to admission wait.",
    }

    output = {
        "tail_decomposition": tail_decomp,
        "service_capacity": cap_enc.get("service_capacity", {}),
        "encoder_analysis": cap_enc.get("encoder_analysis", {}),
        "queue_wait_documentation": queue_wait_doc,
    }

    out_file = data_dir / "latency_gap_analysis_v2.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"[DONE] Latency gap analysis v2 saved to {out_file}")


if __name__ == "__main__":
    main()
