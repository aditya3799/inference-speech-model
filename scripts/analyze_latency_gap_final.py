"""
scripts/analyze_latency_gap_final.py

ONE-PASS Latency Gap & Capacity Analysis (Sections A3 & B1):
1. Tail Latency Decomposition:
   - For requests >= p95 on poisson_high (4.0 req/s) and poisson_mid (3.0 req/s):
     reports MEAN admission wait, TTFT, and decode time, and each one's percentage share
     (strictly summing to 100.0%).
   - Also reports decomposition for all requests (mean).
2. Service Capacity:
   - Evaluates burst capacity (100 / wall time) per repeat, median, range, comparing with 4.0 req/s.
3. Variant Comparisons (Requirement A3):
   - Uses ENGINE BASELINE as the reference (not static).
   - Reports delta vs engine baseline.
   - Labels a difference 'not distinguishable' if |delta| <= combined range.
4. Outputs to data/latency_gap_analysis_final.json.
"""

import json
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import argparse


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


def main():
    parser = argparse.ArgumentParser(description="Final Latency Gap Analysis")
    parser.add_argument("--session-id", type=str, default="")
    parser.add_argument("--timestamp", type=str, default="")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    data_dir = repo_root / "data"

    bench_file = data_dir / "real_speech_evaluation_results_profile_final.json"
    step_file = data_dir / "step_profile_results_final.json"

    with open(bench_file, "r", encoding="utf-8") as f:
        bench_data = json.load(f)
    with open(step_file, "r", encoding="utf-8") as f:
        step_data = json.load(f)

    session_id = args.session_id or bench_data.get("session_id", "")
    timestamp = args.timestamp or bench_data.get("timestamp", "")

    # Decompose tail latencies for poisson profiles
    profiles = bench_data.get("profiles", {})
    tail_decompositions = {}

    for prof_name in ["poisson_high", "poisson_mid"]:
        if prof_name in profiles:
            systems = profiles[prof_name].get("systems", {})
            tail_decompositions[prof_name] = {}
            for sys_name, sys_info in systems.items():
                runs = sys_info.get("runs", [])
                if runs:
                    reqs = runs[0].get("per_request", [])
                    if reqs:
                        tail_decompositions[prof_name][sys_name] = analyze_tail_decomposition(reqs)

    # Burst Service Capacity and Variant Comparisons vs Engine Baseline (Requirement A3)
    burst_systems = profiles.get("burst", {}).get("systems", {})
    eng_base = burst_systems.get("engine_baseline", {})
    eng_base_cap_med = eng_base.get("service_capacity_req_per_sec_median", 0.0)
    eng_base_cap_rng = eng_base.get("service_capacity_req_per_sec_range", 0.0)

    variant_comparisons_burst = {}
    for sys_k, sys_info in burst_systems.items():
        if sys_k == "engine_baseline":
            continue
        cur_med = sys_info.get("service_capacity_req_per_sec_median", 0.0)
        cur_rng = sys_info.get("service_capacity_req_per_sec_range", 0.0)
        delta_cap = cur_med - eng_base_cap_med
        combined_rng = max(cur_rng, eng_base_cap_rng)
        is_distinguishable = abs(delta_cap) > combined_rng

        variant_comparisons_burst[sys_k] = {
            "system": sys_k,
            "median_capacity_req_per_sec": cur_med,
            "range_capacity_req_per_sec": cur_rng,
            "delta_vs_engine_baseline": float(delta_cap),
            "reference_engine_baseline_median": eng_base_cap_med,
            "reference_engine_baseline_range": eng_base_cap_rng,
            "combined_range": float(combined_rng),
            "is_distinguishable": is_distinguishable,
            "assessment_label": (
                f"Distinguishable (|Δ|={abs(delta_cap):.2f} > range {combined_rng:.2f})"
                if is_distinguishable
                else f"Not distinguishable (|Δ|={abs(delta_cap):.2f} <= range {combined_rng:.2f})"
            ),
        }

    # Poisson p95 Latency Comparisons vs Engine Baseline (Requirement A3)
    variant_comparisons_poisson_high = {}
    ph_systems = profiles.get("poisson_high", {}).get("systems", {})
    eng_ph = ph_systems.get("engine_baseline", {})
    eng_ph_p95_med = eng_ph.get("latency_p95_median", 0.0)
    eng_ph_p95_rng = eng_ph.get("latency_p95_range", 0.0)

    for sys_k, sys_info in ph_systems.items():
        if sys_k == "engine_baseline":
            continue
        cur_p95 = sys_info.get("latency_p95_median", 0.0)
        cur_rng = sys_info.get("latency_p95_range", 0.0)
        delta_p95 = cur_p95 - eng_ph_p95_med
        combined_rng = max(cur_rng, eng_ph_p95_rng)
        is_dist = abs(delta_p95) > combined_rng

        variant_comparisons_poisson_high[sys_k] = {
            "system": sys_k,
            "median_p95_latency_s": cur_p95,
            "range_p95_latency_s": cur_rng,
            "delta_vs_engine_baseline_s": float(delta_p95),
            "combined_range_s": float(combined_rng),
            "is_distinguishable": is_dist,
            "assessment_label": (
                f"Distinguishable (|Δ|={abs(delta_p95):.2f}s > range {combined_rng:.2f}s)"
                if is_dist
                else f"Not distinguishable (|Δ|={abs(delta_p95):.2f}s <= range {combined_rng:.2f}s)"
            ),
        }

    output = {
        "session_id": session_id,
        "timestamp": timestamp,
        "tail_decompositions": tail_decompositions,
        "variant_comparisons_burst_vs_engine_baseline": variant_comparisons_burst,
        "variant_comparisons_poisson_high_vs_engine_baseline": variant_comparisons_poisson_high,
    }

    out_file = data_dir / "latency_gap_analysis_final.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\n[DONE] Latency gap analysis saved to {out_file}")


if __name__ == "__main__":
    main()
