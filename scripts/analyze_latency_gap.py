"""
scripts/analyze_latency_gap.py

Decomposes end-to-end latency per request into:
  (a) Admission wait = start_time - arrival_time
  (b) Start-to-first-token (Prefill/TTFT) = first_token_time - start_time
  (c) First-token-to-finish (Decode) = finish_time - first_token_time

Compares static batching vs continuous engine (flag ON) on poisson_high, burst, and poisson_low.
Reports mean, p50, p95, shares of p95 latency, and arrival rate vs achieved service rate.
"""

import sys
import json
import argparse
from pathlib import Path
import numpy as np
from typing import Dict, Any, List


def analyze_system_latency(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    adm_waits = []
    prefills = []
    decodes = []
    e2es = []
    wall_times = []

    for r in runs:
        wall_times.append(r["wall_time_s"])
        for req in r.get("per_request", []):
            arr = req["arrival_time"]
            start = req["start_time"]
            first = req["first_token_time"]
            fin = req["finish_time"]

            adm_waits.append(max(0.0, start - arr))
            prefills.append(max(0.0, first - start))
            decodes.append(max(0.0, fin - first))
            e2es.append(max(0.0, fin - arr))

    n_req = len(e2es)
    total_reqs = sum(len(r.get("per_request", [])) for r in runs)
    mean_wall = float(np.mean(wall_times)) if wall_times else 0.0
    service_rate = (len(runs[0].get("per_request", [])) / mean_wall) if mean_wall > 0 and runs else 0.0

    return {
        "num_requests_total": n_req,
        "mean_wall_time_s": mean_wall,
        "service_rate_req_per_s": service_rate,
        "admission_wait": {
            "mean": float(np.mean(adm_waits)),
            "std": float(np.std(adm_waits)),
            "p50": float(np.percentile(adm_waits, 50)),
            "p95": float(np.percentile(adm_waits, 95)),
        },
        "start_to_first_token": {
            "mean": float(np.mean(prefills)),
            "std": float(np.std(prefills)),
            "p50": float(np.percentile(prefills, 50)),
            "p95": float(np.percentile(prefills, 95)),
        },
        "first_token_to_finish": {
            "mean": float(np.mean(decodes)),
            "std": float(np.std(decodes)),
            "p50": float(np.percentile(decodes, 50)),
            "p95": float(np.percentile(decodes, 95)),
        },
        "e2e_latency": {
            "mean": float(np.mean(e2es)),
            "std": float(np.std(e2es)),
            "p50": float(np.percentile(e2es, 50)),
            "p95": float(np.percentile(e2es, 95)),
        },
    }


def run_latency_decomposition(json_path: str, profile_filter: str = "poisson_high") -> Dict[str, Any]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    profiles = [profile_filter] if profile_filter in data["systems"] else list(data["systems"].keys())
    results = {}

    arrival_rates = {
        "poisson_high": 4.0,
        "poisson_low": 1.0,
        "burst": float("inf"),
    }

    for prof in profiles:
        prof_data = data["systems"][prof]
        results[prof] = {}
        arr_rate = arrival_rates.get(prof, 0.0)

        for sys_name in ["static_batching", "engine_flag_on", "engine_flag_off", "sequential"]:
            if sys_name in prof_data:
                results[prof][sys_name] = analyze_system_latency(prof_data[sys_name]["runs"])

        # Compute gap decomposition between engine_flag_on and static_batching
        if "static_batching" in results[prof] and "engine_flag_on" in results[prof]:
            sb = results[prof]["static_batching"]
            eng = results[prof]["engine_flag_on"]

            gap_total_p95 = eng["e2e_latency"]["p95"] - sb["e2e_latency"]["p95"]
            gap_adm_p95 = eng["admission_wait"]["p95"] - sb["admission_wait"]["p95"]
            gap_ttft_p95 = eng["start_to_first_token"]["p95"] - sb["start_to_first_token"]["p95"]
            gap_dec_p95 = eng["first_token_to_finish"]["p95"] - sb["first_token_to_finish"]["p95"]

            gap_total_mean = eng["e2e_latency"]["mean"] - sb["e2e_latency"]["mean"]
            gap_adm_mean = eng["admission_wait"]["mean"] - sb["admission_wait"]["mean"]
            gap_ttft_mean = eng["start_to_first_token"]["mean"] - sb["start_to_first_token"]["mean"]
            gap_dec_mean = eng["first_token_to_finish"]["mean"] - sb["first_token_to_finish"]["mean"]

            results[prof]["gap_analysis"] = {
                "arrival_rate_req_per_s": arr_rate,
                "static_service_rate": sb["service_rate_req_per_s"],
                "engine_service_rate": eng["service_rate_req_per_s"],
                "engine_queue_unstable": (eng["service_rate_req_per_s"] < arr_rate) if arr_rate != float("inf") else True,
                "p95_gap_total_s": gap_total_p95,
                "p95_gap_shares": {
                    "admission_wait_gap_s": gap_adm_p95,
                    "admission_wait_share_pct": (gap_adm_p95 / gap_total_p95 * 100.0) if gap_total_p95 > 0 else 0.0,
                    "prefill_gap_s": gap_ttft_p95,
                    "prefill_share_pct": (gap_ttft_p95 / gap_total_p95 * 100.0) if gap_total_p95 > 0 else 0.0,
                    "decode_gap_s": gap_dec_p95,
                    "decode_share_pct": (gap_dec_p95 / gap_total_p95 * 100.0) if gap_total_p95 > 0 else 0.0,
                },
                "mean_gap_total_s": gap_total_mean,
                "mean_gap_shares": {
                    "admission_wait_gap_s": gap_adm_mean,
                    "admission_wait_share_pct": (gap_adm_mean / gap_total_mean * 100.0) if gap_total_mean > 0 else 0.0,
                    "prefill_gap_s": gap_ttft_mean,
                    "prefill_share_pct": (gap_ttft_mean / gap_total_mean * 100.0) if gap_total_mean > 0 else 0.0,
                    "decode_gap_s": gap_dec_mean,
                    "decode_share_pct": (gap_dec_mean / gap_total_mean * 100.0) if gap_total_mean > 0 else 0.0,
                }
            }

    return results


def print_markdown_table(results: Dict[str, Any], profile: str = "poisson_high"):
    if profile not in results:
        print(f"Profile {profile} not in results")
        return

    data = results[profile]
    print(f"\n### Latency Decomposition: {profile.upper()}\n")

    print("| System | Phase | Mean (s) | p50 (s) | p95 (s) | Share of System p95 (%) |")
    print("|:---|:---|:---:|:---:|:---:|:---:|")

    for sys_name in ["static_batching", "engine_flag_on"]:
        if sys_name not in data:
            continue
        s_data = data[sys_name]
        tot_p95 = s_data["e2e_latency"]["p95"]

        phases = [
            ("(a) Admission Wait (start - arrival)", s_data["admission_wait"]),
            ("(b) Prefill/TTFT (first_token - start)", s_data["start_to_first_token"]),
            ("(c) Decode (finish - first_token)", s_data["first_token_to_finish"]),
            ("Total End-to-End Latency", s_data["e2e_latency"]),
        ]

        for p_label, p_stats in phases:
            share = (p_stats["p95"] / tot_p95 * 100.0) if p_label != "Total End-to-End Latency" else 100.0
            print(f"| {sys_name} | {p_label} | {p_stats['mean']:.4f}s | {p_stats['p50']:.4f}s | {p_stats['p95']:.4f}s | {share:.1f}% |")

    if "gap_analysis" in data:
        ga = data["gap_analysis"]
        print(f"\n#### Gap Contribution Analysis (Engine flag ON minus Static Batching)")
        print(f"- **Arrival Rate ($\lambda$):** {ga['arrival_rate_req_per_s']} req/s")
        print(f"- **Static Service Rate ($\mu_{{sb}}$):** {ga['static_service_rate']:.2f} req/s (Queue stable: $\lambda < \mu$)")
        print(f"- **Engine Service Rate ($\mu_{{eng}}$):** {ga['engine_service_rate']:.2f} req/s (Queue unstable: $\lambda > \mu \implies {ga['engine_queue_unstable']}$)")
        print(f"- **Total p95 Latency Gap:** {ga['p95_gap_total_s']:.4f}s")
        print(f"  - **Admission Wait Gap:** {ga['p95_gap_shares']['admission_wait_gap_s']:.4f}s ({ga['p95_gap_shares']['admission_wait_share_pct']:.1f}% of gap)")
        print(f"  - **Prefill/TTFT Gap:** {ga['p95_gap_shares']['prefill_gap_s']:.4f}s ({ga['p95_gap_shares']['prefill_share_pct']:.1f}% of gap)")
        print(f"  - **Decode Step Gap:** {ga['p95_gap_shares']['decode_gap_s']:.4f}s ({ga['p95_gap_shares']['decode_share_pct']:.1f}% of gap)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, default="data/real_speech_evaluation_results_v2.json")
    parser.add_argument("--profile", type=str, default="poisson_high")
    args = parser.parse_args()

    res = run_latency_decomposition(args.json, args.profile)
    print_markdown_table(res, args.profile)
