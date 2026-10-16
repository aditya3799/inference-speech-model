"""Generate the phase-timing Markdown report exclusively from result JSON."""

import argparse
import json
from pathlib import Path


def value(item, key, digits=3):
    return "not measured" if not isinstance(item, dict) or key not in item else f"{item[key]:.{digits}f}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    method = data["methodology"]
    lines = [
        "# Phase-timing diagnosis", "",
        f"Session status: **{data['status']}**.", "",
        f"Traffic: {method['traffic']}; utterances: {method['utterances']}; measured repeats: {method['repeats']}; "
        f"discarded warm-up cycles: {method['discarded_warmup_cycles']}; interleaving: {method['interleaving']}.", "",
        f"Phase semantics: {method['request_phase_semantics']}.", "",
        f"Power state: {data['environment'].get('power_state', 'not measured')}.", "",
        "## Correctness and capacity", "",
        "| System | Capacity req/s | Wall s | Exact % | WER % | Status |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for name, system in data["systems"].items():
        parity = data["parity_gate"][name]
        status = "VALID" if parity["exact_match_pct"] >= 99.0 else "INVALID"
        lines.append(
            f"| {name} | {system['capacity_req_per_s_median']:.3f} | {system['wall_time_s_median']:.3f} | "
            f"{parity['exact_match_pct']:.1f} | {parity['wer_pct']:.3f} | {status} |"
        )

    lines += ["", "## Per-request phase totals", "",
              "| Phase | Static total ms | Static mean ms/request | Static % request wall | Hybrid total ms | Hybrid mean ms/request | Hybrid % request wall | Hybrid - static mean ms |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    static = data["systems"]["static_graph"]
    hybrid = data["systems"]["hybrid_after"]
    difference = {row["phase"]: row for row in data["phase_difference_ranked"]}
    for phase in method.get("reported_phases", static["phases"].keys()):
        s, h, d = static["phases"][phase], hybrid["phases"][phase], difference[phase]
        lines.append(
            f"| {phase} | {value(s, 'total_ms')} | {value(s, 'mean_ms_per_request')} | {value(s, 'pct_of_request_wall_time')} | "
            f"{value(h, 'total_ms')} | {value(h, 'mean_ms_per_request')} | {value(h, 'pct_of_request_wall_time')} | "
            f"{value(d, 'hybrid_minus_static_mean_ms')} |"
        )

    lines += ["", "## Ranked contribution to capacity gap", "",
              "| Rank | Phase | Hybrid - static mean ms/request |", "|---:|---|---:|"]
    for rank, row in enumerate(data["phase_difference_ranked"], 1):
        lines.append(f"| {rank} | {row['phase']} | {value(row, 'hybrid_minus_static_mean_ms')} |")

    lines += ["", "## Batch and profiler timing", "",
              "| System | Actual batch-of-4 mean ms | Equivalent batch-of-4 ms | GPU kernel union ms | Idle GPU ms | Idle GPU % |",
              "|---|---:|---:|---:|---:|---:|"]
    for name, system in data["systems"].items():
        actual = system["actual_batch4_wall_ms"]
        profiler = system["profiler_gpu"]
        lines.append(
            f"| {name} | {value(actual, 'mean_ms')} | {system['equivalent_batch4_wall_ms']:.3f} | "
            f"{value(profiler, 'gpu_kernel_union_ms_median')} | {value(profiler, 'idle_gpu_time_ms_median')} | "
            f"{value(profiler, 'idle_gpu_fraction_pct_median')} |"
        )

    lines += ["", "## Slot hold after EOS", "",
              "| System | Total steps | Mean steps/request |", "|---|---:|---:|"]
    for name, system in data["systems"].items():
        hold = system["slot_hold_steps_after_eos"]
        lines.append(f"| {name} | {hold['total']} | {hold['mean_per_request']:.3f} |")

    lines += ["", "## Instrumentation overhead", "",
              "| System | Flag off wall s | Flag on wall s | Difference s | Overhead % |", "|---|---:|---:|---:|---:|"]
    for name, row in data["instrumentation_overhead"].items():
        lines.append(
            f"| {name} | {row['flag_off_wall_time_s']:.3f} | {row['flag_on_wall_time_s']:.3f} | "
            f"{row['overhead_s']:.3f} | {row['overhead_pct']:.2f} |"
        )

    gate = data["session_gate"]
    lines += ["", "## Session gate", "",
              "| Criterion | Observed | Required | Status |", "|---|---:|---:|---|"]
    gate_rows = [
        ("Busy throttle", gate["max_busy_throttle_pct"], 10.0, gate["max_busy_throttle_pct"] <= 10.0, "%"),
        ("Clock spread", gate["clock_spread_pct"], 10.0, gate["clock_spread_pct"] <= 10.0, "%"),
        ("Canary drift", gate["canary_drift_pct"], 15.0, gate["canary_drift_pct"] <= 15.0, "%"),
        ("Maximum temperature", gate["max_temperature_c"], 88.0, gate["max_temperature_c"] <= 88.0, " C"),
    ]
    for label, observed, required, passed, unit in gate_rows:
        lines.append(f"| {label} | {observed:.2f}{unit} | <= {required:.2f}{unit} | {'PASS' if passed else 'FAIL'} |")
    lines += ["", "Failing criteria:"]
    if gate["failures"]:
        lines.extend(f"- {failure}" for failure in gate["failures"])
    else:
        lines.append("- none")

    positive = [r for r in data["phase_difference_ranked"] if isinstance(r.get("hybrid_minus_static_mean_ms"), (int, float)) and r["hybrid_minus_static_mean_ms"] > 0]
    lines += ["", "## JSON-derived summary", ""]
    if positive:
        top = positive[0]
        lines.append(f"The largest measured positive hybrid-minus-static phase difference was `{top['phase']}` at {top['hybrid_minus_static_mean_ms']:.3f} ms per request.")
    else:
        lines.append("No measured phase had a positive hybrid-minus-static mean difference.")
    lines.append(f"The session gate {'passed' if gate['gate_passed'] else 'failed'}; performance conclusions are {'eligible' if gate['gate_passed'] else 'not eligible'} for clean comparison.")

    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("Generated", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
