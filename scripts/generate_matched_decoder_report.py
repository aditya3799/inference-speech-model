"""Generate eager/graph static/hybrid comparisons for every profile in JSON."""

import argparse
import json
from pathlib import Path


COMPARISONS = [
    ("static_eager_to_graph", "static_batching", "static_graph"),
    ("hybrid_before_to_after", "hybrid_before", "hybrid_after"),
    ("eager_architecture", "static_batching", "hybrid_before"),
    ("graph_architecture", "static_graph", "hybrid_after"),
]


def compare(name, left_name, right_name, rows):
    left, right = rows[left_name], rows[right_name]
    return {
        "name": name, "left": left_name, "right": right_name,
        "capacity_ratio": right["service_capacity_req_per_sec_median"] / left["service_capacity_req_per_sec_median"],
        "wall_time_ratio": right["wall_time_s_median"] / left["wall_time_s_median"],
        "p50_latency_ratio": right["latency_p50_s_median"] / left["latency_p50_s_median"],
        "p95_latency_ratio": right["latency_p95_s_median"] / left["latency_p95_s_median"],
    }


def derive(rows):
    return [compare(*item, rows) for item in COMPARISONS]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    profiles = data.get("profiles", {"burst": {"systems": data["speed_rows"]}})
    derived_profiles = {}
    lines = ["# Fresh matched decoder comparison", "", f"Session: `{data['session_id']}`; status: **{data['status']}**."]

    for profile_name, profile in profiles.items():
        rows = profile["systems"]
        rate_label = "Capacity req/s" if profile_name == "burst" else "Observed completion req/s (arrival-paced)"
        comparisons = derive(rows)
        derived_profiles[profile_name] = {"comparisons": comparisons}
        lines += ["", f"## {profile_name}", "",
                  f"| System | Decode | Wall s | {rate_label} | p50 latency s | p95 latency s | Exact % | WER % | Status |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---|"]
        for name, row in rows.items():
            status = "VALID" if row["exact_match_pct"] >= 99 else "INVALID"
            lines.append(
                f"| {name} | {data['system_configs'][name]['decode']} | {row['wall_time_s_median']:.3f} | "
                f"{row['service_capacity_req_per_sec_median']:.3f} | {row['latency_p50_s_median']:.3f} | "
                f"{row['latency_p95_s_median']:.3f} | {row['exact_match_pct']:.1f} | {row['wer_pct']:.3f} | {status} |"
            )
        lines += ["", f"| Matched comparison (right / left) | {rate_label} ratio | Wall-time ratio | p50 ratio | p95 ratio |",
                  "|---|---:|---:|---:|---:|---:|"]
        for item in comparisons:
            lines.append(
                f"| {item['name']} | {item['capacity_ratio']:.3f}x | {item['wall_time_ratio']:.3f}x | "
                f"{item['p50_latency_ratio']:.3f}x | {item['p95_latency_ratio']:.3f}x |"
            )

    data["derived_profiles"] = derived_profiles
    gate, canary = data["session_gate"], data["canary"]
    lines += ["", "## Session gate", "",
              "| Metric | Observed | Limit | Status |", "|---|---:|---:|---|",
              f"| Busy throttling | {gate['max_busy_throttle_pct']:.2f}% | <=10% | {'PASS' if gate['max_busy_throttle_pct'] <= 10 else 'FAIL'} |",
              f"| Clock spread | {gate['clock_spread_pct']:.2f}% | <=10% | {'PASS' if gate['clock_spread_pct'] <= 10 else 'FAIL'} |",
              f"| Canary drift | {canary['drift_pct']:.2f}% | <=15% | {'PASS' if canary['passed'] else 'FAIL'} |",
              f"| Maximum temperature | {gate['max_temperature_c']:.1f} C | <=88 C | {'PASS' if gate['max_temperature_c'] <= 88 else 'FAIL'} |",
              "", "Failing criteria:"]
    lines.extend([f"- {failure}" for failure in gate["failures"]] or ["- none"])

    lines += ["", "## JSON-derived analysis", ""]
    for profile_name, derived in derived_profiles.items():
        hybrid = next(x for x in derived["comparisons"] if x["name"] == "hybrid_before_to_after")
        graph = next(x for x in derived["comparisons"] if x["name"] == "graph_architecture")
        rate_word = "capacity" if profile_name == "burst" else "arrival-paced completion rate"
        lines.append(
            f"For {profile_name}, the matched hybrid after/before {rate_word} ratio was "
            f"{hybrid['capacity_ratio']:.3f}x and its p95 latency ratio was {hybrid['p95_latency_ratio']:.3f}x. "
            f"The matched graph hybrid/static {rate_word} ratio was {graph['capacity_ratio']:.3f}x."
        )
    lines.append(f"The telemetry gate {'passed' if gate['gate_passed'] else 'failed'}; results are {'eligible' if gate['gate_passed'] else 'not eligible'} for clean conclusions.")

    data["report_policy"] = {"conclusions_allowed": gate["gate_passed"]}
    args.input.write_text(json.dumps(data, indent=2), encoding="utf-8", newline="\n")
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("Generated", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
