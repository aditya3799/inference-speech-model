"""Generate the burst-only five-system decoder comparison notebook."""

import json
from pathlib import Path


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def py(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(keepends=True)}


cells = [
    md("""# Hybrid Decoder Before/After — Matched Static Graph Comparison

This notebook compares the same 100 real-speech utterances under burst traffic using three interleaved measured repeats:

1. `sequential` — eager, batch size 1
2. `static_batching` — eager fixed batching
3. `static_graph` — fixed batching through `GraphDecodeBuffers`
4. `hybrid_before` — original eager hybrid decoder
5. `hybrid_after` — optimized hybrid graph decoder

It performs a 100-utterance correctness gate before timing, a discarded interleaved warm-up, GPU cooldowns, and sequential canaries at the start and end. Every speed row prints exact match and WER; exact match below 99% is marked `INVALID`.
"""),
    py("""import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path.cwd()
RUNNER = REPO_ROOT / "scripts" / "run_hybrid_before_after_burst.py"
OUTPUT_JSON = REPO_ROOT / "data" / "hybrid_before_after_burst.json"
if not RUNNER.exists():
    raise RuntimeError("Launch Jupyter from the repository root")

RUN_MODE = os.environ.get("REAL_SPEECH_RUN_MODE", "full")
REPEATS = 3
assert RUN_MODE in {"full", "cached"}
print({"run_mode": RUN_MODE, "repeats": REPEATS, "python": sys.executable})
"""),
    md("""## 1. Code-path configuration

The table is loaded from the executable runner configuration, rather than manually duplicated in the notebook.
"""),
    py("""from scripts.run_hybrid_before_after_burst import SYSTEMS

config_table = pd.DataFrame([
    {"system": name, **cfg, "buckets": "64,128,256,448" if cfg["decode"] == "cuda_graph" else "N/A"}
    for name, cfg in SYSTEMS
])
display(config_table)
"""),
    md("""## 2. Execute

`full` runs the complete parity-gated session. Set `REAL_SPEECH_RUN_MODE=cached` before starting Jupyter only when rendering an existing result.
"""),
    py("""if RUN_MODE == "full":
    subprocess.run(
        [sys.executable, "-u", str(RUNNER), "--repeats", str(REPEATS), "--output", str(OUTPUT_JSON)],
        cwd=REPO_ROOT,
        check=True,
    )
elif not OUTPUT_JSON.exists():
    raise FileNotFoundError("No cached result exists")

results = json.loads(OUTPUT_JSON.read_text(encoding="utf-8"))
print("Session:", results["session_id"])
print("Status:", results["status"])
"""),
    md("""## 3. Correctness preflight

The performance run proceeds only if exact match is at least 99% and absolute WER difference from the Hugging Face reference is at most 0.2 percentage points.
"""),
    py("""parity = pd.DataFrame([
    {"system": name, **gate}
    for name, gate in results["parity_gate"].items()
])
display(parity.round(3))
"""),
    md("""## 4. Median speed and correctness rows

Correctness is deliberately printed beside performance. Do not compare any row marked `INVALID`.
"""),
    py("""speed = pd.DataFrame([
    {
        "system": name,
        "status": row["status"],
        "capacity_req_s": row["service_capacity_req_per_sec_median"],
        "wall_time_s": row["wall_time_s_median"],
        "audio_throughput_x": row["throughput_audio_per_sec_median"],
        "TTFT_p50_s": row["ttft_p50_s_median"],
        "TTFT_p95_s": row["ttft_p95_s_median"],
        "latency_p50_s": row["latency_p50_s_median"],
        "latency_p95_s": row["latency_p95_s_median"],
        "peak_VRAM_MB": row["peak_vram_allocated_mb_median"],
        "exact_match_pct": row["exact_match_pct"],
        "WER_pct": row["wer_pct"],
    }
    for name, row in results["speed_rows"].items()
])
display(speed.round(3))
"""),
    md("""## 5. Hybrid before/after and matched static comparison

This distinguishes the practical improvement (`hybrid_before` → `hybrid_after`) from the architecture comparison (`static_graph` versus `hybrid_after`).
"""),
    py("""rows = results["speed_rows"]

def delta(left, right):
    a, b = rows[left], rows[right]
    return {
        "comparison": f"{left} -> {right}",
        "capacity_speedup_x": b["service_capacity_req_per_sec_median"] / a["service_capacity_req_per_sec_median"],
        "capacity_change_pct": 100 * (b["service_capacity_req_per_sec_median"] / a["service_capacity_req_per_sec_median"] - 1),
        "wall_reduction_pct": 100 * (1 - b["wall_time_s_median"] / a["wall_time_s_median"]),
        "p50_latency_reduction_pct": 100 * (1 - b["latency_p50_s_median"] / a["latency_p50_s_median"]),
        "p95_latency_reduction_pct": 100 * (1 - b["latency_p95_s_median"] / a["latency_p95_s_median"]),
        "VRAM_delta_MB": b["peak_vram_allocated_mb_median"] - a["peak_vram_allocated_mb_median"],
        "left_exact_pct": a["exact_match_pct"], "right_exact_pct": b["exact_match_pct"],
        "left_WER_pct": a["wer_pct"], "right_WER_pct": b["wer_pct"],
    }

display(pd.DataFrame([
    delta("hybrid_before", "hybrid_after"),
    delta("static_batching", "static_graph"),
    delta("static_graph", "hybrid_after"),
]).round(3))
"""),
    md("""## 6. Repeat-level rows

These rows expose execution order and variation hidden by medians. Exact match and WER remain adjacent to every speed measurement.
"""),
    py("""repeat_rows = []
for name, summary in results["speed_rows"].items():
    for run in summary["runs"]:
        repeat_rows.append({
            "system": name, "repeat": run["repeat"], "order_index": run["order_index"],
            "status": run["status"], "capacity_req_s": run["service_capacity_req_per_sec"],
            "wall_time_s": run["wall_time_s"], "latency_p95_s": run["latency_p95_s"],
            "exact_match_pct": run["exact_match_pct"], "WER_pct": run["wer_pct"],
            "busy_clock_MHz": run["telemetry_gpu"]["busy_median_clock_mhz"],
            "busy_throttle_pct": run["telemetry_gpu"]["busy_throttle_sample_pct"],
            "run_gate": "PASS" if run["run_gate_passed"] else "FAIL",
        })
display(pd.DataFrame(repeat_rows).sort_values(["repeat", "order_index"]).round(3))
"""),
    md("""## 7. Session gate

The session is clean only when busy throttling is at most 10%, cross-system busy median clock spread is at most 10%, temperature is at most 88°C, and sequential canary drift is at most 15%.
"""),
    py("""gate = results["session_gate"]
canary = results["canary"]
gate_table = pd.DataFrame([{
    "session_status": results["status"],
    "gate_passed": gate["gate_passed"],
    "busy_throttle_fraction": gate["max_busy_throttle_pct"] / 100.0,
    "busy_throttle_pct": gate["max_busy_throttle_pct"],
    "clock_spread_pct": gate["clock_spread_pct"],
    "canary_start_audio_x": canary["start_throughput_audio_per_sec"],
    "canary_end_audio_x": canary["end_throughput_audio_per_sec"],
    "canary_drift_pct": canary["drift_pct"],
    "max_temperature_C": gate["max_temperature_c"],
    "failures": "; ".join(gate["failures"]) or "none",
}])
display(gate_table.round(3))
"""),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

output = Path(__file__).resolve().parent.parent / "hybrid_decoder_before_after.ipynb"
output.write_text(json.dumps(notebook, indent=1), encoding="utf-8", newline="\n")
print(f"Generated {output}")
