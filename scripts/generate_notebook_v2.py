"""Generate the streamlined real-speech serving comparison notebook."""

import json
from pathlib import Path


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def py(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(keepends=True)}


cells = [
    md("""# Real-Speech Serving Comparison Lab

This notebook keeps the execution-first style of the original paged dual-cache notebook while making the production comparison compact and repeatable. Every row uses the same Whisper checkpoint, LibriSpeech subset, arrival schedule, token limit, and parity criteria.

### Systems compared

1. **Sequential** — one request at a time.
2. **Static batching** — fixed eager batches.
3. **Static graph** — fixed batches with CUDA-graph decode.
4. **Paged KV engine** — continuous batching with cross-KV gathered during decode.
5. **Cross-KV engine** — continuous batching with cross-KV gathered once and reused.
6. **Graph engine** — persistent CUDA-graph decoder buffers.
7. **Encoder-optimized graph engine** — batched encoder and direct encoder-KV-to-graph-slot handoff.

Correctness stays beside performance: WER and exact-token match are displayed in every main comparison table.
"""),
    md("""---
## 1. Environment Setup & Experiment Controls

`full` runs the configured real-speech experiment. `quick` uses a smaller sample and one measured repeat. `cached` only renders the existing JSON. Launch Jupyter from the repository root.
"""),
    py("""%load_ext autoreload
%autoreload 2

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
from IPython.display import display

REPO_ROOT = Path.cwd()
RUNNER = REPO_ROOT / "scripts" / "run_hybrid_before_after_burst.py"
OUTPUT_JSON = REPO_ROOT / "data" / "streamlined_serving_comparison.json"
if not RUNNER.exists():
    raise RuntimeError("Launch Jupyter from the repository root")

RUN_MODE = os.environ.get("REAL_SPEECH_RUN_MODE", "full")  # full | quick | cached
REPEATS = 3
NUM_REQUESTS = 100
INCLUDE_POISSON = True
SYSTEMS_TO_RUN = [
    "sequential",
    "static_batching",
    "static_graph",
    "hybrid_before",
    "hybrid_cross_kv",
    "hybrid_after",
    "hybrid_encoder_optimized",
]

if RUN_MODE == "quick":
    REPEATS, NUM_REQUESTS = 1, 20
assert RUN_MODE in {"full", "quick", "cached"}
print(f"Mode={RUN_MODE} | requests={NUM_REQUESTS} | repeats={REPEATS}")
print("Systems:", ", ".join(SYSTEMS_TO_RUN))
"""),
    py("""import torch
import transformers

try:
    import triton
    triton_version = triton.__version__
except ImportError:
    triton_version = "not installed"

environment = {
    "Python": sys.version.split()[0],
    "CUDA available": torch.cuda.is_available(),
    "GPU": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
    "PyTorch": torch.__version__,
    "Transformers": transformers.__version__,
    "Triton": triton_version,
}
display(pd.DataFrame(environment.items(), columns=["Component", "Value"]))
"""),
    md("""---
## 2. Confirm the Executable Configurations

The table comes directly from the runner. This prevents notebook labels from drifting away from the actual flags used by the engine.
"""),
    py("""from scripts.run_hybrid_before_after_burst import SYSTEMS

system_configs = {name: config for name, config in SYSTEMS}
config_rows = []
for name in SYSTEMS_TO_RUN:
    config = system_configs[name]
    config_rows.append({
        "System": name,
        "Kind": config["kind"],
        "Decoder": config["decode"],
        "Cross-KV reuse": config.get("cache_cross_kv", False),
        "CUDA graph": config.get("graph_decode", False),
        "Encoder batch": config.get("batch_encoder", False),
        "Direct encoder KV": config.get("graph_direct_encoder_kv", False),
        "Readback K": config.get("readback_k", 1),
    })
display(pd.DataFrame(config_rows).style.hide(axis="index"))
"""),
    md("""---
## 3. Run the Parity-Gated Comparison

The runner performs a correctness preflight, discarded warm-up, cyclic execution ordering, GPU/CPU telemetry, and a sequential canary. A failed telemetry gate does not erase results, but it prevents treating them as clean performance conclusions.
"""),
    py("""if RUN_MODE != "cached":
    command = [
        sys.executable, "-u", str(RUNNER),
        "--repeats", str(REPEATS),
        "--num-requests", str(NUM_REQUESTS),
        "--systems", ",".join(SYSTEMS_TO_RUN),
        "--output", str(OUTPUT_JSON),
    ]
    if INCLUDE_POISSON:
        command.append("--include-poisson-high")
    subprocess.run(command, cwd=REPO_ROOT, check=True)
elif not OUTPUT_JSON.exists():
    raise FileNotFoundError(f"Cached result not found: {OUTPUT_JSON}")

results = json.loads(OUTPUT_JSON.read_text(encoding="utf-8"))
print("Session:", results["session_id"])
print("Status:", results["status"])
"""),
    md("""---
## 4. Correctness Gate

Performance rows are valid only when exact token match is at least the runner threshold and WER remains within its allowed distance from the Hugging Face reference.
"""),
    py("""parity = pd.DataFrame([
    {
        "System": name,
        "Status": gate["status"],
        "Exact match (%)": gate["exact_match_pct"],
        "WER (%)": gate["wer_pct"],
        "HF WER (%)": gate["reference_wer_pct"],
        "WER delta (points)": gate["wer_delta_points"],
    }
    for name, gate in results["parity_gate"].items()
])
display(parity.round(3).style.hide(axis="index"))
"""),
    md("""---
## 5. Real-Speech Summary

This is the primary comparison. It follows the original notebook’s wide-table behavior: latency, TTFT, memory, WER, exact match, and throughput are visible together instead of being split across unrelated outputs.
"""),
    py("""summary_rows = []
for profile_name, profile in results["profiles"].items():
    for system_name, metrics in profile["systems"].items():
        summary_rows.append({
            "Traffic": profile_name,
            "System": system_name,
            "Status": metrics["status"],
            "Capacity (req/s)": metrics["service_capacity_req_per_sec_median"],
            "Audio throughput (x)": metrics["throughput_audio_per_sec_median"],
            "Wall (s)": metrics["wall_time_s_median"],
            "TTFT p50 (s)": metrics["ttft_p50_s_median"],
            "TTFT p95 (s)": metrics["ttft_p95_s_median"],
            "Latency p50 (s)": metrics["latency_p50_s_median"],
            "Latency p95 (s)": metrics["latency_p95_s_median"],
            "Peak VRAM (MB)": metrics["peak_vram_allocated_mb_median"],
            "WER (%)": metrics["wer_pct"],
            "Exact match (%)": metrics["exact_match_pct"],
        })

summary = pd.DataFrame(summary_rows)
for profile_name in results["profiles"]:
    print("=" * 120)
    print(profile_name.upper())
    display(
        summary[summary["Traffic"] == profile_name]
        .drop(columns="Traffic")
        .reset_index(drop=True)
        .round(3)
        .style.hide(axis="index")
        .background_gradient(subset=["Capacity (req/s)"], cmap="Greens")
        .background_gradient(subset=["Latency p95 (s)"], cmap="Reds")
    )
"""),
    md("""---
## 6. Streamlined Optimization Comparisons

Ratios are right/left. Capacity above `1.0x` is faster; latency below `1.0x` is lower. These comparisons isolate the main experimental questions without mixing unrelated architectures or decoder modes.
"""),
    py("""comparison_pairs = [
    ("sequential", "static_batching", "batching vs sequential"),
    ("static_batching", "static_graph", "graph effect on static"),
    ("hybrid_before", "hybrid_cross_kv", "one-time cross-KV reuse"),
    ("hybrid_cross_kv", "hybrid_after", "graph effect on engine"),
    ("hybrid_after", "hybrid_encoder_optimized", "encoder handoff optimization"),
    ("static_graph", "hybrid_encoder_optimized", "matched graph architecture"),
]

ratio_rows = []
for profile_name, profile in results["profiles"].items():
    systems = profile["systems"]
    for left, right, question in comparison_pairs:
        if left not in systems or right not in systems:
            continue
        a, b = systems[left], systems[right]
        ratio_rows.append({
            "Traffic": profile_name,
            "Question": question,
            "Comparison": f"{right} / {left}",
            "Capacity ratio": b["service_capacity_req_per_sec_median"] / a["service_capacity_req_per_sec_median"],
            "Wall ratio": b["wall_time_s_median"] / a["wall_time_s_median"],
            "TTFT p95 ratio": b["ttft_p95_s_median"] / a["ttft_p95_s_median"],
            "Latency p95 ratio": b["latency_p95_s_median"] / a["latency_p95_s_median"],
            "WER delta": b["wer_pct"] - a["wer_pct"],
        })

ratios = pd.DataFrame(ratio_rows)
display(ratios.round(3).style.hide(axis="index"))
"""),
    md("""---
## 7. Repeat-Level Results

Medians can hide order and thermal effects. Use these rows to inspect variability before accepting an optimization.
"""),
    py("""repeat_rows = []
for profile_name, profile in results["profiles"].items():
    for system_name, metrics in profile["systems"].items():
        for run in metrics["runs"]:
            repeat_rows.append({
                "Traffic": profile_name,
                "System": system_name,
                "Repeat": run["repeat"],
                "Order": run["order_index"],
                "Capacity (req/s)": run["service_capacity_req_per_sec"],
                "Latency p95 (s)": run["latency_p95_s"],
                "TTFT p95 (s)": run["ttft_p95_s"],
                "Exact match (%)": run["exact_match_pct"],
                "WER (%)": run["wer_pct"],
                "Run gate": run["run_gate_passed"],
            })
display(pd.DataFrame(repeat_rows).round(3).style.hide(axis="index"))
"""),
    md("""---
## 8. Session Quality Gate

Do not publish speed conclusions unless this gate passes. A parity-valid but telemetry-invalid run remains useful for debugging only.
"""),
    py("""gate = results["session_gate"]
canary = results["canary"]
gate_rows = [{
    "Session status": results["status"],
    "Gate passed": gate["gate_passed"],
    "Busy throttling (%)": gate["max_busy_throttle_pct"],
    "Clock spread (%)": gate["clock_spread_pct"],
    "Max temperature (C)": gate["max_temperature_c"],
    "Canary drift (%)": canary["drift_pct"],
}]
display(pd.DataFrame(gate_rows).round(3).style.hide(axis="index"))

if gate["failures"]:
    print("Failing criteria:")
    for failure in gate["failures"]:
        print(" -", failure)
else:
    print("All telemetry criteria passed.")
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

output = Path(__file__).resolve().parent.parent / "real_speech_serving_v2.ipynb"
output.write_text(json.dumps(notebook, indent=1), encoding="utf-8", newline="\n")
print(f"Generated {output}")
