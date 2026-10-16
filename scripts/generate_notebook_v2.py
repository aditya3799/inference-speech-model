"""Generate the five-stage, 100-utterance real-speech comparison notebook."""

import json
from pathlib import Path


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def py(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(keepends=True)}


cells = [
    md("""# Real-Speech Serving: Five-Stage Optimization Comparison

This notebook compares the complete progression on the same 100 LibriSpeech utterances under **burst** and **Poisson-high (4 requests/s)** traffic:

1. `sequential`
2. `static_batching`
3. `hybrid_engine` — paged continuous batching with per-step cross-KV gathers
4. `hybrid_one_time_cross_kv` — cross-KV gathered once at handoff and reused
5. `hybrid_graph_no_host_sync` — one-time cross-KV plus removed per-step host tensor creation/CPU token sync, CUDA graph replay, GPU argmax, persistent buffers, and K=4 readback

Every implementation must first pass the 100-utterance parity gate. A failing implementation is not benchmarked.
"""),
    py("""import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path.cwd()
if not (REPO_ROOT / "scripts" / "run_five_system_comparison.py").exists():
    raise RuntimeError("Launch Jupyter from the repository root")

PYTHON = sys.executable
OUTPUT_JSON = REPO_ROOT / "data" / "real_speech_five_system_comparison.json"

# full runs the complete comparison. cached only renders an existing result.
RUN_MODE = os.environ.get("REAL_SPEECH_RUN_MODE", "full")
REPEATS = 3
assert RUN_MODE in {"full", "cached"}
print("Run mode:", RUN_MODE)
print("Measured repeats:", REPEATS)
print("Python:", PYTHON)
"""),
    md("""## 1. Environment

The benchmark uses the current notebook kernel's Python environment. Confirm that CUDA and the pinned package versions are active before starting the long run.
"""),
    py("""import torch
import transformers
import triton

print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("Torch:", torch.__version__)
print("Transformers:", transformers.__version__)
print("Triton:", triton.__version__)
subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,clocks.gr,power.draw", "--format=csv,noheader"], check=False)
"""),
    md("""## 2. Confirm the five implementations

These are separate runtime configurations, not renamed copies. In particular, the legacy hybrid engine gathers encoder cross-KV from paged blocks on every decode step, while the one-time version caches it at handoff. The final graph version additionally removes per-step host tensor construction and per-token CPU synchronization.
"""),
    py("""from scripts.run_five_system_comparison import DESCRIPTIONS, SYSTEMS

for index, (name, options) in enumerate(SYSTEMS, 1):
    print(f"{index}. {name}")
    print("   options:", options)
    print("   ", DESCRIPTIONS[name])
"""),
    md("""## 3. Execute parity gate and measured comparison

The runner performs:

- 100-utterance burst parity preflight for all five systems;
- a discarded warm-up for every system;
- three measured repeats of burst and Poisson-high traffic;
- cyclic system-order rotation to reduce systematic thermal/order bias;
- median latency, throughput, capacity, VRAM, WER, and exact-match aggregation.

Expect this cell to take a substantial amount of time.
"""),
    py("""if RUN_MODE == "full":
    subprocess.run(
        [PYTHON, "-u", "scripts/run_five_system_comparison.py", "--repeats", str(REPEATS)],
        cwd=REPO_ROOT,
        check=True,
    )
elif not OUTPUT_JSON.exists():
    raise FileNotFoundError("No cached result exists. Set RUN_MODE='full'.")

results = json.loads(OUTPUT_JSON.read_text(encoding="utf-8"))
print("Result status:", results["status"])
assert results["status"] == "VALID", results.get("invalid_systems")
"""),
    md("""## 4. Parity gate

Exact match must be at least 99%, and WER must remain within 0.2 percentage points of the Hugging Face reference.
"""),
    py("""import pandas as pd

parity_rows = []
for name, gate in results["parity_gate"].items():
    parity_rows.append({"system": name, **gate})
parity_table = pd.DataFrame(parity_rows)
display(parity_table)
"""),
    md("""## 5. Real-speech comparison tables

Each row is the median of the measured repeats. Burst service capacity is `100 / wall_time`; Poisson throughput is arrival-limited, so latency and queueing are the important measures there.
"""),
    py("""comparison_rows = []
for profile_name, profile in results["profiles"].items():
    for system_name, metrics in profile["systems"].items():
        comparison_rows.append({
            "profile": profile_name,
            "system": system_name,
            "capacity_req_s": metrics["service_capacity_req_per_sec_median"],
            "audio_throughput_x": metrics["throughput_audio_per_sec_median"],
            "wall_time_s": metrics["wall_time_s_median"],
            "TTFT_p50_s": metrics["ttft_p50_s_median"],
            "TTFT_p95_s": metrics["ttft_p95_s_median"],
            "latency_p50_s": metrics["latency_p50_s_median"],
            "latency_p95_s": metrics["latency_p95_s_median"],
            "peak_VRAM_MB": metrics["peak_vram_allocated_mb_median"],
            "WER_pct": metrics["wer_pct"],
            "exact_match_pct": metrics["exact_match_pct"],
        })

comparison = pd.DataFrame(comparison_rows)
for profile_name in results["profiles"]:
    print("\\n", profile_name.upper())
    display(comparison[comparison["profile"] == profile_name].reset_index(drop=True))
"""),
    md("""## 6. Incremental effect of each optimization

This table makes the requested progression explicit. Positive capacity change and negative latency change are improvements.
"""),
    py("""system_order = [name for name, _ in SYSTEMS]
delta_rows = []
for profile_name, profile in results["profiles"].items():
    previous_name = None
    for system_name in system_order:
        current = profile["systems"][system_name]
        if previous_name is not None:
            previous = profile["systems"][previous_name]
            delta_rows.append({
                "profile": profile_name,
                "transition": f"{previous_name} -> {system_name}",
                "capacity_change_pct": (
                    current["service_capacity_req_per_sec_median"] / previous["service_capacity_req_per_sec_median"] - 1
                ) * 100,
                "latency_p50_change_pct": (
                    current["latency_p50_s_median"] / previous["latency_p50_s_median"] - 1
                ) * 100,
                "latency_p95_change_pct": (
                    current["latency_p95_s_median"] / previous["latency_p95_s_median"] - 1
                ) * 100,
                "VRAM_change_MB": current["peak_vram_allocated_mb_median"] - previous["peak_vram_allocated_mb_median"],
            })
        previous_name = system_name

display(pd.DataFrame(delta_rows))
"""),
    md("""## Interpretation

The comparison that directly answers Nathan's question is:

- `hybrid_one_time_cross_kv` versus `hybrid_graph_no_host_sync`

They share the one-time cross-KV optimization; the principal difference is removal of per-step host tensor construction and per-token CPU synchronization through persistent CUDA-graph buffers and K=4 readback. Use burst capacity for saturation behavior and Poisson-high latency for queueing impact.
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
