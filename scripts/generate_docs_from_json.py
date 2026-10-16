"""Regenerate FINDINGS.md numeric results from benchmark JSON.

README.md uses the newer sectioned median benchmark document and is maintained
separately so this legacy matched-run generator cannot overwrite it.
"""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MATCHED_PATH = ROOT / "data" / "matched_decoder_comparison_burst_poisson.json"
PARITY_PATH = ROOT / "data" / "serving_parity_graph.json"


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def comparison(profile, name):
    return next(item for item in profile["comparisons"] if item["name"] == name)


def ratio_table(matched):
    labels = {
        "static_eager_to_graph": "static graph / static eager",
        "hybrid_before_to_after": "hybrid graph / hybrid eager",
        "eager_architecture": "hybrid / static, eager decoder",
        "graph_architecture": "hybrid / static, graphed decoder",
    }
    lines = [
        "| Traffic | Matched comparison | Completion-rate ratio | Wall-time ratio | p50 ratio | p95 ratio |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for profile_name, profile in matched["derived_profiles"].items():
        for item in profile["comparisons"]:
            lines.append(
                f"| {profile_name} | {labels[item['name']]} | {item['capacity_ratio']:.3f}x | "
                f"{item['wall_time_ratio']:.3f}x | {item['p50_latency_ratio']:.3f}x | "
                f"{item['p95_latency_ratio']:.3f}x |"
            )
    return "\n".join(lines)


def environment_table(matched):
    env = matched["environment"]
    return "\n".join([
        "| Component | Tested version |",
        "|---|---|",
        f"| Python | {env['python']} |",
        f"| PyTorch | {env['torch']} |",
        f"| Transformers | {env['transformers']} |",
        f"| Triton | {env['triton']} |",
        f"| NVIDIA driver | {env['driver']} |",
    ])


def parity_table(parity):
    lines = [
        "| Traffic | Requests | Exact token match | WER | Reference WER |",
        "|---|---:|---:|---:|---:|",
    ]
    for profile in parity["profiles"].values():
        lines.append(
            f"| {profile['name']} | {profile['num_requests']} | {profile['exact_match_pct']:.1f}% | "
            f"{profile['wer_pct']:.3f}% | {profile['reference_wer_pct']:.3f}% |"
        )
    return "\n".join(lines)


def write_docs():
    matched = load(MATCHED_PATH)
    parity = load(PARITY_PATH)
    ratios = ratio_table(matched)
    environment = environment_table(matched)
    parity_rows = parity_table(parity)
    gate = matched["session_gate"]
    canary = matched["canary"]

    readme = f"""# Paged Dual-Cache Speech Serving Engine

This repository explores a phase-aware Whisper serving engine with paged encoder and decoder KV caches, eager and CUDA-graph decode paths, real-speech parity checks, and reproducible benchmark reports.

The current matched benchmark is useful as an observed run, but not as a clean performance conclusion: its telemetry gate failed because cross-system busy-clock spread was {gate['clock_spread_pct']:.2f}% and sequential-canary drift was {canary['drift_pct']:.2f}%. Source: [`data/matched_decoder_comparison_burst_poisson.json`](data/matched_decoder_comparison_burst_poisson.json).

## Architecture: Dual-Cache Speech Serving

The serving path separates long-lived encoder cross-attention KV from autoregressive decoder KV. `EncoderBlockManager` and `BlockManager` own the physical pools; `DualCacheManager` coordinates allocation; `PhaseAwareSpeechScheduler` admits, encodes, prefills, batches decode work, and releases blocks deterministically.

The eager decoder remains the default. CUDA-graph decode is opt-in through `graph_decode=True`; graph buffers, slots, static masks, and captured buckets are not created when the flag is off. Optional phase timing is independently default-off.

### Key Architectural Invariants

- Encoder positions, not mel frames, determine encoder-block admission.
- Shared encoder blocks are read-only on cache hits; only freshly allocated blocks are written.
- Decoder blocks and graph slots are released on completion and failure paths.
- Variable-length batches carry explicit self-attention and cross-attention masks.
- The graph path uses fixed buffers and buckets while preserving the eager path as the compatibility baseline.
- Whisper prompt tokens, EOS handling, suppression behavior, and cached cross-attention projections must preserve generation parity.

## Matched decoder results

All figures below are regenerated from the same four-system run. Ratios are right/left, so they compare the same architecture across decoder modes or the same decoder mode across architectures. This replaces the former unmatched eager-static versus graphed-engine comparison.

{ratios}

The headline matched ratios are: burst hybrid graph/eager completion rate {comparison(matched['derived_profiles']['burst'], 'hybrid_before_to_after')['capacity_ratio']:.3f}x, burst graph hybrid/static completion rate {comparison(matched['derived_profiles']['burst'], 'graph_architecture')['capacity_ratio']:.3f}x, arrival-paced hybrid graph/eager completion rate {comparison(matched['derived_profiles']['poisson_high_4_0'], 'hybrid_before_to_after')['capacity_ratio']:.3f}x, and arrival-paced graph hybrid/static completion rate {comparison(matched['derived_profiles']['poisson_high_4_0'], 'graph_architecture')['capacity_ratio']:.3f}x.

## Serving parity

{parity_rows}

Source: [`data/serving_parity_graph.json`](data/serving_parity_graph.json). The parity gate passed even though one utterance differed in each traffic profile; WER matched the reference.

## Tested environment

{environment}

These are the versions recorded by the accepted matched JSON and therefore the versions pinned in `requirements.txt`.

## Repository structure

- `speech_serving/`: model wrapper, cache managers, scheduler, graph buffers, and timing support.
- `scripts/`: benchmark, parity, profiling, report, notebook, and documentation generators.
- `data/`: source JSON plus generated Markdown reports.
- `test_speech_engine.py`: unit and integration coverage for eager, cache, pretrained, and graph behavior.
- `real_speech_serving_v2.ipynb`: real-speech evaluation notebook.
- `hybrid_decoder_before_after.ipynb`: matched decoder comparison notebook.

## Quickstart & reproduction

Create a Python environment with the CUDA runtime expected by your PyTorch wheel, then install the pinned dependencies:

```powershell
python -m pip install -r requirements.txt
```

Run the complete automated suite:

```powershell
python -m pytest -q
```

Regenerate the matched report and these documents from JSON:

```powershell
python scripts/generate_matched_decoder_report.py --input data/matched_decoder_comparison_burst_poisson.json --output data/matched_decoder_comparison_burst_poisson.md
python scripts/generate_docs_from_json.py
```

To reproduce the benchmark itself on a CUDA-capable system, run `scripts/run_hybrid_before_after_burst.py`; treat a run as suitable for conclusions only when its parity and telemetry gates pass.

## Limitations

- The documented matched session is telemetry-invalid, so its ratios are observations rather than portable claims.
- Results come from one laptop GPU and one real-speech subset.
- CUDA-graph decode uses delayed token readback, which can execute extra slot steps after a logical stop condition.
- Static batching and the hybrid scheduler have different orchestration and memory-management responsibilities even when decoder mode is matched.
"""

    findings = f"""# Findings — Paged Dual-Cache Speech Serving Engine

## Review conclusion

The CUDA-graph work is isolated behind default-off flags. With `graph_decode=False`, the scheduler retains the eager allocation, prefill, decode, sampling, EOS, and release path; it does not instantiate graph buffers or dispatch graph replay. The regression suite exercises the eager baseline as well as eager/graph token parity.

The accepted benchmark environment used the versions below, so these—not the untested alternative versions—are pinned:

{environment}

## Matched observations

{ratios}

The most direct matched decoder ratios are {comparison(matched['derived_profiles']['burst'], 'hybrid_before_to_after')['capacity_ratio']:.3f}x for burst hybrid graph/eager completion rate and {comparison(matched['derived_profiles']['poisson_high_4_0'], 'hybrid_before_to_after')['capacity_ratio']:.3f}x for arrival-paced hybrid graph/eager completion rate. The matched graph hybrid/static ratios are {comparison(matched['derived_profiles']['burst'], 'graph_architecture')['capacity_ratio']:.3f}x for burst and {comparison(matched['derived_profiles']['poisson_high_4_0'], 'graph_architecture')['capacity_ratio']:.3f}x for arrival-paced traffic.

No cross-mode architecture-gap percentage is reported. The previous eager-static versus graphed-engine comparison mixed decoder mode with architecture and has been retired.

## Correctness

{parity_rows}

The graph parity artifact reports a passing gate. Exact match was not perfect, but aggregate WER equaled the reference in both profiles. Source: [`data/serving_parity_graph.json`](data/serving_parity_graph.json).

## Evidence quality

The matched run's telemetry gate failed: busy-clock spread was {gate['clock_spread_pct']:.2f}% and canary drift was {canary['drift_pct']:.2f}%. Maximum temperature was {gate['max_temperature_c']:.1f} C and peak busy throttling was {gate['max_busy_throttle_pct']:.2f}%. Accordingly, the table is retained for transparency but not promoted as a clean hardware-performance result.

## Reproducibility policy

- Performance numbers in README and FINDINGS are generated from `data/matched_decoder_comparison_burst_poisson.json`.
- Parity numbers are generated from `data/serving_parity_graph.json`.
- `scripts/generate_matched_decoder_report.py` recomputes matched ratios from raw system rows.
- `scripts/generate_docs_from_json.py` rewrites `FINDINGS.md`; the README uses the newer sectioned median benchmark document.
"""

    (ROOT / "FINDINGS.md").write_text(findings, encoding="utf-8", newline="\n")


if __name__ == "__main__":
    write_docs()
