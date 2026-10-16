# Paged Dual-Cache Speech Serving Engine

This repository explores a phase-aware Whisper serving engine with paged encoder and decoder KV caches, eager and CUDA-graph decode paths, real-speech parity checks, and reproducible benchmark reports.

The current matched benchmark is useful as an observed run, but not as a clean performance conclusion: its telemetry gate failed because cross-system busy-clock spread was 52.09% and sequential-canary drift was 15.28%. Source: [`data/matched_decoder_comparison_burst_poisson.json`](data/matched_decoder_comparison_burst_poisson.json).

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

| Traffic | Matched comparison | Completion-rate ratio | Wall-time ratio | p50 ratio | p95 ratio |
|---|---|---:|---:|---:|---:|
| burst | static graph / static eager | 2.897x | 0.345x | 0.472x | 0.355x |
| burst | hybrid graph / hybrid eager | 2.898x | 0.345x | 0.365x | 0.346x |
| burst | hybrid / static, eager decoder | 0.547x | 1.828x | 2.179x | 1.877x |
| burst | hybrid / static, graphed decoder | 0.547x | 1.827x | 1.686x | 1.830x |
| poisson_high_4_0 | static graph / static eager | 1.026x | 0.975x | 0.243x | 0.363x |
| poisson_high_4_0 | hybrid graph / hybrid eager | 1.323x | 0.756x | 0.080x | 0.232x |
| poisson_high_4_0 | hybrid / static, eager decoder | 0.751x | 1.331x | 8.959x | 8.626x |
| poisson_high_4_0 | hybrid / static, graphed decoder | 0.969x | 1.032x | 2.938x | 5.511x |

The headline matched ratios are: burst hybrid graph/eager completion rate 2.898x, burst graph hybrid/static completion rate 0.547x, arrival-paced hybrid graph/eager completion rate 1.323x, and arrival-paced graph hybrid/static completion rate 0.969x.

## Serving parity

| Traffic | Requests | Exact token match | WER | Reference WER |
|---|---:|---:|---:|---:|
| burst | 100 | 99.0% | 5.282% | 5.282% |
| poisson_4_0 | 100 | 99.0% | 5.282% | 5.282% |

Source: [`data/serving_parity_graph.json`](data/serving_parity_graph.json). The parity gate passed even though one utterance differed in each traffic profile; WER matched the reference.

## Tested environment

| Component | Tested version |
|---|---|
| Python | 3.10.11 |
| PyTorch | 2.6.0+cu124 |
| Transformers | 4.57.0 |
| Triton | 3.8.0 |
| NVIDIA driver | 551.76 |

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
