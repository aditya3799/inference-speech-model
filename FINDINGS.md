# Findings — Paged Dual-Cache Speech Serving Engine

## Review conclusion

The CUDA-graph work is isolated behind default-off flags. With `graph_decode=False`, the scheduler retains the eager allocation, prefill, decode, sampling, EOS, and release path; it does not instantiate graph buffers or dispatch graph replay. The regression suite exercises the eager baseline as well as eager/graph token parity.

The accepted benchmark environment used the versions below, so these—not the untested alternative versions—are pinned:

| Component | Tested version |
|---|---|
| Python | 3.10.11 |
| PyTorch | 2.6.0+cu124 |
| Transformers | 4.57.0 |
| Triton | 3.8.0 |
| NVIDIA driver | 551.76 |

## Matched observations

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

The most direct matched decoder ratios are 2.898x for burst hybrid graph/eager completion rate and 1.323x for arrival-paced hybrid graph/eager completion rate. The matched graph hybrid/static ratios are 0.547x for burst and 0.969x for arrival-paced traffic.

No cross-mode architecture-gap percentage is reported. The previous eager-static versus graphed-engine comparison mixed decoder mode with architecture and has been retired.

## Correctness

| Traffic | Requests | Exact token match | WER | Reference WER |
|---|---:|---:|---:|---:|
| burst | 100 | 99.0% | 5.282% | 5.282% |
| poisson_4_0 | 100 | 99.0% | 5.282% | 5.282% |

The graph parity artifact reports a passing gate. Exact match was not perfect, but aggregate WER equaled the reference in both profiles. Source: [`data/serving_parity_graph.json`](data/serving_parity_graph.json).

## Evidence quality

The matched run's telemetry gate failed: busy-clock spread was 52.09% and canary drift was 15.28%. Maximum temperature was 72.0 C and peak busy throttling was 0.00%. Accordingly, the table is retained for transparency but not promoted as a clean hardware-performance result.

## Reproducibility policy

- Performance numbers in README and FINDINGS are generated from `data/matched_decoder_comparison_burst_poisson.json`.
- Parity numbers are generated from `data/serving_parity_graph.json`.
- `scripts/generate_matched_decoder_report.py` recomputes matched ratios from raw system rows.
- `scripts/generate_docs_from_json.py` rewrites both documents, preventing hand-copied numeric drift.
