# Fresh matched decoder comparison

Session: `burst_20261016_010916_42fad9`; status: **TELEMETRY_INVALID**.

## burst

| System | Decode | Wall s | Capacity req/s | p50 latency s | p95 latency s | Exact % | WER % | Status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| static_batching | eager | 24.549 | 4.073 | 10.937 | 23.168 | 100.0 | 5.282 | VALID |
| static_graph | cuda_graph | 8.474 | 11.801 | 5.161 | 8.227 | 100.0 | 5.282 | VALID |
| hybrid_before | eager | 44.865 | 2.229 | 23.833 | 43.477 | 99.0 | 5.282 | VALID |
| hybrid_after | cuda_graph | 15.483 | 6.459 | 8.704 | 15.058 | 99.0 | 5.282 | VALID |

| Matched comparison (right / left) | Capacity req/s ratio | Wall-time ratio | p50 ratio | p95 ratio |
|---|---:|---:|---:|---:|---:|
| static_eager_to_graph | 2.897x | 0.345x | 0.472x | 0.355x |
| hybrid_before_to_after | 2.898x | 0.345x | 0.365x | 0.346x |
| eager_architecture | 0.547x | 1.828x | 2.179x | 1.877x |
| graph_architecture | 0.547x | 1.827x | 1.686x | 1.830x |

## poisson_high_4_0

| System | Decode | Wall s | Observed completion req/s (arrival-paced) | p50 latency s | p95 latency s | Exact % | WER % | Status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| static_batching | eager | 23.561 | 4.244 | 0.661 | 0.975 | 99.0 | 5.282 | VALID |
| static_graph | cuda_graph | 22.967 | 4.354 | 0.161 | 0.354 | 99.0 | 5.282 | VALID |
| hybrid_before | eager | 31.352 | 3.190 | 5.921 | 8.414 | 99.0 | 5.282 | VALID |
| hybrid_after | cuda_graph | 23.694 | 4.220 | 0.473 | 1.953 | 99.0 | 5.282 | VALID |

| Matched comparison (right / left) | Observed completion req/s (arrival-paced) ratio | Wall-time ratio | p50 ratio | p95 ratio |
|---|---:|---:|---:|---:|---:|
| static_eager_to_graph | 1.026x | 0.975x | 0.243x | 0.363x |
| hybrid_before_to_after | 1.323x | 0.756x | 0.080x | 0.232x |
| eager_architecture | 0.751x | 1.331x | 8.959x | 8.626x |
| graph_architecture | 0.969x | 1.032x | 2.938x | 5.511x |

## Session gate

| Metric | Observed | Limit | Status |
|---|---:|---:|---|
| Busy throttling | 0.00% | <=10% | PASS |
| Clock spread | 52.09% | <=10% | FAIL |
| Canary drift | 15.28% | <=15% | FAIL |
| Maximum temperature | 72.0 C | <=88 C | PASS |

Failing criteria:
- Cross-system busy median clock spread (52.1%) exceeds 10.0% limit (min=870 MHz, max=1815 MHz).
- Sequential canary drift (15.3%) exceeds 15.0% limit.

## JSON-derived analysis

For burst, the matched hybrid after/before capacity ratio was 2.898x and its p95 latency ratio was 0.346x. The matched graph hybrid/static capacity ratio was 0.547x.
For poisson_high_4_0, the matched hybrid after/before arrival-paced completion rate ratio was 1.323x and its p95 latency ratio was 0.232x. The matched graph hybrid/static arrival-paced completion rate ratio was 0.969x.
The telemetry gate failed; results are not eligible for clean conclusions.
