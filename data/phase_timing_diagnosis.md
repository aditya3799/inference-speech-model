# Phase-timing diagnosis

Session status: **TELEMETRY_INVALID**.

Traffic: burst; utterances: 100; measured repeats: 3; discarded warm-up cycles: 1; interleaving: alternating order.

Phase semantics: shared batch work is request-experienced time and is recorded for every participating request.

Power state: {'status': 'measured after session', 'ac_line_status': 1, 'plugged_in': True, 'battery_percent': 100}.

## Correctness and capacity

| System | Capacity req/s | Wall s | Exact % | WER % | Status |
|---|---:|---:|---:|---:|---|
| static_graph | 15.118 | 6.615 | 100.0 | 5.282 | VALID |
| hybrid_after | 8.034 | 12.447 | 99.0 | 5.282 | VALID |

## Per-request phase totals

| Phase | Static total ms | Static mean ms/request | Static % request wall | Hybrid total ms | Hybrid mean ms/request | Hybrid % request wall | Hybrid - static mean ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| queue_admission_wait | 1088237.349 | 3627.458 | 93.820 | 1854859.992 | 6182.867 | 92.956 | 2555.409 |
| encoder_forward | 14018.415 | 46.728 | 1.209 | 4652.317 | 15.508 | 0.233 | -31.220 |
| cross_kv_pool_copy | not measured | not measured | not measured | 15811.973 | 52.707 | 0.792 | not measured |
| cross_kv_contiguous_or_graph_copy | 11918.532 | 39.728 | 1.028 | 3652.229 | 12.174 | 0.183 | -27.554 |
| prefill | 3841.966 | 12.807 | 0.331 | 3997.268 | 13.324 | 0.200 | 0.518 |
| graph_replay | 19137.749 | 63.792 | 1.650 | 21745.806 | 72.486 | 1.090 | 8.694 |
| host_between_replays | 4201.944 | 14.006 | 0.362 | 6919.525 | 23.065 | 0.347 | 9.059 |
| readback_wait | 537.354 | 1.791 | 0.046 | 580.217 | 1.934 | 0.029 | 0.143 |
| block_allocate | not measured | not measured | not measured | 138.266 | 0.461 | 0.007 | not measured |
| block_release | not measured | not measured | not measured | 33.908 | 0.113 | 0.002 | not measured |
| other_host_between_requests | 18028.429 | 60.095 | 1.554 | 83018.566 | 276.729 | 4.160 | 216.634 |

## Ranked contribution to capacity gap

| Rank | Phase | Hybrid - static mean ms/request |
|---:|---|---:|
| 1 | queue_admission_wait | 2555.409 |
| 2 | other_host_between_requests | 216.634 |
| 3 | host_between_replays | 9.059 |
| 4 | graph_replay | 8.694 |
| 5 | prefill | 0.518 |
| 6 | readback_wait | 0.143 |
| 7 | cross_kv_contiguous_or_graph_copy | -27.554 |
| 8 | encoder_forward | -31.220 |
| 9 | cross_kv_pool_copy | not measured |
| 10 | block_allocate | not measured |
| 11 | block_release | not measured |

## Batch and profiler timing

| System | Actual batch-of-4 mean ms | Equivalent batch-of-4 ms | GPU kernel union ms | Idle GPU ms | Idle GPU % |
|---|---:|---:|---:|---:|---:|
| static_graph | 238.948 | 264.590 | not measured | not measured | not measured |
| hybrid_after | not measured | 497.874 | not measured | not measured | not measured |

## Slot hold after EOS

| System | Total steps | Mean steps/request |
|---|---:|---:|
| static_graph | 6507 | 21.690 |
| hybrid_after | 504 | 1.680 |

## Instrumentation overhead

| System | Flag off wall s | Flag on wall s | Difference s | Overhead % |
|---|---:|---:|---:|---:|
| static_graph | 6.252 | 6.496 | 0.244 | 3.90 |
| hybrid_after | 11.852 | 12.024 | 0.172 | 1.45 |

## Session gate

| Criterion | Observed | Required | Status |
|---|---:|---:|---|
| Busy throttle | 25.00% | <= 10.00% | FAIL |
| Clock spread | 19.01% | <= 10.00% | FAIL |
| Canary drift | 3.22% | <= 15.00% | PASS |
| Maximum temperature | 73.00 C | <= 88.00 C | PASS |

Failing criteria:
- Peak busy throttling ratio 25.0% exceeded 10.0%.
- Cross-system busy median clock spread (19.0%) exceeds 10.0% limit (min=1470 MHz, max=1815 MHz).

## JSON-derived summary

The largest measured positive hybrid-minus-static phase difference was `queue_admission_wait` at 2555.409 ms per request.
The session gate failed; performance conclusions are not eligible for clean comparison.
