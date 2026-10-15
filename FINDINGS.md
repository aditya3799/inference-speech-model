# FINDINGS.md — Paged Dual-Cache Speech Serving Engine

**Date:** October 2026  
**Scope:** Empirical Evaluation & Rigorous Gap Profiling (Session: `session_20261015_010431_1d96d3`).

> *Hardware & Environment*: GPU: `NVIDIA GeForce RTX 3050 6GB Laptop GPU` | OS: `Windows 10 Home Single Language 26H2 (Build 26300)` | Driver: `551.76` | Session: `session_20261015_010431_1d96d3`

---

## Executive Summary & Non-Negotiable Findings

1. **The 1.5x–3x Throughput Claims Are Fully Audited and Retracted:**
   - Prior documentation cited throughput gains of 1.44x to 3.0x. An audit revealed these originated from external text-LLM KV-cache repositories or early unbatched runs.
   - The scheduler microbenchmark achieves steady mean speedups of **1.03x to 1.13x** on synthetic workloads.

2. **Encoder Context Sensitivity & Full-Utterance Caching:**
   - chunk-level sharing is not exactly equivalent (E1 divergence rate: 28.0% [14/50 pairs], spliced-WER 95% CI: [-0.0067, +0.0092]; E2 divergence rate: 31.1% [46/148 shifts], spliced-WER 95% CI: [-0.0040, +0.0230]), with no detectable average WER effect, so the engine hashes full utterances.

3. **Burst Capacity & Tail Latency Under Load (Session `session_20261015_010431_1d96d3`):**
   - Static batching achieves **6.03 req/s** median burst capacity (stable queue).
   - Engine baseline achieves **3.50 req/s** median burst capacity (unstable queue).
   - Under arrival rate $\lambda = 4.0$ req/s, admission wait represents **72.6%** of tail latency ($\ge p95$), showing that queuing backlog accounts for the majority of tail latency.

4. **Step Profile & Hardware Work Measurement:**
   - True GPU busy time and fraction from PyTorch profiler union intervals are **not reliable** because profiler instrumentation overhead and CPU operator time range overlaps inflate durations beyond unprofiled wall time (12.52 ms static / 16.16 ms engine).
   - CUDA graph replay provides an isolated hardware cross-check: static batching executes in **1.58 ms** per step (a 7.67x speedup over unprofiled 12.12 ms wall time).
   - Engine dispatches **1799 operator/kernel calls** (+828 dispatches over static's 971), causing host launch dispatch saturation.

5. **CUDA Graph Test Result:**
   - Static batching achieves a **7.67x speedup** under CUDA graph replay (12.12 ms without graph vs 1.58 ms with graph).
   - Engine baseline capture is blocked: host-to-device tensor allocation/copy in `phase_aware_scheduler.py:658` (`token_in = torch.tensor(...)`) violates CUDA graph capture stream constraints.

6. **Memory Right-Sizing & Contiguous Cross-KV Optimization:**
   - Default pools allocated 1344.0 MB. Sizing to 127.4 MB and avoiding redundant contiguous cross-KV copies drops peak VRAM allocation from 446.1 MB to **389.4 MB** (a **56.7 MB reduction**) with **100.0% exact match parity** across 20 utterances.

---

## Detailed Empirical Data Tables

### Burst Capacity & Variant Comparison Table

| System | Median Capacity (req/s) | Range (req/s) | CPU Perf (Med/Min) | Under-Load Busy Clock | Stability Status |
|:---|:---:|:---:|:---:|:---:|:---:|
| `Static Batching (B=4)` | **6.03** | 0.56 | 97% / 91% | 1822 MHz | **Stable** |
| `Engine Baseline` | **3.50** | 0.23 | 94% / 90% | 750 MHz | **UNSTABLE** |
| `Engine (no_sync_decode=True)` | **3.26** | 0.36 | 94% / 83% | 750 MHz | **UNSTABLE** |
| `Engine (batch_encoder=True)` | **3.38** | 0.89 | 94% / 77% | 791 MHz | **UNSTABLE** |
| `Engine (avoid_redundant_cross_kv=True)` | **3.51** | 0.38 | 97% / 90% | 787 MHz | **UNSTABLE** |

### Memory Breakdown & Allocator Table

> *Source Dataset Timestamp*: `2026-10-15T01:04:31.802737` (from `data/memory_breakdown_results_final.json`, fresh process evaluations)

| Component | Static Batching | Engine Default | Engine Right-Sized | Engine Avoid-Redundant |
|:---|:---:|:---:|:---:|:---:|
| Model Weights | 138.5 MB | 139.4 MB | 139.4 MB | 139.4 MB |
| Pool Blocks in Use | 0.0 MB | 78.0 MB | 58.5 MB | 58.5 MB |
| - `req.cached_encoder_kv` | 0.0 MB | 70.3 MB | 52.7 MB | 0.0 MB |
| - `_cached_batched_enc_kv` | 70.3 MB | 70.3 MB | 52.7 MB | 52.7 MB |
| Contiguous Cross-KV Total | 70.3 MB | 140.6 MB | 105.5 MB | 52.7 MB |
| Pool Allocated | 0.0 MB | 1344.0 MB | 127.4 MB | 127.4 MB |
| Peak Allocated VRAM | **307.6 MB** | **1715.5 MB** | **446.1 MB** | **389.4 MB** |
| Peak Reserved VRAM | 328.0 MB | 1754.0 MB | 468.0 MB | 428.0 MB |

### Snapshot Accounting
- Status: `offset unexplained`

---

## Limitations (Section C2)

1. **Laptop GPU Telemetry**: Evaluated on an NVIDIA GeForce RTX 3050 6GB Laptop GPU. Laptop thermal limits restrict continuous peak TDP to ~35-45W.
2. **Session-to-Session Variance**: Measured static burst capacity ranges from 5.38 req/s to 6.03 req/s (and throughput across all 5 LibriSpeech sessions spanned from 36.56x to 70.95x) across all sessions in `data/`.
3. **In-Flight Concurrency Limit**: Active concurrency capped at 4 by scheduler max batch size.
4. **Triton Memory Overhead**: Direct paged kernel allocates 806.8 MB peak, but the internal cause (intermediate compiler workspaces vs unshared block tables) was **not measured**.