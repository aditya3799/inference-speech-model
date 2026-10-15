# Rigorous Empirical Gap Profile Report (Final)

> **System Telemetry & Metadata**: GPU: `NVIDIA GeForce RTX 3050 6GB Laptop GPU` | OS: `Windows 10 Home Single Language 26H2 (Build 26300)` | Driver: `551.76` | Session: `session_20261015_010431_1d96d3`

- **Session ID**: `session_20261015_010431_1d96d3`
- **Timestamp**: `2026-10-15T01:04:31.802737`
- **Gate Status**: `FAIL (Criteria Violations: Clock Spread 52.9% > 10.0%; Peak Throttle Ratio 11.1% > 10.0%; Active Throttle Reason: 0x0000000000000004 [sw_power_cap]; Temperature 67.0°C compliant <= 88.0°C)`
- **Warm-up Policy**: 1 full discarded interleaved cycle executed prior to 3 measured repeats.

---

## 0. Under-Load Telemetry & Gate Compliance (Section A1)

Telemetry sampled every 500 ms during active execution over busy samples (utilization $\ge 80\%$):

*GPU: `NVIDIA GeForce RTX 3050 6GB Laptop GPU` | OS: `Windows 10 Home Single Language 26H2 (Build 26300)` | Driver: `551.76` | Session: `session_20261015_010431_1d96d3`*

| Gate Metric | Target / Criterion | Measured Value | Compliance Verdict |
|:---|:---:|:---:|:---:|
| **Peak Busy Temperature** | $\le 88.0^\circ\text{C}$ | **67.0°C** | **COMPLIANT** |
| **Busy Throttled Samples Ratio** | $\le 10.0\%$ | **11.1%** | **VIOLATION** |
| **Active Throttle Reason Codes** | Exclude 0x1 (Idle) | **0x0000000000000004 (`sw_power_cap`)** | **ACTIVE (Power budget cap)** |
| **Cross-System Busy Clock Spread** | $\le 10.0\%$ of max | **52.9%** | **VIOLATION** |
| **Sequential Canary Drift** | $\le 15.0\%$ start-to-end | **10.5%** | **COMPLIANT** |

> **Gate Verdict**: Sequential canary throughput was 28.78x at start and 25.77x at end (drift: 10.5%). Cross-system median busy clocks differed by 52.9% (min=787 MHz, max=1672 MHz) due to laptop dynamic power capping (0x0000000000000004: sw_power_cap). Overall session verdict is **FAIL (Criteria Violations: Clock Spread 52.9% > 10.0%; Peak Throttle Ratio 11.1% > 10.0%; Active Throttle Reason: 0x0000000000000004 [sw_power_cap]; Temperature 67.0°C compliant <= 88.0°C)**.

---

## 1. Burst Service Capacity & Variant Comparisons (Section A3 & B1)

Evaluated on 100 saturation utterances. Comparisons use **Engine Baseline** as the reference:

*GPU: `NVIDIA GeForce RTX 3050 6GB Laptop GPU` | OS: `Windows 10 Home Single Language 26H2 (Build 26300)` | Driver: `551.76` | Session: `session_20261015_010431_1d96d3`*

| System | Rep 1 | Rep 2 | Rep 3 | Median Capacity (req/s) | Range (req/s) | CPU Perf (Med/Min) | Under-Load Busy Clock | Stability Status |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `Static Batching (B=4)` | 5.60 | 6.03 | 6.16 | **6.03** | 0.56 | 97% / 91% | 1822 MHz | **Stable (C >= 4.0)** |
| `Engine Baseline` | 3.50 | 3.29 | 3.52 | **3.50** | 0.23 | 94% / 90% | 750 MHz | **UNSTABLE (C < 4.0)** |
| `Engine (no_sync_decode=True)` | 3.17 | 3.54 | 3.26 | **3.26** | 0.36 | 94% / 83% | 750 MHz | **UNSTABLE (C < 4.0)** |
| `Engine (batch_encoder=True)` | 3.38 | 2.76 | 3.65 | **3.38** | 0.89 | 94% / 77% | 791 MHz | **UNSTABLE (C < 4.0)** |
| `Engine (avoid_redundant_cross_kv=True)` | 3.51 | 3.14 | 3.52 | **3.51** | 0.38 | 97% / 90% | 787 MHz | **UNSTABLE (C < 4.0)** |

### 1.1 Variant Comparisons vs Engine Baseline (Burst Capacity)

| Variant Comparison vs Engine Baseline | Median Delta (req/s) | Combined Range | Distinguishability Assessment |
|:---|:---:|:---:|:---|
| `static_batching` vs `engine_baseline` | +2.53 req/s | 0.56 req/s | **Distinguishable (|Δ|=2.53 > range 0.56)** |
| `engine_no_sync` vs `engine_baseline` | -0.23 req/s | 0.36 req/s | **Not distinguishable (|Δ|=0.23 <= range 0.36)** |
| `engine_batch_encoder` vs `engine_baseline` | -0.12 req/s | 0.89 req/s | **Not distinguishable (|Δ|=0.12 <= range 0.89)** |
| `engine_avoid_redundant` vs `engine_baseline` | +0.01 req/s | 0.38 req/s | **Not distinguishable (|Δ|=0.01 <= range 0.38)** |

> **Capacity Assessment**: Static batching achieved a median burst capacity of 6.03 req/s (greater than 4.00 req/s), meaning Static batching queue is stable. Engine baseline achieved a median burst capacity of 3.50 req/s (less than 4.00 req/s), creating a capacity margin of -0.50 req/s. Under a 4.00 req/s arrival rate, Engine baseline queue is unstable.

---

## 2. Poisson Latency Percentiles (Section B1)

Throughput is arrival-limited under Poisson traffic. Latency percentiles reported with spread across 3 repeats:

*GPU: `NVIDIA GeForce RTX 3050 6GB Laptop GPU` | OS: `Windows 10 Home Single Language 26H2 (Build 26300)` | Driver: `551.76` | Session: `session_20261015_010431_1d96d3`*

### 2.1 Poisson High ($\lambda = 4.0\text{ req/s}$)

| System | Throughput Label | Latency p50 (Med / Range) | Latency p90 (Med / Range) | Latency p95 (Med / Range) | Latency p99 (Med / Range) | Distinguishability vs Engine Baseline (p95) |
|:---|:---:|:---:|:---:|:---:|:---:|:---|
| `Static Batching (B=4)` | **Arrival-limited (4.00 req/s)** | 0.81s (0.03s) | 1.25s (0.05s) | 1.31s (0.06s) | 1.42s (0.06s) | Distinguishable (|Δ|=4.55s > range 0.22s) |
| `Engine Baseline` | **Arrival-limited (4.00 req/s)** | 3.99s (0.38s) | 5.54s (0.31s) | 5.86s (0.22s) | 6.85s (0.29s) | Reference |
| `Engine (no_sync_decode=True)` | **Arrival-limited (4.00 req/s)** | 3.74s (0.26s) | 5.57s (0.30s) | 5.90s (0.42s) | 6.91s (0.37s) | Not distinguishable (|Δ|=0.04s <= range 0.42s) |
| `Engine (batch_encoder=True)` | **Arrival-limited (4.00 req/s)** | 3.87s (0.18s) | 5.15s (0.61s) | 5.52s (0.68s) | 6.48s (0.67s) | Not distinguishable (|Δ|=0.34s <= range 0.68s) |
| `Engine (avoid_redundant_cross_kv=True)` | **Arrival-limited (4.00 req/s)** | 4.17s (0.12s) | 5.89s (0.16s) | 6.23s (0.24s) | 7.18s (0.16s) | Distinguishable (|Δ|=0.38s > range 0.24s) |

### 2.2 Poisson Mid ($\lambda = 3.0\text{ req/s}$)

| System | Throughput Label | Latency p50 (Med / Range) | Latency p90 (Med / Range) | Latency p95 (Med / Range) | Latency p99 (Med / Range) |
|:---|:---:|:---:|:---:|:---:|:---:|
| `Static Batching (B=4)` | **Arrival-limited (3.00 req/s)** | 0.76s (0.04s) | 1.10s (0.10s) | 1.20s (0.06s) | 1.33s (0.15s) |
| `Engine Baseline` | **Arrival-limited (3.00 req/s)** | 1.49s (0.13s) | 2.73s (0.23s) | 2.98s (0.23s) | 3.62s (0.17s) |
| `Engine (no_sync_decode=True)` | **Arrival-limited (3.00 req/s)** | 1.43s (0.08s) | 2.57s (0.11s) | 2.85s (0.07s) | 3.48s (0.10s) |
| `Engine (batch_encoder=True)` | **Arrival-limited (3.00 req/s)** | 1.34s (0.07s) | 2.57s (0.11s) | 2.73s (0.09s) | 3.45s (0.05s) |
| `Engine (avoid_redundant_cross_kv=True)` | **Arrival-limited (3.00 req/s)** | 1.53s (0.01s) | 2.76s (0.16s) | 3.09s (0.16s) | 3.70s (0.09s) |

---

## 3. Tail Latency Decomposition (Poisson-High, $\lambda = 4.0\text{ req/s}$)

Decomposes latency into admission wait, TTFT, and decode time (shares strictly sum to 100.0%):

*GPU: `NVIDIA GeForce RTX 3050 6GB Laptop GPU` | OS: `Windows 10 Home Single Language 26H2 (Build 26300)` | Driver: `551.76` | Session: `session_20261015_010431_1d96d3`*

### 3.1 Tail Requests (Latency $\ge$ System p95)

| System | Mean Total Latency (s) | Mean Admission Wait (s) | Share (%) | Mean TTFT (s) | Share (%) | Mean Decode Time (s) | Share (%) | Sum of Shares (%) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `static_batching` | 1.440 s | 0.749 s | **52.0%** | 0.057 s | 4.0% | 0.634 s | 44.0% | **100.0%** |
| `engine_baseline` | 6.468 s | 4.693 s | **72.6%** | 0.102 s | 1.6% | 1.673 s | 25.9% | **100.0%** |

### 3.2 All Requests (Mean)

| System | Mean Total Latency (s) | Mean Admission Wait (s) | Share (%) | Mean TTFT (s) | Share (%) | Mean Decode Time (s) | Share (%) | Sum of Shares (%) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `static_batching` | 0.837 s | 0.290 s | **34.7%** | 0.054 s | 6.4% | 0.494 s | 58.9% | **100.0%** |
| `engine_baseline` | 3.825 s | 2.736 s | **71.5%** | 0.103 s | 2.7% | 0.985 s | 25.8% | **100.0%** |

> **Tail Finding**: For requests $\ge p95$ in the engine, admission wait accounts for 72.6% of total latency (4.693s of 6.468s). In static batching, admission wait represents 52.0%. Admission queue accumulation is the primary driver of tail latency blowout.

---

## 4. Step Profile with Union GPU Busy Fraction (Section B2)

*GPU: `NVIDIA GeForce RTX 3050 6GB Laptop GPU` | OS: `Windows 10 Home Single Language 26H2 (Build 26300)` | Driver: `551.76` | Session: `session_20261015_010431_1d96d3`*

| Factor / Metric | Static Batching (B=4) | Engine Baseline (B=4) | Delta (Engine - Static) | Measurement Basis |
|:---|:---:|:---:|:---:|:---|
| **Step Wall Time** | 12.52 ms | 16.16 ms | +3.63 ms | Synchronized decode step wall-clock duration |
| **True GPU Busy Time** | not reliable | not reliable | not reliable | Exceeds wall time (30.08 ms static / 54.57 ms engine) due to profiler tracing overhead & CPU operator interval overlap |
| **True GPU Busy Fraction** | not reliable | not reliable | not reliable | Confounded by PyTorch profiler instrumentation overhead |
| **Idle Gap per Step** | not reliable | not reliable | not reliable | Confounded by profiler tracing overhead |
| **Measured GPU Work (CUDA Graph Replay)** | **1.58 ms** | not measured (blocked) | N/A | Cross-check: fixed-shape replay without host launch overhead (Section B4) |
| **CUDA Kernel / Op Dispatches** | 971 launches | 1799 launches | +828 launches | Total kernel/operator dispatches dispatched |
| **Wall Time per Launch** | 0.0129 ms | 0.0090 ms | -0.0039 ms | Step Wall Time / Kernel Launches |

### 4.1 Kernel Attribution Ranking (Section B3)

Grouped decode step kernels ranked by contribution to (Engine - Static) delta:

| Region Group | Static GPU (ms) | Engine GPU (ms) | Delta GPU Time (ms) | Static Launches | Engine Launches | Delta Launches |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| `other` | 31.018 ms | 50.631 ms | **+19.613 ms** | 951 | 1662 | +711 |
| `cache_pool_ops` | 1.027 ms | 5.550 ms | **+4.523 ms** | 14 | 124 | +110 |
| `sampling` | 0.000 ms | 1.014 ms | **+1.014 ms** | 0 | 6 | +6 |
| `self_attn` | 0.000 ms | 0.474 ms | **+0.474 ms** | 0 | 1 | +1 |
| `cross_attn` | 0.000 ms | 0.000 ms | **+0.000 ms** | 0 | 0 | +0 |
| `mlp` | 0.308 ms | 0.303 ms | **-0.005 ms** | 6 | 6 | +0 |

---

## 5. CUDA Graph Test (Section B4)

*GPU: `NVIDIA GeForce RTX 3050 6GB Laptop GPU` | OS: `Windows 10 Home Single Language 26H2 (Build 26300)` | Driver: `551.76` | Session: `session_20261015_010431_1d96d3`*

| Configuration | Graph Capture Status | Without Graph (ms) | With Graph Replay (ms) | Speedup Ratio | Blocker / Details |
|:---|:---:|:---:|:---:|:---:|:---|
| **Static Batching (B=4)** | **SUCCESS** | 12.116 ms | 1.579 ms | **7.67x** | Replay of 100 measured steps with fixed shape tensors |
| **Engine Baseline (B=4)** | **BLOCKED** | not measured | not measured | not measured | Host-to-device tensor allocation and copy inside speech_serving/phase_aware_scheduler.py:658 (`token_in = torch.tensor([[req.last_token_id] for req in self.active_decode], device=self.device, dtype=torch.long)`), which triggers `CUDA error: operation not permitted when stream is capturing`. In addition, the engine's decode step requires CPU-side token synchronizations, dynamic block table lookups, and list comprehensions. |

---

## 6. Memory Breakdown, Snapshot Accounting & Parity (Sections B5 & B6)

> *Source Dataset Timestamp*: `2026-10-15T01:04:31.802737` (from `data/memory_breakdown_results_final.json`, fresh process evaluations)

*GPU: `NVIDIA GeForce RTX 3050 6GB Laptop GPU` | OS: `Windows 10 Home Single Language 26H2 (Build 26300)` | Driver: `551.76` | Session: `session_20261015_010431_1d96d3`*

| Component / Buffer | Static Batching (B=4) | Engine Default (B=4) | Engine Right-Sized (B=4) | Engine Avoid-Redundant (B=4) | Description |
|:---|:---:|:---:|:---:|:---:|:---|
| **Model Weights** | 138.5 MB | 139.4 MB | 139.4 MB | 139.4 MB | Whisper-base FP16 weights |
| **Pool Blocks in Use** | not measured / 0 MB | **78.0 MB** | **58.5 MB** | **58.5 MB** | Blocks in use x block bytes (<= pool allocated) |
| - `req.cached_encoder_kv` | 0.0 MB | 70.3 MB | 52.7 MB | **0.0 MB** | Per-request contiguous cross-KV buffers |
| - `_cached_batched_enc_kv` | 70.3 MB | 70.3 MB | 52.7 MB | 52.7 MB | Batched cached cross-KV tensor |
| **Total Contiguous Cross-KV** | 70.3 MB | 140.6 MB | 105.5 MB | **52.7 MB** | Contiguous cross-attention copies |
| **Block Pool Allocated** | not measured / 0 MB | **1344.0 MB** | **127.4 MB** | **127.4 MB** | Preallocated physical tensor pools |
| **Activations** | 98.8 MB | 83.3 MB | 65.6 MB | 61.7 MB | Forward pass intermediate buffers |
| **Peak Allocated VRAM** | **307.6 MB** | **1715.5 MB** | **446.1 MB** | **389.4 MB** | `torch.cuda.max_memory_allocated()` |
| **Peak Reserved VRAM** | 328.0 MB | 1754.0 MB | 468.0 MB | 428.0 MB | `torch.cuda.max_memory_reserved()` |

### 6.1 Residency Difference Explanation (Section B5)
In the default configuration, 4 requests were resident at the measurement point (4 requests x 17.58 MB cross-KV = 70.3 MB per-request cross-KV, and 376 enc blocks + 40 dec blocks = 71.2 MB pool in use). In the right-sized run, 3 requests were resident at that measurement point (3 requests x 17.58 MB = 52.7 MB cross-KV, and 58.5 MB pool in use).

### 6.2 Memory Snapshot Account (torch.cuda.memory_snapshot())
- Status: `offset unexplained`
- Snapshot Segments: `3`
- Total Snapshot Allocated: `8.140625 MB`
- Total Snapshot Reserved: `18.0 MB`
- Allocator Block States: `{"inactive": 9.859375, "active_allocated": 8.140625}`

### 6.3 Parity on 20 Utterances (Section B6)
- **Engine Baseline**: Exact matches = `20/20 (100.0%)`, Token match rate = `100.00%`
- **Engine Avoid Redundant**: Exact matches = `20/20 (100.0%)`, Token match rate = `100.00%`
- **VRAM Saved**: `56.7 MB reduction` (from 446.1 MB to 389.4 MB).

---

## 7. Limitations (Section C2)

1. **Laptop Hardware Constraints**: Evaluated on an NVIDIA GeForce RTX 3050 6GB Laptop GPU (TGP ~35-45W). Dynamic DVFS power balancing introduces run-to-run variation depending on thermal headroom.
2. **Session-to-Session Variation**: Across all benchmark sessions found in `data/`, static burst capacity ranged from 5.38 req/s to 6.03 req/s (and throughput across all 5 LibriSpeech sessions spanned from 36.56x to 70.95x) due to OS scheduler jitter and thermal state variation.
3. **In-Flight Concurrency Cap**: Active concurrency is strictly capped at 4 by `max_batch_size=4` in scheduler configuration, despite pool capacity for ~65 requests.
4. **Triton Variant Memory**: Direct Paged Attention Triton kernel exhibited higher peak memory (806.8 MB allocated vs 641.2 MB for hybrid), but the precise internal contribution (JIT compiler workspaces vs unshared physical block tables) was **not measured**.