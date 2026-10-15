# Paged Dual-Cache Speech Serving Engine

**Status: Research Prototype & Empirical Evaluation.** This repository explores PagedAttention-style continuous batching, dual physical block pools, and lookup-before-encode prefix caching for speech models (Whisper-class ASR and streaming multi-codebook TTS).

> [!NOTE]
> **Key Empirical Highlights:**
> 1. **Real-Speech Serving**: Integrated with `openai/whisper-base` on LibriSpeech `test-clean` (100 stratified utterances, 3 repeats), achieving **5.28% corpus WER** (95% bootstrap CI: `[3.95%, 6.89%]`) and **99.0% exact token match rate (99/100)** against Hugging Face `generate()`. Exactly one utterance diverged (`2830-3980-0051` at token 8: `"God,"` vs `"God"`, identical under Whisper text normalization).
> 2. **Full-Utterance Caching**: Chunk-level sharing is not exactly equivalent (E1: 14/50 pairs diverge at token level; E2: 46/148), with no detectable average WER effect (spliced WER delta CIs include zero), so the engine hashes full utterances.
> 3. **Batching vs. Paging Overhead**: Decode-path ablation isolates the benefit of batching (**1.99x speedup** of static batched over sequential) from paging overhead: re-gathering 1500-position cross-attention KV accounts for **90.0%** of the hybrid-vs-static wall-time gap. Caching cross-attention KV once recovers throughput from 28.10 to 59.99 tok/s.
> 4. **Microbenchmark Speedups**: Continuous batching scheduler achieves steady **1.03x to 1.13x** speedups over unpaged execution under varying arrival traffic on an RTX 3050 Laptop GPU.
> 5. **Concurrency Capacity Analysis**: Accounting for Whisper's full 17.58 MB cross-attention KV, paged memory allocation provides **1.27x** (30s padded audio) to **1.82x** (trimmed audio, analytical/hypothetical) concurrent request capacity gains over static allocations under a fixed 512 MB memory budget.
>
> Full empirical logs, raw telemetry, and reproduction methodology are documented in [FINDINGS.md](FINDINGS.md).

---

## Architecture: Dual-Cache Speech Serving

Speech models exhibit asymmetric memory patterns: a compute-dense, bidirectional audio encoder pass coupled with an autoregressive causal decoder that cross-attends into encoder activations.

```
Incoming Speech Requests (Audio Mel Features / TTS Tokens)
               │
               ▼
┌─────────────────────────────────────────────────────────────┐
│             Phase-Aware Speech Scheduler                    │
│  - Lookup BEFORE encode: identical audio skips encoder      │
│  - Interleaves compute-dense ENCODE & memory-bound DECODE   │
│  - Continuous batch composition & dynamic queue management  │
│  - 4D cross-attention padding mask calculation & caching   │
└──────────────┬───────────────────────────────┬──────────────┘
               │                               │
       [Audio Features]                 [Text Tokens]
               ▼                               ▼
┌───────────────────────────────┐ ┌───────────────────────────┐
│     Encoder Block Manager     │ │   Decoder Block Manager   │
│ - Full-input hash (safe for   │ │ - Causal autoregressive   │
│   bidirectional encoders)     │ │   paged KV-cache pool     │
│ - Reference-counted blocks    │ │ - Incremental allocation  │
│ - LRU cache & bulk eviction   │ │ - Immediate block release │
└───────────────────────────────┘ └───────────────────────────┘
               │                               │
               └───────────────┬───────────────┘
                               ▼
            ┌────────────────────────────────────┐
            │       Memory Guard System          │
            │ - High-water mark tracking (75%)   │
            │ - Admission backpressure rejection │
            │ - Zero-leak lifecycle invariants   │
            └────────────────────────────────────┘
```

### Key Architectural Invariants
1. **Decoupled Dual-Cache Design**: Non-causal encoder representations are computed once, stored in physical block tables, and read repeatedly during all decode steps. Causal decoder KV caches are grown token-by-token.
2. **Lookup-Before-Encode**: Audio features are hashed *before* the encoder runs. On a cache hit, the encoder forward pass is completely skipped, ref-counts are incremented, and pre-projected cross-attention KV tensors are reused.
3. **Full-Utterance Hashing**: Chunk-level sharing is not exactly equivalent (E1: 14/50 pairs diverge at token level; E2: 46/148), with no detectable average WER effect (spliced WER delta CIs include zero), so the engine hashes full utterances.
4. **Pretrained Whisper Integration**: Wrapped via `PretrainedWhisperModel`, caching layer-specific cross-attention projections ($K_l, V_l$) during the encode pass and applying Whisper `suppress_tokens` for exact generation parity.
5. **Memory Guard & High-Water Mark**: Backpressures incoming requests when projected utilization exceeds 75%, preventing CUDA out-of-memory crashes with zero block leaks.

---

## Empirical Benchmark Findings

All numbers below come from executed benchmarks and saved JSON logs. No numbers are hand-edited.

### 1. Parity Check Against Hugging Face `generate()`
- **Single-Stream Verification** (`data/parity_check_results.json`):
  - Verified across 20 LibriSpeech `test-clean` utterances with `openai/whisper-base`:
  - **FP32 Exact Token Match Rate:** **100.0% (20/20)**, WER delta = 0.000.
  - **FP16 Exact Token Match Rate:** **100.0% (20/20)**, WER delta = 0.000.
  - Confirms the custom cross-attention caching and greedy decoding logic matches Hugging Face token-for-token under single-stream execution.
- **Continuous Serving Parity** (`data/real_speech_evaluation_results.json`):
  - Across 100 LibriSpeech test-clean utterances (3 repeats), the continuous batching engine achieves **99.0% token match rate (99/100 utterances identical)**.
  - Exactly one utterance diverged: `2830-3980-0051` (sample index 67) at token position 8 where continuous batching FP16 accumulation emitted token 11 (`,`) (`"God,"` vs `"God"`). Both normalize to 100% identical text with Whisper `EnglishTextNormalizer`.

### 2. Encoder Context Sensitivity (Real Whisper-base, 50 Pairs)
Evaluated in `scripts/evaluate_context_sensitivity.py` (`data/encoder_context_sensitivity_results.json`):
- **Calibration Noise Floor:** $2.22 \times 10^{-3}$ relative $L_2$ error (FP16 vs FP32).
- **Experiment E1 (Trailing Context, $n=50$ pairs):**
  - Relative $L_2$ error: Median = **0.1914** ($86.1\times$ noise floor), p95 = **0.3197** ($143.8\times$ noise floor), Max = **0.3799** ($170.8\times$ noise floor).
  - Token-level divergence: **14 / 50 pairs diverged** (Exact match rate: **72.0%**).
  - Spliced WER delta: **+0.0009** (95% bootstrap CI: `[-0.0067, +0.0092]`).
- **Experiment E2 (Offset Shifts, $n=148$ shifts across 2s, 5s, 10s):**
  - Relative $L_2$ error: Median = **0.8540** ($384.0\times$ noise floor), p95 = **0.9956** ($447.6\times$ noise floor), Max = **1.0067** ($452.6\times$ noise floor).
  - Token-level divergence: **46 / 148 shifts diverged** (Exact match rate: **68.9%**).
  - Spliced WER delta: **+0.0074** (95% bootstrap CI: `[-0.0040, +0.0230]`).
- **Verdict:**
  Chunk-level sharing is not exactly equivalent (E1: 14/50 pairs diverge at token level; E2: 46/148), with no detectable average WER effect (spliced WER delta CIs include zero), so the engine hashes full utterances.

### 3. Paging vs. Batching Ablation & Concurrency Capacity
Evaluated in `scripts/evaluate_paging_variants.py` and `scripts/profile_gather_decomposition.py`:
*Note: Ablation rows 4–6 evaluate cached-cross-KV, zero-copy, and direct Triton kernel variants. The real-speech serving, concurrency, and capacity figures in Section 4 were evaluated on the continuous batching engine path BEFORE the cached-cross-KV optimization.*

- **Gather Cost Decomposition ($B=4, L=32$ to $128$):**
  - Cross-attention gather (1500 positions) consumes **50.0% to 51.4%** of decode step time (mean $71.56\text{--}71.60\text{ms}$, copying $70.3\text{ MB/step}$).
  - Self-attention gather consumes **5.7% to 6.9%** (mean $7.95\text{--}9.82\text{ms}$, copying $1.5\text{--}6.0\text{ MB/step}$).
  - Attention compute itself consumes **12.7% to 13.8%** (mean $17.62\text{--}19.81\text{ms}$).
  - Other (projections, MLP, scatter, bookkeeping) consumes **28.4% to 29.2%** (mean $40.65\text{--}40.68\text{ms}$).
  - *Note:* Millisecond values are inflated by synchronization timing (`torch.cuda.synchronize()` around each sub-layer hook region), so the percentage proportions are the reliable metric.

- **Measured 6-Row Ablation Matrix ($B=4$, 32 tokens generated, 3 repeats):**

| Variant | Throughput (tok/s) | Latency per Req (s) | Step Wall Time (s) | Peak VRAM Alloc (MB) | Peak VRAM Res (MB) | HF Exact Match |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Row 1: Sequential** (No Batching, No Paging) | 34.55 $\pm$ 0.01 | 0.926 | 3.704 | 248.3 | 262.0 | 20/20 (100%) |
| **Row 2: Static Batched Contiguous** (Batching, No Paging) | **68.60** $\pm$ 0.16 | **1.866** | **1.866** | 305.4 | 324.0 | 20/20 (100%) |
| **Row 3: Hybrid Engine** (Gather Every Step, Incl. Cross-KV) | 28.10 $\pm$ 0.43 | 4.554 | 4.554 | 641.2 | 728.0 | 20/20 (100%) |
| **Row 4: Hybrid + Cached Cross-KV** (Gather Self-KV Only) | 59.99 $\pm$ 0.96 | 2.134 | 2.134 | 641.2 | 728.0 | 20/20 (100%) |
| **Row 5: Paged Storage, Contiguous Alloc** (Zero-Copy View) | 68.11 $\pm$ 0.03 | 1.879 | 1.879 | 643.6 | 728.0 | 20/20 (100%) |
| **Row 6: Direct Paged Kernel** (Triton) | 59.23 $\pm$ 1.83 | 2.161 | 2.161 | 806.8 | 928.0 | 20/20 (100%) |

- **Isolating the Paging Cost & Gap Breakdown:**
  - **Batching Speedup:** Static batched contiguous decoding provides a **1.99x speedup** over sequential decoding ($68.60\text{ tok/s}$ vs $34.55\text{ tok/s}$).
  - **Cross-KV Gather Overhead:** Re-gathering 1500-position cross-attention KV accounts for **$90.0\%$ of the throughput gap** between Hybrid and Static Batched Contiguous serving ($2.421\text{s}$ out of $2.688\text{s}$ wall-time penalty). Gathering cross-KV once at encode-to-decode handoff lifts throughput from $28.10\text{ tok/s}$ to $59.99\text{ tok/s}$.
  - **Self-KV Gather Overhead:** Re-gathering self-attention KV accounts for **$9.5\%$ of the total gap** ($0.254\text{s}$ out of $2.688\text{s}$, lifting throughput to $68.11\text{ tok/s}$).
  - **Engine/Bookkeeping Overhead:** The zero-copy control (Row 5) achieves $68.11\text{ tok/s}$ vs static batching's $68.60\text{ tok/s}$—a throughput ratio of **0.993** (99.3% of native batched throughput)—proving that scheduler wrapping overhead is less than **$0.7\%$** ($0.013\text{s}$).
  - **Theoretical Upper Bound:** Row 5 demonstrates that eliminating gather/scatter memory copies recovers near-native batched throughput without needing custom attention kernels.
  - **Direct Paged Attention Kernel (Row 6):** Measured on Windows with Triton 3.8.0. Achieves $59.23 \pm 1.83\text{ tok/s}$ vs Row 4's $59.99 \pm 0.96\text{ tok/s}$. The difference of $0.76\text{ tok/s}$ is well within run-to-run standard deviation ($\pm 0.96$ and $\pm 1.83$). Peak memory was higher (Allocated: 806.8 MB vs 641.2 MB; Reserved: 928.0 MB vs 728.0 MB) due to additional Triton JIT compilation workspace buffers, intermediate launch allocations, and extra unshared physical block tables (128 blocks). Numerical accuracy achieved 100% (20/20) exact token match parity against Hugging Face.

- **Analytical Concurrency Capacity Test (Fixed 512 MB Budget):**

| Audio Representation (Analytical) | Cross-KV / Req (MB) | Unpaged Static Mode (Reqs) | Paged KV-Cache Mode (Reqs) | Concurrency Gain | Basis / Derivation |
|:---|:---:|:---:|:---:|:---:|:---|
| **Standard 30s Padded Audio** (1500 positions) | 17.58 MB | 22 reqs (22.83 MB/req) | 28 reqs (18.14 MB/req) | **1.27x** (+27%) | Analytical: derived from allocator sizes, no run at this concurrency (12,288 B/tok, worst-case 448 vs avg 3 blocks = 48 tokens) |
| **Trimmed Audio** (~441 positions avg) | 5.17 MB | 49 reqs (10.42 MB/req) | 89 reqs (5.73 MB/req) | **1.82x** (+82%) | *Analytical, hypothetical, no parity check* (LibriSpeech mean 8.82s = 441 pos, avg 35 dec tokens = 3 blocks) |

### 4. Real-Speech Serving Evaluation (Whisper-base on LibriSpeech)
Evaluated across 100 stratified LibriSpeech `test-clean` utterances with 3 repeats per condition (`data/real_speech_evaluation_results.json`):
*Note: Evaluated on the continuous batching engine path BEFORE the cached-cross-KV optimization.*

- **Corpus WER:** All systems achieved **5.28%** corpus WER, and 95% bootstrap confidence intervals are within 0.1 percentage points of each other (Sequential: `[4.00%, 6.81%]`, Static Batching: `[4.00%, 6.81%]`, Continuous Engine: `[3.95%, 6.89%]`).
- **Throughput & Latency by Traffic Profile:**

| Traffic Profile | System | Throughput (audio-sec / wall-sec) | Latency p50 (s) | Latency p95 (s) | TTFT p50 (s) | TTFT p95 (s) | Peak VRAM Alloc / Res (MB) | Corpus WER (95% CI) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Burst** ($t=0$) | Sequential | 36.03x $\pm$ 1.60x | 12.674s | 23.524s | 12.396s | 23.472s | 189.5 / 214.0 | 5.28% [4.00%, 6.81%] |
| | Static Batching | **70.95x** $\pm$ 1.06x | **6.531s** | **11.957s** | **6.124s** | **11.742s** | 312.4 / 388.0 | 5.28% [4.00%, 6.81%] |
| | Continuous Engine | 38.53x $\pm$ 0.31x | 12.024s | 22.013s | 11.230s | 21.462s | 1761.6 / 1830.0 | 5.28% [3.95%, 6.89%] |
| **Poisson Low** ($\lambda=1.0$) | Sequential | 9.60x $\pm$ 0.01x | 0.720s | 1.758s | 0.185s | 1.127s | 189.5 / 214.0 | 5.28% [4.00%, 6.81%] |
| | Static Batching | **9.60x** $\pm$ 0.04x | **0.779s** | **1.763s** | **0.174s** | **0.808s** | 280.5 / 308.7 | 5.28% [4.00%, 6.81%] |
| | Continuous Engine | 9.59x $\pm$ 0.01x | 0.920s | 2.296s | 0.165s | 0.463s | 1762.2 / 1806.0 | 5.28% [4.00%, 6.86%] |
| **Poisson High** ($\lambda=4.0$) | Sequential | 25.36x $\pm$ 4.16x | 7.669s | 12.273s | 7.428s | 12.019s | 189.5 / 214.0 | 5.28% [4.00%, 6.81%] |
| | Static Batching | **36.10x** $\pm$ 0.43x | **1.254s** | **2.030s** | **0.612s** | **1.386s** | 312.1 / 401.3 | 5.28% [4.00%, 6.81%] |
| | Continuous Engine | 25.18x $\pm$ 0.08x | 7.161s | 11.914s | 6.112s | 10.847s | 1762.4 / 1831.3 | 5.28% [4.04%, 6.84%] |

*Footnote: Latency is defined as end-to-end latency per request, from arrival to last token.*

- **Traffic Analysis:**
  - **Sequential Parity:** The continuous engine ties sequential under load (Poisson High: 25.18x vs 25.36x, Poisson Low: 9.59x vs 9.60x) and is only about 7% above it in burst (38.53x vs 36.03x). Static batching achieves higher peak throughput (70.95x in burst, 36.10x in Poisson High) because it avoids dynamic scheduler per-step gather/scatter overhead.
  - **Memory Footprint Note:** The engine's ~1.76 GB allocated VRAM reflects the preallocated 1344 MB physical block pools (6144 encoder blocks = 1152.0 MB + 1024 decoder blocks = 192.0 MB) plus model weights, not memory dynamically used by active tokens. Actually-used pool memory was not measured during this benchmark run.

- **Repeated-Audio Workload (Cache ON vs Cache OFF, 100 requests):**

| Metric | Cache OFF | Cache ON | Delta / Reduction | Effect Characterization |
|:---|:---:|:---:|:---:|:---|
| Wall Time (s) | 33.26s $\pm$ 1.00s | 30.00s $\pm$ 1.16s | -3.26s (-9.8%) | Modest (2.1x combined $\sigma$) |
| Latency p50 (s) | 15.538s $\pm$ 0.603s | 14.480s $\pm$ 0.732s | -1.057s (-6.8%) | Modest (1.1x combined $\sigma$) |
| Encoder Calls | 100 | 50 | -50 (-50.0%) | Exact 50% skip |
| Skipped Calls | 0 (0.0%) | 50 (50.0%) | +50 skips | 50/100 duplicate hits |
| Encoder Time Saved | 0.00s | 1.40s | +1.40s | Pure encoder compute avoided |

  - *Characterization:* The wall-time reduction is 9.8% (33.26s to 30.00s; difference is 3.26s), which is ~2.1x the combined run-to-run standard deviation ($\sqrt{1.00^2 + 1.16^2} = 1.53\text{s}$). The p50 latency reduction is 6.8% (15.538s to 14.480s; difference is 1.057s). Because these differences are within about 3x run-to-run standard deviation, the speedup effect on end-to-end serving is described as modest.

- **Concurrency & Queue Telemetry:**

| Profile | Avg In-Flight Reqs | Max In-Flight Reqs | Queue Wait Mean (s) | Queue Wait p50 (s) | Queue Wait p95 (s) | Rejections |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Burst** ($t=0$) | 3.80 | 4 | 1.627s | 1.694s | 2.070s | 0 |
| **Poisson Low** ($\lambda=1.0$) | 1.58 | 4 | 0.037s | 0.003s | 0.183s | 0 |
| **Poisson High** ($\lambda=4.0$) | 3.64 | 4 | 2.244s | 2.455s | 3.191s | 0 |

  - *Capacity Cap Finding:* The preallocated pools hold 6144 encoder blocks (1152.0 MB) and 1024 decoder blocks (192.0 MB). Each 30s utterance requires 94 encoder blocks (17.625 MB), meaning the pools physically fit ~65 concurrent encoder requests ($6144 / 94 \approx 65.3$).
  - However, max in-flight requests across all profiles is strictly 4.
  - This cap comes directly from the scheduler batch limit configured as `max_batch_size: int = 4` in `scripts/evaluate_real_speech.py` (line 277, line 306, line 480) and enforced in `speech_serving/phase_aware_scheduler.py` line 357 (`while len(self.active_encode) + len(self.active_decode) < self.max_batch_size and self.queue:`).
  - Because active concurrency was capped at 4 by the scheduler, memory utilization never reached the 75% threshold, and the high-water-mark memory guard never engaged.

### 5. Scheduler Microbenchmark (Random Weights, Synthetic Inputs)
Evaluated on NVIDIA RTX 3050 Laptop GPU (FP16 precision, 3 repeats per scenario, saved in `microbenchmark_results.json`):

| Workload Scenario | Naive Mean (tok/s) | Naive Peak VRAM (MB) | Paged Engine (tok/s) | Paged Peak VRAM (MB) | Prefix Hit Rate (%) | Speedup (Mean) | Speedup Range |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Uniform Dense** (6s window, n=24) | 138.23 | 25.42 | 154.85 | 31.06 | 13.39% | **1.13x** | 1.05x - 1.27x |
| **Bursty Clustered** (n=24) | 65.84 | 25.90 | 70.42 | 31.54 | 12.71% | **1.07x** | 1.05x - 1.10x |
| **Uniform Moderate** (15s window, n=24) | 80.62 | 24.97 | 82.92 | 30.60 | 12.93% | **1.03x** | 0.98x - 1.12x |

- **Notes & Telemetry Explanation:**
  - In this benchmark, paged peak VRAM is higher than naive (30.60–31.54 MB vs 24.97–25.90 MB) due to block pool allocation.
  - Note: This synthetic workload uses prefix sharing, which is NOT safe for real bidirectional encoders per the context-sensitivity results, testing scheduler mechanics only.
  - In the synthetic workload generator, 33% of requests (`i % 3 == 0`, 8 of 24 requests) share a fixed 48-frame audio prefix. Previously, `enc_bm.hit_rate` reported 0.0% because the scheduler only incremented its private `encoder_calls_skipped` counter and omitted calls to `enc_bm.total_lookups` and `enc_bm.cache_hits`. Hooking up the block manager hit telemetry accurately reflects the true ~13% physical block reuse rate.

---

## Repository Structure

```
├── speech_serving/
│   ├── block_manager.py                 # Self-contained BlockManager (O(1) allocation, zero leaks)
│   ├── models.py                        # PretrainedWhisperModel & StreamingMultiCodebookTTSModel
│   ├── phase_aware_scheduler.py         # Continuous batching scheduler with lookup-before-encode
│   ├── encoder_block_manager.py         # Paged non-causal encoder memory & ref-counting
│   ├── encoder_attention.py             # Non-causal paged attention kernel
│   ├── dual_cache_manager.py            # Coordinator managing dual physical block pools
│   ├── tts_engine.py                    # Streaming multi-codebook TTS engine
│   └── memory_guard.py                  # High-water mark tracking & LRU bulk eviction
├── paged_kv_cache/                      # Compatibility shim re-exporting BlockManager
├── scripts/
│   ├── check_wer_sanity.py              # WER calculation sanity checker
│   ├── evaluate_context_sensitivity.py  # Encoder context sensitivity benchmark
│   ├── evaluate_paging_ablation.py      # Decode breakdown & 2x2 ablation
│   ├── evaluate_paging_variants.py      # Full 6-row decode-path ablation matrix (Triton)
│   ├── evaluate_real_speech.py          # Real-speech serving evaluation runner
│   ├── generate_report_tables.py        # Reproducible table and metrics report generator
│   ├── parity_check.py                  # 20-utterance parity check vs HF generate
│   ├── prepare_dataset.py               # Stratified LibriSpeech test-clean subset generator
│   ├── profile_gather_decomposition.py  # Gather/compute time breakdown profiler
│   └── update_notebook.py               # Notebook sync and cell generator
├── data/
│   ├── encoder_context_sensitivity_results.json # 50-pair context sensitivity results
│   ├── librispeech_subset_100.json              # 100-utterance stratified metadata
│   ├── paging_ablation_results.json             # Full ablation & capacity test results
│   ├── parity_check_results.json                # 100% exact match parity results
│   ├── real_speech_evaluation_results.json      # Full 100-utterance 3-repeat benchmarks
│   └── step1_gather_decomposition.json          # CUDA event breakdown (B=1,4,8,16; L=32,128)
├── microbenchmark_results.json          # Scheduler microbenchmark raw results
├── test_speech_engine.py                # Automated unit test suite (10/10 PASS)
├── requirements.txt                     # Pinned environment versions
├── FINDINGS.md                          # Detailed empirical evaluation report
└── paged_dual_cache_speech_serving.ipynb # Interactive walkthrough notebook
```

---

## Quickstart & Reproduction

### 1. Environment Setup
```bash
git clone https://github.com/aditya3799/inference-speech-model.git
cd inference-speech-model
pip install -r requirements.txt
```

### 2. Run Automated Verification (10/10 PASS)
```bash
py -3.10 -m pytest test_speech_engine.py -v
```

### 3. Generate Report Tables Directly from JSONs
```bash
py -3.10 scripts/generate_report_tables.py
```

### 4. Run Hugging Face Parity Check
```bash
py -3.10 scripts/parity_check.py --num-samples 20
```

### 5. Run Context-Sensitivity Benchmark (50 Pairs)
```bash
py -3.10 scripts/evaluate_context_sensitivity.py --num-pairs 50
```

### 6. Run Gather Decomposition Profiling
```bash
py -3.10 scripts/profile_gather_decomposition.py
```

### 7. Run Full 6-Row Decode-Path Ablation Matrix
```bash
py -3.10 scripts/evaluate_paging_variants.py
```

### 8. Run Real-Speech Serving Evaluation (100 Utterances, 3 Repeats)
```bash
py -3.10 scripts/evaluate_real_speech.py --repeats 3 --num-samples 100 --output data/real_speech_evaluation_results.json
```
