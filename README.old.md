# Paged Dual-Cache Speech Serving Engine

**Status: Research Prototype & Empirical Evaluation.** This repository explores PagedAttention-style continuous batching, dual physical block pools, and lookup-before-encode prefix caching for speech models (Whisper-class ASR and streaming multi-codebook TTS).

> [!IMPORTANT]
> **Evaluation & Audit (October 2026):**
> Following reviewer feedback, this repository has been comprehensively audited and evaluated on real speech:
> 1. All prior synthetic 1.5x–3x throughput claims (which stemmed from noisy unbatched baselines or external text-LLM figures) have been retracted. Measured microbenchmark speedups are **1.03x to 1.13x**.
> 2. The real-speech evaluation integrates `openai/whisper-base` on LibriSpeech `test-clean` (100 stratified utterances, 3 repeats) achieving **5.28% corpus WER** (95% CI: `[4.00%, 6.81%]`) and **99.0% token match rate (99/100)** against Hugging Face `generate()`.
> 3. Encoder context-sensitivity experiments across 50 pairs demonstrate that chunk-level sharing is **UNSOUND** for bidirectional speech encoders ($171\times$ numerical noise floor and downstream token divergence in 14/50 pairs).
> 4. Decode-path ablation isolates the benefit of batching (3.53x) from paging overhead (38.6%–39.7% decode step copy tax when reconstructing contiguous tensors).
> 5. Concurrency capacity analysis incorporates Whisper's 17.58 MB cross-attention KV, yielding realistic gains of **1.27x** (30s padded) and **1.88x** (trimmed audio), retracting the flawed 12.5x decoder-only claim.
>
> See [FINDINGS.md](FINDINGS.md) for full empirical details and raw JSON figures.

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
2. **Lookup-Before-Encode**: Audio features are hashed *before* the encoder runs. On a cache hit, the encoder forward pass is completely skipped, and pre-projected cross-attention KV tensors are gathered directly.
3. **Sound Full-Utterance Hashing**: Because Whisper's encoder is bidirectional, chunk-level representations vary depending on surrounding context. The cache indexes full audio inputs to guarantee mathematical representation invariance.
4. **Pretrained Whisper Integration**: Wrapped via `PretrainedWhisperModel`, caching layer-specific cross-attention projections ($K_l, V_l$) during the encode pass and applying Whisper `suppress_tokens` for exact generation parity.
5. **Memory Guard & High-Water Mark**: Backpressures incoming requests when projected utilization exceeds 75%, preventing CUDA out-of-memory crashes with zero block leaks.

---

## Empirical Benchmark Findings

All numbers below come from executed benchmarks and saved JSON logs. No numbers are hand-edited.

### 1. Parity Check Against Hugging Face `generate()`
Verified across 20 LibriSpeech `test-clean` utterances with `openai/whisper-base`:
- **FP32 Exact Token Match Rate:** **100.0% (20/20)**, WER delta = 0.000.
- **FP16 Exact Token Match Rate:** **100.0% (20/20)**, WER delta = 0.000.
- Confirms the custom cross-attention caching and greedy decoding logic matches Hugging Face token-for-token under single-stream execution.

### 2. Task 3: Encoder Context Sensitivity (Real Whisper-base, 50 Pairs)
Evaluated in `scripts/evaluate_context_sensitivity.py`:
- **Calibration Noise Floor:** $2.22 \times 10^{-3}$ relative $L_2$ error (FP16 vs FP32).
- **Experiment E1 (Trailing Context, $n=50$ pairs):**
  - Relative $L_2$ error: Median = **0.1914**, p95 = **0.3197**, Max = **0.3799** ($170.8\times$ noise floor).
  - Token-level divergence: **14 / 50 pairs diverged** (Exact match rate: **72.0%**).
  - Spliced WER delta: **+0.0009** (95% bootstrap CI: `[-0.0067, +0.0092]`).
- **Experiment E2 (Offset Shifts, $n=150$ shifts across 2s, 5s, 10s):**
  - Relative $L_2$ error: Median = **0.8540**, p95 = **0.9956**, Max = **1.0067** ($452.6\times$ noise floor).
  - Token-level divergence: **46 / 150 shifts diverged** (Exact match rate: **68.9%**).
  - Spliced WER delta: **+0.0074** (95% bootstrap CI: `[-0.0040, +0.0230]`).
- **Evidence-Grounded Verdict:**
  `UNSOUND: E1 trailing context diverges at 171x noise floor (14/50 diverged pairs, 72.0% match rate). Chunk-level prefix cache sharing violates bidirectional encoder attention invariance.` Full-utterance caching is mathematically and empirically required.

### 3. Task 5: Paging vs Batching Ablation & Concurrency Capacity
Evaluated in `scripts/evaluate_paging_variants.py` and `scripts/profile_gather_decomposition.py`:
- **Gather Cost Decomposition ($B=4, L=32$ to $128$):**
  - Cross-attention gather (1500 positions) consumes **50.0% to 57.7%** of decode step time ($71.6\text{ms}$ at $B=4$, copying $70.3\text{ MB/step}$).
  - Self-attention gather consumes **5.7% to 6.9%** ($8.0\text{--}9.8\text{ms}$, copying $1.5\text{--}6.0\text{ MB/step}$).
  - Attention compute itself consumes **15.6% to 20.9%** ($19.8\text{--}29.8\text{ms}$).
- **Measured 6-Row Ablation Matrix ($B=4$, 32 tokens generated, 3 repeats):**

| Variant | Throughput (tok/s) | Latency per Req (s) | Step Wall Time (s) | Peak VRAM Alloc (MB) | Peak VRAM Res (MB) | HF Parity |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Row 1: Sequential** (No Batching, No Paging) | 34.55 $\pm$ 0.01 | 0.926 | 3.704 | 248.3 | 262.0 | 20/20 (100%) |
| **Row 2: Static Batched Contiguous** (Batching, No Paging) | **68.60** $\pm$ 0.16 | **0.467** | **1.866** | 305.4 | 324.0 | 20/20 (100%) |
| **Row 3: Hybrid Engine** (Gather Every Step, Incl. Cross-KV) | 28.10 $\pm$ 0.43 | 1.139 | 4.554 | 641.2 | 728.0 | 20/20 (100%) |
| **Row 4: Hybrid + Cached Cross-KV** (Gather Self-KV Only) | 59.99 $\pm$ 0.96 | 0.534 | 2.134 | 641.2 | 728.0 | 20/20 (100%) |
| **Row 5: Paged Storage, Contiguous Alloc** (Zero-Copy View) | 68.11 $\pm$ 0.03 | 0.470 | 1.879 | 643.6 | 728.0 | 20/20 (100%) |
| **Row 6: Direct Paged Kernel** (Triton) | 59.23 $\pm$ 1.83 | 0.540 | 2.161 | 806.8 | 928.0 | 20/20 (100%) |

- **Isolating the Paging Cost & Gap Breakdown:**
  - **Cross-KV Gather Overhead:** Re-gathering 1500-position cross-attention KV accounts for **$90.0\%$ of the throughput gap** between Hybrid and Static Batched Contiguous serving ($2.420\text{s}$ out of $2.688\text{s}$ penalty). Gathering cross-KV once at encode-to-decode handoff brings throughput from $28.10\text{ tok/s}$ to $59.99\text{ tok/s}$.
  - **Self-KV Gather Overhead:** Re-gathering self-attention KV accounts for **$9.5\%$ of the remaining gap** ($0.255\text{s}$, lifting throughput from $59.99\text{ tok/s}$ to $68.11\text{ tok/s}$).
  - **Engine/Bookkeeping Overhead:** The zero-copy control (Row 5) achieves $68.11\text{ tok/s}$ vs static batching's $68.60\text{ tok/s}$ ($99.3\%$ of native batched throughput), showing that scheduler wrapping overhead is less than **$0.7\%$** ($0.013\text{s}$).
  - **Theoretical Upper Bound:** Row 5 demonstrates that eliminating gather/scatter tensor copies recovers near-native batched throughput without needing custom attention kernels.
  - **Direct Paged Attention Kernel (Row 6):** Measured on Windows with Triton 3.8.0. Achieves $59.23\text{ tok/s}$ ($2.11\times$ faster than full gather) with $100\%$ HF exact match token parity.

- **Analytical Concurrency Capacity Test (Fixed 512 MB Budget):**
  Accounting for both decoder self-attention KV ($12,288\text{ bytes/token}$) and Whisper's cross-attention KV ($12,288\text{ bytes/token}$):
  - **Standard 30s Padded Audio (1500 positions, 17.58 MB cross KV):**
    - Unpaged Static Mode (448 tokens worst-case): 22.83 MB/req $\to$ **22 concurrent requests**.
    - Paged KV-Cache Mode (avg ~35 tokens): 18.14 MB/req $\to$ **28 concurrent requests**.
    - Capacity Multiplier: **1.27x** (+27% capacity).
  - **Trimmed Audio (LibriSpeech mean 8.82s = 441 positions, 5.17 MB cross KV):**
    - Unpaged Static Mode: 10.42 MB/req $\to$ **49 concurrent requests**.
    - Paged KV-Cache Mode: 5.73 MB/req $\to$ **92 concurrent requests**.
    - Capacity Multiplier: **1.88x** (+88% capacity).
  - *Retraction:* The prior 12.5x capacity claim omitted the 17.58 MB cross-attention KV and is retracted.

### 4. Real-Speech Serving Evaluation (Whisper-base on LibriSpeech)
Evaluated across 100 stratified LibriSpeech `test-clean` utterances with 3 repeats per condition (`data/real_speech_evaluation_results.json`):
- **Corpus WER:** **5.28%** (95% bootstrap CI: `[4.00%, 6.81%]`) across all systems.
- **Hugging Face Token Match:** **99.0% (99/100 utterances identical)**. Exactly one utterance diverged: `2830-3980-0051` at token position 8 where continuous batching FP16 accumulation emitted token 11 (`,`) (`"God,"` vs `"God"`). Both normalize to 100% identical text with Whisper `EnglishTextNormalizer`.
- **Traffic Profile Results:**

| Traffic Profile | System | Throughput (audio-sec / wall-sec) | Latency p50 (s) | TTFT p50 (s) | Peak VRAM Alloc / Res (MB) |
|:---|:---|:---:|:---:|:---:|:---:|
| **Burst** ($t=0$) | Sequential | 19.17x $\pm$ 1.91x | 25.41s | 24.85s | 189.5 / 214.0 |
| | Static Batching | **46.10x** $\pm$ 2.87x | **9.58s** | **9.01s** | 312.4 / 388.0 |
| | Continuous Engine | 32.70x $\pm$ 4.03x | 13.86s | 12.76s | 1761.6 / 1830.0 |
| **Poisson Low** ($\lambda=1.0$) | Sequential | 9.65x $\pm$ 0.00x | 0.40s | 0.08s | 189.5 / 214.0 |
| | Static Batching | 9.63x $\pm$ 0.02x | **0.36s** | **0.05s** | 228.2 / 252.0 |
| | Continuous Engine | 9.64x $\pm$ 0.00x | 0.43s | 0.09s | 1742.5 / 1806.0 |
| **Poisson High** ($\lambda=4.0$) | Sequential | 35.52x $\pm$ 1.08x | 1.34s | 1.07s | 189.5 / 214.0 |
| | Static Batching | **37.73x** $\pm$ 0.04x | **0.55s** | **0.20s** | 309.0 / 346.0 |
| | Continuous Engine | 36.25x $\pm$ 0.05x | 1.42s | 0.68s | 1762.1 / 1830.0 |

- **Repeated Workload (Cache ON vs Cache OFF, 100 requests):**
  - **Cache OFF:** 22.42s wall time, 100 encoder calls, 0 skipped, 0.00s saved, Latency p50: 9.63s.
  - **Cache ON:** 19.95s wall time, 50 encoder calls, **50 skipped (50.0%)**, **0.97s saved**, Latency p50: **8.64s** (10.3% latency reduction).
- **Concurrency & Resource Telemetry:**
  - Block Pools: Encoder pool = 6144 blocks (1152.0 MB), Decoder pool = 1024 blocks (192.0 MB), Total pool = 1344.0 MB.
  - In-flight concurrency: Avg 3.80 requests (Burst), 3.34 (Poisson High), 1.22 (Poisson Low), Max = 4.
  - Queue wait time: 1.959s (Burst), 0.632s (Poisson High), 0.004s (Poisson Low).
  - Rejections: 0 rejections across all runs.

### 5. Scheduler Microbenchmark (Random Weights, Synthetic Inputs)
Evaluated on NVIDIA RTX 3050 Laptop GPU (FP16 precision, 3 repeats per scenario, saved in `microbenchmark_results.json`):

| Workload | Arrival Profile | Paged Engine TPS | Naive TPS | Speedup (min / mean / max) | Prefix Hit Rate |
|:---|:---|:---:|:---:|:---:|:---:|
| **Uniform Dense** | 6s arrival window, high concurrency | 154.9 tok/s | 138.2 tok/s | **1.05x / 1.13x / 1.27x** | 13.4% |
| **Bursty** | Clustered arrival surges | 70.4 tok/s | 65.8 tok/s | **1.05x / 1.07x / 1.10x** | 12.7% |
| **Uniform Moderate** | Dispersed arrivals across 15s window | 82.9 tok/s | 80.6 tok/s | **0.98x / 1.03x / 1.12x** | 12.9% |

> **Prefix Hit Telemetry Explanation:**
> In the synthetic workload generator, 33% of requests (`i % 3 == 0`, 8 of 24 requests) share a fixed 48-frame audio prefix. Previously, `enc_bm.hit_rate` reported 0.0% because the scheduler only incremented its private `encoder_calls_skipped` counter and omitted calls to `enc_bm.total_lookups` and `enc_bm.cache_hits`. Hooking up the block manager hit telemetry accurately reflects the true ~13% physical block reuse rate.

---

## Repository Structure

```
├── speech_serving/
│   ├── block_manager.py           # Self-contained BlockManager (O(1) allocation, zero leaks)
│   ├── models.py                  # PretrainedWhisperModel & StreamingMultiCodebookTTSModel
│   ├── phase_aware_scheduler.py   # Continuous batching scheduler with lookup-before-encode
│   ├── encoder_block_manager.py   # Paged non-causal encoder memory & ref-counting
│   ├── encoder_attention.py       # Non-causal paged attention kernel
│   ├── dual_cache_manager.py      # Coordinator managing dual physical block pools
│   ├── tts_engine.py              # Streaming multi-codebook TTS engine
│   └── memory_guard.py            # High-water mark tracking & LRU bulk eviction
├── paged_kv_cache/                # Compatibility shim re-exporting BlockManager
├── scripts/
│   ├── prepare_dataset.py         # Stratified LibriSpeech test-clean subset generator
│   ├── parity_check.py            # 20-utterance parity check vs HF generate
│   ├── evaluate_real_speech.py    # Real-speech serving evaluation runner
│   ├── evaluate_context_sensitivity.py  # Task 3 encoder context sensitivity benchmark
│   ├── evaluate_paging_ablation.py      # Task 5 decode breakdown & 2x2 ablation
│   └── update_notebook.py         # Notebook sync and cell generator
├── data/
│   ├── librispeech_subset_100.json              # 100-utterance stratified metadata
│   ├── parity_check_results.json               # 100% exact match parity results
│   ├── real_speech_evaluation_results.json      # Full 100-utterance 3-repeat benchmarks
│   ├── encoder_context_sensitivity_results.json # 50-pair context sensitivity results
│   └── paging_ablation_results.json             # Full ablation & capacity test results
├── test_speech_engine.py          # Automated unit test suite (10/10 PASS)
├── requirements.txt               # Pinned environment versions
├── FINDINGS.md                    # Detailed empirical evaluation report
└── paged_dual_cache_speech_serving.ipynb  # Interactive walkthrough notebook
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

### 3. Run Hugging Face Parity Check
```bash
py -3.10 scripts/parity_check.py --num-samples 20
```

### 4. Run Task 3 Context-Sensitivity Benchmark (50 Pairs)
```bash
py -3.10 scripts/evaluate_context_sensitivity.py --num-pairs 50
```

### 5. Run Task 5 Paging vs Batching Ablation Benchmark
```bash
py -3.10 scripts/evaluate_paging_ablation.py
```

### 6. Run Real-Speech Serving Evaluation (100 Utterances, 3 Repeats)
```bash
py -3.10 scripts/evaluate_real_speech.py --repeats 3 --num-samples 100 --output data/real_speech_evaluation_results.json
```
