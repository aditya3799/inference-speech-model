# FINDINGS.md — Paged Dual-Cache Speech Serving Engine

**Date:** October 2026  
**Scope:** Complete evaluation, code hardening, and empirical analysis (Tasks 1 through 6).

---

## Executive Summary & Non-Negotiable Findings

1. **The 1.5x–3x Throughput Claims Are Fully Audited and Retracted:**
   - Prior documentation cited throughput gains of 1.44x to 3.0x. An audit revealed these originated from an external text-LLM KV-cache repository or early unbatched runs with noisy baselines.
   - The scheduler microbenchmark (`microbenchmark_results.json` on RTX 3050 Laptop GPU, FP16) achieves steady mean speedups of **1.13x** under dense traffic (154.85 vs 138.23 tok/s), **1.07x** under bursty traffic (70.42 vs 65.84 tok/s), and **1.03x** under moderate traffic (82.92 vs 80.62 tok/s).
   - In this benchmark, paged peak VRAM is higher than naive (30.60–31.54 MB vs 24.97–25.90 MB) due to block pool allocation.
   - Note: This synthetic workload uses prefix sharing, which is NOT safe for real bidirectional encoders per the context-sensitivity results, so it tests scheduler mechanics only.
   - The ~13% prefix hit rate reflects true physical block reuse for the 33% shared-prefix synthetic stream (`is_shared_prefix = (i % 3 == 0)`); it was previously reported as 0.0% due to an uninstrumented hit-path counter.

2. **Full Parity with Hugging Face Whisper-base (Task 4):**
   - The engine integrates `openai/whisper-base` (HF commit `e37978b90ca9030d5170a5c07aadb050351a65bb`) via `PretrainedWhisperModel`.
   - On 20 LibriSpeech test-clean utterances (`data/parity_check_results.json`), single-stream greedy decoding achieved **100.0% Exact Token Match (20/20)** in both FP32 and FP16 with Hugging Face `generate()`.
   - On the 100-utterance continuous batching serving run (`data/real_speech_evaluation_results.json`), the token match rate is **99.0% (99/100 utterances identical)** across all 3 repeats. Exactly one utterance diverged: `2830-3980-0051` (sample index 67), where continuous batching FP16 accumulation at token position 8 emitted token 11 (`,`) (`"God,"` vs `"God"`). Both normalize to 100% identical text with Whisper `EnglishTextNormalizer`.

3. **Encoder Context Sensitivity & Full-Utterance Caching (Task 3):**
   - Evaluated across 50 LibriSpeech utterance pairs (`data/encoder_context_sensitivity_results.json`). Calibration established a numerical noise floor of $2.22 \times 10^{-3}$ relative $L_2$ error (FP16 vs FP32).
   - In Experiment E1 (trailing context effect, $n=50$ pairs), **14 / 50 pairs diverged** at the token level (**72.0%** exact token match rate). Relative $L_2$ error: median = **0.1914** ($86.1\times$ noise floor), p95 = **0.3197** ($143.8\times$ noise floor), max = **0.3799** ($170.8\times$ noise floor). Mean spliced WER delta: **+0.0009** (95% bootstrap CI: `[-0.0067, +0.0092]`).
   - In Experiment E2 (offset shifts, $n=148$ shifts across 2s, 5s, 10s), **46 / 148 shifts diverged** at the token level (**68.9%** exact token match rate). Relative $L_2$ error: median = **0.8540** ($384.0\times$ noise floor), p95 = **0.9956** ($447.6\times$ noise floor), max = **1.0067** ($452.6\times$ noise floor). Mean spliced WER delta: **+0.0074** (95% bootstrap CI: `[-0.0040, +0.0230]`).
   - **Verdict:** Chunk-level sharing is not exactly equivalent (E1: 14/50 pairs diverge at token level; E2: 46/148), with no detectable average WER effect (spliced WER delta CIs include zero), so the engine hashes full utterances.

4. **Batching Drives Throughput; Cross-KV Gather Accounts for 90% of the Paging Gap (Task 5):**
   - Static batched contiguous decoding provides a **1.99x speedup** over sequential decoding (68.60 tok/s vs 34.55 tok/s at B=4, 32 tokens, from `data/paging_ablation_results.json`).
   - In the paged decode path, re-gathering 1500-position cross-attention KV every step accounts for **90.0% of the throughput gap** between hybrid gather paging and static batching ($2.421\text{s}$ out of $2.688\text{s}$ wall-time penalty).
   - Caching cross-attention KV gathered once at encode-to-decode handoff lifts throughput from $28.10\text{ tok/s}$ to $59.99\text{ tok/s}$.
   - Self-attention gather accounts for **9.5% of the total gap** ($0.254\text{s}$, lifting throughput to $68.11\text{ tok/s}$).
   - The zero-copy control achieves a throughput ratio of **0.993** relative to static batching ($68.11 / 68.60$, 99.3%), demonstrating that scheduler bookkeeping overhead is **$< 0.7\%$**.

5. **Concurrency Capacity Math Corrected (Cross-Attention KV Included):**
   - Whisper's 30-second audio creates 1500 encoder positions requiring **17.58 MB** of cross-attention KV per active request ($12,288\text{ bytes/token} \times 1500$).
   - At a 512 MB VRAM budget (`data/paging_ablation_results.json`):
     - **Standard 30s Padded Audio (1500 positions, 17.58 MB cross KV):** Unpaged baseline fits **22 requests** (22.83 MB/req); Paged engine fits **28 requests** (18.14 MB/req), yielding a **1.27x capacity gain** (+27%).
     - **Trimmed Audio (~441 positions avg, 5.17 MB cross KV):** *Analytical, hypothetical, no parity check.* Unpaged baseline fits **49 requests** (10.42 MB/req); Paged engine fits **89 requests** (5.73 MB/req), yielding a **1.82x capacity gain** (+82%), assuming average 35 decoder tokens (3 blocks = 48 tokens).
   - The prior 12.5x claim accounted only for decoder self-attention (5.25 MB vs 0.41 MB) and is retracted.

---

## Detailed Task Breakdown

### Task 1: Reproducibility & Dependency Fixes
- **Self-Contained Allocator:** Replaced external dependencies with `speech_serving/block_manager.py` (free-list allocator, shadow leak-checking, O(1) allocation/free).
- **Environment Pinning:** `requirements.txt` pins exact versions: `torch==2.6.0+cu124`, `transformers==4.57.0`, `datasets==2.21.0`, `jiwer==4.0.0`, `soundfile==0.14.0`, `pytest==8.4.2`.
- **Unit Test Suite:** All automated tests passing in `test_speech_engine.py` (10/10 PASS), including `test_encoder_not_called_on_hit` (sequential hit semantics documented and validated).

### Task 2: Lookup-Before-Encode & Cache Semantics
- **Lookup Order:** `_process_encode_phase` hashes full audio features *before* invoking the encoder.
- **Cache Hit Path:** On a sequential hash hit with `retain_encoder_cache=True`, the encoder forward pass is completely skipped, ref-counts are incremented, and pre-projected cross-attention KV tensors are reused.
- **Bounded Eviction:** Implemented bounded LRU eviction (`max_cached_encoder_inputs=100`) ensuring zero block leaks across hit and miss paths.

### Task 3: Encoder Context-Sensitivity (Real Whisper-base, 50 Pairs)
Evaluated in `scripts/evaluate_context_sensitivity.py` (`data/encoder_context_sensitivity_results.json`):
- **Numerical Noise Floor:**
  - Determinism (same input twice): $\text{rel\_L2} = 0.0$, $\max |diff| = 0.0$.
  - Batch position invariance (pos 0 vs pos 1): $\text{rel\_L2} = 0.0$, $\max |diff| = 0.0$.
  - FP16 vs FP32: $\text{rel\_L2} = 2.2241 \times 10^{-3}$, $\max |diff| = 0.6491$.
- **Experiment E1 (Trailing Context Effect, $n=50$ pairs):**
  - Utterance A alone vs Utterance A followed by Utterance B at offset 0s.
  - Position embeddings for A are identical. Divergence is pure bidirectional self-attention across context boundaries.
  - Token-level divergence: **14 / 50 pairs diverged** at the token level (exact match rate: **72.0%**).
  - Relative $L_2$ error: Median = **0.1914** ($86.1\times$ noise floor), p95 = **0.3197** ($143.8\times$ noise floor), Max = **0.3799** ($170.8\times$ noise floor).
  - Downstream Transcription: Mean spliced WER delta = **+0.0009** (95% bootstrap CI: `[-0.0067, +0.0092]`).
- **Experiment E2 (Offset Shift Effect, $n=148$ shifts):**
  - Utterance A shifted by 2s, 5s, 10s.
  - Token-level divergence: **46 / 148 shifts diverged** (exact match rate: **68.9%**).
  - Relative $L_2$ error: Median = **0.8540** ($384.0\times$ noise floor), p95 = **0.9956** ($447.6\times$ noise floor), Max = **1.0067** ($452.6\times$ noise floor).
  - Downstream Transcription: Mean spliced WER delta = **+0.0074** (95% bootstrap CI: `[-0.0040, +0.0230]`).
- **Verdict:**
  Chunk-level sharing is not exactly equivalent (E1: 14/50 pairs diverge at token level; E2: 46/148), with no detectable average WER effect (spliced WER delta CIs include zero), so the engine hashes full utterances.

### Task 4: Real-Speech Serving Evaluation (Whisper-base on LibriSpeech)
Evaluated across 100 stratified LibriSpeech `test-clean` utterances with 3 repeats per condition (`data/real_speech_evaluation_results.json`):
*Note: The serving throughput, repeated-audio cache, and concurrency figures below were evaluated on the continuous batching engine path BEFORE the cached-cross-KV optimization.*

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

### Task 5: Paging vs. Batching Ablation Benchmark & Direct Paged Attention
Evaluated in `scripts/evaluate_paging_variants.py` and `scripts/profile_gather_decomposition.py`:
*Note: Ablation rows 4–6 evaluate the cached-cross-KV, zero-copy, and direct Triton kernel variants.*

- **Step 1 Gather Cost Decomposition ($B=4, L=32$ to $128$):**
  - Cross-attention gather (1500 positions) consumes **50.0% to 51.4%** of decode step time (mean $71.56\text{--}71.60\text{ms}$, copying $70.3\text{ MB/step}$).
  - Self-attention gather consumes **5.7% to 6.9%** (mean $7.95\text{--}9.82\text{ms}$, copying $1.5\text{--}6.0\text{ MB/step}$).
  - Attention compute itself consumes **12.7% to 13.8%** (mean $17.62\text{--}19.81\text{ms}$).
  - Other (projections, MLP, scatter, bookkeeping) consumes **28.4% to 29.2%** (mean $40.65\text{--}40.68\text{ms}$).
  - *Note:* The millisecond values are inflated by synchronization overhead (`torch.cuda.synchronize()` around each sub-layer hook region), so the percentage proportions are the reliable metric.

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
  - **Cross-KV Gather Overhead:** Re-gathering 1500-position cross-attention KV accounts for **$90.0\%$ of the entire throughput gap** between Hybrid and Static Batched Contiguous serving ($2.421\text{s}$ out of $2.688\text{s}$ wall-time penalty). Gathering cross-KV once at encode-to-decode handoff boosts throughput from $28.10\text{ tok/s}$ to $59.99\text{ tok/s}$.
  - **Self-KV Gather Overhead:** Re-gathering self-attention KV accounts for **$9.5\%$ of the total gap** ($0.254\text{s}$ out of $2.688\text{s}$, moving throughput from $59.99\text{ tok/s}$ to $68.11\text{ tok/s}$).
  - **Engine/Bookkeeping Overhead:** The zero-copy control (Row 5) achieves $68.11\text{ tok/s}$ vs static batching's $68.60\text{ tok/s}$—a throughput ratio of **0.993** (99.3% of native batched throughput)—proving that scheduler wrapping overhead is less than **$0.7\%$** ($0.013\text{s}$).
  - **Theoretical Upper Bound:** Row 5 demonstrates that eliminating gather/scatter memory copies recovers near-native batched throughput without needing custom attention kernels.
  - **Direct Paged Attention Kernel (Row 6):** Measured on Windows with Triton 3.8.0. Achieves $59.23 \pm 1.83\text{ tok/s}$ vs Row 4's $59.99 \pm 0.96\text{ tok/s}$. The difference of $0.76\text{ tok/s}$ is well within run-to-run standard deviation. Peak memory was higher (Allocated: 806.8 MB vs 641.2 MB; Reserved: 928.0 MB vs 728.0 MB) due to additional Triton JIT compilation workspace buffers, intermediate launch allocations, and extra unshared physical block tables (128 blocks). Numerical accuracy achieved 100% (20/20) exact token match parity against Hugging Face.

- **Analytical Concurrency Capacity Test (Fixed 512 MB Budget):**

| Audio Representation (Analytical) | Cross-KV / Req (MB) | Unpaged Static Mode (Reqs) | Paged KV-Cache Mode (Reqs) | Concurrency Gain | Basis / Derivation |
|:---|:---:|:---:|:---:|:---:|:---|
| **Standard 30s Padded Audio** (1500 positions) | 17.58 MB | 22 reqs (22.83 MB/req) | 28 reqs (18.14 MB/req) | **1.27x** (+27%) | Analytical: derived from allocator sizes, no run at this concurrency (12,288 B/tok, worst-case 448 vs avg 3 blocks = 48 tokens) |
| **Trimmed Audio** (~441 positions avg) | 5.17 MB | 49 reqs (10.42 MB/req) | 89 reqs (5.73 MB/req) | **1.82x** (+82%) | *Analytical, hypothetical, no parity check* (LibriSpeech mean 8.82s = 441 pos, avg 35 dec tokens = 3 blocks) |

  - *Retraction:* The prior 12.5x capacity claim omitted the 17.58 MB cross-attention KV and is retracted.
