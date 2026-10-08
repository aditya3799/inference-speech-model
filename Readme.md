# Paged Dual-Cache Speech Serving Engine

**Status: Early-stage / Prototype.** This is an experimental research prototype exploring PagedAttention-style continuous batching for speech models. Shared to demonstrate architectural direction, discuss findings, and invite feedback.

> [!IMPORTANT]
> **Reviewer action taken (October 2026):** This README was updated to accurately reflect what the benchmarks measure following reviewer feedback. See [Limitations and What Changed](#limitations-and-what-changed) and [FINDINGS.md](FINDINGS.md) for a full account.

---

## Overview: What This Is

Most state-of-the-art paged-attention and continuous batching serving frameworks (such as vLLM, SGLang, and TensorRT-LLM) are designed exclusively for autoregressive decoder-only text LLMs.

This project investigates applying paged memory management to **speech inference**—specifically for:
1. **Encoder-Decoder Speech Recognition (Whisper-class ASR)**: Characterized by a compute-dense, non-causal bidirectional encoder pass over audio mel-spectrogram features, coupled with an autoregressive causal decoder that cross-attends into encoder activations.
2. **Streaming Neural Codec Speech Synthesis (TTS)**: Continuous batching across multi-codebook discrete audio tokens (e.g., EnCodec, SNAC, Bark) with real-time frame streaming callbacks.

---

## Architecture: What We Built

Speech inference requires fundamentally different memory orchestration than text LLMs because the encoder and decoder exhibit contrasting lifecycle and attention patterns.

```
Incoming Speech Requests (Audio Mel Features / TTS Tokens)
               │
               ▼
┌─────────────────────────────────────────────────────────────┐
│             Phase-Aware Speech Scheduler                    │
│  - Hash BEFORE encode: identical audio skips encoder        │
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

### 1. Decoupled Dual-Cache Design
- **Non-Causal Encoder Pool (`EncoderBlockManager`)**: Computed once during the audio encoding pass. Blocks are bidirectional, fixed-length representations stored in physical block tables and read repeatedly during all subsequent decoding steps.
- **Causal Decoder Pool (`BlockManager`)**: Dynamically allocated and grown token-by-token during autoregressive sequence generation, utilizing paged KV cache tables.

### 2. Lookup-Before-Encode (Fixed in October 2026)
The original code ran the encoder **before** the cache lookup, so a cache hit still wasted encoder compute. This has been fixed:
- A MD5 hash of the **full input** mel-spectrogram (keyed by content + dtype) is computed **before** calling the encoder.
- On a hit, the encoder is skipped entirely: ref-counts are bumped, cached KV is reused.
- Blocks with `ref_count > 1` are never written (shared-block safety invariant).
- This is the correct design for a **bidirectional encoder**: the same mel chunk produces different hidden states at different positions in different full sequences, so chunk-level sharing is unsafe. Only full-input equality guarantees identical encoder outputs.

### 3. Phase-Aware Continuous Batching Scheduler
- **Encode Phase (Compute-Bound)**: Executes non-causal encoder forward passes on mel-spectrogram features, or skips them on a cache hit.
- **Decode Phase (Memory-Bound)**: Batches multi-request decoder steps concurrently using paged cross-attention into the respective encoder caches.
- **Exact Cross-Attention Masking**: Dynamically constructs and caches 4D attention masks with `torch.finfo(dtype).min` masked fill.

### 4. GPU Memory Guard & Backpressure
- **Admission Control**: Tracks physical block pool saturation. When projected memory crosses the configurable high-water mark, incoming requests are backpressured to prevent CUDA OOM.
- **LRU Bulk Eviction**: Evicts retained idle encoder prefix blocks under memory pressure.
- **Zero-Leak Guarantee**: Strict reference counting and deterministic reclamation.

---

## Verified Results & Benchmarks

### Scheduler Microbenchmark (Synthetic)

> [!WARNING]
> The figures below come from the **scheduler microbenchmark**: random Whisper-shaped weights on a synthetic model, with random mel-spectrogram inputs. They demonstrate continuous-batching scheduling behaviour, not real ASR/TTS performance on speech. Do not interpret them as claims about WER, real-audio latency, or production throughput.

The serving engine was benchmarked at synthetic model scale (**6 encoder layers, 6 decoder layers, hidden dimension 512, 8 attention heads**, random weights) against an unbatched sequential baseline across three streaming traffic scenarios. Each benchmark was run with 3 repeated trials (`--repeats 3`) on an NVIDIA RTX GPU (FP16 precision). Raw results are saved in `microbenchmark_results.json` (from `output.txt`).

| Traffic Workload | Arrival Profile | Paged Dual-Cache Engine | Unbatched Baseline | Speedup / Impact |
|:---|:---|:---:|:---:|:---:|
| **Uniform Dense** | Compressed 6s arrival window, high concurrency | **154.9 tok/s** | 138.2 tok/s | **1.05x – 1.27x** (mean **1.13x**) |
| **Bursty** | Clustered arrival surges (Beta/Poisson distribution) | **70.4 tok/s** | 65.8 tok/s | **1.05x – 1.10x** (mean **1.07x**) |
| **Uniform Moderate** | Dispersed arrivals across a 15s window | **82.9 tok/s** | 80.6 tok/s | **0.98x – 1.12x** (mean **1.03x**) |

> **What these numbers mean:** Under high concurrency (`uniform_dense`), co-scheduling requests yields modest throughput improvements (mean ~1.13x, peak 1.27x). Under sparse traffic (`uniform_moderate`), effective concurrency remains at ~1 and paging overhead produces near-parity (mean 1.03x). Note: Previous documentation cited a 1.44x mean and 1.56x figure; auditing revealed those figures originated from a separate text-LLM KV-cache project and were erroneously copied over. They have been removed.

> **What these numbers do NOT mean:** WER on real speech, latency with a real Whisper checkpoint, or generalisability to any production workload. See [Open Questions](#open-questions).

![Benchmark Results](speech_serving_benchmark.png)

### Memory Safety & Backpressure Stress Test
Under a high-concurrency burst stress test against a constrained capacity pool (`max_blocks_pool = 100`, high-water mark = 75%):
- **Requests Submitted**: 15 unique, uncacheable audio requests.
- **Admission Invariants**: 7 requests admitted, 8 requests backpressured via high-water mark activation.
- **Bulk Eviction**: Exactly 1 bulk LRU eviction event triggered, freeing 70 physical blocks without interrupting active requests.
- **Leak Verification**: After completion, allocated encoder and decoder blocks both returned to **0 / 100** (zero memory leak).

### Cross-Attention Heterogeneous Length Verification
Automated regression tests verify token-for-token numerical equivalence between:
- Requests decoded concurrently in a mixed-length batch (`audio_lengths = [40, 100, 160, 200]`).
- Requests decoded individually in isolation.

All generated tokens match reference generation 100%, confirming that the 4D attention mask prevents zero-padded audio frame contamination.

---

## Limitations and What Changed

### What was fixed (October 2026)
1. **`paged_kv_cache.BlockManager` was an external dependency** that did not ship with the repo. A clean checkout could not import it. Fix: `speech_serving/block_manager.py` now provides a complete, self-contained re-implementation of the interface. The `paged_kv_cache/` package is a local compatibility shim that re-exports it. The `sys.path` hack in `__init__.py` was removed.

2. **Encoder ran before the cache lookup** in `_process_encode_phase`. On a cache hit, the encoder was called anyway (saving no compute). Fix: full-input hash is now computed before the encoder call; on a hit the encoder is completely skipped.

3. **Block over-reservation**: `needed_enc_blocks` was computed from `mel_len` instead of `enc_seq_len` (≈ `mel_len // 2` after Conv2d stride-2), causing 2× more blocks to be reserved than actually needed.

4. **Shared-block write hazard**: `store_encoder_kv` was called even on cache hits, overwriting blocks that may have `ref_count > 1`. Fixed: blocks are only written on miss paths where `ref_count == 1`.

5. **Benchmark labelling**: Previous README claimed the 1.44x–1.56x speedup figure reflected real speech serving performance. It does not; it is a scheduler microbenchmark on random weights. Relabelled accordingly.

6. **`requirements.txt` added** with pinned/range versions so a clean checkout can reproduce the environment.

### What was NOT done (outstanding)
- **Real-speech evaluation (Task 4)**: No WER measurement on LibriSpeech with a pretrained Whisper checkpoint. No p50/p95 latency or TTFT numbers on real audio. This is the most important missing piece.
- **Encoder context-sensitivity experiment (Task 3)**: No quantified comparison of chunk-level vs. full-input sharing on real Whisper encoder outputs.
- **Paging vs. batching ablation (Task 5)**: The decode-cache overhead (contiguous copy vs. direct block read) has not been measured.
- **EOS handling**: The decode loop stops only at `max_new_tokens`. For a real Whisper evaluation, EOS token detection and special prompt tokens (start-of-transcript, language, no-timestamps) are required.

### Open Questions
- Does the encoder-skip give measurable latency reduction on real Whisper with real audio? (Depends on audio repeat rate in the workload.)
- How much of the scheduling speedup survives when the model has real weights and the encoder takes significant time?
- Is paging adding measurable overhead compared to a plain contiguous cache when batch sizes are small?

---

## Repository Structure

```
├── speech_serving/
│   ├── block_manager.py           # Self-contained BlockManager (decoder KV pool)
│   ├── models.py                  # Whisper-class ASR & multi-codebook TTS architectures
│   ├── phase_aware_scheduler.py   # Continuous batching scheduler with lookup-before-encode
│   ├── encoder_block_manager.py   # Paged non-causal encoder memory & ref-counting
│   ├── encoder_attention.py       # Non-causal paged attention kernel
│   ├── dual_cache_manager.py      # Coordinator managing dual physical block pools
│   ├── tts_engine.py              # Streaming multi-codebook TTS engine
│   └── memory_guard.py            # High-water mark tracking & LRU bulk eviction
├── paged_kv_cache/                # Compatibility shim (re-exports BlockManager)
│   ├── __init__.py
│   └── block_manager.py
├── paged_dual_cache_speech_serving.ipynb  # Interactive walkthrough and visual demo
├── benchmark_speech_serving.py    # Comparative benchmarking suite
├── stress_test.py                 # Memory backpressure and high-water mark stress test
├── test_speech_engine.py          # Automated unit test suite (8 tests)
├── requirements.txt               # Pinned environment for reproducibility
├── FINDINGS.md                    # One-page summary of what was fixed and what is open
└── speech_serving_benchmark.png   # Benchmark throughput plot (synthetic)
```

---

## Quickstart

### 1. Environment Setup
```bash
git clone https://github.com/aditya3799/inference-speech-model.git
cd inference-speech-model
pip install -r requirements.txt
```

### 2. Run Automated Verification Suite (CPU is sufficient)
Executes the full test suite (8 tests: TTS streaming, encoder prefix sharing, phase transitions, memory safety, cross-attention padding, and 3 Task-2 encoder-skip tests):
```bash
py -3.10 -m pytest test_speech_engine.py -v
```

### 3. Run Memory Stress Test
Tests admission backpressure, LRU bulk evictions, and zero-leak memory release under heavy pool contention:
```bash
python stress_test.py
```

### 4. Run Scheduler Microbenchmark
**Note: this uses random weights and synthetic audio. Results are labelled accordingly.**
```bash
python benchmark_speech_serving.py --repeats 3 --output_image speech_serving_benchmark.png
```

### 5. Interactive Notebook
An end-to-end interactive notebook is provided in [`paged_dual_cache_speech_serving.ipynb`](paged_dual_cache_speech_serving.ipynb).

---

## License & Feedback

This project is shared under the MIT License for research and exploration. Feedback, questions, and discussions are welcome!
