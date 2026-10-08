# FINDINGS.md — Paged Dual-Cache Speech Serving Engine

**Date:** October 2026  
**Scope:** Tasks 1 and 2 of the reviewer-requested evaluation plan.

---

## What Was Fixed

### 1. Missing `paged_kv_cache.BlockManager` dependency (Task 1)
**Problem:** The scheduler, tests, and benchmark all imported `from paged_kv_cache.block_manager import BlockManager`, but no such package existed in the repository. A clean checkout could not run any code.

**Fix:** `speech_serving/block_manager.py` now contains a complete, self-contained re-implementation of the `BlockManager` interface:
- Free-list allocator (O(1) `allocate_block()` / `free_blocks()`)
- Shadow `set` for leak detection (`allocated_block_count`, `num_free_blocks`)
- `get_stats()` returning `num_blocks`, `allocated_blocks`, `utilization_pct`
- `check_no_leaks()` helper for test assertions

`paged_kv_cache/` is a local compatibility shim that re-exports it, so all existing `from paged_kv_cache.block_manager import BlockManager` imports continue to work without any code changes.

The fragile `sys.path` hack in `speech_serving/__init__.py` was removed.

**Verification:** `py -3.10 -m pytest test_speech_engine.py -v` → 8/8 PASSED on CPU (no GPU required).

---

### 2. Encoder ran before cache lookup (Task 2)
**Problem (reviewer b):** In the original `_process_encode_phase`, the order was:
```
1. model.encode(mel) ← encoder runs
2. hash(mel chunks)
3. allocate_encoder_blocks(hashes)  ← cache lookup happens after
4. store_encoder_kv(blocks, kv)     ← writes even on a hit
```
A cache hit therefore saved no encoder compute. The lookup happened after the most expensive operation.

**Fix:** The method now follows lookup-before-encode order:
```
1. compute_full_input_hash(mel)   ← O(mel_len) hash, no GPU
2. if hash in cache: SKIP encoder, bump ref_counts, reuse cached KV
3. else: model.encode(mel), write fresh blocks, store in cache
4. decoder prefill (always runs)
```

**Why full-input hash, not chunk hash?** The encoder is **bidirectional**. Every encoder position attends to every other position in the full sequence. The same mel chunk at position `i` produces different hidden states depending on what appears at positions `j ≠ i`. Chunk-level sharing is only safe if encoder outputs are context-invariant for the chunk, which is not guaranteed for a bidirectional architecture. Full-input equality guarantees identical encoder outputs. (Chunk-level context-sensitivity is left for Task 3 quantification.)

**Shared-block safety:** On a hit path, `store_encoder_kv` is never called. Blocks are only written on miss paths where `ref_count == 1`. An assertion guards against accidental writes into shared blocks.

**Block over-reservation fixed:** `needed_enc_blocks` was computed from `mel_len`, but the encoder Conv2d has stride 2, so `enc_seq_len ≈ mel_len // 2`. This caused 2× more blocks to be reserved than actually needed. Fixed to use `(mel_len + 1) // 2` (matching the admission check in `submit()`), then the actual `enc_seq_len` from the encoder output on miss paths.

**Telemetry added:** `encoder_calls`, `encoder_calls_skipped`, `encode_time_saved_s`, accessible via `scheduler.get_encoder_telemetry()`.

---

### 3. Benchmark labelling corrected (Task 6 — relabelling)
**Problem:** The README presented an unverified 1.44x–1.56x throughput figure as evidence of "real speech serving performance". Investigation revealed those numbers were not from this codebase's execution, but were copied from a separate text-LLM KV-cache project.

**Fix:** Removed all references to 1.44x and 1.56x. Audited the actual execution log in this repository (`output.txt` and `microbenchmark_results.json`), which ran 3 repeats at synthetic Whisper-Base scale (6 enc, 6 dec, hidden_dim 512, 8 heads) on an RTX 3050 GPU (FP16). Results are explicitly labelled "Scheduler Microbenchmark (random Whisper-shaped weights, synthetic mel inputs)".

---

## Headline Numbers (Scheduler Microbenchmark from output.txt / microbenchmark_results.json)

| Scenario | Naive (mean) | Paged Engine (mean) | Speedup (min / mean / max) |
|:---|:---:|:---:|:---:|
| Uniform Dense (6s window, high concurrency) | 138.23 tok/s | 154.85 tok/s | **1.05x / 1.13x / 1.27x** |
| Bursty (clustered arrivals) | 65.84 tok/s | 70.42 tok/s | **1.05x / 1.07x / 1.10x** |
| Uniform Moderate (15s window, sparse) | 80.62 tok/s | 82.92 tok/s | **0.98x / 1.03x / 1.12x** |

> **Context:** These are throughput numbers on a synthetic model with random weights and synthetic mels, measuring only the scheduler and continuous-batching queue machinery. They are not WER, latency, or real-audio results.

---

## Results That Came Out Worse (or Neutral) Than Before

- **Sparse traffic is truly near parity (~1.03x):** When requests arrive widely spaced, effective concurrency stays at ~1 and the paging overhead (block allocation, block gather, ref-count bookkeeping) adds cost with negligible benefit.
- **The actual microbenchmark speedup is 1.13x, not 1.44x:** The previously cited 1.44x (and peak 1.56x) originated from a text-LLM KV-cache serving project and did not reflect this speech repo's execution.
- **The full-input hash adds CPU overhead** proportional to `mel_len` (flattening + MD5). For very long audio this is on the critical path.
- **`encode_time_saved_s` telemetry is not yet populated** with real encoder timing. The counter increments on every skip, but the time estimate is `0.0` until Task 4's real-speech evaluation measures actual encoder wall-clock time.

---

## Test Status

```
9/9 tests PASSED  (py -3.10 -m pytest test_speech_engine.py -v)

Tests:
  test_dual_cache_phase_aware_scheduler                     PASS
  test_empty_pool_admission                                 PASS
  test_encoder_hit_produces_identical_output                PASS (tokens + torch.allclose KV)
  test_encoder_not_called_on_hit                            PASS (call count == 1)
  test_encoder_paged_attention_and_prefix_sharing           PASS
  test_generation_stops_on_eos_and_uses_whisper_prompt       PASS (EOS termination + prompts)
  test_streaming_tts_engine                                 PASS
  test_variable_length_batch_cross_attention_correctness     PASS
  test_zero_blocks_leaked_after_hit_path                   PASS (no manual flush)
```

---

## Open Questions / Outstanding Work

| Task | Status | Notes |
|:---|:---|:---|
| Task 1: Reproducibility | ✅ Done | BlockManager shim, requirements.txt, sys.path removed, 8 tests pass |
| Task 2: Lookup-before-encode | ✅ Done | Full-input hash, encoder skip, shared-block safety, 3 new tests |
| Task 3: Encoder context-sensitivity | ❌ Not done | Needs real Whisper weights and real audio |
| Task 4: Real-speech evaluation | ❌ Not done | Needs LibriSpeech, pretrained Whisper, WER/latency/VRAM table |
| Task 5: Paging vs. batching ablation | ❌ Not done | Needs decode-cache mode comparison |
| Task 6: Reporting | ⚠️ Partial | README and FINDINGS.md updated; notebook results section not done (no real numbers yet) |

**The most important missing piece is Task 4 (real-speech evaluation).** Without WER, peak VRAM, and p50/p95 latency from a pretrained Whisper checkpoint on LibriSpeech, the claimed speedups cannot be interpreted as evidence of real-world speech serving gains.

---

## How to Reproduce

```bash
git clone https://github.com/aditya3799/inference-speech-model.git
cd inference-speech-model
pip install -r requirements.txt          # torch>=2.3.1, transformers>=4.41.2
py -3.10 -m pytest test_speech_engine.py -v   # 8/8 PASS (CPU)
python benchmark_speech_serving.py --repeats 3  # synthetic microbenchmark
```

All results in this document come from executed code. No numbers were hand-edited.
