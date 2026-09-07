# Paged Dual-Cache Speech Serving Engine

**Status: early-stage / work in progress.** This is a prototype, not a production system — sharing it to show direction and get feedback, not as a finished result.

## What this is

An experimental serving engine for encoder-decoder speech models (Whisper-class ASR, streaming codec-based TTS), applying PagedAttention-style memory management to speech inference. Most of the current paged-attention/continuous-batching work in the ecosystem (vLLM, SGLang, etc.) targets text LLMs; this project explores what that same approach looks like for speech models, which have a different shape: a non-causal encoder pass over audio, plus a causal decoder that also cross-attends into the encoder's output.

## Architecture, in brief

- **Dual-cache design**: separate paged block pools for encoder KV (built once per audio input, non-causal) and decoder KV (grows per generated token, causal + cross-attention into the encoder cache).
- **Phase-aware scheduler**: batches encode-phase (compute-bound) and decode-phase (memory-bound) work with awareness of the two-phase structure, rather than assuming a single growing sequence like text-only schedulers do.
- **Encoder-side prefix sharing**: block-level hashing to reuse cached encoder state across repeated/overlapping audio segments.
- **Memory safety**: explicit block release paths, high-water-mark backpressure, and eviction — built to avoid CUDA OOM under sustained concurrent load.

## Current state

- Phases 1–3 (streaming TTS decode loop, encoder paged attention, dual-cache phase-aware scheduling) are implemented and passing unit tests.
- Benchmarked against a naive unbatched baseline across three synthetic streaming-load scenarios (varying arrival density/burstiness). Current results: **~1.0x–1.3x throughput** over naive, averaged across repeated runs per scenario — modest but consistent gains, not yet a dramatic win.
- Memory stress testing (backpressure/eviction under pool pressure) is still being debugged — admission logic works, but forcing genuine multi-request contention in the test harness itself is an open issue.
- This has all been built and tested on a small toy model configuration; behavior at realistic model scale is untested.

## What's next

- Nail down the stress test so eviction/backpressure is actually exercised under real contention, not just verified in the no-pressure case.
- Investigate where the remaining throughput ceiling is coming from — likely a mix of Python-level scheduling overhead and toy-model size limiting how much paging can pay off.
- Extend benchmarking to a more realistic model size and real (not synthetic) audio.

