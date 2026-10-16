# Hybrid Paged Speech Serving Engine

This repository implements a phase-aware Whisper serving engine that batches encoder work, hands immutable encoder cross-attention KV directly to CUDA-graph slots, and retains paged KV management for the decoder's growing self-attention state.

The benchmark medians below are indicative rather than final performance claims because the runs failed the GPU clock-stability gate. Each displayed metric is the independent median of three runs, so a table column does not necessarily represent one physical run. Source: [`Speech-Serving Engine Median Benchmark Results - Single Table - Sectioned.docx`](Speech-Serving%20Engine%20Median%20Benchmark%20Results%20-%20Single%20Table%20-%20Sectioned.docx).

## Architecture

The optimized path is a hybrid cache architecture:

```text
batched Whisper encoder
    -> contiguous cross-attention KV
    -> direct one-time handoff into fixed CUDA-graph slots

autoregressive decoder
    -> continuously batched CUDA-graph decode
    -> dynamically paged self-attention KV
```

`PhaseAwareSpeechScheduler` admits requests, batches compatible encoder work, performs decoder prefill, continuously batches active decoding requests, and releases graph slots and decoder blocks deterministically.

### Encoder-side changes

- `batch_encoder=True` combines multiple queued audio inputs into one encoder forward pass when at least two requests and multiple engine slots are available.
- `graph_direct_encoder_kv=True` keeps the encoder-produced cross-attention K/V contiguous and hands it directly to the graph path.
- The direct path skips encoder paged-block allocation, block writes, and the subsequent gather back into contiguous tensors.
- `cache_cross_kv=True` prepares immutable cross-attention KV once at encoder-to-decoder handoff and reuses it for every generated token.
- Encoder-cache retention is disabled for this benchmark (`retain_encoder_cache=False`), so completed requests release their contiguous encoder state instead of preserving it for reuse.

The encoder network itself is unchanged: the optimization does not introduce quantization, pruning, layer fusion, reduced audio length, or a different Whisper architecture.

### Decoder-side behavior

- Decoder self-attention KV remains paged because it grows token by token and benefits from block-level allocation and release.
- CUDA-graph decoding uses fixed slots and captured buckets to reduce Python and kernel-launch overhead.
- Graph completion state is read back every four steps (`graph_readback_k=4`) rather than synchronizing after every token.
- Supported graph-buffer transfers are non-blocking, and released graph slots are overwritten on reuse instead of being unconditionally zeroed.
- The eager decoder remains available as the compatibility baseline; graph-specific buffers are created only when graph decoding is enabled.

### Key invariants

- Encoder positions, rather than mel frames, determine encoder sequence lengths and any non-direct-path block admission.
- Direct encoder-KV handoff requires graph decoding and cannot be combined with retained encoder-cache reuse.
- Decoder blocks and graph slots are released on completion and failure paths.
- Variable-length batches carry explicit self-attention and cross-attention masks.
- Whisper prompt tokens, EOS handling, token suppression, and cached cross-attention projections preserve generation quality.

## Median benchmark results

### Burst traffic

| Metric | Sequential | Static batching | Static graph | Hybrid before | Hybrid cross-KV | Hybrid after | Hybrid encoder optimized |
|---|---:|---:|---:|---:|---:|---:|---:|
| Capacity (req/s) | 4.300 | 8.854 | 17.519 | 4.083 | 4.750 | 11.008 | **19.640** |
| Audio throughput (x) | 37.914 | 78.067 | 154.461 | 35.997 | 41.882 | 97.051 | **173.158** |
| Wall time (s) | 23.255 | 11.294 | 5.708 | 24.493 | 21.052 | 9.085 | **5.092** |
| TTFT p50 (s) | 11.539 | 5.589 | 3.296 | 11.963 | 10.457 | 4.547 | **2.590** |
| TTFT p95 (s) | 22.239 | 10.678 | 5.446 | 22.950 | 19.797 | 8.591 | **4.790** |
| Latency p50 (s) | 11.802 | 5.974 | 3.423 | 12.809 | 11.191 | 4.792 | **2.728** |
| Latency p95 (s) | 22.288 | 10.884 | 5.521 | 23.507 | 20.258 | 8.846 | **4.913** |
| Peak VRAM (MB) | 286.96 | 385.49 | 592.08 | 1842.81 | 1842.81 | 1870.31 | **732.89** |
| WER (%) | 5.282 | 5.282 | 5.282 | 5.282 | 5.282 | 5.282 | 5.282 |
| Exact match (%) | 100 | 100 | 100 | 99 | 99 | 99 | 99 |

For burst traffic, the encoder-optimized hybrid reached 19.640 req/s versus 17.519 req/s for static graph, while reducing median wall time from 5.708 s to 5.092 s and p50 TTFT from 3.296 s to 2.590 s. Its 732.89 MB peak VRAM was substantially below the 1870.31 MB hybrid-after result because the direct path removed the redundant paged encoder-KV representation and gather, although it remained above static graph's 592.08 MB.

### Poisson high traffic

Arrival rate: `lambda = 4.0` requests per second.

| Metric | Sequential | Static batching | Static graph | Hybrid before | Hybrid cross-KV | Hybrid after | Hybrid encoder optimized |
|---|---:|---:|---:|---:|---:|---:|---:|
| Capacity (req/s) | 4.087 | 4.294 | 4.364 | 3.961 | 4.205 | 4.347 | **4.369** |
| Audio throughput (x) | 36.038 | 37.857 | 38.477 | 34.927 | 37.077 | 38.327 | **38.525** |
| Wall time (s) | 24.465 | 23.290 | 22.914 | 25.243 | 23.780 | 23.004 | **22.886** |
| TTFT p50 (s) | 1.092 | 0.163 | 0.037 | 1.136 | 0.416 | 0.078 | **0.032** |
| TTFT p95 (s) | 1.979 | 0.429 | 0.157 | 2.122 | 1.261 | 0.164 | **0.052** |
| Latency p50 (s) | 1.309 | 0.510 | 0.131 | 2.009 | 1.071 | 0.221 | **0.124** |
| Latency p95 (s) | 2.191 | 0.795 | 0.244 | 3.177 | 2.219 | 0.426 | **0.226** |
| Peak VRAM (MB) | 376.34 | 496.31 | **586.33** | 1933.94 | 1932.66 | 1958.69 | 796.44 |
| WER (%) | 5.282 | 5.282 | 5.282 | 5.282 | 5.282 | 5.282 | 5.282 |
| Exact match (%) | 100 | 99 | 99 | 99 | 99 | 99 | 99 |

At this arrival rate, throughput is capped near 4.36 req/s, so capacity is effectively tied across the graph configurations. The clearer benefit is latency: the encoder-optimized hybrid recorded 0.052 s p95 TTFT and 0.226 s p95 end-to-end latency, compared with 0.157 s and 0.244 s for static graph.

## Correctness and evidence quality

- WER remained 5.282% for every configuration in both traffic profiles.
- Exact token match was 99% for the optimized hybrid, while the aggregate WER matched the other systems.
- Before external publication, record the exact Whisper checkpoint and speech-dataset subset associated with the reported WER.
- The benchmark failed its GPU clock-stability gate, so differences should be confirmed in a clean-clock session before being treated as portable performance conclusions.

## Tested environment

| Component | Tested version |
|---|---|
| Python | 3.10.11 |
| PyTorch | 2.6.0+cu124 |
| Transformers | 4.57.0 |
| Triton | 3.8.0 |
| NVIDIA driver | 551.76 |

## Repository structure

- `speech_serving/`: model wrapper, cache managers, phase-aware scheduler, graph buffers, and timing support.
- `paged_kv_cache/`: paged decoder-cache implementation.
- `scripts/`: benchmark, parity, profiling, report, and notebook generators.
- `data/`: benchmark JSON and generated reports.
- `test_speech_engine.py`: unit and integration coverage for eager, cache, pretrained, and graph behavior.
- `real_speech_serving_v2.ipynb`: streamlined real-speech comparison notebook.
- `hybrid_decoder_before_after.ipynb`: hybrid decoder comparison notebook.

## Quickstart and reproduction

Create a Python environment with the CUDA runtime expected by the PyTorch wheel, then install the pinned dependencies:

```powershell
python -m pip install -r requirements.txt
```

Run the automated suite:

```powershell
python -m pytest -q
```

Run the real-speech comparison, including the encoder-handoff optimization:

```powershell
python scripts/run_hybrid_before_after_burst.py
```

Use `--encoder-ablation-only` to compare static graph, the graph-engine baseline, and the encoder-optimized hybrid directly. Treat a run as suitable for conclusions only when its parity and telemetry gates pass.

## Limitations

- Current medians come from three runs per metric on one laptop GPU and one real-speech subset.
- The session failed the clock-stability gate, so small differences may reflect hardware-state variation.
- Poisson-high throughput is arrival-rate limited and should be interpreted primarily through latency.
- CUDA-graph decode uses delayed token readback and can execute extra slot steps after a logical stop condition.
- Static batching and the hybrid scheduler have different orchestration and memory-management responsibilities.
