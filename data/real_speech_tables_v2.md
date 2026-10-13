# Real-Speech Serving Evaluation (v2 - Cached Cross-KV Ablation)

- **Model**: openai/whisper-base (FP16)
- **Workload**: LibriSpeech test-clean 100 utterances, 3 repeats per system, max_batch_size=4
- **Evaluated Systems**:
  1. `sequential` (unbatched baseline)
  2. `static_batching` (fixed batch size 4, padded decode)
  3. `engine_flag_off` (continuous batching + dual paged cache, re-gather cross-KV every step)
  4. `engine_flag_on` (continuous batching + one-time cross-KV gather at handoff)

## 1. Traffic Profile Serving Performance (4 Rows per Profile)
| Traffic Profile | System | Throughput (audio-sec / wall-sec) | Latency p50 (s) | Latency p95 (s) | TTFT p50 (s) | TTFT p95 (s) | Peak VRAM Alloc / Res (MB) | Actually Used Pool (MB) | Corpus WER (95% CI) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Burst** ($t=0$) | Sequential | 24.51x $\pm$ 0.45x | 18.105s | 34.498s | 17.718s | 34.417s | 189.5 / 214.0 | not measured | 5.28% [4.00%, 6.81%] |
| | Static Batching | 39.65x $\pm$ 5.80x | 12.395s | 21.977s | 11.512s | 21.583s | 312.4 / 388.0 | not measured | 5.28% [4.00%, 6.81%] |
| | Engine (Flag OFF, re-gather) | 20.56x $\pm$ 0.82x | 22.900s | 41.289s | 21.490s | 40.314s | 1761.6 / 1830.0 | 72.94 MB | 5.28% [3.95%, 6.89%] |
| | Engine (Flag ON, cached cross-KV) | **24.51x** $\pm$ 0.59x | **18.333s** | **34.655s** | 17.094s | 33.754s | 1761.6 / 1830.0 | 143.25 MB | 5.28% [3.95%, 6.89%] |
| **Poisson Low** ($\lambda=1.0$) | Sequential | 9.60x $\pm$ 0.02x | 0.580s | 2.950s | 0.141s | 2.487s | 189.5 / 214.0 | not measured | 5.28% [4.00%, 6.81%] |
| | Static Batching | 9.60x $\pm$ 0.01x | 0.591s | 1.227s | 0.120s | 0.558s | 280.9 / 317.3 | not measured | 5.28% [4.00%, 6.81%] |
| | Engine (Flag OFF, re-gather) | 9.56x $\pm$ 0.01x | 0.981s | 2.151s | 0.155s | 0.440s | 1760.0 / 1806.0 | 72.19 MB | 5.28% [3.96%, 6.69%] |
| | Engine (Flag ON, cached cross-KV) | **9.59x** $\pm$ 0.01x | **0.855s** | **1.744s** | 0.155s | 0.339s | 1763.4 / 1806.0 | 142.50 MB | 5.28% [3.99%, 6.76%] |
| **Poisson High** ($\lambda=4.0$) | Sequential | 22.21x $\pm$ 0.41x | 8.910s | 16.325s | 8.620s | 16.001s | 189.5 / 214.0 | not measured | 5.28% [4.00%, 6.81%] |
| | Static Batching | 36.27x $\pm$ 0.46x | 1.346s | 2.101s | 0.667s | 1.280s | 312.9 / 390.0 | not measured | 5.28% [4.00%, 6.81%] |
| | Engine (Flag OFF, re-gather) | 20.03x $\pm$ 0.39x | 12.193s | 20.616s | 10.820s | 19.257s | 1761.9 / 1830.7 | 72.81 MB | 5.28% [4.08%, 6.81%] |
| | Engine (Flag ON, cached cross-KV) | **23.55x** $\pm$ 0.12x | **8.006s** | **14.104s** | 6.951s | 12.892s | 1762.0 / 1830.7 | 143.25 MB | 5.28% [4.04%, 6.96%] |

*Footnote: Latency is defined as end-to-end latency per request, from arrival to last token. Actually used pool memory tracks active pool blocks in use x block bytes, plus contiguous cross-KV copies for Flag ON.*

## 2. Telemetry and Runtime Concurrency
| Profile | Engine Variant | Avg In-Flight Reqs | Max In-Flight Reqs | Queue Wait Mean (s) | Queue Wait p50 (s) | Queue Wait p95 (s) | Actually Used Peak (MB) | Rejections |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Burst ($t=0$) | Flag OFF (re-gather) | 3.80 | 4 | 3.069s | 3.179s | 4.156s | 72.94 MB | 0 |
| Burst ($t=0$) | Flag ON (cached cross-KV) | 3.80 | 4 | 2.549s | 2.642s | 3.339s | 143.25 MB | 0 |
| Poisson Low ($\lambda=1.0$) | Flag OFF (re-gather) | 1.57 | 4 | 0.028s | 0.003s | 0.188s | 72.19 MB | 0 |
| Poisson Low ($\lambda=1.0$) | Flag ON (cached cross-KV) | 1.48 | 4 | 0.018s | 0.003s | 0.149s | 142.50 MB | 0 |
| Poisson High ($\lambda=4.0$) | Flag OFF (re-gather) | 3.65 | 4 | 2.923s | 3.161s | 3.944s | 72.81 MB | 0 |
| Poisson High ($\lambda=4.0$) | Flag ON (cached cross-KV) | 3.64 | 4 | 2.401s | 2.646s | 3.300s | 143.25 MB | 0 |

## 3. Drift Analysis (Row 3 Reproducibility)
### Baseline Drift Verification (Engine Flag OFF vs Previous Run)

| Profile | Metric | Previous (v1 Engine) | Current (v2 Flag OFF) | Delta ($\Delta$) | Combined Std ($\sigma_{comb}$) | Drift Status |
|:---|:---|:---:|:---:|:---:|:---:|:---|
| burst | Throughput (audio-s/wall-s) | 38.533x $\pm$ 0.314 | 20.558x $\pm$ 0.821 | -17.975x | 0.879x | **DRIFT EXCEEDS $\sigma$** (|$\Delta$|=17.975 > 0.879) |
| burst | Latency p50 (s) | 12.024s $\pm$ 0.134 | 22.900s $\pm$ 1.284 | +10.877s | 1.291s | **DRIFT EXCEEDS $\sigma$** (|$\Delta$|=10.877 > 1.291) |
| burst | Latency p95 (s) | 22.013s $\pm$ 0.175 | 41.289s $\pm$ 1.653 | +19.277s | 1.662s | **DRIFT EXCEEDS $\sigma$** (|$\Delta$|=19.277 > 1.662) |
| burst | Peak VRAM Alloc (MB) | 1761.560MB $\pm$ 0.000 | 1761.560MB $\pm$ 0.000 | +0.000MB | 0.000MB | Reproduced within noise (|$\Delta$|=0.000 $\le$ 0.000) |
| poisson_low | Throughput (audio-s/wall-s) | 9.591x $\pm$ 0.013 | 9.563x $\pm$ 0.014 | -0.028x | 0.019x | **DRIFT EXCEEDS $\sigma$** (|$\Delta$|=0.028 > 0.019) |
| poisson_low | Latency p50 (s) | 0.920s $\pm$ 0.091 | 0.981s $\pm$ 0.029 | +0.061s | 0.096s | Reproduced within noise (|$\Delta$|=0.061 $\le$ 0.096) |
| poisson_low | Latency p95 (s) | 2.296s $\pm$ 0.383 | 2.151s $\pm$ 0.022 | -0.145s | 0.383s | Reproduced within noise (|$\Delta$|=0.145 $\le$ 0.383) |
| poisson_low | Peak VRAM Alloc (MB) | 1762.191MB $\pm$ 1.417 | 1759.951MB $\pm$ 0.400 | -2.240MB | 1.472MB | **DRIFT EXCEEDS $\sigma$** (|$\Delta$|=2.240 > 1.472) |
| poisson_high | Throughput (audio-s/wall-s) | 25.176x $\pm$ 0.077 | 20.026x $\pm$ 0.388 | -5.150x | 0.396x | **DRIFT EXCEEDS $\sigma$** (|$\Delta$|=5.150 > 0.396) |
| poisson_high | Latency p50 (s) | 7.161s $\pm$ 0.391 | 12.193s $\pm$ 0.606 | +5.032s | 0.721s | **DRIFT EXCEEDS $\sigma$** (|$\Delta$|=5.032 > 0.721) |
| poisson_high | Latency p95 (s) | 11.914s $\pm$ 0.273 | 20.616s $\pm$ 0.818 | +8.703s | 0.862s | **DRIFT EXCEEDS $\sigma$** (|$\Delta$|=8.703 > 0.862) |
| poisson_high | Peak VRAM Alloc (MB) | 1762.388MB $\pm$ 0.104 | 1761.867MB $\pm$ 0.362 | -0.521MB | 0.376MB | **DRIFT EXCEEDS $\sigma$** (|$\Delta$|=0.521 > 0.376) |

> **Verdict:** 9 metric(s) showed drift larger than run-to-run std.

## 4. Token Parity Verification
### Token Parity vs Hugging Face Reference (All Systems)

| Traffic Profile | System | HF Match Rate (%) | Total Requests | Differing Utterances |
|:---|:---|:---:|:---:|:---|
| burst | sequential | 100.0% | 100 | None (100% exact match) |
| burst | static_batching | 100.0% | 100 | None (100% exact match) |
| burst | engine_flag_off | 99.0% | 100 | 2830-3980-0051 |
| burst | engine_flag_on | 99.0% | 100 | 2830-3980-0051 |
| poisson_low | sequential | 100.0% | 100 | None (100% exact match) |
| poisson_low | static_batching | 100.0% | 100 | None (100% exact match) |
| poisson_low | engine_flag_off | 100.0% | 100 | None (100% exact match) |
| poisson_low | engine_flag_on | 100.0% | 100 | None (100% exact match) |
| poisson_high | sequential | 100.0% | 100 | None (100% exact match) |
| poisson_high | static_batching | 100.0% | 100 | None (100% exact match) |
| poisson_high | engine_flag_off | 99.0% | 100 | 2830-3980-0051 |
| poisson_high | engine_flag_on | 99.0% | 100 | 2830-3980-0051 |
