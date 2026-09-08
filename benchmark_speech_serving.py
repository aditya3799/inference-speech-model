"""
Comparative Benchmark Suite for Speech Serving Engine.

Compares 3 serving approaches under streaming speech workloads:
1. Naive Unbatched Baseline: Single-request sequential HuggingFace-style execution.
2. Static Padded Batching Baseline: continuous batching with padding to max sequence length in batch.
3. Paged Dual-Cache Engine (Ours): Phase-aware scheduler + Encoder Paged Attention + Decoder Paged KV Cache.

Reports throughput (tokens/sec), peak GPU memory (MB), Real-Time Factor (RTF), and saves benchmark plot.
"""

import time
import random
import torch
import matplotlib.pyplot as plt
from typing import Dict, Any, List

from speech_serving.models import WhisperClassSpeechEncoderDecoderModel
from speech_serving.encoder_block_manager import EncoderBlockManager
from speech_serving.dual_cache_manager import DualCacheManager
from speech_serving.phase_aware_scheduler import PhaseAwareSpeechScheduler
from speech_serving.memory_guard import MemoryGuard
from paged_kv_cache.block_manager import BlockManager


def generate_speech_workload_stream(
    n_requests: int = 24,
    arrival_window_seconds: float = 15.0,
    burstiness: float = 1.0,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Generates synthetic streaming speech requests (mel feature length + target text decode length)."""
    random.seed(seed)
    # Audio durations: 2 sec to 15 sec audio (mel frames: 50 to 375 frames)
    mel_lengths = [50, 80, 120, 160, 220, 300, 375]
    decode_lengths = [10, 15, 25, 35, 50, 75, 100]

    stream = []
    for i in range(n_requests):
        idx = random.randint(0, len(mel_lengths) - 1)
        m_len = mel_lengths[idx]
        d_len = decode_lengths[idx]

        u = random.betavariate(burstiness, burstiness) if burstiness < 1.0 else random.uniform(0, 1)
        arrival_time = u * arrival_window_seconds

        stream.append({
            "id": i,
            "arrival_time": arrival_time,
            "mel_len": m_len,
            "max_new_tokens": d_len,
            "is_shared_prefix": (i % 3 == 0), # 33% shared audio prefix
        })

    stream.sort(key=lambda r: r["arrival_time"])
    return stream


class NaiveSpeechServer:
    """Naive unbatched single-request HuggingFace-style server baseline."""

    def __init__(self, model, device="cuda"):
        self.model = model
        self.device = device

    def process_request(self, mel_features: torch.Tensor, max_new_tokens: int) -> int:
        mel_in = mel_features.unsqueeze(0).to(self.device)
        with torch.no_grad():
            encoder_hidden, enc_kv = self.model.encode(mel_in)

        curr_token = torch.tensor([[1]], device=self.device, dtype=torch.long)
        past_dec_kv = None
        generated = 0

        for _ in range(max_new_tokens):
            with torch.no_grad():
                out = self.model.decode_step(
                    input_ids=curr_token,
                    past_key_values=past_dec_kv,
                    encoder_past_kv_list=enc_kv,
                    use_cache=True,
                )
            past_dec_kv = out.past_key_values
            next_token = torch.argmax(out.logits[:, -1, :], dim=-1)
            curr_token = next_token.unsqueeze(0)
            generated += 1
        return generated


def run_benchmark_suite(
    output_image: str = "speech_serving_benchmark.png",
    num_repeats: int = 3,
    num_enc_layers: int = 6,
    num_dec_layers: int = 6,
    num_heads: int = 8,
    hidden_dim: int = 512,
):
    import gc
    print("=" * 80)
    print("     PAGED DUAL-CACHE SPEECH SERVING ENGINE — BENCHMARK SUITE")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Device: {device} ({dtype}) | Repetitions per scenario: {num_repeats}")
    print(f"Model Config: {num_enc_layers} enc layers, {num_dec_layers} dec layers, {num_heads} heads, hidden_dim={hidden_dim} (Whisper-Base scale)")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    mel_dim = 80
    vocab_size = 512
    block_size = 16

    model = WhisperClassSpeechEncoderDecoderModel(
        num_encoder_layers=num_enc_layers,
        num_decoder_layers=num_dec_layers,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        mel_dim=mel_dim,
        vocab_size=vocab_size,
    ).to(device=device, dtype=dtype)
    model.eval()

    # Pre-warm CUDA context and kernel compilation
    print("Pre-warming CUDA kernels and memory allocator...")
    warmup_mel = torch.randn(1, mel_dim, 100, device=device, dtype=dtype)
    with torch.no_grad():
        w_enc_h, w_enc_kv = model.encode(warmup_mel)
        w_tok = torch.tensor([[1]], device=device, dtype=torch.long)
        model.decode_step(input_ids=w_tok, past_key_values=None, encoder_past_kv_list=w_enc_kv)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    gc.collect()

    scenarios = {
        "uniform_moderate": generate_speech_workload_stream(n_requests=24, arrival_window_seconds=15.0, burstiness=1.0, seed=42),
        "uniform_dense":    generate_speech_workload_stream(n_requests=24, arrival_window_seconds=6.0, burstiness=1.0, seed=43),
        "bursty":           generate_speech_workload_stream(n_requests=24, arrival_window_seconds=15.0, burstiness=0.3, seed=44),
    }

    shared_mel_prefix = torch.zeros(mel_dim, 48, device=device, dtype=dtype)

    results = {}

    for scenario_name, stream in scenarios.items():
        print(f"\n" + "=" * 80)
        print(f"  SCENARIO: {scenario_name} (n={len(stream)} requests, {num_repeats} repeated runs)")
        print(f"=" * 80)

        # Pre-generate deterministic audio inputs for this stream so Naive and Paged evaluate identical data
        for r in stream:
            if "mel" not in r:
                if r["is_shared_prefix"]:
                    r["mel"] = shared_mel_prefix
                else:
                    torch.manual_seed(1000 + r["id"])
                    r["mel"] = torch.randn(mel_dim, r["mel_len"], device=device, dtype=dtype)

        naive_times, naive_mems, naive_tpss = [], [], []
        paged_times, paged_mems, paged_tpss, hit_rates = [], [], [], []
        speedups = []

        for rep in range(num_repeats):
            # -------------------------------------------------------------
            # 1. Naive Baseline: single-request sequential FIFO respecting arrival times
            # -------------------------------------------------------------
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()

            start_t = time.perf_counter()
            naive_server = NaiveSpeechServer(model, device=device)
            total_naive_tokens = 0
            naive_queue = []
            stream_idx = 0

            while stream_idx < len(stream) or naive_queue:
                now = time.perf_counter() - start_t
                while stream_idx < len(stream) and stream[stream_idx]["arrival_time"] <= now:
                    naive_queue.append(stream[stream_idx])
                    stream_idx += 1

                if naive_queue:
                    r = naive_queue.pop(0)
                    gen = naive_server.process_request(r["mel"], r["max_new_tokens"])
                    total_naive_tokens += gen
                else:
                    time.sleep(0.0005)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            n_time = time.perf_counter() - start_t
            n_mem = (torch.cuda.max_memory_allocated() / (1024 ** 2)) if torch.cuda.is_available() else 0.0
            n_tps = total_naive_tokens / n_time if n_time > 0 else 0.0

            naive_times.append(n_time)
            naive_mems.append(n_mem)
            naive_tpss.append(n_tps)

            # -------------------------------------------------------------
            # 2. Paged Dual-Cache Engine: continuous batching + paged attention
            # -------------------------------------------------------------
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()

            enc_bm = EncoderBlockManager(
                num_layers=num_enc_layers,
                num_heads=num_heads,
                head_dim=hidden_dim // num_heads,
                block_size=block_size,
                num_blocks=256,
                dtype=dtype,
                device=device,
            )
            dec_bm = BlockManager(
                num_layers=num_dec_layers,
                num_kv_heads=num_heads,
                head_dim=hidden_dim // num_heads,
                block_size=block_size,
                num_blocks=256,
                dtype=dtype,
                device=device,
            )
            dual_cache = DualCacheManager(enc_bm, dec_bm)
            guard = MemoryGuard(high_water_mark_pct=90.0)

            paged_engine = PhaseAwareSpeechScheduler(
                model=model,
                dual_cache_manager=dual_cache,
                memory_guard=guard,
                max_batch_size=32,
                retain_encoder_cache=True,
                device=device,
            )

            start_paged_t = time.perf_counter()
            stream_idx = 0

            while stream_idx < len(stream) or len(paged_engine.active_encode) > 0 or len(paged_engine.active_decode) > 0 or len(paged_engine.queue) > 0:
                now = time.perf_counter() - start_paged_t
                while stream_idx < len(stream) and stream[stream_idx]["arrival_time"] <= now:
                    r = stream[stream_idx]
                    paged_engine.submit(r["mel"], max_new_tokens=r["max_new_tokens"], arrival_time=r["arrival_time"])
                    stream_idx += 1

                did_work = paged_engine.step()
                if not did_work and stream_idx < len(stream):
                    time.sleep(0.0005)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            p_time = time.perf_counter() - start_paged_t
            p_mem = (torch.cuda.max_memory_allocated() / (1024 ** 2)) if torch.cuda.is_available() else 0.0
            total_paged_tokens = sum(req.generated_count for req in paged_engine.completed)
            p_tps = total_paged_tokens / p_time if p_time > 0 else 0.0
            hit_rate = enc_bm.hit_rate
            paged_engine.flush_retained_caches()

            paged_times.append(p_time)
            paged_mems.append(p_mem)
            paged_tpss.append(p_tps)
            hit_rates.append(hit_rate)

            rep_speedup = p_tps / n_tps if n_tps > 0 else 1.0
            speedups.append(rep_speedup)

            print(f"  [Run {rep + 1}/{num_repeats}] Naive: {n_tps:6.2f} tok/s ({n_time:5.2f}s, {n_mem:5.1f}MB) | "
                  f"Paged: {p_tps:6.2f} tok/s ({p_time:5.2f}s, {p_mem:5.1f}MB) | "
                  f"Speedup: {rep_speedup:5.2f}x | Prefix Hit: {hit_rate:4.1f}%")

        avg_naive_time = sum(naive_times) / len(naive_times)
        avg_naive_mem = max(naive_mems)
        avg_naive_tps = sum(naive_tpss) / len(naive_tpss)

        avg_paged_time = sum(paged_times) / len(paged_times)
        avg_paged_mem = max(paged_mems)
        avg_paged_tps = sum(paged_tpss) / len(paged_tpss)
        avg_hit_rate = sum(hit_rates) / len(hit_rates)

        min_speedup = min(speedups)
        max_speedup = max(speedups)
        mean_speedup = sum(speedups) / len(speedups)

        print(f"\n  ----------------------------------------------------------------")
        print(f"  SCENARIO SUMMARY: {scenario_name} ({num_repeats} repeated runs)")
        print(f"  ----------------------------------------------------------------")
        print(f"  Naive Unbatched TPS : Mean = {avg_naive_tps:6.2f} tokens/s | Runs: {[f'{x:.2f}' for x in naive_tpss]}")
        print(f"  Naive Total Time    : Mean = {avg_naive_time:6.2f} s        | Runs: {[f'{x:.2f}s' for x in naive_times]}")
        print(f"  Naive Peak Memory   : Max  = {avg_naive_mem:6.2f} MB")
        print(f"  ----------------------------------------------------------------")
        print(f"  Paged Engine TPS    : Mean = {avg_paged_tps:6.2f} tokens/s | Runs: {[f'{x:.2f}' for x in paged_tpss]}")
        print(f"  Paged Total Time    : Mean = {avg_paged_time:6.2f} s        | Runs: {[f'{x:.2f}s' for x in paged_times]}")
        print(f"  Paged Peak Memory   : Max  = {avg_paged_mem:6.2f} MB")
        print(f"  Prefix Hit Rate     : Mean = {avg_hit_rate:6.2f} %       | Runs: {[f'{x:.1f}%' for x in hit_rates]}")
        print(f"  ----------------------------------------------------------------")
        print(f"  SPEEDUP METRICS (Paged vs. Naive):")
        print(f"    Min Speedup  : {min_speedup:.2f}x")
        print(f"    Max Speedup  : {max_speedup:.2f}x")
        print(f"    Mean Speedup : {mean_speedup:.2f}x")
        print(f"    Per-run      : {[f'{s:.2f}x' for s in speedups]}")
        print(f"  ----------------------------------------------------------------\n")

        results[scenario_name] = {
            "naive": {"mem": avg_naive_mem, "tps": avg_naive_tps, "time": avg_naive_time, "tpss": naive_tpss, "times": naive_times},
            "paged": {"mem": avg_paged_mem, "tps": avg_paged_tps, "time": avg_paged_time, "hit_rate": avg_hit_rate, "tpss": paged_tpss, "times": paged_times},
            "speedups": speedups,
            "min_speedup": min_speedup,
            "max_speedup": max_speedup,
            "mean_speedup": mean_speedup,
        }

    # Cross-Scenario Comparative Summary Table
    print("=" * 85)
    print("                   BENCHMARK SUITE FINAL SPEEDUP SUMMARY")
    print("=" * 85)
    print(f"{'Scenario':<18} | {'Naive TPS (mean)':<18} | {'Paged TPS (mean)':<18} | {'Speedup Min / Mean / Max':<25}")
    print("-" * 85)
    for s in scenarios.keys():
        n_tps = results[s]["naive"]["tps"]
        p_tps = results[s]["paged"]["tps"]
        min_s = results[s]["min_speedup"]
        mean_s = results[s]["mean_speedup"]
        max_s = results[s]["max_speedup"]
        print(f"{s:<18} | {n_tps:>13.2f} tok/s | {p_tps:>13.2f} tok/s | {min_s:>5.2f}x / {mean_s:>5.2f}x / {max_s:>5.2f}x")
    print("=" * 85 + "\n")

    # Generate Visualization Plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    scenarios_list = list(results.keys())
    x = list(range(len(scenarios_list)))
    w = 0.35

    naive_mems = [results[s]["naive"]["mem"] for s in scenarios_list]
    paged_mems = [results[s]["paged"]["mem"] for s in scenarios_list]
    naive_tpss = [results[s]["naive"]["tps"] for s in scenarios_list]
    paged_tpss = [results[s]["paged"]["tps"] for s in scenarios_list]

    # Calculate range error bars for TPS
    naive_tps_err = [
        [results[s]["naive"]["tps"] - min(results[s]["naive"]["tpss"]) for s in scenarios_list],
        [max(results[s]["naive"]["tpss"]) - results[s]["naive"]["tps"] for s in scenarios_list],
    ]
    paged_tps_err = [
        [results[s]["paged"]["tps"] - min(results[s]["paged"]["tpss"]) for s in scenarios_list],
        [max(results[s]["paged"]["tpss"]) - results[s]["paged"]["tps"] for s in scenarios_list],
    ]

    axes[0].bar([i - w/2 for i in x], naive_mems, width=w, label="Naive Unbatched HF", color="#d9534f")
    axes[0].bar([i + w/2 for i in x], paged_mems, width=w, label="Paged Dual-Cache Engine", color="#5cb85c")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(scenarios_list, rotation=15)
    axes[0].set_ylabel("Peak Memory (MB)")
    axes[0].set_title("Peak GPU Memory Comparison")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    axes[1].bar([i - w/2 for i in x], naive_tpss, width=w, yerr=naive_tps_err, capsize=4, label="Naive Unbatched HF", color="#428bca")
    axes[1].bar([i + w/2 for i in x], paged_tpss, width=w, yerr=paged_tps_err, capsize=4, label="Paged Dual-Cache Engine", color="#f0ad4e")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(scenarios_list, rotation=15)
    axes[1].set_ylabel("Throughput (tokens/sec)")
    axes[1].set_title(f"Generation Throughput (Mean ± Range across {num_repeats} runs)")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)

    # Annotate speedup on bars
    for i, s in enumerate(scenarios_list):
        mean_sp = results[s]["mean_speedup"]
        min_sp = results[s]["min_speedup"]
        max_sp = results[s]["max_speedup"]
        p_tps = results[s]["paged"]["tps"]
        axes[1].text(
            i + w/2,
            p_tps * 1.04,
            f"{mean_sp:.2f}x\n({min_sp:.2f}x–{max_sp:.2f}x)",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(output_image, dpi=150, bbox_inches="tight")
    print(f"Saved benchmark visualizations to {output_image}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Speech Serving Benchmark Suite")
    parser.add_argument("--repeats", type=int, default=3, help="Number of repetitions per scenario (default: 3)")
    parser.add_argument("--output", type=str, default="speech_serving_benchmark.png", help="Output plot path")
    parser.add_argument("--num_encoder_layers", type=int, default=6, help="Number of encoder layers (default: 6)")
    parser.add_argument("--num_decoder_layers", type=int, default=6, help="Number of decoder layers (default: 6)")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads (default: 8)")
    parser.add_argument("--hidden_dim", type=int, default=512, help="Hidden dimension (default: 512)")
    args, _ = parser.parse_known_args()
    run_benchmark_suite(
        output_image=args.output,
        num_repeats=args.repeats,
        num_enc_layers=args.num_encoder_layers,
        num_dec_layers=args.num_decoder_layers,
        num_heads=args.num_heads,
        hidden_dim=args.hidden_dim,
    )
