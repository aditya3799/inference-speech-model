"""
scripts/run_controlled_experiments.py

PART 5: Controlled Experiments for Engine Optimization Flags.
Measures:
  1. Encoder batching: batch_encoder=True (batches encoder across admitted requests).
  2. Sync removal: no_sync_decode=True (removes per-request .item() syncs in decode loop).
  3. Combined: batch_encoder=True AND no_sync_decode=True.
  4. Baseline: unmodified engine (both flags False, gather-once flag ON).

Verifies:
  - 20-utterance parity check with HuggingFace greedy generate.
  - 3 repeats on poisson_high (primary), burst, and poisson_low.
  - Mean +/- std for latency p50, p95, and throughput.
  - Flags remain default OFF.

Outputs saved to data/controlled_experiments_results.json.
"""

import os
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import json
import time
import gc
import numpy as np
import torch
from typing import Dict, Any, List

from scripts.evaluate_real_speech import (
    RealSpeechBenchmark,
    check_session_gate,
    generate_burst_traffic,
    generate_poisson_traffic,
)
from speech_serving.phase_aware_scheduler import (
    WHISPER_DEFAULT_PREFIX_TOKENS,
    WHISPER_EOS_TOKEN_ID,
)


def verify_hf_parity(bench: RealSpeechBenchmark, num_samples: int = 20) -> Dict[str, Any]:
    print(f"\n==================================================================")
    print(f"PART 5.1: 20-UTTERANCE PARITY CHECK WITH HUGGINGFACE GENERATE")
    print(f"==================================================================")

    test_samples = bench.samples[:num_samples]
    traffic = [{"req_id": i, "sample_idx": i, "arrival_time": 0.0} for i in range(num_samples)]

    configs = {
        "engine_baseline": {"batch_encoder": False, "no_sync_decode": False},
        "engine_batch_encoder": {"batch_encoder": True, "no_sync_decode": False},
        "engine_no_sync": {"batch_encoder": False, "no_sync_decode": True},
        "engine_combined": {"batch_encoder": True, "no_sync_decode": True},
    }

    parity_results = {}

    for cfg_name, flags in configs.items():
        gc.collect()
        torch.cuda.empty_cache()
        res = bench.run_engine(
            traffic,
            max_batch_size=4,
            max_new_tokens=150,
            retain_encoder_cache=False,
            cache_cross_kv=True,
            batch_encoder=flags["batch_encoder"],
            no_sync_decode=flags["no_sync_decode"],
        )

        exact_utterances = 0
        total_tokens_matched = 0
        total_tokens_ref = 0

        for r in res["per_request"]:
            s_data = bench.samples[r["req_id"]]
            hf_toks = s_data["hf_tokens"]
            gen_toks = r["gen_tokens"]

            if gen_toks == hf_toks:
                exact_utterances += 1

            min_len = min(len(gen_toks), len(hf_toks))
            matched = sum(1 for a, b in zip(gen_toks[:min_len], hf_toks[:min_len]) if a == b)
            total_tokens_matched += matched
            total_tokens_ref += max(len(gen_toks), len(hf_toks))

        utt_rate = (exact_utterances / num_samples) * 100.0
        tok_rate = (total_tokens_matched / total_tokens_ref * 100.0) if total_tokens_ref > 0 else 0.0

        parity_results[cfg_name] = {
            "exact_utterance_matches": exact_utterances,
            "total_utterances": num_samples,
            "exact_utterance_match_pct": utt_rate,
            "token_match_rate_pct": tok_rate,
            "wer_pct": res["wer_pct"],
        }
        print(f"  [{cfg_name}] Exact Utterances: {exact_utterances}/{num_samples} ({utt_rate:.1f}%) | Token Match: {tok_rate:.2f}% | WER: {res['wer_pct']:.2f}%")

    return parity_results


def run_controlled_benchmarks(bench: RealSpeechBenchmark, num_repeats: int = 3, max_samples: int = 100) -> Dict[str, Any]:
    print(f"\n==================================================================")
    print(f"PART 5.2: CONTROLLED BENCHMARKS (3 REPEATS, 100 UTTERANCES)")
    print(f"==================================================================")

    num_samples = min(len(bench.samples), max_samples)
    traffic_poisson_high = generate_poisson_traffic(num_samples, rate=4.0, seed=42)
    traffic_burst = generate_burst_traffic(num_samples)
    traffic_poisson_low = generate_poisson_traffic(num_samples, rate=1.0, seed=42)

    profiles = {
        "poisson_high": traffic_poisson_high,
        "burst": traffic_burst,
        "poisson_low": traffic_poisson_low,
    }

    configs = {
        "static_batching": None,
        "engine_baseline": {"batch_encoder": False, "no_sync_decode": False},
        "engine_batch_encoder": {"batch_encoder": True, "no_sync_decode": False},
        "engine_no_sync": {"batch_encoder": False, "no_sync_decode": True},
        "engine_combined": {"batch_encoder": True, "no_sync_decode": True},
    }

    # Warm-up (discarded)
    print("Running short warm-up...")
    warm_traffic = [{"req_id": i, "sample_idx": i, "arrival_time": 0.0} for i in range(5)]
    bench.run_static_batching(warm_traffic, batch_size=4)
    bench.run_engine(warm_traffic, max_batch_size=4, cache_cross_kv=True, batch_encoder=False, no_sync_decode=False)
    bench.run_engine(warm_traffic, max_batch_size=4, cache_cross_kv=True, batch_encoder=True, no_sync_decode=True)

    benchmark_results = {}

    for prof_name, traffic_data in profiles.items():
        print(f"\n--- Traffic Profile: {prof_name.upper()} ---")
        benchmark_results[prof_name] = {}

        for cfg_name, flags in configs.items():
            print(f"  Configuration: {cfg_name} ({num_repeats} repeats)...")
            runs = []
            for rep in range(num_repeats):
                gc.collect()
                torch.cuda.empty_cache()
                if cfg_name == "static_batching":
                    res = bench.run_static_batching(traffic_data, batch_size=4)
                else:
                    res = bench.run_engine(
                        traffic_data,
                        max_batch_size=4,
                        max_new_tokens=150,
                        retain_encoder_cache=False,
                        cache_cross_kv=True,
                        batch_encoder=flags["batch_encoder"],
                        no_sync_decode=flags["no_sync_decode"],
                    )
                runs.append({
                    "throughput": res["throughput_audio_per_sec"],
                    "latency_p50": res["latency_p50_s"],
                    "latency_p95": res["latency_p95_s"],
                    "ttft_p50": res["ttft_p50_s"],
                    "ttft_p95": res["ttft_p95_s"],
                    "peak_vram_allocated_mb": res["peak_vram_allocated_mb"],
                })

            tps = [r["throughput"] for r in runs]
            p50s = [r["latency_p50"] for r in runs]
            p95s = [r["latency_p95"] for r in runs]
            vrams = [r["peak_vram_allocated_mb"] for r in runs]

            benchmark_results[prof_name][cfg_name] = {
                "throughput_mean": float(np.mean(tps)),
                "throughput_std": float(np.std(tps)),
                "latency_p50_mean": float(np.mean(p50s)),
                "latency_p50_std": float(np.std(p50s)),
                "latency_p95_mean": float(np.mean(p95s)),
                "latency_p95_std": float(np.std(p95s)),
                "peak_vram_mb": float(np.mean(vrams)),
                "runs": runs,
            }

            print(f"    -> Throughput: {np.mean(tps):.2f} +/- {np.std(tps):.2f}x | Latency p50: {np.mean(p50s):.3f} +/- {np.std(p50s):.3f}s | p95: {np.mean(p95s):.3f} +/- {np.std(p95s):.3f}s")

    return benchmark_results


def main():
    gate_res = check_session_gate(force_unsafe=False)
    bench = RealSpeechBenchmark(device="cuda", dtype=torch.float16)

    parity_res = verify_hf_parity(bench, num_samples=20)
    bench_res = run_controlled_benchmarks(bench, num_repeats=3, max_samples=100)

    output = {
        "session_gate": gate_res,
        "parity_check_20_utterances": parity_res,
        "benchmarks": bench_res,
    }

    out_file = Path("data/controlled_experiments_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\n[DONE] Controlled experiments saved to {out_file}")


if __name__ == "__main__":
    main()
