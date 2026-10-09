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
import random
import gc
import numpy as np
import soundfile as sf
import torch
import jiwer
from typing import List, Dict, Any, Tuple, Optional
import logging
logging.getLogger("speech_serving").setLevel(logging.ERROR)
logging.getLogger("SpeechMemoryGuard").setLevel(logging.ERROR)
logging.getLogger("PhaseAwareSpeechScheduler").setLevel(logging.ERROR)
from transformers import WhisperProcessor
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

from speech_serving.models import PretrainedWhisperModel
from speech_serving.encoder_block_manager import EncoderBlockManager
from speech_serving.dual_cache_manager import DualCacheManager
from speech_serving.phase_aware_scheduler import (
    PhaseAwareSpeechScheduler,
    WHISPER_DEFAULT_PREFIX_TOKENS,
    WHISPER_EOS_TOKEN_ID,
)
from paged_kv_cache.block_manager import BlockManager


def sync_time() -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


class RealSpeechBenchmark:
    def __init__(self, device: str = "cuda", dtype: torch.dtype = torch.float16):
        self.device = device
        self.dtype = dtype
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-base")
        self.normalizer = self.processor.tokenizer.normalize

        print("Initializing PretrainedWhisperModel...")
        self.model = PretrainedWhisperModel(
            model_name_or_path="openai/whisper-base",
            device=device,
            dtype=dtype,
        )

        # Precompute HF greedy reference tokens for all 100 samples
        with open("data/librispeech_subset_100.json", "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

        print(f"Loading and pre-extracting log-mels for {len(self.metadata)} utterances...")
        self.samples = []
        for item in self.metadata:
            audio, sr = sf.read(item["path"])
            if sr != 16000:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            mel = self.processor(audio, sampling_rate=16000, return_tensors="pt").input_features[0].to(dtype=dtype)
            self.samples.append({
                "id": item["id"],
                "duration": item["duration"],
                "stratum": item["stratum"],
                "text": item["text"],
                "norm_ref": self.normalizer(item["text"]),
                "mel": mel,
            })

        print("Precomputing HF baseline tokens for exact match rate check...")
        for s in self.samples:
            mel_in = s["mel"].unsqueeze(0).to(self.device)
            with torch.no_grad():
                hf_out = self.model.hf_model.generate(
                    mel_in,
                    num_beams=1,
                    do_sample=False,
                    language="en",
                    task="transcribe",
                    max_new_tokens=150,
                )
            tokens = hf_out[0].tolist()
            if tokens[:4] == WHISPER_DEFAULT_PREFIX_TOKENS:
                tokens = tokens[4:]
            tokens = [t for t in tokens if t != WHISPER_EOS_TOKEN_ID]
            s["hf_tokens"] = tokens

    def run_sequential(self, traffic: List[Dict[str, Any]], max_new_tokens: int = 150) -> Dict[str, Any]:
        """System 1: Sequential processing (one request at a time)."""
        torch.cuda.reset_peak_memory_stats()
        t_bench_start = sync_time()
        requests_sorted = sorted(traffic, key=lambda x: x["arrival_time"])

        per_request = []
        curr_sim_time = 0.0

        for req in requests_sorted:
            arr_time = req["arrival_time"]
            wait_for_arrival = arr_time - (sync_time() - t_bench_start)
            if wait_for_arrival > 0:
                time.sleep(wait_for_arrival)

            t_req_start = sync_time()
            s_data = self.samples[req["sample_idx"]]
            mel_in = s_data["mel"].unsqueeze(0).to(self.device)

            # 1. Encode
            with torch.no_grad():
                enc_out, enc_kv = self.model.encode(mel_in)

            # 2. Decoder Prefill
            prompt = torch.tensor([WHISPER_DEFAULT_PREFIX_TOKENS], device=self.device, dtype=torch.long)
            with torch.no_grad():
                dec_out = self.model.decode_step(
                    input_ids=prompt,
                    encoder_past_kv_list=enc_kv,
                    use_cache=True,
                )
            t_first_tok = sync_time()
            next_tok = torch.argmax(dec_out.logits[:, -1, :], dim=-1).item()
            gen_tokens = [next_tok]

            # 3. Decode loop
            curr_cache = dec_out.past_key_values
            while next_tok != WHISPER_EOS_TOKEN_ID and len(gen_tokens) < max_new_tokens:
                tok_in = torch.tensor([[next_tok]], device=self.device, dtype=torch.long)
                with torch.no_grad():
                    dec_out = self.model.decode_step(
                        input_ids=tok_in,
                        past_key_values=curr_cache,
                        encoder_past_kv_list=enc_kv,
                        use_cache=True,
                    )
                next_tok = torch.argmax(dec_out.logits[:, -1, :], dim=-1).item()
                gen_tokens.append(next_tok)
                curr_cache = dec_out.past_key_values

            t_req_finish = sync_time()
            per_request.append({
                "req_id": req["req_id"],
                "sample_id": s_data["id"],
                "arrival_time": arr_time,
                "start_time": t_req_start - t_bench_start,
                "first_token_time": t_first_tok - t_bench_start,
                "finish_time": t_req_finish - t_bench_start,
                "ttft": t_first_tok - (t_bench_start + arr_time),
                "e2e_latency": t_req_finish - (t_bench_start + arr_time),
                "gen_tokens": [t for t in gen_tokens if t != WHISPER_EOS_TOKEN_ID],
                "duration": s_data["duration"],
                "norm_ref": s_data["norm_ref"],
                "hf_tokens": s_data["hf_tokens"],
            })

        t_bench_end = sync_time()
        wall_time = t_bench_end - t_bench_start
        peak_alloc = torch.cuda.max_memory_allocated() / (1024 * 1024)
        peak_res = torch.cuda.max_memory_reserved() / (1024 * 1024)
        gc.collect()
        torch.cuda.empty_cache()

        return self._compute_metrics(per_request, wall_time, peak_alloc, peak_res)

    def run_static_batching(self, traffic: List[Dict[str, Any]], batch_size: int = 4, max_new_tokens: int = 150) -> Dict[str, Any]:
        """System 2: Static batching (fixed batch size, padded decode, waits for slowest)."""
        torch.cuda.reset_peak_memory_stats()
        t_bench_start = sync_time()
        queue = sorted(traffic, key=lambda x: x["arrival_time"])
        per_request = []

        q_idx = 0
        while q_idx < len(queue):
            # Wait for at least 1 request to arrive
            next_arr = queue[q_idx]["arrival_time"]
            wait_t = next_arr - (sync_time() - t_bench_start)
            if wait_t > 0:
                time.sleep(wait_t)

            now_rel = sync_time() - t_bench_start
            # Gather up to batch_size available requests
            batch_reqs = []
            while q_idx < len(queue) and len(batch_reqs) < batch_size and queue[q_idx]["arrival_time"] <= now_rel:
                batch_reqs.append(queue[q_idx])
                q_idx += 1
            if not batch_reqs:
                batch_reqs.append(queue[q_idx])
                q_idx += 1

            bsz = len(batch_reqs)
            t_batch_start = sync_time()

            # 1. Encode batch
            b_mels = torch.stack([self.samples[r["sample_idx"]]["mel"] for r in batch_reqs], dim=0).to(self.device)
            with torch.no_grad():
                enc_out, enc_kv = self.model.encode(b_mels)

            # 2. Prefill
            prompt = torch.tensor([WHISPER_DEFAULT_PREFIX_TOKENS] * bsz, device=self.device, dtype=torch.long)
            with torch.no_grad():
                dec_out = self.model.decode_step(
                    input_ids=prompt,
                    encoder_past_kv_list=enc_kv,
                    use_cache=True,
                )
            t_first_tok = sync_time()
            next_tokens = torch.argmax(dec_out.logits[:, -1, :], dim=-1).tolist()
            req_gen_tokens = [[tok] for tok in next_tokens]
            req_dones = [tok == WHISPER_EOS_TOKEN_ID for tok in next_tokens]
            req_finish_times = [t_first_tok if d else None for d in req_dones]

            # 3. Synchronous batch decode loop (waits for all requests to finish)
            curr_cache = dec_out.past_key_values
            step = 1
            while not all(req_dones) and step < max_new_tokens:
                step += 1
                tok_in = torch.tensor([[tok] for tok in next_tokens], device=self.device, dtype=torch.long)
                with torch.no_grad():
                    dec_out = self.model.decode_step(
                        input_ids=tok_in,
                        past_key_values=curr_cache,
                        encoder_past_kv_list=enc_kv,
                        use_cache=True,
                    )
                next_tokens = torch.argmax(dec_out.logits[:, -1, :], dim=-1).tolist()
                t_step = sync_time()
                for i in range(bsz):
                    if not req_dones[i]:
                        req_gen_tokens[i].append(next_tokens[i])
                        if next_tokens[i] == WHISPER_EOS_TOKEN_ID or step >= max_new_tokens:
                            req_dones[i] = True
                            req_finish_times[i] = t_step
                curr_cache = dec_out.past_key_values

            # In static batching, all batch slots are held until the entire batch finishes
            t_batch_finish = sync_time()
            for i, r in enumerate(batch_reqs):
                s_data = self.samples[r["sample_idx"]]
                arr_time = r["arrival_time"]
                # Finish time is when batch finishes
                fin_time = t_batch_finish
                per_request.append({
                    "req_id": r["req_id"],
                    "sample_id": s_data["id"],
                    "arrival_time": arr_time,
                    "start_time": t_batch_start - t_bench_start,
                    "first_token_time": t_first_tok - t_bench_start,
                    "finish_time": fin_time - t_bench_start,
                    "ttft": t_first_tok - (t_bench_start + arr_time),
                    "e2e_latency": fin_time - (t_bench_start + arr_time),
                    "gen_tokens": [t for t in req_gen_tokens[i] if t != WHISPER_EOS_TOKEN_ID],
                    "duration": s_data["duration"],
                    "norm_ref": s_data["norm_ref"],
                    "hf_tokens": s_data["hf_tokens"],
                })

        t_bench_end = sync_time()
        wall_time = t_bench_end - t_bench_start
        peak_alloc = torch.cuda.max_memory_allocated() / (1024 * 1024)
        peak_res = torch.cuda.max_memory_reserved() / (1024 * 1024)
        gc.collect()
        torch.cuda.empty_cache()

        return self._compute_metrics(per_request, wall_time, peak_alloc, peak_res)

    def run_engine(
        self,
        traffic: List[Dict[str, Any]],
        max_batch_size: int = 4,
        max_new_tokens: int = 150,
        retain_encoder_cache: bool = False,
    ) -> Dict[str, Any]:
        """System 3: Engine (continuous batching + paged dual cache)."""
        torch.cuda.reset_peak_memory_stats()

        enc_bm = EncoderBlockManager(
            num_layers=self.model.num_encoder_layers,
            num_heads=self.model.num_heads,
            head_dim=self.model.head_dim,
            block_size=16,
            num_blocks=6144,
            dtype=self.dtype,
            device=self.device,
        )
        dec_bm = BlockManager(
            num_layers=self.model.num_decoder_layers,
            num_kv_heads=self.model.num_heads,
            head_dim=self.model.head_dim,
            block_size=16,
            num_blocks=1024,
            dtype=self.dtype,
            device=self.device,
        )
        cache_mgr = DualCacheManager(enc_bm, dec_bm)
        scheduler = PhaseAwareSpeechScheduler(
            model=self.model,
            dual_cache_manager=cache_mgr,
            max_batch_size=max_batch_size,
            retain_encoder_cache=retain_encoder_cache,
            eos_token_id=WHISPER_EOS_TOKEN_ID,
            max_cached_encoder_inputs=100,
            device=self.device,
        )

        t_bench_start = sync_time()
        requests_sorted = sorted(traffic, key=lambda x: x["arrival_time"])
        q_idx = 0
        prompt_tensor = torch.tensor([WHISPER_DEFAULT_PREFIX_TOKENS], device=self.device, dtype=torch.long)

        # Simulation / Execution loop
        while q_idx < len(requests_sorted) or scheduler.has_work():
            now_rel = sync_time() - t_bench_start
            # Submit arrived requests into engine as long as scheduler queue has room
            while q_idx < len(requests_sorted) and requests_sorted[q_idx]["arrival_time"] <= now_rel and len(scheduler.queue) < max_batch_size * 2:
                r = requests_sorted[q_idx]
                s_data = self.samples[r["sample_idx"]]
                req_id = scheduler.submit(
                    mel_features=s_data["mel"].to(self.device),
                    prompt_token_ids=prompt_tensor,
                    max_new_tokens=max_new_tokens,
                    arrival_time=r["arrival_time"],
                    eos_token_id=WHISPER_EOS_TOKEN_ID,
                )
                if req_id is None:
                    # Engine backpressure active (high water mark reached)
                    break
                q_idx += 1

            if scheduler.has_work():
                scheduler.step()
            elif q_idx < len(requests_sorted):
                # Idle: sleep until next arrival
                wait_t = requests_sorted[q_idx]["arrival_time"] - (sync_time() - t_bench_start)
                if wait_t > 0:
                    time.sleep(min(wait_t, 0.005))

        t_bench_end = sync_time()
        wall_time = t_bench_end - t_bench_start
        peak_alloc = torch.cuda.max_memory_allocated() / (1024 * 1024)
        peak_res = torch.cuda.max_memory_reserved() / (1024 * 1024)

        per_request = []
        for r_comp in scheduler.completed:
            traffic_meta = next(x for x in traffic if x["req_id"] == r_comp.id)
            s_data = self.samples[traffic_meta["sample_idx"]]
            arr_time = traffic_meta["arrival_time"]
            per_request.append({
                "req_id": r_comp.id,
                "sample_id": s_data["id"],
                "arrival_time": arr_time,
                "start_time": r_comp.start_time - t_bench_start,
                "first_token_time": r_comp.first_token_time - t_bench_start,
                "finish_time": r_comp.finish_time - t_bench_start,
                "ttft": r_comp.first_token_time - (t_bench_start + arr_time),
                "e2e_latency": r_comp.finish_time - (t_bench_start + arr_time),
                "gen_tokens": [t for t in r_comp.generated_tokens if t != WHISPER_EOS_TOKEN_ID],
                "duration": s_data["duration"],
                "norm_ref": s_data["norm_ref"],
                "hf_tokens": s_data["hf_tokens"],
            })

        metrics = self._compute_metrics(per_request, wall_time, peak_alloc, peak_res)
        metrics["encoder_telemetry"] = {
            "encoder_calls": scheduler.encoder_calls,
            "encoder_calls_skipped": scheduler.encoder_calls_skipped,
            "encode_time_saved_s": scheduler.encode_time_saved_s,
        }
        metrics["concurrency_stats"] = scheduler.get_concurrency_stats()
        del scheduler, cache_mgr, enc_bm, dec_bm
        gc.collect()
        torch.cuda.empty_cache()
        return metrics

    def _compute_metrics(
        self,
        per_request: List[Dict[str, Any]],
        wall_time: float,
        peak_alloc_mb: float,
        peak_res_mb: float,
    ) -> Dict[str, Any]:
        e2e_lats = [r["e2e_latency"] for r in per_request]
        ttfts = [r["ttft"] for r in per_request]
        total_audio_s = sum(r["duration"] for r in per_request)
        throughput = total_audio_s / wall_time if wall_time > 0 else 0.0

        # Exact token match vs HF generate
        matches = sum(1 for r in per_request if r["gen_tokens"] == r["hf_tokens"])
        token_match_rate = matches / len(per_request) * 100.0 if per_request else 0.0

        # WER calculation
        refs = []
        hyps = []
        for r in per_request:
            hyp_text = self.processor.tokenizer.decode(r["gen_tokens"])
            norm_hyp = self.normalizer(hyp_text)
            r["hyp_text"] = hyp_text
            r["norm_hyp"] = norm_hyp
            refs.append(r["norm_ref"])
            hyps.append(norm_hyp)
        overall_wer = jiwer.wer(refs, hyps) * 100.0

        # Bootstrap 95% confidence interval for corpus WER
        wer_ci_95 = [overall_wer, overall_wer]
        if len(refs) > 1:
            rng = np.random.RandomState(42)
            n_boot = 1000
            idx_arr = np.arange(len(refs))
            boot_wers = []
            for _ in range(n_boot):
                b_idx = rng.choice(idx_arr, size=len(refs), replace=True)
                b_refs = [refs[i] for i in b_idx]
                b_hyps = [hyps[i] for i in b_idx]
                boot_wers.append(jiwer.wer(b_refs, b_hyps) * 100.0)
            wer_ci_95 = [float(np.percentile(boot_wers, 2.5)), float(np.percentile(boot_wers, 97.5))]

        return {
            "num_requests": len(per_request),
            "wall_time_s": wall_time,
            "throughput_audio_per_sec": throughput,
            "ttft_p50_s": float(np.percentile(ttfts, 50)),
            "ttft_p95_s": float(np.percentile(ttfts, 95)),
            "latency_p50_s": float(np.percentile(e2e_lats, 50)),
            "latency_p95_s": float(np.percentile(e2e_lats, 95)),
            "peak_vram_allocated_mb": peak_alloc_mb,
            "peak_vram_reserved_mb": peak_res_mb,
            "wer_pct": overall_wer,
            "wer_ci_95": wer_ci_95,
            "token_match_rate_pct": token_match_rate,
            "per_request": per_request,
        }


def generate_burst_traffic(num_requests: int) -> List[Dict[str, Any]]:
    return [{"req_id": i, "sample_idx": i, "arrival_time": 0.0} for i in range(num_requests)]


def generate_poisson_traffic(num_requests: int, rate: float, seed: int = 42) -> List[Dict[str, Any]]:
    random.seed(seed)
    np.random.seed(seed)
    intervals = np.random.exponential(scale=1.0 / rate, size=num_requests)
    arrivals = np.cumsum(intervals) - intervals[0] # first arrives at t=0
    return [{"req_id": i, "sample_idx": i, "arrival_time": float(arrivals[i])} for i in range(num_requests)]


def generate_repeated_traffic(num_unique: int = 50, seed: int = 42) -> List[Dict[str, Any]]:
    """
    50 utterances submitted twice (100 total requests).
    - 25 pairs burst together at t=0 (concurrent duplicates).
    - 25 pairs staggered with arrival offset (sequential duplicates).
    """
    random.seed(seed)
    traffic = []
    # 25 concurrent pairs (burst at t=0)
    req_id = 0
    for i in range(25):
        traffic.append({"req_id": req_id, "sample_idx": i, "arrival_time": 0.0})
        req_id += 1
        traffic.append({"req_id": req_id, "sample_idx": i, "arrival_time": 0.0})
        req_id += 1
    # 25 staggered pairs (first at t=1.0, second at t=10.0)
    for i in range(25, 50):
        traffic.append({"req_id": req_id, "sample_idx": i, "arrival_time": float(i - 25) * 0.2})
        req_id += 1
        traffic.append({"req_id": req_id, "sample_idx": i, "arrival_time": float(i - 25) * 0.2 + 5.0})
        req_id += 1
    return traffic


def run_benchmark_suite(
    num_repeats: int = 3,
    max_samples: int = 100,
    batch_size: int = 4,
    output_path: str = "task4_real_speech_results.json",
):
    bench = RealSpeechBenchmark(device="cuda", dtype=torch.float16)

    # 1. Warm-up run (discarded)
    print("\n--- Running Warm-up Run (5 samples, discarded) ---")
    warm_traffic = [{"req_id": i, "sample_idx": i, "arrival_time": 0.0} for i in range(min(5, len(bench.samples)))]
    bench.run_sequential(warm_traffic)
    bench.run_static_batching(warm_traffic, batch_size=batch_size)
    bench.run_engine(warm_traffic, max_batch_size=batch_size)
    print("Warm-up complete. Starting measured benchmark runs...")

    num_samples = min(len(bench.samples), max_samples)
    traffic_burst = generate_burst_traffic(num_samples)
    traffic_poisson_low = generate_poisson_traffic(num_samples, rate=1.0, seed=42)
    traffic_poisson_high = generate_poisson_traffic(num_samples, rate=4.0, seed=42)

    profiles = {
        "burst": traffic_burst,
        "poisson_low": traffic_poisson_low,
        "poisson_high": traffic_poisson_high,
    }

    results = {
        "metadata": {
            "gpu_name": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
            "model": "openai/whisper-base",
            "commit_hash": "e37978b90ca9030d5170a5c07aadb050351a65bb",
            "dataset": "LibriSpeech test-clean (100 stratified utterances)",
            "repeats": num_repeats,
        },
        "systems": {},
        "repeated_workload": {},
    }

    # Systems: Sequential, Static Batching, Engine
    system_names = ["sequential", "static_batching", "engine"]

    for prof_name, traffic_data in profiles.items():
        print(f"\n==================================================================")
        print(f"EVALUATING TRAFFIC PROFILE: {prof_name.upper()} (100 utterances)")
        print(f"==================================================================")
        results["systems"][prof_name] = {}

        for sys_name in system_names:
            print(f"\n--- System: {sys_name} ({num_repeats} repeats) ---")
            repeat_runs = []
            for rep in range(num_repeats):
                gc.collect()
                torch.cuda.empty_cache()
                print(f"  [Repeat {rep+1}/{num_repeats}] running {sys_name} on {prof_name}...", end="", flush=True)
                if sys_name == "sequential":
                    run_res = bench.run_sequential(traffic_data)
                elif sys_name == "static_batching":
                    run_res = bench.run_static_batching(traffic_data, batch_size=batch_size)
                elif sys_name == "engine":
                    run_res = bench.run_engine(traffic_data, max_batch_size=batch_size, retain_encoder_cache=False)

                print(f" Done in {run_res['wall_time_s']:.2f}s | Throughput: {run_res['throughput_audio_per_sec']:.2f}x (audio-sec / wall-sec) | WER: {run_res['wer_pct']:.2f}% | Lat p50: {run_res['latency_p50_s']:.2f}s")
                repeat_runs.append(run_res)

            # Compute mean and std across repeats
            agg = {
                "wall_time_s_mean": float(np.mean([r["wall_time_s"] for r in repeat_runs])),
                "wall_time_s_std": float(np.std([r["wall_time_s"] for r in repeat_runs])),
                "throughput_mean": float(np.mean([r["throughput_audio_per_sec"] for r in repeat_runs])),
                "throughput_std": float(np.std([r["throughput_audio_per_sec"] for r in repeat_runs])),
                "throughput_label": "audio-sec / wall-sec (RTF factor x)",
                "ttft_p50_s_mean": float(np.mean([r["ttft_p50_s"] for r in repeat_runs])),
                "ttft_p50_s_std": float(np.std([r["ttft_p50_s"] for r in repeat_runs])),
                "ttft_p95_s_mean": float(np.mean([r["ttft_p95_s"] for r in repeat_runs])),
                "ttft_p95_s_std": float(np.std([r["ttft_p95_s"] for r in repeat_runs])),
                "latency_p50_s_mean": float(np.mean([r["latency_p50_s"] for r in repeat_runs])),
                "latency_p50_s_std": float(np.std([r["latency_p50_s"] for r in repeat_runs])),
                "latency_p95_s_mean": float(np.mean([r["latency_p95_s"] for r in repeat_runs])),
                "latency_p95_s_std": float(np.std([r["latency_p95_s"] for r in repeat_runs])),
                "peak_vram_alloc_mb_mean": float(np.mean([r["peak_vram_allocated_mb"] for r in repeat_runs])),
                "peak_vram_alloc_mb_std": float(np.std([r["peak_vram_allocated_mb"] for r in repeat_runs])),
                "peak_vram_res_mb_mean": float(np.mean([r["peak_vram_reserved_mb"] for r in repeat_runs])),
                "peak_vram_res_mb_std": float(np.std([r["peak_vram_reserved_mb"] for r in repeat_runs])),
                "wer_pct": repeat_runs[0]["wer_pct"],
                "wer_ci_95": repeat_runs[0].get("wer_ci_95", [repeat_runs[0]["wer_pct"], repeat_runs[0]["wer_pct"]]),
                "token_match_rate_pct": repeat_runs[0]["token_match_rate_pct"],
            }
            if sys_name == "engine" and "concurrency_stats" in repeat_runs[0]:
                cs0 = repeat_runs[0]["concurrency_stats"]
                agg["concurrency"] = {
                    "encoder_pool_blocks": cs0["encoder_pool_blocks"],
                    "encoder_pool_mb": cs0["encoder_pool_mb"],
                    "decoder_pool_blocks": cs0["decoder_pool_blocks"],
                    "decoder_pool_mb": cs0["decoder_pool_mb"],
                    "total_pool_mb": cs0["total_pool_mb"],
                    "req_encoder_blocks": cs0["req_encoder_blocks"],
                    "req_encoder_mb": cs0["req_encoder_mb"],
                    "req_decoder_blocks": cs0["req_decoder_blocks"],
                    "req_decoder_mb": cs0["req_decoder_mb"],
                    "avg_in_flight_mean": float(np.mean([r["concurrency_stats"]["avg_in_flight_requests"] for r in repeat_runs])),
                    "max_in_flight_max": int(max([r["concurrency_stats"]["max_in_flight_requests"] for r in repeat_runs])),
                    "queue_wait_time_mean_s": float(np.mean([r["concurrency_stats"]["queue_wait_time_mean_s"] for r in repeat_runs])),
                    "queue_wait_time_p50_s": float(np.mean([r["concurrency_stats"]["queue_wait_time_p50_s"] for r in repeat_runs])),
                    "queue_wait_time_p95_s": float(np.mean([r["concurrency_stats"]["queue_wait_time_p95_s"] for r in repeat_runs])),
                    "rejections_mean": float(np.mean([r["concurrency_stats"]["rejections_per_run"] for r in repeat_runs])),
                }
            results["systems"][prof_name][sys_name] = {
                "aggregate": agg,
                "runs": repeat_runs,
            }

    # Repeated Audio Workload (Cache ON vs Cache OFF)
    print("\n==================================================================")
    print("EVALUATING REPEATED-AUDIO WORKLOAD (Cache ON vs Cache OFF)")
    print("==================================================================")
    rep_traffic = generate_repeated_traffic(num_unique=50, seed=42)

    for cache_mode, retain_cache in [("cache_off", False), ("cache_on", True)]:
        print(f"\n--- Repeated Workload: {cache_mode} ({num_repeats} repeats) ---")
        repeat_runs = []
        for rep in range(num_repeats):
            print(f"  [Repeat {rep+1}/{num_repeats}] running {cache_mode}...", end="", flush=True)
            run_res = bench.run_engine(rep_traffic, max_batch_size=batch_size, retain_encoder_cache=retain_cache)
            telem = run_res["encoder_telemetry"]
            print(f" Done in {run_res['wall_time_s']:.2f}s | Encoder calls: {telem['encoder_calls']}, Skipped: {telem['encoder_calls_skipped']}, Time saved: {telem['encode_time_saved_s']:.2f}s")
            repeat_runs.append(run_res)

        agg = {
            "wall_time_s_mean": float(np.mean([r["wall_time_s"] for r in repeat_runs])),
            "wall_time_s_std": float(np.std([r["wall_time_s"] for r in repeat_runs])),
            "latency_p50_s_mean": float(np.mean([r["latency_p50_s"] for r in repeat_runs])),
            "latency_p50_s_std": float(np.std([r["latency_p50_s"] for r in repeat_runs])),
            "latency_p95_s_mean": float(np.mean([r["latency_p95_s"] for r in repeat_runs])),
            "latency_p95_s_std": float(np.std([r["latency_p95_s"] for r in repeat_runs])),
            "encoder_calls_mean": float(np.mean([r["encoder_telemetry"]["encoder_calls"] for r in repeat_runs])),
            "encoder_calls_skipped_mean": float(np.mean([r["encoder_telemetry"]["encoder_calls_skipped"] for r in repeat_runs])),
            "encode_time_saved_s_mean": float(np.mean([r["encoder_telemetry"]["encode_time_saved_s"] for r in repeat_runs])),
        }
        results["repeated_workload"][cache_mode] = {
            "aggregate": agg,
            "runs": repeat_runs,
        }

    # Save to JSON
    out_file = Path(output_path)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll benchmark results successfully saved to {out_file}!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate real speech serving on Whisper-base LibriSpeech")
    parser.add_argument("--repeats", type=int, default=3, help="Number of measured repeats per condition (default: 3)")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of utterances to benchmark (default: 100)")
    parser.add_argument("--batch-size", type=int, default=4, help="Max batch size for batching/engine (default: 4)")
    parser.add_argument("--output", type=str, default="data/real_speech_evaluation_results.json", help="Output JSON path")
    parser.add_argument("--quick", action="store_true", help="Quick smoke test (5 samples, 1 repeat)")
    args = parser.parse_args()

    reps = 1 if args.quick else args.repeats
    samples_count = 5 if args.quick else args.num_samples
    run_benchmark_suite(num_repeats=reps, max_samples=samples_count, batch_size=args.batch_size, output_path=args.output)
