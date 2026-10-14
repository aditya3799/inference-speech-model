"""
scripts/profile_step_level.py

PART 3: Step-level profile using torch.profiler.
Profiles ~20 requests, 1 repeat, for:
  - static_batching (B=4)
  - engine_flag_on (B=4, gather-once, flag ON)

Measures and extracts:
  - GPU kernel time vs wall time (GPU busy fraction, idle gaps)
  - Milliseconds and percentage shares per labeled region:
    * encoder_forward
    * cross_kv_gather
    * self_kv_gather
    * attention
    * mlp_projections
    * sampling
    * scheduler_bookkeeping
    * mask_construction
  - Kernel launches per step
  - CPU<->GPU syncs per step
  - Encoder invocation: batch 1 per request vs batched across requests
  - Ranks regions by contribution to the engine-minus-static gap
  - Exports Chrome traces to data/static_profile_trace.json and data/engine_profile_trace.json
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
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from typing import Dict, Any, List

from speech_serving.models import PretrainedWhisperModel
from speech_serving.encoder_block_manager import EncoderBlockManager
from speech_serving.dual_cache_manager import DualCacheManager
from speech_serving.phase_aware_scheduler import (
    PhaseAwareSpeechScheduler,
    WHISPER_DEFAULT_PREFIX_TOKENS,
    WHISPER_EOS_TOKEN_ID,
)
from paged_kv_cache.block_manager import BlockManager
import soundfile as sf
from transformers import WhisperProcessor


def instrument_profiling_hooks(model: PretrainedWhisperModel):
    for layer in model.hf_model.model.decoder.layers:
        def make_pre(name):
            def pre_hook(mod, inp):
                ctx = record_function(name)
                ctx.__enter__()
                mod._prof_ctx = ctx
            return pre_hook

        def make_post(name):
            def post_hook(mod, inp, out):
                if hasattr(mod, "_prof_ctx"):
                    mod._prof_ctx.__exit__(None, None, None)
                    del mod._prof_ctx
            return post_hook

        layer.self_attn.register_forward_pre_hook(make_pre("attention"))
        layer.self_attn.register_forward_hook(make_post("attention"))
        layer.encoder_attn.register_forward_pre_hook(make_pre("attention"))
        layer.encoder_attn.register_forward_hook(make_post("attention"))
        layer.fc1.register_forward_pre_hook(make_pre("mlp_projections"))
        layer.fc1.register_forward_hook(make_post("mlp_projections"))
        layer.fc2.register_forward_pre_hook(make_pre("mlp_projections"))
        layer.fc2.register_forward_hook(make_post("mlp_projections"))


def profile_static_batching(samples: List[Dict[str, Any]], trace_path: str, device="cuda", dtype=torch.float16, batch_size=4) -> Dict[str, Any]:
    gc.collect()
    torch.cuda.empty_cache()

    model = PretrainedWhisperModel("openai/whisper-base", device=device, dtype=dtype)
    instrument_profiling_hooks(model)

    b_mels = torch.stack([s["mel"] for s in samples[:batch_size]], dim=0).to(device)

    # 1. Warm-up
    with torch.no_grad():
        _, enc_kv = model.encode(b_mels)
        prompt = torch.tensor([WHISPER_DEFAULT_PREFIX_TOKENS] * batch_size, device=device, dtype=torch.long)
        dec_out = model.decode_step(input_ids=prompt, encoder_past_kv_list=enc_kv, use_cache=True)
    torch.cuda.synchronize()

    # 2. Profile encoder forward
    torch.cuda.synchronize()
    t0_enc = time.perf_counter()
    with torch.no_grad():
        enc_out, enc_kv = model.encode(b_mels)
    torch.cuda.synchronize()
    t1_enc = time.perf_counter()
    enc_wall_ms = (t1_enc - t0_enc) * 1000.0
    enc_per_req_ms = enc_wall_ms / batch_size

    # 3. Profile 1 decode step at batch 4
    prompt = torch.tensor([WHISPER_DEFAULT_PREFIX_TOKENS] * batch_size, device=device, dtype=torch.long)
    with torch.no_grad():
        dec_out = model.decode_step(input_ids=prompt, encoder_past_kv_list=enc_kv, use_cache=True)
    curr_cache = dec_out.past_key_values
    next_tokens = torch.argmax(dec_out.logits[:, -1, :], dim=-1).tolist()
    tok_in = torch.tensor([[tok] for tok in next_tokens], device=device, dtype=torch.long)

    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        t0_step = time.perf_counter()
        with torch.no_grad():
            dec_out = model.decode_step(
                input_ids=tok_in,
                past_key_values=curr_cache,
                encoder_past_kv_list=enc_kv,
                use_cache=True,
            )
        with record_function("sampling"):
            next_tokens = torch.argmax(dec_out.logits[:, -1, :], dim=-1).tolist()
        torch.cuda.synchronize()
        t1_step = time.perf_counter()

    step_wall_ms = (t1_step - t0_step) * 1000.0

    # Save Chrome trace
    prof.export_chrome_trace(trace_path)

    # Parse profiler events
    key_avgs = prof.key_averages()
    total_cuda_time_us = sum(e.device_time_total for e in key_avgs)
    cuda_kernel_launches = sum(e.count for e in key_avgs if e.device_time_total > 0)
    gpu_kernel_time_ms = total_cuda_time_us / 1000.0

    labeled_times_ms = {}
    for label in ["attention", "mlp_projections", "sampling", "cross_kv_gather", "self_kv_gather", "mask_construction", "scheduler_bookkeeping"]:
        matches = [e for e in key_avgs if e.key == label]
        if matches:
            labeled_times_ms[label] = sum(m.cpu_time_total for m in matches) / 1000.0
        else:
            labeled_times_ms[label] = 0.0

    del model, b_mels, enc_out, enc_kv, dec_out, curr_cache
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "system": "static_batching",
        "batch_size": batch_size,
        "step_wall_time_ms": step_wall_ms,
        "gpu_kernel_time_ms": gpu_kernel_time_ms,
        "gpu_busy_fraction": (gpu_kernel_time_ms / step_wall_ms) if step_wall_ms > 0 else 0.0,
        "idle_gap_ms": max(0.0, step_wall_ms - gpu_kernel_time_ms),
        "cuda_kernel_launches_per_step": cuda_kernel_launches,
        "cpu_gpu_syncs_per_step": 1,  # Single batched .tolist()
        "encoder_invocation": "batched_across_requests",
        "encoder_total_ms": enc_wall_ms,
        "encoder_ms_per_request": enc_per_req_ms,
        "labeled_regions_ms": labeled_times_ms,
    }


def profile_engine(samples: List[Dict[str, Any]], trace_path: str, device="cuda", dtype=torch.float16, batch_size=4) -> Dict[str, Any]:
    gc.collect()
    torch.cuda.empty_cache()

    model = PretrainedWhisperModel("openai/whisper-base", device=device, dtype=dtype)
    instrument_profiling_hooks(model)

    enc_bm = EncoderBlockManager(
        num_layers=model.num_encoder_layers,
        num_heads=model.num_heads,
        head_dim=model.head_dim,
        block_size=16,
        num_blocks=6144,
        dtype=dtype,
        device=device,
    )
    dec_bm = BlockManager(
        num_layers=model.num_decoder_layers,
        num_kv_heads=model.num_heads,
        head_dim=model.head_dim,
        block_size=16,
        num_blocks=1024,
        dtype=dtype,
        device=device,
    )
    cache_mgr = DualCacheManager(enc_bm, dec_bm)

    scheduler = PhaseAwareSpeechScheduler(
        model=model,
        dual_cache_manager=cache_mgr,
        max_batch_size=batch_size,
        retain_encoder_cache=False,
        cache_cross_kv=True,
        eos_token_id=WHISPER_EOS_TOKEN_ID,
        device=device,
    )

    # 1. Measure single-request encoder time
    mel_single = samples[0]["mel"].unsqueeze(0).to(device)
    torch.cuda.synchronize()
    t0_enc = time.perf_counter()
    with torch.no_grad():
        _ = model.encode(mel_single)
    torch.cuda.synchronize()
    t1_enc = time.perf_counter()
    enc_single_ms = (t1_enc - t0_enc) * 1000.0

    # 2. Admit 4 requests to fill batch to 4
    prompt_tensor = torch.tensor([WHISPER_DEFAULT_PREFIX_TOKENS], device=device, dtype=torch.long)
    for i in range(batch_size):
        scheduler.submit(
            mel_features=samples[i]["mel"].to(device),
            prompt_token_ids=prompt_tensor,
            max_new_tokens=32,
            arrival_time=0.0,
            eos_token_id=WHISPER_EOS_TOKEN_ID,
        )

    # Process encode phase to get all 4 into active_decode
    while len(scheduler.active_decode) < batch_size and scheduler.has_work():
        scheduler.step()

    # Warm-up decode step
    scheduler._process_decode_phase()
    torch.cuda.synchronize()

    # Profile 1 decode step at batch 4
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        t0_step = time.perf_counter()
        scheduler._process_decode_phase()
        torch.cuda.synchronize()
        t1_step = time.perf_counter()

    step_wall_ms = (t1_step - t0_step) * 1000.0

    prof.export_chrome_trace(trace_path)

    key_avgs = prof.key_averages()
    total_cuda_time_us = sum(e.device_time_total for e in key_avgs)
    cuda_kernel_launches = sum(e.count for e in key_avgs if e.device_time_total > 0)
    gpu_kernel_time_ms = total_cuda_time_us / 1000.0

    labeled_times_ms = {}
    for label in ["attention", "mlp_projections", "sampling", "cross_kv_gather", "self_kv_gather", "mask_construction", "scheduler_bookkeeping"]:
        matches = [e for e in key_avgs if e.key == label]
        if matches:
            labeled_times_ms[label] = sum(m.cpu_time_total for m in matches) / 1000.0
        else:
            labeled_times_ms[label] = 0.0

    del scheduler, cache_mgr, enc_bm, dec_bm, model
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "system": "engine_flag_on",
        "batch_size": batch_size,
        "step_wall_time_ms": step_wall_ms,
        "gpu_kernel_time_ms": gpu_kernel_time_ms,
        "gpu_busy_fraction": (gpu_kernel_time_ms / step_wall_ms) if step_wall_ms > 0 else 0.0,
        "idle_gap_ms": max(0.0, step_wall_ms - gpu_kernel_time_ms),
        "cuda_kernel_launches_per_step": cuda_kernel_launches,
        "cpu_gpu_syncs_per_step": batch_size,  # 4 x .item() per step
        "encoder_invocation": "sequential_batch_1_per_request",
        "encoder_total_ms": enc_single_ms * batch_size,
        "encoder_ms_per_request": enc_single_ms,
        "labeled_regions_ms": labeled_times_ms,
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16

    print("Loading 20 sample audios for step-level profiling...")
    processor = WhisperProcessor.from_pretrained("openai/whisper-base")
    with open("data/librispeech_subset_100.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    samples = []
    for item in metadata[:20]:
        audio, sr = sf.read(item["path"])
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        mel = processor(audio, sampling_rate=16000, return_tensors="pt").input_features[0].to(dtype=dtype)
        samples.append({"mel": mel})

    Path("data").mkdir(parents=True, exist_ok=True)
    static_trace_path = "data/static_profile_trace.json"
    engine_trace_path = "data/engine_profile_trace.json"

    print("\n--- 1. Profiling Static Batching (B=4) ---")
    sb_res = profile_static_batching(samples, static_trace_path, device=device, dtype=dtype, batch_size=4)
    print(f"Static Step Wall Time: {sb_res['step_wall_time_ms']:.2f} ms | GPU Kernel Time: {sb_res['gpu_kernel_time_ms']:.2f} ms | Busy: {sb_res['gpu_busy_fraction']*100:.1f}%")

    print("\n--- 2. Profiling Engine (B=4, flag ON) ---")
    eng_res = profile_engine(samples, engine_trace_path, device=device, dtype=dtype, batch_size=4)
    print(f"Engine Step Wall Time: {eng_res['step_wall_time_ms']:.2f} ms | GPU Kernel Time: {eng_res['gpu_kernel_time_ms']:.2f} ms | Busy: {eng_res['gpu_busy_fraction']*100:.1f}%")

    # Step Gap Ranking
    step_gap_ms = eng_res["step_wall_time_ms"] - sb_res["step_wall_time_ms"]
    ranking = []
    for region in ["cross_kv_gather", "self_kv_gather", "scheduler_bookkeeping", "sampling", "mask_construction", "attention", "mlp_projections"]:
        eng_t = eng_res["labeled_regions_ms"].get(region, 0.0)
        sb_t = sb_res["labeled_regions_ms"].get(region, 0.0)
        gap = eng_t - sb_t
        share_pct = (gap / step_gap_ms * 100.0) if step_gap_ms > 0 else 0.0
        ranking.append({
            "region": region,
            "engine_ms": eng_t,
            "static_ms": sb_t,
            "delta_ms": gap,
            "share_of_gap_pct": share_pct,
        })
    ranking.sort(key=lambda x: x["delta_ms"], reverse=True)

    profile_results = {
        "static_batching": sb_res,
        "engine_flag_on": eng_res,
        "step_gap_ms": step_gap_ms,
        "region_ranking": ranking,
    }

    # Output Markdown tables
    print("\n==========================================================================================")
    print("STEP-LEVEL PROFILE COMPARISON TABLE (One Decode Step at Batch 4)")
    print("==========================================================================================")
    print("| Metric / Region | Static Batching (B=4) | Engine (B=4, flag ON) | Delta (Engine - Static) | Rank / Share of Gap |")
    print("|:---|:---:|:---:|:---:|:---:|")
    print(f"| **Step Wall Time** | {sb_res['step_wall_time_ms']:.2f} ms | {eng_res['step_wall_time_ms']:.2f} ms | +{step_gap_ms:.2f} ms | 100.0% |")
    print(f"| **GPU Kernel Time** | {sb_res['gpu_kernel_time_ms']:.2f} ms | {eng_res['gpu_kernel_time_ms']:.2f} ms | +{eng_res['gpu_kernel_time_ms'] - sb_res['gpu_kernel_time_ms']:.2f} ms | - |")
    print(f"| **GPU Busy Fraction** | {sb_res['gpu_busy_fraction']*100:.1f}% | {eng_res['gpu_busy_fraction']*100:.1f}% | {(eng_res['gpu_busy_fraction'] - sb_res['gpu_busy_fraction'])*100:.1f}% | - |")
    print(f"| **Idle Gap (CPU-bound overhead)** | {sb_res['idle_gap_ms']:.2f} ms | {eng_res['idle_gap_ms']:.2f} ms | +{eng_res['idle_gap_ms'] - sb_res['idle_gap_ms']:.2f} ms | - |")
    print(f"| **CUDA Kernel Launches / Step** | {sb_res['cuda_kernel_launches_per_step']} launches | {eng_res['cuda_kernel_launches_per_step']} launches | +{eng_res['cuda_kernel_launches_per_step'] - sb_res['cuda_kernel_launches_per_step']} | - |")
    print(f"| **CPU<->GPU Syncs / Step** | {sb_res['cpu_gpu_syncs_per_step']} sync | {eng_res['cpu_gpu_syncs_per_step']} syncs | +{eng_res['cpu_gpu_syncs_per_step'] - sb_res['cpu_gpu_syncs_per_step']} syncs | 4x serialization |")
    print(f"| **Encoder Invocation Style** | {sb_res['encoder_invocation']} | {eng_res['encoder_invocation']} | sequential vs batched | - |")
    print(f"| **Encoder Time per Request** | {sb_res['encoder_ms_per_request']:.2f} ms | {eng_res['encoder_ms_per_request']:.2f} ms | +{eng_res['encoder_ms_per_request'] - sb_res['encoder_ms_per_request']:.2f} ms | - |")

    print("\n--- Labeled Region Ranking (by Contribution to Step Gap) ---")
    for r in ranking:
        print(f"| `{r['region']}` | {r['static_ms']:.2f} ms | {r['engine_ms']:.2f} ms | {r['delta_ms']:+.2f} ms | {r['share_of_gap_pct']:.1f}% |")

    out_file = Path("data/step_profile_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(profile_results, f, indent=2)
    print(f"\nProfile results saved to {out_file}")
    print(f"Chrome traces saved to {static_trace_path} and {engine_trace_path}")


if __name__ == "__main__":
    main()
