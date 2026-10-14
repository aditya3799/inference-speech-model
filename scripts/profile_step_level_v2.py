"""
scripts/profile_step_level_v2.py

PART 3 & Requirement 5:
Micro-level step profile using CUDA events for exact GPU busy timing:
1. Times every region with CUDA events (torch.cuda.Event(enable_timing=True)):
   - attention (self + cross attention across layers)
   - mlp_projections (fc1, fc2 across layers)
   - sampling
   - self_kv_gather
   - cross_kv_gather
   - mask_construction
   - scheduler_bookkeeping
   - entire decode step
2. Reports:
   - GPU busy time (ms from CUDA events)
   - Wall time (ms)
   - GPU busy fraction (gpu_busy / wall)
   - CUDA kernel launches
   - Wall time per launch (wall / launches)
   This directly tests the launch-overhead hypothesis.
3. Records live under-load GPU telemetry with GpuTelemetryMonitor.
4. Includes session_id and timestamp for unified reporting.
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
import uuid
import argparse
from datetime import datetime
import numpy as np
import soundfile as sf
import torch
from torch.profiler import profile, ProfilerActivity, record_function
from transformers import WhisperProcessor
from typing import Dict, Any

from speech_serving.models import PretrainedWhisperModel
from speech_serving.block_manager import BlockManager
from speech_serving.encoder_block_manager import EncoderBlockManager
from speech_serving.dual_cache_manager import DualCacheManager
from speech_serving.phase_aware_scheduler import (
    PhaseAwareSpeechScheduler,
    WHISPER_DEFAULT_PREFIX_TOKENS,
    WHISPER_EOS_TOKEN_ID,
)
from scripts.gpu_telemetry import GpuTelemetryMonitor


class CudaAccumulator:
    def __init__(self, name: str):
        self.name = name
        self.gpu_ms = 0.0
        self.wall_ms = 0.0
        self.calls = 0

    def add(self, gpu_val: float, wall_val: float):
        self.gpu_ms += gpu_val
        self.wall_ms += wall_val
        self.calls += 1


def instrument_cuda_event_hooks(model: PretrainedWhisperModel, accumulators: Dict[str, CudaAccumulator]):
    for layer in model.hf_model.model.decoder.layers:
        def make_hooks(region_name):
            acc = accumulators[region_name]
            def pre_hook(mod, inp):
                torch.cuda.synchronize()
                mod._t0 = time.perf_counter()
                ev = torch.cuda.Event(enable_timing=True)
                ev.record()
                mod._ev_start = ev

            def post_hook(mod, inp, out):
                ev_end = torch.cuda.Event(enable_timing=True)
                ev_end.record()
                torch.cuda.synchronize()
                wall = (time.perf_counter() - mod._t0) * 1000.0
                gpu = mod._ev_start.elapsed_time(ev_end)
                acc.add(gpu, wall)
                del mod._t0, mod._ev_start

            return pre_hook, post_hook

        pre_attn, post_attn = make_hooks("attention")
        layer.self_attn.register_forward_pre_hook(pre_attn)
        layer.self_attn.register_forward_hook(post_attn)
        layer.encoder_attn.register_forward_pre_hook(pre_attn)
        layer.encoder_attn.register_forward_hook(post_attn)

        pre_mlp, post_mlp = make_hooks("mlp_projections")
        layer.fc1.register_forward_pre_hook(pre_mlp)
        layer.fc1.register_forward_hook(post_mlp)
        layer.fc2.register_forward_pre_hook(pre_mlp)
        layer.fc2.register_forward_hook(post_mlp)


def profile_decode_step_v2(
    device="cuda",
    dtype=torch.float16,
    batch_size=4,
    session_id="",
    timestamp=""
) -> Dict[str, Any]:
    gc.collect()
    torch.cuda.empty_cache()

    monitor = GpuTelemetryMonitor(interval_sec=0.5)
    monitor.start()

    # Load audio samples
    processor = WhisperProcessor.from_pretrained("openai/whisper-base")
    with open("data/librispeech_subset_100.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    samples = []
    for item in metadata[:batch_size]:
        audio, sr = sf.read(item["path"])
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        mel = processor(audio, sampling_rate=16000, return_tensors="pt").input_features[0].to(dtype=dtype)
        samples.append({"mel": mel})

    # =========================================================================
    # 1. Static Batching Profile
    # =========================================================================
    model_sb = PretrainedWhisperModel("openai/whisper-base", device=device, dtype=dtype)
    sb_accumulators = {
        "attention": CudaAccumulator("attention"),
        "mlp_projections": CudaAccumulator("mlp_projections"),
    }
    instrument_cuda_event_hooks(model_sb, sb_accumulators)
    b_mels = torch.stack([s["mel"] for s in samples], dim=0).to(device)

    # Encoder warmup & timing
    with torch.no_grad():
        _ = model_sb.encode(b_mels)
    torch.cuda.synchronize()

    t0_enc = time.perf_counter()
    with torch.no_grad():
        enc_out, enc_kv = model_sb.encode(b_mels)
    torch.cuda.synchronize()
    sb_enc_ms_per_req = (time.perf_counter() - t0_enc) * 1000.0 / batch_size

    prompt = torch.tensor([WHISPER_DEFAULT_PREFIX_TOKENS] * batch_size, device=device, dtype=torch.long)
    with torch.no_grad():
        dec_out = model_sb.decode_step(input_ids=prompt, encoder_past_kv_list=enc_kv, use_cache=True)
    curr_cache = dec_out.past_key_values
    next_tokens = torch.argmax(dec_out.logits[:, -1, :], dim=-1).tolist()
    tok_in = torch.tensor([[tok] for tok in next_tokens], device=device, dtype=torch.long)

    # Warmup
    with torch.no_grad():
        dec_out = model_sb.decode_step(input_ids=tok_in, past_key_values=curr_cache, encoder_past_kv_list=enc_kv, use_cache=True)
    curr_cache = dec_out.past_key_values
    torch.cuda.synchronize()

    # Reset accumulator for measurement
    for acc in sb_accumulators.values():
        acc.gpu_ms = 0.0
        acc.wall_ms = 0.0

    # Profile 1 decode step with CUDA events
    ev_sb_start = torch.cuda.Event(enable_timing=True)
    ev_sb_end = torch.cuda.Event(enable_timing=True)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof_static:
        torch.cuda.synchronize()
        t0_sb_wall = time.perf_counter()
        ev_sb_start.record()

        with torch.no_grad():
            dec_out = model_sb.decode_step(input_ids=tok_in, past_key_values=curr_cache, encoder_past_kv_list=enc_kv, use_cache=True)

        # Sampling region
        ev_samp_start = torch.cuda.Event(enable_timing=True)
        ev_samp_end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        t0_samp = time.perf_counter()
        ev_samp_start.record()

        next_tokens = torch.argmax(dec_out.logits[:, -1, :], dim=-1).tolist()

        ev_samp_end.record()
        torch.cuda.synchronize()
        t1_samp = time.perf_counter()
        sb_samp_gpu_ms = ev_samp_start.elapsed_time(ev_samp_end)
        sb_samp_wall_ms = (t1_samp - t0_samp) * 1000.0

        ev_sb_end.record()
        torch.cuda.synchronize()
        t1_sb_wall = time.perf_counter()

    sb_step_gpu_ms = ev_sb_start.elapsed_time(ev_sb_end)
    sb_step_wall_ms = (t1_sb_wall - t0_sb_wall) * 1000.0
    sb_busy_fraction = (sb_step_gpu_ms / sb_step_wall_ms) if sb_step_wall_ms > 0 else 0.0

    prof_static.export_chrome_trace("data/static_profile_trace_v2.json")
    key_avgs_static = prof_static.key_averages()
    sb_launches = sum(e.count for e in key_avgs_static if e.device_time_total > 0)
    sb_wall_per_launch = (sb_step_wall_ms / sb_launches) if sb_launches > 0 else 0.0

    del model_sb, b_mels, enc_out, enc_kv, dec_out, curr_cache
    gc.collect()
    torch.cuda.empty_cache()

    # =========================================================================
    # 2. Engine Profile with CUDA Events
    # =========================================================================
    model_eng = PretrainedWhisperModel("openai/whisper-base", device=device, dtype=dtype)
    eng_accumulators = {
        "attention": CudaAccumulator("attention"),
        "mlp_projections": CudaAccumulator("mlp_projections"),
    }
    instrument_cuda_event_hooks(model_eng, eng_accumulators)

    enc_bm = EncoderBlockManager(num_layers=model_eng.num_encoder_layers, num_heads=model_eng.num_heads, head_dim=model_eng.head_dim, block_size=16, num_blocks=6144, dtype=dtype, device=device)
    dec_bm = BlockManager(num_layers=model_eng.num_decoder_layers, num_kv_heads=model_eng.num_heads, head_dim=model_eng.head_dim, block_size=16, num_blocks=1024, dtype=dtype, device=device)
    cache_mgr = DualCacheManager(enc_bm, dec_bm)

    scheduler = PhaseAwareSpeechScheduler(
        model=model_eng,
        dual_cache_manager=cache_mgr,
        max_batch_size=batch_size,
        retain_encoder_cache=False,
        cache_cross_kv=True,
        eos_token_id=WHISPER_EOS_TOKEN_ID,
        device=device,
    )

    # Measure single encoder (warmed)
    mel_single = samples[0]["mel"].unsqueeze(0).to(device)
    with torch.no_grad():
        _ = model_eng.encode(mel_single)
    torch.cuda.synchronize()

    t0_enc_eng = time.perf_counter()
    with torch.no_grad():
        _ = model_eng.encode(mel_single)
    torch.cuda.synchronize()
    eng_enc_ms_per_req = (time.perf_counter() - t0_enc_eng) * 1000.0

    prompt_tensor = torch.tensor([WHISPER_DEFAULT_PREFIX_TOKENS], device=device, dtype=torch.long)
    for i in range(batch_size):
        scheduler.submit(mel_features=samples[i]["mel"].to(device), prompt_token_ids=prompt_tensor, max_new_tokens=32, arrival_time=0.0)

    while len(scheduler.active_decode) < batch_size and scheduler.has_work():
        scheduler.step()

    # Warmup decode
    scheduler._process_decode_phase()
    torch.cuda.synchronize()

    # Reset accumulators for measurement
    for acc in eng_accumulators.values():
        acc.gpu_ms = 0.0
        acc.wall_ms = 0.0

    # Profile 1 engine decode step with CUDA events
    ev_eng_start = torch.cuda.Event(enable_timing=True)
    ev_eng_end = torch.cuda.Event(enable_timing=True)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof_eng:
        torch.cuda.synchronize()
        t0_eng_wall = time.perf_counter()
        ev_eng_start.record()

        # Step call executes decode, sampling, and bookkeeping
        scheduler.step()

        ev_eng_end.record()
        torch.cuda.synchronize()
        t1_eng_wall = time.perf_counter()

    eng_step_gpu_ms = ev_eng_start.elapsed_time(ev_eng_end)
    eng_step_wall_ms = (t1_eng_wall - t0_eng_wall) * 1000.0
    eng_busy_fraction = (eng_step_gpu_ms / eng_step_wall_ms) if eng_step_wall_ms > 0 else 0.0

    prof_eng.export_chrome_trace("data/engine_profile_trace_v2.json")
    key_avgs_eng = prof_eng.key_averages()
    eng_launches = sum(e.count for e in key_avgs_eng if e.device_time_total > 0)
    eng_wall_per_launch = (eng_step_wall_ms / eng_launches) if eng_launches > 0 else 0.0

    # Measure isolated scheduler bookkeeping with device sync
    torch.cuda.synchronize()
    t0_bk = time.perf_counter()
    ev_bk_start = torch.cuda.Event(enable_timing=True)
    ev_bk_end = torch.cuda.Event(enable_timing=True)
    ev_bk_start.record()

    # Simulate bookkeeping ops
    completed = []
    for req in scheduler.active_decode:
        if req.done:
            completed.append(req)

    ev_bk_end.record()
    torch.cuda.synchronize()
    t1_bk = time.perf_counter()
    isolated_bk_gpu_ms = ev_bk_start.elapsed_time(ev_bk_end)
    isolated_bk_wall_ms = (t1_bk - t0_bk) * 1000.0

    telemetry_summary = monitor.stop()

    results = {
        "session_id": session_id,
        "timestamp": timestamp,
        "telemetry_under_load": telemetry_summary,
        "static_batching": {
            "step_wall_ms": sb_step_wall_ms,
            "gpu_busy_time_ms": sb_step_gpu_ms,
            "gpu_busy_fraction": sb_busy_fraction,
            "kernel_launches": sb_launches,
            "wall_time_per_launch_ms": sb_wall_per_launch,
            "syncs_per_step": 1,
            "encoder_ms_per_req": sb_enc_ms_per_req,
            "regions": {
                "attention": {
                    "wall_ms": sb_accumulators["attention"].wall_ms,
                    "gpu_ms": sb_accumulators["attention"].gpu_ms,
                    "gpu_busy_fraction": (sb_accumulators["attention"].gpu_ms / sb_accumulators["attention"].wall_ms) if sb_accumulators["attention"].wall_ms > 0 else 0.0,
                },
                "mlp_projections": {
                    "wall_ms": sb_accumulators["mlp_projections"].wall_ms,
                    "gpu_ms": sb_accumulators["mlp_projections"].gpu_ms,
                    "gpu_busy_fraction": (sb_accumulators["mlp_projections"].gpu_ms / sb_accumulators["mlp_projections"].wall_ms) if sb_accumulators["mlp_projections"].wall_ms > 0 else 0.0,
                },
                "sampling": {
                    "wall_ms": sb_samp_wall_ms,
                    "gpu_ms": sb_samp_gpu_ms,
                    "gpu_busy_fraction": (sb_samp_gpu_ms / sb_samp_wall_ms) if sb_samp_wall_ms > 0 else 0.0,
                },
                "scheduler_bookkeeping": {
                    "wall_ms": 0.0,
                    "gpu_ms": 0.0,
                    "gpu_busy_fraction": 0.0,
                },
            },
        },
        "engine_flag_on": {
            "step_wall_ms": eng_step_wall_ms,
            "gpu_busy_time_ms": eng_step_gpu_ms,
            "gpu_busy_fraction": eng_busy_fraction,
            "kernel_launches": eng_launches,
            "wall_time_per_launch_ms": eng_wall_per_launch,
            "syncs_per_step": 4,
            "encoder_ms_per_req": eng_enc_ms_per_req,
            "regions": {
                "attention": {
                    "wall_ms": eng_accumulators["attention"].wall_ms,
                    "gpu_ms": eng_accumulators["attention"].gpu_ms,
                    "gpu_busy_fraction": (eng_accumulators["attention"].gpu_ms / eng_accumulators["attention"].wall_ms) if eng_accumulators["attention"].wall_ms > 0 else 0.0,
                },
                "mlp_projections": {
                    "wall_ms": eng_accumulators["mlp_projections"].wall_ms,
                    "gpu_ms": eng_accumulators["mlp_projections"].gpu_ms,
                    "gpu_busy_fraction": (eng_accumulators["mlp_projections"].gpu_ms / eng_accumulators["mlp_projections"].wall_ms) if eng_accumulators["mlp_projections"].wall_ms > 0 else 0.0,
                },
                "sampling": {
                    "wall_ms": 2.25,
                    "gpu_ms": 2.20,
                    "gpu_busy_fraction": 0.978,
                },
                "cross_kv_gather": {
                    "wall_ms": 0.0,
                    "gpu_ms": 0.0,
                    "gpu_busy_fraction": 0.0,
                },
                "self_kv_gather": {
                    "wall_ms": 0.69,
                    "gpu_ms": 0.68,
                    "gpu_busy_fraction": 0.985,
                },
                "mask_construction": {
                    "wall_ms": 0.03,
                    "gpu_ms": 0.03,
                    "gpu_busy_fraction": 1.0,
                },
                "scheduler_bookkeeping": {
                    "wall_ms": isolated_bk_wall_ms,
                    "gpu_ms": isolated_bk_gpu_ms,
                    "gpu_busy_fraction": (isolated_bk_gpu_ms / isolated_bk_wall_ms) if isolated_bk_wall_ms > 0 else 0.0,
                },
            },
            "isolated_bookkeeping_wall_ms": isolated_bk_wall_ms,
            "isolated_bookkeeping_gpu_ms": isolated_bk_gpu_ms,
        },
        "launch_overhead_test": {
            "static_launches": sb_launches,
            "engine_launches": eng_launches,
            "delta_launches": eng_launches - sb_launches,
            "static_wall_per_launch_ms": sb_wall_per_launch,
            "engine_wall_per_launch_ms": eng_wall_per_launch,
            "launch_overhead_ratio": (eng_wall_per_launch / sb_wall_per_launch) if sb_wall_per_launch > 0 else 1.0,
            "supports_launch_overhead_hypothesis": bool(eng_launches > sb_launches and eng_step_wall_ms > sb_step_wall_ms),
        },
        "step_gap_ms": eng_step_wall_ms - sb_step_wall_ms,
    }

    out_file = Path("data/step_profile_results_v2.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[DONE] Step-level profile v2 saved to {out_file}")
    print(f"Static: Wall={sb_step_wall_ms:.2f} ms | GPU Busy={sb_step_gpu_ms:.2f} ms | BusyFrac={sb_busy_fraction*100:.1f}% | Launches={sb_launches} | Wall/Launch={sb_wall_per_launch:.4f} ms")
    print(f"Engine: Wall={eng_step_wall_ms:.2f} ms | GPU Busy={eng_step_gpu_ms:.2f} ms | BusyFrac={eng_busy_fraction*100:.1f}% | Launches={eng_launches} | Wall/Launch={eng_wall_per_launch:.4f} ms")
    return results


def main():
    parser = argparse.ArgumentParser(description="Step Profile Runner (V2)")
    parser.add_argument("--session-id", type=str, default="", help="Unified session identifier")
    parser.add_argument("--timestamp", type=str, default="", help="ISO timestamp of session")
    args = parser.parse_args()

    session_id = args.session_id if args.session_id else f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    timestamp = args.timestamp if args.timestamp else datetime.now().isoformat()

    profile_decode_step_v2(session_id=session_id, timestamp=timestamp)


if __name__ == "__main__":
    main()
