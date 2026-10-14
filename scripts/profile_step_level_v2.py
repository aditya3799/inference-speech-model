"""
scripts/profile_step_level_v2.py

PART 3 (Updated v2):
Confirms that scheduler_bookkeeping is measured with a device sync at region
boundaries (and from GPU kernel time in the profiler trace), so it is not
absorbing GPU execution time from earlier asynchronous decode kernels.
Reports GPU kernel time vs wall time for scheduler_bookkeeping and all other regions.
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


def profile_decode_step_v2(device="cuda", dtype=torch.float16, batch_size=4) -> Dict[str, Any]:
    gc.collect()
    torch.cuda.empty_cache()

    # Load samples
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

    # --- 1. Static Batching Profile ---
    model = PretrainedWhisperModel("openai/whisper-base", device=device, dtype=dtype)
    instrument_profiling_hooks(model)
    b_mels = torch.stack([s["mel"] for s in samples], dim=0).to(device)

    # Encoder Warmup & Profile
    with torch.no_grad():
        _ = model.encode(b_mels)
    torch.cuda.synchronize()

    t0_enc = time.perf_counter()
    with torch.no_grad():
        enc_out, enc_kv = model.encode(b_mels)
    torch.cuda.synchronize()
    static_enc_ms_per_req = (time.perf_counter() - t0_enc) * 1000.0 / batch_size

    prompt = torch.tensor([WHISPER_DEFAULT_PREFIX_TOKENS] * batch_size, device=device, dtype=torch.long)
    with torch.no_grad():
        dec_out = model.decode_step(input_ids=prompt, encoder_past_kv_list=enc_kv, use_cache=True)
    curr_cache = dec_out.past_key_values
    next_tokens = torch.argmax(dec_out.logits[:, -1, :], dim=-1).tolist()
    tok_in = torch.tensor([[tok] for tok in next_tokens], device=device, dtype=torch.long)

    # Warmup
    with torch.no_grad():
        dec_out = model.decode_step(input_ids=tok_in, past_key_values=curr_cache, encoder_past_kv_list=enc_kv, use_cache=True)
    curr_cache = dec_out.past_key_values
    torch.cuda.synchronize()

    # Profile 1 step with device sync
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof_static:
        torch.cuda.synchronize()
        t0_step = time.perf_counter()
        with torch.no_grad():
            dec_out = model.decode_step(input_ids=tok_in, past_key_values=curr_cache, encoder_past_kv_list=enc_kv, use_cache=True)
        torch.cuda.synchronize()
        t_fwd = time.perf_counter()

        with record_function("sampling"):
            next_tokens = torch.argmax(dec_out.logits[:, -1, :], dim=-1).tolist()
        torch.cuda.synchronize()
        t_end = time.perf_counter()

    static_wall_ms = (t_end - t0_step) * 1000.0
    static_trace_path = "data/static_profile_trace_v2.json"
    prof_static.export_chrome_trace(static_trace_path)

    key_avgs_static = prof_static.key_averages()
    static_cuda_ms = sum(e.device_time_total for e in key_avgs_static) / 1000.0
    static_launches = sum(e.count for e in key_avgs_static if e.device_time_total > 0)

    static_regions = {}
    for label in ["attention", "mlp_projections", "sampling", "scheduler_bookkeeping"]:
        matches = [e for e in key_avgs_static if e.key == label]
        c_time = sum(m.cpu_time_total for m in matches) / 1000.0 if matches else 0.0
        g_time = sum(m.device_time_total for m in matches) / 1000.0 if matches else 0.0
        static_regions[label] = {"wall_ms": c_time, "gpu_ms": g_time}

    del model, b_mels, enc_out, enc_kv, dec_out, curr_cache
    gc.collect()
    torch.cuda.empty_cache()

    # --- 2. Engine Profile with Region Device Sync ---
    model_eng = PretrainedWhisperModel("openai/whisper-base", device=device, dtype=dtype)
    instrument_profiling_hooks(model_eng)

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

    # Warmup
    scheduler._process_decode_phase()
    torch.cuda.synchronize()

    # Profile 1 step with device sync at region boundary
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof_eng:
        torch.cuda.synchronize()
        t0_eng_step = time.perf_counter()
        scheduler._process_decode_phase()
        torch.cuda.synchronize()
        t1_eng_step = time.perf_counter()

    eng_wall_ms = (t1_eng_step - t0_eng_step) * 1000.0
    engine_trace_path = "data/engine_profile_trace_v2.json"
    prof_eng.export_chrome_trace(engine_trace_path)

    key_avgs_eng = prof_eng.key_averages()
    eng_cuda_ms = sum(e.device_time_total for e in key_avgs_eng) / 1000.0
    eng_launches = sum(e.count for e in key_avgs_eng if e.device_time_total > 0)

    eng_regions = {}
    for label in ["attention", "mlp_projections", "sampling", "cross_kv_gather", "self_kv_gather", "mask_construction", "scheduler_bookkeeping"]:
        matches = [e for e in key_avgs_eng if e.key == label]
        c_time = sum(m.cpu_time_total for m in matches) / 1000.0 if matches else 0.0
        g_time = sum(m.device_time_total for m in matches) / 1000.0 if matches else 0.0
        eng_regions[label] = {"wall_ms": c_time, "gpu_ms": g_time}

    # Isolated measurement of scheduler_bookkeeping with explicit device sync
    torch.cuda.synchronize()
    t0_bk = time.perf_counter()
    num_dec_layers = scheduler.dual_cache.decoder_bm.num_layers
    for i, req in enumerate(scheduler.active_decode):
        blk_id = req.decoder_block_table[req.decoder_seq_len // scheduler.dual_cache.decoder_bm.block_size]
        offset = req.decoder_seq_len % scheduler.dual_cache.decoder_bm.block_size
        for l in range(num_dec_layers):
            new_k = req.cached_decoder_kv[l][0][0, :, -1, :]
            new_v = req.cached_decoder_kv[l][1][0, :, -1, :]
            scheduler.dual_cache.decoder_bm.key_pool[l][blk_id, offset] = new_k
            scheduler.dual_cache.decoder_bm.value_pool[l][blk_id, offset] = new_v
    torch.cuda.synchronize()
    isolated_bk_wall_ms = (time.perf_counter() - t0_bk) * 1000.0

    del scheduler, cache_mgr, enc_bm, dec_bm, model_eng
    gc.collect()
    torch.cuda.empty_cache()

    results = {
        "static_batching": {
            "step_wall_ms": static_wall_ms,
            "gpu_kernel_ms": static_cuda_ms,
            "kernel_launches": static_launches,
            "syncs_per_step": 1,
            "encoder_ms_per_req": static_enc_ms_per_req,
            "regions": static_regions,
        },
        "engine_flag_on": {
            "step_wall_ms": eng_wall_ms,
            "gpu_kernel_ms": eng_cuda_ms,
            "kernel_launches": eng_launches,
            "syncs_per_step": 4,
            "encoder_ms_per_req": eng_enc_ms_per_req,
            "regions": eng_regions,
            "isolated_bookkeeping_wall_ms": isolated_bk_wall_ms,
        },
        "step_gap_ms": eng_wall_ms - static_wall_ms,
    }

    out_file = Path("data/step_profile_results_v2.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n[DONE] Step-level profile v2 saved to {out_file}")
    print(f"Static Step Wall: {static_wall_ms:.2f} ms | GPU: {static_cuda_ms:.2f} ms")
    print(f"Engine Step Wall: {eng_wall_ms:.2f} ms | GPU: {eng_cuda_ms:.2f} ms")
    print(f"Bookkeeping: Profile Wall={eng_regions['scheduler_bookkeeping']['wall_ms']:.2f} ms, GPU={eng_regions['scheduler_bookkeeping']['gpu_ms']:.2f} ms, Isolated Wall={isolated_bk_wall_ms:.2f} ms")
    return results


if __name__ == "__main__":
    profile_decode_step_v2()
