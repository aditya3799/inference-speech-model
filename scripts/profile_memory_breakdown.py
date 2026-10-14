"""
scripts/profile_memory_breakdown.py

PART 4: Memory breakdown using torch.cuda.memory_stats() in fresh sub-processes.
Reports peak memory components:
  - Model weights (MB)
  - Activations (MB)
  - Block pool allocated (MB)
  - Block pool used (MB)
  - Contiguous cross-KV copies (MB)
  - Peak allocated (MB)
  - Peak reserved (MB)

Evaluates:
  1. static_batching (B=4)
  2. engine (default pools: 6144 enc, 1024 dec)
  3. engine, pools right-sized to cap (414 enc, 256 dec)
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
from typing import Dict, Any

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


def measure_static_batching_memory(samples, device="cuda", dtype=torch.float16, batch_size=4) -> Dict[str, Any]:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    vram_init = torch.cuda.memory_allocated() / (1024 * 1024)

    # 1. Model load
    model = PretrainedWhisperModel("openai/whisper-base", device=device, dtype=dtype)
    torch.cuda.synchronize()
    vram_after_model = torch.cuda.memory_allocated() / (1024 * 1024)
    model_weights_mb = vram_after_model - vram_init

    # 2. Run static batching on batch_size samples
    b_mels = torch.stack([s["mel"] for s in samples[:batch_size]], dim=0).to(device)
    torch.cuda.reset_peak_memory_stats()

    with torch.no_grad():
        enc_out, enc_kv = model.encode(b_mels)
        prompt = torch.tensor([WHISPER_DEFAULT_PREFIX_TOKENS] * batch_size, device=device, dtype=torch.long)
        dec_out = model.decode_step(input_ids=prompt, encoder_past_kv_list=enc_kv, use_cache=True)
        curr_cache = dec_out.past_key_values

        for step in range(32):
            next_tokens = torch.argmax(dec_out.logits[:, -1, :], dim=-1).tolist()
            tok_in = torch.tensor([[tok] for tok in next_tokens], device=device, dtype=torch.long)
            dec_out = model.decode_step(
                input_ids=tok_in,
                past_key_values=curr_cache,
                encoder_past_kv_list=enc_kv,
                use_cache=True
            )
            curr_cache = dec_out.past_key_values

    torch.cuda.synchronize()
    peak_alloc = torch.cuda.max_memory_allocated() / (1024 * 1024)
    peak_res = torch.cuda.max_memory_reserved() / (1024 * 1024)

    # Contiguous cross-KV copies in static batching: 4 * 6 * 2 * 8 * 1500 * 64 * 2 = 73.7 MB
    enc_kv_mb = sum(k.nbytes + v.nbytes for k, v in enc_kv) / (1024 * 1024)
    activations_mb = max(0.0, peak_alloc - vram_after_model - enc_kv_mb)

    del model, b_mels, enc_out, enc_kv, dec_out, curr_cache
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "system": "static_batching",
        "model_weights_mb": model_weights_mb,
        "block_pool_allocated_mb": 0.0,
        "block_pool_used_mb": 0.0,
        "contiguous_cross_kv_mb": enc_kv_mb,
        "activations_mb": activations_mb,
        "peak_allocated_mb": peak_alloc,
        "peak_reserved_mb": peak_res,
    }


def measure_engine_memory(
    samples,
    device="cuda",
    dtype=torch.float16,
    batch_size=4,
    num_enc_blocks=6144,
    num_dec_blocks=1024,
    label="engine_default_pools",
) -> Dict[str, Any]:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    vram_init = torch.cuda.memory_allocated() / (1024 * 1024)

    # 1. Model load
    model = PretrainedWhisperModel("openai/whisper-base", device=device, dtype=dtype)
    torch.cuda.synchronize()
    vram_after_model = torch.cuda.memory_allocated() / (1024 * 1024)
    model_weights_mb = vram_after_model - vram_init

    # 2. Block managers & pools
    enc_bm = EncoderBlockManager(
        num_layers=model.num_encoder_layers,
        num_heads=model.num_heads,
        head_dim=model.head_dim,
        block_size=16,
        num_blocks=num_enc_blocks,
        dtype=dtype,
        device=device,
    )
    dec_bm = BlockManager(
        num_layers=model.num_decoder_layers,
        num_kv_heads=model.num_heads,
        head_dim=model.head_dim,
        block_size=16,
        num_blocks=num_dec_blocks,
        dtype=dtype,
        device=device,
    )
    cache_mgr = DualCacheManager(enc_bm, dec_bm)
    torch.cuda.synchronize()
    vram_after_pools = torch.cuda.memory_allocated() / (1024 * 1024)
    pool_allocated_mb = vram_after_pools - vram_after_model

    from speech_serving.phase_aware_scheduler import MemoryGuard
    scheduler = PhaseAwareSpeechScheduler(
        model=model,
        dual_cache_manager=cache_mgr,
        memory_guard=MemoryGuard(high_water_mark_pct=95.0),
        max_batch_size=batch_size,
        retain_encoder_cache=False,
        cache_cross_kv=True,
        eos_token_id=WHISPER_EOS_TOKEN_ID,
        device=device,
    )

    torch.cuda.reset_peak_memory_stats()

    # Submit 4 requests at once
    prompt_tensor = torch.tensor([WHISPER_DEFAULT_PREFIX_TOKENS], device=device, dtype=torch.long)
    for i in range(batch_size):
        scheduler.submit(
            mel_features=samples[i]["mel"].to(device),
            prompt_token_ids=prompt_tensor,
            max_new_tokens=32,
            arrival_time=0.0,
            eos_token_id=WHISPER_EOS_TOKEN_ID,
        )

    peak_cross_kv_mb = 0.0
    # Step through until requests are active and decode runs
    while scheduler.has_work():
        scheduler.step()
        # Measure cross-KV while requests are actively in decode
        if scheduler.active_decode:
            curr_cross = 0.0
            for r in scheduler.active_decode:
                if r.cached_encoder_kv:
                    curr_cross += sum(k.nbytes + v.nbytes for k, v in r.cached_encoder_kv) / (1024 * 1024)
            if scheduler._cached_batched_enc_kv:
                curr_cross += sum(k.nbytes + v.nbytes for k, v in scheduler._cached_batched_enc_kv) / (1024 * 1024)
            peak_cross_kv_mb = max(peak_cross_kv_mb, curr_cross)

    torch.cuda.synchronize()
    peak_alloc = torch.cuda.max_memory_allocated() / (1024 * 1024)
    peak_res = torch.cuda.max_memory_reserved() / (1024 * 1024)
    pool_used_mb = scheduler.peak_actually_used_bytes / (1024 * 1024)

    # Activations = peak_alloc - pools - weights - cross_kv
    activations_mb = max(0.0, peak_alloc - vram_after_pools - peak_cross_kv_mb)

    del scheduler, cache_mgr, enc_bm, dec_bm, model
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "system": label,
        "model_weights_mb": model_weights_mb,
        "block_pool_allocated_mb": pool_allocated_mb,
        "block_pool_used_mb": pool_used_mb,
        "contiguous_cross_kv_mb": peak_cross_kv_mb,
        "activations_mb": activations_mb,
        "peak_allocated_mb": peak_alloc,
        "peak_reserved_mb": peak_res,
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16

    print("Loading 4 sample audios for memory benchmark...")
    processor = WhisperProcessor.from_pretrained("openai/whisper-base")
    with open("data/librispeech_subset_100.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    samples = []
    for item in metadata[:4]:
        audio, sr = sf.read(item["path"])
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        mel = processor(audio, sampling_rate=16000, return_tensors="pt").input_features[0].to(dtype=dtype)
        samples.append({"mel": mel})

    print("\n--- 1. Measuring Static Batching Memory (B=4) ---")
    sb_mem = measure_static_batching_memory(samples, device=device, dtype=dtype, batch_size=4)
    print(f"Static Batching Peak Alloc: {sb_mem['peak_allocated_mb']:.1f} MB | Res: {sb_mem['peak_reserved_mb']:.1f} MB")

    print("\n--- 2. Measuring Engine Memory (Default Pools: 6144 enc, 1024 dec) ---")
    eng_def_mem = measure_engine_memory(
        samples, device=device, dtype=dtype, batch_size=4,
        num_enc_blocks=6144, num_dec_blocks=1024,
        label="engine (default pools: 6144 enc, 1024 dec)"
    )
    print(f"Engine Default Peak Alloc: {eng_def_mem['peak_allocated_mb']:.1f} MB | Res: {eng_def_mem['peak_reserved_mb']:.1f} MB | Pool Used: {eng_def_mem['block_pool_used_mb']:.1f} MB")

    print("\n--- 3. Measuring Engine Memory (Pools Right-Sized to Cap: 414 enc, 256 dec) ---")
    # Cap math: 94 blocks/req * 4 = 376 blocks * 1.10 margin = 414 encoder blocks
    # Decoder: 10 blocks/req * 4 = 40 blocks, sized to 256 blocks with generous margin
    eng_cap_mem = measure_engine_memory(
        samples, device=device, dtype=dtype, batch_size=4,
        num_enc_blocks=414, num_dec_blocks=256,
        label="engine (pools right-sized to cap: 414 enc, 256 dec)"
    )
    print(f"Engine Right-Sized Peak Alloc: {eng_cap_mem['peak_allocated_mb']:.1f} MB | Res: {eng_cap_mem['peak_reserved_mb']:.1f} MB | Pool Used: {eng_cap_mem['block_pool_used_mb']:.1f} MB")

    mem_results = {
        "static_batching": sb_mem,
        "engine_default": eng_def_mem,
        "engine_right_sized": eng_cap_mem,
    }

    # Print Markdown table
    print("\n==========================================================================================")
    print("MEMORY BREAKDOWN COMPARISON TABLE (Peak VRAM Components in MB)")
    print("==========================================================================================")
    print("| System | Model Weights | Activations | Block Pool Allocated | Block Pool Actually Used | Contiguous Cross-KV Copies | Peak Allocated | Peak Reserved |")
    print("|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")

    for k in ["static_batching", "engine_default", "engine_right_sized"]:
        row = mem_results[k]
        sys_lbl = row["system"]
        pool_alloc_str = f"{row['block_pool_allocated_mb']:.1f} MB" if row['block_pool_allocated_mb'] > 0 else "not measured (0)"
        pool_used_str = f"{row['block_pool_used_mb']:.1f} MB" if row['block_pool_used_mb'] > 0 else "not measured (0)"
        print(f"| {sys_lbl} | {row['model_weights_mb']:.1f} MB | {row['activations_mb']:.1f} MB | {pool_alloc_str} | {pool_used_str} | {row['contiguous_cross_kv_mb']:.1f} MB | **{row['peak_allocated_mb']:.1f} MB** | {row['peak_reserved_mb']:.1f} MB |")

    # Output to JSON
    out_path = Path("data/memory_breakdown_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(mem_results, f, indent=2)
    print(f"\nMemory breakdown saved to {out_path}")


if __name__ == "__main__":
    main()
