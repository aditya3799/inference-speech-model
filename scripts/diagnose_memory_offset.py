"""
scripts/diagnose_memory_offset.py

Diagnoses the exact ~8 MB offset between component sum and max_memory_allocated.
Lists every active tensor in memory at peak decode and inspects PyTorch allocator stats.
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

import gc
import json
import torch
import soundfile as sf
from transformers import WhisperProcessor

from speech_serving.models import PretrainedWhisperModel
from speech_serving.encoder_block_manager import EncoderBlockManager
from speech_serving.dual_cache_manager import DualCacheManager
from speech_serving.phase_aware_scheduler import (
    PhaseAwareSpeechScheduler,
    WHISPER_DEFAULT_PREFIX_TOKENS,
    WHISPER_EOS_TOKEN_ID,
)
from paged_kv_cache.block_manager import BlockManager


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16

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

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Load Model
    model = PretrainedWhisperModel("openai/whisper-base", device=device, dtype=dtype)
    torch.cuda.synchronize()
    vram_model = torch.cuda.memory_allocated() / (1024 * 1024)

    # Right-sized pools (414 enc, 256 dec)
    enc_bm = EncoderBlockManager(num_layers=model.num_encoder_layers, num_heads=model.num_heads, head_dim=model.head_dim, block_size=16, num_blocks=414, dtype=dtype, device=device)
    dec_bm = BlockManager(num_layers=model.num_decoder_layers, num_kv_heads=model.num_heads, head_dim=model.head_dim, block_size=16, num_blocks=256, dtype=dtype, device=device)
    cache_mgr = DualCacheManager(enc_bm, dec_bm)
    torch.cuda.synchronize()
    vram_pools = torch.cuda.memory_allocated() / (1024 * 1024)
    pools_alloc_mb = vram_pools - vram_model

    scheduler = PhaseAwareSpeechScheduler(
        model=model,
        dual_cache_manager=cache_mgr,
        max_batch_size=4,
        retain_encoder_cache=False,
        cache_cross_kv=True,
        eos_token_id=WHISPER_EOS_TOKEN_ID,
        device=device,
    )

    prompt_tensor = torch.tensor([WHISPER_DEFAULT_PREFIX_TOKENS], device=device, dtype=torch.long)
    for i in range(4):
        scheduler.submit(mel_features=samples[i]["mel"].to(device), prompt_token_ids=prompt_tensor, max_new_tokens=32, arrival_time=0.0)

    # Run step by step and track memory stats
    step_num = 0
    while scheduler.has_work() and step_num < 5:
        scheduler.step()
        step_num += 1

    torch.cuda.synchronize()
    peak_alloc = torch.cuda.max_memory_allocated() / (1024 * 1024)
    curr_alloc = torch.cuda.memory_allocated() / (1024 * 1024)
    stats = torch.cuda.memory_stats()

    # Find all tensors in gc
    tensors = []
    total_tensor_bytes = 0
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) and obj.is_cuda:
                tensors.append({
                    "shape": list(obj.shape),
                    "dtype": str(obj.dtype),
                    "bytes": obj.nelement() * obj.element_size(),
                    "mb": (obj.nelement() * obj.element_size()) / (1024 * 1024),
                })
                total_tensor_bytes += obj.nelement() * obj.element_size()
        except Exception:
            pass

    tensors.sort(key=lambda x: x["bytes"], reverse=True)
    tensor_mb_sum = total_tensor_bytes / (1024 * 1024)

    print(f"Current VRAM Allocated: {curr_alloc:.2f} MB")
    print(f"Peak VRAM Allocated: {peak_alloc:.2f} MB")
    print(f"Sum of all live PyTorch CUDA tensors: {tensor_mb_sum:.2f} MB")
    print(f"Offset (Peak Allocated - Current Live): {peak_alloc - curr_alloc:.2f} MB")
    print(f"Offset (Current Allocated - Tensor Bytes): {curr_alloc - tensor_mb_sum:.2f} MB")

    # Group tensors by size
    large_tensors = [t for t in tensors if t["mb"] >= 1.0]
    small_tensors = [t for t in tensors if t["mb"] < 1.0]
    small_sum_mb = sum(t["mb"] for t in small_tensors)
    print(f"\nLarge Tensors count: {len(large_tensors)}")
    print(f"Small Tensors count: {len(small_tensors)}, total sum: {small_sum_mb:.2f} MB")
    for lt in large_tensors[:15]:
        print(f"  - Shape: {lt['shape']}, dtype: {lt['dtype']}, size: {lt['mb']:.2f} MB")

    # Inspect PyTorch allocator rounding overhead
    # PyTorch allocator rounds every allocation to multiples of 512 bytes
    allocated_bytes = stats.get("allocated_bytes.all.current", 0)
    overshoot_bytes = stats.get("oversize_allocations.all.allocated", 0)
    print(f"\nAllocator Stats:")
    print(f"  allocated_bytes: {allocated_bytes / (1024*1024):.2f} MB")
    print(f"  active_bytes: {stats.get('active_bytes.all.current', 0) / (1024*1024):.2f} MB")
    print(f"  num_alloc_retries: {stats.get('num_alloc_retries', 0)}")


if __name__ == "__main__":
    main()
