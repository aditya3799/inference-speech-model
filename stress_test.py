"""
GPU Memory Stress Test Suite & Prefix Deduplication Test for Dual-Cache Speech Engine.

Features:
1. run_memory_stress_test(): Pure memory pressure stress test using unique audio tensors
   (no deduplication offset) to force high-water mark contention, triggering bulk evictions and/or
   backpressure rejections with zero memory leaks.
2. run_prefix_dedup_test(): Realistic audio prefix deduplication test using distinct repeated audio clips
   with retain_encoder_cache=True to measure true prefix cache hit rates under plausible production traffic.
"""

import sys
import time
import torch

from speech_serving.models import WhisperClassSpeechEncoderDecoderModel
from speech_serving.encoder_block_manager import EncoderBlockManager
from speech_serving.dual_cache_manager import DualCacheManager
from speech_serving.phase_aware_scheduler import PhaseAwareSpeechScheduler
from speech_serving.memory_guard import MemoryGuard
from paged_kv_cache.block_manager import BlockManager


def run_memory_stress_test(
    num_requests: int = 15,
    max_blocks_pool: int = 100, # Pool capacity (100 blocks) allowing 3-4 concurrent requests before 75% HWM contention
    high_water_mark_pct: float = 75.0,
    step_delay: float = 0.01,
):
    print("=" * 80)
    print("       PAGED DUAL-CACHE SPEECH ENGINE — GPU MEMORY STRESS TEST")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    print(f"Device: {device} | High-Water Mark: {high_water_mark_pct}% | Capacity Pool: {max_blocks_pool} blocks")
    print(f"Submitting {num_requests} unique long audio requests (160 mel frames, 30 decode tokens) with step_delay={step_delay}s...\n")

    num_enc_layers = 2
    num_dec_layers = 2
    num_heads = 4
    hidden_dim = 64
    mel_dim = 80
    vocab_size = 256
    block_size = 8

    model = WhisperClassSpeechEncoderDecoderModel(
        num_encoder_layers=num_enc_layers,
        num_decoder_layers=num_dec_layers,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        mel_dim=mel_dim,
        vocab_size=vocab_size,
    ).to(device=device, dtype=dtype)
    model.eval()

    enc_bm = EncoderBlockManager(
        num_layers=num_enc_layers,
        num_heads=num_heads,
        head_dim=hidden_dim // num_heads,
        block_size=block_size,
        num_blocks=max_blocks_pool,
        dtype=dtype,
        device=device,
    )

    dec_bm = BlockManager(
        num_layers=num_dec_layers,
        num_kv_heads=num_heads,
        head_dim=hidden_dim // num_heads,
        block_size=block_size,
        num_blocks=max_blocks_pool,
        dtype=dtype,
        device=device,
    )

    dual_cache = DualCacheManager(enc_bm, dec_bm)
    guard = MemoryGuard(high_water_mark_pct=high_water_mark_pct)

    scheduler = PhaseAwareSpeechScheduler(
        model=model,
        dual_cache_manager=dual_cache,
        memory_guard=guard,
        max_batch_size=32,
        retain_encoder_cache=True, # Retain LRU encoder caches to test high-water mark bulk eviction
        step_delay=step_delay,
        verbose=True,
        device=device,
    )

    submitted_ids = []
    rejected_count = 0

    start_bench_time = time.perf_counter()

    # 1. Submit burst wave of ALL-UNIQUE random tensors (no deduplication offset)
    for i in range(num_requests):
        # Unique 160 mel frames (requires 20 encoder blocks per request)
        mel = torch.randn(mel_dim, 160, device=device, dtype=dtype)
        req_id = scheduler.submit(mel, max_new_tokens=30)
        
        enc_u = enc_bm.get_stats()["utilization_pct"]
        dec_u = dec_bm.get_stats()["utilization_pct"]

        if req_id is not None:
            submitted_ids.append(req_id)
            print(f"  [ADMITTED] Req #{i:<2} (ID={req_id:<2}) | Pool Util: Enc={enc_u:.1f}%, Dec={dec_u:.1f}%")
        else:
            rejected_count += 1
            print(f"  [BACKPRESSURE REJECTED] Req #{i:<2} | Pool Util: Enc={enc_u:.1f}%, Dec={dec_u:.1f}% (High-Water Mark Triggered!)")

    print(f"\nBurst Submission Complete: {len(submitted_ids)} Admitted, {rejected_count} Backpressured.\n")

    # 2. Step through scheduler loop under sustained in-flight load
    max_steps = 2000
    step_count = 0
    peak_enc_util = 0.0
    peak_dec_util = 0.0

    while scheduler.step() and step_count < max_steps:
        step_count += 1
        e_stat = enc_bm.get_stats()
        d_stat = dec_bm.get_stats()
        in_flight = len(scheduler.active_encode) + len(scheduler.active_decode)

        if e_stat["utilization_pct"] > peak_enc_util:
            peak_enc_util = e_stat["utilization_pct"]
        if d_stat["utilization_pct"] > peak_dec_util:
            peak_dec_util = d_stat["utilization_pct"]

        if step_count % 5 == 0 or guard.eviction_events_count > 0:
            print(f"  [STEP {step_count:<3}] In-Flight Reqs: {in_flight} (Enc: {len(scheduler.active_encode)}, Dec: {len(scheduler.active_decode)}) | Pool Util: Enc={e_stat['utilization_pct']:.1f}%, Dec={d_stat['utilization_pct']:.1f}%")

    # Flush retained LRU encoder caches to trigger eviction cleanup telemetry
    flushed_blocks = scheduler.flush_retained_caches()

    total_time = time.perf_counter() - start_bench_time
    completed_reqs = scheduler.completed

    print("\n" + "=" * 80)
    print("                      STRESS TEST RESULTS SUMMARY")
    print("=" * 80)
    print(f"Total Requests Submitted     : {num_requests}")
    print(f"Requests Admitted            : {len(submitted_ids)}")
    print(f"Requests Backpressured       : {rejected_count}")
    print(f"Peak Simultaneous In-Flight  : {scheduler.peak_in_flight_requests} requests")
    print(f"Peak Encoder Pool Util       : {peak_enc_util:.1f}%")
    print(f"Peak Decoder Pool Util       : {peak_dec_util:.1f}%")
    print(f"Bulk Eviction Events         : {guard.eviction_events_count}")
    print(f"Total Wall-Clock Time        : {total_time:.2f}s")
    print(f"Final Encoder Allocated      : {enc_bm.allocated_block_count} / {enc_bm.num_blocks}")
    print(f"Final Decoder Allocated      : {dec_bm.allocated_block_count} / {dec_bm.num_blocks}")

    # Strict mechanism engagement assertion (must catch backpressure OR eviction)
    assert rejected_count > 0 or guard.eviction_events_count > 0, \
        "Neither backpressure nor eviction was triggered — high-water mark contention was not genuinely tested!"

    # Strict zero memory leak assertion
    assert enc_bm.allocated_block_count == 0, "Encoder block leak detected!"
    assert dec_bm.allocated_block_count == 0, "Decoder block leak detected!"

    print("\n>>> STRESS TEST PASSED: High-water mark eviction & backpressure engagement verified with ZERO memory leaks! <<<\n")


def run_prefix_dedup_test(num_requests: int = 15):
    """
    Dedicated audio prefix deduplication test using distinct repeated audio clips
    to measure true prefix cache hit rates under realistic production traffic patterns.
    """
    print("=" * 80)
    print("       PAGED DUAL-CACHE SPEECH ENGINE — PREFIX DEDUP & HIT RATE TEST")
    print("=" * 80)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    num_enc_layers = 2
    num_dec_layers = 2
    num_heads = 4
    hidden_dim = 64
    mel_dim = 80
    vocab_size = 256
    block_size = 8

    model = WhisperClassSpeechEncoderDecoderModel(
        num_encoder_layers=num_enc_layers,
        num_decoder_layers=num_dec_layers,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        mel_dim=mel_dim,
        vocab_size=vocab_size,
    ).to(device=device, dtype=dtype)
    model.eval()

    enc_bm = EncoderBlockManager(num_enc_layers, num_heads, hidden_dim // num_heads, block_size, num_blocks=64, dtype=dtype, device=device)
    dec_bm = BlockManager(num_dec_layers, num_heads, hidden_dim // num_heads, block_size, num_blocks=64, dtype=dtype, device=device)
    dual_cache = DualCacheManager(enc_bm, dec_bm)
    guard = MemoryGuard(high_water_mark_pct=90.0)

    scheduler = PhaseAwareSpeechScheduler(
        model=model,
        dual_cache_manager=dual_cache,
        memory_guard=guard,
        max_batch_size=4,
        retain_encoder_cache=True, # Enable retention so completed clips stay cached for subsequent lookups
        device=device,
    )

    # 5 distinct realistic audio clips (e.g. standard audio prompt intros / recurring phrases)
    torch.manual_seed(123)
    audio_clips = [torch.randn(mel_dim, 64, device=device, dtype=dtype) for _ in range(5)]

    print(f"Submitting {num_requests} requests referencing 5 distinct recurring audio clips...\n")

    for i in range(num_requests):
        clip_idx = i % 5 # Rotate through distinct audio clips
        mel = audio_clips[clip_idx]
        scheduler.submit(mel, max_new_tokens=15)

    completed = scheduler.run_until_done()
    scheduler.flush_retained_caches()

    print("\n" + "=" * 80)
    print("                    PREFIX DEDUP TEST RESULTS SUMMARY")
    print("=" * 80)
    print(f"Total Requests Completed     : {len(completed)}")
    print(f"Total Encoder Block Lookups  : {enc_bm.total_lookups}")
    print(f"Encoder Prefix Cache Hits    : {enc_bm.cache_hits}")
    print(f"Encoder Prefix Hit Rate      : {enc_bm.hit_rate}%")
    print(f"Final Encoder Allocated      : {enc_bm.allocated_block_count} / {enc_bm.num_blocks}")
    print(f"Final Decoder Allocated      : {dec_bm.allocated_block_count} / {dec_bm.num_blocks}")

    assert enc_bm.hit_rate > 0.0, "Prefix cache hit rate should be > 0 for repeated audio clips!"
    assert enc_bm.allocated_block_count == 0, "Encoder block leak detected!"
    assert dec_bm.allocated_block_count == 0, "Decoder block leak detected!"

    print("\n>>> PREFIX DEDUP TEST PASSED: Realistic audio prefix cache hit rate verified! <<<\n")


if __name__ == "__main__":
    run_memory_stress_test()
    run_prefix_dedup_test()
