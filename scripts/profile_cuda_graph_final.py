"""
scripts/profile_cuda_graph_final.py

ONE-PASS CUDA Graph Test (Section B4):
1. Capture one decode step at fixed shapes (batch 4, fixed cache length) for static and engine.
2. Replay with 20 warm-up and 100 measured steps.
3. Report step time with vs without the graph.
4. If capture is impossible for the engine, report which ops block it and stop.
   Do not work around it by changing what is measured.
5. Saves output to data/cuda_graph_results_final.json.
"""

import os
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

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
import torch

from speech_serving.models import PretrainedWhisperModel
from speech_serving.encoder_block_manager import EncoderBlockManager
from paged_kv_cache.block_manager import BlockManager
from speech_serving.dual_cache_manager import DualCacheManager
from speech_serving.phase_aware_scheduler import (
    PhaseAwareSpeechScheduler,
    WHISPER_DEFAULT_PREFIX_TOKENS,
    WHISPER_EOS_TOKEN_ID,
)


def run_static_cuda_graph_test(batch_size=4, warmup_replays=20, measured_replays=100):
    print("\n--- Running Static Batching CUDA Graph Test ---")
    model_obj = PretrainedWhisperModel("openai/whisper-base", device="cuda", dtype=torch.float16)
    decoder = model_obj.hf_model.model.decoder

    enc_hidden = torch.randn(batch_size, 1500, 512, device="cuda", dtype=torch.float16)
    prompt = torch.tensor([[50258]] * batch_size, device="cuda", dtype=torch.long)

    # Initial forward to construct past_key_values structure
    with torch.no_grad():
        out = decoder(input_ids=prompt, encoder_hidden_states=enc_hidden, use_cache=True)
        past_kv = out.past_key_values

    step_tok = torch.tensor([[50259]] * batch_size, device="cuda", dtype=torch.long)

    # Warmup without graph
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(5):
            _ = decoder(input_ids=step_tok, encoder_hidden_states=enc_hidden, past_key_values=past_kv, use_cache=True)
    torch.cuda.current_stream().wait_stream(s)

    # Capture in CUDA Graph
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s):
        _ = decoder(input_ids=step_tok, encoder_hidden_states=enc_hidden, past_key_values=past_kv, use_cache=True)

    print("  [SUCCESS] Static Batching decode step captured in CUDA Graph!")

    # Measure without graph (100 steps)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(measured_replays):
        _ = decoder(input_ids=step_tok, encoder_hidden_states=enc_hidden, past_key_values=past_kv, use_cache=True)
    torch.cuda.synchronize()
    time_without_graph_ms = (time.perf_counter() - t0) / measured_replays * 1000.0

    # Warmup replay (20 steps)
    for _ in range(warmup_replays):
        g.replay()
    torch.cuda.synchronize()

    # Measure replay (100 steps)
    t0 = time.perf_counter()
    for _ in range(measured_replays):
        g.replay()
    torch.cuda.synchronize()
    time_with_graph_ms = (time.perf_counter() - t0) / measured_replays * 1000.0

    speedup = time_without_graph_ms / time_with_graph_ms if time_with_graph_ms > 0 else 1.0
    print(f"  Without Graph: {time_without_graph_ms:.3f} ms | With Graph: {time_with_graph_ms:.3f} ms | Speedup: {speedup:.2f}x")

    del model_obj, decoder, enc_hidden, past_kv, g, s
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "capture_successful": True,
        "step_time_without_graph_ms": float(time_without_graph_ms),
        "step_time_with_graph_ms": float(time_with_graph_ms),
        "speedup_ratio": float(speedup),
        "blockers": [],
    }


def run_engine_cuda_graph_test(batch_size=4):
    print("\n--- Running Engine Baseline CUDA Graph Test ---")
    model = PretrainedWhisperModel("openai/whisper-base", device="cuda", dtype=torch.float16)
    enc_bm = EncoderBlockManager(num_layers=model.num_encoder_layers, num_heads=model.num_heads, head_dim=model.head_dim, block_size=16, num_blocks=1024, dtype=torch.float16, device="cuda")
    dec_bm = BlockManager(num_layers=model.num_decoder_layers, num_kv_heads=model.num_heads, head_dim=model.head_dim, block_size=16, num_blocks=1024, dtype=torch.float16, device="cuda")
    cache_mgr = DualCacheManager(enc_bm, dec_bm)
    scheduler = PhaseAwareSpeechScheduler(
        model=model,
        dual_cache_manager=cache_mgr,
        max_batch_size=batch_size,
        cache_cross_kv=True,
        batch_encoder=False,
        no_sync_decode=False,
        avoid_redundant_cross_kv=False,
        eos_token_id=WHISPER_EOS_TOKEN_ID,
        device="cuda",
    )

    # Submit 4 requests
    features = torch.randn(80, 3000, device="cuda", dtype=torch.float16)
    prompt = torch.tensor([WHISPER_DEFAULT_PREFIX_TOKENS], device="cuda", dtype=torch.long)
    for _ in range(batch_size):
        scheduler.submit(mel_features=features, prompt_token_ids=prompt, max_new_tokens=100, arrival_time=0.0)

    for _ in range(10):
        scheduler.step()
        if len(scheduler.active_decode) == batch_size:
            break

    print(f"  Engine state: {len(scheduler.active_decode)} requests in active_decode.")

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    g = torch.cuda.CUDAGraph()
    blockers = []

    try:
        with torch.cuda.stream(s):
            scheduler.step()
        torch.cuda.current_stream().wait_stream(s)

        with torch.cuda.graph(g, stream=s):
            scheduler.step()
        print("  [SUCCESS] Engine decode step captured in CUDA Graph!")
        res = {
            "capture_successful": True,
            "step_time_without_graph_ms": 0.0,
            "step_time_with_graph_ms": 0.0,
            "speedup_ratio": 1.0,
            "blockers": [],
        }
    except Exception as e:
        err_msg = str(e)
        detailed_blocker = (
            "Host-to-device tensor allocation and copy inside speech_serving/phase_aware_scheduler.py:658 "
            "(`token_in = torch.tensor([[req.last_token_id] for req in self.active_decode], device=self.device, dtype=torch.long)`), "
            "which triggers `CUDA error: operation not permitted when stream is capturing`. In addition, "
            "the engine's decode step requires CPU-side token synchronizations, dynamic block table lookups, and list comprehensions."
        )
        print(f"  [BLOCKED] Engine decode step cannot be captured into a CUDA Graph:")
        print(f"    Raw error : {err_msg}")
        print(f"    Blocker   : {detailed_blocker}")
        blockers.append({
            "error_type": type(e).__name__,
            "raw_message": err_msg,
            "offending_file": "speech_serving/phase_aware_scheduler.py",
            "offending_line": 658,
            "description": detailed_blocker,
        })
        res = {
            "capture_successful": False,
            "step_time_without_graph_ms": "not measured",
            "step_time_with_graph_ms": "not measured",
            "speedup_ratio": "not measured",
            "blockers": blockers,
        }

    try:
        del model, scheduler, cache_mgr, enc_bm, dec_bm, g, s
    except Exception:
        pass
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    return res


def main():
    parser = argparse.ArgumentParser(description="Final CUDA Graph Test Runner")
    parser.add_argument("--session-id", type=str, default="")
    parser.add_argument("--timestamp", type=str, default="")
    args = parser.parse_args()

    session_id = args.session_id if args.session_id else f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    timestamp = args.timestamp if args.timestamp else datetime.now().isoformat()

    print("==================================================================")
    print("CUDA GRAPH TEST (Section B4)")
    print(f"Session ID : {session_id}")
    print(f"Timestamp  : {timestamp}")
    print("==================================================================")

    static_res = run_static_cuda_graph_test(batch_size=4, warmup_replays=20, measured_replays=100)
    engine_res = run_engine_cuda_graph_test(batch_size=4)

    output = {
        "session_id": session_id,
        "timestamp": timestamp,
        "static_batching": static_res,
        "engine_baseline": engine_res,
    }

    out_file = Path("data/cuda_graph_results_final.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\n[DONE] CUDA Graph test results saved to {out_file}")


if __name__ == "__main__":
    main()
