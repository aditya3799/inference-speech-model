import os
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import time
import torch
from speech_serving.models import PretrainedWhisperModel
from speech_serving.phase_aware_scheduler import PhaseAwareSpeechScheduler, WHISPER_DEFAULT_PREFIX_TOKENS

def test_static_cuda_graph():
    print("Testing Static Batching CUDA Graph Capture...")
    model_obj = PretrainedWhisperModel("openai/whisper-base", device="cuda", dtype=torch.float16)
    decoder = model_obj.hf_model.model.decoder

    batch_size = 4
    enc_hidden = torch.randn(batch_size, 1500, 512, device="cuda", dtype=torch.float16)
    input_ids = torch.tensor([[50258]] * batch_size, device="cuda", dtype=torch.long)

    # Initial forward to build past_key_values structure
    with torch.no_grad():
        out = decoder(input_ids=input_ids, encoder_hidden_states=enc_hidden, use_cache=True)
        past_kv = out.past_key_values

    # Step input
    step_input_ids = torch.tensor([[50259]] * batch_size, device="cuda", dtype=torch.long)

    # Warmup stream
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(5):
            _ = decoder(input_ids=step_input_ids, encoder_hidden_states=enc_hidden, past_key_values=past_kv, use_cache=True)
    torch.cuda.current_stream().wait_stream(s)

    g = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(g, stream=s):
            static_out = decoder(input_ids=step_input_ids, encoder_hidden_states=enc_hidden, past_key_values=past_kv, use_cache=True)
        print("  [SUCCESS] Static Batching decode step captured in CUDA Graph!")
        
        # Benchmark without graph
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(100):
            _ = decoder(input_ids=step_input_ids, encoder_hidden_states=enc_hidden, past_key_values=past_kv, use_cache=True)
        torch.cuda.synchronize()
        t_without = (time.perf_counter() - t0) / 100 * 1000

        # Benchmark with graph replay
        for _ in range(20):
            g.replay()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(100):
            g.replay()
        torch.cuda.synchronize()
        t_with = (time.perf_counter() - t0) / 100 * 1000

        print(f"  Step time WITHOUT graph: {t_without:.3f} ms | WITH graph: {t_with:.3f} ms (Speedup: {t_without/t_with:.2f}x)")
        return True, None, t_without, t_with
    except Exception as e:
        print("  [FAILED] Static Batching graph capture failed:", e)
        return False, str(e), 0.0, 0.0

def test_engine_cuda_graph():
    print("\nTesting Engine Baseline CUDA Graph Capture...")
    from speech_serving.encoder_block_manager import EncoderBlockManager
    from paged_kv_cache.block_manager import BlockManager
    from speech_serving.dual_cache_manager import DualCacheManager
    from speech_serving.phase_aware_scheduler import PhaseAwareSpeechScheduler, WHISPER_DEFAULT_PREFIX_TOKENS, WHISPER_EOS_TOKEN_ID
    from speech_serving.models import PretrainedWhisperModel

    model = PretrainedWhisperModel("openai/whisper-base", device="cuda", dtype=torch.float16)

    enc_bm = EncoderBlockManager(
        num_layers=model.num_encoder_layers,
        num_heads=model.num_heads,
        head_dim=model.head_dim,
        block_size=16,
        num_blocks=1024,
        dtype=torch.float16,
        device="cuda",
    )
    dec_bm = BlockManager(
        num_layers=model.num_decoder_layers,
        num_kv_heads=model.num_heads,
        head_dim=model.head_dim,
        block_size=16,
        num_blocks=1024,
        dtype=torch.float16,
        device="cuda",
    )
    cache_mgr = DualCacheManager(enc_bm, dec_bm)
    scheduler = PhaseAwareSpeechScheduler(
        model=model,
        dual_cache_manager=cache_mgr,
        max_batch_size=4,
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
    for i in range(4):
        scheduler.submit(
            mel_features=features,
            prompt_token_ids=prompt,
            max_new_tokens=100,
            arrival_time=0.0,
        )

    # Step until all 4 requests are in decode phase
    for _ in range(10):
        scheduler.step()
        if len(scheduler.active_decode) == 4:
            break

    print(f"  Engine state: {len(scheduler.active_decode)} requests in active_decode.")

    # Try capturing a single decode step in a CUDA graph
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    g = torch.cuda.CUDAGraph()
    blockers = []
    try:
        # Warmup on stream
        with torch.cuda.stream(s):
            for _ in range(3):
                scheduler.step()
        torch.cuda.current_stream().wait_stream(s)

        with torch.cuda.graph(g, stream=s):
            scheduler.step()
        print("  [SUCCESS] Engine decode step captured in CUDA Graph!")
    except Exception as e:
        import traceback
        err_msg = str(e)
        blockers.append(err_msg)
        print("  [BLOCKED] Engine decode step could not be captured:")
        print(f"    Error: {err_msg}")
        traceback.print_exc()

    return blockers

if __name__ == "__main__":
    test_static_cuda_graph()
    test_engine_cuda_graph()
