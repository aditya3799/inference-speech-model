import os
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import gc
import json
import logging
from pathlib import Path
import numpy as np
import torch
from transformers.cache_utils import DynamicCache, DynamicLayer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from speech_serving.block_manager import BlockManager
from speech_serving.encoder_block_manager import EncoderBlockManager
from speech_serving.models import PretrainedWhisperModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("gather_profile")


def run_gather_decomposition(output_path: str = "data/step1_gather_decomposition.json"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "CUDA required for precise event timing"

    logger.info("Initializing PretrainedWhisperModel on cuda...")
    model = PretrainedWhisperModel(device="cuda", dtype=torch.float16)
    model.eval()

    num_layers = model.num_decoder_layers
    num_heads = model.num_decoder_kv_heads
    head_dim = model.head_dim
    dtype = model.dtype
    dtype_bytes = 2  # fp16
    bytes_per_token = 2 * num_layers * num_heads * head_dim * dtype_bytes  # 12,288 bytes

    batch_sizes = [1, 4, 8, 16]
    seq_lens = [32, 128]
    block_size = 16
    num_warmup = 10
    num_measured = 50

    results = {
        "metadata": {
            "device": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
            "model": "openai/whisper-base",
            "bytes_per_token_position": bytes_per_token,
            "enc_seq_len": 1500,
        },
        "configurations": {},
    }

    for bsz in batch_sizes:
        for slen in seq_lens:
            cfg_name = f"B{bsz}_L{slen}"
            logger.info("Profiling configuration: Batch Size = %d, Seq Len = %d", bsz, slen)

            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            # 1. Setup Decoder BlockManager (Paged Self-Attention)
            needed_dec_blocks = (slen + block_size - 1) // block_size + 2
            dec_bm = BlockManager(
                num_layers=num_layers,
                num_kv_heads=num_heads,
                head_dim=head_dim,
                block_size=block_size,
                num_blocks=needed_dec_blocks * bsz + 64,
                dtype=dtype,
                device=device,
            )
            dec_req_tables = [[dec_bm.allocate_block() for _ in range(needed_dec_blocks)] for _ in range(bsz)]

            # Pre-fill physical decoder blocks
            for i in range(bsz):
                for s in range(slen):
                    blk_id = dec_req_tables[i][s // block_size]
                    off = s % block_size
                    for l in range(num_layers):
                        dec_bm.key_pool[l][blk_id, off] = torch.randn(num_heads, head_dim, device=device, dtype=dtype)
                        dec_bm.value_pool[l][blk_id, off] = torch.randn(num_heads, head_dim, device=device, dtype=dtype)

            # 2. Setup Encoder BlockManager (Paged Cross-Attention, 1500 positions)
            enc_len = 1500
            needed_enc_blocks = (enc_len + block_size - 1) // block_size
            enc_bm = EncoderBlockManager(
                num_layers=num_layers,
                num_heads=num_heads,
                head_dim=head_dim,
                block_size=block_size,
                num_blocks=needed_enc_blocks * bsz + 64,
                dtype=dtype,
                device=device,
            )
            enc_req_tables = [[enc_bm.allocate_block() for _ in range(needed_enc_blocks)] for _ in range(bsz)]

            # Pre-fill physical encoder blocks
            for i in range(bsz):
                for s in range(enc_len):
                    blk_id = enc_req_tables[i][s // block_size]
                    off = s % block_size
                    for l in range(num_layers):
                        enc_bm.key_pool[l][blk_id, off] = torch.randn(num_heads, head_dim, device=device, dtype=dtype)
                        enc_bm.value_pool[l][blk_id, off] = torch.randn(num_heads, head_dim, device=device, dtype=dtype)

            # Pre-gathered cached cross-attention KV for Step 2 comparison
            cached_enc_kv = []
            for l in range(num_layers):
                ck = torch.cat([
                    enc_bm.key_pool[l][enc_req_tables[i]].view(-1, num_heads, head_dim)[:enc_len].permute(1, 0, 2).unsqueeze(0)
                    for i in range(bsz)
                ], dim=0)
                cv = torch.cat([
                    enc_bm.value_pool[l][enc_req_tables[i]].view(-1, num_heads, head_dim)[:enc_len].permute(1, 0, 2).unsqueeze(0)
                    for i in range(bsz)
                ], dim=0)
                cached_enc_kv.append((ck, cv))

            token_in = torch.tensor([[50258]] * bsz, device=device, dtype=torch.long)

            # 3. Setup CUDA events for sub-layer hook timing
            # We hook self_attn and encoder_attn on each layer to measure pure compute
            layer_self_events = []
            layer_cross_events = []
            for l in range(num_layers):
                layer_self_events.append((torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)))
                layer_cross_events.append((torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)))

            hooks = []
            for l in range(num_layers):
                l_idx = l
                def make_pre_hook_self(idx):
                    def hook(module, args):
                        layer_self_events[idx][0].record()
                    return hook
                def make_post_hook_self(idx):
                    def hook(module, args, output):
                        layer_self_events[idx][1].record()
                    return hook
                def make_pre_hook_cross(idx):
                    def hook(module, args):
                        layer_cross_events[idx][0].record()
                    return hook
                def make_post_hook_cross(idx):
                    def hook(module, args, output):
                        layer_cross_events[idx][1].record()
                    return hook

                h1 = model.hf_model.model.decoder.layers[l].self_attn.register_forward_pre_hook(make_pre_hook_self(l_idx))
                h2 = model.hf_model.model.decoder.layers[l].self_attn.register_forward_hook(make_post_hook_self(l_idx))
                h3 = model.hf_model.model.decoder.layers[l].encoder_attn.register_forward_pre_hook(make_pre_hook_cross(l_idx))
                h4 = model.hf_model.model.decoder.layers[l].encoder_attn.register_forward_hook(make_post_hook_cross(l_idx))
                hooks.extend([h1, h2, h3, h4])

            # Events for step stages
            ev_step_start = torch.cuda.Event(enable_timing=True)
            ev_gather_self_start = torch.cuda.Event(enable_timing=True)
            ev_gather_self_end = torch.cuda.Event(enable_timing=True)
            ev_gather_cross_start = torch.cuda.Event(enable_timing=True)
            ev_gather_cross_end = torch.cuda.Event(enable_timing=True)
            ev_fwd_start = torch.cuda.Event(enable_timing=True)
            ev_fwd_end = torch.cuda.Event(enable_timing=True)
            ev_scatter_start = torch.cuda.Event(enable_timing=True)
            ev_scatter_end = torch.cuda.Event(enable_timing=True)
            ev_step_end = torch.cuda.Event(enable_timing=True)

            # Measurement collectors
            step_times = []
            gather_self_times = []
            gather_cross_times = []
            attn_self_times = []
            attn_cross_times = []
            scatter_times = []
            other_times = []

            # Theoretical bytes copied per step
            # Self-attn gather: reads slen tokens for each request across all layers
            bytes_gather_self = bsz * slen * bytes_per_token
            # Cross-attn gather: reads 1500 tokens for each request across all layers
            bytes_gather_cross = bsz * enc_len * bytes_per_token
            # Scatter: writes 1 new token for each request across all layers
            bytes_scatter_self = bsz * 1 * bytes_per_token

            for step_idx in range(num_warmup + num_measured):
                curr_slen = slen + (step_idx % 8)

                torch.cuda.synchronize()
                ev_step_start.record()

                # --- A1. Gather Self-Attention KV from BlockManager ---
                ev_gather_self_start.record()
                dec_cache = DynamicCache()
                for l in range(num_layers):
                    layer = DynamicLayer()
                    if bsz == 1:
                        blks = dec_req_tables[0][:(curr_slen + block_size - 1) // block_size]
                        k_blks = dec_bm.key_pool[l][blks]
                        v_blks = dec_bm.value_pool[l][blks]
                        layer.keys = k_blks.reshape(-1, num_heads, head_dim)[:curr_slen].permute(1, 0, 2).unsqueeze(0)
                        layer.values = v_blks.reshape(-1, num_heads, head_dim)[:curr_slen].permute(1, 0, 2).unsqueeze(0)
                    else:
                        b_k, b_v = [], []
                        for i in range(bsz):
                            blks = dec_req_tables[i][:(curr_slen + block_size - 1) // block_size]
                            k_blks = dec_bm.key_pool[l][blks]
                            v_blks = dec_bm.value_pool[l][blks]
                            b_k.append(k_blks.reshape(-1, num_heads, head_dim)[:curr_slen].permute(1, 0, 2).unsqueeze(0))
                            b_v.append(v_blks.reshape(-1, num_heads, head_dim)[:curr_slen].permute(1, 0, 2).unsqueeze(0))
                        layer.keys = torch.cat(b_k, dim=0)
                        layer.values = torch.cat(b_v, dim=0)
                    layer.is_initialized = True
                    dec_cache.layers.append(layer)
                ev_gather_self_end.record()

                # --- A2. Gather Cross-Attention KV from EncoderBlockManager ---
                ev_gather_cross_start.record()
                enc_kv_layers = []
                for l in range(num_layers):
                    if bsz == 1:
                        blks = enc_req_tables[0]
                        k_blks = enc_bm.key_pool[l][blks]
                        v_blks = enc_bm.value_pool[l][blks]
                        ek = k_blks.reshape(-1, num_heads, head_dim)[:enc_len].permute(1, 0, 2).unsqueeze(0)
                        ev = v_blks.reshape(-1, num_heads, head_dim)[:enc_len].permute(1, 0, 2).unsqueeze(0)
                    else:
                        b_k, b_v = [], []
                        for i in range(bsz):
                            blks = enc_req_tables[i]
                            k_blks = enc_bm.key_pool[l][blks]
                            v_blks = enc_bm.value_pool[l][blks]
                            b_k.append(k_blks.reshape(-1, num_heads, head_dim)[:enc_len].permute(1, 0, 2).unsqueeze(0))
                            b_v.append(v_blks.reshape(-1, num_heads, head_dim)[:enc_len].permute(1, 0, 2).unsqueeze(0))
                        ek = torch.cat(b_k, dim=0)
                        ev = torch.cat(b_v, dim=0)
                    enc_kv_layers.append((ek, ev))
                ev_gather_cross_end.record()

                # --- B. Forward Pass (Attention Compute + MLPs + LayerNorms + Projections) ---
                ev_fwd_start.record()
                with torch.no_grad():
                    dec_out = model.decode_step(
                        input_ids=token_in,
                        past_key_values=dec_cache,
                        encoder_past_kv_list=enc_kv_layers,
                        use_cache=True,
                    )
                # Next token argmax (sampling)
                next_tok = torch.argmax(dec_out.logits[:, -1, :], dim=-1, keepdim=True)
                ev_fwd_end.record()

                # --- C. Scatter New KV back to physical block pool ---
                ev_scatter_start.record()
                for i in range(bsz):
                    blk_id = dec_req_tables[i][curr_slen // block_size]
                    off = curr_slen % block_size
                    for l in range(num_layers):
                        new_k = dec_out.past_key_values.layers[l].keys[i, :, -1, :]
                        new_v = dec_out.past_key_values.layers[l].values[i, :, -1, :]
                        dec_bm.key_pool[l][blk_id, off] = new_k
                        dec_bm.value_pool[l][blk_id, off] = new_v
                ev_scatter_end.record()

                ev_step_end.record()
                torch.cuda.synchronize()

                if step_idx >= num_warmup:
                    t_step = ev_step_start.elapsed_time(ev_step_end)
                    t_g_self = ev_gather_self_start.elapsed_time(ev_gather_self_end)
                    t_g_cross = ev_gather_cross_start.elapsed_time(ev_gather_cross_end)
                    t_sc = ev_scatter_start.elapsed_time(ev_scatter_end)

                    # Sum hook times across 6 layers
                    t_a_self = sum(layer_self_events[l][0].elapsed_time(layer_self_events[l][1]) for l in range(num_layers))
                    t_a_cross = sum(layer_cross_events[l][0].elapsed_time(layer_cross_events[l][1]) for l in range(num_layers))

                    # Everything else: (t_fwd - attn compute) + argmax + bookkeeping
                    t_fwd = ev_fwd_start.elapsed_time(ev_fwd_end)
                    t_other = (t_fwd - (t_a_self + t_a_cross))

                    step_times.append(t_step)
                    gather_self_times.append(t_g_self)
                    gather_cross_times.append(t_g_cross)
                    attn_self_times.append(t_a_self)
                    attn_cross_times.append(t_a_cross)
                    scatter_times.append(t_sc)
                    other_times.append(t_other)

            for h in hooks:
                h.remove()

            # Clean memory
            del dec_bm, enc_bm, dec_req_tables, enc_req_tables, cached_enc_kv
            gc.collect()
            torch.cuda.empty_cache()

            def stats(arr):
                return {
                    "mean_ms": float(np.mean(arr)),
                    "p95_ms": float(np.percentile(arr, 95)),
                    "std_ms": float(np.std(arr)),
                }

            t_step_st = stats(step_times)
            t_g_self_st = stats(gather_self_times)
            t_g_cross_st = stats(gather_cross_times)
            t_a_self_st = stats(attn_self_times)
            t_a_cross_st = stats(attn_cross_times)
            t_scatter_st = stats(scatter_times)
            t_other_st = stats(other_times)

            total_gather_mean = t_g_self_st["mean_ms"] + t_g_cross_st["mean_ms"]
            total_copy_mean = total_gather_mean + t_scatter_st["mean_ms"]

            gather_fraction = total_gather_mean / t_step_st["mean_ms"]
            total_copy_fraction = total_copy_mean / t_step_st["mean_ms"]
            gather_self_fraction = t_g_self_st["mean_ms"] / t_step_st["mean_ms"]
            gather_cross_fraction = t_g_cross_st["mean_ms"] / t_step_st["mean_ms"]

            cfg_result = {
                "batch_size": bsz,
                "seq_len": slen,
                "total_step_time_ms": t_step_st,
                "self_attention": {
                    "gather_time_ms": t_g_self_st,
                    "compute_time_ms": t_a_self_st,
                    "gather_fraction": gather_self_fraction,
                    "bytes_copied": bytes_gather_self,
                    "bytes_copied_mb": bytes_gather_self / (1024 * 1024),
                },
                "cross_attention": {
                    "gather_time_ms": t_g_cross_st,
                    "compute_time_ms": t_a_cross_st,
                    "gather_fraction": gather_cross_fraction,
                    "bytes_copied": bytes_gather_cross,
                    "bytes_copied_mb": bytes_gather_cross / (1024 * 1024),
                },
                "scatter_write_back": {
                    "scatter_time_ms": t_scatter_st,
                    "scatter_fraction": t_scatter_st["mean_ms"] / t_step_st["mean_ms"],
                    "bytes_copied": bytes_scatter_self,
                    "bytes_copied_mb": bytes_scatter_self / (1024 * 1024),
                },
                "other_compute_and_bookkeeping": {
                    "other_time_ms": t_other_st,
                    "other_fraction": t_other_st["mean_ms"] / t_step_st["mean_ms"],
                },
                "summary": {
                    "total_gather_time_ms": total_gather_mean,
                    "total_gather_fraction": gather_fraction,
                    "total_copy_time_ms": total_copy_mean,
                    "total_copy_fraction": total_copy_fraction,
                    "pure_attention_compute_time_ms": t_a_self_st["mean_ms"] + t_a_cross_st["mean_ms"],
                    "pure_attention_fraction": (t_a_self_st["mean_ms"] + t_a_cross_st["mean_ms"]) / t_step_st["mean_ms"],
                },
            }

            results["configurations"][cfg_name] = cfg_result
            logger.info(
                "  [%s] Total=%.2fms | GatherSelf=%.2fms (%.1f%%) | GatherCross=%.2fms (%.1f%%) | AttnCompute=%.2fms (%.1f%%) | Other=%.2fms (%.1f%%) | GatherFraction=%.1f%%",
                cfg_name,
                t_step_st["mean_ms"],
                t_g_self_st["mean_ms"], gather_self_fraction * 100.0,
                t_g_cross_st["mean_ms"], gather_cross_fraction * 100.0,
                t_a_self_st["mean_ms"] + t_a_cross_st["mean_ms"], ((t_a_self_st["mean_ms"] + t_a_cross_st["mean_ms"]) / t_step_st["mean_ms"]) * 100.0,
                t_other_st["mean_ms"], (t_other_st["mean_ms"] / t_step_st["mean_ms"]) * 100.0,
                gather_fraction * 100.0,
            )

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info("Saved complete gather decomposition to %s", out_file)
    return results


if __name__ == "__main__":
    run_gather_decomposition()
