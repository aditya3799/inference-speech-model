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
import subprocess
import ctypes
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


def check_session_gate(force_unsafe: bool = False) -> Dict[str, Any]:
    """
    Session gate (Part 0): Verify machine and GPU environment before running benchmarks.
    Logs and checks: plugged-in state, power plan, GPU temp, power draw, graphics clock
    under a 10s warm-up load, throttle reasons, other GPU processes, and pre-load VRAM.
    Aborts if on battery, sw_power_cap or hw_slowdown is active, or clock < 1400 MHz,
    unless force_unsafe is True.
    """
    print("\n==================================================================")
    print("SESSION GATE PRE-FLIGHT CHECKS")
    print("==================================================================")

    # 1. Baseline VRAM before model load
    pre_load_vram_mb = 0.0
    if torch.cuda.is_available():
        pre_load_vram_mb = torch.cuda.memory_allocated() / (1024 * 1024)
        print(f"[GATE] GPU Memory Allocated before model load: {pre_load_vram_mb:.2f} MB")
        if pre_load_vram_mb > 50.0:
            print(f"  [GATE WARNING] Allocated memory > 50 MB ({pre_load_vram_mb:.2f} MB), potential leftover model in process!")

    # 2. AC Line / Plugged-in state
    plugged_in = True
    battery_pct = 100
    try:
        class SYSTEM_POWER_STATUS(ctypes.Structure):
            _fields_ = [
                ("ACLineStatus", ctypes.c_byte),
                ("BatteryFlag", ctypes.c_byte),
                ("BatteryLifePercent", ctypes.c_byte),
                ("SystemStatusFlag", ctypes.c_byte),
                ("BatteryLifeTime", ctypes.c_ulong),
                ("BatteryFullLifeTime", ctypes.c_ulong),
            ]
        sp = SYSTEM_POWER_STATUS()
        if hasattr(ctypes, "windll") and hasattr(ctypes.windll, "kernel32"):
            ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(sp))
            plugged_in = (sp.ACLineStatus != 0)
            battery_pct = int(sp.BatteryLifePercent)
        print(f"[GATE] Power Source: {'AC Wall Power (Plugged In)' if plugged_in else 'BATTERY'} (Battery: {battery_pct}%)")
    except Exception as e:
        print(f"[GATE] Could not determine battery status: {e}")

    # 3. Windows Power Plan
    power_plan = "Unknown"
    try:
        plan_out = subprocess.check_output(["powercfg", "/getactivescheme"], stderr=subprocess.DEVNULL).decode().strip()
        power_plan = plan_out
        print(f"[GATE] Windows Power Plan: {power_plan}")
    except Exception as e:
        print(f"[GATE] Could not query power scheme: {e}")

    # 4. Competing GPU processes
    other_gpu_procs = []
    try:
        apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"], stderr=subprocess.DEVNULL).decode().strip()
        my_pid = str(os.getpid())
        if apps:
            for line in apps.splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3 and parts[0] != my_pid:
                    other_gpu_procs.append({"pid": parts[0], "name": parts[1], "used_memory": parts[2]})
        print(f"[GATE] Other GPU processes detected: {len(other_gpu_procs)}")
        for proc in other_gpu_procs:
            print(f"  - PID {proc['pid']}: {proc['name']} ({proc['used_memory']})")
    except Exception as e:
        print(f"[GATE] Could not query GPU compute apps: {e}")

    # 5. 10s Warm-up Load & GPU Clocks under Load
    print("[GATE] Running 10-second GPU warm-up load to measure clock & throttle state under load...")
    clock_load = 0.0
    power_load = 0.0
    temp_load = 0.0
    throttle_active_load = "0x0"
    sw_power_cap_load = "Not Active"
    hw_slowdown_load = "Not Active"

    if torch.cuda.is_available():
        t_end = time.perf_counter() + 10.0
        mat = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
        while time.perf_counter() < t_end - 0.5:
            _ = torch.matmul(mat, mat)

        try:
            smi_out = subprocess.check_output([
                "nvidia-smi",
                "--query-gpu=clocks.gr,power.draw,temperature.gpu,clocks_throttle_reasons.active,clocks_throttle_reasons.sw_power_cap,clocks_throttle_reasons.hw_slowdown",
                "--format=csv,noheader,nounits"
            ], stderr=subprocess.DEVNULL).decode().strip()
            parts = [p.strip() for p in smi_out.split(",")]
            clock_load = float(parts[0])
            power_load = float(parts[1])
            temp_load = float(parts[2])
            throttle_active_load = parts[3]
            sw_power_cap_load = parts[4]
            hw_slowdown_load = parts[5]
        except Exception as e:
            print(f"[GATE] Could not query nvidia-smi under load: {e}")

        torch.cuda.synchronize()
        del mat
        torch.cuda.empty_cache()

    print(f"[GATE] Under Load: Clock={clock_load:.0f} MHz | Power={power_load:.1f} W | Temp={temp_load:.0f} C")
    print(f"[GATE] Throttles Under Load: SW Power Cap: {sw_power_cap_load} | HW Slowdown: {hw_slowdown_load} | Active: {throttle_active_load}")

    # 6. Evaluate Gate Pass/Fail
    failures = []
    if not plugged_in:
        failures.append(f"Laptop is running on battery ({battery_pct}%). Plug into AC wall power.")
    if "active" in sw_power_cap_load.lower() and "not" not in sw_power_cap_load.lower():
        failures.append(f"GPU SW Power Cap is active ({sw_power_cap_load}).")
    if "active" in hw_slowdown_load.lower() and "not" not in hw_slowdown_load.lower():
        failures.append(f"GPU HW Slowdown is active ({hw_slowdown_load}).")
    if clock_load < 1400.0:
        failures.append(f"Graphics clock under load is {clock_load:.0f} MHz (below 1400 MHz threshold).")

    gate_passed = len(failures) == 0
    if not gate_passed:
        msg = "[GATE ABORT] Session gate checks failed:\n  * " + "\n  * ".join(failures) + "\nAborting to guarantee reproducible performance. Override with --force-unsafe."
        if not force_unsafe:
            print(f"\n{msg}\n")
            raise RuntimeError(msg)
        else:
            print(f"\n[WARNING] Gate checks failed but proceeding due to --force-unsafe:\n  * " + "\n  * ".join(failures))

    unsafe_state = (not gate_passed) or force_unsafe
    print(f"[GATE] Session Gate Status: {'PASSED (SAFE)' if gate_passed else 'OVERRIDDEN (UNSAFE)'}\n")

    return {
        "gate_passed": gate_passed,
        "unsafe_power_state": unsafe_state,
        "plugged_in": plugged_in,
        "battery_pct": battery_pct,
        "power_plan": power_plan,
        "other_gpu_processes": other_gpu_procs,
        "vram_before_load_mb": pre_load_vram_mb,
        "gpu_temp_under_load_c": temp_load,
        "power_draw_under_load_w": power_load,
        "graphics_clock_under_load_mhz": clock_load,
        "sw_power_cap": sw_power_cap_load,
        "hw_slowdown": hw_slowdown_load,
        "throttle_active": throttle_active_load,
    }


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

        metrics = self._compute_metrics(per_request, wall_time, peak_alloc, peak_res)
        metrics["peak_actually_used_mb"] = "not measured"
        metrics["concurrency_stats"] = "not measured"
        return metrics

    def run_static_batching(
        self,
        traffic: List[Dict[str, Any]],
        batch_size: int = 4,
        max_new_tokens: int = 150,
        graph_decode: bool = False,
        graph_readback_k: int = 4,
        phase_timing: bool = False,
    ) -> Dict[str, Any]:
        """System 2: Static batching (fixed batch size, padded decode, waits for slowest)."""
        torch.cuda.reset_peak_memory_stats()
        t_bench_start = sync_time()
        queue = sorted(traffic, key=lambda x: x["arrival_time"])
        per_request = []

        from speech_serving.phase_timing import PhaseTimer, finalize_phases, new_phase_record
        phase_timer = PhaseTimer(phase_timing, device=self.device)
        batch_wall_times_ms = []
        buf = None
        if graph_decode:
            from speech_serving.graph_decode_buffers import GraphDecodeBuffers
            buf = GraphDecodeBuffers(
                model=self.model,
                device=self.device,
                dtype=self.dtype,
                max_slots=batch_size,
                readback_k=graph_readback_k,
            )

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
            request_phase_records = [new_phase_record() for _ in batch_reqs] if phase_timing else [None] * bsz
            if phase_timing:
                for req_meta, phase_record in zip(batch_reqs, request_phase_records):
                    phase_record["encoder_batched"] = bsz > 1
                    queue_ms = max(0.0, t_batch_start - (t_bench_start + req_meta["arrival_time"])) * 1000.0
                    phase_timer.add([phase_record], "queue_admission_wait", queue_ms, 0.0)

            # 1. Encode batch
            b_mels = torch.stack([self.samples[r["sample_idx"]]["mel"] for r in batch_reqs], dim=0).to(self.device)
            with phase_timer.measure(request_phase_records, "encoder_forward"):
                with torch.no_grad():
                    enc_out, enc_kv = self.model.encode(b_mels)

            # 2. Prefill
            prompt = torch.tensor([WHISPER_DEFAULT_PREFIX_TOKENS] * bsz, device=self.device, dtype=torch.long)
            with phase_timer.measure(request_phase_records, "prefill"):
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

            if not graph_decode:
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
            else:
                # 3. CUDA graph batch decode loop using identical graph machinery
                with phase_timer.measure(request_phase_records, "cross_kv_contiguous_or_graph_copy"):
                    for i in range(bsz):
                        dummy_req = type("StaticReq", (), {
                            "id": batch_reqs[i]["req_id"],
                            "max_new_tokens": max_new_tokens,
                            "eos_token_id": WHISPER_EOS_TOKEN_ID,
                            "done": False,
                        })()
                        enc_kv_i = [(enc_kv[l][0][i:i+1], enc_kv[l][1][i:i+1]) for l in range(len(enc_kv))]
                        init_self_kv_i = [
                            (dec_out.past_key_values.layers[l].keys[i:i+1], dec_out.past_key_values.layers[l].values[i:i+1])
                            for l in range(len(dec_out.past_key_values.layers))
                        ]
                        buf.assign_slot(
                            req=dummy_req,
                            enc_kv_layers=enc_kv_i,
                            prompt_len=prompt.shape[1],
                            initial_tok=next_tokens[i],
                            initial_self_kv=init_self_kv_i,
                        )

                step = 1
                eos_seen_step = [step if done else None for done in req_dones]
                while not all(req_dones) and step < max_new_tokens:
                    step += 1
                    active_records = [request_phase_records[i] for i in range(bsz) if not req_dones[i]]
                    with phase_timer.measure(active_records, "host_between_replays"):
                        bucket = buf.prepare_replay()
                    with phase_timer.measure(active_records, "graph_replay"):
                        buf.replay_step(bucket)
                    if buf.should_readback():
                        with phase_timer.measure(active_records, "readback_wait"):
                            sync_res = buf.sync_tokens()
                        t_step = sync_time()
                        with phase_timer.measure(active_records, "host_between_replays"):
                            for slot, toks in sync_res:
                                idx = slot.slot_idx
                                if idx < bsz:
                                    for token_offset, tok in enumerate(toks):
                                        if req_dones[idx]:
                                            break
                                        req_gen_tokens[idx].append(tok)
                                        if tok == WHISPER_EOS_TOKEN_ID or len(req_gen_tokens[idx]) >= max_new_tokens:
                                            req_dones[idx] = True
                                            req_finish_times[idx] = t_step
                                            eos_seen_step[idx] = step - (len(toks) - token_offset - 1)
                                            break

                if buf._step_in_k > 0:
                    active_records = [request_phase_records[i] for i in range(bsz) if not req_dones[i]]
                    with phase_timer.measure(active_records, "readback_wait"):
                        sync_res = buf.sync_tokens()
                    t_step = sync_time()
                    with phase_timer.measure(active_records, "host_between_replays"):
                        for slot, toks in sync_res:
                            idx = slot.slot_idx
                            if idx < bsz:
                                for token_offset, tok in enumerate(toks):
                                    if req_dones[idx]:
                                        break
                                    req_gen_tokens[idx].append(tok)
                                    if tok == WHISPER_EOS_TOKEN_ID or len(req_gen_tokens[idx]) >= max_new_tokens:
                                        req_dones[idx] = True
                                        req_finish_times[idx] = t_step
                                        eos_seen_step[idx] = step - (len(toks) - token_offset - 1)
                                        break

                if phase_timing:
                    for i, phase_record in enumerate(request_phase_records):
                        seen = eos_seen_step[i]
                        phase_record["slot_hold_steps_after_eos"] = max(0, step - seen) if seen is not None else 0

                for i in range(bsz):
                    buf.release_slot(i)

                # Discard tokens generated after EOS to ensure exact parity
                for i in range(bsz):
                    if WHISPER_EOS_TOKEN_ID in req_gen_tokens[i]:
                        eos_pos = req_gen_tokens[i].index(WHISPER_EOS_TOKEN_ID)
                        req_gen_tokens[i] = req_gen_tokens[i][:eos_pos]

            # In static batching, all batch slots are held until the entire batch finishes
            t_batch_finish = sync_time()
            batch_wall_times_ms.append((t_batch_finish - t_batch_start) * 1000.0)
            for i, r in enumerate(batch_reqs):
                s_data = self.samples[r["sample_idx"]]
                arr_time = r["arrival_time"]
                # Finish time is when batch finishes
                fin_time = t_batch_finish
                if phase_timing:
                    service_ms = (t_batch_finish - t_batch_start) * 1000.0
                    measured_ms = sum(
                        value["wall_ms"] for key, value in request_phase_records[i]["phases"].items()
                        if key != "queue_admission_wait" and isinstance(value, dict)
                    )
                    phase_timer.add([request_phase_records[i]], "other_host_between_requests", max(0.0, service_ms - measured_ms), 0.0)
                    finalize_phases(request_phase_records[i])
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
                    "phase_timing": request_phase_records[i] if phase_timing else "not measured",
                })

        t_bench_end = sync_time()
        wall_time = t_bench_end - t_bench_start
        peak_alloc = torch.cuda.max_memory_allocated() / (1024 * 1024)
        peak_res = torch.cuda.max_memory_reserved() / (1024 * 1024)
        gc.collect()
        torch.cuda.empty_cache()

        metrics = self._compute_metrics(per_request, wall_time, peak_alloc, peak_res)
        metrics["peak_actually_used_mb"] = "not measured"
        metrics["concurrency_stats"] = "not measured"
        metrics["batch_wall_times_ms"] = batch_wall_times_ms if phase_timing else "not measured"
        return metrics

    def run_engine(
        self,
        traffic: List[Dict[str, Any]],
        max_batch_size: int = 4,
        max_new_tokens: int = 150,
        retain_encoder_cache: bool = False,
        cache_cross_kv: bool = False,
        batch_encoder: bool = False,
        no_sync_decode: bool = False,
        avoid_redundant_cross_kv: bool = False,
        graph_decode: bool = False,
        graph_readback_k: int = 4,
        graph_non_blocking_copies: bool = True,
        graph_zero_slot_buffers_on_release: bool = False,
        graph_force_bucket: Optional[int] = None,
        phase_timing: bool = False,
    ) -> Dict[str, Any]:
        """Engine (continuous batching + paged dual cache)."""
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
            cache_cross_kv=cache_cross_kv,
            batch_encoder=batch_encoder,
            no_sync_decode=no_sync_decode,
            avoid_redundant_cross_kv=avoid_redundant_cross_kv,
            eos_token_id=WHISPER_EOS_TOKEN_ID,
            max_cached_encoder_inputs=100,
            device=self.device,
            graph_decode=graph_decode,
            graph_readback_k=graph_readback_k,
            graph_non_blocking_copies=graph_non_blocking_copies,
            graph_zero_slot_buffers_on_release=graph_zero_slot_buffers_on_release,
            graph_force_bucket=graph_force_bucket,
            phase_timing=phase_timing,
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
                    arrival_wall_time=t_bench_start + r["arrival_time"],
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
                "slot_id": r_comp.graph_slot_idx,
                "slot_reused": r_comp.graph_slot_reused,
                "graph_token_trace": r_comp.graph_token_trace,
                "graph_extra_slot_steps": r_comp.graph_extra_slot_steps,
                "phase_timing": r_comp.phase_timing if phase_timing else "not measured",
            })

        if phase_timing:
            from speech_serving.phase_timing import finalize_phases
            for request in per_request:
                record = request["phase_timing"]
                service_ms = max(0.0, request["finish_time"] - request["start_time"]) * 1000.0
                measured_ms = sum(
                    value["wall_ms"] for key, value in record["phases"].items()
                    if key != "queue_admission_wait" and isinstance(value, dict)
                )
                scheduler.phase_timer.add([record], "other_host_between_requests", max(0.0, service_ms - measured_ms), 0.0)
                finalize_phases(record)

        metrics = self._compute_metrics(per_request, wall_time, peak_alloc, peak_res)
        metrics["encoder_telemetry"] = {
            "encoder_calls": scheduler.encoder_calls,
            "encoder_calls_skipped": scheduler.encoder_calls_skipped,
            "encode_time_saved_s": scheduler.encode_time_saved_s,
        }
        concurrency_stats = scheduler.get_concurrency_stats()
        metrics["concurrency_stats"] = concurrency_stats
        metrics["peak_actually_used_mb"] = concurrency_stats["peak_actually_used_mb"]
        metrics["batch_wall_times_ms"] = "not measured"
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
    output_path: str = "data/real_speech_evaluation_results_v2.json",
    include_repeated: bool = False,
    bench: Optional[RealSpeechBenchmark] = None,
    force_unsafe: bool = False,
) -> Dict[str, Any]:
    # Part 0: Session gate check before model allocation
    gate_telemetry = check_session_gate(force_unsafe=force_unsafe)

    if bench is None:
        bench = RealSpeechBenchmark(device="cuda", dtype=torch.float16)

    # Canary Run at START (20 utterances, sequential)
    canary_samples_count = min(20, len(bench.samples))
    canary_traffic = [{"req_id": i, "sample_idx": i, "arrival_time": 0.0} for i in range(canary_samples_count)]
    print(f"\n--- Running Sequential Canary ({canary_samples_count} utterances, START of session) ---")
    canary_start_res = bench.run_sequential(canary_traffic)
    canary_start_tp = canary_start_res["throughput_audio_per_sec"]
    print(f"Canary START throughput: {canary_start_tp:.2f}x audio-time")

    # 1. Warm-up run (discarded)
    print("\n--- Running Warm-up Run (5 samples, discarded) ---")
    warm_traffic = [{"req_id": i, "sample_idx": i, "arrival_time": 0.0} for i in range(min(5, len(bench.samples)))]
    bench.run_sequential(warm_traffic)
    bench.run_static_batching(warm_traffic, batch_size=batch_size)
    bench.run_engine(warm_traffic, max_batch_size=batch_size, cache_cross_kv=False)
    bench.run_engine(warm_traffic, max_batch_size=batch_size, cache_cross_kv=True)
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

    start_iso = time.strftime("%Y-%m-%d %H:%M:%S")
    results = {
        "metadata": {
            "gpu_name": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
            "model": "openai/whisper-base",
            "benchmark_branch": "profile-gap",
            "dataset": "LibriSpeech test-clean (100 stratified utterances)",
            "repeats": num_repeats,
            "session_start_time": start_iso,
            "background_load": "idle (dedicated single session)",
            "unsafe_power_state": gate_telemetry.get("unsafe_power_state", False),
            "session_gate": gate_telemetry,
        },
        "systems": {},
    }

    # Systems: 1. sequential, 2. static_batching, 3. engine_flag_off, 4. engine_flag_on
    system_names = ["sequential", "static_batching", "engine_flag_off", "engine_flag_on"]

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
                elif sys_name == "engine_flag_off":
                    run_res = bench.run_engine(traffic_data, max_batch_size=batch_size, retain_encoder_cache=False, cache_cross_kv=False)
                elif sys_name == "engine_flag_on":
                    run_res = bench.run_engine(traffic_data, max_batch_size=batch_size, retain_encoder_cache=False, cache_cross_kv=True)

                used_str = f" | Pool used: {run_res.get('peak_actually_used_mb', 'N/A')} MB" if sys_name.startswith("engine") else ""
                print(f" Done in {run_res['wall_time_s']:.2f}s | Throughput: {run_res['throughput_audio_per_sec']:.2f}x | WER: {run_res['wer_pct']:.2f}% | Lat p50: {run_res['latency_p50_s']:.2f}s{used_str}")
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
            if sys_name.startswith("engine") and "concurrency_stats" in repeat_runs[0]:
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
                    "peak_actually_used_mb_mean": float(np.mean([r["concurrency_stats"]["peak_actually_used_mb"] for r in repeat_runs])),
                    "peak_actually_used_mb_std": float(np.std([r["concurrency_stats"]["peak_actually_used_mb"] for r in repeat_runs])),
                }
                agg["peak_actually_used_mb_mean"] = float(np.mean([r["concurrency_stats"]["peak_actually_used_mb"] for r in repeat_runs]))
                agg["peak_actually_used_mb_std"] = float(np.std([r["concurrency_stats"]["peak_actually_used_mb"] for r in repeat_runs]))
            else:
                agg["concurrency"] = "not measured"
                agg["peak_actually_used_mb_mean"] = "not measured"
                agg["peak_actually_used_mb_std"] = "not measured"

            results["systems"][prof_name][sys_name] = {
                "aggregate": agg,
                "runs": repeat_runs,
            }

    if include_repeated:
        # Repeated Audio Workload (Cache ON vs Cache OFF)
        print("\n==================================================================")
        print("EVALUATING REPEATED-AUDIO WORKLOAD (Cache ON vs Cache OFF)")
        print("==================================================================")
        results["repeated_workload"] = {}
        rep_traffic = generate_repeated_traffic(num_unique=50, seed=42)

        for cache_mode, retain_cache in [("cache_off", False), ("cache_on", True)]:
            print(f"\n--- Repeated Workload: {cache_mode} ({num_repeats} repeats) ---")
            repeat_runs = []
            for rep in range(num_repeats):
                print(f"  [Repeat {rep+1}/{num_repeats}] running {cache_mode}...", end="", flush=True)
                run_res = bench.run_engine(rep_traffic, max_batch_size=batch_size, retain_encoder_cache=retain_cache, cache_cross_kv=False)
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

    # Part 0: Reproducibility check against reference engine numbers
    ref_engine_throughputs = {
        "burst": 38.533,
        "poisson_low": 9.591,
        "poisson_high": 25.176,
    }
    repro_failures = []
    print("\n==================================================================")
    print("PART 0 REPRODUCIBILITY CHECK (Engine Flag OFF vs Reference)")
    print("==================================================================")
    for prof, ref_tp in ref_engine_throughputs.items():
        if prof in results["systems"] and "engine_flag_off" in results["systems"][prof]:
            meas_tp = results["systems"][prof]["engine_flag_off"]["aggregate"]["throughput_mean"]
            delta_pct = abs(meas_tp - ref_tp) / ref_tp * 100.0
            passed = delta_pct <= 15.0
            status_tag = "PASS" if passed else "FAIL"
            print(f"  [{status_tag}] Profile: {prof.upper()} | Measured: {meas_tp:.2f}x | Reference: {ref_tp:.2f}x | Delta: {delta_pct:.1f}%")
            if not passed:
                repro_failures.append({
                    "profile": prof,
                    "measured_throughput": meas_tp,
                    "reference_throughput": ref_tp,
                    "delta_pct": delta_pct,
                })

    if repro_failures:
        results["metadata"]["session_status"] = "not reportable"
        results["metadata"]["reproducibility_failures"] = repro_failures
        print(f"\n[REPRODUCIBILITY WARNING] Session marked 'not reportable' because {len(repro_failures)} profiles deviated by >15% from reference:")
        for f in repro_failures:
            print(f"  - {f['profile']}: {f['measured_throughput']:.2f}x vs {f['reference_throughput']:.2f}x ({f['delta_pct']:.1f}% delta)")
    else:
        results["metadata"]["session_status"] = "reportable"
        results["metadata"]["reproducibility_failures"] = []
        print("\n[REPRODUCIBILITY] All engine flag OFF profiles within 15% of reference.")

    # Part 0: Canary Run at END (20 utterances, sequential)
    print(f"\n--- Running Sequential Canary ({canary_samples_count} utterances, END of session) ---")
    canary_end_res = bench.run_sequential(canary_traffic)
    canary_end_tp = canary_end_res["throughput_audio_per_sec"]
    drift_pct = abs(canary_end_tp - canary_start_tp) / canary_start_tp * 100.0 if canary_start_tp > 0 else 0.0
    print(f"Canary END throughput: {canary_end_tp:.2f}x (Start: {canary_start_tp:.2f}x, Drift: {drift_pct:.1f}%)")
    if drift_pct > 15.0:
        print(f"[CANARY WARNING] Sequential canary throughput drifted by {drift_pct:.1f}% (> 15%) between session start and end!")
    results["metadata"]["canary"] = {
        "start_throughput": canary_start_tp,
        "end_throughput": canary_end_tp,
        "drift_pct": drift_pct,
        "drift_exceeded_15pct": drift_pct > 15.0,
    }

    # Save to JSON
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll benchmark results successfully saved to {out_file}!")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate real speech serving on Whisper-base LibriSpeech")
    parser.add_argument("--repeats", type=int, default=3, help="Number of measured repeats per condition (default: 3)")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of utterances to benchmark (default: 100)")
    parser.add_argument("--batch-size", type=int, default=4, help="Max batch size for batching/engine (default: 4)")
    parser.add_argument("--output", type=str, default="data/real_speech_evaluation_results_v2.json", help="Output JSON path")
    parser.add_argument("--include-repeated", action="store_true", help="Also evaluate repeated workload (cache on vs off)")
    parser.add_argument("--quick", action="store_true", help="Quick smoke test (5 samples, 1 repeat)")
    parser.add_argument("--force-unsafe", action="store_true", help="Override session gate aborts on low clock or battery")
    args = parser.parse_args()

    reps = 1 if args.quick else args.repeats
    samples_count = 5 if args.quick else args.num_samples
    run_benchmark_suite(
        num_repeats=reps,
        max_samples=samples_count,
        batch_size=args.batch_size,
        output_path=args.output,
        include_repeated=args.include_repeated,
        force_unsafe=args.force_unsafe,
    )
