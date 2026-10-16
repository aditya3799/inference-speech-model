"""
Phase 3: Dual-Cache Phase-Aware Scheduler for Encoder-Decoder Speech Models.

Phase-aware Scheduling:
1. ENCODE_PHASE: Compute-bound audio feature encoding pass.  A full-input hash
   is computed BEFORE calling the encoder.  On a hit the encoder is skipped
   entirely; ref-counts are bumped and cached KV is reused.  This fixes the
   original bug where the encoder ran before the cache lookup.
2. DECODE_PHASE: Memory-bound token-by-token continuous batching with dual
   attention (causal self-attention + cross-attention over encoder KV).
3. Overlapping: Schedules compute-heavy encode steps and memory-bound decode
   steps within batch execution cycles.
4. Non-Negotiable Memory Safety: Deterministic cleanup paths via try/finally,
   high-water mark eviction (>90%), and periodic empty_cache maintenance.

Block-alignment note: the model Conv2d uses stride-2 along the time axis, so
  enc_seq_len ≈ mel_len // 2.
  `needed_enc_blocks` is computed from `enc_seq_len`, not `mel_len`, to avoid
  2× over-reservation.

Shared-block safety: a full-input hash keyed on
  (md5(mel_bytes), model_dtype) is used.  On a cache hit the block pool is
  NOT written; blocks are only written for miss paths.

Telemetry added: encoder_calls, encoder_calls_skipped, encode_time_saved_s.
"""

import os
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import time
import enum
import torch
from torch.profiler import record_function
import hashlib
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from transformers.cache_utils import DynamicCache, DynamicLayer

from .models import WhisperClassSpeechEncoderDecoderModel
from .dual_cache_manager import DualCacheManager
from .memory_guard import MemoryGuard, logger
from .graph_decode_buffers import GraphDecodeBuffers, MAX_SLOTS, DEFAULT_READBACK_K
from .phase_timing import PhaseTimer


WHISPER_SOT_TOKEN_ID = 50258
WHISPER_LANG_EN_TOKEN_ID = 50259
WHISPER_TRANSCRIBE_TOKEN_ID = 50359
WHISPER_NO_TIMESTAMPS_TOKEN_ID = 50363
WHISPER_EOS_TOKEN_ID = 50257

WHISPER_DEFAULT_PREFIX_TOKENS = [
    WHISPER_SOT_TOKEN_ID,
    WHISPER_LANG_EN_TOKEN_ID,
    WHISPER_TRANSCRIBE_TOKEN_ID,
    WHISPER_NO_TIMESTAMPS_TOKEN_ID,
]


class RequestPhase(enum.Enum):
    QUEUED = "QUEUED"
    ENCODE_PHASE = "ENCODE_PHASE"
    DECODE_PHASE = "DECODE_PHASE"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass
class DualCacheRequest:
    """Represents a request in the Dual-Cache Phase-Aware Serving Engine."""
    id: int
    mel_features: torch.Tensor # [mel_dim, mel_len]
    prompt_token_ids: torch.Tensor # [1, prompt_len]
    max_new_tokens: int
    arrival_time: float = 0.0
    eos_token_id: Optional[int] = None
    full_input_hash: Optional[str] = None

    # Phase tracking
    phase: RequestPhase = RequestPhase.QUEUED

    # Encoder KV state
    encoder_seq_len: int = 0
    encoder_block_table: List[int] = field(default_factory=list)

    # Decoder KV state
    decoder_seq_len: int = 0
    decoder_block_table: List[int] = field(default_factory=list)

    # Output generation state
    generated_tokens: List[int] = field(default_factory=list)
    generated_count: int = 0
    last_token_id: Optional[int] = None
    done: bool = False
    slot_idx: Optional[int] = None
    graph_slot_idx: Optional[int] = None
    graph_slot_reused: bool = False
    graph_token_trace: List[Dict[str, int]] = field(default_factory=list)
    graph_extra_slot_steps: int = 0

    # Cached encoder KV tensors — materialized once at encode time, reused every decode step.
    # List of (K, V) per encoder layer: K/V shape = [1, num_heads, enc_seq_len, head_dim]
    cached_encoder_kv: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None

    # Cached decoder KV tensors — active contiguous working buffer per layer:
    # List of (K, V) per decoder layer: K/V shape = [1, num_kv_heads, seq_len, head_dim]
    cached_decoder_kv: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None

    # Timing metrics
    submission_time: Optional[float] = None
    arrival_wall_time: Optional[float] = None
    queue_wait_time: Optional[float] = None
    start_time: Optional[float] = None
    encode_done_time: Optional[float] = None
    first_token_time: Optional[float] = None
    finish_time: Optional[float] = None
    phase_timing: Optional[Dict[str, Any]] = None


class PhaseAwareSpeechScheduler:
    """
    Phase-aware serving scheduler orchestrating dual-cache pools for speech models.
    """

    def __init__(
        self,
        model: WhisperClassSpeechEncoderDecoderModel,
        dual_cache_manager: DualCacheManager,
        memory_guard: Optional[MemoryGuard] = None,
        max_batch_size: int = 32,
        retain_encoder_cache: bool = False,
        step_delay: float = 0.0,
        verbose: bool = False,
        device: str = "cuda",
        eos_token_id: Optional[int] = None,
        max_cached_encoder_inputs: int = 100,
        cache_cross_kv: bool = False,
        batch_encoder: bool = False,
        no_sync_decode: bool = False,
        avoid_redundant_cross_kv: bool = False,
        graph_decode: bool = False,
        graph_readback_k: int = DEFAULT_READBACK_K,
        graph_non_blocking_copies: bool = True,
        graph_zero_slot_buffers_on_release: bool = False,
        graph_force_bucket: Optional[int] = None,
        graph_direct_encoder_kv: bool = False,
        phase_timing: bool = False,
    ):
        self.model = model
        self.dual_cache = dual_cache_manager
        self.memory_guard = memory_guard or MemoryGuard()
        self.max_batch_size = max_batch_size
        self.retain_encoder_cache = retain_encoder_cache
        self.step_delay = step_delay
        self.verbose = verbose
        self.device = device
        self.eos_token_id = eos_token_id
        self.max_cached_encoder_inputs = max_cached_encoder_inputs
        self.cache_cross_kv = cache_cross_kv
        self.batch_encoder = batch_encoder
        self.no_sync_decode = no_sync_decode
        self.avoid_redundant_cross_kv = avoid_redundant_cross_kv
        self.graph_decode = graph_decode
        self.graph_readback_k = graph_readback_k
        self.graph_direct_encoder_kv = bool(graph_direct_encoder_kv)
        if self.graph_direct_encoder_kv and (not self.graph_decode or self.retain_encoder_cache):
            raise ValueError("graph_direct_encoder_kv requires graph_decode=True and retain_encoder_cache=False")
        self.phase_timing = bool(phase_timing)
        self.phase_timer = PhaseTimer(self.phase_timing, device=self.device)
        self.graph_buffers: Optional[GraphDecodeBuffers] = None
        if self.graph_decode:
            self.max_batch_size = min(self.max_batch_size, MAX_SLOTS)
            dt = getattr(self.dual_cache.decoder_bm, "dtype", torch.float16)
            self.graph_buffers = GraphDecodeBuffers(
                model=self.model,
                device=self.device,
                dtype=dt,
                max_slots=self.max_batch_size,
                readback_k=self.graph_readback_k,
                non_blocking_copies=graph_non_blocking_copies,
                zero_slot_buffers_on_release=graph_zero_slot_buffers_on_release,
                force_bucket=graph_force_bucket,
            )

        self.queue: List[DualCacheRequest] = []
        self.active_encode: List[DualCacheRequest] = []
        self.active_decode: List[DualCacheRequest] = []
        self.completed: List[DualCacheRequest] = []
        self._next_id = 0

        # Concurrency & telemetry tracking
        self.peak_in_flight_requests: int = 0
        self.peak_actually_used_bytes: int = 0
        self.rejected_requests_count: int = 0
        self.in_flight_history: List[int] = []
        self.queue_wait_times: List[float] = []
        self.retained_encoder_caches: Dict[int, Tuple[List[int], float]] = {} # req_id -> (block_table, timestamp)

        # Encoder-skip telemetry (Task 2)
        self.encoder_calls: int = 0                 # actual encoder forward passes executed
        self.encoder_calls_skipped: int = 0         # encoder forwards avoided via cache hit
        self.encode_time_saved_s: float = 0.0       # cumulative wall-clock seconds saved

        # Full-input encoder cache: full_input_hash -> entry dict:
        # {
        #     "block_table": List[int],
        #     "enc_seq_len": int,
        #     "cached_enc_kv": List[Tuple[torch.Tensor, torch.Tensor]],
        #     "last_access_time": float,
        #     "held_by_cache": bool,
        # }
        self._full_input_enc_cache: Dict[str, Dict[str, Any]] = {}

        # Batch-level encoder KV cache reuse across decode steps
        self._cached_batch_key: Optional[Tuple[int, ...]] = None
        self._cached_batched_enc_kv: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
        self._cached_enc_mask: Optional[torch.Tensor] = None

    def submit(
        self,
        mel_features: torch.Tensor,
        prompt_token_ids: Optional[torch.Tensor] = None,
        max_new_tokens: int = 50,
        arrival_time: float = 0.0,
        eos_token_id: Optional[int] = None,
        arrival_wall_time: Optional[float] = None,
    ) -> Optional[int]:
        """
        Submit a new encoder-decoder speech request.

        Returns:
            request_id (int) or None if request was rejected under high-water mark backpressure.
        """
        # Check high-water mark for encoder and decoder pools independently before admitting
        enc_stats = self.dual_cache.encoder_bm.get_stats()
        dec_stats = self.dual_cache.decoder_bm.get_stats()

        enc_block_size = self.dual_cache.encoder_bm.block_size
        dec_block_size = self.dual_cache.decoder_bm.block_size

        mel_in = mel_features.unsqueeze(0) if mel_features.dim() == 2 else mel_features
        full_hash = self._compute_full_input_hash(mel_in) if self.retain_encoder_cache else None
        is_hit = (full_hash in self._full_input_enc_cache) if (full_hash and self.retain_encoder_cache) else False

        mel_len = mel_features.shape[-1] if mel_features.dim() > 1 else mel_features.shape[0]
        # Speech Conv2 stride 2 downsampling: enc_seq_len = (mel_len + 1) // 2
        req_enc_seq_len = (mel_len + 1) // 2
        req_enc_blocks = 0 if is_hit or self.graph_direct_encoder_kv else ((req_enc_seq_len + enc_block_size - 1) // enc_block_size)
        req_dec_blocks = (max_new_tokens + dec_block_size - 1) // dec_block_size

        queued_enc_blocks = 0 if self.graph_direct_encoder_kv else sum(
            0 if (self.retain_encoder_cache and r.full_input_hash and r.full_input_hash in self._full_input_enc_cache)
            else (((r.mel_features.shape[-1] + 1) // 2) + enc_block_size - 1) // enc_block_size
            for r in self.queue
        )
        queued_dec_blocks = sum(
            (r.max_new_tokens + dec_block_size - 1) // dec_block_size
            for r in self.queue
        )

        enc_capacity = enc_stats["num_blocks"]
        enc_allocated = enc_stats["allocated_blocks"] + queued_enc_blocks + req_enc_blocks
        proj_enc_util = (enc_allocated / enc_capacity * 100.0) if enc_capacity > 0 else 0.0

        dec_capacity = dec_stats["num_blocks"]
        dec_allocated = dec_stats["allocated_blocks"] + queued_dec_blocks + req_dec_blocks
        proj_dec_util = (dec_allocated / dec_capacity * 100.0) if dec_capacity > 0 else 0.0

        if self.verbose:
            logger.info(
                f"[ADMISSION CHECK] Req #{self._next_id} | "
                f"Enc: curr={enc_stats['allocated_blocks']}, queued={queued_enc_blocks}, req={req_enc_blocks}, total={enc_capacity} -> proj={proj_enc_util:.1f}% (HWM={self.memory_guard.high_water_mark_pct}%) | "
                f"Dec: curr={dec_stats['allocated_blocks']}, queued={queued_dec_blocks}, req={req_dec_blocks}, total={dec_capacity} -> proj={proj_dec_util:.1f}%"
            )

        if proj_enc_util >= self.memory_guard.high_water_mark_pct or proj_dec_util >= self.memory_guard.high_water_mark_pct:
            # Trigger eviction of completed requests' encoder caches first
            freed = self._evict_retained_encoder_caches()
            if freed > 0:
                self.memory_guard.trigger_bulk_eviction_cleanup(freed, reason="high_water_mark_eviction")

            # Re-check utilization after eviction
            enc_stats = self.dual_cache.encoder_bm.get_stats()
            dec_stats = self.dual_cache.decoder_bm.get_stats()
            enc_allocated = enc_stats["allocated_blocks"] + queued_enc_blocks + req_enc_blocks
            proj_enc_util = (enc_allocated / enc_capacity * 100.0) if enc_capacity > 0 else 0.0

            dec_allocated = dec_stats["allocated_blocks"] + queued_dec_blocks + req_dec_blocks
            proj_dec_util = (dec_allocated / dec_capacity * 100.0) if dec_capacity > 0 else 0.0

            if proj_enc_util >= self.memory_guard.high_water_mark_pct or proj_dec_util >= self.memory_guard.high_water_mark_pct:
                max_proj = max(proj_enc_util, proj_dec_util)
                logger.warning(
                    f"[BACKPRESSURE] High-water mark active! Utilization would reach "
                    f"{max_proj:.1f}%. Rejecting request to prevent CUDA OOM."
                )
                self.rejected_requests_count += 1
                return None

        if prompt_token_ids is None:
            if getattr(self.model, "vocab_size", 0) >= 50364:
                prompt_token_ids = torch.tensor([WHISPER_DEFAULT_PREFIX_TOKENS], device=self.device, dtype=torch.long)
            else:
                prompt_token_ids = torch.tensor([[1]], device=self.device, dtype=torch.long)

        effective_eos = eos_token_id if eos_token_id is not None else self.eos_token_id
        if effective_eos is None and getattr(self.model, "vocab_size", 0) >= 50258:
            effective_eos = WHISPER_EOS_TOKEN_ID

        now = time.perf_counter()
        req = DualCacheRequest(
            id=self._next_id,
            mel_features=mel_features.to(self.device),
            prompt_token_ids=prompt_token_ids.to(self.device),
            max_new_tokens=max_new_tokens,
            arrival_time=arrival_time,
            arrival_wall_time=arrival_wall_time if arrival_wall_time is not None else now,
            submission_time=now,
            eos_token_id=effective_eos,
            full_input_hash=full_hash,
        )
        self._next_id += 1
        self.queue.append(req)
        return req.id

    def _evict_retained_encoder_caches(self, min_blocks_to_free: int = 1) -> int:
        """Evicts oldest retained encoder caches under memory pressure until min_blocks_to_free are released."""
        freed = 0
        if not self.retained_encoder_caches and not self._full_input_enc_cache:
            return 0
        if self.retained_encoder_caches:
            sorted_keys = sorted(self.retained_encoder_caches.keys(), key=lambda k: self.retained_encoder_caches[k][1])
            for k in sorted_keys:
                blk_table, _ = self.retained_encoder_caches.pop(k)
                freed += self.dual_cache.encoder_bm.release_blocks(blk_table)
                if min_blocks_to_free > 0 and freed >= min_blocks_to_free:
                    return freed
        if self._full_input_enc_cache:
            sorted_hashes = sorted(self._full_input_enc_cache.keys(), key=lambda h: self._full_input_enc_cache[h]["last_access_time"])
            for h in sorted_hashes:
                entry = self._full_input_enc_cache.pop(h)
                if entry.get("held_by_cache", False):
                    freed += self.dual_cache.encoder_bm.release_blocks(entry["block_table"])
                if min_blocks_to_free > 0 and freed >= min_blocks_to_free:
                    return freed
        return freed

    def flush_retained_caches(self) -> int:
        """Flushes all retained LRU encoder caches and triggers cleanup telemetry."""
        freed = self._evict_retained_encoder_caches(min_blocks_to_free=0)
        if freed > 0:
            self.memory_guard.trigger_bulk_eviction_cleanup(freed, reason="flush_retained_caches")
        return freed

    def _compute_full_input_hash(self, mel_in: torch.Tensor) -> str:
        """
        Compute a deterministic MD5 hash that identifies the FULL mel-spectrogram
        input tensor (not chunks of it).

        Key design rationale:
        - The model encoder is **bidirectional**: every position attends to every
          other position in the full sequence.  Chunk-level sharing is unsafe
          because the same mel chunk produces different encoder hidden states
          depending on surrounding audio.
        - By keying on the full input we guarantee that two requests that hit
          the same hash will have produced identical encoder outputs.
        - The dtype string is included so that fp16 and fp32 runs never collide.

        Args:
            mel_in: [1, mel_dim, mel_len] tensor (already on device or CPU).

        Returns:
            Hex string (MD5 of the flat bytes of the CPU float32 representation
            + dtype tag).
        """
        flat = mel_in.detach().cpu().to(torch.float32).numpy().tobytes()
        dtype_tag = str(mel_in.dtype).encode()
        return hashlib.md5(flat + dtype_tag).hexdigest()

    def _process_encode_phase(self):
        """
        Admit queued requests for the encode phase.

        FIXED (Task 2): the full-input hash is computed BEFORE calling the
        encoder.  On a cache hit the encoder is skipped entirely:
          - ref_counts on the existing blocks are incremented
          - the cached enc_kv_list is cloned onto the request
          - `encoder_calls_skipped` and `encode_time_saved_s` are updated

        On a miss the encoder runs, the result is written into fresh (unshared)
        blocks, and the entry is stored in `_full_input_enc_cache`.

        Block-count fix: `needed_enc_blocks` is computed from `enc_seq_len`
        (≈ mel_len // 2 after Conv2 stride-2), not from `mel_len`, to avoid
        the 2× over-reservation observed in earlier memory profiling.

        Shared-block safety: on a miss path, blocks with ref_count > 1 are
        never overwritten (allocate_block() always returns freshly allocated
        blocks for a miss; the shared block is only reused via ref-count bump
        on the hit path where we do NOT write).
        """
        # Batch encoder fast-path (Part 5 controlled experiment, default OFF)
        available_slots = self.max_batch_size - (len(self.active_encode) + len(self.active_decode))
        if self.batch_encoder and available_slots > 1 and len(self.queue) > 1 and not self.retain_encoder_cache:
            batch_reqs = []
            enc_block_size = self.dual_cache.encoder_bm.block_size
            total_needed = 0
            while self.queue and len(batch_reqs) < available_slots:
                cand = self.queue[0]
                est_seq = (cand.mel_features.shape[-1] + 1) // 2
                needed = (est_seq + enc_block_size - 1) // enc_block_size
                if not self.graph_direct_encoder_kv and self.dual_cache.encoder_bm.num_free_blocks < total_needed + needed:
                    break
                r_pop = self.queue.pop(0)
                r_pop.start_time = time.perf_counter()
                ref_arr = r_pop.arrival_wall_time if r_pop.arrival_wall_time is not None else r_pop.submission_time
                if ref_arr is not None:
                    r_pop.queue_wait_time = r_pop.start_time - ref_arr
                    self.queue_wait_times.append(r_pop.queue_wait_time)
                    self.phase_timer.add([r_pop], "queue_admission_wait", r_pop.queue_wait_time * 1000.0, 0.0)
                if self.phase_timing:
                    self.phase_timer.ensure(r_pop)["encoder_batched"] = True
                r_pop.phase = RequestPhase.ENCODE_PHASE
                batch_reqs.append(r_pop)
                total_needed += needed

            if len(batch_reqs) > 1:
                b_mels = torch.stack([
                    r.mel_features.squeeze(0) if r.mel_features.dim() == 3 else r.mel_features
                    for r in batch_reqs
                ], dim=0).to(self.device)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_enc_start = time.perf_counter()
                with self.phase_timer.measure(batch_reqs, "encoder_forward"):
                    with record_function("encoder_forward"):
                        with torch.no_grad():
                            b_hidden, b_kv_list = self.model.encode(b_mels)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t_enc_end = time.perf_counter()
                enc_time = t_enc_end - t_enc_start
                self.encoder_calls += 1

                for i, req in enumerate(batch_reqs):
                    req_hidden = b_hidden[i:i+1]
                    req_kv = [(b_kv_list[l][0][i:i+1], b_kv_list[l][1][i:i+1]) for l in range(len(b_kv_list))]
                    self._setup_request_after_encode(req, req_hidden, req_kv, full_hash=None, enc_time=enc_time)
                return
            elif len(batch_reqs) == 1:
                self.queue.insert(0, batch_reqs[0])

        while len(self.active_encode) + len(self.active_decode) < self.max_batch_size and self.queue:
            next_req = self.queue[0]
            mel_len = next_req.mel_features.shape[-1]

            # --- Compute expected enc_seq_len from Conv2 stride-2 ---
            # This is the number of encoder positions, not mel frames.
            # Formula matches models.py Conv2d(stride=2): enc_seq_len = (mel_len + 1) // 2
            # (same as the submit() admission check uses)
            est_enc_seq_len = (mel_len + 1) // 2
            enc_block_size = self.dual_cache.encoder_bm.block_size
            needed_enc_blocks = (est_enc_seq_len + enc_block_size - 1) // enc_block_size

            # --- STEP 0: Compute full-input hash BEFORE any encoder work ---
            mel_in = (
                next_req.mel_features.unsqueeze(0)
                if next_req.mel_features.dim() == 2
                else next_req.mel_features
            )
            full_hash = self._compute_full_input_hash(mel_in)
            next_req.full_input_hash = full_hash
            cache_hit = (full_hash in self._full_input_enc_cache) if self.retain_encoder_cache else False

            # On a hit we need blocks already allocated; no new blocks required.
            blocks_needed_for_admission = 0 if cache_hit or self.graph_direct_encoder_kv else needed_enc_blocks

            # Check capacity before dequeuing
            if self.dual_cache.encoder_bm.num_free_blocks < blocks_needed_for_admission:
                freed = self._evict_retained_encoder_caches()
                if freed > 0:
                    self.memory_guard.trigger_bulk_eviction_cleanup(freed, reason="encode_phase_capacity")
                if self.dual_cache.encoder_bm.num_free_blocks < blocks_needed_for_admission:
                    # Not enough physical blocks available yet; wait for active requests to free memory
                    break

            req = self.queue.pop(0)
            req.start_time = time.perf_counter()
            ref_arrival = req.arrival_wall_time if req.arrival_wall_time is not None else req.submission_time
            if ref_arrival is not None:
                req.queue_wait_time = req.start_time - ref_arrival
                self.queue_wait_times.append(req.queue_wait_time)
                self.phase_timer.add([req], "queue_admission_wait", req.queue_wait_time * 1000.0, 0.0)
            if self.phase_timing:
                self.phase_timer.ensure(req)["encoder_batched"] = False
            req.phase = RequestPhase.ENCODE_PHASE

            try:
                if cache_hit:
                    # -------------------------------------------------------
                    # HIT PATH: encoder skipped entirely
                    # -------------------------------------------------------
                    entry = self._full_input_enc_cache[full_hash]
                    entry["last_access_time"] = time.perf_counter()
                    cached_block_table = entry["block_table"]
                    enc_seq_len = entry["enc_seq_len"]
                    cached_enc_kv = entry["cached_enc_kv"]

                    # Bump ref-counts and telemetry for every shared block (do NOT write)
                    for blk_id in cached_block_table:
                        self.dual_cache.encoder_bm.ref_counts[blk_id] += 1
                        self.dual_cache.encoder_bm.total_lookups += 1
                        self.dual_cache.encoder_bm.cache_hits += 1

                    req.encoder_block_table = list(cached_block_table)
                    req.encoder_seq_len = enc_seq_len

                    # Re-use the cached KV tensors (same memory; safe because
                    # encoder KV is read-only after the miss path wrote it)
                    req.cached_encoder_kv = cached_enc_kv

                    # Telemetry
                    self.encoder_calls_skipped += 1
                    saved_t = entry.get("encode_time", 0.0)
                    self.encode_time_saved_s += saved_t

                    if self.verbose:
                        logger.info(
                            f"[ENCODE HIT] req {req.id}: skipped encoder, "
                            f"reusing {len(cached_block_table)} blocks "
                            f"(hash={full_hash[:8]}…)"
                        )

                else:
                    # -------------------------------------------------------
                    # MISS PATH: run encoder, write into fresh blocks
                    # -------------------------------------------------------
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t_enc_start = time.perf_counter()
                    with self.phase_timer.measure([req], "encoder_forward"):
                        with record_function("encoder_forward"):
                            with torch.no_grad():
                                encoder_hidden, encoder_kv_list = self.model.encode(mel_in)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    t_enc_end = time.perf_counter()
                    enc_time = t_enc_end - t_enc_start
                    self.encoder_calls += 1

                    self._setup_request_after_encode(req, encoder_hidden, encoder_kv_list, full_hash=full_hash, enc_time=enc_time)

            except Exception as e:
                logger.error(f"Error in ENCODE_PHASE for req {req.id}: {e}", exc_info=True)
                self._complete_request(req, status=RequestPhase.FAILED)

    def _setup_request_after_encode(
        self,
        req: DualCacheRequest,
        encoder_hidden: torch.Tensor,
        encoder_kv_list: List[Tuple[torch.Tensor, torch.Tensor]],
        full_hash: Optional[str] = None,
        enc_time: float = 0.0,
    ):
        enc_block_size = self.dual_cache.encoder_bm.block_size
        enc_seq_len = encoder_hidden.shape[1]
        req.encoder_seq_len = enc_seq_len

        # Direct graph handoff keeps the encoder projections contiguous and
        # copies them once into the graph slot after prefill. The paged encoder
        # pool remains the source of truth for eager/cached configurations.
        direct_graph_handoff = self.graph_direct_encoder_kv
        real_needed = (enc_seq_len + enc_block_size - 1) // enc_block_size
        enc_block_table = []
        if not direct_graph_handoff:
            with self.phase_timer.measure([req], "block_allocate"):
                with record_function("scheduler_bookkeeping"):
                    for _ in range(real_needed):
                        blk_id, _ = self.dual_cache.encoder_bm.allocate_block(content_hash=None)
                        enc_block_table.append(blk_id)
                    req.encoder_block_table = enc_block_table

        # Write encoder KV into fresh blocks ONLY
        if not direct_graph_handoff:
            with self.phase_timer.measure([req], "cross_kv_pool_copy"):
                with record_function("scheduler_bookkeeping"):
                    num_enc_layers = len(encoder_kv_list)
                    for l in range(num_enc_layers):
                        k = encoder_kv_list[l][0][0]
                        v = encoder_kv_list[l][1][0]
                        for blk_idx_in_req, blk_id in enumerate(enc_block_table):
                            assert self.dual_cache.encoder_bm.ref_counts[blk_id] == 1, (
                                f"Attempted to write into shared block {blk_id} "
                                f"(ref_count={self.dual_cache.encoder_bm.ref_counts[blk_id]})"
                            )
                            start_t = blk_idx_in_req * enc_block_size
                            end_t = min(start_t + enc_block_size, enc_seq_len)
                            tok_count = end_t - start_t
                            k_slice = k[:, start_t:end_t, :].permute(1, 0, 2)
                            v_slice = v[:, start_t:end_t, :].permute(1, 0, 2)
                            self.dual_cache.encoder_bm.store_encoder_kv(blk_id, l, k_slice, v_slice)

        # Gather enc KV for this request (done once at handoff)
        if direct_graph_handoff:
            enc_kv_layers = encoder_kv_list
            req.cached_encoder_kv = enc_kv_layers
        else:
            with self.phase_timer.measure([req], "cross_kv_contiguous_or_graph_copy"):
                with record_function("cross_kv_gather"):
                    enc_kv_layers = []
                    for l in range(num_enc_layers):
                        gathered_k = self.dual_cache.encoder_bm.key_pool[l][enc_block_table]
                        gathered_v = self.dual_cache.encoder_bm.value_pool[l][enc_block_table]
                        flat_k = gathered_k.view(-1, self.dual_cache.encoder_bm.num_heads, self.dual_cache.encoder_bm.head_dim)[:enc_seq_len]
                        flat_v = gathered_v.view(-1, self.dual_cache.encoder_bm.num_heads, self.dual_cache.encoder_bm.head_dim)[:enc_seq_len]
                        ck = flat_k.permute(1, 0, 2).unsqueeze(0)
                        cv = flat_v.permute(1, 0, 2).unsqueeze(0)
                        enc_kv_layers.append((ck, cv))
                    req.cached_encoder_kv = enc_kv_layers

        if self.retain_encoder_cache and full_hash:
            for blk_id in enc_block_table:
                self.dual_cache.encoder_bm.ref_counts[blk_id] += 1
            held_by_cache = True
            while len(self._full_input_enc_cache) >= self.max_cached_encoder_inputs:
                oldest_hash = min(
                    self._full_input_enc_cache.keys(),
                    key=lambda h: self._full_input_enc_cache[h]["last_access_time"]
                )
                evicted_entry = self._full_input_enc_cache.pop(oldest_hash)
                if evicted_entry.get("held_by_cache", False):
                    self.dual_cache.encoder_bm.release_blocks(evicted_entry["block_table"])

            self._full_input_enc_cache[full_hash] = {
                "block_table": list(enc_block_table),
                "enc_seq_len": enc_seq_len,
                "cached_enc_kv": enc_kv_layers,
                "last_access_time": time.perf_counter(),
                "held_by_cache": held_by_cache,
                "encode_time": enc_time,
            }

        # Decoder prefill
        req.encode_done_time = time.perf_counter()
        req.phase = RequestPhase.DECODE_PHASE

        prompt_len = req.prompt_token_ids.shape[1]
        req.decoder_seq_len = prompt_len
        if self.graph_decode:
            total_dec_len = prompt_len + req.max_new_tokens
            needed_dec_blocks = (total_dec_len + self.dual_cache.decoder_bm.block_size - 1) // self.dual_cache.decoder_bm.block_size
        else:
            needed_dec_blocks = (prompt_len + self.dual_cache.decoder_bm.block_size - 1) // self.dual_cache.decoder_bm.block_size
        with self.phase_timer.measure([req], "block_allocate"):
            with record_function("scheduler_bookkeeping"):
                for _ in range(needed_dec_blocks):
                    req.decoder_block_table.append(self.dual_cache.allocate_decoder_block())

        with self.phase_timer.measure([req], "prefill"):
            with torch.no_grad():
                dec_out = self.model.decode_step(
                    input_ids=req.prompt_token_ids,
                    encoder_past_kv_list=enc_kv_layers,
                    use_cache=True
                )

        with record_function("scheduler_bookkeeping"):
            req.cached_decoder_kv = [
                (dec_out.past_key_values.layers[l].keys.contiguous(),
                 dec_out.past_key_values.layers[l].values.contiguous())
                for l in range(self.dual_cache.decoder_bm.num_layers)
            ]
            for l in range(self.dual_cache.decoder_bm.num_layers):
                dk = dec_out.past_key_values.layers[l].keys
                dv = dec_out.past_key_values.layers[l].values
                for blk_idx_in_req, blk_id in enumerate(req.decoder_block_table):
                    start_t = blk_idx_in_req * self.dual_cache.decoder_bm.block_size
                    if start_t >= prompt_len:
                        break
                    end_t = min(start_t + self.dual_cache.decoder_bm.block_size, prompt_len)
                    tok_count = end_t - start_t
                    dk_slice = dk[0, :, start_t:end_t, :].permute(1, 0, 2)
                    dv_slice = dv[0, :, start_t:end_t, :].permute(1, 0, 2)
                    self.dual_cache.decoder_bm.key_pool[l][blk_id, :tok_count] = dk_slice
                    self.dual_cache.decoder_bm.value_pool[l][blk_id, :tok_count] = dv_slice

        with record_function("sampling"):
            next_tok = torch.argmax(dec_out.logits[:, -1, :], dim=-1).item()
        req.last_token_id = next_tok
        req.generated_tokens.append(next_tok)
        req.generated_count += 1
        req.first_token_time = time.perf_counter()

        is_eos = (req.eos_token_id is not None and next_tok == req.eos_token_id)
        if is_eos or req.generated_count >= req.max_new_tokens:
            self._complete_request(req, status=RequestPhase.COMPLETED)
        else:
            self.active_decode.append(req)
            if self.graph_decode:
                with self.phase_timer.measure([req], "cross_kv_contiguous_or_graph_copy"):
                    slot_idx = self.graph_buffers.assign_slot(
                        req=req,
                        enc_kv_layers=enc_kv_layers,
                        prompt_len=prompt_len,
                        initial_tok=next_tok,
                        initial_self_kv=req.cached_decoder_kv,
                    )
                req.slot_idx = slot_idx
                req.graph_slot_idx = slot_idx
                req.graph_slot_reused = self.graph_buffers.slots[slot_idx].reused
            if self.avoid_redundant_cross_kv:
                req.cached_encoder_kv = None
            self._cached_batch_key = None
            self._cached_enc_mask = None


    def _process_decode_phase_graph(self):
        """Execute one decode step across active decode slots using CUDA graph replay."""
        if not self.active_decode:
            return

        active_at_start = list(self.active_decode)
        with self.phase_timer.measure(active_at_start, "host_between_replays"):
            bucket = self.graph_buffers.prepare_replay()
        with self.phase_timer.measure(active_at_start, "graph_replay"):
            with record_function("graph_decode_step"):
                self.graph_buffers.replay_step(bucket)

        if self.graph_buffers.should_readback():
            with self.phase_timer.measure(active_at_start, "readback_wait"):
                results = self.graph_buffers.sync_tokens()
            completed_in_step = []
            with self.phase_timer.measure(active_at_start, "host_between_replays"):
                for slot, tokens in results:
                    req = slot.req
                    for token_offset, (tok, trace) in enumerate(zip(tokens, slot.last_readback_trace)):
                        if req.done:
                            break
                        req.graph_token_trace.append({
                            "token_index": len(req.generated_tokens),
                            "bucket": trace["bucket"],
                            "readback_position": trace["readback_position"],
                        })
                        req.last_token_id = tok
                        req.generated_tokens.append(tok)
                        req.generated_count = len(req.generated_tokens)
                        req.decoder_seq_len = req.prompt_token_ids.shape[1] + req.generated_count

                        is_eos = (req.eos_token_id is not None and tok == req.eos_token_id)
                        if is_eos or req.generated_count >= req.max_new_tokens:
                            req.graph_extra_slot_steps = len(tokens) - token_offset - 1
                            if self.phase_timing:
                                self.phase_timer.ensure(req)["slot_hold_steps_after_eos"] = req.graph_extra_slot_steps
                            if is_eos and req.eos_token_id in req.generated_tokens:
                                eos_pos = req.generated_tokens.index(req.eos_token_id)
                                req.generated_tokens = req.generated_tokens[:eos_pos]
                            self.graph_buffers.release_slot(slot.slot_idx)
                            req.slot_idx = None
                            self._complete_request(req, status=RequestPhase.COMPLETED)
                            completed_in_step.append(req)
                            break

            if completed_in_step:
                self.active_decode = [r for r in self.active_decode if r not in completed_in_step]

    def _process_decode_phase(self):
        """Execute one vectorized decode step across active decode requests in parallel."""
        if not self.active_decode:
            return

        if self.graph_decode:
            self._process_decode_phase_graph()
            return

        # 1. Allocate decoder blocks for any requests crossing block boundaries
        for req in self.active_decode:
            if req.decoder_seq_len % self.dual_cache.decoder_bm.block_size == 0:
                req.decoder_block_table.append(self.dual_cache.allocate_decoder_block())

        batch_size = len(self.active_decode)

        # 2. Build batched token_in tensor [batch_size, 1]
        token_in = torch.tensor([[req.last_token_id] for req in self.active_decode], device=self.device, dtype=torch.long)

        # 3. Build batched decoder past KV cache directly from contiguous cached_decoder_kv (zero block gathers, zero permutes)
        num_dec_layers = self.dual_cache.decoder_bm.num_layers
        num_kv_heads = self.dual_cache.decoder_bm.num_kv_heads
        head_dim = self.dual_cache.decoder_bm.head_dim

        dec_cache = DynamicCache()
        self_attn_mask = None
        with record_function("self_kv_gather"):
            if batch_size == 1:
                req = self.active_decode[0]
                for l in range(num_dec_layers):
                    layer = DynamicLayer()
                    layer.keys = req.cached_decoder_kv[l][0]
                    layer.values = req.cached_decoder_kv[l][1]
                    layer.is_initialized = True
                    dec_cache.layers.append(layer)
            else:
                max_dec_len = max(req.decoder_seq_len for req in self.active_decode)
                min_dec_len = min(req.decoder_seq_len for req in self.active_decode)
                if min_dec_len == max_dec_len:
                    for l in range(num_dec_layers):
                        layer = DynamicLayer()
                        layer.keys = torch.cat([req.cached_decoder_kv[l][0] for req in self.active_decode], dim=0)
                        layer.values = torch.cat([req.cached_decoder_kv[l][1] for req in self.active_decode], dim=0)
                        layer.is_initialized = True
                        dec_cache.layers.append(layer)
                else:
                    for l in range(num_dec_layers):
                        batched_k = torch.zeros(
                            (batch_size, num_kv_heads, max_dec_len, head_dim),
                            device=self.device, dtype=self.dual_cache.decoder_bm.dtype
                        )
                        batched_v = torch.zeros(
                            (batch_size, num_kv_heads, max_dec_len, head_dim),
                            device=self.device, dtype=self.dual_cache.decoder_bm.dtype
                        )
                        for i, req in enumerate(self.active_decode):
                            ck = req.cached_decoder_kv[l][0]
                            cv = req.cached_decoder_kv[l][1]
                            batched_k[i, :, :req.decoder_seq_len, :] = ck[0]
                            batched_v[i, :, :req.decoder_seq_len, :] = cv[0]
                        layer = DynamicLayer()
                        layer.keys = batched_k
                        layer.values = batched_v
                        layer.is_initialized = True
                        dec_cache.layers.append(layer)

        with record_function("mask_construction"):
            if batch_size > 1 and min_dec_len < max_dec_len:
                self_attn_mask = torch.ones(
                    (batch_size, max_dec_len + 1),
                    device=self.device,
                    dtype=torch.long,
                )
                for i, req in enumerate(self.active_decode):
                    if req.decoder_seq_len < max_dec_len:
                        self_attn_mask[i, req.decoder_seq_len:max_dec_len] = 0

        # 4. Build batched encoder past KV cache list
        enc_mask = None
        if not self.cache_cross_kv:
            # Legacy / pre-fix path: re-gather cross-attention KV from physical blocks on EVERY decode step
            num_enc_layers = self.model.num_encoder_layers
            enc_num_heads = self.dual_cache.encoder_bm.num_heads
            enc_head_dim = self.dual_cache.encoder_bm.head_dim
            enc_dtype = self.dual_cache.encoder_bm.dtype
            max_enc_len = max(req.encoder_seq_len for req in self.active_decode)
            min_enc_len = min(req.encoder_seq_len for req in self.active_decode)

            enc_kv_layers = []
            for l in range(num_enc_layers):
                if min_enc_len == max_enc_len:
                    b_k = torch.cat([
                        self.dual_cache.encoder_bm.key_pool[l][req.encoder_block_table].view(
                            -1, enc_num_heads, enc_head_dim
                        )[:req.encoder_seq_len].permute(1, 0, 2).unsqueeze(0)
                        for req in self.active_decode
                    ], dim=0)
                    b_v = torch.cat([
                        self.dual_cache.encoder_bm.value_pool[l][req.encoder_block_table].view(
                            -1, enc_num_heads, enc_head_dim
                        )[:req.encoder_seq_len].permute(1, 0, 2).unsqueeze(0)
                        for req in self.active_decode
                    ], dim=0)
                else:
                    b_k = torch.zeros((batch_size, enc_num_heads, max_enc_len, enc_head_dim), device=self.device, dtype=enc_dtype)
                    b_v = torch.zeros((batch_size, enc_num_heads, max_enc_len, enc_head_dim), device=self.device, dtype=enc_dtype)
                    for i, req in enumerate(self.active_decode):
                        k_blks = self.dual_cache.encoder_bm.key_pool[l][req.encoder_block_table]
                        v_blks = self.dual_cache.encoder_bm.value_pool[l][req.encoder_block_table]
                        b_k[i, :, :req.encoder_seq_len, :] = k_blks.view(-1, enc_num_heads, enc_head_dim)[:req.encoder_seq_len].permute(1, 0, 2).unsqueeze(0)[0]
                        b_v[i, :, :req.encoder_seq_len, :] = v_blks.view(-1, enc_num_heads, enc_head_dim)[:req.encoder_seq_len].permute(1, 0, 2).unsqueeze(0)[0]
                enc_kv_layers.append((b_k, b_v))

            if min_enc_len < max_enc_len:
                enc_mask = torch.zeros((batch_size, 1, 1, max_enc_len), device=self.device, dtype=torch.bool)
                for i, req in enumerate(self.active_decode):
                    enc_mask[i, 0, 0, :req.encoder_seq_len] = True
            else:
                enc_mask = None
        else:
            # Optimized path (cache_cross_kv=True): cross-KV gathered once at handoff, reused across decode steps
            if batch_size == 1:
                if self.avoid_redundant_cross_kv:
                    batch_key = tuple(req.id for req in self.active_decode)
                    if self._cached_batch_key == batch_key and self._cached_batched_enc_kv is not None:
                        enc_kv_layers = self._cached_batched_enc_kv
                    else:
                        req = self.active_decode[0]
                        num_enc_layers = self.model.num_encoder_layers
                        enc_num_heads = self.dual_cache.encoder_bm.num_heads
                        enc_head_dim = self.dual_cache.encoder_bm.head_dim
                        enc_kv_layers = []
                        for l in range(num_enc_layers):
                            k_blks = self.dual_cache.encoder_bm.key_pool[l][req.encoder_block_table]
                            v_blks = self.dual_cache.encoder_bm.value_pool[l][req.encoder_block_table]
                            ck = k_blks.view(-1, enc_num_heads, enc_head_dim)[:req.encoder_seq_len].permute(1, 0, 2).unsqueeze(0)
                            cv = v_blks.view(-1, enc_num_heads, enc_head_dim)[:req.encoder_seq_len].permute(1, 0, 2).unsqueeze(0)
                            enc_kv_layers.append((ck, cv))
                        self._cached_batch_key = batch_key
                        self._cached_batched_enc_kv = enc_kv_layers
                else:
                    enc_kv_layers = self.active_decode[0].cached_encoder_kv
            else:
                batch_key = tuple(req.id for req in self.active_decode)
                if self._cached_batch_key == batch_key and self._cached_batched_enc_kv is not None:
                    enc_kv_layers = self._cached_batched_enc_kv
                    enc_mask = self._cached_enc_mask
                else:
                    max_enc_len = max(req.encoder_seq_len for req in self.active_decode)
                    min_enc_len = min(req.encoder_seq_len for req in self.active_decode)
                    num_enc_layers = self.model.num_encoder_layers
                    enc_num_heads = self.dual_cache.encoder_bm.num_heads
                    enc_head_dim = self.dual_cache.encoder_bm.head_dim
                    enc_dtype = self.dual_cache.encoder_bm.dtype

                    with record_function("cross_kv_gather"):
                        enc_kv_layers = []
                        for l in range(num_enc_layers):
                            if self.avoid_redundant_cross_kv:
                                if min_enc_len == max_enc_len:
                                    b_k = torch.cat([
                                        self.dual_cache.encoder_bm.key_pool[l][req.encoder_block_table].view(
                                            -1, enc_num_heads, enc_head_dim
                                        )[:req.encoder_seq_len].permute(1, 0, 2).unsqueeze(0)
                                        for req in self.active_decode
                                    ], dim=0)
                                    b_v = torch.cat([
                                        self.dual_cache.encoder_bm.value_pool[l][req.encoder_block_table].view(
                                            -1, enc_num_heads, enc_head_dim
                                        )[:req.encoder_seq_len].permute(1, 0, 2).unsqueeze(0)
                                        for req in self.active_decode
                                    ], dim=0)
                                else:
                                    b_k = torch.zeros((batch_size, enc_num_heads, max_enc_len, enc_head_dim), device=self.device, dtype=enc_dtype)
                                    b_v = torch.zeros((batch_size, enc_num_heads, max_enc_len, enc_head_dim), device=self.device, dtype=enc_dtype)
                                    for i, req in enumerate(self.active_decode):
                                        k_blks = self.dual_cache.encoder_bm.key_pool[l][req.encoder_block_table]
                                        v_blks = self.dual_cache.encoder_bm.value_pool[l][req.encoder_block_table]
                                        b_k[i, :, :req.encoder_seq_len, :] = k_blks.view(-1, enc_num_heads, enc_head_dim)[:req.encoder_seq_len].permute(1, 0, 2).unsqueeze(0)[0]
                                        b_v[i, :, :req.encoder_seq_len, :] = v_blks.view(-1, enc_num_heads, enc_head_dim)[:req.encoder_seq_len].permute(1, 0, 2).unsqueeze(0)[0]
                            else:
                                if min_enc_len == max_enc_len:
                                    b_k = torch.cat([req.cached_encoder_kv[l][0] for req in self.active_decode], dim=0)
                                    b_v = torch.cat([req.cached_encoder_kv[l][1] for req in self.active_decode], dim=0)
                                else:
                                    b_k = torch.zeros((batch_size, enc_num_heads, max_enc_len, enc_head_dim), device=self.device, dtype=enc_dtype)
                                    b_v = torch.zeros((batch_size, enc_num_heads, max_enc_len, enc_head_dim), device=self.device, dtype=enc_dtype)
                                    for i, req in enumerate(self.active_decode):
                                        ck, cv = req.cached_encoder_kv[l]
                                        b_k[i, :, :req.encoder_seq_len, :] = ck[0]
                                        b_v[i, :, :req.encoder_seq_len, :] = cv[0]
                            enc_kv_layers.append((b_k, b_v))

                    with record_function("mask_construction"):
                        if min_enc_len < max_enc_len:
                            enc_mask = torch.zeros((batch_size, 1, 1, max_enc_len), device=self.device, dtype=torch.bool)
                            for i, req in enumerate(self.active_decode):
                                enc_mask[i, 0, 0, :req.encoder_seq_len] = True
                        else:
                            enc_mask = None

                    self._cached_batch_key = batch_key
                    self._cached_batched_enc_kv = enc_kv_layers
                    self._cached_enc_mask = enc_mask

        # 5. Run single batched forward pass on GPU
        position_ids = torch.tensor(
            [[req.decoder_seq_len] for req in self.active_decode],
            device=self.device,
            dtype=torch.long,
        )
        with torch.no_grad():
            dec_out = self.model.decode_step(
                input_ids=token_in,
                past_key_values=dec_cache,
                encoder_past_kv_list=enc_kv_layers,
                attention_mask=self_attn_mask,
                encoder_attention_mask=enc_mask,
                position_ids=position_ids,
                use_cache=True,
            )

        # Sampling optimization: batched argmax and sync if no_sync_decode is True
        batch_next_tokens = None
        if self.no_sync_decode:
            with record_function("sampling"):
                batch_next_tokens = torch.argmax(dec_out.logits[:, -1, :], dim=-1).tolist()

        # 6. Scatter new decoder token KV back to physical block table & update request working cache
        still_active = []
        for i, req in enumerate(self.active_decode):
            try:
                with record_function("scheduler_bookkeeping"):
                    blk_id = req.decoder_block_table[req.decoder_seq_len // self.dual_cache.decoder_bm.block_size]
                    offset = req.decoder_seq_len % self.dual_cache.decoder_bm.block_size
                    for l in range(num_dec_layers):
                        new_k = dec_out.past_key_values.layers[l].keys[i, :, -1, :]
                        new_v = dec_out.past_key_values.layers[l].values[i, :, -1, :]
                        self.dual_cache.decoder_bm.key_pool[l][blk_id, offset] = new_k
                        self.dual_cache.decoder_bm.value_pool[l][blk_id, offset] = new_v

                        # Incremental O(1) append to request working buffer [1, num_kv_heads, 1, head_dim]
                        new_k_slice = new_k.unsqueeze(0).unsqueeze(2)
                        new_v_slice = new_v.unsqueeze(0).unsqueeze(2)
                        req.cached_decoder_kv[l] = (
                            torch.cat([req.cached_decoder_kv[l][0], new_k_slice], dim=2),
                            torch.cat([req.cached_decoder_kv[l][1], new_v_slice], dim=2)
                        )

                    req.decoder_seq_len += 1

                with record_function("sampling"):
                    if self.no_sync_decode and batch_next_tokens is not None:
                        next_tok = batch_next_tokens[i]
                    else:
                        next_tok = torch.argmax(dec_out.logits[i, -1, :], dim=-1).item()

                req.last_token_id = next_tok
                req.generated_tokens.append(next_tok)
                req.generated_count += 1

                is_eos = (req.eos_token_id is not None and next_tok == req.eos_token_id)
                if is_eos or req.generated_count >= req.max_new_tokens:
                    self._complete_request(req, status=RequestPhase.COMPLETED)
                else:
                    still_active.append(req)
            except Exception as e:
                logger.error(f"Error in DECODE_PHASE for req {req.id}: {e}")
                self._complete_request(req, status=RequestPhase.FAILED)

        if len(still_active) != len(self.active_decode):
            self._cached_batch_key = None  # Invalidate batch cache when any request completes
            self._cached_enc_mask = None
        self.active_decode = still_active

    def _complete_request(self, req: DualCacheRequest, status: RequestPhase):
        """
        Completes a request and executes EXPLICIT, DETERMINISTIC block release.
        Decoder blocks are released immediately; encoder blocks are released or retained in LRU cache.
        """
        req.phase = status
        req.done = True
        req.finish_time = time.perf_counter()

        if self.graph_decode and getattr(req, "slot_idx", None) is not None:
            self.graph_buffers.release_slot(req.slot_idx)
            req.slot_idx = None

        try:
            with self.phase_timer.measure([req], "block_release"):
                # 1. Immediate decoder block release
                if req.decoder_block_table:
                    self.dual_cache.decoder_bm.free_blocks(req.decoder_block_table)
                    req.decoder_block_table = []

                # 2. Encoder block release or LRU cache retention
                if req.encoder_block_table:
                    self.dual_cache.encoder_bm.release_blocks(req.encoder_block_table)
                    # If retain_encoder_cache is False, remove cache entry when no active requests reference its blocks
                    if not self.retain_encoder_cache and req.full_input_hash in self._full_input_enc_cache:
                        cached_entry = self._full_input_enc_cache[req.full_input_hash]
                        if all(self.dual_cache.encoder_bm.ref_counts[b] == 0 for b in cached_entry["block_table"]):
                            self._full_input_enc_cache.pop(req.full_input_hash, None)
                    req.encoder_block_table = []
        finally:
            req.cached_decoder_kv = None
            req.cached_encoder_kv = None
            self._cached_batch_key = None
            self._cached_batched_enc_kv = None
            self._cached_enc_mask = None
            self.completed.append(req)

    def _update_peak_actually_used(self):
        """Update peak pool memory + contiguous cross-KV copies in use."""
        enc_bm = self.dual_cache.encoder_bm
        dec_bm = self.dual_cache.decoder_bm
        dt_bytes = 2 if enc_bm.dtype in (torch.float16, torch.bfloat16) else 4
        enc_blk_bytes = 2 * enc_bm.num_layers * enc_bm.num_heads * enc_bm.block_size * enc_bm.head_dim * dt_bytes
        dec_blk_bytes = 2 * dec_bm.num_layers * dec_bm.num_kv_heads * dec_bm.block_size * dec_bm.head_dim * dt_bytes
        used_pool_bytes = (enc_bm.allocated_block_count * enc_blk_bytes) + (dec_bm.allocated_block_count * dec_blk_bytes)

        cached_cross_bytes = 0
        if self.cache_cross_kv:
            for req in self.active_decode:
                if req.cached_encoder_kv is not None:
                    cached_cross_bytes += (
                        req.encoder_seq_len * 2 * enc_bm.num_layers * enc_bm.num_heads * enc_bm.head_dim * dt_bytes
                    )

        if not hasattr(self, "peak_pool_blocks_used_bytes"):
            self.peak_pool_blocks_used_bytes = 0
        if used_pool_bytes > self.peak_pool_blocks_used_bytes:
            self.peak_pool_blocks_used_bytes = used_pool_bytes

        now_used = used_pool_bytes + cached_cross_bytes
        if now_used > self.peak_actually_used_bytes:
            self.peak_actually_used_bytes = now_used

    def step(self) -> bool:
        """
        Phase-aware scheduler step.
        Interleaves compute-heavy encode processing with memory-bound decode steps.
        """
        in_flight = len(self.active_encode) + len(self.active_decode)
        if in_flight > self.peak_in_flight_requests:
            self.peak_in_flight_requests = in_flight
        self.in_flight_history.append(in_flight)
        self._update_peak_actually_used()

        if self.verbose:
            enc_stats = self.dual_cache.encoder_bm.get_stats()
            dec_stats = self.dual_cache.decoder_bm.get_stats()
            self.memory_guard.log_telemetry(
                active_requests=in_flight,
                queued_requests=len(self.queue),
                encoder_stats=enc_stats,
                decoder_stats=dec_stats,
            )

        self._process_encode_phase()
        self._update_peak_actually_used()
        self._process_decode_phase()
        self._update_peak_actually_used()

        if self.step_delay > 0:
            time.sleep(self.step_delay)

        return len(self.queue) > 0 or len(self.active_encode) > 0 or len(self.active_decode) > 0

    def has_work(self) -> bool:
        """Returns True if there are any queued or active requests."""
        return len(self.queue) > 0 or len(self.active_encode) > 0 or len(self.active_decode) > 0

    def run_until_done(self, max_steps: int = 10000) -> List[DualCacheRequest]:
        """Execute phase-aware loop until queue and active requests are finished."""
        steps = 0
        while self.step() and steps < max_steps:
            steps += 1
        return self.completed

    def reset_telemetry(self):
        """Reset all per-run counters and histories cleanly."""
        self.encoder_calls = 0
        self.encoder_calls_skipped = 0
        self.encode_time_saved_s = 0.0
        self.rejected_requests_count = 0
        self.in_flight_history = []
        self.queue_wait_times = []
        self.completed = []
        self.peak_in_flight_requests = 0
        self.peak_actually_used_bytes = 0

    def get_concurrency_stats(self) -> Dict[str, Any]:
        """Return memory pool sizing, per-request demand, and runtime concurrency statistics."""
        enc_bm = self.dual_cache.encoder_bm
        dec_bm = self.dual_cache.decoder_bm

        dtype_bytes = 2 if enc_bm.dtype in (torch.float16, torch.bfloat16) else 4
        # Encoder KV: 2 * num_layers * num_heads * block_size * head_dim * bytes
        enc_block_bytes = 2 * enc_bm.num_layers * enc_bm.num_heads * enc_bm.block_size * enc_bm.head_dim * dtype_bytes
        enc_pool_mb = (enc_bm.num_blocks * enc_block_bytes) / (1024 * 1024)

        # Decoder KV: 2 * num_layers * num_kv_heads * block_size * head_dim * bytes
        dec_block_bytes = 2 * dec_bm.num_layers * dec_bm.num_kv_heads * dec_bm.block_size * dec_bm.head_dim * dtype_bytes
        dec_pool_mb = (dec_bm.num_blocks * dec_block_bytes) / (1024 * 1024)

        # Per-request blocks: 30s mel -> 1500 tokens -> 94 blocks; 150 max tokens -> 10 blocks
        req_enc_blocks = (1500 + enc_bm.block_size - 1) // enc_bm.block_size
        req_enc_mb = (req_enc_blocks * enc_block_bytes) / (1024 * 1024)
        req_dec_blocks = (150 + dec_bm.block_size - 1) // dec_bm.block_size
        req_dec_mb = (req_dec_blocks * dec_block_bytes) / (1024 * 1024)

        q_waits = self.queue_wait_times if self.queue_wait_times else [0.0]
        in_flights = self.in_flight_history if self.in_flight_history else [0]

        return {
            "encoder_pool_blocks": enc_bm.num_blocks,
            "encoder_pool_mb": enc_pool_mb,
            "decoder_pool_blocks": dec_bm.num_blocks,
            "decoder_pool_mb": dec_pool_mb,
            "total_pool_mb": enc_pool_mb + dec_pool_mb,
            "req_encoder_blocks": req_enc_blocks,
            "req_encoder_mb": req_enc_mb,
            "req_decoder_blocks": req_dec_blocks,
            "req_decoder_mb": req_dec_mb,
            "avg_in_flight_requests": float(np.mean(in_flights)),
            "max_in_flight_requests": int(max(in_flights)),
            "queue_wait_time_mean_s": float(np.mean(q_waits)),
            "queue_wait_time_p50_s": float(np.percentile(q_waits, 50)),
            "queue_wait_time_p95_s": float(np.percentile(q_waits, 95)),
            "rejections_per_run": self.rejected_requests_count,
            "peak_actually_used_mb": float(self.peak_actually_used_bytes / (1024 * 1024)),
        }

    def get_encoder_telemetry(self) -> Dict[str, Any]:
        """
        Return a summary of encoder-skip telemetry.

        Keys:
            encoder_calls              : number of encoder forward passes actually run
            encoder_calls_skipped      : number of encoder passes avoided via cache hit
            encode_time_saved_s        : estimated wall-clock seconds saved (approximate)
            total_requests             : sum of calls + skipped
            skip_rate_pct              : percentage of requests that skipped the encoder
            full_input_cache_entries   : number of entries currently resident
            max_cached_encoder_inputs  : capacity bound of the cache
        """
        total = self.encoder_calls + self.encoder_calls_skipped
        skip_rate = (self.encoder_calls_skipped / total * 100.0) if total > 0 else 0.0
        return {
            "encoder_calls": self.encoder_calls,
            "encoder_calls_skipped": self.encoder_calls_skipped,
            "encode_time_saved_s": round(self.encode_time_saved_s, 4),
            "total_requests": total,
            "skip_rate_pct": round(skip_rate, 2),
            "full_input_cache_entries": len(self._full_input_enc_cache),
            "max_cached_encoder_inputs": self.max_cached_encoder_inputs,
        }

    def flush_full_input_cache(self) -> int:
        """
        Clear the full-input encoder cache, releasing ref-counts for all
        cached block tables held by the cache.

        Returns:
            Number of unique full-input entries cleared.
        """
        count = len(self._full_input_enc_cache)
        for full_hash, entry in list(self._full_input_enc_cache.items()):
            if entry.get("held_by_cache", False):
                self.dual_cache.encoder_bm.release_blocks(entry["block_table"])
        self._full_input_enc_cache.clear()
        return count
