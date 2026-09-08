"""
Phase 3: Dual-Cache Phase-Aware Scheduler for Encoder-Decoder Speech Models.

Phase-aware Scheduling:
1. ENCODE_PHASE: Compute-bound audio feature encoding pass. Calculates encoder KV and writes to EncoderBlockManager.
2. DECODE_PHASE: Memory-bound token-by-token continuous batching with dual attention (causal self-attention + cross-attention over encoder KV).
3. Overlapping: Schedules compute-heavy encode steps and memory-bound decode steps within batch execution cycles.
4. Non-Negotiable Memory Safety: Deterministic cleanup paths via try/finally, high-water mark eviction (>90%), and periodic empty_cache maintenance.
"""

import time
import enum
import torch
import hashlib
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from transformers.cache_utils import DynamicCache, DynamicLayer

from .models import WhisperClassSpeechEncoderDecoderModel
from .dual_cache_manager import DualCacheManager
from .memory_guard import MemoryGuard, logger


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

    # Cached encoder KV tensors — materialized once at encode time, reused every decode step.
    # List of (K, V) per encoder layer: K/V shape = [1, num_heads, enc_seq_len, head_dim]
    cached_encoder_kv: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None

    # Cached decoder KV tensors — active contiguous working buffer per layer:
    # List of (K, V) per decoder layer: K/V shape = [1, num_kv_heads, seq_len, head_dim]
    cached_decoder_kv: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None

    # Timing metrics
    start_time: Optional[float] = None
    encode_done_time: Optional[float] = None
    first_token_time: Optional[float] = None
    finish_time: Optional[float] = None


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
    ):
        self.model = model
        self.dual_cache = dual_cache_manager
        self.memory_guard = memory_guard or MemoryGuard()
        self.max_batch_size = max_batch_size
        self.retain_encoder_cache = retain_encoder_cache
        self.step_delay = step_delay
        self.verbose = verbose
        self.device = device

        self.queue: List[DualCacheRequest] = []
        self.active_encode: List[DualCacheRequest] = []
        self.active_decode: List[DualCacheRequest] = []
        self.completed: List[DualCacheRequest] = []
        self._next_id = 0

        # Concurrency & telemetry tracking
        self.peak_in_flight_requests: int = 0
        self.rejected_requests_count: int = 0
        self.retained_encoder_caches: Dict[int, Tuple[List[int], float]] = {} # req_id -> (block_table, timestamp)

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

        mel_len = mel_features.shape[-1] if mel_features.dim() > 1 else mel_features.shape[0]
        # Speech Conv2 stride 2 downsampling: enc_seq_len = (mel_len + 1) // 2
        req_enc_seq_len = (mel_len + 1) // 2
        req_enc_blocks = (req_enc_seq_len + enc_block_size - 1) // enc_block_size
        req_dec_blocks = (max_new_tokens + dec_block_size - 1) // dec_block_size

        queued_enc_blocks = sum(
            (((r.mel_features.shape[-1] + 1) // 2) + enc_block_size - 1) // enc_block_size
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
            prompt_token_ids = torch.tensor([[1]], device=self.device, dtype=torch.long)

        req = DualCacheRequest(
            id=self._next_id,
            mel_features=mel_features.to(self.device),
            prompt_token_ids=prompt_token_ids.to(self.device),
            max_new_tokens=max_new_tokens,
            arrival_time=arrival_time,
        )
        self._next_id += 1
        self.queue.append(req)
        return req.id

    def _evict_retained_encoder_caches(self) -> int:
        """Evicts oldest retained encoder caches under memory pressure."""
        freed = 0
        if not self.retained_encoder_caches:
            return 0
        sorted_keys = sorted(self.retained_encoder_caches.keys(), key=lambda k: self.retained_encoder_caches[k][1])
        for k in sorted_keys:
            blk_table, _ = self.retained_encoder_caches.pop(k)
            freed += self.dual_cache.encoder_bm.release_blocks(blk_table)
        return freed

    def flush_retained_caches(self) -> int:
        """Flushes all retained LRU encoder caches and triggers cleanup telemetry."""
        freed = self._evict_retained_encoder_caches()
        if freed > 0:
            self.memory_guard.trigger_bulk_eviction_cleanup(freed, reason="flush_retained_caches")
        return freed

    def _compute_audio_block_hashes(self, mel_features: torch.Tensor, block_size: int = 16) -> List[str]:
        """Calculates MD5 content hashes for slices of audio mel frames."""
        mel_len = mel_features.shape[1]
        hashes = []
        for start_t in range(0, mel_len, block_size):
            end_t = min(start_t + block_size, mel_len)
            slice_tensor = mel_features[:, start_t:end_t]
            h = hashlib.md5(slice_tensor.detach().cpu().numpy().tobytes()).hexdigest()
            hashes.append(h)
        return hashes

    def _process_encode_phase(self):
        """Admit requests for encode phase and run audio encoder forward pass."""
        while len(self.active_encode) + len(self.active_decode) < self.max_batch_size and self.queue:
            next_req = self.queue[0]
            mel_len = next_req.mel_features.shape[-1]
            needed_enc_blocks = (mel_len + self.dual_cache.encoder_bm.block_size - 1) // self.dual_cache.encoder_bm.block_size

            # Check capacity before dequeuing
            if self.dual_cache.encoder_bm.num_free_blocks < needed_enc_blocks:
                freed = self._evict_retained_encoder_caches()
                if freed > 0:
                    self.memory_guard.trigger_bulk_eviction_cleanup(freed, reason="encode_phase_capacity")
                if self.dual_cache.encoder_bm.num_free_blocks < needed_enc_blocks:
                    # Not enough physical blocks available yet; wait for active requests to finish and free memory
                    break

            req = self.queue.pop(0)
            req.start_time = time.perf_counter()
            req.phase = RequestPhase.ENCODE_PHASE

            try:
                # 1. Forward audio mel features through model encoder
                # mel_features input: [1, mel_dim, mel_len]
                mel_in = req.mel_features.unsqueeze(0) if req.mel_features.dim() == 2 else req.mel_features
                with torch.no_grad():
                    encoder_hidden, encoder_kv_list = self.model.encode(mel_in)

                enc_seq_len = encoder_hidden.shape[1]
                req.encoder_seq_len = enc_seq_len

                # 2. Compute audio block content hashes for prefix/block sharing
                hashes = self._compute_audio_block_hashes(mel_in[0], block_size=self.dual_cache.encoder_bm.block_size)

                # 3. Allocate physical encoder blocks from EncoderBlockManager
                enc_block_table, hits = self.dual_cache.allocate_encoder_blocks(enc_seq_len, content_hashes=hashes)
                req.encoder_block_table = enc_block_table

                # 4. Write encoder Key/Value tensors into EncoderBlockManager physical pool
                num_enc_layers = len(encoder_kv_list)
                block_size = self.dual_cache.encoder_bm.block_size
                for l in range(num_enc_layers):
                    k = encoder_kv_list[l][0][0] # [num_heads, enc_seq_len, head_dim]
                    v = encoder_kv_list[l][1][0]

                    for blk_idx_in_req, blk_id in enumerate(enc_block_table):
                        start_t = blk_idx_in_req * block_size
                        end_t = min(start_t + block_size, enc_seq_len)
                        tok_count = end_t - start_t

                        k_slice = k[:, start_t:end_t, :].permute(1, 0, 2) # [tok_count, num_heads, head_dim]
                        v_slice = v[:, start_t:end_t, :].permute(1, 0, 2)

                        self.dual_cache.encoder_bm.store_encoder_kv(blk_id, l, k_slice, v_slice)

                req.encode_done_time = time.perf_counter()
                req.phase = RequestPhase.DECODE_PHASE

                # Initialize decoder prompt state
                prompt_len = req.prompt_token_ids.shape[1]
                req.decoder_seq_len = prompt_len
                needed_dec_blocks = (prompt_len + self.dual_cache.decoder_bm.block_size - 1) // self.dual_cache.decoder_bm.block_size
                for _ in range(needed_dec_blocks):
                    req.decoder_block_table.append(self.dual_cache.allocate_decoder_block())

                # Prefill decoder pass.
                # Gather encoder KV list from EncoderBlockManager — this is the ONLY gather for this request;
                # the result is stored in req.cached_encoder_kv and reused on every subsequent decode step.
                enc_kv_layers = []
                for l in range(num_enc_layers):
                    gathered_k = self.dual_cache.encoder_bm.key_pool[l][req.encoder_block_table]
                    gathered_v = self.dual_cache.encoder_bm.value_pool[l][req.encoder_block_table]
                    flat_k = gathered_k.view(-1, self.dual_cache.encoder_bm.num_heads, self.dual_cache.encoder_bm.head_dim)[:enc_seq_len]
                    flat_v = gathered_v.view(-1, self.dual_cache.encoder_bm.num_heads, self.dual_cache.encoder_bm.head_dim)[:enc_seq_len]
                    ck = flat_k.permute(1, 0, 2).unsqueeze(0)  # [1, num_heads, enc_seq_len, head_dim]
                    cv = flat_v.permute(1, 0, 2).unsqueeze(0)
                    enc_kv_layers.append((ck, cv))

                # Cache encoder KV on the request — never gathered again during decode
                req.cached_encoder_kv = enc_kv_layers

                with torch.no_grad():
                    dec_out = self.model.decode_step(
                        input_ids=req.prompt_token_ids,
                        encoder_past_kv_list=enc_kv_layers,
                        use_cache=True
                    )

                # Store decoder prefill KV into decoder working cache (contiguous, zero-permute)
                req.cached_decoder_kv = [
                    (dec_out.past_key_values.layers[l].keys.contiguous(), dec_out.past_key_values.layers[l].values.contiguous())
                    for l in range(self.dual_cache.decoder_bm.num_layers)
                ]

                # Store decoder prefill KV into decoder block pool
                for l in range(self.dual_cache.decoder_bm.num_layers):
                    dk = dec_out.past_key_values.layers[l].keys
                    dv = dec_out.past_key_values.layers[l].values

                    for blk_idx_in_req, blk_id in enumerate(req.decoder_block_table):
                        start_t = blk_idx_in_req * self.dual_cache.decoder_bm.block_size
                        end_t = min(start_t + self.dual_cache.decoder_bm.block_size, prompt_len)
                        tok_count = end_t - start_t
                        dk_slice = dk[0, :, start_t:end_t, :].permute(1, 0, 2)
                        dv_slice = dv[0, :, start_t:end_t, :].permute(1, 0, 2)

                        self.dual_cache.decoder_bm.key_pool[l][blk_id, :tok_count] = dk_slice
                        self.dual_cache.decoder_bm.value_pool[l][blk_id, :tok_count] = dv_slice

                next_tok = torch.argmax(dec_out.logits[:, -1, :], dim=-1).item()
                req.last_token_id = next_tok
                req.generated_tokens.append(next_tok)
                req.generated_count += 1
                req.first_token_time = time.perf_counter()

                if req.generated_count >= req.max_new_tokens:
                    self._complete_request(req, status=RequestPhase.COMPLETED)
                else:
                    self.active_decode.append(req)
                    self._cached_batch_key = None  # Invalidate batch-level encoder cache on new admission
                    self._cached_enc_mask = None

            except Exception as e:
                logger.error(f"Error in ENCODE_PHASE for req {req.id}: {e}")
                self._complete_request(req, status=RequestPhase.FAILED)

    def _process_decode_phase(self):
        """Execute one vectorized decode step across active decode requests in parallel."""
        if not self.active_decode:
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

                # Decoder self-attention padding mask (key len = max_dec_len + 1 past + current)
                self_attn_mask = torch.ones(
                    (batch_size, 1, 1, max_dec_len + 1),
                    device=self.device,
                    dtype=torch.bool,
                )
                for i, req in enumerate(self.active_decode):
                    if req.decoder_seq_len < max_dec_len:
                        self_attn_mask[i, 0, 0, req.decoder_seq_len:max_dec_len] = False

        # 4. Build batched encoder past KV cache list (with cross-step batch reuse)
        enc_mask = None
        if batch_size == 1:
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

                enc_kv_layers = []
                for l in range(num_enc_layers):
                    if min_enc_len == max_enc_len:
                        b_k = torch.cat([req.cached_encoder_kv[l][0] for req in self.active_decode], dim=0)
                        b_v = torch.cat([req.cached_encoder_kv[l][1] for req in self.active_decode], dim=0)
                    else:
                        b_k = torch.zeros(
                            (batch_size, enc_num_heads, max_enc_len, enc_head_dim),
                            device=self.device, dtype=enc_dtype
                        )
                        b_v = torch.zeros(
                            (batch_size, enc_num_heads, max_enc_len, enc_head_dim),
                            device=self.device, dtype=enc_dtype
                        )
                        for i, req in enumerate(self.active_decode):
                            ck, cv = req.cached_encoder_kv[l]
                            b_k[i, :, :req.encoder_seq_len, :] = ck[0]
                            b_v[i, :, :req.encoder_seq_len, :] = cv[0]
                    enc_kv_layers.append((b_k, b_v))

                if min_enc_len < max_enc_len:
                    enc_mask = torch.zeros(
                        (batch_size, 1, 1, max_enc_len),
                        device=self.device,
                        dtype=torch.bool,
                    )
                    for i, req in enumerate(self.active_decode):
                        enc_mask[i, 0, 0, :req.encoder_seq_len] = True
                else:
                    enc_mask = None

                self._cached_batch_key = batch_key
                self._cached_batched_enc_kv = enc_kv_layers
                self._cached_enc_mask = enc_mask

        # 5. Run single batched forward pass on GPU
        with torch.no_grad():
            dec_out = self.model.decode_step(
                input_ids=token_in,
                past_key_values=dec_cache,
                encoder_past_kv_list=enc_kv_layers,
                attention_mask=self_attn_mask,
                encoder_attention_mask=enc_mask,
                use_cache=True,
            )

        # 6. Scatter new decoder token KV back to physical block table & update request working cache
        still_active = []
        for i, req in enumerate(self.active_decode):
            try:
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
                next_tok = torch.argmax(dec_out.logits[i, -1, :], dim=-1).item()
                req.last_token_id = next_tok
                req.generated_tokens.append(next_tok)
                req.generated_count += 1

                if req.generated_count >= req.max_new_tokens:
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

        try:
            # 1. Immediate decoder block release
            if req.decoder_block_table:
                self.dual_cache.decoder_bm.free_blocks(req.decoder_block_table)
                req.decoder_block_table = []

            # 2. Encoder block release or LRU cache retention
            if self.retain_encoder_cache and status == RequestPhase.COMPLETED and req.encoder_block_table:
                self.retained_encoder_caches[req.id] = (list(req.encoder_block_table), time.perf_counter())
            else:
                if req.encoder_block_table:
                    self.dual_cache.encoder_bm.release_blocks(req.encoder_block_table)
                    req.encoder_block_table = []
        finally:
            req.cached_decoder_kv = None
            req.cached_encoder_kv = None
            self._cached_batch_key = None
            self._cached_batched_enc_kv = None
            self._cached_enc_mask = None
            self.completed.append(req)

    def step(self) -> bool:
        """
        Phase-aware scheduler step.
        Interleaves compute-heavy encode processing with memory-bound decode steps.
        """
        in_flight = len(self.active_encode) + len(self.active_decode)
        if in_flight > self.peak_in_flight_requests:
            self.peak_in_flight_requests = in_flight

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
        self._process_decode_phase()

        if self.step_delay > 0:
            time.sleep(self.step_delay)

        return len(self.queue) > 0 or len(self.active_encode) > 0 or len(self.active_decode) > 0

    def run_until_done(self, max_steps: int = 10000) -> List[DualCacheRequest]:
        """Execute phase-aware loop until queue and active requests are finished."""
        steps = 0
        while self.step() and steps < max_steps:
            steps += 1
        return self.completed
