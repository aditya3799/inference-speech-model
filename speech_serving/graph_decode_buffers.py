"""
speech_serving/graph_decode_buffers.py

Slot-based persistent GPU buffers and CUDA graph infrastructure for graph-mode decode.
All behavior is gated behind graph_decode=True. Eager path is completely untouched.

Specifications (Step 2):
- Persistent GPU buffers per slot (max_batch_size=4):
  token_ids[B], positions[B], lengths[B], active_mask[B], finished[B], block_table[B, max_blocks].
  All updated by in-place copy_ from pinned host buffers BEFORE each replay.
- Self-attention KV:
  Length buckets: [64, 128, 256, 448].
  One captured graph per (batch=4, bucket).
  Static self-KV buffer written in-place via scatter_/index_copy_ using GPU index tensors.
  Attention mask for positions >= length and inactive slots.
- Cross-attention KV:
  Static buffer per slot, filled once at encode-to-decode handoff, reused every step.
- Sampling on GPU:
  argmax with suppress mask, written into token_ids buffer (no .item()).
  EOS and max-length tracked by on-GPU finished flags.
  Host reads token buffer once every K steps (K=4, configurable) via non-blocking copy + event.
  Tokens after EOS discarded so outputs match 100%.
- Warmup on side stream before capture, shared graph memory pool.
- Report extra VRAM overhead.
"""

import time
import math
import torch
import torch.nn as nn
from typing import Optional, List, Tuple, Dict, Any
from transformers.cache_utils import Cache, DynamicCache, EncoderDecoderCache


SELF_KV_BUCKETS = [64, 128, 256, 448]
MAX_SLOTS = 4
DEFAULT_READBACK_K = 1
MAX_BLOCKS = 64


def get_next_bucket(length: int) -> int:
    """Return the smallest bucket >= length, or the max bucket."""
    for b in SELF_KV_BUCKETS:
        if length <= b:
            return b
    return SELF_KV_BUCKETS[-1]


class StaticBucketLayer:
    """Fixed-size key/value buffer for one decoder layer and one bucket length."""
    def __init__(self, batch_size: int, num_heads: int, bucket_len: int, head_dim: int, device: str, dtype: torch.dtype):
        self.keys = torch.zeros(batch_size, num_heads, bucket_len, head_dim, device=device, dtype=dtype)
        self.values = torch.zeros(batch_size, num_heads, bucket_len, head_dim, device=device, dtype=dtype)
        self.is_initialized = True

    def update(self, k: torch.Tensor, v: torch.Tensor, pos_idx: torch.Tensor, *args, **kwargs):
        self.keys.scatter_(2, pos_idx, k)
        self.values.scatter_(2, pos_idx, v)
        return self.keys, self.values


class StaticBucketSelfCache(Cache):
    """
    Subclasses HuggingFace Cache to supply pre-allocated static bucket KV buffers
    to WhisperDecoder without dynamic allocations or torch.cat growth.
    """
    def __init__(
        self,
        batch_size: int,
        num_heads: int,
        bucket_len: int,
        head_dim: int,
        num_layers: int,
        pos_tensor: torch.Tensor,
        device: str,
        dtype: torch.dtype,
    ):
        layers = [
            StaticBucketLayer(batch_size, num_heads, bucket_len, head_dim, device=device, dtype=dtype)
            for _ in range(num_layers)
        ]
        super().__init__(layers=layers)
        self.bucket_len = bucket_len
        self.pos_tensor = pos_tensor  # [B, 1]
        self.num_heads = num_heads
        self.head_dim = head_dim

    def update(self, k: torch.Tensor, v: torch.Tensor, layer_idx: int, *args, **kwargs):
        pos_idx = self.pos_tensor.view(-1, 1, 1, 1).expand(-1, self.num_heads, 1, self.head_dim).clamp(0, self.bucket_len - 1)
        return self.layers[layer_idx].update(k, v, pos_idx)

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return 0

    def get_max_cache_shape(self) -> int:
        return self.bucket_len

    def to_legacy_cache(self):
        return tuple((layer.keys, layer.values) for layer in self.layers)


class SlotInfo:
    """Tracks CPU-side metadata for one slot in the slot-based decode engine."""
    def __init__(self, slot_idx: int):
        self.slot_idx = slot_idx
        self.active = False
        self.req = None
        self.decoder_seq_len = 0
        self.last_token_id = 0
        self.generated_count = 0
        self.max_new_tokens = 0
        self.eos_token_id = None
        self.bucket = SELF_KV_BUCKETS[0]
        self.pending_tokens: List[int] = []
        self.eos_reached = False
        self.enc_seq_len = 0
        self.readback_start = 0
        self.needs_token_upload = False
        self.assignment_count = 0
        self.reused = False
        self.last_readback_trace: List[Dict[str, int]] = []


class GraphDecodeBuffers:
    """
    Manages slot allocations, persistent GPU buffers, and CUDA graph captures
    for the PhaseAwareSpeechScheduler.
    """
    def __init__(
        self,
        model: Any,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        max_slots: int = MAX_SLOTS,
        readback_k: int = DEFAULT_READBACK_K,
        max_blocks: int = MAX_BLOCKS,
        max_enc_len: int = 1500,
        non_blocking_copies: bool = True,
        zero_slot_buffers_on_release: bool = False,
        force_bucket: Optional[int] = None,
    ):
        self.model = model
        self.device = device
        self.dtype = dtype
        self.max_slots = max_slots
        self.readback_k = readback_k
        self.max_blocks = max_blocks
        self.max_enc_len = max_enc_len
        self.non_blocking_copies = non_blocking_copies
        self.zero_slot_buffers_on_release = zero_slot_buffers_on_release
        self.force_bucket = force_bucket
        if force_bucket is not None and force_bucket not in SELF_KV_BUCKETS:
            raise ValueError(f"force_bucket must be one of {SELF_KV_BUCKETS}, got {force_bucket}")

        # Model geometry
        self.is_pretrained_whisper = hasattr(model, "hf_model")
        self.num_decoder_layers = model.num_decoder_layers
        self.num_heads = model.num_heads
        self.head_dim = model.head_dim
        self.hidden_dim = model.hidden_dim
        self.vocab_size = model.vocab_size
        self.num_encoder_layers = getattr(model, "num_encoder_layers", 6)

        # Slots
        self.slots = [SlotInfo(i) for i in range(max_slots)]

        # -------------------------------------------------------------
        # Pinned Host Buffers (written by CPU before each replay)
        # -------------------------------------------------------------
        self.h_token_ids = torch.zeros(max_slots, dtype=torch.long).pin_memory()
        self.h_positions = torch.zeros(max_slots, dtype=torch.long).pin_memory()
        self.h_lengths = torch.zeros(max_slots, dtype=torch.long).pin_memory()
        self.h_active_mask = torch.zeros(max_slots, dtype=torch.bool).pin_memory()
        self.h_finished = torch.zeros(max_slots, dtype=torch.bool).pin_memory()
        self.h_block_table = torch.zeros((max_slots, max_blocks), dtype=torch.long).pin_memory()
        self.h_output_tokens = torch.zeros(max_slots, dtype=torch.long).pin_memory()
        self.h_token_ring = torch.zeros((max_slots, max(readback_k, 1)), dtype=torch.long).pin_memory()
        self._step_in_k = 0
        self._window_buckets: List[int] = []

        # Host attention mask pinned buffers per bucket: [max_slots, 1, 1, bucket]
        self.h_attn_masks: Dict[int, torch.Tensor] = {}
        for b in SELF_KV_BUCKETS:
            self.h_attn_masks[b] = torch.zeros((max_slots, 1, 1, b), dtype=torch.float16).pin_memory()
        self.h_cross_mask = torch.zeros((max_slots, 1, 1, max_enc_len), dtype=torch.float16).pin_memory()
        # K-sized pinned input ring.  A bank is not reused until its H2D event
        # completes; the normal K-step readback synchronization retires all
        # banks without forcing a CPU wait before every replay.
        self._input_ring_size = max(readback_k, 2)
        self._input_banks = []
        for _ in range(self._input_ring_size):
            self._input_banks.append({
                "token_ids": torch.zeros_like(self.h_token_ids).pin_memory(),
                "positions": torch.zeros_like(self.h_positions).pin_memory(),
                "lengths": torch.zeros_like(self.h_lengths).pin_memory(),
                "active_mask": torch.zeros_like(self.h_active_mask).pin_memory(),
                "finished": torch.zeros_like(self.h_finished).pin_memory(),
                "attn_masks": {
                    b: torch.zeros_like(self.h_attn_masks[b]).pin_memory() for b in SELF_KV_BUCKETS
                },
                "cross_mask": torch.zeros_like(self.h_cross_mask).pin_memory(),
            })

        # -------------------------------------------------------------
        # Persistent GPU Buffers (read/written inside or around replay)
        # -------------------------------------------------------------
        self.g_token_ids = torch.zeros((max_slots, 1), dtype=torch.long, device=device)
        self.g_positions = torch.zeros((max_slots, 1), dtype=torch.long, device=device)
        self.g_lengths = torch.zeros(max_slots, dtype=torch.long, device=device)
        self.g_active_mask = torch.zeros(max_slots, dtype=torch.bool, device=device)
        self.g_finished = torch.zeros(max_slots, dtype=torch.bool, device=device)
        self.g_block_table = torch.zeros((max_slots, max_blocks), dtype=torch.long, device=device)
        self.g_output_tokens = torch.zeros(max_slots, dtype=torch.long, device=device)

        # GPU attention masks per bucket
        self.g_attn_masks: Dict[int, torch.Tensor] = {}
        for b in SELF_KV_BUCKETS:
            self.g_attn_masks[b] = torch.zeros((max_slots, 1, 1, b), dtype=torch.float16, device=device)
        self.g_cross_mask = torch.zeros((max_slots, 1, 1, max_enc_len), dtype=torch.float16, device=device)

        # Dummy encoder hidden states for Whisper decoder input
        self.dummy_enc = torch.empty((max_slots, max_enc_len, self.hidden_dim), device=device, dtype=dtype)
        self.g_cache_position = torch.zeros(1, dtype=torch.long, device=device)

        # -------------------------------------------------------------
        # Static Cross-KV Buffer (per slot, filled at handoff)
        # -------------------------------------------------------------
        self.cross_cache: Optional[DynamicCache] = None
        self.g_cross_kv: Optional[torch.Tensor] = None
        if self.is_pretrained_whisper:
            self.cross_cache = DynamicCache()
            for l in range(self.num_decoder_layers):
                ck = torch.zeros((max_slots, self.num_heads, max_enc_len, self.head_dim), device=device, dtype=dtype)
                cv = torch.zeros((max_slots, self.num_heads, max_enc_len, self.head_dim), device=device, dtype=dtype)
                self.cross_cache.update(ck, cv, layer_idx=l)
        else:
            self.g_cross_kv = torch.zeros(
                (max_slots, self.num_decoder_layers, 2, self.num_heads, max_enc_len, self.head_dim),
                device=device, dtype=dtype
            )

        # -------------------------------------------------------------
        # Static Self-KV Caches & Buffers per Bucket
        # -------------------------------------------------------------
        self.self_caches: Dict[int, StaticBucketSelfCache] = {}
        self.enc_dec_caches: Dict[int, EncoderDecoderCache] = {}
        self.g_self_kv_native: Dict[int, torch.Tensor] = {}

        for b in SELF_KV_BUCKETS:
            if self.is_pretrained_whisper:
                sc = StaticBucketSelfCache(
                    batch_size=max_slots,
                    num_heads=self.num_heads,
                    bucket_len=b,
                    head_dim=self.head_dim,
                    num_layers=self.num_decoder_layers,
                    pos_tensor=self.g_positions,
                    device=device,
                    dtype=dtype,
                )
                self.self_caches[b] = sc
                edc = EncoderDecoderCache(sc, self.cross_cache)
                for l in range(len(self.cross_cache.layers)):
                    edc.is_updated[l] = True
                self.enc_dec_caches[b] = edc
            else:
                self.g_self_kv_native[b] = torch.zeros(
                    (max_slots, self.num_decoder_layers, 2, self.num_heads, b, self.head_dim),
                    device=device, dtype=dtype
                )

        # -------------------------------------------------------------
        # CUDA Graph Captures
        # -------------------------------------------------------------
        self.graphs: Dict[int, torch.cuda.CUDAGraph] = {}
        self._graph_pool = None
        self._step_counter = 0
        self._readback_event = torch.cuda.Event()
        self._input_copy_events = [torch.cuda.Event() for _ in range(self._input_ring_size)]
        self._input_copy_in_flight = [False] * self._input_ring_size
        self.capture_all_buckets()

    def assign_slot(
        self,
        req: Any,
        enc_kv_layers: List[Tuple[torch.Tensor, torch.Tensor]],
        prompt_len: int,
        initial_tok: int,
        initial_self_kv: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> int:
        """
        Assign an open slot to a request completing the encode phase.
        Fills the persistent cross-KV buffer and initial self-KV for that slot.
        """
        slot_idx = self.find_free_slot()
        if slot_idx is None:
            raise RuntimeError("No free slot available in GraphDecodeBuffers!")

        slot = self.slots[slot_idx]
        slot.active = True
        slot.req = req
        slot.decoder_seq_len = prompt_len
        slot.last_token_id = initial_tok
        slot.generated_count = 1
        slot.max_new_tokens = req.max_new_tokens
        slot.eos_token_id = req.eos_token_id
        slot.bucket = get_next_bucket(prompt_len + 1)
        slot.pending_tokens = [initial_tok]
        slot.eos_reached = False
        slot.enc_seq_len = min(enc_kv_layers[0][0].shape[2], self.max_enc_len) if enc_kv_layers else self.max_enc_len
        slot.readback_start = self._step_in_k
        slot.needs_token_upload = True
        slot.reused = slot.assignment_count > 0
        slot.assignment_count += 1
        slot.last_readback_trace = []

        # Copy cross-KV into static cross-cache at slot_idx
        for l in range(len(enc_kv_layers)):
            ck, cv = enc_kv_layers[l]
            # ck shape: [1, num_heads, enc_len, head_dim]
            c_len = min(ck.shape[2], self.max_enc_len)
            if self.is_pretrained_whisper:
                self.cross_cache.layers[l].keys[slot_idx, :, :c_len, :].copy_(ck[0, :, :c_len, :])
                self.cross_cache.layers[l].values[slot_idx, :, :c_len, :].copy_(cv[0, :, :c_len, :])
            else:
                self.g_cross_kv[slot_idx, l, 0, :, :c_len, :].copy_(ck[0, :, :c_len, :])
                self.g_cross_kv[slot_idx, l, 1, :, :c_len, :].copy_(cv[0, :, :c_len, :])

        # Zero self-KV buffers for slot_idx to prevent any stale state from prior requests
        for b in SELF_KV_BUCKETS:
            for l in range(self.num_decoder_layers):
                if self.is_pretrained_whisper:
                    self.self_caches[b].layers[l].keys[slot_idx].zero_()
                    self.self_caches[b].layers[l].values[slot_idx].zero_()
                else:
                    self.g_self_kv_native[b][slot_idx, l].zero_()

        # Copy initial self-KV into all bucket buffers for slot_idx
        if initial_self_kv is not None:
            for l in range(len(initial_self_kv)):
                sk, sv = initial_self_kv[l]
                # sk shape: [1, num_heads, prompt_len, head_dim]
                s_len = min(sk.shape[2], prompt_len)
                for b in SELF_KV_BUCKETS:
                    if s_len <= b:
                        if self.is_pretrained_whisper:
                            self.self_caches[b].layers[l].keys[slot_idx, :, :s_len, :].copy_(sk[0, :, :s_len, :])
                            self.self_caches[b].layers[l].values[slot_idx, :, :s_len, :].copy_(sv[0, :, :s_len, :])
                        else:
                            self.g_self_kv_native[b][slot_idx, l, 0, :, :s_len, :].copy_(sk[0, :, :s_len, :])
                            self.g_self_kv_native[b][slot_idx, l, 1, :, :s_len, :].copy_(sv[0, :, :s_len, :])

        return slot_idx

    def release_slot(self, slot_idx: int) -> None:
        """Mark a slot as inactive and clear references."""
        slot = self.slots[slot_idx]
        slot.active = False
        slot.req = None
        slot.pending_tokens = []
        slot.eos_reached = False
        slot.needs_token_upload = False
        slot.readback_start = 0
        slot.last_readback_trace = []
        if self.zero_slot_buffers_on_release:
            for b in SELF_KV_BUCKETS:
                for l in range(self.num_decoder_layers):
                    if self.is_pretrained_whisper:
                        self.self_caches[b].layers[l].keys[slot_idx].zero_()
                        self.self_caches[b].layers[l].values[slot_idx].zero_()
                    else:
                        self.g_self_kv_native[b][slot_idx, l].zero_()
            for l in range(self.num_decoder_layers):
                if self.is_pretrained_whisper:
                    self.cross_cache.layers[l].keys[slot_idx].zero_()
                    self.cross_cache.layers[l].values[slot_idx].zero_()
                else:
                    self.g_cross_kv[slot_idx, l].zero_()

    def find_free_slot(self) -> Optional[int]:
        for s in self.slots:
            if not s.active:
                return s.slot_idx
        return None

    def active_slot_count(self) -> int:
        return sum(1 for s in self.slots if s.active)

    def capture_all_buckets(self, warmup_steps: int = 3) -> None:
        """
        Pre-captures CUDA graphs for all length buckets [64, 128, 256, 448] on batch_size=4.
        Uses a shared graph memory pool across all bucket graphs.
        """
        side_stream = torch.cuda.Stream()
        side_stream.wait_stream(torch.cuda.current_stream())

        dummy_tok = 50259 if (self.is_pretrained_whisper or self.vocab_size >= 50364) else 1
        for b in SELF_KV_BUCKETS:
            # Prepare dummy mask and positions for bucket capture
            self.g_attn_masks[b].zero_()
            self.g_positions.zero_()
            self.g_token_ids.fill_(dummy_tok)

            # Warmup runs on side stream
            with torch.cuda.stream(side_stream):
                for _ in range(warmup_steps):
                    self._forward_step(b)
            torch.cuda.current_stream().wait_stream(side_stream)

            # Capture in CUDA Graph
            g = torch.cuda.CUDAGraph()
            pool_kwargs = {}
            if self._graph_pool is not None:
                pool_kwargs["pool"] = self._graph_pool

            with torch.cuda.graph(g, stream=side_stream, **pool_kwargs):
                self._forward_step(b)

            if self._graph_pool is None:
                self._graph_pool = g.pool()

            self.graphs[b] = g

        # Clean all self-KV buffers after capture warmup so initial serving sees zeroed buffers
        for b in SELF_KV_BUCKETS:
            if self.is_pretrained_whisper:
                for l in range(self.num_decoder_layers):
                    self.self_caches[b].layers[l].keys.zero_()
                    self.self_caches[b].layers[l].values.zero_()
            else:
                self.g_self_kv_native[b].zero_()

    def _forward_step(self, bucket: int) -> None:
        """Execute one decode step on the static buffers."""
        if self.is_pretrained_whisper:
            out = self.model.hf_model.model.decoder(
                input_ids=self.g_token_ids,
                encoder_hidden_states=self.dummy_enc,
                past_key_values=self.enc_dec_caches[bucket],
                attention_mask=self.g_attn_masks[bucket],
                position_ids=self.g_positions,
                cache_position=self.g_cache_position,
                use_cache=True,
            )
            logits = self.model.hf_model.proj_out(out.last_hidden_state[:, -1, :])
            if self.model.suppress_mask is not None:
                logits = logits.masked_fill(self.model.suppress_mask, -float("inf"))
            tokens = torch.argmax(logits, dim=-1)
            self.g_output_tokens.copy_(tokens)
            self.g_token_ids[:, 0].copy_(tokens)
        else:
            self.model.decode_step_graph_fixed(
                g_token_ids=self.g_token_ids,
                g_positions=self.g_positions,
                g_self_kv=self.g_self_kv_native[bucket],
                g_cross_kv=self.g_cross_kv,
                g_output_tokens=self.g_output_tokens,
                bucket=bucket,
                max_slots=self.max_slots,
                g_attn_mask=self.g_attn_masks[bucket],
                g_cross_mask=self.g_cross_mask,
            )
            self.g_token_ids[:, 0].copy_(self.g_output_tokens)

    def prepare_replay(self) -> int:
        """
        Updates pinned host buffers and copies to GPU persistent buffers before graph replay.
        Returns the active bucket length.
        """
        # Determine maximum sequence length among active slots
        max_len = 0
        for s in self.slots:
            if s.active:
                if s.decoder_seq_len > max_len:
                    max_len = s.decoder_seq_len
        bucket = self.force_bucket or get_next_bucket(max_len + 1)
        if self.force_bucket is None:
            # A slot promoted to a larger bucket has received subsequent KV
            # updates only in that bucket.  Never demote while that slot is
            # active, or replay would resume from an older, stale cache image.
            bucket = max(bucket, max((s.bucket for s in self.slots if s.active), default=bucket))

        # If any slot crossed a bucket boundary, upgrade its KV to the new bucket
        for s in self.slots:
            if s.active and s.bucket < bucket:
                old_b = s.bucket
                cur_len = s.decoder_seq_len
                for l in range(self.num_decoder_layers):
                    # Copy from old bucket to new bucket
                    if self.is_pretrained_whisper:
                        old_k = self.self_caches[old_b].layers[l].keys[s.slot_idx, :, :cur_len, :]
                        old_v = self.self_caches[old_b].layers[l].values[s.slot_idx, :, :cur_len, :]
                        self.self_caches[bucket].layers[l].keys[s.slot_idx, :, :cur_len, :].copy_(old_k)
                        self.self_caches[bucket].layers[l].values[s.slot_idx, :, :cur_len, :].copy_(old_v)
                    else:
                        old_k = self.g_self_kv_native[old_b][s.slot_idx, l, 0, :, :cur_len, :]
                        old_v = self.g_self_kv_native[old_b][s.slot_idx, l, 1, :, :cur_len, :]
                        self.g_self_kv_native[bucket][s.slot_idx, l, 0, :, :cur_len, :].copy_(old_k)
                        self.g_self_kv_native[bucket][s.slot_idx, l, 1, :, :cur_len, :].copy_(old_v)
                s.bucket = bucket

        bank_idx = self._step_counter % self._input_ring_size
        if self._input_copy_in_flight[bank_idx]:
            self._input_copy_events[bank_idx].synchronize()
            self._input_copy_in_flight[bank_idx] = False
        bank = self._input_banks[bank_idx]
        h_token_ids = bank["token_ids"]
        h_positions = bank["positions"]
        h_lengths = bank["lengths"]
        h_active_mask = bank["active_mask"]
        h_finished = bank["finished"]

        # Fill host pinned buffers
        h_mask = bank["attn_masks"][bucket]
        h_mask.zero_()
        h_cross_mask = bank["cross_mask"]
        h_cross_mask.zero_()
        dummy_tok = 50259 if self.is_pretrained_whisper else 1

        for i, s in enumerate(self.slots):
            if s.active:
                h_token_ids[i] = s.last_token_id
                h_positions[i] = s.decoder_seq_len
                h_lengths[i] = s.decoder_seq_len
                h_active_mask[i] = True
                h_finished[i] = s.eos_reached
                # Mask positions > s.decoder_seq_len (i.e. positions >= s.decoder_seq_len + 1)
                h_mask[i, 0, 0, s.decoder_seq_len + 1:] = -10000.0
                h_cross_mask[i, 0, 0, s.enc_seq_len:] = -10000.0
            else:
                h_token_ids[i] = dummy_tok
                h_positions[i] = 0
                h_lengths[i] = 0
                h_active_mask[i] = False
                h_finished[i] = True
                h_mask[i, 0, 0, :] = -10000.0
                h_cross_mask[i, 0, 0, :] = -10000.0

        # In-place copy to persistent GPU buffers before replay
        nb = self.non_blocking_copies
        if self._step_in_k == 0:
            self.g_token_ids[:, 0].copy_(h_token_ids, non_blocking=nb)
            for s in self.slots:
                s.needs_token_upload = False
        else:
            # Requests may be admitted midway through another request's K-window.
            # Upload only fresh slots; established slots keep the autoregressive
            # token written by the preceding graph replay.
            for s in self.slots:
                if s.active and s.needs_token_upload:
                    self.g_token_ids[s.slot_idx, 0].copy_(
                        h_token_ids[s.slot_idx], non_blocking=nb
                    )
                    s.needs_token_upload = False
        self.g_positions[:, 0].copy_(h_positions, non_blocking=nb)
        self.g_lengths.copy_(h_lengths, non_blocking=nb)
        self.g_active_mask.copy_(h_active_mask, non_blocking=nb)
        self.g_finished.copy_(h_finished, non_blocking=nb)
        self.g_attn_masks[bucket].copy_(h_mask, non_blocking=nb)
        self.g_cross_mask.copy_(h_cross_mask, non_blocking=nb)
        if nb:
            self._input_copy_events[bank_idx].record()
            self._input_copy_in_flight[bank_idx] = True

        return bucket

    def replay_step(self, bucket: int) -> None:
        """Replay the pre-captured graph for the given bucket."""
        if bucket not in self.graphs:
            self.capture_all_buckets()
        self.graphs[bucket].replay()
        if self._step_in_k < self.h_token_ring.shape[1]:
            self.h_token_ring[:, self._step_in_k].copy_(
                self.g_output_tokens, non_blocking=self.non_blocking_copies
            )
            self._window_buckets.append(bucket)
            self._step_in_k += 1
        self._step_counter += 1

        # Advance slot sequence lengths host-side
        for s in self.slots:
            if s.active:
                s.decoder_seq_len += 1
                s.generated_count += 1

    def should_readback(self) -> bool:
        """Check if output tokens should be synced back to host."""
        if self._step_counter % self.readback_k == 0:
            return True
        for s in self.slots:
            if s.active and s.generated_count >= s.max_new_tokens:
                return True
        return False

    def sync_tokens(self) -> List[Tuple[SlotInfo, List[int]]]:
        """
        Synchronizes output tokens from GPU to pinned host buffer.
        Returns list of (slot, tokens_list) for active slots.
        """
        self._readback_event.record()
        self._readback_event.synchronize()
        self._input_copy_in_flight = [False] * self._input_ring_size

        results = []
        for i, s in enumerate(self.slots):
            if s.active:
                start = s.readback_start
                toks = [int(self.h_token_ring[i, k].item()) for k in range(start, self._step_in_k)]
                s.last_readback_trace = [
                    {"bucket": self._window_buckets[k], "readback_position": k}
                    for k in range(start, self._step_in_k)
                ]
                if toks:
                    s.last_token_id = toks[-1]
                    self.h_token_ids[i] = toks[-1]
                    s.pending_tokens.extend(toks)
                results.append((s, toks))
                s.readback_start = 0
        self._step_in_k = 0
        self._window_buckets = []
        return results

    def vram_overhead_mb(self) -> float:
        """Report estimated VRAM overhead of persistent static buffers and captured graphs."""
        dt_bytes = 2 if self.dtype in (torch.float16, torch.bfloat16) else 4
        # Self-KV across all 4 buckets
        self_kv_bytes = sum(
            self.max_slots * self.num_decoder_layers * 2 * self.num_heads * b * self.head_dim * dt_bytes
            for b in SELF_KV_BUCKETS
        )
        # Cross-KV
        cross_kv_bytes = (
            self.max_slots * self.num_encoder_layers * 2 * self.num_heads * self.max_enc_len * self.head_dim * dt_bytes
        )
        # Attention masks
        mask_bytes = sum(self.max_slots * b * dt_bytes for b in SELF_KV_BUCKETS)
        total_mb = (self_kv_bytes + cross_kv_bytes + mask_bytes) / (1024 * 1024)
        return total_mb
