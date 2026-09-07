"""
Phase 1: Streaming Neural Codec TTS Engine (Decoder-Only Paged Attention).

Implements autoregressive continuous batching for multi-codebook neural codec audio decoders
(e.g., EnCodec / SNAC / Bark / Parler-TTS style codebook tokens per frame).
Reuses paged_kv_cache.block_manager.BlockManager without modification.
"""

import time
import torch
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Callable
from transformers.cache_utils import DynamicCache, DynamicLayer

from paged_kv_cache.block_manager import BlockManager
from .models import StreamingMultiCodebookTTSModel


@dataclass
class TTSRequest:
    """Represents a streaming TTS generation request."""
    id: int
    prompt_codebooks: torch.Tensor # [seq_len, num_codebooks]
    max_new_frames: int
    arrival_time: float = 0.0
    
    # State tracking
    seq_len: int = 0
    block_table: List[int] = field(default_factory=list)
    generated_frames: List[torch.Tensor] = field(default_factory=list) # List of [1, num_codebooks] tensors
    generated_count: int = 0
    done: bool = False
    
    # Timing & streaming telemetry
    start_time: Optional[float] = None
    first_token_time: Optional[float] = None
    last_frame_codebooks: Optional[torch.Tensor] = None # [1, 1, num_codebooks]
    on_frame_callback: Optional[Callable[[int, torch.Tensor], None]] = None


class StreamingTTSEngine:
    """
    Continuous batching engine for streaming multi-codebook TTS generation.
    Allocates KV-cache blocks dynamically via BlockManager as sequence length grows.
    """

    def __init__(
        self,
        model: StreamingMultiCodebookTTSModel,
        block_manager: BlockManager,
        max_batch_size: int = 8,
        device: str = "cuda",
    ):
        self.model = model
        self.block_manager = block_manager
        self.max_batch_size = max_batch_size
        self.device = device

        self.active: List[TTSRequest] = []
        self.queue: List[TTSRequest] = []
        self.completed: List[TTSRequest] = []
        self._next_id = 0

    def submit(
        self,
        prompt_codebooks: torch.Tensor,
        max_new_frames: int = 100,
        arrival_time: float = 0.0,
        on_frame_callback: Optional[Callable[[int, torch.Tensor], None]] = None,
    ) -> int:
        """
        Submit a new TTS audio generation request.
        Args:
            prompt_codebooks: Initial prompt codec frames [prompt_len, num_codebooks]
            max_new_frames: Max audio codebook frames to generate
            arrival_time: Request arrival timestamp
            on_frame_callback: Streaming callback fired when a new codec frame is generated
        """
        req = TTSRequest(
            id=self._next_id,
            prompt_codebooks=prompt_codebooks.to(self.device),
            max_new_frames=max_new_frames,
            arrival_time=arrival_time,
            on_frame_callback=on_frame_callback,
        )
        self._next_id += 1
        self.queue.append(req)
        return req.id

    def _admit_from_queue(self):
        """Admit queued requests, allocate initial KV blocks, and execute prompt prefill."""
        while len(self.active) < self.max_batch_size and self.queue:
            req = self.queue.pop(0)
            req.start_time = time.perf_counter()
            prompt_len = req.prompt_codebooks.shape[0]
            req.seq_len = prompt_len

            # 1. Allocate blocks for prompt prefill
            needed_blocks = (prompt_len + self.block_manager.block_size - 1) // self.block_manager.block_size
            for _ in range(needed_blocks):
                req.block_table.append(self.block_manager.allocate_block())

            # 2. Forward prefill pass
            input_tensor = req.prompt_codebooks.unsqueeze(0) # [1, prompt_len, num_codebooks]
            with torch.no_grad():
                out = self.model(input_codebooks=input_tensor, use_cache=True)

            # 3. Store prefill KV states into physical block pool across all layers
            for l in range(self.block_manager.num_layers):
                k = out.past_key_values.layers[l].keys   # [1, num_heads, prompt_len, head_dim]
                v = out.past_key_values.layers[l].values # [1, num_heads, prompt_len, head_dim]

                for blk_idx_in_req, blk_id in enumerate(req.block_table):
                    start_t = blk_idx_in_req * self.block_manager.block_size
                    end_t = min(start_t + self.block_manager.block_size, prompt_len)
                    tok_count = end_t - start_t

                    k_slice = k[0, :, start_t:end_t, :].permute(1, 0, 2)
                    v_slice = v[0, :, start_t:end_t, :].permute(1, 0, 2)

                    self.block_manager.key_pool[l][blk_id, :tok_count] = k_slice
                    self.block_manager.value_pool[l][blk_id, :tok_count] = v_slice

            # 4. Sample first frame codebooks [1, num_codebooks]
            # out.logits shape: [1, prompt_len, num_codebooks, vocab_size]
            next_frame_logits = out.logits[0, -1, :, :] # [num_codebooks, vocab_size]
            next_frame_tokens = torch.argmax(next_frame_logits, dim=-1) # [num_codebooks]

            req.last_frame_codebooks = next_frame_tokens.unsqueeze(0).unsqueeze(0) # [1, 1, num_codebooks]
            req.generated_frames.append(next_frame_tokens)
            req.generated_count += 1
            req.first_token_time = time.perf_counter()

            if req.on_frame_callback is not None:
                try:
                    req.on_frame_callback(req.id, next_frame_tokens)
                except Exception as e:
                    print(f"Warning: Exception in frame callback for req {req.id}: {e}")

            if req.generated_count >= req.max_new_frames:
                req.done = True
                self.block_manager.free_blocks(req.block_table)
                req.block_table = []
                self.completed.append(req)
            else:
                self.active.append(req)

    def step(self) -> bool:
        """
        Execute one decode step across active TTS requests.
        For each request:
        1. Allocate block if crossing block boundary.
        2. Gather KV history from non-contiguous blocks into DynamicCache.
        3. Forward single new multi-codebook frame.
        4. Write newly computed (K, V) into physical block pool.
        5. Evict completed requests and immediately free physical blocks.
        """
        self._admit_from_queue()
        if not self.active:
            return False

        still_active = []
        for req in self.active:
            try:
                # 1. Allocate block if boundary crossed
                if req.seq_len % self.block_manager.block_size == 0:
                    req.block_table.append(self.block_manager.allocate_block())

                # 2. Gather KV history into DynamicCache
                cache = DynamicCache()
                for l in range(self.block_manager.num_layers):
                    gathered_k = self.block_manager.key_pool[l][req.block_table]
                    gathered_v = self.block_manager.value_pool[l][req.block_table]

                    flat_k = gathered_k.view(-1, self.block_manager.num_kv_heads, self.block_manager.head_dim)[:req.seq_len]
                    flat_v = gathered_v.view(-1, self.block_manager.num_kv_heads, self.block_manager.head_dim)[:req.seq_len]

                    k_seq = flat_k.permute(1, 0, 2).contiguous().unsqueeze(0)
                    v_seq = flat_v.permute(1, 0, 2).contiguous().unsqueeze(0)

                    layer = DynamicLayer()
                    layer.keys = k_seq
                    layer.values = v_seq
                    layer.is_initialized = True
                    cache.layers.append(layer)

                # 3. Forward step with single input frame
                with torch.no_grad():
                    out = self.model(input_codebooks=req.last_frame_codebooks, past_key_values=cache, use_cache=True)

                # 4. Write new (K, V) token to block pool
                blk_id = req.block_table[req.seq_len // self.block_manager.block_size]
                offset = req.seq_len % self.block_manager.block_size
                for l in range(self.block_manager.num_layers):
                    new_k = out.past_key_values.layers[l].keys[0, :, -1, :]
                    new_v = out.past_key_values.layers[l].values[0, :, -1, :]
                    self.block_manager.key_pool[l][blk_id, offset] = new_k
                    self.block_manager.value_pool[l][blk_id, offset] = new_v

                # 5. Update sequence state and sample next multi-codebook frame
                req.seq_len += 1
                next_frame_logits = out.logits[0, -1, :, :] # [num_codebooks, vocab_size]
                next_frame_tokens = torch.argmax(next_frame_logits, dim=-1) # [num_codebooks]

                req.last_frame_codebooks = next_frame_tokens.unsqueeze(0).unsqueeze(0)
                req.generated_frames.append(next_frame_tokens)
                req.generated_count += 1

                if req.on_frame_callback is not None:
                    try:
                        req.on_frame_callback(req.id, next_frame_tokens)
                    except Exception as e:
                        print(f"Warning: Exception in frame callback for req {req.id}: {e}")

                # 6. Check completion
                if req.generated_count >= req.max_new_frames:
                    req.done = True
                    self.block_manager.free_blocks(req.block_table)
                    req.block_table = []
                    self.completed.append(req)
                else:
                    still_active.append(req)

            except Exception as e:
                # Disconnect / error cleanup guard: deterministic block release
                print(f"Error executing step for TTS request {req.id}: {e}")
                self.block_manager.free_blocks(req.block_table)
                req.block_table = []
                req.done = True
                self.completed.append(req)

        self.active = still_active
        return True

    def run_until_done(self, max_steps: int = 10000) -> List[TTSRequest]:
        """Run continuous batching until queue and active requests are finished."""
        steps = 0
        while (self.active or self.queue) and steps < max_steps:
            did_work = self.step()
            if not did_work:
                break
            steps += 1
        return self.completed
