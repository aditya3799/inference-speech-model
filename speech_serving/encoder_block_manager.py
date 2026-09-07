"""
Phase 2: Encoder-Side Paged Attention Cache Manager with Audio Prefix/Block Sharing.

Implements non-causal block-based memory management for encoder Key/Value states:
- Non-causal: computed once per audio sequence or streaming chunk.
- Block/Prefix Sharing: Content-hashes audio mel-spectrogram block features for deduplication.
- Deterministic Reference Counting: Tracks ref_count per physical block. Blocks are freed ONLY when ref_count == 0.
- Explicit Hit-Rate Telemetry: Measures hit rate separately from decoder-side prefix caching.
"""

import math
import hashlib
import torch
from typing import List, Tuple, Optional, Dict, Any


class EncoderBlockManager:
    """
    Manages physical KV-cache memory blocks for speech encoders (non-causal, bidirectional).
    Supports reference counting and audio prefix block sharing.
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        block_size: int = 16,
        num_blocks: Optional[int] = None,
        gpu_memory_utilization: float = 0.85,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.dtype = dtype
        self.device = device
        self.gpu_memory_utilization = gpu_memory_utilization

        self.bytes_per_token = num_layers * 2 * num_heads * head_dim * torch.tensor([], dtype=dtype).element_size()
        self.bytes_per_block = self.bytes_per_token * block_size

        if num_blocks is None:
            self.num_blocks, self.available_kv_memory_bytes = self._calculate_available_blocks()
        else:
            self.num_blocks = num_blocks
            self.available_kv_memory_bytes = self.num_blocks * self.bytes_per_block

        # Physical pool allocation: [num_blocks, block_size, num_heads, head_dim]
        self.key_pool: List[torch.Tensor] = [
            torch.zeros(
                (self.num_blocks, self.block_size, self.num_heads, self.head_dim),
                dtype=self.dtype,
                device=self.device,
            )
            for _ in range(self.num_layers)
        ]
        self.value_pool: List[torch.Tensor] = [
            torch.zeros(
                (self.num_blocks, self.block_size, self.num_heads, self.head_dim),
                dtype=self.dtype,
                device=self.device,
            )
            for _ in range(self.num_layers)
        ]

        # Free block tracking & Reference counting
        self.free_block_list: List[int] = list(range(self.num_blocks))
        self.ref_counts: Dict[int, int] = {i: 0 for i in range(self.num_blocks)}

        # Audio Prefix / Block Sharing table: content_hash -> physical_block_id
        self.prefix_hash_table: Dict[str, int] = {}
        self.block_to_hash: Dict[int, str] = {}

        # Retained audio prompt LRU cache tracking: block_id -> last_access_timestamp
        self.retained_caches: Dict[str, List[int]] = {}

        # Hit Rate Telemetry
        self.total_lookups: int = 0
        self.cache_hits: int = 0

    def _calculate_available_blocks(self) -> Tuple[int, int]:
        """Calculates available memory and max block capacity for encoder KV pool."""
        if not torch.cuda.is_available() or self.device == "cpu":
            default_blocks = 512
            return default_blocks, default_blocks * self.bytes_per_block

        total_gpu_mem = torch.cuda.get_device_properties(self.device).total_memory
        allocated_mem = torch.cuda.memory_allocated(self.device)
        target_usable = int(total_gpu_mem * self.gpu_memory_utilization)
        available_kv_mem = max(0, target_usable - allocated_mem)

        calc_blocks = available_kv_mem // self.bytes_per_block
        num_blocks = max(64, int(calc_blocks))
        return num_blocks, available_kv_mem

    def _hash_audio_block(self, audio_features_slice: torch.Tensor) -> str:
        """
        Computes a deterministic MD5 hash for a slice of audio features / mel frames.
        Args:
            audio_features_slice: Tensor slice [block_size, ...]
        """
        feat_bytes = audio_features_slice.detach().cpu().numpy().tobytes()
        return hashlib.md5(feat_bytes).hexdigest()

    def allocate_block(self, content_hash: Optional[str] = None) -> Tuple[int, bool]:
        """
        Allocates a physical block, or reuses an existing block if content_hash matches.

        Returns:
            Tuple[int, bool]: (physical_block_id, is_cache_hit)
        """
        self.total_lookups += 1

        # Check prefix cache hit
        if content_hash is not None and content_hash in self.prefix_hash_table:
            blk_id = self.prefix_hash_table[content_hash]
            if self.ref_counts[blk_id] > 0:
                self.ref_counts[blk_id] += 1
                self.cache_hits += 1
                return blk_id, True
            else:
                # Stale hash entry, remove
                del self.prefix_hash_table[content_hash]

        if not self.free_block_list:
            raise RuntimeError(
                f"Out of Encoder KV-cache blocks! All {self.num_blocks} blocks are allocated or referenced."
            )

        blk_id = self.free_block_list.pop(0)
        self.ref_counts[blk_id] = 1

        if content_hash is not None:
            self.prefix_hash_table[content_hash] = blk_id
            self.block_to_hash[blk_id] = content_hash

        return blk_id, False

    def release_block(self, blk_id: int) -> bool:
        """
        Decrements the reference count of a physical block.
        Returns block to free list IMMEDIATELY when ref_count hits 0.

        Returns:
            bool: True if block was freed, False if still referenced elsewhere.
        """
        if blk_id not in self.ref_counts or self.ref_counts[blk_id] <= 0:
            return False

        self.ref_counts[blk_id] -= 1
        if self.ref_counts[blk_id] == 0:
            # Clean hash table mapping
            if blk_id in self.block_to_hash:
                h = self.block_to_hash.pop(blk_id)
                self.prefix_hash_table.pop(h, None)

            self.free_block_list.append(blk_id)
            return True
        return False

    def release_blocks(self, block_ids: List[int]) -> int:
        """
        Release a list of physical blocks deterministically.
        Returns count of newly freed physical blocks.
        """
        freed = 0
        for b in block_ids:
            if self.release_block(b):
                freed += 1
        return freed

    def store_encoder_kv(
        self,
        block_id: int,
        layer_idx: int,
        k_tensor: torch.Tensor,
        v_tensor: torch.Tensor,
    ):
        """
        Stores computed key and value tensors for a block slice into physical pool.
        Args:
            k_tensor: [tok_count, num_heads, head_dim]
            v_tensor: [tok_count, num_heads, head_dim]
        """
        tok_count = k_tensor.shape[0]
        self.key_pool[layer_idx][block_id, :tok_count] = k_tensor
        self.value_pool[layer_idx][block_id, :tok_count] = v_tensor

    @property
    def allocated_block_count(self) -> int:
        return self.num_blocks - len(self.free_block_list)

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_block_list)

    @property
    def hit_rate(self) -> float:
        if self.total_lookups == 0:
            return 0.0
        return round((self.cache_hits / self.total_lookups) * 100, 2)

    def get_stats(self) -> Dict[str, Any]:
        """Return metrics and pool utilization statistics."""
        allocated = self.allocated_block_count
        return {
            "num_blocks": self.num_blocks,
            "block_size": self.block_size,
            "allocated_blocks": allocated,
            "free_blocks": len(self.free_block_list),
            "utilization_pct": round((allocated / self.num_blocks) * 100, 2) if self.num_blocks > 0 else 0.0,
            "total_lookups": self.total_lookups,
            "cache_hits": self.cache_hits,
            "prefix_hit_rate_pct": self.hit_rate,
            "pool_memory_mb": round((self.num_blocks * self.bytes_per_block) / (1024 ** 2), 2),
        }
