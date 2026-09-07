"""
Phase 3: Dual-Cache Manager (Encoder + Decoder Independent Cache Pools).

Orchestrates two independent cache pools per request:
1. Encoder Cache: Non-causal, read-only, reference-counted, shared audio prefix blocks.
2. Decoder Cache: Causal, growing incrementally per step, sequence-bound.
Enforces explicit, deterministic resource release paths on completion or failure.
"""

import torch
from typing import List, Tuple, Optional, Dict, Any

from paged_kv_cache.block_manager import BlockManager
from .encoder_block_manager import EncoderBlockManager


class DualCacheManager:
    """
    Coordinator for dual-cache serving engines.
    Tracks independent block allocation tables for encoder and decoder caches.
    """

    def __init__(
        self,
        encoder_block_manager: EncoderBlockManager,
        decoder_block_manager: BlockManager,
    ):
        self.encoder_bm = encoder_block_manager
        self.decoder_bm = decoder_block_manager

    def allocate_encoder_blocks(
        self,
        seq_len: int,
        content_hashes: Optional[List[Optional[str]]] = None,
    ) -> Tuple[List[int], int]:
        """
        Allocate physical blocks for an encoder sequence.
        Returns:
            Tuple[List[int], int]: (encoder_block_table, cache_hits_count)
        """
        block_size = self.encoder_bm.block_size
        needed_blocks = (seq_len + block_size - 1) // block_size
        block_table = []
        cache_hits = 0

        try:
            for b_idx in range(needed_blocks):
                ch = content_hashes[b_idx] if (content_hashes is not None and b_idx < len(content_hashes)) else None
                blk_id, hit = self.encoder_bm.allocate_block(content_hash=ch)
                block_table.append(blk_id)
                if hit:
                    cache_hits += 1
        except Exception as e:
            self.encoder_bm.release_blocks(block_table)
            raise e

        return block_table, cache_hits

    def allocate_decoder_block(self) -> int:
        """Allocate a single physical block for decoder paged KV cache."""
        return self.decoder_bm.allocate_block()

    def release_request_resources(
        self,
        encoder_block_table: List[int],
        decoder_block_table: List[int],
    ) -> Dict[str, int]:
        """
        Deterministic, immediate release of all physical blocks allocated to a request.
        Decrements ref_counts for encoder blocks and returns decoder blocks immediately to pool.

        Returns:
            Dict[str, int]: Summary of freed blocks count.
        """
        freed_encoder = self.encoder_bm.release_blocks(encoder_block_table)
        
        decoder_count = len(decoder_block_table)
        self.decoder_bm.free_blocks(decoder_block_table)

        return {
            "freed_encoder_blocks": freed_encoder,
            "freed_decoder_blocks": decoder_count,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Return combined status telemetry across both cache pools."""
        return {
            "encoder": self.encoder_bm.get_stats(),
            "decoder": self.decoder_bm.get_stats(),
        }
