"""
BlockManager: Paged KV-Cache Physical Block Pool for Causal Decoders.

This is a self-contained re-implementation of the `paged_kv_cache.BlockManager`
interface used by the decoder-side paged attention engine.

NOTE: The public `paged_kv_cache` package does not ship with this repository.
This module provides a fully compatible replacement so that a clean checkout
can install and pass all tests without any external dependency on that package.

Interface implemented:
    BlockManager(num_layers, num_kv_heads, head_dim, block_size,
                 num_blocks=None, dtype, device)
    Attributes:
        num_layers, num_kv_heads, head_dim, block_size, dtype, num_blocks
        key_pool[l]   : Tensor [num_blocks, block_size, num_kv_heads, head_dim]
        value_pool[l] : Tensor [num_blocks, block_size, num_kv_heads, head_dim]
    Methods:
        allocate_block() -> int
        free_blocks(block_ids: List[int])
        get_stats() -> dict  (keys: num_blocks, allocated_blocks, utilization_pct)
    Properties:
        allocated_block_count -> int
        num_free_blocks       -> int
"""

from __future__ import annotations

import torch
from typing import Dict, List, Optional, Tuple, Any


class BlockManager:
    """
    Manages a fixed-size pool of physical KV-cache blocks for causal (decoder)
    autoregressive generation.

    Each block stores ``block_size`` tokens across all KV heads and layers.
    Allocation is via a free list (O(1) pop/append). Blocks are freed immediately
    and unconditionally — no reference counting; caller is responsible for not
    freeing blocks that are still in use.

    Leak detection: ``get_stats()`` reports ``allocated_blocks``; callers
    should assert this is 0 after all requests complete.
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        block_size: int,
        num_blocks: Optional[int] = None,
        dtype: torch.dtype = torch.float16,
        device: str = "cuda",
    ):
        if num_layers <= 0:
            raise ValueError(f"num_layers must be > 0, got {num_layers}")
        if num_kv_heads <= 0:
            raise ValueError(f"num_kv_heads must be > 0, got {num_kv_heads}")
        if head_dim <= 0:
            raise ValueError(f"head_dim must be > 0, got {head_dim}")
        if block_size <= 0:
            raise ValueError(f"block_size must be > 0, got {block_size}")

        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.dtype = dtype
        self.device = device

        if num_blocks is None:
            num_blocks = self._auto_num_blocks()
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be > 0, got {num_blocks}")
        self.num_blocks = num_blocks

        # Physical pool: per-layer tensors of shape
        # [num_blocks, block_size, num_kv_heads, head_dim]
        self.key_pool: List[torch.Tensor] = [
            torch.zeros(
                (num_blocks, block_size, num_kv_heads, head_dim),
                dtype=dtype,
                device=device,
            )
            for _ in range(num_layers)
        ]
        self.value_pool: List[torch.Tensor] = [
            torch.zeros(
                (num_blocks, block_size, num_kv_heads, head_dim),
                dtype=dtype,
                device=device,
            )
            for _ in range(num_layers)
        ]

        # Free list — all blocks available at start.
        self._free_list: List[int] = list(range(num_blocks))
        # Shadow set for O(1) leak detection (not needed for correctness).
        self._allocated: set = set()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _auto_num_blocks(self) -> int:
        """Estimate a reasonable block count from available GPU memory."""
        if not torch.cuda.is_available() or self.device == "cpu":
            return 512

        total = torch.cuda.get_device_properties(self.device).total_memory
        allocated = torch.cuda.memory_allocated(self.device)
        available = max(0, int((total - allocated) * 0.40))  # use ≤40% of free mem
        element_size = torch.tensor([], dtype=self.dtype).element_size()
        bytes_per_block = (
            self.num_layers
            * 2  # key + value
            * self.block_size
            * self.num_kv_heads
            * self.head_dim
            * element_size
        )
        blocks = max(64, available // bytes_per_block)
        return int(blocks)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def allocate_block(self) -> int:
        """
        Allocate a single physical block from the free list.

        Returns:
            block_id (int): Index into key_pool / value_pool.

        Raises:
            RuntimeError: When all blocks are exhausted.
        """
        if not self._free_list:
            raise RuntimeError(
                f"BlockManager: out of physical blocks! "
                f"All {self.num_blocks} blocks are allocated."
            )
        blk_id = self._free_list.pop()
        self._allocated.add(blk_id)
        return blk_id

    def free_blocks(self, block_ids: List[int]) -> None:
        """
        Return a list of blocks to the free pool immediately.

        Duplicate or already-free block IDs are silently ignored to prevent
        double-free panics; determinism is the caller's responsibility.

        Args:
            block_ids: List of physical block indices to free.
        """
        for blk_id in block_ids:
            if blk_id in self._allocated:
                self._allocated.discard(blk_id)
                self._free_list.append(blk_id)
            # else: already free / invalid — silently skip

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def allocated_block_count(self) -> int:
        """Number of currently allocated (in-use) blocks."""
        return len(self._allocated)

    @property
    def num_free_blocks(self) -> int:
        """Number of blocks available for allocation."""
        return len(self._free_list)

    # ------------------------------------------------------------------
    # Telemetry
    # ------------------------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        """
        Return a telemetry snapshot of pool utilisation.

        Keys guaranteed (compatible with the paged_kv_cache interface):
            num_blocks          : total blocks in pool
            allocated_blocks    : currently allocated
            utilization_pct     : float percentage (0.0–100.0)
            free_blocks         : currently free
            pool_memory_mb      : approximate VRAM consumed by both pools
        """
        allocated = self.allocated_block_count
        utilization = (
            round(allocated / self.num_blocks * 100.0, 2)
            if self.num_blocks > 0
            else 0.0
        )
        element_size = torch.tensor([], dtype=self.dtype).element_size()
        pool_bytes = (
            self.num_blocks
            * self.block_size
            * self.num_kv_heads
            * self.head_dim
            * 2  # key + value
            * self.num_layers
            * element_size
        )
        return {
            "num_blocks": self.num_blocks,
            "allocated_blocks": allocated,
            "free_blocks": self.num_free_blocks,
            "utilization_pct": utilization,
            "pool_memory_mb": round(pool_bytes / (1024 ** 2), 2),
        }

    def check_no_leaks(self) -> bool:
        """
        Return True iff all blocks have been returned (zero leaks).
        Useful for post-test assertions.
        """
        return self.allocated_block_count == 0

    def __repr__(self) -> str:
        s = self.get_stats()
        return (
            f"BlockManager(layers={self.num_layers}, heads={self.num_kv_heads}, "
            f"head_dim={self.head_dim}, block_size={self.block_size}, "
            f"blocks={s['allocated_blocks']}/{self.num_blocks} "
            f"({s['utilization_pct']}% used), "
            f"dtype={self.dtype}, device={self.device})"
        )
