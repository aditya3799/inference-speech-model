"""
paged_kv_cache.block_manager — compatibility shim.

Re-exports BlockManager from speech_serving.block_manager so existing imports
of `from paged_kv_cache.block_manager import BlockManager` work without the
external package.
"""

from speech_serving.block_manager import BlockManager

__all__ = ["BlockManager"]
