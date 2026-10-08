"""
paged_kv_cache compatibility shim.

The public ``paged_kv_cache`` package is not distributed with this repository.
This package re-exports the self-contained ``BlockManager`` from
``speech_serving.block_manager`` so that all existing imports of the form

    from paged_kv_cache.block_manager import BlockManager

continue to work on a clean checkout without any external dependencies.

See ``speech_serving/block_manager.py`` for the full implementation and
interface documentation.
"""

from speech_serving.block_manager import BlockManager

__all__ = ["BlockManager"]
