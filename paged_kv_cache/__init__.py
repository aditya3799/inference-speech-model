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

import os
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from speech_serving.block_manager import BlockManager

__all__ = ["BlockManager"]
