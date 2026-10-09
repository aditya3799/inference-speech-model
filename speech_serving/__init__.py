"""
Speech Serving Package: Paged Dual-Cache Serving Engine for Speech Models.
Includes Streaming Neural Codec TTS, Encoder Paged Attention, Dual-Cache Cross-Attention,
and Phase-Aware Scheduling.

NOTE: ``paged_kv_cache`` is a local compatibility shim (see ``paged_kv_cache/``
in the repository root).  No external ``paged_kv_cache`` package is required.
"""
import os
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from .models import StreamingMultiCodebookTTSModel, WhisperClassSpeechEncoderDecoderModel
from .tts_engine import StreamingTTSEngine
from .block_manager import BlockManager
from .encoder_block_manager import EncoderBlockManager
from .encoder_attention import EncoderAttentionBackend
from .dual_cache_manager import DualCacheManager
from .phase_aware_scheduler import PhaseAwareSpeechScheduler, RequestPhase
from .memory_guard import MemoryGuard

