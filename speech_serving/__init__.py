"""
Speech Serving Package: Paged Dual-Cache Serving Engine for Speech Models.
Includes Streaming Neural Codec TTS, Encoder Paged Attention, Dual-Cache Cross-Attention,
and Phase-Aware Scheduling.
"""

import os
import sys

# Ensure parent directory is in sys.path so existing paged_kv_cache module can be imported
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from .models import StreamingMultiCodebookTTSModel, WhisperClassSpeechEncoderDecoderModel
from .tts_engine import StreamingTTSEngine
from .encoder_block_manager import EncoderBlockManager
from .encoder_attention import EncoderAttentionBackend
from .dual_cache_manager import DualCacheManager
from .phase_aware_scheduler import PhaseAwareSpeechScheduler, RequestPhase
from .memory_guard import MemoryGuard

