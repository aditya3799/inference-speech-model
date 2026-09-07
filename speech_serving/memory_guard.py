"""
Memory Safety & Telemetry Module for Dual-Cache Speech Inference Engine.

Enforces strict GPU memory management rules:
1. High-water mark monitoring: Triggers bulk eviction when pool utilization exceeds high-water mark (e.g. 90%).
2. Periodic maintenance torch.cuda.empty_cache(): Invoked ONLY after bulk evictions, NOT per request.
3. Telemetry & Telemetry Logging: Real-time tracking of pool capacity, allocated blocks, active requests, and eviction events.
"""

import time
import logging
import torch
from typing import Dict, Any, List, Optional

# Set up dedicated logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("SpeechMemoryGuard")


class MemoryGuard:
    """
    Monitors dual-cache block pool memory utilization and enforces memory safety.
    """

    def __init__(
        self,
        high_water_mark_pct: float = 90.0,
        enable_periodic_gc: bool = True,
    ):
        self.high_water_mark_pct = high_water_mark_pct
        self.enable_periodic_gc = enable_periodic_gc
        
        # Telemetry metrics
        self.eviction_events_count: int = 0
        self.last_empty_cache_time: float = time.perf_counter()
        self.total_freed_blocks_count: int = 0

    def check_high_water_mark(
        self,
        encoder_allocated: int,
        encoder_total: int,
        decoder_allocated: int,
        decoder_total: int,
    ) -> bool:
        """
        Calculates combined pool utilization percentage.
        Returns True if total utilization crosses high-water mark.
        """
        total_blocks = encoder_total + decoder_total
        if total_blocks == 0:
            return False

        allocated_blocks = encoder_allocated + decoder_allocated
        utilization_pct = (allocated_blocks / total_blocks) * 100.0

        if utilization_pct >= self.high_water_mark_pct:
            logger.warning(
                f"[HIGH WATER MARK EXCEEDED] Utilization={utilization_pct:.1f}% "
                f"(Allocated={allocated_blocks}/{total_blocks} blocks). Triggering memory pressure alert!"
            )
            return True
        return False

    def trigger_bulk_eviction_cleanup(self, freed_blocks: int, reason: str = "high_water_mark"):
        """
        Executes maintenance operations after bulk block eviction events.
        Calls torch.cuda.empty_cache() safely after bulk evictions.
        """
        self.eviction_events_count += 1
        self.total_freed_blocks_count += freed_blocks
        now_str = time.strftime("%H:%M:%S")

        msg = (
            f"  >>> [BULK EVICTION EVENT #{self.eviction_events_count} at {now_str}] "
            f"Reason: '{reason}' | Freed {freed_blocks} physical blocks. "
            f"Executing maintenance torch.cuda.empty_cache()."
        )
        print(msg)
        logger.info(msg)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            self.last_empty_cache_time = time.perf_counter()

    def log_telemetry(
        self,
        active_requests: int,
        queued_requests: int,
        encoder_stats: Dict[str, Any],
        decoder_stats: Dict[str, Any],
    ):
        """Logs continuous structured memory telemetry."""
        logger.info(
            f"[TELEMETRY] ActiveReqs={active_requests} QueuedReqs={queued_requests} | "
            f"EncUtil={encoder_stats['utilization_pct']}% ({encoder_stats['allocated_blocks']}/{encoder_stats['num_blocks']} blks, "
            f"HitRate={encoder_stats.get('prefix_hit_rate_pct', 0)}%) | "
            f"DecUtil={decoder_stats['utilization_pct']}% ({decoder_stats['allocated_blocks']}/{decoder_stats['num_blocks']} blks) | "
            f"Evictions={self.eviction_events_count}"
        )
