"""Default-off diagnostic phase timing helpers."""

import time
from contextlib import contextmanager
from typing import Any, Dict, Iterable

import torch


PHASES = (
    "queue_admission_wait",
    "encoder_forward",
    "cross_kv_pool_copy",
    "cross_kv_contiguous_or_graph_copy",
    "prefill",
    "graph_replay",
    "host_between_replays",
    "readback_wait",
    "block_allocate",
    "block_release",
    "other_host_between_requests",
)


def new_phase_record() -> Dict[str, Any]:
    return {
        "phases": {},
        "encoder_batched": None,
        "slot_hold_steps_after_eos": 0,
        "semantics": "request-experienced elapsed time; shared batch work is recorded for every participating request",
    }


class PhaseTimer:
    """Uses boundary synchronization, perf_counter, and CUDA events only when enabled."""

    def __init__(self, enabled: bool, device: str = "cuda"):
        self.enabled = bool(enabled)
        self.device = device
        self.cuda_enabled = self.enabled and torch.cuda.is_available() and str(device).startswith("cuda")

    @staticmethod
    def ensure(target: Any) -> Dict[str, Any]:
        if isinstance(target, dict):
            if "phases" in target:
                return target
            return target.setdefault("phase_timing", new_phase_record())
        if getattr(target, "phase_timing", None) is None:
            target.phase_timing = new_phase_record()
        return target.phase_timing

    def add(self, targets: Iterable[Any], phase: str, wall_ms: float, cuda_ms: float, calls: int = 1) -> None:
        if not self.enabled:
            return
        for target in targets:
            record = self.ensure(target)
            item = record["phases"].setdefault(phase, {"wall_ms": 0.0, "cuda_ms": 0.0, "calls": 0})
            item["wall_ms"] += float(wall_ms)
            item["cuda_ms"] += float(cuda_ms)
            item["calls"] += int(calls)

    @contextmanager
    def measure(self, targets: Iterable[Any], phase: str):
        targets = list(targets)
        if not self.enabled:
            yield
            return
        if self.cuda_enabled:
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        else:
            start_event = end_event = None
        started = time.perf_counter()
        try:
            yield
        finally:
            if self.cuda_enabled:
                end_event.record()
                torch.cuda.synchronize()
                cuda_ms = start_event.elapsed_time(end_event)
            else:
                cuda_ms = 0.0
            wall_ms = (time.perf_counter() - started) * 1000.0
            self.add(targets, phase, wall_ms, cuda_ms)


def finalize_phases(record: Dict[str, Any], measured_phases=PHASES) -> Dict[str, Any]:
    """Emit explicit not-measured entries instead of converting missing data to zero."""
    phases = record.setdefault("phases", {})
    for phase in measured_phases:
        phases.setdefault(phase, "not measured")
    return record
