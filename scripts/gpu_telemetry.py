"""
scripts/gpu_telemetry.py

Background GPU & CPU telemetry monitors for empirical gating.
Requirements (A1):
1. Poll nvidia-smi every 500 ms during runs, including GPU utilization.
2. Compute gate metrics over BUSY samples only (utilization >= 80%):
   - median and min clock
   - max temperature
   - fraction of busy samples with active throttle reason
3. Gate conditions:
   - throttle fraction on busy samples <= 10%
   - max temperature <= 88 C
   - busy-sample median clock of every system within 10% of each other
   - sequential canary drift between start and end <= 15%
   (Absolute clock threshold dropped, but logged).
4. CPU state monitor: poll typeperf "% Processor Performance" every 1s,
   reporting median and min per system.
"""

import threading
import subprocess
import time
import os
from typing import List, Dict, Any, Tuple, Optional
import numpy as np


def get_max_boost_clock() -> float:
    """Queries max graphics boost clock from nvidia-smi."""
    try:
        out = subprocess.check_output([
            "nvidia-smi",
            "--query-gpu=clocks.max.gr",
            "--format=csv,noheader,nounits"
        ], stderr=subprocess.DEVNULL).decode().strip()
        val = float(out.split("\n")[0].strip())
        return val if val > 0 else 2100.0
    except Exception:
        return 2100.0


class GpuTelemetryMonitor:
    """
    Background thread that samples nvidia-smi every interval_sec during benchmark execution.
    Captures clock, power, temperature, throttle reason, and GPU utilization.
    """
    def __init__(self, interval_sec: float = 0.5):
        self.interval = interval_sec
        self.samples: List[Dict[str, Any]] = []
        self._stop_event = threading.Event()
        self._thread = None
        self._lock = threading.Lock()

    def start(self):
        with self._lock:
            self.samples.clear()
            self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self) -> Dict[str, Any]:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        return self.get_summary()

    def _poll_loop(self):
        while not self._stop_event.is_set():
            try:
                out = subprocess.check_output([
                    "nvidia-smi",
                    "--query-gpu=clocks.gr,power.draw,temperature.gpu,clocks_throttle_reasons.active,utilization.gpu",
                    "--format=csv,noheader,nounits"
                ], stderr=subprocess.DEVNULL).decode().strip()
                line = out.split("\n")[0].strip()
                parts = [p.strip() for p in line.split(",")]
                clock = float(parts[0])
                power = float(parts[1])
                temp = float(parts[2])
                active_throttle = parts[3]
                util_gpu = float(parts[4])

                # Determine active throttling (exclude 0x0000000000000001 = gpu_idle)
                raw_int = int(active_throttle, 16) if active_throttle.startswith("0x") else 0
                is_throttled = (raw_int & ~0x1) != 0

                with self._lock:
                    self.samples.append({
                        "timestamp": time.time(),
                        "clock_mhz": clock,
                        "power_w": power,
                        "temp_c": temp,
                        "utilization_gpu": util_gpu,
                        "active_throttle_hex": active_throttle,
                        "is_throttled": is_throttled,
                    })
            except Exception:
                pass
            time.sleep(self.interval)

    def get_summary(self) -> Dict[str, Any]:
        with self._lock:
            if not self.samples:
                return {
                    "sample_count": 0,
                    "busy_sample_count": 0,
                    "median_clock_mhz": 0.0,
                    "min_clock_mhz": 0.0,
                    "max_clock_mhz": 0.0,
                    "max_temp_c": 0.0,
                    "mean_power_w": 0.0,
                    "mean_utilization_pct": 0.0,
                    "throttle_sample_pct": 0.0,
                    "busy_median_clock_mhz": 0.0,
                    "busy_min_clock_mhz": 0.0,
                    "busy_max_clock_mhz": 0.0,
                    "busy_max_temp_c": 0.0,
                    "busy_throttle_sample_pct": 0.0,
                }

            all_clocks = [s["clock_mhz"] for s in self.samples]
            all_temps = [s["temp_c"] for s in self.samples]
            all_powers = [s["power_w"] for s in self.samples]
            all_utils = [s["utilization_gpu"] for s in self.samples]
            all_throttled_count = sum(1 for s in self.samples if s["is_throttled"])

            # Filter for BUSY samples only (utilization >= 80%) per Requirement A1
            busy_samples = [s for s in self.samples if s["utilization_gpu"] >= 80.0]
            if not busy_samples:
                # If no samples reached 80% (e.g. short test), fallback to top 25% highest utilization samples
                thresh = float(np.percentile(all_utils, 75)) if all_utils else 0.0
                busy_samples = [s for s in self.samples if s["utilization_gpu"] >= thresh]

            busy_clocks = [s["clock_mhz"] for s in busy_samples]
            busy_temps = [s["temp_c"] for s in busy_samples]
            busy_throttled_count = sum(1 for s in busy_samples if s["is_throttled"])

            return {
                "sample_count": len(self.samples),
                "busy_sample_count": len(busy_samples),
                # All samples metrics (logged)
                "all_median_clock_mhz": float(np.median(all_clocks)),
                "all_min_clock_mhz": float(np.min(all_clocks)),
                "all_max_clock_mhz": float(np.max(all_clocks)),
                "all_max_temp_c": float(np.max(all_temps)),
                "all_mean_power_w": float(np.mean(all_powers)),
                "all_mean_utilization_pct": float(np.mean(all_utils)),
                "all_throttle_sample_pct": float(all_throttled_count / len(self.samples) * 100.0),
                # BUSY samples metrics (gated on these per Requirement A1)
                "median_clock_mhz": float(np.median(busy_clocks)),
                "min_clock_mhz": float(np.min(busy_clocks)),
                "max_clock_mhz": float(np.max(busy_clocks)),
                "max_temp_c": float(np.max(busy_temps)),
                "throttle_sample_pct": float(busy_throttled_count / len(busy_samples) * 100.0) if busy_samples else 0.0,
                "busy_median_clock_mhz": float(np.median(busy_clocks)),
                "busy_min_clock_mhz": float(np.min(busy_clocks)),
                "busy_max_clock_mhz": float(np.max(busy_clocks)),
                "busy_max_temp_c": float(np.max(busy_temps)),
                "busy_throttle_sample_pct": float(busy_throttled_count / len(busy_samples) * 100.0) if busy_samples else 0.0,
            }


class CpuTelemetryMonitor:
    """
    Background monitor polling typeperf for '% Processor Performance' every interval_sec.
    Measures CPU performance state during runs (median and min per system).
    """
    def __init__(self, interval_sec: int = 1):
        self.interval = interval_sec
        self.samples: List[float] = []
        self._proc = None
        self._thread = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

    def start(self):
        with self._lock:
            self.samples.clear()
            self._stop_event.clear()
        try:
            self._proc = subprocess.Popen(
                ["typeperf", r"\Processor Information(_Total)\% Processor Performance", "-si", str(self.interval)],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1
            )
            self._thread = threading.Thread(target=self._reader_loop, daemon=True)
            self._thread.start()
        except Exception:
            pass

    def _reader_loop(self):
        if not self._proc or not self._proc.stdout:
            return
        while not self._stop_event.is_set():
            line = self._proc.stdout.readline()
            if not line:
                break
            parts = [p.strip(' "\r\n') for p in line.split(',')]
            if len(parts) >= 2:
                try:
                    val = float(parts[1])
                    with self._lock:
                        self.samples.append(val)
                except ValueError:
                    pass

    def stop(self) -> Dict[str, Any]:
        self._stop_event.set()
        if self._proc:
            try:
                self._proc.terminate()
                self._proc.kill()
            except Exception:
                pass
        if self._thread:
            self._thread.join(timeout=1.0)
        with self._lock:
            if not self.samples:
                return {
                    "sample_count": 0,
                    "median_cpu_perf_pct": 0.0,
                    "min_cpu_perf_pct": 0.0,
                    "max_cpu_perf_pct": 0.0,
                }
            return {
                "sample_count": len(self.samples),
                "median_cpu_perf_pct": float(np.median(self.samples)),
                "min_cpu_perf_pct": float(np.min(self.samples)),
                "max_cpu_perf_pct": float(np.max(self.samples)),
            }


def evaluate_run_gate(summary: Dict[str, Any], max_temp_threshold_c: float = 88.0) -> Tuple[bool, List[str]]:
    """
    Evaluates individual run conditions on busy samples:
    1. throttle reasons active on busy samples <= 10.0%
    2. max temperature <= 88.0 C
    (Absolute clock threshold dropped per Requirement A1).
    """
    failures = []
    max_temp = summary.get("busy_max_temp_c", summary.get("max_temp_c", 0.0))
    throttle_pct = summary.get("busy_throttle_sample_pct", summary.get("throttle_sample_pct", 0.0))

    if max_temp > max_temp_threshold_c:
        failures.append(f"Max temperature {max_temp:.1f} C exceeds threshold ({max_temp_threshold_c:.1f} C).")
    if throttle_pct > 10.0:
        failures.append(f"Busy sample throttling was active for {throttle_pct:.1f}% of samples (exceeds 10.0% limit).")

    return len(failures) == 0, failures


def evaluate_session_gate(
    system_median_busy_clocks: Dict[str, float],
    all_runs_summaries: List[Dict[str, Any]],
    canary_drift_pct: float,
    max_temp_threshold_c: float = 88.0,
) -> Tuple[bool, Dict[str, Any]]:
    """
    Evaluates session-wide gate per Requirement A1:
    1. Throttle fraction on busy samples <= 10% across all runs.
    2. Max temperature <= 88 C.
    3. Busy-sample median clock of every system within 10% of each other.
    4. Sequential canary throughput at start and end within 15% (fail if drift > 15%).
    """
    failures = []

    # 1 & 2: Throttle & Temp
    max_temp = max((s.get("busy_max_temp_c", 0.0) for s in all_runs_summaries), default=0.0)
    max_throttle = max((s.get("busy_throttle_sample_pct", 0.0) for s in all_runs_summaries), default=0.0)

    if max_temp > max_temp_threshold_c:
        failures.append(f"Peak temperature {max_temp:.1f} C exceeded {max_temp_threshold_c:.1f} C.")
    if max_throttle > 10.0:
        failures.append(f"Peak busy throttling ratio {max_throttle:.1f}% exceeded 10.0%.")

    # 3. Busy-sample median clock of every system within 10% of each other
    valid_clocks = {sys: clk for sys, clk in system_median_busy_clocks.items() if clk > 0}
    clock_spread_pct = 0.0
    if len(valid_clocks) >= 2:
        max_c = max(valid_clocks.values())
        min_c = min(valid_clocks.values())
        clock_spread_pct = (max_c - min_c) / max_c * 100.0
        if clock_spread_pct > 10.0:
            failures.append(
                f"Cross-system busy median clock spread ({clock_spread_pct:.1f}%) exceeds 10.0% limit "
                f"(min={min_c:.0f} MHz, max={max_c:.0f} MHz)."
            )

    # 4. Sequential Canary
    if canary_drift_pct > 15.0:
        failures.append(f"Sequential canary drift ({canary_drift_pct:.1f}%) exceeds 15.0% limit.")

    passed = len(failures) == 0
    gate_details = {
        "gate_passed": passed,
        "max_temperature_c": max_temp,
        "max_busy_throttle_pct": max_throttle,
        "system_median_busy_clocks": system_median_busy_clocks,
        "clock_spread_pct": clock_spread_pct,
        "canary_drift_pct": canary_drift_pct,
        "failures": failures,
    }
    return passed, gate_details
