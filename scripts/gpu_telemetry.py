"""
scripts/gpu_telemetry.py

Background GPU telemetry monitor for empirical gating and telemetry logging.
Polls nvidia-smi every 500 ms during runs to capture under-load GPU clock,
power, temperature, and throttle events.
"""

import threading
import subprocess
import time
from typing import List, Dict, Any, Tuple
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
    Captures live under-load telemetry: clock, power draw, temperature, and throttle events.
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
                    "--query-gpu=clocks.gr,power.draw,temperature.gpu,clocks_throttle_reasons.active,clocks_throttle_reasons.sw_thermal_slowdown,clocks_throttle_reasons.hw_slowdown,clocks_throttle_reasons.sw_power_cap",
                    "--format=csv,noheader,nounits"
                ], stderr=subprocess.DEVNULL).decode().strip()
                line = out.split("\n")[0].strip()
                parts = [p.strip() for p in line.split(",")]
                clock = float(parts[0])
                power = float(parts[1])
                temp = float(parts[2])
                active_throttle = parts[3]
                sw_thermal = parts[4]
                hw_slowdown = parts[5]
                sw_power = parts[6]

                # Determine active throttling (exclude 0x0000000000000001 = gpu_idle)
                raw_int = int(active_throttle, 16) if active_throttle.startswith("0x") else 0
                is_throttled = (raw_int & ~0x1) != 0

                with self._lock:
                    self.samples.append({
                        "timestamp": time.time(),
                        "clock_mhz": clock,
                        "power_w": power,
                        "temp_c": temp,
                        "active_throttle_hex": active_throttle,
                        "is_throttled": is_throttled,
                        "sw_thermal": sw_thermal,
                        "hw_slowdown": hw_slowdown,
                        "sw_power_cap": sw_power,
                    })
            except Exception:
                pass
            time.sleep(self.interval)

    def get_summary(self) -> Dict[str, Any]:
        with self._lock:
            if not self.samples:
                return {
                    "sample_count": 0,
                    "median_clock_mhz": 0.0,
                    "min_clock_mhz": 0.0,
                    "max_clock_mhz": 0.0,
                    "max_temp_c": 0.0,
                    "mean_power_w": 0.0,
                    "throttle_sample_pct": 0.0,
                    "throttled_samples_count": 0,
                }

            clocks = [s["clock_mhz"] for s in self.samples]
            temps = [s["temp_c"] for s in self.samples]
            powers = [s["power_w"] for s in self.samples]
            throttled_count = sum(1 for s in self.samples if s["is_throttled"])
            throttle_pct = (throttled_count / len(self.samples)) * 100.0

            return {
                "sample_count": len(self.samples),
                "median_clock_mhz": float(np.median(clocks)),
                "min_clock_mhz": float(np.min(clocks)),
                "max_clock_mhz": float(np.max(clocks)),
                "max_temp_c": float(np.max(temps)),
                "mean_power_w": float(np.mean(powers)),
                "throttle_sample_pct": float(throttle_pct),
                "throttled_samples_count": throttled_count,
            }


def evaluate_run_gate(summary: Dict[str, Any], min_clock_target_mhz: float = 500.0, max_temp_threshold_c: float = 88.0) -> Tuple[bool, List[str]]:
    """
    Evaluates whether a run satisfies the gate criteria:
    1. median under-load clock >= min_clock_target_mhz (default 500 MHz, active working state)
    2. max temperature <= max_temp_threshold_c (default 88 C)
    3. throttle reasons active for <= 10% of samples
    """
    failures = []
    med_clock = summary.get("median_clock_mhz", 0.0)
    max_temp = summary.get("max_temp_c", 0.0)
    throttle_pct = summary.get("throttle_sample_pct", 0.0)

    if med_clock < min_clock_target_mhz:
        failures.append(f"Median clock {med_clock:.1f} MHz is below active operational threshold ({min_clock_target_mhz:.1f} MHz).")
    if max_temp > max_temp_threshold_c:
        failures.append(f"Max temperature {max_temp:.1f} C exceeds threshold ({max_temp_threshold_c:.1f} C).")
    if throttle_pct > 10.0:
        failures.append(f"Throttling was active for {throttle_pct:.1f}% of samples (exceeds 10.0% limit).")

    passed = len(failures) == 0
    return passed, failures
