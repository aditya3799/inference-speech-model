"""
scripts/run_unified_session_final.py

ONE-PASS Master Orchestrator (Section B & C):
Runs all benchmarking and profiling steps in a single unified session under the
exact same session_id:
1. Interleaved Benchmark (Section B1: burst, poisson 4.0, poisson 3.0, canary start/end, CPU telemetry)
2. Step Profile & Kernel Attribution (Section B2 & B3: union GPU busy fraction, attribution ranking)
3. CUDA Graph Test (Section B4: static speedup, engine blocker documentation)
4. Memory Breakdown & Parity (Section B5 & B6: allocator breakdown, snapshot accounting, 20-utterance parity)
5. Latency Gap Analysis (Section A3: variant comparisons vs engine baseline, tail decomposition)
6. Gap Profile Report (Section A2: data/gap_profile_tables_final.md & data/gap_profile_results_final.json)
7. Regenerate Documentation (Section C: README.md & FINDINGS.md)
"""

import os
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sys
import subprocess
import time
import uuid
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON_EXE = sys.executable


def cooldown_gpu(target_temp: float = 60.0, max_wait_s: int = 60):
    print(f"\n[COOLDOWN] Waiting for GPU to cool to <= {target_temp:.0f} C before unified session...")
    t0 = time.time()
    while time.time() - t0 < max_wait_s:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                stderr=subprocess.DEVNULL
            ).decode().strip()
            temp = float(out.split("\n")[0].strip())
            print(f"  Current GPU temperature: {temp:.0f} C", end="\r")
            if temp <= target_temp or temp == 0.0:
                print(f"\n  GPU cooled to {temp:.0f} C. Ready.")
                return
        except Exception:
            break
        time.sleep(2)
    print("\n  Cooldown timeout reached; proceeding with execution.")


def run_step(step_num: int, step_name: str, cmd_list: list):
    print("\n" + "=" * 70)
    print(f"STEP {step_num}: {step_name.upper()}")
    print("=" * 70)
    t0 = time.time()
    res = subprocess.run(cmd_list, cwd=str(REPO_ROOT))
    elapsed = time.time() - t0
    if res.returncode != 0:
        print(f"\n[ERROR] Step {step_num} ({step_name}) failed with exit code {res.returncode}")
        sys.exit(res.returncode)
    print(f"[SUCCESS] Step {step_num} finished in {elapsed:.1f}s.")


def main():
    session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    timestamp = datetime.now().isoformat()

    print("######################################################################")
    print("MASTER UNIFIED MEASUREMENT SESSION (ONE-PASS TASK)")
    print(f"Session ID : {session_id}")
    print(f"Timestamp  : {timestamp}")
    print("######################################################################")

    cooldown_gpu(target_temp=58.0, max_wait_s=60)

    # Step 1: Interleaved Benchmark (Section B1)
    run_step(1, "Interleaved Benchmark (B1)", [
        PYTHON_EXE, "scripts/run_interleaved_benchmark_final.py",
        "--session-id", session_id,
        "--timestamp", timestamp,
    ])

    # Step 2: Step Profile & Kernel Attribution (Section B2 & B3)
    run_step(2, "Step Profile & Kernel Attribution (B2 & B3)", [
        PYTHON_EXE, "scripts/profile_step_level_final.py",
        "--session-id", session_id,
        "--timestamp", timestamp,
    ])

    # Step 3: CUDA Graph Test (Section B4)
    run_step(3, "CUDA Graph Test (B4)", [
        PYTHON_EXE, "scripts/profile_cuda_graph_final.py",
        "--session-id", session_id,
        "--timestamp", timestamp,
    ])

    # Step 4: Memory Breakdown & Parity (Section B5 & B6)
    run_step(4, "Memory Breakdown & Parity (B5 & B6)", [
        PYTHON_EXE, "scripts/profile_memory_breakdown_final.py",
        "--session-id", session_id,
        "--timestamp", timestamp,
    ])

    # Step 5: Latency Gap Analysis (Section A3)
    run_step(5, "Latency Gap Analysis (A3)", [
        PYTHON_EXE, "scripts/analyze_latency_gap_final.py",
        "--session-id", session_id,
        "--timestamp", timestamp,
    ])

    # Step 6: Gap Profile Report Generator (Section A2)
    run_step(6, "Gap Profile Report Generator (A2)", [
        PYTHON_EXE, "scripts/generate_gap_report_final.py",
        "--session-id", session_id,
        "--timestamp", timestamp,
    ])

    # Step 7: Documentation Regeneration (Section C)
    run_step(7, "Regenerate Documentation (README & FINDINGS) (C)", [
        PYTHON_EXE, "scripts/generate_final_docs.py",
    ])

    print("\n######################################################################")
    print("UNIFIED MEASUREMENT SESSION & DOCS REGENERATION COMPLETE")
    print(f"Unified Session ID : {session_id}")
    print("######################################################################")


if __name__ == "__main__":
    main()
