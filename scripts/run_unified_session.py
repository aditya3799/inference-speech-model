"""
scripts/run_unified_session.py

Unified Master Session Runner:
Executes the benchmark, step profile, memory breakdown, latency analysis,
and report generation in a single continuous session with a shared session_id
and timestamp. Guarantees zero stale data across all report sections.
"""

import sys
import subprocess
import uuid
from datetime import datetime
from pathlib import Path


def main():
    repo_root = Path(__file__).resolve().parent.parent
    session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    timestamp = datetime.now().isoformat()

    print("==================================================================")
    print("STARTING UNIFIED RIGOROUS PROFILING SESSION")
    print(f"Session ID : {session_id}")
    print(f"Timestamp  : {timestamp}")
    print("==================================================================")

    python_exe = sys.executable

    steps = [
        ("Interleaved Benchmark (Burst & Poisson-High)", [
            python_exe, str(repo_root / "scripts" / "run_interleaved_benchmark_v2.py"),
            "--session-id", session_id,
            "--timestamp", timestamp,
        ]),
        ("CUDA Events Step-Level Profile & Launch Overhead", [
            python_exe, str(repo_root / "scripts" / "profile_step_level_v2.py"),
            "--session-id", session_id,
            "--timestamp", timestamp,
        ]),
        ("Memory Breakdown & Parity Check", [
            python_exe, str(repo_root / "scripts" / "profile_memory_breakdown_v2.py"),
            "--session-id", session_id,
            "--timestamp", timestamp,
        ]),
        ("Latency Gap & Capacity Analysis", [
            python_exe, str(repo_root / "scripts" / "analyze_latency_gap_v2.py"),
            "--session-id", session_id,
            "--timestamp", timestamp,
        ]),
        ("Report & Consistency Generator", [
            python_exe, str(repo_root / "scripts" / "generate_gap_report_v2.py"),
        ]),
    ]

    import os
    env = dict(os.environ)
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["USE_TF"] = "0"
    env["TRANSFORMERS_NO_TF"] = "1"

    for name, cmd in steps:
        print(f"\n>>> [STEP] Executing: {name} <<<")
        res = subprocess.run(cmd, cwd=str(repo_root), env=env)
        if res.returncode != 0:
            print(f"\n[ERROR] Step '{name}' failed with exit code {res.returncode}. Aborting session.")
            sys.exit(res.returncode)

    print("\n==================================================================")
    print("UNIFIED PROFILING SESSION COMPLETED SUCCESSFULLY")
    print(f"Session ID: {session_id}")
    print("Output Artifacts:")
    print("  - data/gap_profile_tables_v2.md")
    print("  - data/gap_profile_results_v2.json")
    print("==================================================================")


if __name__ == "__main__":
    main()
