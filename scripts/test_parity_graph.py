"""100-utterance serving parity diagnostics for engine_graph.

This is a correctness workload, not a performance benchmark.  It exercises the
real scheduler under burst and Poisson(4.0) arrivals and records enough graph
provenance to diagnose every token mismatch.
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import jiwer
import torch

from scripts.evaluate_real_speech import (
    RealSpeechBenchmark,
    generate_burst_traffic,
    generate_poisson_traffic,
)


def first_divergence(actual: List[int], expected: List[int]) -> int:
    for idx, (lhs, rhs) in enumerate(zip(actual, expected)):
        if lhs != rhs:
            return idx
    return min(len(actual), len(expected))


def mismatch_report(request: Dict[str, Any]) -> Dict[str, Any]:
    actual = request["gen_tokens"]
    expected = request["hf_tokens"]
    idx = first_divergence(actual, expected)
    trace = next(
        (item for item in request.get("graph_token_trace", []) if item["token_index"] == idx),
        None,
    )
    return {
        "utterance": request["sample_id"],
        "first_divergent_token_index": idx,
        "output_length": len(actual),
        "hf_length": len(expected),
        "length_relation": "shorter" if len(actual) < len(expected) else "longer" if len(actual) > len(expected) else "equal",
        "slot_id": request.get("slot_id"),
        "slot_reused": request.get("slot_reused", False),
        "bucket_at_divergence": trace.get("bucket") if trace else None,
        "readback_position": trace.get("readback_position") if trace else None,
        "actual_token": actual[idx] if idx < len(actual) else None,
        "hf_token": expected[idx] if idx < len(expected) else None,
    }


def hf_reference_wer(bench: RealSpeechBenchmark) -> float:
    refs = [sample["norm_ref"] for sample in bench.samples]
    hyps = [bench.normalizer(bench.processor.tokenizer.decode(sample["hf_tokens"])) for sample in bench.samples]
    return jiwer.wer(refs, hyps) * 100.0


def summarize(name: str, result: Dict[str, Any], reference_wer: float) -> Dict[str, Any]:
    mismatches = [
        mismatch_report(req)
        for req in result["per_request"]
        if req["gen_tokens"] != req["hf_tokens"]
    ]
    extra_steps = [req.get("graph_extra_slot_steps", 0) for req in result["per_request"]]
    summary = {
        "name": name,
        "num_requests": result["num_requests"],
        "exact_match_pct": result["token_match_rate_pct"],
        "wer_pct": result["wer_pct"],
        "reference_wer_pct": reference_wer,
        "wer_delta_points": result["wer_pct"] - reference_wer,
        "gate_passed": result["token_match_rate_pct"] >= 99.0 and abs(result["wer_pct"] - reference_wer) <= 0.2,
        "mismatches": mismatches,
        "cost_of_k": {
            "extra_slot_steps_total": sum(extra_steps),
            "extra_slot_steps_mean": sum(extra_steps) / len(extra_steps),
            "extra_slot_steps_per_request": extra_steps,
        },
    }
    print(
        f"{name}: exact={summary['exact_match_pct']:.1f}% "
        f"WER={summary['wer_pct']:.3f}% (HF {reference_wer:.3f}%, "
        f"delta={summary['wer_delta_points']:+.3f}) gate={'PASS' if summary['gate_passed'] else 'FAIL'}"
    )
    for mismatch in mismatches:
        print("  MISMATCH " + json.dumps(mismatch, sort_keys=True))
    return summary


def run_all(output: Path, run_isolations: bool) -> Dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for engine_graph serving parity")

    bench = RealSpeechBenchmark(
        device="cuda",
        dtype=torch.float16,
        local_files_only=True,
    )
    reference_wer = hf_reference_wer(bench)
    burst = generate_burst_traffic(100)
    poisson = generate_poisson_traffic(100, rate=4.0, seed=42)
    report: Dict[str, Any] = {
        "created_at": datetime.now().isoformat(),
        "reference_wer_pct": reference_wer,
        "profiles": {},
        "isolations": {},
    }

    profiles = {"burst": burst, "poisson_4_0": poisson}
    for profile_name, traffic in profiles.items():
        result = bench.run_engine(
            traffic,
            max_batch_size=4,
            graph_decode=True,
            graph_readback_k=4,
            batch_encoder=True,
            graph_direct_encoder_kv=True,
        )
        report["profiles"][profile_name] = summarize(profile_name, result, reference_wer)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8", newline="\n")

    if run_isolations:
        isolations = {
            "k_1": {"graph_readback_k": 1},
            "synchronous_copies": {"graph_non_blocking_copies": False},
            "max_batch_size_1": {"max_batch_size": 1},
            "zero_slot_buffers_on_release": {"graph_zero_slot_buffers_on_release": True},
            "only_bucket_448": {"graph_force_bucket": 448},
            "equal_length_requests": {"batch_encoder": True},
        }
        for name, overrides in isolations.items():
            options = {
                "max_batch_size": 4,
                "graph_decode": True,
                "graph_readback_k": 4,
                "batch_encoder": True,
                "graph_direct_encoder_kv": True,
            }
            options.update(overrides)
            result = bench.run_engine(burst, **options)
            report["isolations"][name] = summarize(name, result, reference_wer)
            output.write_text(json.dumps(report, indent=2), encoding="utf-8", newline="\n")

    report["gate_passed"] = all(item["gate_passed"] for item in report["profiles"].values())
    output.write_text(json.dumps(report, indent=2), encoding="utf-8", newline="\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "data" / "serving_parity_graph.json")
    parser.add_argument("--skip-isolations", action="store_true")
    args = parser.parse_args()
    report = run_all(args.output, run_isolations=not args.skip_isolations)
    return 0 if report["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
