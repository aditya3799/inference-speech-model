import json
from pathlib import Path

eval_path = Path("data/real_speech_evaluation_results.json")
if not eval_path.exists():
    print("Running real speech evaluation suite...")
    # !py -3.10 scripts/evaluate_real_speech.py --repeats 3 --output data/real_speech_evaluation_results.json

with open(eval_path, "r", encoding="utf-8") as f:
    eval_data = json.load(f)

meta = eval_data["metadata"]
print("=" * 125)
print(f"REAL-SPEECH EVALUATION SUMMARY: {meta['model']} on {meta['dataset']}")
print(f"Hardware: {meta['gpu_name']} | PyTorch {meta['torch_version']} | Repeats: {meta['repeats']}")
print("=" * 125)

fmt = "{:<14} | {:<16} | {:<24} | {:<9} | {:<16} | {:<15} | {:<15} | {:<10}"
print(fmt.format("Traffic", "System", "Corpus WER (95% CI)", "HF Match", "VRAM Alloc/Res", "TTFT p50/p95", "Lat p50/p95", "Throughput"))
print("-" * 125)

for prof, systems in eval_data["systems"].items():
    for sys_name, s_data in systems.items():
        agg = s_data["aggregate"]
        ci_str = f"[{agg['wer_ci_95'][0]:.1f}%, {agg['wer_ci_95'][1]:.1f}%]"
        wer_str = f"{agg['wer_pct']:.2f}% {ci_str}"
        match_str = f"{agg['token_match_rate_pct']:.1f}%"
        vram_str = f"{agg['peak_vram_alloc_mb_mean']:.0f} / {agg['peak_vram_res_mb_mean']:.0f} MB"
        ttft_str = f"{agg['ttft_p50_s_mean']:.2f}s / {agg['ttft_p95_s_mean']:.2f}s"
        lat_str = f"{agg['latency_p50_s_mean']:.2f}s / {agg['latency_p95_s_mean']:.2f}s"
        thput_str = f"{agg['throughput_mean']:.2f}x"
        print(fmt.format(prof.upper(), sys_name, wer_str, match_str, vram_str, ttft_str, lat_str, thput_str))

print("-" * 125)
print("REPEATED AUDIO WORKLOAD (50 Duplicate Utterance Pairs):")
for mode, m_data in eval_data.get("repeated_workload", {}).items():
    agg = m_data["aggregate"]
    print(f"  [{mode.upper()}]: Wall Time = {agg['wall_time_s_mean']:.2f}s (std {agg['wall_time_s_std']:.2f}s) | Lat p50 = {agg['latency_p50_s_mean']:.2f}s | Encoder Calls = {agg['encoder_calls_mean']:.0f} | Skipped = {agg['encoder_calls_skipped_mean']:.0f}")
print("=" * 125)
