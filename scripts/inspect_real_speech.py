import json

with open("data/real_speech_evaluation_results.json", "r", encoding="utf-8") as f:
    d = json.load(f)

print("METADATA:", d["metadata"])

print("\n--- TRAFFIC PROFILES ---")
for prof in ["burst", "poisson_low", "poisson_high"]:
    print(f"\nPROFILE: {prof}")
    for sys_name in ["sequential", "static_batching", "engine"]:
        if sys_name in d["systems"][prof]:
            agg = d["systems"][prof][sys_name]["aggregate"]
            print(f"  {sys_name}:")
            print(f"    throughput: {agg['throughput_mean']:.2f} +/- {agg['throughput_std']:.2f}")
            print(f"    latency p50: {agg['latency_p50_s_mean']:.3f} s")
            print(f"    ttft p50: {agg['ttft_p50_s_mean']:.3f} s")
            print(f"    vram alloc: {agg['peak_vram_alloc_mb_mean']:.1f} MB, res: {agg['peak_vram_res_mb_mean']:.1f} MB")
            print(f"    wer: {agg['wer_pct']:.2f}%, CI: {agg['wer_ci_95']}")
            print(f"    token match rate: {agg['token_match_rate_pct']:.1f}%")
            for k in agg:
                if any(x in k for x in ['flight', 'wait', 'queue', 'reject', 'pool', 'used', 'concurrency']):
                    print(f"    {k}: {agg[k]}")

print("\n--- ENGINE RUN TELEMETRY ---")
for prof in ["burst", "poisson_low", "poisson_high"]:
    if "engine" in d["systems"][prof]:
        runs = d["systems"][prof]["engine"].get("runs", [])
        if runs:
            r0 = runs[0]
            print(f"Profile {prof} Run 0 keys related to concurrency/memory:")
            for k, v in r0.items():
                if any(x in k for x in ['flight', 'wait', 'queue', 'reject', 'pool', 'used', 'concurrency', 'telemetry', 'memory']):
                    print(f"  {k}: {v}")

