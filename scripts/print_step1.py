import json

d = json.load(open('data/step1_gather_decomposition.json', encoding='utf-8'))
print(f"{'Config':<8} | {'Total Step (ms)':<18} | {'Gather Self (ms)':<18} | {'Gather Cross (ms)':<18} | {'Attn Comp (ms)':<16} | {'Other (ms)':<14} | {'Bytes Copied':<16} | {'Gather Frac'}")
print("-" * 130)

for cfg, v in d['configurations'].items():
    tot = v['total_step_time_ms']
    gs = v['self_attention']['gather_time_ms']
    gc = v['cross_attention']['gather_time_ms']
    aself = v['self_attention']['compute_time_ms']
    across = v['cross_attention']['compute_time_ms']
    other = v['other_compute_and_bookkeeping']['other_time_ms']
    g_frac = v['summary']['total_gather_fraction'] * 100
    b_self_mb = v['self_attention']['bytes_copied_mb']
    b_cross_mb = v['cross_attention']['bytes_copied_mb']
    tot_bytes = b_self_mb + b_cross_mb

    tot_str = f"{tot['mean_ms']:.2f} (p95 {tot['p95_ms']:.2f})"
    gs_str = f"{gs['mean_ms']:.2f} (p95 {gs['p95_ms']:.2f})"
    gc_str = f"{gc['mean_ms']:.2f} (p95 {gc['p95_ms']:.2f})"
    ac_str = f"{aself['mean_ms']+across['mean_ms']:.2f}"
    ot_str = f"{other['mean_ms']:.2f}"
    by_str = f"{tot_bytes:.1f} MB"
    gf_str = f"{g_frac:.1f}%"

    print(f"{cfg:<8} | {tot_str:<18} | {gs_str:<18} | {gc_str:<18} | {ac_str:<16} | {ot_str:<14} | {by_str:<16} | {gf_str}")
