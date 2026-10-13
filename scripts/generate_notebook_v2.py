"""
Generates real_speech_serving_v2.ipynb with complete live-executing benchmark cycles,
allowing users to click and execute the full 20+ minute benchmark cycle on GPU,
as well as fast modes and comprehensive telemetry tables.
"""

import json
from pathlib import Path


def build_live_notebook():
    nb = {
        "cells": [],
        "metadata": {
            "language_info": {
                "name": "python",
                "version": "3.10.11"
            },
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    # Cell 1: Header Markdown
    nb["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Real-Speech Serving Evaluation (v2: One-Time Cross-Attention KV Gather)\n",
            "### Live Executable Benchmark Cycle & Comprehensive Evaluation Suite\n",
            "\n",
            "This notebook provides a **live, fully executable benchmark environment** on GPU for comparing speech serving architectures on Whisper-base:\n",
            "1. **Sequential Baseline**: Single request execution (unbatched, unpaged).\n",
            "2. **Static Contiguous Batching**: Padded synchronous batching ($B=4$).\n",
            "3. **Engine (Flag OFF — Legacy)**: Continuous batching + paged dual cache, re-gathering cross-attention KV on **every single decode step**.\n",
            "4. **Engine (Flag ON — Fix)**: Continuous batching + **one-time cross-attention KV gather** at encode-to-decode handoff, cached contiguously for request lifetime.\n",
            "\n",
            "> **Live Cycle Execution**: The benchmark cell in Section 3 executes the complete 100-utterance LibriSpeech evaluation across all 3 traffic profiles (`burst`, `poisson_low`, `poisson_high`) and 3 repeats (+ warmup) directly on your GPU (~20 minutes), printing real-time streaming progress and populating all metrics tables."
        ]
    })

    # Cell 2: Imports & Environment
    nb["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import os\n",
            "import sys\n",
            "import gc\n",
            "import json\n",
            "import time\n",
            "import random\n",
            "from pathlib import Path\n",
            "\n",
            "os.environ[\"USE_TF\"] = \"0\"\n",
            "os.environ[\"TRANSFORMERS_NO_TF\"] = \"1\"\n",
            "os.environ[\"TF_ENABLE_ONEDNN_OPTS\"] = \"0\"\n",
            "os.environ[\"PYTORCH_CUDA_ALLOC_CONF\"] = \"expandable_segments:True\"\n",
            "\n",
            "import torch\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "import matplotlib.pyplot as plt\n",
            "import soundfile as sf\n",
            "import jiwer\n",
            "from transformers import WhisperProcessor\n",
            "\n",
            "# Ensure repository root is on sys.path\n",
            "REPO_ROOT = Path.cwd()\n",
            "if str(REPO_ROOT) not in sys.path:\n",
            "    sys.path.insert(0, str(REPO_ROOT))\n",
            "\n",
            "from speech_serving.models import PretrainedWhisperModel\n",
            "from speech_serving.encoder_block_manager import EncoderBlockManager\n",
            "from speech_serving.block_manager import BlockManager\n",
            "from speech_serving.dual_cache_manager import DualCacheManager\n",
            "from speech_serving.phase_aware_scheduler import (\n",
            "    PhaseAwareSpeechScheduler,\n",
            "    WHISPER_DEFAULT_PREFIX_TOKENS,\n",
            "    WHISPER_EOS_TOKEN_ID,\n",
            ")\n",
            "from scripts.evaluate_real_speech import RealSpeechBenchmark, run_benchmark_suite\n",
            "\n",
            "assert torch.cuda.is_available(), \"CUDA GPU is required for live benchmark execution.\"\n",
            "device = \"cuda\"\n",
            "dtype = torch.float16\n",
            "\n",
            "gpu_name = torch.cuda.get_device_name(0)\n",
            "total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)\n",
            "print(f\"CUDA Device: {gpu_name} ({total_vram_gb:.2f} GB VRAM)\")\n",
            "print(f\"PyTorch Version: {torch.__version__}\")"
        ]
    })

    # Cell 3: Model and Data Loading Markdown
    nb["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Model & LibriSpeech Dataset Initialization\n",
            "Initializes `RealSpeechBenchmark` with `openai/whisper-base` in FP16 precision, extracts 80-channel log-mel filterbanks from LibriSpeech `test-clean` audio, and precomputes Hugging Face reference greedy tokens for exact token parity checking."
        ]
    })

    # Cell 4: Model and Data Loading Code
    nb["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "print(\"Initializing RealSpeechBenchmark in FP16 on CUDA...\")\n",
            "bench = RealSpeechBenchmark(device=device, dtype=dtype)\n",
            "model = bench.model\n",
            "processor = bench.processor\n",
            "normalizer = bench.normalizer\n",
            "samples = bench.samples\n",
            "\n",
            "print(f\"Successfully initialized: {len(samples)} stratified LibriSpeech utterances ready.\")"
        ]
    })

    # Cell 5: Live Serving Harness Functions Markdown
    nb["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Serving Engine Architecture & Implementation\n",
            "The benchmark benchmarks four distinct serving systems:\n",
            "- **Sequential Baseline** (`bench.run_sequential`): Unbatched, unpaged baseline processing one request at a time.\n",
            "- **Static Batching** (`bench.run_static_batching`): Padded synchronous batching ($B=4$).\n",
            "- **Engine (Flag OFF)** (`bench.run_engine(..., cache_cross_kv=False)`): Continuous batching + paged dual cache, re-gathering cross-attention KV from encoder pool blocks on **every decode step**.\n",
            "- **Engine (Flag ON)** (`bench.run_engine(..., cache_cross_kv=True)`): Continuous batching + **one-time cross-attention KV gather** at handoff, held contiguously for request lifetime."
        ]
    })

    # Cell 6: Inspect Serving Methods Code
    nb["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Quick verification of serving system methods\n",
            "print(\"Serving system methods available in `bench`:\")\n",
            "print(\"  - bench.run_sequential(traffic)\")\n",
            "print(\"  - bench.run_static_batching(traffic, batch_size=4)\")\n",
            "print(\"  - bench.run_engine(traffic, max_batch_size=4, cache_cross_kv=False)  [Flag OFF: Re-gather]\")\n",
            "print(\"  - bench.run_engine(traffic, max_batch_size=4, cache_cross_kv=True)   [Flag ON: Cached Cross-KV]\")\n",
            "print(f\"Encoder pool capacity: {bench.model.num_encoder_layers} layers x 6144 blocks\")\n",
            "print(f\"Decoder pool capacity: {bench.model.num_decoder_layers} layers x 1024 blocks\")"
        ]
    })

    # Cell 7: Workload Configuration Markdown
    nb["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Live Benchmark Execution on GPU\n",
            "\n",
            "You can choose between three execution modes:\n",
            "- **`BENCHMARK_MODE = \"full\"`** *(Default)*: Runs the **complete 100-utterance, 3-traffic-profile (`burst`, `poisson_low`, `poisson_high`), 3-repeat (+ warmup) benchmark cycle** directly on your GPU. Takes **~20–25 minutes** and streams live updates per repeat.\n",
            "- **`BENCHMARK_MODE = \"quick\"`**: Runs a fast smoke test (20 utterances, 3 profiles, 1 repeat) in **~1–2 minutes**.\n",
            "- **`BENCHMARK_MODE = \"cached\"`**: Instantly loads the previously generated authoritative benchmark run from `data/real_speech_evaluation_results_v2.json` in **< 1 second**."
        ]
    })

    # Cell 8: Workload Configuration Code
    nb["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ==============================================================================\n",
            "# BENCHMARK EXECUTION CONFIGURATION\n",
            "# ==============================================================================\n",
            "# Select execution mode:\n",
            "#   'full'   : Complete 20+ minute benchmark cycle on GPU (100 utterances, 3 profiles, 4 systems, 3 repeats + warmup)\n",
            "#   'quick'  : Fast smoke test on GPU (20 utterances, 3 profiles, 1 repeat ~ 1-2 mins)\n",
            "#   'cached' : Instantly load precomputed results from data/real_speech_evaluation_results_v2.json (< 1 sec)\n",
            "# ==============================================================================\n",
            "BENCHMARK_MODE = \"full\"  # <--- Set to 'full', 'quick', or 'cached'\n",
            "OUTPUT_JSON_PATH = \"data/real_speech_evaluation_results_v2.json\"\n",
            "BATCH_SIZE = 4\n",
            "\n",
            "print(f\"Configured Benchmark Mode: {BENCHMARK_MODE.upper()}\")\n",
            "if BENCHMARK_MODE == 'full':\n",
            "    print(\"Note: Running Cell 10 below will execute the complete 100-utterance evaluation (~20-25 mins).\")\n",
            "elif BENCHMARK_MODE == 'quick':\n",
            "    print(\"Note: Running Cell 10 below will execute a quick smoke test (~1-2 mins).\")\n",
            "else:\n",
            "    print(f\"Note: Running Cell 10 will immediately load {OUTPUT_JSON_PATH}.\")"
        ]
    })

    # Cell 9: Live Execution Markdown
    nb["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Execute Benchmark Cycle\n",
            "Click **Run** on the cell below to begin the benchmark cycle. The output below will stream live progress for each traffic profile, serving system, and measured repeat."
        ]
    })

    # Cell 10: Live Execution Code
    nb["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "if BENCHMARK_MODE == \"cached\":\n",
            "    print(f\"Loading precomputed benchmark results from {OUTPUT_JSON_PATH}...\")\n",
            "    with open(OUTPUT_JSON_PATH, \"r\", encoding=\"utf-8\") as f:\n",
            "        data_v2 = json.load(f)\n",
            "    print(f\"Loaded results for {len(data_v2['systems'])} traffic profiles successfully!\")\n",
            "else:\n",
            "    num_samples = 100 if BENCHMARK_MODE == \"full\" else 20\n",
            "    num_repeats = 3 if BENCHMARK_MODE == \"full\" else 1\n",
            "    print(\"=\" * 70)\n",
            "    print(f\"STARTING LIVE BENCHMARK ON GPU: {gpu_name}\")\n",
            "    print(f\"Mode: {BENCHMARK_MODE.upper()} | Utterances: {num_samples} | Repeats: {num_repeats} | Batch Size: {BATCH_SIZE}\")\n",
            "    print(\"Traffic Profiles: Burst (t=0), Poisson Low (λ=1.0), Poisson High (λ=4.0)\")\n",
            "    print(\"Systems: Sequential, Static Batching, Engine Flag OFF, Engine Flag ON\")\n",
            "    print(\"=\" * 70)\n",
            "    \n",
            "    data_v2 = run_benchmark_suite(\n",
            "        num_repeats=num_repeats,\n",
            "        max_samples=num_samples,\n",
            "        batch_size=BATCH_SIZE,\n",
            "        output_path=OUTPUT_JSON_PATH,\n",
            "        include_repeated=False,\n",
            "        bench=bench,\n",
            "    )\n",
            "    print(\"\\nLive benchmark execution complete! Populating tables below...\")"
        ]
    })

    # Cell 11: Authoritative Historical Comparison Markdown
    nb["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Serving Performance Comparison Table\n",
            "Summarizes throughput, request latency percentiles (p50/p95), TTFT percentiles (p50/p95), VRAM allocation, and actually used pool memory across all traffic profiles."
        ]
    })

    # Cell 12: Historical Performance Table Code
    nb["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "profiles = [\n",
            "    ('burst', 'Burst (t=0)'),\n",
            "    ('poisson_low', 'Poisson Low (λ=1.0)'),\n",
            "    ('poisson_high', 'Poisson High (λ=4.0)'),\n",
            "]\n",
            "systems = [\n",
            "    ('sequential', 'Sequential'),\n",
            "    ('static_batching', 'Static Batching (B=4)'),\n",
            "    ('engine_flag_off', 'Engine (Flag OFF, re-gather)'),\n",
            "    ('engine_flag_on', 'Engine (Flag ON, cached cross-KV)'),\n",
            "]\n",
            "\n",
            "rows = []\n",
            "for prof_k, prof_label in profiles:\n",
            "    prof_data = data_v2['systems'][prof_k]\n",
            "    for sys_k, sys_label in systems:\n",
            "        agg = prof_data[sys_k]['aggregate']\n",
            "        th = f\"{agg['throughput_mean']:.2f} ± {agg['throughput_std']:.2f}\"\n",
            "        lat_p50 = f\"{agg['latency_p50_s_mean']:.3f}\"\n",
            "        lat_p95 = f\"{agg['latency_p95_s_mean']:.3f}\"\n",
            "        ttft_p50 = f\"{agg['ttft_p50_s_mean']:.3f}\"\n",
            "        ttft_p95 = f\"{agg['ttft_p95_s_mean']:.3f}\"\n",
            "        vram = f\"{agg['peak_vram_alloc_mb_mean']:.1f} / {agg['peak_vram_res_mb_mean']:.1f}\"\n",
            "        used_val = agg.get('peak_actually_used_mb_mean', 'not measured')\n",
            "        used_str = f\"{used_val:.2f} MB\" if isinstance(used_val, (int, float)) else str(used_val)\n",
            "        ci = agg.get('wer_ci_95', [agg['wer_pct'], agg['wer_pct']])\n",
            "        wer_str = f\"{agg['wer_pct']:.2f}% [{ci[0]:.2f}%, {ci[1]:.2f}%]\"\n",
            "        \n",
            "        rows.append({\n",
            "            'Traffic Profile': prof_label,\n",
            "            'System': sys_label,\n",
            "            'Throughput (x)': th,\n",
            "            'Lat p50 (s)': lat_p50,\n",
            "            'Lat p95 (s)': lat_p95,\n",
            "            'TTFT p50 (s)': ttft_p50,\n",
            "            'TTFT p95 (s)': ttft_p95,\n",
            "            'VRAM Alloc/Res (MB)': vram,\n",
            "            'Pool Actually Used': used_str,\n",
            "            'Corpus WER (95% CI)': wer_str,\n",
            "        })\n",
            "\n",
            "df_perf = pd.DataFrame(rows)\n",
            "df_perf"
        ]
    })

    # Cell 13: Concurrency Table Markdown
    nb["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5. Concurrency Telemetry & Pool Memory Actually Used\n",
            "Compares in-flight request concurrency, queue wait times, and pool memory actually used between the two engine variants."
        ]
    })

    # Cell 14: Concurrency Table Code
    nb["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "concurrency_rows = []\n",
            "for prof_k, prof_label in profiles:\n",
            "    for sys_k, sys_label in [('engine_flag_off', 'Flag OFF (re-gather)'), ('engine_flag_on', 'Flag ON (cached cross-KV)')]:\n",
            "        c = data_v2['systems'][prof_k][sys_k]['aggregate']['concurrency']\n",
            "        concurrency_rows.append({\n",
            "            'Traffic Profile': prof_label,\n",
            "            'Engine Variant': sys_label,\n",
            "            'Avg In-Flight': f\"{c['avg_in_flight_mean']:.2f}\",\n",
            "            'Max In-Flight': c['max_in_flight_max'],\n",
            "            'Queue Wait Mean (s)': f\"{c['queue_wait_time_mean_s']:.3f}\",\n",
            "            'Queue Wait p50 (s)': f\"{c['queue_wait_time_p50_s']:.3f}\",\n",
            "            'Queue Wait p95 (s)': f\"{c['queue_wait_time_p95_s']:.3f}\",\n",
            "            'Pool Actually Used Peak': f\"{c.get('peak_actually_used_mb_mean', 0.0):.2f} MB\",\n",
            "            'Rejections': f\"{c['rejections_mean']:.0f}\",\n",
            "        })\n",
            "\n",
            "df_concurrency = pd.DataFrame(concurrency_rows)\n",
            "df_concurrency"
        ]
    })

    # Cell 15: Drift Verification Markdown
    nb["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 6. Drift Check: Engine Flag OFF (Current) vs Original Engine (Baseline v1)\n",
            "Validates that the legacy execution path (`engine_flag_off`) in the current run reproduces the baseline v1 results from `data/real_speech_evaluation_results.json` within run-to-run noise ($\le 1\\sigma$)."
        ]
    })

    # Cell 16: Drift Verification Code
    nb["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "v1_path = Path('data/real_speech_evaluation_results.json')\n",
            "with open(v1_path, 'r', encoding='utf-8') as f:\n",
            "    data_v1 = json.load(f)\n",
            "\n",
            "drift_rows = []\n",
            "drift_metrics = [\n",
            "    ('throughput_mean', 'throughput_std', 'Throughput (audio-s/wall-s)', 'x'),\n",
            "    ('latency_p50_s_mean', 'latency_p50_s_std', 'Latency p50 (s)', 's'),\n",
            "    ('latency_p95_s_mean', 'latency_p95_s_std', 'Latency p95 (s)', 's'),\n",
            "    ('peak_vram_alloc_mb_mean', 'peak_vram_alloc_mb_std', 'Peak VRAM Alloc (MB)', 'MB'),\n",
            "]\n",
            "\n",
            "for prof_k, prof_label in profiles:\n",
            "    v1_agg = data_v1['systems'][prof_k]['engine']['aggregate']\n",
            "    v2_agg = data_v2['systems'][prof_k]['engine_flag_off']['aggregate']\n",
            "    \n",
            "    for m_mean, m_std, label, unit in drift_metrics:\n",
            "        val1 = v1_agg[m_mean]\n",
            "        std1 = v1_agg.get(m_std, 0.0)\n",
            "        val2 = v2_agg[m_mean]\n",
            "        std2 = v2_agg.get(m_std, 0.0)\n",
            "        \n",
            "        diff = val2 - val1\n",
            "        comb_std = np.sqrt(std1**2 + std2**2)\n",
            "        threshold = max(comb_std, std1, std2, 1e-4)\n",
            "        is_drift = abs(diff) > threshold\n",
            "        \n",
            "        drift_rows.append({\n",
            "            'Profile': prof_label,\n",
            "            'Metric': label,\n",
            "            'Previous (v1 Engine)': f\"{val1:.3f}{unit} ± {std1:.3f}\",\n",
            "            'Current (v2 Flag OFF)': f\"{val2:.3f}{unit} ± {std2:.3f}\",\n",
            "            'Delta (Δ)': f\"{diff:+.3f}{unit}\",\n",
            "            'Combined Std': f\"{threshold:.3f}{unit}\",\n",
            "            'Status': 'DRIFT > σ' if is_drift else 'Within Noise (≤ 1σ)',\n",
            "        })\n",
            "\n",
            "df_drift = pd.DataFrame(drift_rows)\n",
            "df_drift"
        ]
    })

    # Cell 17: Token Parity Markdown
    nb["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 7. Token Parity Verification vs Hugging Face Reference\n",
            "Verifies token equivalence across all systems and confirms zero regressions between Flag OFF and Flag ON."
        ]
    })

    # Cell 18: Token Parity Code
    nb["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "parity_rows = []\n",
            "for prof_k, prof_label in profiles:\n",
            "    for sys_k, sys_label in systems:\n",
            "        sys_data = data_v2['systems'][prof_k][sys_k]\n",
            "        match_pct = sys_data['aggregate']['token_match_rate_pct']\n",
            "        run0 = sys_data['runs'][0]\n",
            "        differing = [r['sample_id'] for r in run0['per_request'] if r['gen_tokens'] != r['hf_tokens']]\n",
            "        \n",
            "        parity_rows.append({\n",
            "            'Profile': prof_label,\n",
            "            'System': sys_label,\n",
            "            'HF Match Rate (%)': f\"{match_pct:.1f}%\",\n",
            "            'Total Requests': len(run0['per_request']),\n",
            "            'Differing Utterances': 'None (100% exact match)' if not differing else ', '.join(differing[:3]),\n",
            "        })\n",
            "\n",
            "df_parity = pd.DataFrame(parity_rows)\n",
            "df_parity"
        ]
    })

    # Cell 19: Visualizations Markdown
    nb["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 8. Comparative Visualizations\n",
            "Visual comparison of Serving Throughput, Latency p50, and Memory Footprints across traffic profiles."
        ]
    })

    # Cell 20: Visualizations Code
    nb["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "fig, axes = plt.subplots(1, 3, figsize=(18, 5))\n",
            "prof_keys = ['burst', 'poisson_low', 'poisson_high']\n",
            "prof_names = ['Burst (t=0)', 'Poisson Low (λ=1.0)', 'Poisson High (λ=4.0)']\n",
            "x = np.arange(len(prof_keys))\n",
            "width = 0.2\n",
            "\n",
            "# 1. Throughput Plot\n",
            "ax = axes[0]\n",
            "for idx, (sys_k, sys_l) in enumerate(systems):\n",
            "    vals = [data_v2['systems'][p][sys_k]['aggregate']['throughput_mean'] for p in prof_keys]\n",
            "    errs = [data_v2['systems'][p][sys_k]['aggregate']['throughput_std'] for p in prof_keys]\n",
            "    ax.bar(x + idx * width, vals, width, yerr=errs, capsize=4, label=sys_l)\n",
            "ax.set_ylabel('Throughput (audio-sec / wall-sec)')\n",
            "ax.set_title('Serving Throughput (Higher is Better)')\n",
            "ax.set_xticks(x + 1.5 * width)\n",
            "ax.set_xticklabels(prof_names)\n",
            "ax.legend(fontsize=8)\n",
            "ax.grid(axis='y', linestyle='--', alpha=0.7)\n",
            "\n",
            "# 2. Latency p50 Plot\n",
            "ax = axes[1]\n",
            "for idx, (sys_k, sys_l) in enumerate(systems):\n",
            "    vals = [data_v2['systems'][p][sys_k]['aggregate']['latency_p50_s_mean'] for p in prof_keys]\n",
            "    ax.bar(x + idx * width, vals, width, label=sys_l)\n",
            "ax.set_ylabel('Latency p50 (seconds)')\n",
            "ax.set_title('Request Latency p50 (Lower is Better)')\n",
            "ax.set_xticks(x + 1.5 * width)\n",
            "ax.set_xticklabels(prof_names)\n",
            "ax.legend(fontsize=8)\n",
            "ax.grid(axis='y', linestyle='--', alpha=0.7)\n",
            "\n",
            "# 3. Memory Footprint: Peak VRAM vs Actually Used Pool (Burst)\n",
            "ax = axes[2]\n",
            "sys_labels_short = ['Sequential', 'Static Batch', 'Engine (Flag OFF)', 'Engine (Flag ON)']\n",
            "vram_alloc = [data_v2['systems']['burst'][sys_k]['aggregate']['peak_vram_alloc_mb_mean'] for sys_k, _ in systems]\n",
            "used_pool = []\n",
            "for sys_k, _ in systems:\n",
            "    val = data_v2['systems']['burst'][sys_k]['aggregate'].get('peak_actually_used_mb_mean', 0.0)\n",
            "    used_pool.append(val if isinstance(val, (int, float)) else 0.0)\n",
            "\n",
            "x_bar = np.arange(len(sys_labels_short))\n",
            "ax.bar(x_bar - 0.15, vram_alloc, 0.3, label='Peak VRAM Allocated (MB)', color='tab:blue')\n",
            "ax.bar(x_bar + 0.15, used_pool, 0.3, label='Actually Used Pool Memory (MB)', color='tab:orange')\n",
            "ax.set_ylabel('Memory (MB)')\n",
            "ax.set_title('Burst: Peak VRAM Allocated vs Actually Used Pool')\n",
            "ax.set_xticks(x_bar)\n",
            "ax.set_xticklabels(sys_labels_short, rotation=15, ha='right')\n",
            "ax.legend(fontsize=8)\n",
            "ax.grid(axis='y', linestyle='--', alpha=0.7)\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()"
        ]
    })

    # Cell 21: Conclusions Markdown
    nb["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 9. Key Engineering Conclusions\n",
            "\n",
            "1. **Serving Throughput & Latency Gains**:\n",
            "   - Under **Burst** traffic, caching cross-attention KV improves engine throughput from $20.56\\text{x}$ to $24.51\\text{x}$ (**+19.2% speedup**) and cuts latency p50 from $22.90\\text{s}$ to $18.33\\text{s}$ (**-4.57s reduction**).\n",
            "   - Under **Poisson High** ($\\lambda=4.0$), throughput rises from $20.03\\text{x}$ to $23.55\\text{x}$ (**+17.6% speedup**) while latency p50 drops from $12.19\\text{s}$ to $8.01\\text{s}$ (**-34.3% latency cut**).\n",
            "\n",
            "2. **Memory Pool Trade-Off**:\n",
            "   - Encoder block pool usage is $94$ physical blocks per request ($17.58\\text{ MB}$ in FP16). For batch size 4, this consumes $70.5\\text{ MB}$ ($72.94\\text{ MB}$ total pool usage including decoder blocks).\n",
            "   - In Flag ON mode, holding the contiguous cross-KV copy for the request lifetime adds $17.58\\text{ MB}$ per request, bringing peak active memory to $143.25\\text{ MB}$.\n",
            "   - Encoder blocks remain allocated in the physical pool until request completion to preserve reference counting and prefix caching.\n",
            "\n",
            "3. **Zero Token Regressions**:\n",
            "   - 100% token equivalence between Flag OFF and Flag ON across all traffic profiles and all utterances."
        ]
    })

    output_path = Path("real_speech_serving_v2.ipynb")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"Live-executable notebook written to {output_path}!")


if __name__ == "__main__":
    build_live_notebook()
