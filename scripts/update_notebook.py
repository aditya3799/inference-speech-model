"""Script to update paged_dual_cache_speech_serving.ipynb with:
1. Pinned environment check & seed setting in Cell 1 (no parent_dir hack).
2. Clean relabelling of synthetic microbenchmark (removing 1.44x/1.50x claims).
3. Real-speech evaluation section (Whisper-base on LibriSpeech test-clean, parity check, serving suite).
4. Encoder context-sensitivity section (Task 3: noise floor, E1, E2, downstream decode, decision rule).
5. Paging vs batching ablation section (Task 5: decode breakdown, memory double-buffering, 2x2 matrix).
"""

import json
from pathlib import Path

NOTEBOOK_PATH = Path("paged_dual_cache_speech_serving.ipynb")

def build_updated_notebook():
    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # 1. Update Cell 1: Environment, versions, seeds, sys.path
    nb["cells"][1]["source"] = [
        "%load_ext autoreload\n",
        "%autoreload 2\n",
        "\n",
        "# -------------------------------------------------------------\n",
        "# Environment Setup, Dependency Verification & Seed Pinning\n",
        "# -------------------------------------------------------------\n",
        "import os\n",
        "import sys\n",
        "from pathlib import Path\n",
        "import time\n",
        "import math\n",
        "import random\n",
        "import hashlib\n",
        "import json\n",
        "from typing import List, Tuple, Optional, Dict, Any, Callable\n",
        "\n",
        "import numpy as np\n",
        "import torch\n",
        "import torch.nn as nn\n",
        "import torch.nn.functional as F\n",
        "import matplotlib.pyplot as plt\n",
        "import transformers\n",
        "\n",
        "# Ensure repository root is in sys.path\n",
        "REPO_ROOT = Path(os.getcwd()).resolve()\n",
        "if str(REPO_ROOT) not in sys.path:\n",
        "    sys.path.insert(0, str(REPO_ROOT))\n",
        "\n",
        "# Set fixed reproducibility seeds across all random generators\n",
        "SEED = 42\n",
        "random.seed(SEED)\n",
        "np.random.seed(SEED)\n",
        "torch.manual_seed(SEED)\n",
        "if torch.cuda.is_available():\n",
        "    torch.cuda.manual_seed_all(SEED)\n",
        "\n",
        "device = \"cuda\" if torch.cuda.is_available() else \"cpu\"\n",
        "dtype = torch.float16 if device == \"cuda\" else torch.float32\n",
        "\n",
        "print(\"=\" * 65)\n",
        "print(f\"Python Version:       {sys.version.split()[0]}\")\n",
        "print(f\"PyTorch Version:      {torch.__version__}\")\n",
        "print(f\"Transformers Version: {transformers.__version__}\")\n",
        "print(f\"CUDA Available:       {torch.cuda.is_available()}\")\n",
        "if torch.cuda.is_available():\n",
        "    print(f\"CUDA Device:          {torch.cuda.get_device_name(0)}\")\n",
        "    free_mem = torch.cuda.mem_get_info()[0] / (1024**2)\n",
        "    total_mem = torch.cuda.mem_get_info()[1] / (1024**2)\n",
        "    print(f\"VRAM Free / Total:    {free_mem:.1f} MB / {total_mem:.1f} MB\")\n",
        "print(f\"Execution Precision:  {dtype}\")\n",
        "print(f\"Fixed Random Seed:    {SEED}\")\n",
        "print(\"=\" * 65)\n"
    ]

    # 2. Update Cell 14: Relabel synthetic microbenchmark markdown
    nb["cells"][14]["source"] = [
        "---\n",
        "## 7. Scheduler Microbenchmark (Random Whisper-Shaped Weights, Synthetic Mel Inputs)\n",
        "\n",
        "> **Note:** This microbenchmark measures scheduler queuing, phase interleaving, and continuous-batching mechanics on a synthetic model with random weights and synthetic mel inputs. It **does not** measure word error rate (WER) or real speech serving performance.\n",
        "\n",
        "Evaluates generation throughput (tokens/second), latency, and peak GPU memory across three synthetic streaming traffic patterns:\n",
        "- **Uniform Moderate**: Requests arrive spaced evenly across a 15s window (sparse concurrency, ~1.03x mean speedup).\n",
        "- **Uniform Dense**: Compressed 6s arrival window creating high concurrency spikes (~1.13x mean speedup on RTX 3050).\n",
        "- **Bursty**: Clustered surge arrivals testing queue draining and high-water mark responsiveness (~1.07x mean speedup).\n"
    ]

    # Retain cells 0 to 17, and append new sections
    cells = nb["cells"][:18]

    # -------------------------------------------------------------
    # 3. New Section 8: Real-Speech Evaluation (Whisper-base & LibriSpeech)
    # -------------------------------------------------------------
    sec8_md = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "## 8. Real-Speech Evaluation: Pretrained Whisper-base on LibriSpeech test-clean\n",
            "\n",
            "To evaluate real-speech serving performance, we integrate the pretrained `openai/whisper-base` checkpoint and evaluate on a fixed, stratified subset of 100 utterances from LibriSpeech `test-clean` (short < 5s, medium 5–12s, long 12–30s):\n",
            "\n",
            "### Key Methodological Invariants:\n",
            "1. **Parity Check**: Exact greedy token match comparison against Hugging Face `generate()` across 20 utterances in FP32 and FP16.\n",
            "2. **Three Systems Compared on Identical Arrival Schedules**:\n",
            "   - **Sequential**: One request at a time using identical Whisper forward pass code.\n",
            "   - **Static Batching**: Fixed batch size, zero-padded to longest sequence, waiting for slowest utterance.\n",
            "   - **Continuous Batching Engine**: Paged dual cache with phase-aware continuous scheduler.\n",
            "3. **Traffic Profiles**: Burst ($t=0$), Poisson Low Rate ($\\lambda=1.0$), Poisson High Rate ($\\lambda=4.0$).\n",
            "4. **Repeated-Audio Workload**: Verifies end-to-end full-input encoder cache hits (Cache ON vs Cache OFF).\n"
        ]
    }

    sec8_code = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# Run Parity Check against Hugging Face generate() (20 utterances)\n",
            "!py -3.10 scripts/parity_check.py --num-samples 20\n",
            "\n",
            "# To run the full serving evaluation suite (or quick smoke test):\n",
            "# !py -3.10 scripts/evaluate_real_speech.py --quick\n",
            "# !py -3.10 scripts/evaluate_real_speech.py --repeats 3 --output data/real_speech_evaluation_results.json\n"
        ]
    }

    # -------------------------------------------------------------
    # 4. New Section 9: Task 3 - Encoder Context Sensitivity
    # -------------------------------------------------------------
    sec9_md = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "## 9. Task 3: Encoder Context-Sensitivity & Chunk Sharing Soundness\n",
            "\n",
            "In a bidirectional speech encoder (such as Whisper), every audio position attends to every other position across the entire 30-second window. Does identical local audio produce identical encoder hidden states in different contexts?\n",
            "\n",
            "### Experiments:\n",
            "1. **Numerical Noise Floor**: Determinism, batch-position invariance, and FP16 vs FP32 precision floor.\n",
            "2. **Experiment E1 (Trailing Context)**: Utterance A alone (zero-padded) vs Utterance A followed by Utterance B at identical offset (0s).\n",
            "3. **Experiment E2 (Offset Shift)**: Utterance A at offset 0s vs Utterance A shifted by 2s, 5s, 10s.\n",
            "4. **Downstream Transcription**: Feed spliced encoder representations into greedy decode to observe token divergence and WER impact.\n",
            "5. **Decision Rule**: If relative $L_2$ error $> 10\\times$ noise floor OR any transcript diverges, declare chunk-level sharing **UNSOUND**.\n"
        ]
    }

    sec9_code = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import json\n",
            "from pathlib import Path\n",
            "\n",
            "results_path = Path(\"data/encoder_context_sensitivity_results.json\")\n",
            "if not results_path.exists():\n",
            "    print(\"Running encoder context sensitivity evaluation...\")\n",
            "    !py -3.10 scripts/evaluate_context_sensitivity.py --num-pairs 5\n",
            "\n",
            "with open(results_path, \"r\", encoding=\"utf-8\") as f:\n",
            "    cs_data = json.load(f)\n",
            "\n",
            "dec = cs_data[\"decision_rule\"]\n",
            "print(\"=\" * 65)\n",
            "print(f\"VERDICT:        {dec['verdict']}\")\n",
            "print(f\"RECOMMENDATION: {dec['recommendation']}\")\n",
            "print(f\"Noise Floor Rel L2:      {dec['criteria']['noise_floor_relative_l2']:.6e}\")\n",
            "print(f\"Max Measured Rel L2:     {dec['criteria']['max_measured_relative_l2']:.6e}\")\n",
            "print(f\"Noise Floor Multiplier:  {dec['criteria']['noise_floor_multiplier']:.1f}x (Threshold: 10.0x)\")\n",
            "print(f\"Transcript Diverged E1:  {dec['criteria']['transcript_diverged_e1_trailing']}\")\n",
            "print(\"=\" * 65)\n"
        ]
    }

    # -------------------------------------------------------------
    # 5. New Section 10: Task 5 - Paging vs Batching Ablation
    # -------------------------------------------------------------
    sec10_md = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "---\n",
            "## 10. Task 5: Paging vs Batching Ablation & Copy Overhead Breakdown\n",
            "\n",
            "Because standard Hugging Face attention kernels do not natively accept paged block tables, our hybrid serving engine maintains physical block pools while reconstructing contiguous KV tensors for each decode step.\n",
            "\n",
            "### Key Questions Addressed:\n",
            "1. **Decode Step Copy Overhead**: How much time is spent gathering and scattering contiguous tensors ($T_{\\text{gather}} + T_{\\text{scatter}}$) vs the actual attention forward pass ($T_{\\text{attn}}$)?\n",
            "2. **Double-Buffering Memory**: What is the memory cost of holding both the paged block pool and the contiguous working buffers simultaneously?\n",
            "3. **2x2 Ablation Matrix**: Isolating the throughput contribution of batching from paging.\n",
            "   - [No Batching, No Paging] = Sequential\n",
            "   - [Batching, No Paging] = Static batched contiguous\n",
            "   - [Batching + Paging via Gather] = Current hybrid engine\n",
            "   - [Batching + Direct Paged Attention] = Upper bound without copy overhead\n"
        ]
    }

    sec10_code = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import json\n",
            "from pathlib import Path\n",
            "\n",
            "ablation_path = Path(\"data/paging_ablation_results.json\")\n",
            "if not ablation_path.exists():\n",
            "    print(\"Running paging vs batching ablation benchmark...\")\n",
            "    !py -3.10 scripts/evaluate_paging_ablation.py --quick\n",
            "\n",
            "with open(ablation_path, \"r\", encoding=\"utf-8\") as f:\n",
            "    abl_data = json.load(f)\n",
            "\n",
            "print(\"=\" * 75)\n",
            "print(\"2x2 ABLATION MATRIX SUMMARY (Throughput & Latency):\")\n",
            "print(\"=\" * 75)\n",
            "for k, v in abl_data[\"ablation_matrix_2x2\"].items():\n",
            "    print(f\"{v['description']:<60} | {v['throughput_tok_per_s']:>6.2f} tok/s | {v['wall_time_s']['mean']:>6.3f} s\")\n",
            "print(\"=\" * 75)\n"
        ]
    }

    # Append new sections
    cells.extend([sec8_md, sec8_code, sec9_md, sec9_code, sec10_md, sec10_code])
    nb["cells"] = cells

    with open(NOTEBOOK_PATH, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)

    print(f"Successfully updated {NOTEBOOK_PATH} with {len(nb['cells'])} cells.")

if __name__ == "__main__":
    build_updated_notebook()
