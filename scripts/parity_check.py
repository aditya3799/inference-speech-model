import os
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
import gc
import json
import time
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import soundfile as sf
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

from speech_serving.models import PretrainedWhisperModel
from speech_serving.encoder_block_manager import EncoderBlockManager
from speech_serving.dual_cache_manager import DualCacheManager
from speech_serving.phase_aware_scheduler import (
    PhaseAwareSpeechScheduler,
    WHISPER_DEFAULT_PREFIX_TOKENS,
    WHISPER_EOS_TOKEN_ID,
)
from paged_kv_cache.block_manager import BlockManager


def run_parity_check(dtype_str: str = "float16", num_samples: int = 20):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if dtype_str == "float32" else torch.float16

    print(f"\n=======================================================")
    print(f"Running Parity Check: dtype={dtype_str}, device={device}, num_samples={num_samples}")
    print(f"=======================================================")

    processor = WhisperProcessor.from_pretrained("openai/whisper-base")
    normalizer = BasicTextNormalizer()

    with open("data/librispeech_subset_100.json", "r", encoding="utf-8") as f:
        subset = json.load(f)[:num_samples]

    # Initialize Engine Model & HF Model sharing weights
    engine_model = PretrainedWhisperModel(
        model_name_or_path="openai/whisper-base",
        device=device,
        dtype=dtype,
    )
    hf_model = engine_model.hf_model

    # Allocate physical block pools ONCE outside the loop to avoid continuous reallocation
    enc_bm = EncoderBlockManager(
        num_layers=engine_model.num_encoder_layers,
        num_heads=engine_model.num_heads,
        head_dim=engine_model.head_dim,
        block_size=16,
        num_blocks=256,
        dtype=dtype,
        device=device,
    )
    dec_bm = BlockManager(
        num_layers=engine_model.num_decoder_layers,
        num_kv_heads=engine_model.num_heads,
        head_dim=engine_model.head_dim,
        block_size=16,
        num_blocks=256,
        dtype=dtype,
        device=device,
    )
    cache_mgr = DualCacheManager(enc_bm, dec_bm)
    scheduler = PhaseAwareSpeechScheduler(
        model=engine_model,
        dual_cache_manager=cache_mgr,
        eos_token_id=WHISPER_EOS_TOKEN_ID,
        device=device,
    )

    prompt_tensor = torch.tensor([WHISPER_DEFAULT_PREFIX_TOKENS], device=device, dtype=torch.long)
    results = []
    exact_matches = 0

    for idx, item in enumerate(subset):
        audio, sr = sf.read(item["path"])
        if sr != 16000:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        mel = inputs.input_features.to(device=device, dtype=dtype)

        # 1. Hugging Face Generate
        with torch.no_grad():
            hf_out = hf_model.generate(
                mel,
                num_beams=1,
                do_sample=False,
                language="en",
                task="transcribe",
                max_new_tokens=50,
            )
        hf_full_tokens = hf_out[0].tolist()
        if hf_full_tokens[:4] == WHISPER_DEFAULT_PREFIX_TOKENS:
            hf_gen_tokens = hf_full_tokens[4:]
        else:
            hf_gen_tokens = hf_full_tokens

        # 2. Engine Serving Scheduler (reuses pool, request blocks freed upon completion)
        scheduler.completed.clear()
        req_id = scheduler.submit(
            mel_features=mel[0],
            prompt_token_ids=prompt_tensor,
            max_new_tokens=50,
            eos_token_id=WHISPER_EOS_TOKEN_ID,
        )

        completed = scheduler.run_until_done()
        req = completed[0]
        engine_gen_tokens = list(req.generated_tokens)

        # Standardize EOS handling
        engine_tokens_no_eos = [t for t in engine_gen_tokens if t != WHISPER_EOS_TOKEN_ID]
        hf_tokens_no_eos = [t for t in hf_gen_tokens if t != WHISPER_EOS_TOKEN_ID]

        is_match = (engine_tokens_no_eos == hf_tokens_no_eos)
        if is_match:
            exact_matches += 1
            divergence_pos = None
            div_engine_tok = None
            div_hf_tok = None
        else:
            min_len = min(len(engine_tokens_no_eos), len(hf_tokens_no_eos))
            divergence_pos = min_len
            for pos in range(min_len):
                if engine_tokens_no_eos[pos] != hf_tokens_no_eos[pos]:
                    divergence_pos = pos
                    break
            div_engine_tok = engine_tokens_no_eos[divergence_pos] if divergence_pos < len(engine_tokens_no_eos) else "EOF"
            div_hf_tok = hf_tokens_no_eos[divergence_pos] if divergence_pos < len(hf_tokens_no_eos) else "EOF"

        engine_text = processor.tokenizer.decode(engine_tokens_no_eos)
        hf_text = processor.tokenizer.decode(hf_tokens_no_eos)
        ref_text = item["text"]

        res = {
            "index": idx,
            "id": item["id"],
            "duration": item["duration"],
            "stratum": item["stratum"],
            "is_exact_match": is_match,
            "engine_tokens": engine_tokens_no_eos,
            "hf_tokens": hf_tokens_no_eos,
            "engine_text": engine_text,
            "hf_text": hf_text,
            "reference_text": ref_text,
            "divergence_pos": divergence_pos,
            "div_engine_token": div_engine_tok,
            "div_hf_token": div_hf_tok,
        }
        results.append(res)
        match_str = "MATCH" if is_match else f"MISMATCH at pos {divergence_pos} (Engine={div_engine_tok} vs HF={div_hf_tok})"
        print(f"[{idx+1:02d}/{num_samples}] utt={item['id']} ({item['duration']}s, {item['stratum']}): {match_str}")

    match_rate = exact_matches / num_samples * 100.0
    print(f"\nSummary for {dtype_str}: Exact Match = {exact_matches}/{num_samples} ({match_rate:.1f}%)")

    # Clean up GPU resources
    del scheduler, cache_mgr, enc_bm, dec_bm, engine_model, hf_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "dtype": dtype_str,
        "num_samples": num_samples,
        "exact_matches": exact_matches,
        "exact_match_rate_pct": match_rate,
        "per_sample": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Check exact token parity against Hugging Face generate()")
    parser.add_argument("--num-samples", type=int, default=20, help="Number of utterances to test (default: 20)")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32", "both"], help="Precision to test (default: float16)")
    args = parser.parse_args()

    results = {}
    if args.dtype in ("float16", "both"):
        results["fp16"] = run_parity_check("float16", num_samples=args.num_samples)
    if args.dtype in ("float32", "both"):
        results["fp32"] = run_parity_check("float32", num_samples=args.num_samples)

    output = {
        "environment": {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "model": "openai/whisper-base",
            "commit_hash": "e37978b90ca9030d5170a5c07aadb050351a65bb",
        },
        **results,
    }

    out_path = Path("data/parity_check_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nParity check results successfully saved to {out_path}")


if __name__ == "__main__":
    main()
