"""Task 3: Encoder Context-Sensitivity and Numerical Calibration for Whisper-base.

Evaluates whether chunk-level KV cache sharing is mathematically and linguistically
sound for a bidirectional speech encoder (Whisper-base).

Experiments:
1. Numerical Noise Floor Calibration:
   - Same input encoded twice (determinism)
   - Same input in different batch positions (position invariance)
   - FP16 vs FP32 precision floor
2. Experiment E1 (Trailing Context):
   - Utterance A alone vs Utterance A followed by Utterance B at identical offset (0s).
   - Isolates bidirectional self-attention across audio boundaries.
3. Experiment E2 (Offset Shift):
   - Utterance A at offset 0s vs Utterance A shifted by 2s, 5s, 10s.
   - Mixes context change with positional embedding change.
4. Downstream Transcription Impact:
   - Splicing context-polluted or offset-shifted encoder blocks into greedy decode.
   - Measures token exact match, divergence token position, and WER delta.
5. Decision Rule:
   - If relative L2 error > 10x noise floor OR any transcript diverges,
     declares chunk-level sharing UNSOUND.
"""

import argparse
import json
import logging
from pathlib import Path
import soundfile as sf
import torch
import torch.nn.functional as F
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
)
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
import jiwer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("context_sensitivity")

MODEL_ID = "openai/whisper-base"
SAMPLING_RATE = 16000
MEL_FRAMES_PER_SEC = 100
ENC_POS_PER_SEC = 50  # 2x conv downsampling


def compute_tensor_diff_metrics(t1: torch.Tensor, t2: torch.Tensor) -> dict:
    """Computes max abs diff, mean abs diff, relative L2 error, and cosine similarity."""
    t1_f = t1.to(torch.float32)
    t2_f = t2.to(torch.float32)
    diff = (t1_f - t2_f).abs()
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())

    l2_diff = float(torch.norm(t1_f - t2_f, p=2).item())
    l2_t1 = float(torch.norm(t1_f, p=2).item())
    rel_l2 = l2_diff / (l2_t1 + 1e-12)

    # Cosine similarity across flattened vectors
    v1 = t1_f.reshape(-1)
    v2 = t2_f.reshape(-1)
    cos_sim = float(F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item())

    return {
        "max_abs_diff": max_abs,
        "mean_abs_diff": mean_abs,
        "relative_l2_error": rel_l2,
        "cosine_similarity": cos_sim,
    }


def greedy_decode_hidden(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    encoder_hidden_states: torch.Tensor,
    device: torch.device,
    max_new_tokens: int = 128,
) -> tuple[list[int], str]:
    """Greedy decodes using precomputed encoder hidden states."""
    enc_out = BaseModelOutput(last_hidden_state=encoder_hidden_states.to(device))
    prompt_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe", no_timestamps=True)
    forced_ids = [token_id for _, token_id in prompt_ids]
    # SOT sequence: [50258, 50259, 50359, 50363]
    initial_tokens = [model.config.decoder_start_token_id] + forced_ids

    # Run generate with forced prompt tokens
    output_ids = model.generate(
        encoder_outputs=enc_out,
        decoder_input_ids=torch.tensor([initial_tokens], device=device),
        max_new_tokens=max_new_tokens,
        num_beams=1,
        do_sample=False,
        suppress_tokens=model.generation_config.suppress_tokens,
    )[0].tolist()

    text = processor.batch_decode([output_ids], skip_special_tokens=True)[0]
    return output_ids, text


def find_first_divergence(seq1: list[int], seq2: list[int]) -> int | None:
    min_len = min(len(seq1), len(seq2))
    for i in range(min_len):
        if seq1[i] != seq2[i]:
            return i
    if len(seq1) != len(seq2):
        return min_len
    return None


def run_evaluation(
    subset_path: Path,
    output_path: Path,
    num_pairs: int = 5,
    quick: bool = False,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    device = torch.device(device_str)
    logger.info("Running on device: %s", device)

    # Load processor and model
    processor = WhisperProcessor.from_pretrained(MODEL_ID)
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID).to(device)
    model.eval()
    normalizer = BasicTextNormalizer()

    # Load dataset
    with open(subset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Filter short / medium utterances for pairs
    short_items = [x for x in data if x.get("duration", 0) <= 6.0 and Path(x["path"]).exists()]
    if not short_items:
        raise RuntimeError("No short utterances found in dataset!")

    if quick:
        num_pairs = min(2, len(short_items) // 2)
    else:
        num_pairs = min(num_pairs, len(short_items) // 2)

    logger.info("Selected %d utterance pairs for context sensitivity evaluation", num_pairs)

    # -------------------------------------------------------------
    # 1. Numerical Noise Floor Calibration
    # -------------------------------------------------------------
    logger.info("=== Phase 1: Numerical Noise Floor Calibration ===")
    calib_sample = short_items[0]
    wav_calib, _ = sf.read(calib_sample["path"])
    features_calib = processor.feature_extractor(
        wav_calib, sampling_rate=SAMPLING_RATE, return_tensors="pt"
    ).input_features.to(device)

    with torch.no_grad():
        # Determinism run 1 vs run 2 (FP32)
        model_fp32 = model.to(torch.float32)
        feat_fp32 = features_calib.to(torch.float32)
        enc_run1 = model_fp32.model.encoder(feat_fp32).last_hidden_state
        enc_run2 = model_fp32.model.encoder(feat_fp32).last_hidden_state
        det_metrics_fp32 = compute_tensor_diff_metrics(enc_run1, enc_run2)

        # Batch position invariance: [x] vs [x, x] at pos 0 and pos 1
        batch_feat = torch.cat([feat_fp32, feat_fp32], dim=0)
        enc_batch = model_fp32.model.encoder(batch_feat).last_hidden_state
        batch_pos0_metrics = compute_tensor_diff_metrics(enc_run1[0], enc_batch[0])
        batch_pos1_metrics = compute_tensor_diff_metrics(enc_run1[0], enc_batch[1])

        # FP16 vs FP32
        if device.type == "cuda":
            model_fp16 = WhisperForConditionalGeneration.from_pretrained(
                MODEL_ID, torch_dtype=torch.float16
            ).to(device)
            model_fp16.eval()
            feat_fp16 = features_calib.to(torch.float16)
            enc_fp16 = model_fp16.model.encoder(feat_fp16).last_hidden_state
            fp16_vs_fp32_metrics = compute_tensor_diff_metrics(enc_run1, enc_fp16)
            # Restore model to fp16 for fast evaluation if CUDA
            eval_model = model_fp16
            eval_dtype = torch.float16
        else:
            fp16_vs_fp32_metrics = det_metrics_fp32  # fallback on CPU
            eval_model = model_fp32
            eval_dtype = torch.float32

    calibration_results = {
        "sample_id": calib_sample["id"],
        "duration_sec": calib_sample["duration"],
        "determinism_fp32": det_metrics_fp32,
        "batch_position_0": batch_pos0_metrics,
        "batch_position_1": batch_pos1_metrics,
        "fp16_vs_fp32": fp16_vs_fp32_metrics,
        "noise_floor_relative_l2": max(
            det_metrics_fp32["relative_l2_error"],
            batch_pos0_metrics["relative_l2_error"],
            batch_pos1_metrics["relative_l2_error"],
            fp16_vs_fp32_metrics["relative_l2_error"],
        ),
        "noise_floor_max_abs": max(
            det_metrics_fp32["max_abs_diff"],
            batch_pos0_metrics["max_abs_diff"],
            batch_pos1_metrics["max_abs_diff"],
            fp16_vs_fp32_metrics["max_abs_diff"],
        ),
    }

    noise_floor_rel_l2 = calibration_results["noise_floor_relative_l2"]
    logger.info(
        "Noise floor established: rel_l2=%.6e, max_abs=%.6e",
        noise_floor_rel_l2,
        calibration_results["noise_floor_max_abs"],
    )

    # -------------------------------------------------------------
    # 2. Experiments E1 & E2 across utterance pairs
    # -------------------------------------------------------------
    e1_results = []
    e2_results = []
    downstream_e1_results = []
    downstream_e2_results = []

    for pair_idx in range(num_pairs):
        item_a = short_items[2 * pair_idx]
        item_b = short_items[2 * pair_idx + 1]

        wav_a, sr_a = sf.read(item_a["path"])
        wav_b, sr_b = sf.read(item_b["path"])
        dur_a = len(wav_a) / sr_a
        dur_b = len(wav_b) / sr_b

        # Number of active encoder positions covering A's audio:
        # 16kHz audio: 160 samples = 1 mel frame (100 fps) -> 2 mel frames = 1 encoder position (50 pos/s)
        n_enc_a = min(1500, int(dur_a * ENC_POS_PER_SEC))

        logger.info(
            "Pair %d: Utterance A=%s (%.2fs, %d enc pos), B=%s (%.2fs)",
            pair_idx,
            item_a["id"],
            dur_a,
            n_enc_a,
            item_b["id"],
            dur_b,
        )

        with torch.no_grad():
            # A alone (standard zero-padded to 30s)
            feat_a = processor.feature_extractor(
                wav_a, sampling_rate=SAMPLING_RATE, return_tensors="pt"
            ).input_features.to(device, dtype=eval_dtype)
            enc_out_a = eval_model.model.encoder(
                feat_a, output_hidden_states=True
            )
            h_a_full = enc_out_a.last_hidden_state  # [1, 1500, 512]
            h_a_slice = h_a_full[:, :n_enc_a, :]

            # Reference baseline greedy decode for A
            tokens_ref_a, text_ref_a = greedy_decode_hidden(
                eval_model, processor, h_a_full, device
            )
            norm_ref_a = normalizer(text_ref_a)
            gt_norm_a = normalizer(item_a["text"])
            base_wer = jiwer.wer(gt_norm_a, norm_ref_a)

            # ---------------------------------------------------------
            # Experiment E1: Trailing Context (A alone vs A followed by B)
            # ---------------------------------------------------------
            wav_ab = list(wav_a) + list(wav_b)
            feat_ab = processor.feature_extractor(
                wav_ab, sampling_rate=SAMPLING_RATE, return_tensors="pt"
            ).input_features.to(device, dtype=eval_dtype)
            enc_out_ab = eval_model.model.encoder(
                feat_ab, output_hidden_states=True
            )
            h_ab_full = enc_out_ab.last_hidden_state
            h_ab_slice = h_ab_full[:, :n_enc_a, :]

            # Compare encoder outputs strictly over A's active audio positions [0:n_enc_a]
            e1_final_metrics = compute_tensor_diff_metrics(h_a_slice, h_ab_slice)

            # Per-layer metrics for E1 (layers 0..6)
            layer_metrics_e1 = []
            for l_idx, (hl_a, hl_ab) in enumerate(
                zip(enc_out_a.hidden_states, enc_out_ab.hidden_states)
            ):
                hl_a_slice = hl_a[:, :n_enc_a, :]
                hl_ab_slice = hl_ab[:, :n_enc_a, :]
                m = compute_tensor_diff_metrics(hl_a_slice, hl_ab_slice)
                m["layer"] = l_idx
                layer_metrics_e1.append(m)

            e1_results.append({
                "pair_idx": pair_idx,
                "utterance_a_id": item_a["id"],
                "utterance_b_id": item_b["id"],
                "duration_a": dur_a,
                "duration_b": dur_b,
                "n_enc_positions": n_enc_a,
                "final_layer_metrics": e1_final_metrics,
                "per_layer_metrics": layer_metrics_e1,
            })

            # Downstream impact of E1: Splice h_ab_slice into A's position in h_a_full
            h_spliced_e1 = h_a_full.clone()
            h_spliced_e1[:, :n_enc_a, :] = h_ab_slice

            tokens_e1, text_e1 = greedy_decode_hidden(
                eval_model, processor, h_spliced_e1, device
            )
            norm_e1 = normalizer(text_e1)
            e1_wer = jiwer.wer(gt_norm_a, norm_e1)
            exact_match_e1 = tokens_ref_a == tokens_e1
            diverge_idx_e1 = find_first_divergence(tokens_ref_a, tokens_e1)

            downstream_e1_results.append({
                "pair_idx": pair_idx,
                "utterance_a_id": item_a["id"],
                "baseline_tokens": tokens_ref_a,
                "baseline_text": text_ref_a,
                "baseline_normalized": norm_ref_a,
                "baseline_wer": base_wer,
                "spliced_tokens": tokens_e1,
                "spliced_text": text_e1,
                "spliced_normalized": norm_e1,
                "spliced_wer": e1_wer,
                "exact_match": exact_match_e1,
                "first_divergence_idx": diverge_idx_e1,
                "wer_delta": e1_wer - base_wer,
            })

            # ---------------------------------------------------------
            # Experiment E2: Offset Shift (A at offset 0 vs A at 2s, 5s, 10s)
            # ---------------------------------------------------------
            offsets_sec = [2.0, 5.0, 10.0]
            e2_pair_shifts = []
            downstream_e2_pair = []

            for offset_s in offsets_sec:
                if dur_a + offset_s > 29.5:
                    continue  # fits within 30s window

                silence_samples = [0.0] * int(offset_s * SAMPLING_RATE)
                wav_shifted = silence_samples + list(wav_a)
                feat_shifted = processor.feature_extractor(
                    wav_shifted, sampling_rate=SAMPLING_RATE, return_tensors="pt"
                ).input_features.to(device, dtype=eval_dtype)

                enc_out_shifted = eval_model.model.encoder(
                    feat_shifted, output_hidden_states=True
                )
                h_shifted_full = enc_out_shifted.last_hidden_state

                # Offset start in encoder positions:
                start_enc = int(offset_s * ENC_POS_PER_SEC)
                end_enc = start_enc + n_enc_a
                h_shifted_slice = h_shifted_full[:, start_enc:end_enc, :]

                shift_metrics = compute_tensor_diff_metrics(h_a_slice, h_shifted_slice)
                shift_metrics["offset_sec"] = offset_s
                shift_metrics["start_enc_pos"] = start_enc
                e2_pair_shifts.append(shift_metrics)

                # Downstream impact: Splice shifted slice into offset 0 of A's full tensor
                h_spliced_e2 = h_a_full.clone()
                h_spliced_e2[:, :n_enc_a, :] = h_shifted_slice

                tokens_e2, text_e2 = greedy_decode_hidden(
                    eval_model, processor, h_spliced_e2, device
                )
                norm_e2 = normalizer(text_e2)
                e2_wer = jiwer.wer(gt_norm_a, norm_e2)
                exact_match_e2 = tokens_ref_a == tokens_e2
                diverge_idx_e2 = find_first_divergence(tokens_ref_a, tokens_e2)

                downstream_e2_pair.append({
                    "offset_sec": offset_s,
                    "spliced_tokens": tokens_e2,
                    "spliced_text": text_e2,
                    "spliced_normalized": norm_e2,
                    "spliced_wer": e2_wer,
                    "exact_match": exact_match_e2,
                    "first_divergence_idx": diverge_idx_e2,
                    "wer_delta": e2_wer - base_wer,
                })

            e2_results.append({
                "pair_idx": pair_idx,
                "utterance_a_id": item_a["id"],
                "duration_a": dur_a,
                "n_enc_positions": n_enc_a,
                "shifts": e2_pair_shifts,
            })
            downstream_e2_results.append({
                "pair_idx": pair_idx,
                "utterance_a_id": item_a["id"],
                "baseline_tokens": tokens_ref_a,
                "baseline_text": text_ref_a,
                "shifts": downstream_e2_pair,
            })

    # -------------------------------------------------------------
    # 3. Decision Rule Assessment
    # -------------------------------------------------------------
    # Rule: If relative L2 error > 10x noise floor OR any transcript diverges,
    # declare chunk-level sharing UNSOUND for bidirectional speech encoders.
    max_e1_rel_l2 = max(r["final_layer_metrics"]["relative_l2_error"] for r in e1_results)
    max_e2_rel_l2 = max(
        max(s["relative_l2_error"] for s in r["shifts"]) for r in e2_results if r["shifts"]
    )
    overall_max_rel_l2 = max(max_e1_rel_l2, max_e2_rel_l2)
    noise_floor_ratio = overall_max_rel_l2 / (noise_floor_rel_l2 + 1e-12)

    e1_any_diverged = any(not r["exact_match"] for r in downstream_e1_results)
    e2_any_diverged = any(
        any(not s["exact_match"] for s in r["shifts"]) for r in downstream_e2_results
    )
    any_transcript_diverged = e1_any_diverged or e2_any_diverged

    is_unsound = (noise_floor_ratio > 10.0) or any_transcript_diverged

    verdict = {
        "verdict": "UNSOUND" if is_unsound else "SOUND",
        "recommendation": (
            "full-utterance-only sharing (chunk-level sharing violates bidirectional self-attention invariance)"
            if is_unsound
            else "chunk-level sharing permitted"
        ),
        "criteria": {
            "noise_floor_relative_l2": noise_floor_rel_l2,
            "max_measured_relative_l2": overall_max_rel_l2,
            "noise_floor_multiplier": noise_floor_ratio,
            "threshold_multiplier": 10.0,
            "exceeds_10x_noise_floor": bool(noise_floor_ratio > 10.0),
            "transcript_diverged_e1_trailing": e1_any_diverged,
            "transcript_diverged_e2_offset": e2_any_diverged,
            "any_transcript_diverged": any_transcript_diverged,
        },
    }

    logger.info("=======================================================")
    logger.info("VERDICT: %s", verdict["verdict"])
    logger.info("RECOMMENDATION: %s", verdict["recommendation"])
    logger.info("Noise floor rel L2: %.6e | Max rel L2: %.6e (%.1fx noise floor)",
                noise_floor_rel_l2, overall_max_rel_l2, noise_floor_ratio)
    logger.info("Any transcript diverged: %s (E1: %s, E2: %s)",
                any_transcript_diverged, e1_any_diverged, e2_any_diverged)
    logger.info("=======================================================")

    full_output = {
        "model_id": MODEL_ID,
        "device": device_str,
        "calibration": calibration_results,
        "experiment_e1_trailing_context": e1_results,
        "downstream_e1_transcription": downstream_e1_results,
        "experiment_e2_offset_shift": e2_results,
        "downstream_e2_transcription": downstream_e2_results,
        "decision_rule": verdict,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)

    logger.info("Results successfully saved to %s", output_path)
    return full_output


def main():
    parser = argparse.ArgumentParser(description="Evaluate encoder context-sensitivity on Whisper-base")
    parser.add_argument("--subset", type=Path, default=Path("data/librispeech_subset_100.json"))
    parser.add_argument("--output", type=Path, default=Path("data/encoder_context_sensitivity_results.json"))
    parser.add_argument("--num-pairs", type=int, default=5)
    parser.add_argument("--quick", action="store_true", help="Quick run with 2 pairs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    run_evaluation(
        subset_path=args.subset,
        output_path=args.output,
        num_pairs=args.num_pairs,
        quick=args.quick,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
