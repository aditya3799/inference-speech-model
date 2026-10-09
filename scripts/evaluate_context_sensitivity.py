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

import os
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
import numpy as np

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

    # Use all valid utterances in dataset for pairing
    items = [x for x in data if Path(x["path"]).exists()]
    if not items:
        raise RuntimeError("No valid utterances found in dataset!")

    if quick:
        num_pairs = min(2, len(items) // 2)
    else:
        num_pairs = min(num_pairs, len(items) // 2)

    logger.info("Selected %d utterance pairs for context sensitivity evaluation", num_pairs)

    # -------------------------------------------------------------
    # 1. Numerical Noise Floor Calibration
    # -------------------------------------------------------------
    logger.info("=== Phase 1: Numerical Noise Floor Calibration ===")
    calib_sample = items[0]
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
                MODEL_ID, dtype=torch.float16, low_cpu_mem_usage=True
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
        item_a = items[2 * pair_idx]
        item_b = items[2 * pair_idx + 1]

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
    # 3. Aggregate Summaries: E1 and E2 Reported Separately
    # -------------------------------------------------------------
    # Experiment E1 Aggregates (Trailing Context across identical offset 0s)
    e1_rel_l2s = [r["final_layer_metrics"]["relative_l2_error"] for r in e1_results]
    e1_median_rel_l2 = float(np.median(e1_rel_l2s))
    e1_p95_rel_l2 = float(np.percentile(e1_rel_l2s, 95))
    e1_max_rel_l2 = float(np.max(e1_rel_l2s))

    # E1 Per-layer relative L2 diffs (layers 0..6)
    num_encoder_layers = len(e1_results[0]["per_layer_metrics"])
    e1_per_layer_stats = []
    for l_idx in range(num_encoder_layers):
        l_diffs = [r["per_layer_metrics"][l_idx]["relative_l2_error"] for r in e1_results]
        e1_per_layer_stats.append({
            "layer": l_idx,
            "median_relative_l2": float(np.median(l_diffs)),
            "p95_relative_l2": float(np.percentile(l_diffs, 95)),
            "max_relative_l2": float(np.max(l_diffs)),
        })

    # E1 Downstream spliced transcription statistics
    e1_base_wers = [r["baseline_wer"] for r in downstream_e1_results]
    e1_spliced_wers = [r["spliced_wer"] for r in downstream_e1_results]
    e1_wer_deltas = [r["wer_delta"] for r in downstream_e1_results]
    e1_exact_matches = [r["exact_match"] for r in downstream_e1_results]
    e1_diverged_count = sum(1 for m in e1_exact_matches if not m)

    def _bootstrap_ci(data, n_boot=2000, alpha=0.05):
        if not data:
            return [0.0, 0.0]
        rng = np.random.default_rng(42)
        means = [rng.choice(data, size=len(data), replace=True).mean() for _ in range(n_boot)]
        return [float(np.percentile(means, 100 * (alpha / 2))), float(np.percentile(means, 100 * (1 - alpha / 2)))]

    e1_wer_ci = _bootstrap_ci(e1_wer_deltas)

    e1_summary = {
        "num_pairs_evaluated": len(e1_results),
        "median_relative_l2": e1_median_rel_l2,
        "p95_relative_l2": e1_p95_rel_l2,
        "max_relative_l2": e1_max_rel_l2,
        "per_layer_relative_l2": e1_per_layer_stats,
        "baseline_wer_mean": float(np.mean(e1_base_wers)) if e1_base_wers else 0.0,
        "baseline_wer_median": float(np.median(e1_base_wers)) if e1_base_wers else 0.0,
        "spliced_wer_mean": float(np.mean(e1_spliced_wers)) if e1_spliced_wers else 0.0,
        "spliced_wer_median": float(np.median(e1_spliced_wers)) if e1_spliced_wers else 0.0,
        "wer_delta_mean": float(np.mean(e1_wer_deltas)) if e1_wer_deltas else 0.0,
        "wer_delta_ci_95": e1_wer_ci,
        "wer_delta_max": float(np.max(e1_wer_deltas)) if e1_wer_deltas else 0.0,
        "exact_token_match_rate": float(sum(e1_exact_matches) / len(e1_exact_matches)) if e1_exact_matches else 1.0,
        "diverged_pairs_count": e1_diverged_count,
    }

    # Experiment E2 Aggregates (Offset Shifts)
    e2_rel_l2s = [s["relative_l2_error"] for r in e2_results for s in r["shifts"]]
    e2_median_rel_l2 = float(np.median(e2_rel_l2s)) if e2_rel_l2s else 0.0
    e2_p95_rel_l2 = float(np.percentile(e2_rel_l2s, 95)) if e2_rel_l2s else 0.0
    e2_max_rel_l2 = float(np.max(e2_rel_l2s)) if e2_rel_l2s else 0.0

    # E2 Per-offset statistics
    e2_per_offset = {}
    for offset_val in [2.0, 5.0, 10.0]:
        offs = [s["relative_l2_error"] for r in e2_results for s in r["shifts"] if s["offset_sec"] == offset_val]
        if offs:
            e2_per_offset[f"offset_{offset_val}s"] = {
                "count": len(offs),
                "median_relative_l2": float(np.median(offs)),
                "p95_relative_l2": float(np.percentile(offs, 95)),
                "max_relative_l2": float(np.max(offs)),
            }

    # E2 Downstream spliced transcription statistics
    e2_spliced_wers = [s["spliced_wer"] for r in downstream_e2_results for s in r["shifts"]]
    e2_wer_deltas = [s["wer_delta"] for r in downstream_e2_results for s in r["shifts"]]
    e2_exact_matches = [s["exact_match"] for r in downstream_e2_results for s in r["shifts"]]
    e2_diverged_count = sum(1 for m in e2_exact_matches if not m)
    e2_wer_ci = _bootstrap_ci(e2_wer_deltas)

    e2_summary = {
        "num_shifts_evaluated": len(e2_rel_l2s),
        "median_relative_l2": e2_median_rel_l2,
        "p95_relative_l2": e2_p95_rel_l2,
        "max_relative_l2": e2_max_rel_l2,
        "per_offset_stats": e2_per_offset,
        "spliced_wer_mean": float(np.mean(e2_spliced_wers)) if e2_spliced_wers else 0.0,
        "spliced_wer_median": float(np.median(e2_spliced_wers)) if e2_spliced_wers else 0.0,
        "wer_delta_mean": float(np.mean(e2_wer_deltas)) if e2_wer_deltas else 0.0,
        "wer_delta_ci_95": e2_wer_ci,
        "wer_delta_max": float(np.max(e2_wer_deltas)) if e2_wer_deltas else 0.0,
        "exact_token_match_rate": float(sum(e2_exact_matches) / len(e2_exact_matches)) if e2_exact_matches else 1.0,
        "diverged_shifts_count": e2_diverged_count,
    }

    # -------------------------------------------------------------
    # 4. Decision Rule Assessment (Focused strictly on E1 + Spliced WER)
    # -------------------------------------------------------------
    e1_p95_noise_ratio = e1_p95_rel_l2 / (noise_floor_rel_l2 + 1e-12)
    e1_max_noise_ratio = e1_max_rel_l2 / (noise_floor_rel_l2 + 1e-12)
    e1_transcript_diverged = bool(e1_diverged_count > 0)
    e1_wer_degraded = bool(np.mean(e1_wer_deltas) > 0.0)

    is_unsound = (e1_p95_noise_ratio > 10.0) or (e1_max_noise_ratio > 10.0) or e1_transcript_diverged or e1_wer_degraded

    evidence_verdict = (
        f"UNSOUND: E1 trailing context diverges at {e1_max_noise_ratio:.0f}x noise floor "
        f"({e1_diverged_count}/{len(e1_results)} diverged pairs, {e1_summary['exact_token_match_rate']*100:.1f}% match rate). "
        f"Chunk-level prefix cache sharing violates bidirectional encoder attention invariance."
        if is_unsound
        else "SOUND: representations invariant within noise floor."
    )

    verdict = {
        "verdict": evidence_verdict,
        "verdict_label": "UNSOUND" if is_unsound else "SOUND",
        "basis": "Evaluated strictly using Experiment E1 (trailing context) statistics plus downstream spliced WER",
        "recommendation": (
            "full-utterance-only sharing (chunk-level sharing violates bidirectional self-attention invariance)"
            if is_unsound
            else "chunk-level sharing permitted"
        ),
        "criteria": {
            "noise_floor_relative_l2": noise_floor_rel_l2,
            "max_measured_relative_l2": e1_max_rel_l2,
            "noise_floor_multiplier": e1_max_noise_ratio,
            "transcript_diverged_e1_trailing": e1_transcript_diverged,
            "e1_median_relative_l2": e1_median_rel_l2,
            "e1_p95_relative_l2": e1_p95_rel_l2,
            "e1_max_relative_l2": e1_max_rel_l2,
            "e1_p95_noise_floor_multiplier": e1_p95_noise_ratio,
            "e1_max_noise_floor_multiplier": e1_max_noise_ratio,
            "threshold_multiplier": 10.0,
            "e1_exceeds_10x_noise_floor": bool(e1_p95_noise_ratio > 10.0 or e1_max_noise_ratio > 10.0),
            "e1_transcripts_diverged": e1_transcript_diverged,
            "e1_diverged_pairs_count": e1_diverged_count,
            "e1_exact_token_match_rate": e1_summary["exact_token_match_rate"],
            "e1_mean_wer_delta": e1_summary["wer_delta_mean"],
            "e1_wer_delta_ci_95": e1_wer_ci,
        },
    }

    logger.info("=======================================================")
    logger.info("DECISION RULE VERDICT: %s", verdict["verdict"])
    logger.info("RECOMMENDATION: %s", verdict["recommendation"])
    logger.info("E1 rel L2: median=%.6e, p95=%.6e, max=%.6e (Noise floor: %.6e)",
                e1_median_rel_l2, e1_p95_rel_l2, e1_max_rel_l2, noise_floor_rel_l2)
    logger.info("E1 noise multiplier: p95=%.1fx, max=%.1fx (threshold=10.0x)",
                e1_p95_noise_ratio, e1_max_noise_ratio)
    logger.info("E1 transcript diverged: %s (%d/%d pairs diverged, match_rate=%.1f%%)",
                e1_transcript_diverged, e1_diverged_count, len(e1_results),
                e1_summary["exact_token_match_rate"] * 100.0)
    logger.info("E1 WER: baseline=%.4f, spliced=%.4f, mean_delta=%+.4f",
                e1_summary["baseline_wer_mean"], e1_summary["spliced_wer_mean"], e1_summary["wer_delta_mean"])
    logger.info("=======================================================")

    full_output = {
        "model_id": MODEL_ID,
        "device": device_str,
        "calibration": calibration_results,
        "experiment_e1_summary": e1_summary,
        "experiment_e1_trailing_context": e1_results,
        "downstream_e1_transcription": downstream_e1_results,
        "experiment_e2_summary": e2_summary,
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
