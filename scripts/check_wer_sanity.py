import os
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import json
import soundfile as sf
import torch
import jiwer
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16

processor = WhisperProcessor.from_pretrained("openai/whisper-base")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base").to(device, dtype=dtype).eval()

english_norm = processor.tokenizer.normalize
basic_norm = BasicTextNormalizer()

with open("data/librispeech_subset_100.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print(f"Loaded {len(data)} utterances from LibriSpeech subset.", flush=True)

results = []
for i, item in enumerate(data):
    wav, sr = sf.read(item["path"])
    mel = processor(wav, sampling_rate=16000, return_tensors="pt").input_features.to(device, dtype=dtype)
    
    with torch.no_grad():
        out = model.generate(mel, max_new_tokens=150, do_sample=False, num_beams=1, language="en", task="transcribe")
    
    hyp_text = processor.decode(out[0], skip_special_tokens=True)
    ref_text = item["text"]
    
    norm_ref_en = english_norm(ref_text)
    norm_hyp_en = english_norm(hyp_text)
    norm_ref_basic = basic_norm(ref_text)
    norm_hyp_basic = basic_norm(hyp_text)
    
    utt_wer_en = jiwer.wer(norm_ref_en, norm_hyp_en) * 100.0 if len(norm_ref_en.split()) > 0 else 0.0
    
    results.append({
        "id": item["id"],
        "duration": item["duration"],
        "stratum": item["stratum"],
        "ref": ref_text,
        "hyp": hyp_text,
        "norm_ref_en": norm_ref_en,
        "norm_hyp_en": norm_hyp_en,
        "norm_ref_basic": norm_ref_basic,
        "norm_hyp_basic": norm_hyp_basic,
        "wer_en": utt_wer_en,
        "tokens_gen": len(out[0]),
    })
    if (i + 1) % 20 == 0:
        print(f"Processed {i + 1}/{len(data)} utterances...", flush=True)

# Corpus WER
corpus_wer_en = jiwer.wer([r["norm_ref_en"] for r in results], [r["norm_hyp_en"] for r in results]) * 100.0
corpus_wer_basic = jiwer.wer([r["norm_ref_basic"] for r in results], [r["norm_hyp_basic"] for r in results]) * 100.0

print("\n" + "=" * 80, flush=True)
print(f"OVERALL CORPUS WER (EnglishTextNormalizer, max_new_tokens=150): {corpus_wer_en:.2f}%", flush=True)
print(f"OVERALL CORPUS WER (BasicTextNormalizer, max_new_tokens=150):   {corpus_wer_basic:.2f}%", flush=True)
print("=" * 80, flush=True)

# Per stratum WER
for stratum in ["short", "medium", "long"]:
    strat_items = [r for r in results if r["stratum"] == stratum]
    s_wer = jiwer.wer([r["norm_ref_en"] for r in strat_items], [r["norm_hyp_en"] for r in strat_items]) * 100.0
    ref_tokens = [len(processor.tokenizer.encode(r["ref"])) for r in strat_items]
    gen_tokens = [r["tokens_gen"] for r in strat_items]
    print(f"Stratum '{stratum.upper():<6}' ({len(strat_items)} utterances): WER = {s_wer:.2f}% | Max Gen Tokens: {max(gen_tokens)} (Max Ref: {max(ref_tokens)})", flush=True)

# Top 10 worst utterances
results.sort(key=lambda x: x["wer_en"], reverse=True)
print("\n" + "=" * 80, flush=True)
print("TOP 10 WORST UTTERANCES (ref vs hyp):", flush=True)
print("=" * 80, flush=True)
for idx, r in enumerate(results[:10]):
    print(f"\n[{idx+1:02d}] ID: {r['id']} | Stratum: {r['stratum']} | Duration: {r['duration']:.2f}s | WER: {r['wer_en']:.2f}% | Tokens: {r['tokens_gen']}", flush=True)
    print(f"     REF (norm): \"{r['norm_ref_en']}\"", flush=True)
    print(f"     HYP (norm): \"{r['norm_hyp_en']}\"", flush=True)
