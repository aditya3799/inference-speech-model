import os
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import json
from transformers import WhisperProcessor, WhisperForConditionalGeneration

processor = WhisperProcessor.from_pretrained("openai/whisper-base")
hf = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base")
suppress = set(hf.config.suppress_tokens)

with open("data/parity_check_results.json") as f:
    res = json.load(f)

for s in res["fp32"]["per_sample"]:
    if not s["is_exact_match"]:
        idx = s["index"]
        utt_id = s["id"]
        pos = s["divergence_pos"]
        e_tok = s["div_engine_token"]
        h_tok = s["div_hf_token"]
        e_dec = processor.tokenizer.decode([e_tok]) if isinstance(e_tok, int) else ""
        h_dec = processor.tokenizer.decode([h_tok]) if isinstance(h_tok, int) else ""
        print(f"=== Sample {idx}: {utt_id} ===")
        print(f"Divergence pos: {pos}")
        print(f"Engine tok: {e_tok} ({repr(e_dec)}) | In suppress_tokens? {e_tok in suppress}")
        print(f"HF tok:     {h_tok} ({repr(h_dec)}) | In suppress_tokens? {h_tok in suppress}")
        print(f"Engine text: {repr(s['engine_text'])}")
        print(f"HF text:     {repr(s['hf_text'])}")
        print(f"Ref text:    {repr(s['reference_text'])}\n")
