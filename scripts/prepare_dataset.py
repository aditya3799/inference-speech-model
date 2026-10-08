import soundfile as sf
from pathlib import Path
import random
import json

root = Path("data/LibriSpeech/test-clean")
transcripts = {}
for trans_file in root.glob("**/*.trans.txt"):
    with open(trans_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(" ", 1)
            if len(parts) == 2:
                transcripts[parts[0]] = parts[1]

flac_files = sorted(list(root.glob("**/*.flac")))

items = []
for p in flac_files:
    utt_id = p.stem
    if utt_id in transcripts:
        info = sf.info(str(p))
        duration = info.duration
        if duration <= 30.0:
            items.append({
                "id": utt_id,
                "path": str(p.resolve()).replace("\\", "/"),
                "duration": round(duration, 3),
                "text": transcripts[utt_id]
            })

random.seed(42)
short = sorted([x for x in items if x["duration"] < 5.0], key=lambda x: x["id"])
med = sorted([x for x in items if 5.0 <= x["duration"] <= 12.0], key=lambda x: x["id"])
long = sorted([x for x in items if 12.0 < x["duration"] <= 30.0], key=lambda x: x["id"])

random.shuffle(short)
random.shuffle(med)
random.shuffle(long)

selected_short = [{**x, "stratum": "short"} for x in short[:33]]
selected_med = [{**x, "stratum": "medium"} for x in med[:34]]
selected_long = [{**x, "stratum": "long"} for x in long[:33]]

dataset_100 = selected_short + selected_med + selected_long
random.seed(42)
random.shuffle(dataset_100)

out_file = Path("data/librispeech_subset_100.json")
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(dataset_100, f, indent=2)

print(f"Saved {len(dataset_100)} stratified items to {out_file}")
print(f"Short (<5s): {len([x for x in dataset_100 if x['stratum'] == 'short'])}")
print(f"Medium (5-12s): {len([x for x in dataset_100 if x['stratum'] == 'medium'])}")
print(f"Long (12-30s): {len([x for x in dataset_100 if x['stratum'] == 'long'])}")
