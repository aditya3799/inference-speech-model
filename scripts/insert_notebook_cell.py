import json

with open("paged_dual_cache_speech_serving.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

# Read the cell code from scripts/display_results.py
with open("scripts/display_results.py", "r", encoding="utf-8") as f:
    cell_code = f.readlines()

new_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": cell_code,
}

# Find the index of the cell with parity_check.py
target_idx = None
for idx, cell in enumerate(nb["cells"]):
    src = "".join(cell.get("source", []))
    if "scripts/parity_check.py" in src:
        target_idx = idx
        break

if target_idx is not None:
    nb["cells"].insert(target_idx + 1, new_cell)
    print(f"Inserted new code cell at index {target_idx + 1}")
else:
    nb["cells"].append(new_cell)
    print("Appended new code cell at end")

with open("paged_dual_cache_speech_serving.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully! Total cells:", len(nb["cells"]))
