import json
import re

with open("/Users/ethankok/projects/TIL/asr_results.json", "r", encoding="utf-8") as f:
    preds = json.load(f)

# Indices of interest
indices = [53, 1623, 2270, 3365, 4016, 63, 40, 86, 112, 122, 60, 30, 49, 105, 110, 62, 1644, 2328, 2571, 2751, 626]

for idx in sorted(list(set(indices))):
    print(f"[{idx}]: {preds[idx]}")
