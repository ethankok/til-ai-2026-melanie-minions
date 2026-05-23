import json
import re

gold = []
with open("asr.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        gold.append(json.loads(line))

with open("asr_results.json", "r", encoding="utf-8") as f:
    preds = json.load(f)

# Targets to check in gold
targets = ["canian", "hegemony", "sharpsea"]

for target in targets:
    print(f"=== Target: {target} ===")
    matches_count = 0
    for idx, (g, p) in enumerate(zip(gold, preds)):
        g_text = g["transcript"]
        if re.search(rf"\b{target}\b", g_text, re.I):
            matches_count += 1
            print(f"[{idx}]:")
            print(f"  GOLD: {g_text}")
            print(f"  PRED: {p}")
            print()
    print(f"Total occurrences in gold: {matches_count}\n")
