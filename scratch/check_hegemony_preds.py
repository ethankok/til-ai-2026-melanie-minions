import json
import re

gold = []
with open("asr.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        gold.append(json.loads(line))

with open("asr_results.json", "r", encoding="utf-8") as f:
    preds = json.load(f)

for idx, (g, p) in enumerate(zip(gold, preds)):
    if "hegemony" in g["transcript"].lower():
        # find what represents hegemony in p
        # We can look for words like hegel, hegemony, hegemoni, hegmoni, etc.
        matches = re.findall(r"\b(hegel|hegemoni|hegmoni|hegemony)\b", p, re.I)
        print(f"[{idx}] Gold: {g['transcript']}")
        print(f"      Pred: {p}")
        print(f"      Matches: {matches}")
        print()
