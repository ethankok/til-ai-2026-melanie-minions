import json
import re
from collections import Counter

gold = []
with open("asr.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        gold.append(json.loads(line))

with open("asr_results.json", "r", encoding="utf-8") as f:
    preds = json.load(f)

targets = ["canian", "hegemony", "sharpsea"]

for target in targets:
    print(f"\n=================== TARGET: {target} ===================")
    mappings = Counter()
    for idx, (g, p) in enumerate(zip(gold, preds)):
        g_text = g["transcript"].lower()
        p_text = p.lower()
        if target in g_text:
            # Let's find matches in gold and look at pred
            # We can find all instances of the target in the gold text
            # Since the sentences are short, let's extract the sentence and search.
            # Let's search for what the pred has in the corresponding spot.
            # Let's print the specific match pairs
            # For Canian: typically canyon, kenyan, kanyean, etc.
            # For Hegemony: hegemonic, hegemoni, hegemony, etc.
            # For Sharpsea: sharp sea, sharp c, sharp c block, sharpsea bloc, etc.
            print(f"[{idx}] GOLD: {g['transcript']}")
            print(f"      PRED: {p}")
