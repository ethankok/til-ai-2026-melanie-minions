import json
import re
from collections import Counter

gold = []
with open("asr.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        gold.append(json.loads(line))

with open("asr_results.json", "r", encoding="utf-8") as f:
    preds = json.load(f)

print("=== CANIAN ===")
canian_preds = Counter()
for g, p in zip(gold, preds):
    if "canian" in g["transcript"].lower():
        # find matches in p
        # What words are in p?
        words = re.findall(r"\b\w+\b", p.lower())
        for w in words:
            if w in ["canyon", "canyons", "kenyan", "kenyans", "kanyean", "kanyeans", "khanian", "khanians", "canian", "canians", "cania", "kanyan", "kanyans", "kenyan's"]:
                canian_preds[w] += 1
            # Check for two word phrases like "kenya and" or something
        # Let's print the sentences to be sure
        print(f"G: {g['transcript']}")
        print(f"P: {p}")
        print()

print("=== HEGEMONY ===")
hegemony_preds = Counter()
for g, p in zip(gold, preds):
    if "hegemony" in g["transcript"].lower():
        print(f"G: {g['transcript']}")
        print(f"P: {p}")
        print()

print("=== SHARPSEA ===")
sharpsea_preds = Counter()
for g, p in zip(gold, preds):
    if "sharpsea" in g["transcript"].lower():
        # print first few words of pred matching sharp
        # Let's find patterns in pred: e.g. sharp c, sharp sea, etc.
        matches = re.findall(r"\bsharp\s+\w+\b", p, re.I)
        for m in matches:
            sharpsea_preds[m.lower()] += 1
        print(f"G: {g['transcript']}")
        print(f"P: {p}")
        print()

print("Canian pred word counts:", canian_preds)
print("Sharpsea pred phrase counts:", sharpsea_preds)
