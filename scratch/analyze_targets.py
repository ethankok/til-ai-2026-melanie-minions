import json
import re

# Load gold transcripts
gold = []
with open("asr.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        gold.append(json.loads(line))

# Load predictions
with open("asr_results.json", "r", encoding="utf-8") as f:
    preds = json.load(f)

# Targets to search in gold
targets = {
    "Canian": r"\bcanian(s?)\b",
    "Hegemony": r"\bhegemony(s?)\b",
    "Sharpsea": r"\bsharpsea\b",
    "Sarento": r"\bsarento\b",
    "Phi": r"\bphi\b"
}

for name, pattern in targets.items():
    print(f"\n=================== SEARCHING FOR GOLD TARGET: {name} ===================")
    rx = re.compile(pattern, re.I)
    matches_count = 0
    for idx, (g, p) in enumerate(zip(gold, preds)):
        g_text = g["transcript"]
        if rx.search(g_text):
            matches_count += 1
            if matches_count <= 15: # print up to 15 examples
                print(f"\n--- Index {idx} ---")
                print(f"GOLD: {g_text}")
                print(f"PRED: {p}")
    print(f"Total gold matches for {name}: {matches_count}")
