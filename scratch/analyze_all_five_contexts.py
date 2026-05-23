import json
import re

gold = []
with open("asr.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        gold.append(json.loads(line))

with open("asr_results.json", "r", encoding="utf-8") as f:
    preds = json.load(f)

for idx, (g, p) in enumerate(zip(gold, preds)):
    p_lower = p.lower()
    matches = list(re.finditer(r"\b(hundred|thousand)\s+five\b", p_lower))
    if matches:
        is_phi = "phi" in g["transcript"].lower()
        if is_phi:
            print(f"[{idx}] CORRECT:")
            print(f"  PRED: {p}")
            for m in matches:
                # print 3 words before and after
                words = p_lower.split()
                # find word index
                match_words = m.group(0).split()
                # find start word index
                for i in range(len(words) - 1):
                    if words[i:i+2] == match_words:
                        start_idx = max(0, i - 4)
                        end_idx = min(len(words), i + 6)
                        print(f"    Context: ... {' '.join(words[start_idx:end_idx])} ...")
                        break
            print()
