import json
import re

gold = []
with open("asr.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        gold.append(json.loads(line))

targets = ["pi", "fi", "fai", "phi"]
for t in targets:
    count = 0
    examples = []
    for idx, g in enumerate(gold):
        text = g["transcript"]
        if re.search(rf"\b{t}\b", text, re.I):
            count += 1
            if len(examples) < 5:
                examples.append((idx, text))
    print(f"Target '{t}' count: {count}")
    for idx, text in examples:
        print(f"  [{idx}]: {text[:100]}...")
