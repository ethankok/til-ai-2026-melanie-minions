import json
import re

gold = []
with open("asr.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        gold.append(json.loads(line))

for idx, g in enumerate(gold):
    text = g["transcript"].lower()
    for phrase in ["thousand five", "hundred five"]:
        if phrase in text:
            print(f"[{idx}]: {text}")
