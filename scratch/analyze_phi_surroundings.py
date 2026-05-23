import json
import re
from collections import Counter

gold = []
with open("asr.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        gold.append(json.loads(line))

before_words = Counter()
after_words = Counter()

for g in gold:
    text = g["transcript"].lower()
    # clean punctuation except hyphen and letters
    text_clean = re.sub(r"[^\w\s-]", "", text)
    words = text_clean.split()
    for i, w in enumerate(words):
        if w == "phi":
            # 2 words before
            for offset in [1, 2]:
                if i - offset >= 0:
                    before_words[words[i - offset]] += 1
            # 2 words after
            for offset in [1, 2]:
                if i + offset < len(words):
                    after_words[words[i + offset]] += 1

print("Top words BEFORE 'phi':")
for w, c in before_words.most_common(30):
    print(f"  {w}: {c}")

print("\nTop words AFTER 'phi':")
for w, c in after_words.most_common(30):
    print(f"  {w}: {c}")
