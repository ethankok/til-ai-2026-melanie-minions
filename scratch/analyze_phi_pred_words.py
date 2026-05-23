import json
import re

with open("asr.jsonl", "r", encoding="utf-8") as f:
    gold = [json.loads(line) for line in f]
with open("asr_results.json", "r", encoding="utf-8") as f:
    preds = json.load(f)

# Words to ignore (not part of ASR differences)
# We want to find the exact word in the prediction that corresponds to "Phi" (or "phi") in gold.
# Since we have aligned lines, let's split both into words and look for "phi" in gold.
# To do alignment, we can use a simple edit distance or sequence matcher.
from difflib import SequenceMatcher

word_map = {}

for idx, (g, p) in enumerate(zip(gold, preds)):
    g_text = g["transcript"].lower()
    # clean punctuation for simple word alignment
    g_clean = re.sub(r"[^\w\s-]", "", g_text)
    p_clean = re.sub(r"[^\w\s-]", "", p.lower())
    
    g_words = g_clean.split()
    p_words = p_clean.split()
    
    if "phi" in g_words:
        # Use SequenceMatcher to find alignment
        matcher = SequenceMatcher(None, g_words, p_words)
        opcodes = matcher.get_opcodes()
        
        for tag, i1, i2, j1, j2 in opcodes:
            # We look for blocks where gold has "phi"
            # tag can be 'replace', 'equal', 'delete', 'insert'
            # Check if "phi" is in g_words[i1:i2]
            for idx_g in range(i1, i2):
                if g_words[idx_g] == "phi":
                    # The corresponding words in pred are p_words[j1:j2]
                    pred_sub = p_words[j1:j2]
                    pred_str = " ".join(pred_sub)
                    word_map[pred_str] = word_map.get(pred_str, 0) + 1

# Sort by frequency
for k, v in sorted(word_map.items(), key=lambda x: x[1], reverse=True):
    print(f"'{k}': {v} times")
