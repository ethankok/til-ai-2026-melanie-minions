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
    print(f"=== Target: {target} ===")
    pred_spellings = []
    gold_spellings = []
    for idx, (g, p) in enumerate(zip(gold, preds)):
        g_text = g["transcript"]
        if re.search(rf"\b{target}\b", g_text, re.I):
            # Find the target word in gold
            g_match = re.search(rf"\b({target}s?)\b", g_text, re.I)
            if g_match:
                gold_spellings.append(g_match.group(1).lower())
            
            # Let's see what predictions exist in the corresponding sentence.
            # We can find words around the same position, or print the whole sentence to see the spelling.
            # Let's extract what word matches or overlaps.
            # A simple way: find the word in p that corresponds. Since the sentence lengths and alignments differ,
            # we can print the gold sentence and predicted sentence, but to aggregate, let's look at the actual words.
            # Let's find phrases in PRED that correspond to the target context.
            # e.g., if target is canian, look at canyon, kenyan, kanyean, etc.
            # Let's just print the index, gold phrase, and predicted phrase around the target.
            
            # Find the position of target in gold
            g_pos = re.search(rf"\b{target}\b", g_text, re.I).start()
            # extract a window of 5 words around it
            g_words = g_text.split()
            # find index of word containing target
            t_idx = -1
            for wi, w in enumerate(g_words):
                if re.search(rf"\b{target}\b", w, re.I):
                    t_idx = wi
                    break
            if t_idx != -1:
                g_context = " ".join(g_words[max(0, t_idx-2):t_idx+3])
            else:
                g_context = g_text
                
            print(f"  [{idx}] Gold context: '{g_context}'")
            print(f"        Pred: '{p}'")
            print()
