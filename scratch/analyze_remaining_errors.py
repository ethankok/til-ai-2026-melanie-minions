import json
import re
from collections import Counter
import sys
import os
import jiwer

# Load original helpers
sys.path.append(os.path.abspath("asr/src"))
from asr_postprocess import _preserve_case

# Import the upgraded functions from test_postprocess_upgrade
sys.path.append(os.path.abspath("scratch"))
from test_postprocess_upgrade import digits_to_words, normalize_text

gold = []
with open("asr.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        gold.append(json.loads(line))

with open("asr_results.json", "r", encoding="utf-8") as f:
    preds = json.load(f)

# Collect all word substitutions where gold word is capitalized (proper noun candidate)
substitutions = Counter()
insertions = Counter()
deletions = Counter()

for idx, (p, g) in enumerate(zip(preds, gold)):
    p_proc = digits_to_words(p)
    
    # Let's align them at the word level using jiwer
    p_norm = normalize_text(p_proc)
    g_norm = normalize_text(g["transcript"])
    
    p_words = p_norm.split()
    g_words = g_norm.split()
    
    # We can also look at the original (un-normalized) gold to check capitalization
    g_orig_words = g["transcript"].split()
    
    # Simple check: if they don't match, let's look at alignment
    if p_norm != g_norm:
        output = jiwer.process_words(g_norm, p_norm)
        # output.alignments is a list of AlignmentChunk
        # Each chunk has type: 'equal', 'substitute', 'insert', 'delete'
        # and src_start_idx, src_end_idx, dest_start_idx, dest_end_idx
        for chunk in output.alignments[0]:
            if chunk.type == "substitute":
                g_sub = g_words[chunk.ref_start_idx:chunk.ref_end_idx]
                p_sub = p_words[chunk.hyp_start_idx:chunk.hyp_end_idx]
                # Find matching original capitalization in gold
                g_orig_sub = g_orig_words[chunk.ref_start_idx:chunk.ref_end_idx] if chunk.ref_start_idx < len(g_orig_words) else g_sub
                
                # Check if any word in the gold original is capitalized (but not start of sentence)
                is_proper_noun = False
                for wi, gw in enumerate(g_orig_sub):
                    # check if capitalized and not first word of sentence
                    global_idx = chunk.ref_start_idx + wi
                    if global_idx > 0 and gw and gw[0].isupper():
                        is_proper_noun = True
                        break
                
                g_phrase = " ".join(g_sub)
                p_phrase = " ".join(p_sub)
                substitutions[(g_phrase, p_phrase, is_proper_noun)] += 1
            elif chunk.type == "delete":
                g_del = g_words[chunk.ref_start_idx:chunk.ref_end_idx]
                deletions[" ".join(g_del)] += 1
            elif chunk.type == "insert":
                p_ins = p_words[chunk.hyp_start_idx:chunk.hyp_end_idx]
                insertions[" ".join(p_ins)] += 1

print("=== Top Proper Noun Substitutions ===")
for (g, p, is_prop), count in substitutions.most_common(50):
    if is_prop:
        print(f"Gold: {g:<25} | Pred: {p:<25} | Count: {count}")

print("\n=== Top General Substitutions ===")
for (g, p, is_prop), count in substitutions.most_common(50):
    if not is_prop:
        print(f"Gold: {g:<25} | Pred: {p:<25} | Count: {count}")
