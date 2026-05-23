import json
import re

with open("asr.jsonl", "r", encoding="utf-8") as f:
    gold = [json.loads(line) for line in f]
with open("asr_results.json", "r", encoding="utf-8") as f:
    preds = json.load(f)

phi_contexts = []
for idx, (g, p) in enumerate(zip(gold, preds)):
    g_text = g["transcript"]
    if "phi" in g_text.lower():
        # Find where phi is in gold, and print the local context of both gold and prediction
        phi_contexts.append((idx, g_text, p))

print(f"Total transcripts containing 'Phi' in gold: {len(phi_contexts)}")
print("\nFirst 40 contexts:")
for idx, g_text, p in phi_contexts[:40]:
    # Let's find the words around "Phi" in gold
    g_words = g_text.split()
    p_words = p.split()
    
    # Simple search for "Phi" (case-insensitive) in gold words
    for i, w in enumerate(g_words):
        if "phi" in w.lower():
            # Get window of 5 words around it in gold
            start = max(0, i - 4)
            end = min(len(g_words), i + 5)
            gold_context = " ".join(g_words[start:end])
            
            # Find corresponding part in prediction (by index ratio or word search)
            # Let's search for phonetically similar words in pred_words around index i
            p_start = max(0, i - 5)
            p_end = min(len(p_words), i + 6)
            pred_context = " ".join(p_words[p_start:p_end])
            
            print(f"Index {idx:4d} | GOLD: {gold_context:<60} | PRED: {pred_context}")
            break
