import json
import re

with open("asr.jsonl", "r", encoding="utf-8") as f:
    gold = [json.loads(line) for line in f]
with open("asr_results.json", "r", encoding="utf-8") as f:
    preds = json.load(f)

words_to_check = ["five", "file", "pi", "fi", "fee", "fight", "fire", "pie", "files"]

print("Analyzing contexts where prediction has one of the target misheard words and gold has 'Phi':")
for idx, (g, p) in enumerate(zip(gold, preds)):
    g_text = g["transcript"]
    if "phi" in g_text.lower():
        # Check if any of our target words are in prediction
        p_lower = p.lower()
        matched_words = [w for w in words_to_check if re.search(r"\b" + w + r"\b", p_lower)]
        if matched_words:
            # Print the index, the matched word, and the sentence segment
            # Find the target word in prediction
            for w in matched_words:
                # Find matching context in PRED
                p_match = re.search(r"(\S+\s+){0,3}\b" + w + r"\b(\s+\S+){0,3}", p_lower)
                g_match = re.search(r"(\S+\s+){0,3}\bphi\b(\s+\S+){0,3}", g_text.lower())
                
                pred_ctx = p_match.group(0) if p_match else ""
                gold_ctx = g_match.group(0) if g_match else ""
                
                print(f"Index {idx:4d} | Word: {w:<6} | PRED: '{pred_ctx}' | GOLD: '{gold_ctx}'")
                break
