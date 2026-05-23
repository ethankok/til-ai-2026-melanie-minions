import json
import sys
import os
import jiwer
import re

# Add asr/src to path
sys.path.append(os.path.abspath("asr/src"))
from asr_postprocess import digits_to_words

# Load gold transcripts
gold = []
with open("asr.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        gold.append(json.loads(line))

# Load predictions
with open("asr_results.json", "r", encoding="utf-8") as f:
    preds = json.load(f)

print(f"Loaded {len(preds)} predictions, {len(gold)} gold transcripts.")

wer_transforms = jiwer.Compose(
    [
        jiwer.ToLowerCase(),
        jiwer.SubstituteRegexes({"-": " ", "—": " ", "–": " "}),
        jiwer.RemoveMultipleSpaces(),
        jiwer.RemovePunctuation(),
        jiwer.Strip(),
        jiwer.ReduceToListOfListOfWords(),
    ]
)

cer_transforms = jiwer.Compose(
    [
        jiwer.ToLowerCase(),
        jiwer.SubstituteRegexes({"-": "", "—": "", "–": ""}),
        jiwer.RemoveWhiteSpace(replace_by_space=False),
        jiwer.RemovePunctuation(),
        jiwer.ReduceToListOfListOfChars(),
    ]
)

def normalize_text(text, is_chinese=False):
    transform = cer_transforms if is_chinese else wer_transforms
    words_list = transform([text])
    if is_chinese:
        # words_list is list of list of chars
        return "".join(words_list[0])
    else:
        # words_list is list of list of words
        return " ".join(words_list[0])

mismatches = []
language_totals = {"english": 0, "chinese": 0, "malay": 0, "tamil": 0}
language_errors = {"english": 0, "chinese": 0, "malay": 0, "tamil": 0}

for idx, (p, g) in enumerate(zip(preds, gold)):
    processed_p = digits_to_words(p)
    norm_p = normalize_text(processed_p, g["language"] == "chinese")
    norm_g = normalize_text(g["transcript"], g["language"] == "chinese")
    
    lang = g["language"]
    language_totals[lang] += 1
    
    if norm_p != norm_g:
        language_errors[lang] += 1
        mismatches.append({
            "index": idx,
            "audio": g["audio"],
            "language": lang,
            "gold": norm_g,
            "pred": norm_p,
            "orig_pred": p,
            "orig_gold": g["transcript"]
        })

print("\nMismatch count by language:")
for lang in language_totals:
    total = language_totals[lang]
    errors = language_errors[lang]
    pct_str = f"({errors/total:.4%})" if total > 0 else "(N/A)"
    print(f"  {lang}: {errors} / {total} {pct_str}")

print(f"\nTotal mismatched lines: {len(mismatches)} / {len(preds)}")

print("\nDetail of first 30 mismatches:")
for m in mismatches[:30]:
    print(f"\n--- Index {m['index']} ({m['language']}) ---")
    print(f"GOLD: {m['gold']}")
    print(f"PRED: {m['pred']}")
    print(f"ORIG_PRED: {m['orig_pred']}")
    # Diff words
    gold_words = m['gold'].split()
    pred_words = m['pred'].split()
    diffs = []
    # Print simple word align differences if possible
    max_len = max(len(gold_words), len(pred_words))
    for i in range(max_len):
        gw = gold_words[i] if i < len(gold_words) else "<EOF>"
        pw = pred_words[i] if i < len(pred_words) else "<EOF>"
        if gw != pw:
            diffs.append(f"{gw} vs {pw}")
    print(f"DIFFS: {', '.join(diffs[:10])}")
