import json

# Load predictions
with open("/home/jupyter/til/asr_results.json", "r", encoding="utf-8") as f:
    preds = json.load(f)

# Load gold transcripts
gold = []
with open("/home/jupyter/novice/asr/asr.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        gold.append(json.loads(line))

print(f"Loaded {len(preds)} predictions, {len(gold)} gold transcripts.")

indices = [53, 1623, 2270, 3365, 4016, 63, 40, 86, 112, 122, 60, 30, 49, 105, 110, 62, 1644, 2328, 2571, 2751, 626]

for idx in sorted(list(set(indices))):
    print(f"\n--- INDEX {idx} ---")
    print(f"GOLD KEY: {gold[idx]['key']} AUDIO: {gold[idx]['audio']}")
    print(f"GOLD: {gold[idx]['transcript']}")
    print(f"PRED: {preds[idx]}")
