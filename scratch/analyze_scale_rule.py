import json
import re

gold = []
with open("asr.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        gold.append(json.loads(line))

with open("asr_results.json", "r", encoding="utf-8") as f:
    preds = json.load(f)

_SCALES = r"million|thousand|hundred|billion"
_CANDIDATES = r"five|file|files|fi|pi|fee|fight|fire|pie|fai"

# Scale rule
scale_rx = re.compile(rf"\b({_SCALES})\s+({_CANDIDATES})(s?)\b(?!\s+(?:hundred|thousand|million|billion|ten|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|one|two|three|four|five|six|seven|eight|nine|point)\b)", re.I)

for idx, (g, p) in enumerate(zip(gold, preds)):
    matches = list(scale_rx.finditer(p))
    if matches:
        is_phi = "phi" in g["transcript"].lower()
        print(f"[{idx}] {'CORRECT' if is_phi else 'INCORRECT'}:")
        print(f"  ORIG: {p}")
        for m in matches:
            print(f"    Match: '{m.group(0)}'")
        print(f"  GOLD: {g['transcript']}")
        print()
