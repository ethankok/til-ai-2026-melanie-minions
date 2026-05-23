import json
import re

with open("/Users/ethankok/projects/TIL/asr_results.json", "r", encoding="utf-8") as f:
    preds = json.load(f)

queries = {
    "canyon": r"\bcanyon\b",
    "hegel": r"\bhegel\b",
    "sharp_sea": r"sharp\s*sea",
    "sarano": r"sarano",
    "one_network": r"one\s+network",
    "five_trans": r"five\s+transactions",
    "launch_pad": r"launch\s+pad",
    "percent": r"%",
    "mega_corps": r"mega\s+corps",
    "forty_six": r"forty\s+six"
}

for name, q in queries.items():
    rx = re.compile(q, re.I)
    matches = []
    for idx, p in enumerate(preds):
        if rx.search(p):
            matches.append((idx, p))
    print(f"{name}: {len(matches)} matches")
    for idx, p in matches[:5]:
        print(f"  [{idx}]: {p[:120]}...")
