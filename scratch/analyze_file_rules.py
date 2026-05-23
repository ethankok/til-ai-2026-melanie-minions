import json
import re

gold = []
with open("asr.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        gold.append(json.loads(line))

with open("asr_results.json", "r", encoding="utf-8") as f:
    preds = json.load(f)

# Rules 3-8 regexes
rules = [
    ("Rule 3: currency suffixes", rf"\b(file|files)\s+(ledger|ledgers|transfer|transfers|movement|movements|flow|flows|credit|credits|exchange|exchanges|transaction|transactions|wire|wires|conversion|conversions)\b"),
    ("Rule 4: got/have/had to drop", rf"\b(got|have|had)\s+(file|files)\s+to\s+drop\b"),
    ("Rule 5: sold/sell/selling for", rf"\b(sold|sell|selling)\s+(?:[a-z0-9'-]+\s+){{0,2}}for\s+(file|files)\b"),
    ("Rule 6: bleeding/funneling/saving up", rf"\b(bleeding|funneling|funneled|funnels?|saving\s+up)\s+(file|files)\b"),
    ("Rule 7: throwing around", rf"\b(throwing|throw|threw)\s+(file|files)\s+around\b"),
    ("Rule 8: currency contexts", rf"\b(file|files)\s+(for|minimum|on\s+a|at\s+blackjack|saved|short|in\s+bribes)\b")
]

for name, rx_str in rules:
    rx = re.compile(rx_str, re.I)
    matches_found = 0
    for idx, (g, p) in enumerate(zip(gold, preds)):
        m = rx.search(p)
        if m:
            is_phi = "phi" in g["transcript"].lower()
            print(f"[{idx}] {name} matched '{m.group(0)}' - {'CORRECT' if is_phi else 'INCORRECT'}")
            print(f"  GOLD: {g['transcript']}")
            matches_found += 1
    if matches_found == 0:
        print(f"{name}: No matches found")
