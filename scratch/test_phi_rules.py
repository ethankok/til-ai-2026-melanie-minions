import json
import re
import sys
import os

sys.path.append(os.path.abspath("asr/src"))
from asr_postprocess import _preserve_case

# Scale words
_SCALES = r"million|thousand|hundred|billion"
# Currency indicators (following words)
_CURR_SUFX = r"ledger|ledgers|transfer|transfers|movement|movements|flow|flows|credit|credits|exchange|exchanges|transaction|transactions|wire|wires|conversion|conversions"
# Words to replace
_CANDIDATES = r"five|file|files|fi|pi|fee|fight|fire|pie|fai"

def repl_phi(match: re.Match[str]) -> str:
    # Match can have multiple groups. We want to find which group matched the candidate.
    # To keep it simple: we want to replace the candidate word with Phi, preserving its case.
    # Let's find the candidate word in the match.
    matched = match.group(0)
    # We find the candidate using case-insensitive search
    cand_match = re.search(rf"\b({_CANDIDATES})\b", matched, re.I)
    if not cand_match:
        return matched
    cand = cand_match.group(1)
    phi_str = _preserve_case("Phi", cand)
    # Replace the candidate word in the matched string
    # Using regex to replace only the word boundary candidate
    replaced = re.sub(rf"\b{cand}\b", phi_str, matched)
    return replaced

# Define all Phi rules
PHI_RULES = [
    # 1. Standalone pi/fi/fai
    (re.compile(r"\b(pi|fi|fai)(s?)\b", re.I), lambda m: _preserve_case("Phi", m.group(1)) + m.group(2)),
    
    # 2. Preceded by scale words
    (re.compile(rf"\b({_SCALES})\s+({_CANDIDATES})(s?)\b(?!\s+(?:hundred|thousand|million|billion|ten|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|one|two|three|four|five|six|seven|eight|nine|point)\b)", re.I), repl_phi),
    
    # 3. Followed by currency suffixes
    (re.compile(rf"\b({_CANDIDATES})\s+({_CURR_SUFX})\b", re.I), repl_phi),
    
    # 4. Context phrases: got/have/had ... to drop
    (re.compile(rf"\b(got|have|had)\s+({_CANDIDATES})\s+to\s+drop\b", re.I), repl_phi),
    
    # 5. Context phrases: sold/sell/selling ... for ...
    (re.compile(rf"\b(sold|sell|selling)\s+([a-z0-9'-]+\s+){0,2}for\s+({_CANDIDATES})\b", re.I), repl_phi),
    
    # 6. Context phrases: bleeding/funneling/saving up ...
    (re.compile(rf"\b(bleeding|funneling|funneled|funnels?|saving\s+up)\s+({_CANDIDATES})\b", re.I), repl_phi),
    
    # 7. Context phrases: throwing ... around
    (re.compile(rf"\b(throwing|throw|threw)\s+({_CANDIDATES})\s+around\b", re.I), repl_phi),
    
    # 8. Followed by currency contexts (for, minimum, saved, short, in bribes, at blackjack)
    (re.compile(rf"\b({_CANDIDATES})\s+(for|minimum|on\s+a|at\s+blackjack|saved|short|in\s+bribes)\b", re.I), repl_phi),
]

# Load gold transcripts
gold = []
with open("asr.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        gold.append(json.loads(line))

# Load predictions
with open("asr_results.json", "r", encoding="utf-8") as f:
    preds = json.load(f)

print("Testing Phi rules on predictions:")
changed = 0
correct_changes = 0
incorrect_changes = 0

for idx, (g, p) in enumerate(zip(gold, preds)):
    p_new = p
    for rx, repl in PHI_RULES:
        p_new = rx.sub(repl, p_new)
    
    if p_new != p:
        changed += 1
        # Check if the change brought us closer to gold
        # We check case-insensitive presence of Phi in gold at the same spot
        is_phi_in_gold = "phi" in g["transcript"].lower()
        if is_phi_in_gold:
            correct_changes += 1
            print(f"[{idx}] CORRECT:")
            print(f"  ORIG: {p}")
            print(f"  NEW : {p_new}")
            print(f"  GOLD: {g['transcript']}")
            print()
        else:
            incorrect_changes += 1
            print(f"[{idx}] INCORRECT:")
            print(f"  ORIG: {p}")
            print(f"  NEW : {p_new}")
            print(f"  GOLD: {g['transcript']}")
            print()

print(f"Total changes: {changed}")
print(f"Correct changes (gold has Phi): {correct_changes}")
print(f"Incorrect changes (gold doesn't have Phi): {incorrect_changes}")
