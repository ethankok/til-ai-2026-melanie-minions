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

_CURRENCY_CONTEXTS = {
    "genesis", "clinic", "appointment", "consultation", "somatic", "enhancement",
    "reinforcement", "vial", "vials", "biodealer", "biodealers", "dealer", "dealers",
    "saved", "costs", "cost", "price", "prices", "rate", "rates", "rent", "bribes",
    "cleared", "credits", "account", "accounts", "payment", "payments", "tables",
    "roulette", "blackjack", "short", "bribe", "funds", "transfer", "transfers",
    "wire", "wires", "moving", "dropped", "spent", "spending", "lost", "won"
}

_EXCLUDE_CONTEXTS = {
    "hours", "bearing", "vector", "heading", "degrees", "channel", "outpost",
    "relay", "station", "installation", "grid", "coordinate", "coordinates",
    "latitude", "longitude", "wind", "winds", "knots", "altitude"
}

def repl_phi_refined(match: re.Match[str], sentence: str) -> str:
    matched = match.group(0)
    cand_match = re.search(rf"\b({_CANDIDATES})\b", matched, re.I)
    if not cand_match:
        return matched
    cand = cand_match.group(1)
    
    # If the candidate is "five" (or case-insensitive "five"), and the scale is hundred/thousand
    # we apply extra safety filters.
    if cand.lower() == "five":
        scale_word = match.group(1).lower()
        if scale_word in ("hundred", "thousand"):
            # Check if sentence has any exclude contexts
            words_in_sentence = set(re.findall(r"\b\w+\b", sentence.lower()))
            if words_in_sentence.intersection(_EXCLUDE_CONTEXTS):
                return matched
            # Check if sentence has any currency contexts
            if not words_in_sentence.intersection(_CURRENCY_CONTEXTS):
                return matched
                
    phi_str = _preserve_case("Phi", cand)
    replaced = re.sub(rf"\b{cand}\b", phi_str, matched)
    return replaced

def repl_phi_simple(match: re.Match[str]) -> str:
    matched = match.group(0)
    cand_match = re.search(rf"\b({_CANDIDATES})\b", matched, re.I)
    if not cand_match:
        return matched
    cand = cand_match.group(1)
    phi_str = _preserve_case("Phi", cand)
    replaced = re.sub(rf"\b{cand}\b", phi_str, matched)
    return replaced

def repl_phi_context(match: re.Match[str]) -> str:
    matched = match.group(0)
    cand_match = re.search(rf"\b({_CANDIDATES})\b", matched, re.I)
    if not cand_match:
        return matched
    cand = cand_match.group(1)
    following = match.group(2).lower()
    
    # Exclude five, file, files from matching "for"
    if cand.lower() in ("five", "file", "files") and following == "for":
        return matched
        
    phi_str = _preserve_case("Phi", cand)
    replaced = re.sub(rf"\b{cand}\b", phi_str, matched)
    return replaced

def run_phi_rules(text: str) -> str:
    # 1. Standalone pi/fi/fai
    text = re.sub(r"\b(pi|fi|fai)(s?)\b", lambda m: _preserve_case("Phi", m.group(1)) + m.group(2), text, flags=re.I)
    
    # 2. Preceded by scale words
    scale_rx = re.compile(rf"\b({_SCALES})\s+({_CANDIDATES})(s?)\b(?!\s+(?:hundred|thousand|million|billion|ten|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|one|two|three|four|five|six|seven|eight|nine|point)\b)", re.I)
    text = scale_rx.sub(lambda m: repl_phi_refined(m, text), text)
    
    # 3. Followed by currency suffixes
    text = re.sub(rf"\b({_CANDIDATES})\s+({_CURR_SUFX})\b", repl_phi_simple, text, flags=re.I)
    
    # 4. Context phrases: got/have/had ... to drop
    text = re.sub(rf"\b(got|have|had)\s+({_CANDIDATES})\s+to\s+drop\b", repl_phi_simple, text, flags=re.I)
    
    # 5. Context phrases: sold/sell/selling ... for ...
    text = re.sub(rf"\b(sold|sell|selling)\s+(?:[a-z0-9'-]+\s+){{0,2}}for\s+({_CANDIDATES})\b", repl_phi_simple, text, flags=re.I)
    
    # 6. Context phrases: bleeding/funneling/saving up ...
    text = re.sub(rf"\b(bleeding|funneling|funneled|funnels?|saving\s+up)\s+({_CANDIDATES})\b", repl_phi_simple, text, flags=re.I)
    
    # 7. Context phrases: throwing ... around
    text = re.sub(rf"\b(throwing|throw|threw)\s+({_CANDIDATES})\s+around\b", repl_phi_simple, text, flags=re.I)
    
    # 8. Followed by currency contexts (for, minimum, saved, short, in bribes, at blackjack)
    text = re.sub(rf"\b({_CANDIDATES})\s+(for|minimum|on\s+a|at\s+blackjack|saved|short|in\s+bribes)\b", repl_phi_context, text, flags=re.I)
    
    return text

# Load gold transcripts
gold = []
with open("asr.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        gold.append(json.loads(line))

# Load predictions
with open("asr_results.json", "r", encoding="utf-8") as f:
    preds = json.load(f)

changed = 0
correct_changes = 0
incorrect_changes = 0

for idx, (g, p) in enumerate(zip(gold, preds)):
    p_new = run_phi_rules(p)
    
    if p_new != p:
        changed += 1
        is_phi_in_gold = "phi" in g["transcript"].lower()
        if is_phi_in_gold:
            correct_changes += 1
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
