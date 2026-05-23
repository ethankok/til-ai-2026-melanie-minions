import json
import re

# Simple preserve case helper
def _preserve_case(target: str, match_text: str) -> str:
    if match_text.isupper():
        return target.upper()
    if match_text.islower():
        return target.lower()
    if match_text and match_text[0].isupper():
        return target
    return target

def repl_zonnon(match: re.Match[str]) -> str:
    suffix = match.group(2) if match.group(2) else ""
    base = _preserve_case("Zonnon", match.group(1))
    return base + suffix

def repl_caulfield(match: re.Match[str]) -> str:
    suffix = match.group(2) if match.group(2) else ""
    base = _preserve_case("Caulfield", match.group(1))
    return base + suffix

# Regexes
rx_zonnon = re.compile(r"\b(zonon|zonan|zonun|zondon|zondun|zonkon|zonnan|zonnal|zonone|zono|zonom|zonanun|zonal)(s?|['s]*)\b", re.I)
rx_caulfield = re.compile(r"\b(coalfield|colfield|callfield|coffield|cofield|colefield)(s?|['s]*)\b", re.I)

with open("asr_results.json", "r", encoding="utf-8") as f:
    preds = json.load(f)

changed_zonnon = 0
changed_caulfield = 0

for idx, p in enumerate(preds):
    p_new = rx_zonnon.sub(repl_zonnon, p)
    p_new = rx_caulfield.sub(repl_caulfield, p_new)
    if p_new != p:
        # Find which rule matched
        matched = []
        if rx_zonnon.search(p):
            matched.append("Zonnon")
            changed_zonnon += 1
        if rx_caulfield.search(p):
            matched.append("Caulfield")
            changed_caulfield += 1
        print(f"[{idx}] {matched}:")
        print(f"  ORIG: {p}")
        print(f"  NEW : {p_new}")
        print()

print(f"Total changed Zonnon: {changed_zonnon}")
print(f"Total changed Caulfield: {changed_caulfield}")
