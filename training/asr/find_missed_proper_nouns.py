import json
import re

# Load the current predictions
preds_path = "/Users/ethankok/projects/TIL/asr_results.json"
try:
    with open(preds_path, "r", encoding="utf-8") as f:
        preds = json.load(f)
except Exception as e:
    print(f"Error loading {preds_path}: {e}")
    exit(1)

print(f"Loaded {len(preds)} predictions.")

# We want to search for potential misspellings or phonetic variations of our proper nouns
# that are not already corrected (or check if they exist in the processed predictions).
# We define regexes for the incorrect forms.

patterns = {
    # 1. Devika Oranyan
    "devika_misspellings": (r"\b(divika|davika|devica|de vika)\b", None),
    "oranyan_misspellings": (r"\b(uranyan|aranyan|auranyan|origins)\b", r"\b(devika|divika|davika|de vika)\b"), # check if it occurs near devika or standalone
    "oranyan_standalone_misspellings": (r"\b(uranyan|aranyan|auranyan)\b", None),
    
    # 2. Takeshi Oyelaran
    "takeshi_misspellings": (r"\b(takashi|takeshy)\b", None),
    "oyelaran_misspellings": (r"\b(oilaran|oylaran|olrn|oyelarn|oilaron)\b", None),
    
    # 3. Sarento
    "sarento_misspellings": (r"\b(sorrento|sarrento|serento)\b", None),
    
    # 4. Cyanite
    "cyanite_misspellings": (r"\b(cyanide|syanite|sanite|sinide)\b", None),
    
    # 5. Phyrexis
    "phyrexis_misspellings": (r"\b(pyrexis|pyrex|pyrex's|perex|perexis|firex|firexes|fedex)\b", None),
    
    # 6. New Mewan
    "new_mewan_misspellings": (r"\b(new\s+mee[- ]one|new\s+meeone|new\s+meeon|new\s+miwan|new\s+muvan|new\s+muan|new\s+muon|new\s+mewon)\b", None),
    "mewan_standalone_misspellings": (r"\b(miwan|muvan|numuan|muan|muon|mewon)\b", None),
    
    # 7. Kestrelian
    "kestrelian_misspellings": (r"\b(castilian|kestralian|kestrillian|castralian|castrillian|castrelian|kastrillian)\b", None),
    
    # 8. Tidak
    "tidak_misspellings": (r"\b(tedak|taidak|sidak|tiduck)\b", None),
    
    # 9. Kashikari
    "kashikari_misspellings": (r"\b(kashigari|kashikarikari|kashkari)\b", None),
    
    # 10. Sim Jiahong
    "sim_misspellings": (r"\b(sym)\b", None),
    "jiahong_misspellings": (r"\b(jahong|jiahung)\b", None),
    
    # 11. Park Soo-Hyun
    "park_misspellings": (r"\b(pak|pack)\s+(soo[- ]hyun|su[- ]hyun|suhyon|suhyun|soohyun)\b", None),
    "soo_hyun_misspellings": (r"\b(suhyon|suhyun|soohyun|soohyan|suzanne|shohyan|suyan|suyon|sujan)\b", None),
    
    # 12. Blackshore
    "blackshore_misspellings": (r"\bblack\s+shore\b", None),
    
    # 13. Tavenport
    "tavenport_misspellings": (r"\b(tavernport|davenport)\b", None),
    
    # 14. Delashcastle
    "delash_misspellings": (r"\b(del\s+ash\s+castle|delash|del[- ]ash[- ]castle|ashcastle|ash\s+castle)\b", None),
    
    # 15. Kleros
    "kleros_misspellings": (r"\b(clairos)\b", None),
    
    # 16. Vyanova
    "vyanova_misspellings": (r"\b(vayanova)\b", None),
}

results = {}

for name, (pat, context_pat) in patterns.items():
    rx = re.compile(pat, re.I)
    matches = []
    for idx, pred in enumerate(preds):
        for m in rx.finditer(pred):
            matched_str = m.group(0)
            # If context_pat is specified, verify if the context matches
            if context_pat:
                if not re.search(context_pat, pred, re.I):
                    continue
            # Get some context around the match
            start = max(0, m.start() - 40)
            end = min(len(pred), m.end() + 40)
            context = pred[start:end]
            matches.append((idx, matched_str, context.strip()))
    results[name] = matches

for name, matches in results.items():
    print(f"\n========================================\n{name}: {len(matches)} matches found\n========================================")
    for idx, matched_str, context in matches[:10]:
        print(f"[{idx}] Matched '{matched_str}' in context: ... {context} ...")
