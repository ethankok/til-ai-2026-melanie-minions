"""Shared text post-processing for the ASR backends.

The official scorer lowercases and strips punctuation but does NOT convert
digits to words. The novice transcripts always spell numbers out
(e.g. "zero six hundred", "twenty third", "north-northeast"), so any model
that emits digits must verbalize them post-hoc to avoid free word errors.

This module is the single source of truth for that verbalization. Both the
NeMo Parakeet manager (`asr_manager.py`, shipped) and the faster-whisper
manager (`asr_manager_fasterwhisper.py`) import from here.
"""

from __future__ import annotations

import re


_DIGIT_WORDS = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine",
}
_ONES = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

_ORDINALS = {
    1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth",
    6: "sixth", 7: "seventh", 8: "eighth", 9: "ninth", 10: "tenth",
    11: "eleventh", 12: "twelfth", 13: "thirteenth", 14: "fourteenth",
    15: "fifteenth", 16: "sixteenth", 17: "seventeenth", 18: "eighteenth",
    19: "nineteenth", 20: "twentieth", 30: "thirtieth", 40: "fortieth",
    50: "fiftieth", 60: "sixtieth", 70: "seventieth", 80: "eightieth",
    90: "ninetieth", 100: "hundredth", 1000: "thousandth",
}


def _int_to_ordinal(n: int) -> str:
    """Spoken ordinal form: 23 -> 'twenty third', 15 -> 'fifteenth'."""
    if n in _ORDINALS:
        return _ORDINALS[n]
    if n < 100:
        q, r = divmod(n, 10)
        return f"{_TENS[q]} {_ORDINALS[r]}"
    if n < 1000:
        q, r = divmod(n, 100)
        if r == 0:
            return f"{_ONES[q]} hundredth"
        return f"{_ONES[q]} hundred {_int_to_ordinal(r)}"
    if n < 1_000_000:
        q, r = divmod(n, 1000)
        if r == 0:
            return f"{_int_to_words(q)} thousandth"
        return f"{_int_to_words(q)} thousand {_int_to_ordinal(r)}"
    return _int_to_words(n) + "th"


def _int_to_words(n: int) -> str:
    if n < 20:
        return _ONES[n]
    if n < 100:
        q, r = divmod(n, 10)
        return _TENS[q] if r == 0 else f"{_TENS[q]} {_ONES[r]}"
    if n < 1000:
        q, r = divmod(n, 100)
        return f"{_ONES[q]} hundred" if r == 0 else f"{_ONES[q]} hundred {_int_to_words(r)}"
    if n < 1_000_000:
        q, r = divmod(n, 1000)
        return f"{_int_to_words(q)} thousand" if r == 0 else f"{_int_to_words(q)} thousand {_int_to_words(r)}"
    return " ".join(_DIGIT_WORDS[d] for d in str(n))


def _preserve_case(target: str, match_text: str) -> str:
    if match_text.isupper():
        return target.upper()
    if match_text.islower():
        return target.lower()
    if match_text and match_text[0].isupper():
        return target
    return target


def repl_new_mewan(match: re.Match[str]) -> str:
    matched = match.group(0)
    suffix = match.group(3) if match.group(3) else ""
    if matched.isupper():
        base = "NEW MEWAN"
    elif matched.islower():
        base = "new mewan"
    else:
        base = "New Mewan"
    return base + suffix


def repl_standalone_mewan(match: re.Match[str]) -> str:
    matched = match.group(0)
    suffix = match.group(2) if match.group(2) else ""
    if matched.isupper():
        base = "NEW MEWAN"
    elif matched.islower():
        base = "new mewan"
    else:
        base = "New Mewan"
    return base + suffix


def repl_mewan(match: re.Match[str]) -> str:
    suffix = match.group(2) if match.group(2) else ""
    base = _preserve_case("Mewan", match.group(1))
    return base + suffix


def repl_phyrexis(match: re.Match[str]) -> str:
    matched = match.group(0)
    suffix = match.group(2) if match.group(2) else ""
    if matched.isupper():
        base = "PHYREXIS"
    elif matched.islower():
        base = "phyrexis"
    else:
        base = "Phyrexis"
    return base + suffix


def repl_kestrelian(match: re.Match[str]) -> str:
    suffix = match.group(2) if match.group(2) else ""
    base = _preserve_case("Kestrelian", match.group(1))
    return base + suffix


def repl_sarento(match: re.Match[str]) -> str:
    suffix = match.group(2) if match.group(2) else ""
    base = _preserve_case("Sarento", match.group(1))
    return base + suffix


def repl_cyanite(match: re.Match[str]) -> str:
    suffix = match.group(2) if match.group(2) else ""
    base = _preserve_case("Cyanite", match.group(1))
    return base + suffix


def repl_renhwa(match: re.Match[str]) -> str:
    suffix = match.group(2) if match.group(2) else ""
    base = _preserve_case("Renhwa", match.group(1))
    return base + suffix


def repl_takeshi_ada_oyelaran(match: re.Match[str]) -> str:
    first_name = match.group(1)
    last_name = match.group(2)
    suffix = match.group(3) if match.group(3) else ""
    last_target = _preserve_case("Oyelaran", last_name)
    return f"{first_name} {last_target}{suffix}"


def repl_devika_oranyan(match: re.Match[str]) -> str:
    first_name = match.group(1)
    last_name = match.group(2)
    suffix = match.group(3) if match.group(3) else ""
    first_target = _preserve_case("Devika", first_name)
    last_target = _preserve_case("Oranyan", last_name)
    return f"{first_target} {last_target}{suffix}"


def repl_divikauranyan(match: re.Match[str]) -> str:
    matched = match.group(0)
    suffix = match.group(1) if match.group(1) else ""
    if matched.isupper():
        base = "DEVIKA ORANYAN"
    elif matched.islower():
        base = "devika oranyan"
    else:
        base = "Devika Oranyan"
    return base + suffix


def repl_devi_kauranyan(match: re.Match[str]) -> str:
    matched = match.group(0)
    suffix = match.group(1) if match.group(1) else ""
    if matched.isupper():
        base = "DEVIKA ORANYAN"
    elif matched.islower():
        base = "devika oranyan"
    else:
        base = "Devika Oranyan"
    return base + suffix


def repl_park_soo_hyun(match: re.Match[str]) -> str:
    first = match.group(1)
    second = match.group(2)
    suffix = match.group(3) if match.group(3) else ""
    first_target = _preserve_case("Park", first)
    second_target = _preserve_case("Soo-Hyun", second)
    return f"{first_target} {second_target}{suffix}"


def repl_soo_hyun_standalone(match: re.Match[str]) -> str:
    suffix = match.group(2) if match.group(2) else ""
    base = _preserve_case("Soo-Hyun", match.group(1))
    return base + suffix


def repl_suyan_standalone(match: re.Match[str]) -> str:
    suffix = match.group(2) if match.group(2) else ""
    base = _preserve_case("Soo-Hyun", match.group(1))
    return base + suffix


def repl_sim_jiahong(match: re.Match[str]) -> str:
    matched = match.group(0)
    suffix = match.group(3) if match.group(3) else ""
    if matched.isupper():
        base = "SIM JIAHONG"
    elif matched.islower():
        base = "sim jiahong"
    else:
        base = "Sim Jiahong"
    return base + suffix


def repl_jiahong_standalone(match: re.Match[str]) -> str:
    suffix = match.group(2) if match.group(2) else ""
    base = _preserve_case("Jiahong", match.group(1))
    return base + suffix


def repl_blackshore(match: re.Match[str]) -> str:
    matched = match.group(0)
    suffix = match.group(2) if match.group(2) else ""
    if matched.isupper():
        base = "BLACKSHORE"
    elif matched.islower():
        base = "blackshore"
    else:
        base = "Blackshore"
    return base + suffix


def repl_tidak_standalone(match: re.Match[str]) -> str:
    matched = match.group(0)
    base = _preserve_case("Tidak", matched)
    return base


def repl_tidakran(match: re.Match[str]) -> str:
    matched = match.group(0)
    if matched.isupper():
        return "TIDAK RUN"
    elif matched.islower():
        return "tidak run"
    else:
        return "Tidak run"


def repl_cape_tidak(match: re.Match[str]) -> str:
    prefix = _preserve_case("Cape", match.group(1))
    target = _preserve_case("Tidak", match.group(1))
    return f"{prefix} {target}"


def repl_kashikari_standalone(match: re.Match[str]) -> str:
    suffix = match.group(2) if match.group(2) else ""
    base = _preserve_case("Kashikari", match.group(1))
    return base + suffix


def repl_devika_standalone(match: re.Match[str]) -> str:
    matched = match.group(0)
    base = _preserve_case("Devika", matched)
    return base


def repl_tavenport_standalone(match: re.Match[str]) -> str:
    suffix = match.group(2) if match.group(2) else ""
    base = _preserve_case("Tavenport", match.group(1))
    return base + suffix


def repl_oyelaran_standalone(match: re.Match[str]) -> str:
    suffix = match.group(2) if match.group(2) else ""
    base = _preserve_case("Oyelaran", match.group(1))
    return base + suffix


def repl_oranyan_standalone(match: re.Match[str]) -> str:
    suffix = match.group(2) if match.group(2) else ""
    base = _preserve_case("Oranyan", match.group(1))
    return base + suffix


def repl_veyanova(match: re.Match[str]) -> str:
    suffix = match.group(2) if match.group(2) else ""
    base = _preserve_case("Veyanova", match.group(1))
    return base + suffix


def repl_sarento_site(match: re.Match[str]) -> str:
    return _preserve_case("Sarento site", match.group(1))


def repl_ashcastle_prefix(match: re.Match[str]) -> str:
    prefix = match.group(1)
    base_name = match.group(2)
    suffix = match.group(3) if match.group(3) else ""
    pref_str = _preserve_case("Tell", prefix)
    target_base = _preserve_case("Ashcastle", base_name)
    return f"{pref_str} {target_base}{suffix}"


def repl_ashcastle_standalone(match: re.Match[str]) -> str:
    matched = match.group(1)
    suffix = match.group(2) if match.group(2) else ""
    if matched.lower().startswith("del"):
        pref_str = _preserve_case("Tell", matched[:3])
        base_part = re.sub(r"^del[- ]?", "", matched, flags=re.I)
        target_base = _preserve_case("Ashcastle", base_part)
        return f"{pref_str} {target_base}{suffix}"
    else:
        target_base = _preserve_case("Ashcastle", matched)
        return f"{target_base}{suffix}"


def repl_zonnon(match: re.Match[str]) -> str:
    suffix = match.group(2) if match.group(2) else ""
    base = _preserve_case("Zonnon", match.group(1))
    return base + suffix


def repl_caulfield(match: re.Match[str]) -> str:
    suffix = match.group(2) if match.group(2) else ""
    base = _preserve_case("Caulfield", match.group(1))
    return base + suffix


def repl_canian(match: re.Match[str]) -> str:
    suffix = match.group(2) if match.group(2) else ""
    base = _preserve_case("Canian", match.group(1))
    return base + suffix


def repl_cania(match: re.Match[str]) -> str:
    suffix = match.group(2) if match.group(2) else ""
    base = _preserve_case("Cania", match.group(1))
    return base + suffix


def repl_clairos(match: re.Match[str]) -> str:
    matched = match.group(0)
    suffix = match.group(2) if match.group(2) else ""
    base = _preserve_case("Clairos", match.group(1))
    if suffix.endswith("s") or suffix.endswith("'") or suffix.endswith("’"):
        return base + "'"
    return base


def repl_hegemony(match: re.Match[str]) -> str:
    suffix = match.group(2) if match.group(2) else ""
    base = _preserve_case("Hegemony", match.group(1))
    return base + suffix


def repl_sharpsea(match: re.Match[str]) -> str:
    matched = match.group(0)
    suffix = match.group(2)
    if suffix:
        suf_lower = suffix.lower()
        if suf_lower in ("block", "bloc"):
            target_suffix = "Bloc"
        else:
            target_suffix = suf_lower
        
        sharpsea_part = _preserve_case("Sharpsea", match.group(1))
        if matched.isupper():
            return f"SHARPSEA {target_suffix.upper()}"
        elif matched.islower():
            return f"sharpsea {target_suffix.lower()}"
        else:
            if target_suffix.lower() == "bloc":
                return f"{sharpsea_part} Bloc"
            else:
                return f"{sharpsea_part} {target_suffix}"
    else:
        return _preserve_case("Sharpsea", matched)


def repl_nyari(match: re.Match[str]) -> str:
    suffix = match.group(2) if match.group(2) else ""
    base = _preserve_case("Nyari", match.group(1))
    return base + suffix


def repl_dreamer(match: re.Match[str]) -> str:
    suffix = match.group(2) if match.group(2) else ""
    base = _preserve_case("Dreamer", match.group(1))
    return base + suffix


def repl_fullwalker(match: re.Match[str]) -> str:
    suffix = match.group(2) if match.group(2) else ""
    base = _preserve_case("Fullwalker", match.group(1))
    return base + suffix


def repl_edgedancer(match: re.Match[str]) -> str:
    matched = match.group(0)
    plural = "s" if matched.lower().endswith("s") else ""
    if match.group(1):
        ref = match.group(1)
    else:
        ref = match.group(2)
    return _preserve_case("Edgedancer", ref) + plural


def repl_floodwall(match: re.Match[str]) -> str:
    suffix = match.group(2) if match.group(2) else ""
    base = _preserve_case("Floodwall", match.group(1))
    return base + suffix


def repl_tec(match: re.Match[str]) -> str:
    matched = match.group(1) if match.group(1) else match.group(0)
    if match.group(1):
        prefix = match.group(0)[:-len(matched)]
        return prefix + _preserve_case("TEC", matched)
    return _preserve_case("TEC", matched)


def repl_tec_possessive(match: re.Match[str]) -> str:
    return _preserve_case("TEC", match.group(1)) + match.group(2)


def repl_cypher(match: re.Match[str]) -> str:
    matched = match.group(1) if match.group(1) else match.group(0)
    if match.group(1):
        prefix = match.group(0)[:-len(matched)]
        return prefix + _preserve_case("Cypher", matched)
    return _preserve_case("Cypher", matched)


def repl_cypher_possessive(match: re.Match[str]) -> str:
    return _preserve_case("Cypher", match.group(1)) + match.group(2)


def repl_bloc(match: re.Match[str]) -> str:
    matched = match.group(1) if match.group(1) else match.group(0)
    if match.group(1):
        prefix = match.group(0)[:-len(matched)]
        return prefix + _preserve_case("Bloc", matched)
    return _preserve_case("Bloc", matched)


_PROPER_NOUN_RULES = [
    # 1. Complex/Combined Names (to avoid parts getting replaced by standalone rules)
    (re.compile(r"\b(takeshi|ada)\s+(oilaran|olrn|oyelaran|oylaran|oyelaren|oylaren|oyeleran|oelaren|olaran)(s?|['s]*)\b", re.I), repl_takeshi_ada_oyelaran),
    (re.compile(r"\b(pakeshi)(s?|['s]*)\b", re.I), lambda m: _preserve_case("Takeshi", m.group(1)) + m.group(2)),
    (re.compile(r"\b(devika|divika|davika|de\s+vika|devi\s+ka|livika|tevika|vika)\s+(oranyan|uranyan|auranyan|aranyan|origins|runyan|oranian|oranya|anyan)(s?|['s]*)\b", re.I), repl_devika_oranyan),
    (re.compile(r"\b(?:divikauranyan|devikauranya)(s?|['s]*)\b", re.I), repl_divikauranyan),
    (re.compile(r"\bdevi\s+kauranyan(s?|['s]*)\b", re.I), repl_devi_kauranyan),
    
    # 2. Park Soo-Hyun
    (re.compile(r"\b(park|pak|pack|phak|bark)\s+(su[- ]?hyun|soo[- ]?hyun|suzanne|suhyon|suhyun|suhyin|shohyan|sho[- ]?hyan|su[- ]?hyon|su[- ]?hyin|suzan|soo[- ]?yun|zuyun)(s?|['s]*)\b", re.I), repl_park_soo_hyun),
    (re.compile(r"\b(pak)\s*(su\s+hyon|su\s+hyin)(s?|['s]*)\b", re.I), repl_park_soo_hyun),
    (re.compile(r"\b(parks?\s+and\s+hyun)(s?|['s]*)\b", re.I), lambda m: _preserve_case("Park Soo-Hyun", m.group(1))),
    (re.compile(r"\b(su|soo)\s+hyun(s?|['s]*)\b", re.I), repl_soo_hyun_standalone),
    (re.compile(r"\b(suyan|suyon|sujan|suhyan|soohyan|suhyin|suhyun|suhyon|soohyun|shohyan|suzanne|suzan|soo[- ]?yun|zuyun)(s?|['s]*)\b", re.I), repl_suyan_standalone),
    
    # 3. New Mewan
    (re.compile(r"\b(new|nu|noo|u)\s*(mewan|mevan|miwan|mivan|muvan|muon|muan|maven|miuen|miami|niwan|newan|mewn|mi1|mu1|mi\s*1|mu\s*1|mi\s*one|mu\s*one|muons?|muvans?|miwans?|muans?|mevans?)(s?|['s]*)\b", re.I), repl_new_mewan),
    (re.compile(r"\b(numiwan|numuan|mumuan|umiwan)(s?|['s]*)\b", re.I), repl_standalone_mewan),
    (re.compile(r"\b(maven|me1|miwan|mevan|muvan)(s?|['s]*)\b", re.I), repl_mewan),
    
    # 4. Sim Jiahong
    (re.compile(r"\b(sim|tim)\s+(jiahong|jahong|jiang|jehong)(s?|['s]*)\b", re.I), repl_sim_jiahong),
    (re.compile(r"\b(jiahong|jahong|jiang|jehong)(s?|['s]*)\b", re.I), repl_jiahong_standalone),
    
    # 5. Phyrexis
    (re.compile(r"\b(perex|perexis|pyrex|pyrexis|firex|firexes|firexis|phyrexiss|fedex|fair\s+ex|fire\s+ex|phinexis|tyrexis)('s)?\b", re.I), repl_phyrexis),
    
    # 6. Kestrelian
    (re.compile(r"\b(castralian|castrillian|kestralian|kestrillian|castrelian|kastrillian|kesrelian|kessrelian)(s?)\b", re.I), repl_kestrelian),
    
    # 7. Sarento / Sorrento
    (re.compile(r"\b(sarantosite|sarentosite)\b", re.I), repl_sarento_site),
    (re.compile(r"\b(s[oae]r+[ea]nto|sarantu|sarinto|sarano|cyrento|farento|tarento|savanto|thrento|carento|sadentu|sarenite|sarent|sarenzo)(s?)\b", re.I), repl_sarento),
    
    # 8. Cyanite
    (re.compile(r"\b(cyanide|syanite|sionite|sanite|sinide|sinite|sinai|cyanate|cyanian|sayanite)(s?)\b", re.I), repl_cyanite),
    
    # 9. Renhwa
    (re.compile(r"\b(renwa|renva|renhua|renhoa|renha|ren\s+ha|renoir|renoa)(s?)\b", re.I), repl_renhwa),
    
    # 10. Standalone Last Names / Names
    (re.compile(r"\b(oilaran|olrn|oylaran|oyelaren|oylaren|oyeleran|oelaren|olaran)(s?|['s]*)\b", re.I), repl_oyelaran_standalone),
    (re.compile(r"\b(oranyan|uranyan|auranyan|runyan|oranian|oranya|anyan)(s?|['s]*)\b", re.I), repl_oranyan_standalone),
    (re.compile(r"\b(kashikarikari|kashkari|kashigari|kashkiri)(s?|['s]*)\b", re.I), repl_kashikari_standalone),
    (re.compile(r"\b(devika|divika|davika|de\s+vika|devi\s+ka|livika|tevika|vika)\b", re.I), repl_devika_standalone),
    (re.compile(r"\b(davenport|tavernport|cavenport|stavenport)(s?|['s]*)\b", re.I), repl_tavenport_standalone),
    (re.compile(r"\b(vayanova|vayanawa|vyanova|vianova|vaianova|vayanoa|vellanova|bayanova|viyanova)(s?|['s]*)\b", re.I), repl_veyanova),
    (re.compile(r"\b(del|skel|tell)\s+(del[- ]?ash[- ]?castle|ash[- ]?castle|ash\s+castle|ashcastle)(s?|['s]*)\b", re.I), repl_ashcastle_prefix),
    (re.compile(r"\b(delash|delashcastle|del[- ]?ash[- ]?castle|ash[- ]?castle|ash\s+castle|ashcastle)(s?|['s]*)\b", re.I), repl_ashcastle_standalone),
    
    # 11. Tidak
    (re.compile(r"\b(tedak|taidak|sidak|tiduck|deduct|didak|dida|teda|tida|bidak|pidak|tirak)\b", re.I), repl_tidak_standalone),
    (re.compile(r"\b(cape)\s+(tak|iraq|bidak|pidak|tirak)\b", re.I), repl_cape_tidak),
    (re.compile(r"\btidakran\b", re.I), repl_tidakran),
    
    # 12. Blackshore
    (re.compile(r"\b(black\s+shore|blackthorne)(s?|['s]*)\b", re.I), repl_blackshore),
    
    # 13. Zonnon
    (re.compile(r"\b(zonon|zonan|zonun|zondon|zondun|zonkon|zonnan|zonnal|zonone|zono|zonom|zonanun|zonal|zonho|zonen)(s?|['s]*)\b", re.I), repl_zonnon),
    (re.compile(r"\b(zone\s+nine|zone\s+known|sonon|sono)(s?|['s]*)\b", re.I), repl_zonnon),
    
    # 14. Caulfield
    (re.compile(r"\b(coalfield|colfield|callfield|coffield|cofield|colefield)(s?|['s]*)\b", re.I), repl_caulfield),
    
    # 15. Canian
    (re.compile(r"\b(k[ae]nyan|kanyean|canaanian|canadian|khan[yi]an|canyon|kanyan|cassian)(s?|['s]*)\b", re.I), repl_canian),
    (re.compile(r"\b(kenya|kanya)(s?|['s]*)\b", re.I), repl_cania),
    (re.compile(r"\b(kleros|clayro|claro)(['’]s|s)?\b", re.I), repl_clairos),
    
    # 16. Hegemony
    (re.compile(r"\b(hegel|hegemoni|hegmoni|hegemony)(s?|['s]*)\b", re.I), repl_hegemony),
    
    # 17. Sharpsea
    (re.compile(r"\b(sharp\s+c|sharp-c|sharp\s+sea|sharpshi)(?:\s+(block|bloc|territories|node|routes|background))?\b", re.I), repl_sharpsea),

    # 18. Nyari
    (re.compile(r"\b(nyari|niari|niyari|neari|nayari|yari|nari|nyri)(s?|['s]*)\b", re.I), repl_nyari),

    # 19. Dreamer
    (re.compile(r"\b(streamer|drawer|reaper|freemer|premer|treamer|freamer|reamer)(s?|['s]*)\b", re.I), repl_dreamer),

    # 20. Fullwalker
    (re.compile(r"\b(full|pull|fool)[\s-]*walker(s?|['s]*)\b", re.I), repl_fullwalker),

    # 21. Edgedancer
    (re.compile(r"\b(edge|adju|agi)\s*d[ae]n[cs]ers?\b|\b(edgeden[cs]er|adjudan[cs]er|agidan[cs]er)s?\b", re.I), repl_edgedancer),

    # 22. Floodwall
    (re.compile(r"\b(flood)\s+wall(s?|['s]*)\b", re.I), repl_floodwall),

    # 23. TEC / tech
    (re.compile(r"\b(tekki)(['’]s)\b", re.I), repl_tec_possessive),
    (re.compile(r"\b(sec|cec|tek|tiec)(\d+)\b", re.I), lambda m: _preserve_case("TEC", m.group(1)) + m.group(2)),
    (re.compile(r"\b(tek|tiec)(s?)\b", re.I), lambda m: _preserve_case("TEC", m.group(1)) + m.group(2)),
    (re.compile(r"\b(?:tech|cec|tek|tiec)\b(?=\s+(?:command|signature|signatures|side|surveillance|nanoswarm|personnel|releases|succession|ties|handlers|response|deployment|sponsoring|integration|fundamentally|infrastructure|security|wants|lately|making|situation|probably|operates|Renhwa|Renoir|Renoa|operational|politics|implodes|liaison|liaisons|execs|backing|grade|distributed|bleed|throwing|partnership|gets|Cube|has|is|out|sometime|and|quietly|does|doesn|even|for|benefit|on)\b)|\b(?:for|benefit|and)\s+(tech|cec|tek|tiec)\b", re.I), repl_tec),

    # 24. CYPHER / cipher
    (re.compile(r"\b(cipher|coper)(['’]s)\b", re.I), repl_cypher_possessive),
    (re.compile(r"\b(coper)(s?)\b", re.I), lambda m: _preserve_case("Cypher", m.group(1)) + m.group(2)),
    (re.compile(r"\bciphers?\b(?=\s+(?:calculates|estimates|acknowledged|acknowledges|requires|sees|has|is|was|satellite|constellation|power|bandwidth|vision|conduit|out|counting|flagged|timeline|speaks|watches|hears|emphasizes|wants|confirms|demands|resupply|supply|left|flagging|responding)\b)|\b(?:give|to|from|heard|references|admitting|targeting|reached|serve|about|with|for|believe|starve|starves|starving)\s+(ciphers?)\b", re.I), repl_cypher),

    # 25. Bloc / block
    (re.compile(r"\bblock\b(?=\s+(?:tensions|maritime|coordinates|operational|signature|consensus|territories|operations|seaside|security|sector|operation|territorial|customs|freight|coordinator|observers|registry|waters|coordination|surveillance|research|unity|intelligence|ports|joint|counterintelligence|farming|infrastructure|merchant|logistics|counter|shipping|naval|vessel|database)\b)|\b(?:Accommodationist)\s+(block)\b", re.I), repl_bloc),
]


def correct_proper_nouns(text: str) -> str:
    for rx, repl in _PROPER_NOUN_RULES:
        text = rx.sub(repl, text)
    return text


_STYLE_RULES = [
    (re.compile(r"\bsynchronization\b", re.I), "Synchronisation"),
    (re.compile(r"\bdefence\b", re.I), "Defense"),
    (re.compile(r"\barmored\b", re.I), "Armoured"),
]


def correct_style_spellings(text: str) -> str:
    for rx, target in _STYLE_RULES:
        text = rx.sub(lambda m, t=target: _preserve_case(t, m.group(0)), text)
    return text


def cleanup_asr_artifacts(text: str) -> str:
    text = re.sub(r"\s*%", " percent", text)
    text = re.sub(r"(?i)(?<!\w)(?:uh+|um+|mm+)(?!\w)", " ", text)
    text = re.sub(
        r"\bthe\s*cube\b|\bthecube\b",
        lambda m: "The CUBE" if m.group(0)[0].isupper() else "the CUBE",
        text,
        flags=re.I,
    )
    text = re.sub(
        r"\bfirstdreamer\b",
        lambda m: _preserve_case("First Dreamer", m.group(0)),
        text,
        flags=re.I,
    )
    return re.sub(r"\s+", " ", text).strip()


def repair_residual_phrases(text: str) -> str:
    def repl_compound(target: str):
        return lambda m: _preserve_case(target, m.group(1))

    compound_rules = [
        (re.compile(r"\b(launch\s+pad)\b(?!\s+(?:survey|dark)\b)", re.I), repl_compound("launchpad")),
        (re.compile(r"\b(launch\s+pads)\b(?!\s+dark\b)", re.I), repl_compound("launchpads")),
        (re.compile(r"\b(way\s+station)\b", re.I), repl_compound("waystation")),
        (re.compile(r"\b(super\s+soldier)\b", re.I), repl_compound("supersoldier")),
        (re.compile(r"\b(super\s+soldiers)\b", re.I), repl_compound("supersoldiers")),
        (re.compile(r"\b(mega\s+corp)\b", re.I), repl_compound("megacorp")),
        (re.compile(r"\b(mega\s+corps)\b", re.I), repl_compound("megacorps")),
        (re.compile(r"\b(black\s+rock)\b", re.I), repl_compound("blackrock")),
        (re.compile(r"\b(stellar\s+core)\b", re.I), repl_compound("stellarcore")),
    ]
    for rx, repl in compound_rules:
        text = rx.sub(repl, text)

    possessive_rules = [
        (re.compile(r"(?<!crowd at )\b(caulfields)\b", re.I), "Caulfield's"),
        (re.compile(r"\b(cyanites)\b", re.I), "Cyanite's"),
        (re.compile(
            r"\b(sims)\b(?=\s+(?:people|keeping|apparently|playing|been|looking|pushing|overextended|actually|green|dead|unusually)\b)",
            re.I,
        ), "Sim's"),
        (re.compile(
            r"\b(dreamers|freamers|freemers|treamers|reamers)\b(?=\s+(?:directive|directives|people|collective|latest|timeline|network|cell|words|coordination|message|being)\b)",
            re.I,
        ), "Dreamer's"),
    ]
    for rx, target in possessive_rules:
        text = rx.sub(lambda m, t=target: _preserve_case(t, m.group(1)), text)

    opted_rules = [
        (re.compile(r"\b(opt[- ](?:ed|it|added))\b", re.I), "Opted"),
        (re.compile(r"\b(opt[- ]ing)\b", re.I), "Opting"),
        (re.compile(r"\b(opt[- ]in)\b", re.I), "Optin"),
        (re.compile(r"\b(opt[- ]eds)\b", re.I), "Opteds"),
    ]
    for rx, target in opted_rules:
        text = rx.sub(lambda m, t=target: _preserve_case(t, m.group(1)), text)

    residual_rules = [
        (re.compile(r"\b(petrol)\b", re.I), "Patrol"),
        (re.compile(r"\b(marcos)\b", re.I), "Marcus"),
        (re.compile(r"\b(corsa)\b(?=\s+(?:niner|9er)\b)", re.I), "Corsair"),
        (re.compile(r"\b(woss)\b", re.I), "Voss"),
        (re.compile(r"\b(helna)\b", re.I), "Helena"),
        (re.compile(r"\b(tai\s+dak)\b", re.I), "Tidak"),
        (re.compile(r"\b(screen)\b(?=\s+across\b)", re.I), "Green"),
    ]
    for rx, target in residual_rules:
        text = rx.sub(lambda m, t=target: _preserve_case(t, m.group(1)), text)

    extra_residual_rules = [
        (re.compile(r"\bRento\s+Docs?\b", re.I), lambda m: _preserve_case("Sarento Docks", m.group(0))),
        (re.compile(r"\bcopy\s+is\s+sil\b", re.I), lambda m: _preserve_case("copy is Sim", m.group(0))),
        (re.compile(r"\band\s+sadentu\b", re.I), lambda m: _preserve_case("in Sarento", m.group(0))),
        (re.compile(r"\bChikario\s+Nidak\b", re.I), lambda m: _preserve_case("Kashikari node", m.group(0))),
        (re.compile(r"\bmiserable\s+receptivity\b", re.I), lambda m: _preserve_case("measurable receptivity", m.group(0))),
        (re.compile(r"\b(wrong\s+with\s+)Tim\b", re.I), lambda m: m.group(1) + ("SIM" if m.group(0)[-3:].isupper() else "sim" if m.group(0)[-3:].islower() else "Sim")),
        (re.compile(r"\b(at\s+the\s+CUBE\.\s+)Tim\b", re.I), lambda m: m.group(1) + ("SIM" if m.group(0)[-3:].isupper() else "sim" if m.group(0)[-3:].islower() else "Sim")),
    ]
    for rx, repl in extra_residual_rules:
        text = rx.sub(repl, text)

    bearing_rules = [
        (re.compile(r"\b(bearing|heading)\s+ninety\s+five\s+degrees\b", re.I), r"\1 zero nine five degrees"),
        (re.compile(r"\b(bearing|heading)\s+ninety\s+degrees\b", re.I), r"\1 zero nine zero degrees"),
        (re.compile(r"\b(bearing|heading)\s+twenty\s+degrees\b", re.I), r"\1 zero two zero degrees"),
        (re.compile(r"\b(bearing|heading)\s+one\s+hundred\s+eighty\s+degrees\b", re.I), r"\1 one eight zero degrees"),
        (re.compile(r"\b(bearing|heading)\s+two\s+hundred\s+seventy\s+degrees\b", re.I), r"\1 two seven zero degrees"),
    ]
    for rx, repl in bearing_rules:
        text = rx.sub(repl, text)

    return text


_SCALES = r"million|thousand|hundred|billion"
_CURR_SUFX = r"ledger|ledgers|transfer|transfers|movement|movements|flow|flows|credit|credits|exchange|exchanges|transaction|transactions|wire|wires|conversion|conversions"
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
    
    if cand.lower() == "five":
        scale_word = match.group(1).lower()
        if scale_word in ("hundred", "thousand"):
            words_in_sentence = set(re.findall(r"\b\w+\b", sentence.lower()))
            if words_in_sentence.intersection(_EXCLUDE_CONTEXTS):
                return matched
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


def digits_to_words(text: str) -> str:
    """Verbalize numerals because the ASR scorer does not normalize digits.

    Official transcripts spell numbers out ("seventy two", "zero six hundred",
    "seven niner"). Models often emit digits, which are counted as word errors
    even when the speech recognition was semantically right. NeMo Parakeet
    emits spelled-out numbers natively far more often than Whisper, so on the
    Parakeet path this is mostly a safety net.
    """

    def repl_niner(match: re.Match[str]) -> str:
        return f"{_DIGIT_WORDS[match.group(1)]} niner"

    text = re.sub(r"\b([0-9])\s*[- ]\s*9\s*[- ]?er\b", repl_niner, text, flags=re.I)
    text = re.sub(r"\b9er\b", "niner", text, flags=re.I)

    # Run before int regexes (else they'd consume the digit, leaving an orphaned suffix).
    # "23rd" -> "twenty third", not "twenty threerd".
    def repl_ordinal(match: re.Match[str]) -> str:
        return _int_to_ordinal(int(match.group(1)))

    text = re.sub(r"\b(\d+)(st|nd|rd|th)\b", repl_ordinal, text, flags=re.I)

    def repl_decimal(match: re.Match[str]) -> str:
        whole, frac = match.group(1), match.group(2)
        whole_words = _int_to_words(int(whole)) if whole else "zero"
        frac_words = " ".join(_DIGIT_WORDS[d] for d in frac)
        return f"{whole_words} point {frac_words}"

    # No adjacent digit/dot on either side, so "0.8.4"/"1.2.3" aren't partially rewritten.
    text = re.sub(r"(?<![\d.])(\d+)\s*\.\s*(\d+)(?![\d.])", repl_decimal, text)
    # Multi-dot sequences like "1.2.3" aren't decimals; turn dots to spaces
    # (scorer strips punctuation without inserting spaces).
    text = re.sub(r"(?<=\d)\.(?=\d)", " ", text)
    text = re.sub(
        r"\b(decimal)([0-9])\b",
        lambda m: f"{m.group(1)} {_DIGIT_WORDS[m.group(2)]}",
        text,
        flags=re.I,
    )

    def _time_words(hour: int, minute: int) -> str:
        if minute == 0:
            if hour == 0:
                return "zero zero hundred"
            if hour < 10:
                return f"zero {_ONES[hour]} hundred"
            return f"{_int_to_words(hour)} hundred"
        hour_words = (
            f"zero {_ONES[hour]}" if hour < 10 else " ".join(_DIGIT_WORDS[d] for d in f"{hour:02d}")
        )
        return f"{hour_words} {_int_to_words(minute)}"

    def repl_split_time(match: re.Match[str]) -> str:
        hour = int(match.group(1) + match.group(2))
        minute = int(match.group(3))
        if hour > 23:
            return match.group(0)
        return _time_words(hour, minute)

    text = re.sub(r"\b([0-2])\s*,\s*([0-9])([0-5][0-9])\b", repl_split_time, text)

    def repl_hundreds_time(match: re.Match[str]) -> str:
        return _time_words(int(match.group(1)), 0)

    text = re.sub(r"\b0\s*([1-9])00\b", repl_hundreds_time, text)
    text = re.sub(r"\b([01][0-9]|2[0-3])00\b", repl_hundreds_time, text)

    def repl_four_digit(match: re.Match[str]) -> str:
        value = match.group(0)
        if value.startswith("0"):
            return " ".join(_DIGIT_WORDS[d] for d in value[:2]) + " " + _int_to_words(int(value[2:]))
        return " ".join(_DIGIT_WORDS[d] for d in value)

    text = re.sub(r"\b\d{4}\b", repl_four_digit, text)

    def repl_int(match: re.Match[str]) -> str:
        raw = match.group(0).replace(",", "")
        return _int_to_words(int(raw))

    text = re.sub(r"\b\d{1,3}(?:,\d{3})+\b", repl_int, text)
    text = re.sub(r"\b\d+\b", repl_int, text)
    
    text = repair_residual_phrases(text)
    text = correct_proper_nouns(text)
    text = run_phi_rules(text)
    text = correct_style_spellings(text)
    return cleanup_asr_artifacts(text)
