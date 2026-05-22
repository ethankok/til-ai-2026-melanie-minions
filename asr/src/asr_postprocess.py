"""Shared text post-processing for the ASR backends.

The official scorer lowercases and strips punctuation but does NOT convert
digits to words. The novice transcripts always spell numbers out
(e.g. "zero six hundred", "twenty third", "north-northeast"), so any model
that emits digits must verbalize them post-hoc to avoid free word errors.

This module is the single source of truth for that verbalization. Both the
faster-whisper manager (`asr_manager.py`) and the NeMo Parakeet manager
(`asr_manager_nemo.py`) import from here.
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
    suffix = match.group(1) if match.group(1) else ""
    if matched.isupper():
        base = "SIM JIAHONG"
    elif matched.islower():
        base = "sim jiahong"
    else:
        base = "Sim Jiahong"
    return base + suffix


def repl_jiahong_standalone(match: re.Match[str]) -> str:
    matched = match.group(0)
    suffix = match.group(1) if match.group(1) else ""
    base = _preserve_case("Jiahong", matched[:-len(suffix)] if suffix else matched)
    return base + suffix


def repl_blackshore(match: re.Match[str]) -> str:
    matched = match.group(0)
    suffix = match.group(1) if match.group(1) else ""
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


_PROPER_NOUN_RULES = [
    # 1. Complex/Combined Names (to avoid parts getting replaced by standalone rules)
    (re.compile(r"\b(takeshi|ada)\s+(oilaran|olrn|oyelaran)(s?|['s]*)\b", re.I), repl_takeshi_ada_oyelaran),
    (re.compile(r"\b(devika|divika|davika|de\s+vika|devi\s+ka)\s+(oranyan|uranyan|auranyan|aranyan|origins)(s?|['s]*)\b", re.I), repl_devika_oranyan),
    (re.compile(r"\bdivikauranyan(s?|['s]*)\b", re.I), repl_divikauranyan),
    (re.compile(r"\bdevi\s+kauranyan(s?|['s]*)\b", re.I), repl_devi_kauranyan),
    
    # 2. Park Soo-Hyun
    (re.compile(r"\b(park|pak|pack)\s+(su\s+hyun|soo\s+hyun|suzanne|suhyon|suhyun|shohyan|sho\s+hyan|su\s+hyon)(s?|['s]*)\b", re.I), repl_park_soo_hyun),
    (re.compile(r"\b(su|soo)\s+hyun(s?|['s]*)\b", re.I), repl_soo_hyun_standalone),
    (re.compile(r"\b(suyan|suyon|sujan|suhyan|soohyan|suhyun|suhyon|soohyun|shohyan|suzanne)(s?|['s]*)\b", re.I), repl_suyan_standalone),
    
    # 3. New Mewan
    (re.compile(r"\b(new|nu|noo|u)\s*(mewan|miwan|muvan|muon|muan|mi1|mu1|mi\s*1|mu\s*1|mi\s*one|mu\s*one|muons?|muvans?|miwans?|muans?)(s?|['s]*)\b", re.I), repl_new_mewan),
    (re.compile(r"\b(numiwan|numuan|mumuan|umiwan)(s?|['s]*)\b", re.I), repl_standalone_mewan),
    
    # 4. Sim Jiahong
    (re.compile(r"\bsim\s+jahong(s?|['s]*)\b", re.I), repl_sim_jiahong),
    (re.compile(r"\bjahong(s?|['s]*)\b", re.I), repl_jiahong_standalone),
    
    # 5. Phyrexis
    (re.compile(r"\b(perex|perexis|pyrex|pyrexis|firex|firexes|fedex)('s)?\b", re.I), repl_phyrexis),
    
    # 6. Kestrelian
    (re.compile(r"\b(castralian|castrillian|kestralian|kestrillian|castrelian|kastrillian|kesrelian)(s?)\b", re.I), repl_kestrelian),
    
    # 7. Sarento / Sorrento
    (re.compile(r"\b(s[oae]r+ento)(s?)\b", re.I), repl_sarento),
    
    # 8. Cyanite
    (re.compile(r"\b(cyanide|syanite|sanite|sinide)(s?)\b", re.I), repl_cyanite),
    
    # 9. Renhwa
    (re.compile(r"\b(renwa|renva|renhua|renhoa|renha|ren\s+ha)(s?)\b", re.I), repl_renhwa),
    
    # 10. Standalone Last Names / Names
    (re.compile(r"\b(oilaran|olrn)(s?|['s]*)\b", re.I), repl_oyelaran_standalone),
    (re.compile(r"\b(oranyan|uranyan|auranyan)(s?|['s]*)\b", re.I), repl_oranyan_standalone),
    (re.compile(r"\b(kashikarikari|kashkari|kashigari)(s?|['s]*)\b", re.I), repl_kashikari_standalone),
    (re.compile(r"\b(devika|divika|davika|de\s+vika)\b", re.I), repl_devika_standalone),
    (re.compile(r"\b(davenport|tavernport)(s?|['s]*)\b", re.I), repl_tavenport_standalone),
    (re.compile(r"\b(vayanova|vyanova|vianova|vaianova)(s?|['s]*)\b", re.I), repl_veyanova),
    (re.compile(r"\b(del|skel|tell)\s+(del[- ]?ash[- ]?castle|ash[- ]?castle|ash\s+castle|ashcastle)(s?|['s]*)\b", re.I), repl_ashcastle_prefix),
    (re.compile(r"\b(delash|delashcastle|del[- ]?ash[- ]?castle|ash[- ]?castle|ash\s+castle|ashcastle)(s?|['s]*)\b", re.I), repl_ashcastle_standalone),
    
    # 11. Tidak
    (re.compile(r"\b(tedak|taidak|sidak|tiduck)\b", re.I), repl_tidak_standalone),
    (re.compile(r"\btidakran\b", re.I), repl_tidakran),
    
    # 12. Blackshore
    (re.compile(r"\bblack\s+shore(s?|['s]*)\b", re.I), repl_blackshore),
    
    # 13. Zonnon
    (re.compile(r"\b(zonon|zonan|zonun|zondon|zondun|zonkon|zonnan|zonnal|zonone|zono|zonom|zonanun|zonal)(s?|['s]*)\b", re.I), repl_zonnon),
    
    # 14. Caulfield
    (re.compile(r"\b(coalfield|colfield|callfield|coffield|cofield|colefield)(s?|['s]*)\b", re.I), repl_caulfield),
]


def correct_proper_nouns(text: str) -> str:
    for rx, repl in _PROPER_NOUN_RULES:
        text = rx.sub(repl, text)
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

    # Ordinals such as "23rd", "15th". Run before any int regex catches the
    # digit half and leaves an orphaned suffix. Use spoken ordinal forms so
    # "23rd" -> "twenty third" (not "twenty threerd").
    def repl_ordinal(match: re.Match[str]) -> str:
        return _int_to_ordinal(int(match.group(1)))

    text = re.sub(r"\b(\d+)(st|nd|rd|th)\b", repl_ordinal, text, flags=re.I)

    def repl_decimal(match: re.Match[str]) -> str:
        whole, frac = match.group(1), match.group(2)
        whole_words = _int_to_words(int(whole)) if whole else "zero"
        frac_words = " ".join(_DIGIT_WORDS[d] for d in frac)
        return f"{whole_words} point {frac_words}"

    # Require no adjacent digit or dot on either side so dotted coordinates such
    # as "0.8.4" or "1.2.3" do not get partially rewritten.
    text = re.sub(r"(?<![\d.])(\d+)\s*\.\s*(\d+)(?![\d.])", repl_decimal, text)
    # Multi-dot sequences like "1.2.3" are not decimals. Preserve token
    # boundaries by turning digit-to-digit dots into spaces before integer
    # verbalization; the scorer removes punctuation without inserting spaces.
    text = re.sub(r"(?<=\d)\.(?=\d)", " ", text)

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
    text = correct_proper_nouns(text)
    return text

