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
    return text
