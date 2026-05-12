"""Mine in-world slang terms from the NLP training corpus.

The hackathon notes that ASR transcripts share a world with the NLP dataset and
contain original slang/phrases. We extract tokens that appear repeatedly in the
NLP corpus but are rare in standard English, and write them as a single
space-separated string to ``slang_prompt.txt``. That file is loaded by
``ASRManager`` and passed as ``initial_prompt`` to faster-whisper so the decoder
prior is shifted toward the dataset's vocabulary.

Usage on the GCP Workbench instance::

    python training/asr/extract_slang.py \
        --nlp-dir /home/jupyter/novice/nlp \
        --out asr/models/slang_prompt.txt \
        --top-k 80

Whisper truncates ``initial_prompt`` to roughly the last 224 tokens, so 80
short words is a safe budget.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


_TOKEN_RE = re.compile(r"[A-Za-z]+")


def _iter_text_chunks(nlp_dir: Path):
    """Yield raw text chunks from anything that looks like an NLP corpus file.

    The NLP task ships documents in JSONL and/or plain text. Be permissive: try
    JSONL first (yielding any string-valued field), fall back to plain text.
    """
    for path in sorted(nlp_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".jsonl", ".json", ".txt", ".md"}:
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                if path.suffix.lower() in {".jsonl", ".json"}:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            yield line
                            continue
                        _yield_strings(obj, out=lambda s: None)
                        for s in _collect_strings(obj):
                            yield s
                else:
                    yield f.read()
        except (OSError, UnicodeDecodeError):
            continue


def _collect_strings(obj) -> list[str]:
    out: list[str] = []
    _yield_strings(obj, out=out.append)
    return out


def _yield_strings(obj, out) -> None:
    if isinstance(obj, str):
        out(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            _yield_strings(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _yield_strings(v, out)


def _load_common_english() -> set[str]:
    """Return a lowercase set of common English words to subtract.

    Prefers wordfreq's top-N list; falls back to nltk's words corpus; falls back
    to an empty set (then every alpha token is candidate slang).
    """
    try:
        from wordfreq import top_n_list

        return set(w.lower() for w in top_n_list("en", 50000))
    except Exception:
        pass
    try:
        from nltk.corpus import words  # type: ignore

        return set(w.lower() for w in words.words())
    except Exception:
        return set()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--nlp-dir",
        type=Path,
        default=Path("/home/jupyter/novice/nlp"),
        help="Directory containing the NLP corpus files.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("asr/models/slang_prompt.txt"),
        help="Destination for the slang prompt text file.",
    )
    ap.add_argument(
        "--top-k",
        type=int,
        default=200,
        help=(
            "Number of slang tokens to keep. Whisper truncates the "
            "initial_prompt to roughly the last 224 tokens, so ~200 short "
            "words is the safe budget. ERROR_ANALYSIS shows proper-noun "
            "substitutions (Sarento, Cyanite, Phyrexis, Mewan, etc.) drive a "
            "large share of remaining WER, so we want broad coverage."
        ),
    )
    ap.add_argument(
        "--min-count",
        type=int,
        default=2,
        help="Minimum occurrences in the NLP corpus.",
    )
    ap.add_argument(
        "--min-len",
        type=int,
        default=4,
        help="Minimum token length.",
    )
    args = ap.parse_args()

    if not args.nlp_dir.exists():
        raise SystemExit(f"NLP dir does not exist: {args.nlp_dir}")

    common = _load_common_english()
    counts: Counter[str] = Counter()
    for chunk in _iter_text_chunks(args.nlp_dir):
        for tok in _TOKEN_RE.findall(chunk):
            tok = tok.lower()
            if len(tok) < args.min_len:
                continue
            if tok in common:
                continue
            counts[tok] += 1

    ranked = [t for t, c in counts.most_common() if c >= args.min_count]
    slang = ranked[: args.top_k]

    # NOTE on ordering: faster-whisper truncates initial_prompt to the LAST
    # ~223 tokens (via previous_tokens[-(max_length // 2 - 1):]). With ~200
    # proper nouns the tokenized prompt overflows. Intuition would say to
    # reverse the list so high-frequency terms land at the end and survive
    # truncation -- BUT the vad-off-v2 experiment showed that ordering
    # produced WORSE local WER than the original highest-frequency-first
    # layout (0.060 vs 0.055 on the 1028-clip Workbench manifest). Likely
    # cause: putting the most common in-world nouns immediately before
    # decode-start over-primes the decoder and causes false-positive
    # hallucinations on unrelated clips. Keeping high-frequency first lets
    # those terms be in the prompt as background context but not in the
    # last-attended position. See training/asr/ERROR_ANALYSIS.md.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(" ".join(slang) + "\n", encoding="utf-8")

    print(f"Wrote {len(slang)} slang tokens to {args.out}")
    print("Top 20 (highest frequency, written first in the prompt):",
          " ".join(slang[:20]))


if __name__ == "__main__":
    main()
