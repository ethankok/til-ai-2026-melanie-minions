"""Build an n-gram language model for NGPU-LM shallow fusion on the Parakeet-TDT
ASR path.

The residual ASR WER is almost entirely in-world proper-noun substitutions
(see training/asr/ERROR_ANALYSIS.md). An n-gram LM trained on in-domain text —
the ASR gold transcripts (closest match to the hidden test distribution, with
slang in spoken context) plus the NLP corpus (broad proper-noun coverage) —
directly biases the decoder toward that vocabulary.

This script does two things:

1. ``collect_training_text`` (pure Python, no NeMo): union of ASR transcripts and
   NLP-corpus sentences, de-duplicated. Reuses extract_slang.py's corpus reader
   so collection stays consistent with the slang pipeline. Unit-tested on the Mac.
2. ``main`` (Workbench): writes the collected lines to a temp file and shells out
   to NeMo's ``scripts/asr_language_modeling/ngram_lm/train_kenlm.py``, which
   tokenizes with the acoustic model's own SentencePiece tokenizer (read from the
   ``.nemo``) and trains the KenLM / NGPU-LM model.

Workbench usage::

    python training/asr/build_ngram_lm.py \
        --asr-jsonl /home/jupyter/novice/asr/asr.jsonl \
        --nlp-dir /home/jupyter/novice/nlp \
        --nemo-model asr/models/parakeet-tdt-0.6b-v2.nemo \
        --out asr/models/ngram_lm.nemo \
        --ngram-length 6 \
        --train-kenlm /path/to/NeMo/scripts/asr_language_modeling/ngram_lm/train_kenlm.py

NeMo recommends order 6 for BPE-based models. ``preserve_arpa=true`` keeps an
``.ARPA`` next to the output so the classic ``maes`` strategy can use it if the
NGPU-LM path is unavailable on the installed NeMo build.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

# Reuse the slang pipeline's permissive corpus reader so the LM sees exactly the
# same corpus text the slang prompt was mined from.
from extract_slang import _collect_strings, _iter_text_chunks

# Split on sentence-final punctuation so n-gram stats aren't dominated by whole
# mega-documents. Keep it dumb and deterministic.
_SENT_SPLIT_RE = re.compile(r"[.!?]+")


def _split_sentences(text: str) -> list[str]:
    """Split a text chunk into stripped, non-empty sentence-ish lines."""
    return [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]


def _iter_asr_transcripts(asr_jsonl: Path):
    """Yield the ``transcript`` field of each line in the ASR manifest.

    Tolerates a missing file (returns nothing) so the collector never crashes
    when only one source is present.
    """
    if not asr_jsonl.exists():
        return
    with open(asr_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = obj.get("transcript")
            if isinstance(t, str) and t.strip():
                yield t.strip()


def collect_training_text(asr_jsonl: Path, nlp_dir: Path) -> list[str]:
    """Return de-duplicated in-domain text lines for n-gram LM training.

    Order: ASR transcripts first (tightest distribution match), then NLP-corpus
    sentences. De-duplicated while preserving first-seen order. Blank lines are
    dropped. Pure Python — no NeMo dependency.
    """
    seen: set[str] = set()
    lines: list[str] = []

    def _add(line: str) -> None:
        line = line.strip()
        if not line or line in seen:
            return
        seen.add(line)
        lines.append(line)

    for transcript in _iter_asr_transcripts(asr_jsonl):
        _add(transcript)

    if nlp_dir.exists():
        for chunk in _iter_text_chunks(nlp_dir):
            for sentence in _split_sentences(chunk):
                _add(sentence)

    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--asr-jsonl", type=Path, default=Path("/home/jupyter/novice/asr/asr.jsonl"))
    ap.add_argument("--nlp-dir", type=Path, default=Path("/home/jupyter/novice/nlp"))
    ap.add_argument(
        "--nemo-model",
        type=Path,
        required=True,
        help="Acoustic .nemo whose tokenizer trains the LM (must match the deployed model).",
    )
    ap.add_argument("--out", type=Path, default=Path("asr/models/ngram_lm.nemo"))
    ap.add_argument("--ngram-length", type=int, default=6, help="NeMo recommends 6 for BPE.")
    ap.add_argument(
        "--train-kenlm",
        type=Path,
        default=None,
        help=(
            "Path to NeMo's scripts/asr_language_modeling/ngram_lm/train_kenlm.py. "
            "If omitted, only the corpus text file is written (--text-only behaviour) "
            "and the train_kenlm command is printed for you to run."
        ),
    )
    ap.add_argument("--kenlm-bin", type=Path, default=None, help="KenLM bin folder (kenlm_bin_path=).")
    ap.add_argument(
        "--text-out",
        type=Path,
        default=None,
        help="Where to keep the collected corpus text. Default: a temp file.",
    )
    args = ap.parse_args()

    lines = collect_training_text(args.asr_jsonl, args.nlp_dir)
    if not lines:
        raise SystemExit(
            f"No training text collected from asr={args.asr_jsonl} nlp={args.nlp_dir}"
        )

    if args.text_out is not None:
        text_path = args.text_out
        text_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        tmp = tempfile.NamedTemporaryFile(
            "w", suffix=".txt", delete=False, encoding="utf-8"
        )
        text_path = Path(tmp.name)
        tmp.close()
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Collected {len(lines)} unique lines -> {text_path}", flush=True)

    cmd = [
        sys.executable,
        str(args.train_kenlm) if args.train_kenlm else
        "<NeMo>/scripts/asr_language_modeling/ngram_lm/train_kenlm.py",
        f"nemo_model_file={args.nemo_model}",
        f"train_paths=[{text_path}]",
        f"kenlm_model_file={args.out}",
        f"ngram_length={args.ngram_length}",
        "preserve_arpa=true",
        "save_nemo=True",
    ]
    if args.kenlm_bin is not None:
        cmd.append(f"kenlm_bin_path={args.kenlm_bin}")

    if args.train_kenlm is None:
        print("\n--train-kenlm not given. Run NeMo's trainer yourself:\n")
        print("  " + " ".join(cmd))
        return

    args.out.parent.mkdir(parents=True, exist_ok=True)
    print("Running: " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    print(f"Wrote n-gram LM -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
