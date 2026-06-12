"""Mac-runnable unit tests for the pure logic in build_ngram_lm.py.

The NeMo-dependent parts (train_kenlm subprocess) are Workbench-only and not
covered here. We only test the text-collection / sentence-splitting that runs as
plain Python.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training" / "asr"))

from build_ngram_lm import collect_training_text, _split_sentences  # noqa: E402


def test_split_sentences_basic():
    text = "Approach Sarento. Hold at zero six hundred! Is Phyrexis near?"
    out = _split_sentences(text)
    assert out == [
        "Approach Sarento",
        "Hold at zero six hundred",
        "Is Phyrexis near",
    ]


def test_split_sentences_drops_empty_and_strips():
    assert _split_sentences("  a.  . b  ") == ["a", "b"]
    assert _split_sentences("") == []
    assert _split_sentences("   ") == []


def test_collect_includes_asr_transcripts(tmp_path: Path):
    asr = tmp_path / "asr.jsonl"
    asr.write_text(
        json.dumps({"key": "k1", "audio": "a.wav", "transcript": "hold at sarento", "language": "english"})
        + "\n"
        + json.dumps({"key": "k2", "audio": "b.wav", "transcript": "phyrexis inbound", "language": "english"})
        + "\n",
        encoding="utf-8",
    )
    nlp_dir = tmp_path / "nlp"
    nlp_dir.mkdir()

    lines = collect_training_text(asr, nlp_dir)
    assert "hold at sarento" in lines
    assert "phyrexis inbound" in lines


def test_collect_includes_nlp_corpus_sentences(tmp_path: Path):
    asr = tmp_path / "asr.jsonl"
    asr.write_text("", encoding="utf-8")
    nlp_dir = tmp_path / "nlp"
    nlp_dir.mkdir()
    (nlp_dir / "docs.jsonl").write_text(
        json.dumps({"document": "The Cyanite mines. Floodwall holds."}) + "\n",
        encoding="utf-8",
    )

    lines = collect_training_text(asr, nlp_dir)
    assert "The Cyanite mines" in lines
    assert "Floodwall holds" in lines


def test_collect_dedupes_and_drops_blanks(tmp_path: Path):
    asr = tmp_path / "asr.jsonl"
    asr.write_text(
        json.dumps({"transcript": "repeat me"}) + "\n"
        + json.dumps({"transcript": "repeat me"}) + "\n"
        + json.dumps({"transcript": "   "}) + "\n"
        + json.dumps({"transcript": ""}) + "\n",
        encoding="utf-8",
    )
    nlp_dir = tmp_path / "nlp"
    nlp_dir.mkdir()

    lines = collect_training_text(asr, nlp_dir)
    assert lines.count("repeat me") == 1
    assert "" not in lines
    assert all(line.strip() for line in lines)


def test_collect_tolerates_missing_asr_and_nlp(tmp_path: Path):
    asr = tmp_path / "nope.jsonl"
    nlp_dir = tmp_path / "alsonope"
    lines = collect_training_text(asr, nlp_dir)
    assert lines == []
