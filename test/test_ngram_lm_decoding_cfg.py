"""Mac-runnable unit tests for the pure decoding-config builder in
asr_manager_nemo.py. This does NOT import NeMo: _build_lm_decoding_cfg is a
plain dict transform, so it is importable and testable on the Mac.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "asr" / "src"))

from asr_manager_nemo import _build_lm_decoding_cfg  # noqa: E402


def test_sets_strategy_and_beam_fields():
    base = {"strategy": "greedy_batch", "beam": {}}
    cfg = _build_lm_decoding_cfg(
        base,
        lm_path="/m/ngram_lm.nemo",
        alpha=0.3,
        beam_size=4,
        strategy="malsd_batch",
        pruning_mode="late",
        blank_lm_score_mode="lm_weighted_full",
    )
    assert cfg["strategy"] == "malsd_batch"
    assert cfg["beam"]["beam_size"] == 4
    assert cfg["beam"]["ngram_lm_model"] == "/m/ngram_lm.nemo"
    assert cfg["beam"]["ngram_lm_alpha"] == 0.3
    assert cfg["beam"]["pruning_mode"] == "late"
    assert cfg["beam"]["blank_lm_score_mode"] == "lm_weighted_full"


def test_does_not_mutate_input():
    base = {"strategy": "greedy_batch", "beam": {"beam_size": 1}}
    _build_lm_decoding_cfg(
        base,
        lm_path="/m/x.nemo",
        alpha=0.2,
        beam_size=8,
        strategy="malsd_batch",
        pruning_mode="late",
        blank_lm_score_mode="lm_weighted_full",
    )
    assert base["strategy"] == "greedy_batch"
    assert base["beam"] == {"beam_size": 1}


def test_creates_beam_subdict_when_absent():
    base = {"strategy": "greedy_batch"}  # no 'beam' key
    cfg = _build_lm_decoding_cfg(
        base,
        lm_path="/m/x.nemo",
        alpha=0.1,
        beam_size=2,
        strategy="malsd_batch",
        pruning_mode="early",
        blank_lm_score_mode="no_score",
    )
    assert cfg["beam"]["beam_size"] == 2
    assert cfg["beam"]["ngram_lm_alpha"] == 0.1


def test_preserves_other_base_keys():
    base = {"strategy": "greedy_batch", "compute_timestamps": False, "beam": {"foo": 1}}
    cfg = _build_lm_decoding_cfg(
        base,
        lm_path="/m/x.nemo",
        alpha=0.3,
        beam_size=4,
        strategy="malsd_batch",
        pruning_mode="late",
        blank_lm_score_mode="lm_weighted_full",
    )
    assert cfg["compute_timestamps"] is False
    assert cfg["beam"]["foo"] == 1  # existing beam sub-keys preserved
