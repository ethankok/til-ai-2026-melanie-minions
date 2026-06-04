"""Tests for melee_eval's --confpol-ckpt gate-knob parsing (consultant gate sweep).

Grammar under test: LABEL=PATH (checkpoint only, backward-compatible) or
LABEL=PATH@eps=E,floor=F,ovr=O (also set the confpol gate knobs for this
candidate, so one sweep can register the same checkpoint at several thresholds).
"""

import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "training" / "ae"))

from melee_eval import _parse_confpol_spec


def test_checkpoint_only_is_backward_compatible():
    label, env = _parse_confpol_spec(
        "native-u100=training/ae/checkpoints/confpol-native-u100.pt"
    )
    assert label == "native-u100"
    assert env == {
        "AE_POLICY_CHECKPOINT": str(
            Path("training/ae/checkpoints/confpol-native-u100.pt").resolve()
        )
    }


def test_gate_suffix_sets_all_three_knobs():
    label, env = _parse_confpol_spec("g-less=/tmp/x.pt@eps=2,floor=5,ovr=0")
    assert label == "g-less"
    assert env["AE_POLICY_CHECKPOINT"] == str(Path("/tmp/x.pt").resolve())
    assert env["AE_CONFPOL_MARGIN_EPSILON"] == "2"
    assert env["AE_CONFPOL_TOP_FLOOR"] == "5"
    assert env["AE_CONFPOL_OVERRIDE_TARGET_NONE"] == "0"


def test_partial_gate_suffix_only_sets_given_knobs():
    label, env = _parse_confpol_spec("g-more=/tmp/x.pt@eps=20")
    assert env["AE_CONFPOL_MARGIN_EPSILON"] == "20"
    assert "AE_CONFPOL_TOP_FLOOR" not in env
    assert "AE_CONFPOL_OVERRIDE_TARGET_NONE" not in env


def test_unknown_gate_knob_raises():
    with pytest.raises(ValueError):
        _parse_confpol_spec("bad=/tmp/x.pt@frobnicate=9")
