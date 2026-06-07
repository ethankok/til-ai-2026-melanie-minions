"""Tests for melee_eval's --confpol-ckpt gate-knob parsing (consultant gate sweep).

Grammar under test: LABEL=PATH (checkpoint only, backward-compatible) or
LABEL=PATH@eps=E,floor=F,ovr=O (also set the confpol gate knobs for this
candidate, so one sweep can register the same checkpoint at several thresholds).
"""

import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "training" / "ae"))

from melee_eval import (
    _parse_confpol_spec,
    _wilson_lower,
    _paired_placement_stats,
    _gap_not_widening,
    _promotion_verdict,
)


# ---------------------------------------------------------------------------
# Reform 3 — Wilson lower bound on Probability-of-Improvement
# ---------------------------------------------------------------------------
def test_wilson_lower_bounds_and_monotonicity():
    assert _wilson_lower(0, 0) == 0.0           # no data -> 0
    assert _wilson_lower(0, 10) == 0.0          # all losses -> lower bound pinned at 0
    # all wins: lower bound is high but strictly < 1 (interval has width)
    allwin = _wilson_lower(10, 10)
    assert 0.6 < allwin < 1.0
    # more wins at fixed n -> higher lower bound
    assert _wilson_lower(8, 10) > _wilson_lower(5, 10) > _wilson_lower(2, 10)
    # a coin flip's lower bound sits below 0.5 (can't claim improvement)
    assert _wilson_lower(5, 10) < 0.5
    # the same proportion with more samples gives a tighter (higher) lower bound
    assert _wilson_lower(80, 100) > _wilson_lower(8, 10)


# ---------------------------------------------------------------------------
# Reform 3 — paired per-seed placement stats (lower placement = better)
# ---------------------------------------------------------------------------
def _mk(worst, margin, score, places, suite="m"):
    """Minimal candidate result dict carrying paired run rows under one suite."""
    rows = [{"hash": 0, "sim": i, "mean_placement": p} for i, p in enumerate(places)]
    return {
        "worst_bracket_placement": worst,
        "min_margin": margin,
        "semis_mixed_score": score,
        "per_bracket": {suite: {"run_rows": rows}},
    }


def test_paired_stats_candidate_strictly_better():
    inc = _mk(3.0, -50.0, 0.40, [3, 3, 3, 3, 3, 3])
    cand = _mk(2.0, -50.0, 0.40, [2, 2, 2, 2, 2, 2])
    st = _paired_placement_stats(cand, inc)
    assert st["n_pairs"] == 6
    assert st["wins"] == 6 and st["ties"] == 0 and st["losses"] == 0
    assert abs(st["mean_delta"] - 1.0) < 1e-9   # inc - cand
    assert st["poi"] == 1.0
    assert st["poi_lower"] > 0.5


def test_paired_stats_candidate_worse():
    inc = _mk(3.0, -50.0, 0.40, [2, 2, 2, 2])
    cand = _mk(3.0, -50.0, 0.40, [3, 3, 3, 3])
    st = _paired_placement_stats(cand, inc)
    assert st["mean_delta"] < 0
    assert st["poi"] == 0.0 and st["poi_lower"] == 0.0


def test_paired_stats_within_noise_is_a_coin_flip():
    inc = _mk(3.0, -50.0, 0.40, [3.0, 3.0, 3.0, 3.0])
    cand = _mk(3.0, -50.0, 0.40, [2.95, 3.05, 2.90, 3.10])
    st = _paired_placement_stats(cand, inc)
    assert abs(st["mean_delta"]) < 1e-9          # net zero
    assert st["poi"] == 0.5
    assert st["poi_lower"] < 0.5                  # cannot claim improvement


# ---------------------------------------------------------------------------
# Reform 2 + 3 — promotion verdict: placement primary, off-construct = floors,
# real-and-reliable effect required.
# ---------------------------------------------------------------------------
def test_placement_better_but_reward_shy_is_promotable():
    # Lower reward + more-negative margin (both within the non-crater band) must NOT
    # block a candidate that clearly places better. This is the g00-fixed-03 shape.
    inc = _mk(3.0, -50.0, 0.40, [3, 3, 3, 3, 3, 3])
    cand = _mk(2.0, -70.0, 0.36, [2, 2, 2, 2, 2, 2])  # margin -70 (floor ~-75.5), score 0.36 (floor 0.34)
    v = _promotion_verdict(cand, inc)
    assert v["minimax_placement_ok"]
    assert v["margin_noncrater_ok"]
    assert v["score_noncrater_ok"]
    assert v["effect_ok"] and v["poi_ok"]
    assert v["promotable"] is True


def test_reward_cratered_candidate_is_rejected_by_score_floor():
    inc = _mk(3.0, -50.0, 0.40, [3, 3, 3, 3, 3, 3])
    cand = _mk(2.0, -50.0, 0.20, [2, 2, 2, 2, 2, 2])  # 0.20 < 0.85*0.40 = 0.34
    v = _promotion_verdict(cand, inc)
    assert v["score_noncrater_ok"] is False
    assert v["promotable"] is False


def test_margin_collapse_is_rejected_by_margin_floor():
    inc = _mk(3.0, -50.0, 0.40, [3, 3, 3, 3, 3, 3])
    cand = _mk(2.0, -500.0, 0.40, [2, 2, 2, 2, 2, 2])  # -500 << floor ~ -75.5
    v = _promotion_verdict(cand, inc)
    assert v["margin_noncrater_ok"] is False
    assert v["promotable"] is False


def test_within_noise_placement_gain_is_not_promotable():
    inc = _mk(3.0, -50.0, 0.40, [3.0, 3.0, 3.0, 3.0])
    cand = _mk(2.95, -50.0, 0.40, [2.95, 3.05, 2.90, 3.10])  # |mean_delta| < 0.3
    v = _promotion_verdict(cand, inc)
    assert v["minimax_placement_ok"]              # nominal worst-bracket is "better"
    assert v["effect_ok"] is False                # but the effect is within noise
    assert v["promotable"] is False


def test_placement_worse_fails_minimax():
    inc = _mk(3.0, -50.0, 0.40, [3, 3, 3, 3])
    cand = _mk(3.5, -50.0, 0.40, [3.5, 3.5, 3.5, 3.5])
    v = _promotion_verdict(cand, inc)
    assert v["minimax_placement_ok"] is False
    assert v["promotable"] is False


# ---------------------------------------------------------------------------
# Reform 4 — held-out composition gap (overfit alarm)
# ---------------------------------------------------------------------------
def test_gap_not_widening():
    # candidate's heldout-minus-tune gap may not exceed incumbent's by > tol
    assert _gap_not_widening(0.4, 0.3, tol=0.5) is True    # 0.4 <= 0.3 + 0.5
    assert _gap_not_widening(1.2, 0.3, tol=0.5) is False   # 1.2 > 0.8 -> overfit
    assert _gap_not_widening(-0.2, 0.3, tol=0.5) is True   # candidate transfers better


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
