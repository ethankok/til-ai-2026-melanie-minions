"""Tests for the AdaptiveRewardShaper respawn-reward loophole fix.

The agent-health shaping term used to credit `health_delta_coef * (health -
last_health)`. On death -> respawn, health jumps 0 -> 100, producing a large
POSITIVE bonus that refunds the whole life's accumulated damage penalty, so a
death carried no net shaped deterrent. The fix clamps the *agent*-health term
to negative deltas only (damage/death penalized; the respawn jump contributes
0). The base-health term is deliberately left unchanged (bases don't respawn).
"""

import argparse
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "training" / "ae"))

from train_ppo import AdaptiveRewardShaper


def _make_shaper(health_coef: float = 0.01, base_coef: float = 0.03) -> AdaptiveRewardShaper:
    args = argparse.Namespace(
        health_delta_coef=health_coef,
        base_health_delta_coef=base_coef,
    )
    shaper = AdaptiveRewardShaper(args)
    shaper.start_game()
    return shaper


def test_first_health_observation_sets_baseline_no_bonus():
    shaper = _make_shaper()
    # last_health is None on the first tick -> baseline only, no bonus.
    assert shaper._health_delta_bonus({"health": 100}) == 0.0


def test_taking_damage_is_penalized():
    shaper = _make_shaper(health_coef=0.01)
    shaper._health_delta_bonus({"health": 100})  # baseline
    # 100 -> 40 is a -60 delta -> negative bonus (penalty).
    assert shaper._health_delta_bonus({"health": 40}) == pytest.approx(0.01 * (40 - 100))


def test_respawn_jump_contributes_zero():
    """0 -> 100 on respawn must NOT pay out a positive bonus."""
    shaper = _make_shaper(health_coef=0.01)
    shaper._health_delta_bonus({"health": 0})  # baseline at death
    assert shaper._health_delta_bonus({"health": 100}) == 0.0


def test_full_death_cycle_nets_negative():
    """100 -> 40 -> 0 -> respawn 100 must be a net shaped LOSS, not ~0."""
    shaper = _make_shaper(health_coef=0.01)
    total = 0.0
    for h in (100, 40, 0, 100):
        total += shaper._health_delta_bonus({"health": h})
    # damage -0.6, death -0.4, respawn 0 -> -1.0 (deterrent restored).
    assert total == pytest.approx(0.01 * (40 - 100) + 0.01 * (0 - 40))
    assert total < 0.0


def test_post_respawn_damage_uses_new_baseline():
    """After respawn the next damage delta is measured from full health, not 0."""
    shaper = _make_shaper(health_coef=0.01)
    shaper._health_delta_bonus({"health": 0})  # death baseline
    shaper._health_delta_bonus({"health": 100})  # respawn (0 bonus, baseline -> 100)
    # 100 -> 70 should be -30, proving last_health was updated to 100 on respawn.
    assert shaper._health_delta_bonus({"health": 70}) == pytest.approx(0.01 * (70 - 100))


def test_base_health_term_unchanged_positive_delta_still_credited():
    """Regression guard: the base-health term is NOT clamped."""
    shaper = _make_shaper(base_coef=0.03)
    shaper._health_delta_bonus({"base_health": 100})  # baseline
    assert shaper._health_delta_bonus({"base_health": 130}) == pytest.approx(0.03 * 30)
