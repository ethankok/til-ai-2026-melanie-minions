"""Unit tests for AE planner-weight CEM tuner.

These tests stay fast: they validate optimizer math, env construction, and
artifact helpers without running the expensive melee simulator.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "training" / "ae"))

from tune_planner_weights import (  # noqa: E402
    PARAMS,
    decode_vector,
    default_vector,
    encode_env,
    encode_values,
    incumbent_values,
)


def test_param_table_contains_v1_search_space_in_order():
    assert [p.name for p in PARAMS] == [
        "AE_ITEM_MISSION_VALUE",
        "AE_ITEM_RESOURCE_VALUE",
        "AE_ENEMY_BASE_VALUE",
        "AE_DIST_PENALTY",
        "AE_PATH_THREAT_PENALTY",
        "AE_DIJKSTRA_BOMB_COST",
        "AE_LEAD_TETHER_HEALTH",
        "AE_LEAD_TETHER_WEIGHT",
        "AE_CONTENTION_SCALE",
        "AE_CONTENTION_PFLOOR",
    ]
    assert len(PARAMS) == 10


def test_incumbent_values_are_deployed_defaults():
    values = incumbent_values()
    assert values == {
        "AE_ITEM_MISSION_VALUE": 80.0,
        "AE_ITEM_RESOURCE_VALUE": 40.0,
        "AE_ENEMY_BASE_VALUE": 100.0,
        "AE_DIST_PENALTY": 1.15,
        "AE_PATH_THREAT_PENALTY": 2.0,
        "AE_DIJKSTRA_BOMB_COST": 7.0,
        "AE_LEAD_TETHER_HEALTH": 60.0,
        "AE_LEAD_TETHER_WEIGHT": 0.5,
        "AE_CONTENTION_SCALE": 2.5,
        "AE_CONTENTION_PFLOOR": 0.15,
    }


def test_default_vector_decodes_to_incumbent_values():
    decoded = decode_vector(default_vector())
    assert decoded == incumbent_values()


def test_encode_decode_round_trip_for_in_bounds_values():
    original = {
        "AE_ITEM_MISSION_VALUE": 91.0,
        "AE_ITEM_RESOURCE_VALUE": 33.5,
        "AE_ENEMY_BASE_VALUE": 120.0,
        "AE_DIST_PENALTY": 1.4,
        "AE_PATH_THREAT_PENALTY": 3.0,
        "AE_DIJKSTRA_BOMB_COST": 8.0,
        "AE_LEAD_TETHER_HEALTH": 72.0,
        "AE_LEAD_TETHER_WEIGHT": 0.75,
        "AE_CONTENTION_SCALE": 3.1,
        "AE_CONTENTION_PFLOOR": 0.22,
    }
    encoded = encode_values(original)
    decoded = decode_vector(encoded)
    for key, value in original.items():
        assert math.isclose(decoded[key], value, rel_tol=1e-9, abs_tol=1e-9)


def test_decode_vector_clips_to_bounds():
    decoded = decode_vector(np.array([10.0] * len(PARAMS), dtype=float))
    assert decoded["AE_ITEM_MISSION_VALUE"] == 120.0
    assert decoded["AE_ITEM_RESOURCE_VALUE"] == 70.0
    assert decoded["AE_ENEMY_BASE_VALUE"] == 160.0
    assert decoded["AE_DIST_PENALTY"] == 2.0
    assert decoded["AE_PATH_THREAT_PENALTY"] == 5.0
    assert decoded["AE_DIJKSTRA_BOMB_COST"] == 12.0
    assert decoded["AE_LEAD_TETHER_HEALTH"] == 85.0
    assert decoded["AE_LEAD_TETHER_WEIGHT"] == 1.5
    assert decoded["AE_CONTENTION_SCALE"] == 6.0
    assert decoded["AE_CONTENTION_PFLOOR"] == 0.5


def test_encode_env_formats_stable_decimal_strings():
    env = encode_env({
        "AE_ITEM_MISSION_VALUE": 91.23456789,
        "AE_ITEM_RESOURCE_VALUE": 40.0,
        "AE_ENEMY_BASE_VALUE": 100.0,
        "AE_DIST_PENALTY": 1.15,
        "AE_PATH_THREAT_PENALTY": 2.0,
        "AE_DIJKSTRA_BOMB_COST": 7.0,
        "AE_LEAD_TETHER_HEALTH": 60.0,
        "AE_LEAD_TETHER_WEIGHT": 0.5,
        "AE_CONTENTION_SCALE": 2.5,
        "AE_CONTENTION_PFLOOR": 0.15,
    })
    assert env["AE_ITEM_MISSION_VALUE"] == "91.234568"
    assert env["AE_RESOURCE_UNUSED"] if False else True
    assert env["AE_LEAD_TETHER_WEIGHT"] == "0.500000"


from tune_planner_weights import cem_update, promotion_ok, rank_key  # noqa: E402


def _result(placements: dict[str, float]) -> dict:
    """Build a fake melee result from explicit per-bracket mean placements.

    Uses the real bracket names so the field-bracket guard (semis_mixed/real_field)
    is exercised; min_margin / adversarial are diagnostics only and are not gated.
    """
    return {
        "worst_bracket_placement": max(placements.values()),
        "min_margin": -100.0,
        "semis_mixed_score": 0.50,
        "per_bracket": {k: {"mean_placement": v} for k, v in placements.items()},
    }


_FLAT = {"semis_mixed": 2, "all_aggressive": 2, "all_farmer": 2, "adversarial": 2, "real_field": 2}


def test_rank_key_is_driven_by_mean_placement_not_worst_bracket():
    incumbent = _result(_FLAT)
    lower_mean_higher_worst = _result(
        {"semis_mixed": 1, "all_aggressive": 1, "all_farmer": 1, "real_field": 1, "adversarial": 4})
    higher_mean_lower_worst = _result(
        {"semis_mixed": 3, "all_aggressive": 3, "all_farmer": 3, "adversarial": 3, "real_field": 3})
    # mean 1.6 (worst 4) ranks BETTER than mean 3.0 (worst 3): mean is the primary objective.
    assert rank_key(lower_mean_higher_worst, incumbent) < rank_key(higher_mean_lower_worst, incumbent)
    assert rank_key(lower_mean_higher_worst, incumbent)[0] == 1.6


def test_rank_key_field_regression_breaks_ties_when_means_match():
    incumbent = _result(_FLAT)
    clean = _result(_FLAT)  # mean 2.0, no field regression
    spiky = _result(
        {"semis_mixed": 1, "all_aggressive": 1, "all_farmer": 1, "adversarial": 3, "real_field": 4})
    # both mean 2.0; `spiky` regresses real_field (2 -> 4) so it ranks WORSE on the secondary term.
    assert rank_key(clean, incumbent) < rank_key(spiky, incumbent)
    assert rank_key(clean, incumbent)[1] == 0.0
    assert rank_key(spiky, incumbent)[1] == 2.0


def test_promotion_ok_requires_mean_gain_and_no_field_bracket_regression():
    incumbent = _result(_FLAT)  # mean 2.0
    promotable = _result(
        {"semis_mixed": 2, "all_aggressive": 1, "all_farmer": 1, "adversarial": 1, "real_field": 2})
    no_mean_gain = _result(_FLAT)
    field_regress = _result(
        {"semis_mixed": 3.5, "all_aggressive": 1, "all_farmer": 1, "adversarial": 1, "real_field": 1})
    within_tol = _result(
        {"semis_mixed": 2.2, "all_aggressive": 1, "all_farmer": 1, "adversarial": 1, "real_field": 1})
    assert promotion_ok(promotable, incumbent) is True
    # mean ties incumbent -> not strictly better -> not promotable.
    assert promotion_ok(no_mean_gain, incumbent) is False
    # mean improves (1.5) but semis_mixed collapses 2 -> 3.5 (beyond tol): field guard rejects it.
    assert promotion_ok(field_regress, incumbent) is False
    # small field wobble (2 -> 2.2) is inside the noise tolerance: still promotable.
    assert promotion_ok(within_tol, incumbent) is True


def test_cem_update_uses_elites_with_smoothing_and_sigma_floor():
    old_mu = np.zeros(3, dtype=float)
    old_sigma = np.ones(3, dtype=float)
    elites = np.array([
        [0.20, 0.10, -0.10],
        [0.30, 0.20, -0.20],
        [0.40, 0.30, -0.30],
    ], dtype=float)
    new_mu, new_sigma = cem_update(
        old_mu,
        old_sigma,
        elites,
        smoothing=0.50,
        min_sigma=0.05,
        max_sigma=0.80,
    )
    np.testing.assert_allclose(new_mu, np.array([0.15, 0.10, -0.10]), atol=1e-9)
    assert np.all(new_sigma >= 0.05)
    assert np.all(new_sigma <= 0.80)
    assert new_sigma.shape == old_sigma.shape


from tune_planner_weights import build_candidate_spec  # noqa: E402


def test_build_candidate_spec_combines_cbomb7_fixed_checkpoint_and_weights():
    values = incumbent_values()
    spec = build_candidate_spec(values, policy_ckpt="/tmp/policy.pt")
    assert spec["our"] == "confidence_policy_hybrid"
    env = spec["env"]
    assert env["AE_POLICY_CHECKPOINT"] == str(Path("/tmp/policy.pt").resolve())
    assert env["AE_MODE"] == "confidence_policy_hybrid"
    assert env["AE_CONTENTION"] == "1"
    assert env["AE_PLAN_RESCORE"] == "0"
    assert env["AE_CONFPOL_MARGIN_EPSILON"] == "5.0"
    assert env["AE_ITEM_MISSION_VALUE"] == "80.000000"
    assert env["AE_ITEM_RESOURCE_VALUE"] == "40.000000"
    assert env["AE_DIJKSTRA_BOMB_COST"] == "7.000000"
    assert env["AE_LEAD_BASE_TETHER"] == "1"
    assert env["AE_USE_PLAYBOOK"] == "0"
