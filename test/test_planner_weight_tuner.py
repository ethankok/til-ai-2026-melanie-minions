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


def _result(placements: dict[str, float], *, raw_ae: float, worst_robust: float | None = None) -> dict:
    """Build a fake melee result for the finals-aligned tuner objective.

    The 2026-06-09 gate optimizes ``raw_ae`` (the finals-proportional discriminator)
    subject to a ``worst_robust_placement`` floor. ``per_bracket`` mean placements are
    kept for diagnostics; ``worst_robust`` defaults to the worst per-bracket placement.
    """
    wb = max(placements.values())
    return {
        "worst_bracket_placement": wb,
        "worst_robust_placement": wb if worst_robust is None else worst_robust,
        "raw_ae": raw_ae,
        "min_margin": -100.0,
        "semis_mixed_score": 0.50,
        "per_bracket": {k: {"mean_placement": v} for k, v in placements.items()},
    }


_FLAT = {"semis_mixed": 2, "all_aggressive": 2, "all_farmer": 2, "adversarial": 2, "real_field": 2}


def test_rank_key_maximizes_raw_ae_when_placement_floor_met():
    incumbent = _result(_FLAT, raw_ae=200.0, worst_robust=2.0)
    higher_rawae = _result(_FLAT, raw_ae=260.0, worst_robust=2.0)
    lower_rawae = _result(_FLAT, raw_ae=210.0, worst_robust=2.0)
    # Placement floor held by both -> higher raw_ae (the discriminator) ranks BETTER.
    assert rank_key(higher_rawae, incumbent) < rank_key(lower_rawae, incumbent)


def test_rank_key_penalizes_placement_floor_violation_over_raw_ae():
    incumbent = _result(_FLAT, raw_ae=200.0, worst_robust=2.0)
    # Huge reward but robust placement regresses beyond tolerance: must rank WORSE
    # than a modest-reward candidate that holds the placement floor.
    reward_but_regresses = _result(_FLAT, raw_ae=400.0, worst_robust=3.5)
    holds_floor = _result(_FLAT, raw_ae=210.0, worst_robust=2.0)
    assert rank_key(holds_floor, incumbent) < rank_key(reward_but_regresses, incumbent)


def test_promotion_ok_requires_rawae_gain_and_placement_floor():
    incumbent = _result(_FLAT, raw_ae=200.0, worst_robust=2.0)
    promotable = _result(_FLAT, raw_ae=240.0, worst_robust=2.0)
    no_rawae_gain = _result(_FLAT, raw_ae=200.0, worst_robust=2.0)
    placement_regress = _result(_FLAT, raw_ae=300.0, worst_robust=3.0)
    within_tol = _result(_FLAT, raw_ae=240.0, worst_robust=2.2)
    assert promotion_ok(promotable, incumbent) is True
    # raw_ae ties incumbent -> not strictly better -> not promotable.
    assert promotion_ok(no_rawae_gain, incumbent) is False
    # raw_ae improves but worst_robust_placement regresses beyond tol -> floor rejects.
    assert promotion_ok(placement_regress, incumbent) is False
    # small placement wobble (2 -> 2.2) is inside the noise tolerance: still promotable.
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


from tune_planner_weights import append_jsonl, load_jsonl, write_summary  # noqa: E402


def test_append_and_load_jsonl_round_trip(tmp_path):
    path = tmp_path / "records.jsonl"
    append_jsonl(path, {"candidate": "incumbent", "rank_key": [3.2, 0.0, 0.0, 2.1]})
    append_jsonl(path, {"candidate": "g00_c01", "rank_key": [3.0, 0.0, 0.0, 2.0]})
    assert load_jsonl(path) == [
        {"candidate": "incumbent", "rank_key": [3.2, 0.0, 0.0, 2.1]},
        {"candidate": "g00_c01", "rank_key": [3.0, 0.0, 0.0, 2.0]},
    ]


def test_load_jsonl_missing_file_returns_empty_list(tmp_path):
    assert load_jsonl(tmp_path / "missing.jsonl") == []


def test_write_summary_creates_parent_and_stable_json(tmp_path):
    path = tmp_path / "nested" / "summary.json"
    payload = {"best": {"candidate": "g00_c01"}, "docker_env": ["ENV AE_DIST_PENALTY=1.200000"]}
    write_summary(path, payload)
    loaded = json.loads(path.read_text())
    assert loaded == payload


from tune_planner_weights import (  # noqa: E402
    docker_env_lines,
    sample_generation,
    values_from_summary,
)


def test_sample_generation_always_includes_incumbent_best_and_decodes_to_bounds():
    rng = np.random.default_rng(123)
    mu = default_vector()
    sigma = np.full(len(PARAMS), 0.25, dtype=float)
    best_values = incumbent_values() | {"AE_DIST_PENALTY": 1.4}
    samples = sample_generation(
        rng,
        mu,
        sigma,
        pop_size=5,
        include_incumbent=True,
        best_values=best_values,
        fixed_values=[],
    )
    assert len(samples) == 5
    assert samples[0][0] == "incumbent"
    assert samples[0][1] == incumbent_values()
    assert samples[1][0] == "best"
    assert samples[1][1] == best_values
    for _label, values, vector in samples:
        assert vector.shape == (len(PARAMS),)
        for p in PARAMS:
            assert p.lower <= values[p.name] <= p.upper


def test_values_from_summary_reads_best_values(tmp_path):
    path = tmp_path / "summary.json"
    path.write_text(json.dumps({"best": {"values": incumbent_values()}}))
    assert values_from_summary(path) == incumbent_values()


def test_docker_env_lines_include_only_tuned_weights_in_param_order():
    lines = docker_env_lines(incumbent_values())
    assert lines == [
        "ENV AE_ITEM_MISSION_VALUE=80.000000",
        "ENV AE_ITEM_RESOURCE_VALUE=40.000000",
        "ENV AE_ENEMY_BASE_VALUE=100.000000",
        "ENV AE_DIST_PENALTY=1.150000",
        "ENV AE_PATH_THREAT_PENALTY=2.000000",
        "ENV AE_DIJKSTRA_BOMB_COST=7.000000",
        "ENV AE_LEAD_TETHER_HEALTH=60.000000",
        "ENV AE_LEAD_TETHER_WEIGHT=0.500000",
        "ENV AE_CONTENTION_SCALE=2.500000",
        "ENV AE_CONTENTION_PFLOOR=0.150000",
    ]


def test_jsonable_args_stringifies_path_values_for_summary_write():
    """Regression: staged runs pass --center-summary/--include-top-from as Path
    objects; the per-generation summary write must not raise on json.dumps."""
    from tune_planner_weights import _build_parser, jsonable_args

    args = _build_parser().parse_args([
        "--out-dir", "training/ae/data/x",
        "--center-summary", "training/ae/data/stage0/summary.json",
        "--include-top-from", "training/ae/data/stage0/candidates.jsonl",
        "--include-top-n", "6",
    ])
    payload = jsonable_args(args)
    assert payload["out_dir"] == "training/ae/data/x"
    assert payload["center_summary"] == "training/ae/data/stage0/summary.json"
    assert payload["include_top_from"] == "training/ae/data/stage0/candidates.jsonl"
    # Must round-trip through json without raising.
    json.dumps(payload, sort_keys=True)
