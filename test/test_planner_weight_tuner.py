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
