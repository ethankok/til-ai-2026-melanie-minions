"""Tests for the PlannerWeights module (ae/src/planner_weights.py).

The planner's tunable surface as one typed dataclass: from_env() must read
the historical AE_* vocabulary with identical defaults and clamps, and
AEManager(weights=...) must honor a directly-constructed profile without
touching os.environ.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "ae" / "src"))

from ae_manager import AEManager  # noqa: E402
from planner_weights import PlannerWeights  # noqa: E402


def test_defaults_match_historical_values():
    w = PlannerWeights()
    assert w.mcts_enabled is False
    assert w.mcts_depth == 3
    assert w.enemy_base_value == 80.0
    assert w.dist_penalty == 1.15
    assert w.item_mission_value == 50.0
    assert w.tier1_repeat_kill is True
    assert w.tier1_defense_priority is False
    assert w.no_self_damage is False
    assert w.bomb_detonate_steps == 5


def test_from_env_reads_historical_vocabulary(monkeypatch):
    monkeypatch.setenv("AE_ENEMY_BASE_VALUE", "120.5")
    monkeypatch.setenv("AE_CONTENTION", "1")
    monkeypatch.setenv("AE_MCTS_BUDGET_MS", "40")
    monkeypatch.setenv("AE_FIRST_TARGET_TABLE", "1")
    w = PlannerWeights.from_env()
    assert w.enemy_base_value == 120.5
    assert w.contention_enabled is True
    assert w.mcts_time_budget_s == 0.040
    assert w.first_target_table_enabled is True


def test_clamps_apply_on_every_construction_path():
    w = PlannerWeights(mcts_depth=99, mcts_width=1, contention_pfloor=7.0,
                       stun_tax_mult=-3.0, plan_rescore_k=0)
    assert w.mcts_depth == 8
    assert w.mcts_width == 12
    assert w.contention_pfloor == 1.0
    assert w.stun_tax_mult == 0.0
    assert w.plan_rescore_k == 1


def test_manager_accepts_direct_profile_without_env(monkeypatch):
    # env says one thing; the explicit profile must win
    monkeypatch.setenv("AE_ENEMY_BASE_VALUE", "999")
    m = AEManager(weights=PlannerWeights(enemy_base_value=42.0,
                                         item_mission_value=7.0))
    assert m.ENEMY_BASE_VALUE == 42.0
    assert m.item_mission_value == 7.0
    assert m.ITEM_VALUES["mission"] == 7.0
    assert m.weights.enemy_base_value == 42.0


def test_manager_default_path_still_reads_env(monkeypatch):
    monkeypatch.setenv("AE_ITEM_MISSION_VALUE", "80")
    monkeypatch.setenv("AE_TIME_DANGER", "1")
    m = AEManager()
    assert m.item_mission_value == 80.0
    assert m.time_danger_enabled is True


def test_danger_horizon_floored_at_bomb_timer():
    m = AEManager(weights=PlannerWeights(danger_horizon=1))
    assert m.danger_horizon == m.BOMB_TIMER


def test_as_dict_round_trip():
    w = PlannerWeights(enemy_base_value=99.0)
    d = w.as_dict()
    assert d["enemy_base_value"] == 99.0
    assert PlannerWeights(**d) == w
