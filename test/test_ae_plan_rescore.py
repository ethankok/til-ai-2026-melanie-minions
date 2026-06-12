"""Tests for the top-K forward-sim plan re-score (AE_PLAN_RESCORE).

The re-score re-ranks ONLY the top-K static target candidates by a short
self-plan projection on the known map (items collected en route + whether a
bomb actually lands on a target base + a time tax). It is opponent-light (no
opponent rollout) and default-OFF, so the flag-off path must stay identical to
the legacy planner.
"""

import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "ae" / "src"))

from ae_manager import AEManager


def _open_grid_manager(monkeypatch, *, flag="1", **env):
    """An AEManager on a fully-open 16x16 known grid (no walls)."""
    if flag is None:
        monkeypatch.delenv("AE_PLAN_RESCORE", raising=False)
    else:
        monkeypatch.setenv("AE_PLAN_RESCORE", flag)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    m = AEManager()
    m.seen = {(x, y) for x in range(16) for y in range(16)}
    m.last_step = 1
    m.is_fixed_novice_map = False
    return m


# --- projection arithmetic ------------------------------------------------

def test_projection_sums_items_minus_time_tax(monkeypatch):
    m = _open_grid_manager(monkeypatch)
    m.last_seen_items = {(1, 0): ("mission", 0), (2, 0): ("resource", 0)}
    m.team_bombs = 0
    path = [(0, 0), (1, 0), (2, 0)]
    # mission 5 + resource 2 = 7 ; steps=2 ; tax 0.35*2 = 0.7 -> 6.3
    assert m._project_plan_reward((0, 0), (2, 0), "item_resource", path) == pytest.approx(6.3)


def test_projection_credits_landable_base_full_value(monkeypatch):
    m = _open_grid_manager(monkeypatch)
    m.last_seen_items = {}
    m.team_bombs = 1
    path = [(5, 4), (5, 5), (5, 6)]  # bomb_from = (5,5), base adjacent at (5,6)
    # base value 55 (no shared-credit) ; steps 2 ; tax 0.7 -> 54.3
    assert m._project_plan_reward((5, 4), (5, 6), "enemy_base", path) == pytest.approx(54.3)


def test_projection_demotes_base_when_no_bomb_in_inventory(monkeypatch):
    m = _open_grid_manager(monkeypatch)
    m.last_seen_items = {}
    m.team_bombs = 0  # cannot bomb -> base is a phantom
    path = [(5, 4), (5, 5), (5, 6)]
    # no base credit ; only the time tax -> -0.7
    assert m._project_plan_reward((5, 4), (5, 6), "enemy_base", path) == pytest.approx(-0.7)


# --- top-K override behaviour --------------------------------------------

def _two_candidate_parent():
    # base plan: (5,4)->(5,5)->(5,6) ; item plan: (5,4)->(6,4)->(7,4)
    return {
        (5, 4): None,
        (5, 5): (5, 4),
        (5, 6): (5, 5),
        (6, 4): (5, 4),
        (7, 4): (6, 4),
    }


def test_rescore_overrides_phantom_base_toward_item(monkeypatch):
    m = _open_grid_manager(monkeypatch)
    m.last_seen_items = {(7, 4): ("mission", 0)}
    m.team_bombs = 0  # base unbombable -> phantom
    scored = [(77.7, (5, 6), "enemy_base"), (47.7, (7, 4), "item_mission")]
    target, kind = m._rescore_top_k((5, 4), scored, _two_candidate_parent(),
                                    (5, 6), "enemy_base")
    assert target == (7, 4)
    assert kind == "item_mission"


def test_rescore_keeps_static_winner_when_it_projects_best(monkeypatch):
    m = _open_grid_manager(monkeypatch)
    m.last_seen_items = {(7, 4): ("mission", 0)}
    m.team_bombs = 1  # base lands -> 54.3 beats item 4.3
    scored = [(77.7, (5, 6), "enemy_base"), (47.7, (7, 4), "item_mission")]
    target, kind = m._rescore_top_k((5, 4), scored, _two_candidate_parent(),
                                    (5, 6), "enemy_base")
    assert target == (5, 6)
    assert kind == "enemy_base"


def test_rescore_demote_only_does_not_promote_base_over_static_item(monkeypatch):
    # static winner is an ITEM; a landable base projects higher, but demote-only (default)
    # must KEEP the item -- promoting regresses the tuned heuristic (top_seed_proxy -0.28 smoke)
    m = _open_grid_manager(monkeypatch)
    m.last_seen_items = {(7, 4): ("mission", 0)}
    m.team_bombs = 1  # base lands (projects 54.3) but must not be promoted
    scored = [(50.0, (7, 4), "item_mission"), (45.0, (5, 6), "enemy_base")]
    target, kind = m._rescore_top_k((5, 4), scored, _two_candidate_parent(),
                                    (7, 4), "item_mission")
    assert target == (7, 4)
    assert kind == "item_mission"


def test_rescore_projects_at_most_k_candidates(monkeypatch):
    m = _open_grid_manager(monkeypatch, AE_PLAN_RESCORE_K="2")
    m.last_seen_items = {}
    m.team_bombs = 0
    calls = []
    original = m._project_plan_reward
    m._project_plan_reward = lambda *a, **k: calls.append(1) or original(*a, **k)
    parent = {(0, 0): None}
    scored = []
    for i in range(1, 6):
        parent[(i, 0)] = (i - 1, 0)
        scored.append((100.0 - i, (i, 0), "low_visit"))
    m._rescore_top_k((0, 0), scored, parent, (1, 0), "low_visit")
    assert len(calls) <= 2


# --- flag-off regression guard -------------------------------------------

def _setup_override_scenario(m):
    m.enemy_bases = {(5, 6): 1}
    m.last_seen_items = {(7, 4): ("mission", 0)}
    m.team_bombs = 0  # base is a phantom (no bomb) but still the static winner


def test_flag_on_overrides_phantom_base_target(monkeypatch):
    m = _open_grid_manager(monkeypatch, flag="1")
    _setup_override_scenario(m)
    target, _ = m._choose_target((5, 4), set())
    assert target == (7, 4)


def test_flag_off_keeps_legacy_static_target(monkeypatch):
    m = _open_grid_manager(monkeypatch, flag=None)
    _setup_override_scenario(m)
    target, _ = m._choose_target((5, 4), set())
    assert target == (5, 6)
