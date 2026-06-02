"""Tests for AE_CONTENTION contention-aware item valuation.

Demote-only, item-vs-item: discount item targets an opponent reaches first,
using free fixed-map spawn geometry + live viewcone sightings. Default-OFF, so
the flag-off path must stay identical to the legacy planner.
"""

import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "ae" / "src"))

from ae_manager import AEManager


def _open_grid_manager(monkeypatch, *, flag="1", **env):
    """An AEManager on a fully-open 16x16 known grid (no walls)."""
    if flag is None:
        monkeypatch.delenv("AE_CONTENTION", raising=False)
    else:
        monkeypatch.setenv("AE_CONTENTION", flag)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    m = AEManager()
    m.seen = {(x, y) for x in range(16) for y in range(16)}
    m.grid_size = 16
    m.last_step = 1
    m.is_fixed_novice_map = False
    m.ENEMY_CHASE_VALUE = 0.0  # isolate item contention from chase candidates
    return m


def test_flags_default_off_and_env_override(monkeypatch):
    monkeypatch.delenv("AE_CONTENTION", raising=False)
    m = AEManager()
    assert m.contention_enabled is False
    assert m.contention_scale == pytest.approx(2.5)
    assert m.contention_pfloor == pytest.approx(0.15)
    assert m.contention_topen == 40
    assert m.contention_tfresh == 3

    monkeypatch.setenv("AE_CONTENTION", "1")
    monkeypatch.setenv("AE_CONTENTION_SCALE", "3.0")
    monkeypatch.setenv("AE_CONTENTION_PFLOOR", "0.2")
    monkeypatch.setenv("AE_CONTENTION_TOPEN", "60")
    monkeypatch.setenv("AE_CONTENTION_TFRESH", "5")
    m2 = AEManager()
    assert m2.contention_enabled is True
    assert m2.contention_scale == pytest.approx(3.0)
    assert m2.contention_pfloor == pytest.approx(0.2)
    assert m2.contention_topen == 60
    assert m2.contention_tfresh == 5


def test_is_item_kind(monkeypatch):
    m = _open_grid_manager(monkeypatch)
    assert m._is_item_kind("item_mission") is True
    assert m._is_item_kind("respawn_resource") is True
    assert m._is_item_kind("enemy_base") is False
    assert m._is_item_kind("low_visit") is False


def test_believed_opponents_live_sightings_fresh_only(monkeypatch):
    m = _open_grid_manager(monkeypatch)
    m.enemy_agents = {(1, 0): 5, (8, 8): 1}  # last_seen steps
    # step 6: (1,0) stale=1 (<=3 kept); (8,8) stale=5 (>3 dropped)
    opps = m._believed_opponents(6)
    assert (1, 0) in opps
    assert (8, 8) not in opps


def test_believed_opponents_excludes_own_spawn_on_fixed_map(monkeypatch):
    m = _open_grid_manager(monkeypatch)
    m.is_fixed_novice_map = True
    m.base_location = (13, 9)  # BASE_LOCATIONS[0] -> spawn STARTING_LOCATIONS[0]=(14,9)
    m.enemy_agents = {}
    opps = m._believed_opponents(0)
    assert (14, 9) not in opps  # our own spawn excluded
    assert (9, 14) in opps      # slot 1 spawn included
    assert len(opps) == 5


def test_believed_opponents_no_spawn_seed_off_fixed_map(monkeypatch):
    m = _open_grid_manager(monkeypatch)
    m.is_fixed_novice_map = False  # not the fixed novice map
    m.base_location = (13, 9)
    m.enemy_agents = {}
    assert m._believed_opponents(0) == []


def test_believed_opponents_spawn_seed_expires_after_topen(monkeypatch):
    m = _open_grid_manager(monkeypatch)
    m.is_fixed_novice_map = True
    m.base_location = (13, 9)
    m.enemy_agents = {}
    assert m._believed_opponents(41) == []  # step > TOPEN (40) -> no spawn seed
