"""Tests for the BombSafety module through its own interface.

BombSafety (ae/src/bomb_safety.py) owns blast geometry, chain-resolved
danger, placement vetoes, escape search and commit/rollback, operating on
the host AEManager's belief blackboard. These tests cross the BombSafety
seam directly; the legacy ``AEManager._*`` delegators are covered by the
older test files (test_ae_time_danger, test_ae_no_self_damage,
test_ae_confpol_rollback).
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "ae" / "src"))

from ae_manager import AEManager  # noqa: E402
from bomb_safety import BombSafety  # noqa: E402


def _open_grid_manager(step: int = 5) -> AEManager:
    m = AEManager()
    m.seen = {(x, y) for x in range(16) for y in range(16)}
    m.grid_size = 16
    m.last_step = step
    m.health = 60
    m.enemy_agents = {}
    m.enemy_bases = []
    m.base_location = None
    return m


def test_manager_constructs_bomb_safety():
    m = AEManager()
    assert isinstance(m.bomb_safety, BombSafety)


def test_blast_cells_matches_delegator_and_caches():
    m = _open_grid_manager()
    bs = m.bomb_safety
    blast = bs.blast_cells((8, 8))
    assert blast == m._blast_cells((8, 8))
    assert (8, 8) in blast
    assert all(max(abs(x - 8), abs(y - 8)) <= m.BOMB_RADIUS for x, y in blast)
    # cache lives on the host blackboard, shared with the delegator
    assert (8, 8) in m._blast_cache


def test_danger_cells_includes_only_imminent_bombs():
    m = _open_grid_manager()
    bs = m.bomb_safety
    m.known_bombs = {
        (4, 4): {"timer": 1, "own": False, "last_step": 5},
        (12, 12): {"timer": m.danger_horizon + 5, "own": False, "last_step": 5},
    }
    danger = bs.danger_cells()
    assert (4, 4) in danger
    assert (12, 12) not in danger


def test_danger_layers_resolve_chain_to_earliest_trigger():
    m = _open_grid_manager()
    m.time_danger_enabled = True
    bs = m.bomb_safety
    # B sits inside A's blast; A fires at t=1, so B chains to t=1 despite its own timer=3
    m.known_bombs = {
        (8, 8): {"timer": 1, "own": False, "last_step": 5},
        (8, 9): {"timer": 3, "own": False, "last_step": 5},
    }
    layers = bs.danger_layers()
    assert (8, 9) in layers[1]
    assert bs.on_fire_at((8, 9), 1)
    assert not bs.on_fire_at((8, 9), len(layers) + 1)  # beyond horizon -> safe


def test_safe_escape_within_escapes_blast():
    m = _open_grid_manager()
    bs = m.bomb_safety
    blast = bs.blast_cells((8, 8))
    target = bs.safe_escape_within((8, 8), blast, m.BOMB_TIMER)
    assert target is not None
    assert target not in blast


def test_commit_and_revert_round_trip():
    m = _open_grid_manager()
    bs = m.bomb_safety
    assert (8, 8) not in m.known_bombs
    prior_escape = m.escape_target

    bs.commit_bomb((8, 8), (8, 11))
    assert m.known_bombs[(8, 8)]["own"] is True
    assert m.escape_target == (8, 11)
    assert m._tick_bomb_commit is not None

    assert bs.revert_commit() is True
    assert (8, 8) not in m.known_bombs
    assert m.escape_target == prior_escape
    assert m._tick_bomb_commit is None
    assert bs.revert_commit() is False  # second revert is a no-op


def test_escape_required_respects_no_self_damage(monkeypatch):
    assert _open_grid_manager().bomb_safety.escape_required() is True
    monkeypatch.setenv("AE_NO_SELF_DAMAGE", "1")
    assert _open_grid_manager().bomb_safety.escape_required() is False
