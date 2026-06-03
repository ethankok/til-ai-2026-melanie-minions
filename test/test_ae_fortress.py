"""Tests for AE_FORTRESS posture + the experimental-flag de-contamination
(both AE_STUN_TAX and AE_FORTRESS must be forced OFF on opponent proxies)."""

import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "ae" / "src"))
sys.path.append(str(Path(__file__).resolve().parents[1] / "training" / "ae"))

from ae_manager import AEManager


def test_self_heuristic_opponent_forces_experimental_flags_off(monkeypatch):
    monkeypatch.setenv("AE_STUN_TAX", "1")
    monkeypatch.setenv("AE_FORTRESS", "1")
    from foreign_opponents import SelfHeuristicOpponent
    op = SelfHeuristicOpponent()
    assert op._mgr.stun_tax_enabled is False
    assert op._mgr.fortress_enabled is False


def _open_grid_fortress(monkeypatch, *, flag="1", **env):
    if flag is None:
        monkeypatch.delenv("AE_FORTRESS", raising=False)
    else:
        monkeypatch.setenv("AE_FORTRESS", flag)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    m = AEManager()
    m.seen = {(x, y) for x in range(16) for y in range(16)}
    m.grid_size = 16
    m.is_fixed_novice_map = False
    m.ENEMY_CHASE_VALUE = 0.0
    return m


def test_fortress_flags_default_off_and_override(monkeypatch):
    monkeypatch.delenv("AE_FORTRESS", raising=False)
    m = AEManager()
    assert m.fortress_enabled is False
    assert m.fortress_phase == pytest.approx(0.6)
    assert m.fortress_base_mult == pytest.approx(0.5)
    assert m.fortress_defense_mult == pytest.approx(1.5)
    assert m.fortress_threat_mult == pytest.approx(1.5)
    assert m.fortress_tether_w == pytest.approx(0.3)
    assert m.MATCH_STEPS == 200


def test_posture_farm_when_flag_off(monkeypatch):
    m = _open_grid_fortress(monkeypatch, flag=None)
    m.base_location = None
    m.enemy_agents = {}
    assert m._posture(199) == "farm"  # flag off -> always farm


def test_posture_fortress_late_phase(monkeypatch):
    m = _open_grid_fortress(monkeypatch, flag="1")
    m.base_location = None
    m.enemy_agents = {}
    assert m._posture(119) == "farm"       # 119 < 0.6*200
    assert m._posture(120) == "fortress"   # 120 >= 0.6*200


def test_posture_fortress_on_base_threat(monkeypatch):
    m = _open_grid_fortress(monkeypatch, flag="1")
    m.base_location = (5, 5)
    m.enemy_agents = {(6, 6): 0}  # within BASE_DEFENSE_RADIUS, fresh
    assert m._posture(1) == "fortress"  # early, but base is threatened


def test_fortress_demotes_enemy_base_to_item(monkeypatch):
    # enemy_base (3,0) value=100 -> farm score 100-1.15*3 = 96.55 (wins).
    # fortress base_mult=0.5 -> 50-3.45 = 46.55 ; item (0,2) = 50-2.3 = 47.70 -> item wins.
    m = _open_grid_fortress(monkeypatch, flag="1")
    m.ENEMY_BASE_VALUE = 100.0
    m.tier1_shared_credit = False
    m.team_bombs = 1
    m.base_location = None  # isolate base_mult from the tether
    m.enemy_bases = {(3, 0)}
    m.last_seen_items = {(0, 2): ("mission", 0)}
    m.enemy_agents = {}
    m.last_step = 130  # late phase -> fortress
    target, _ = m._choose_target((0, 0), set())
    assert target == (0, 2)


def test_flag_off_keeps_enemy_base(monkeypatch):
    m = _open_grid_fortress(monkeypatch, flag=None)
    m.ENEMY_BASE_VALUE = 100.0
    m.tier1_shared_credit = False
    m.team_bombs = 1
    m.base_location = None
    m.enemy_bases = {(3, 0)}
    m.last_seen_items = {(0, 2): ("mission", 0)}
    m.enemy_agents = {}
    m.last_step = 130
    target, _ = m._choose_target((0, 0), set())
    assert target == (3, 0)  # farm: base (96.55) beats item (47.70)


def test_fortress_tether_reduces_far_target_score(monkeypatch):
    # Single item (0,5), base (0,0): manhattan 5. Tether (w=0.3) lowers top_score
    # by 0.3*5 = 1.5 vs farm. No threats -> threat_mult irrelevant.
    def top(flag):
        m = _open_grid_fortress(monkeypatch, flag=flag)
        m.base_location = (0, 0)
        m.enemy_bases = set()
        m.enemy_agents = {}
        m.last_seen_items = {(0, 5): ("mission", 0)}
        m.last_step = 130
        m._choose_target((0, 0), set())
        return m.last_decision_confidence["top_score"]

    assert top("1") == pytest.approx(top(None) - 0.3 * 5)
