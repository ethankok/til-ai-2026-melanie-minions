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
