"""Tests for AE_STUN_TAX: a freeze opportunity-cost penalty on farming paths.

Default-OFF; flag-off (and MULT=1.0) must stay identical to the legacy planner.
The tax scales the existing path-threat penalty for ITEM targets only.
"""

import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "ae" / "src"))

from ae_manager import AEManager


def _open_grid_manager(monkeypatch, *, flag="1", **env):
    """An AEManager on a fully-open 16x16 known grid (no walls)."""
    if flag is None:
        monkeypatch.delenv("AE_STUN_TAX", raising=False)
    else:
        monkeypatch.setenv("AE_STUN_TAX", flag)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    m = AEManager()
    m.seen = {(x, y) for x in range(16) for y in range(16)}
    m.grid_size = 16
    m.last_step = 1
    m.is_fixed_novice_map = False
    m.ENEMY_CHASE_VALUE = 0.0
    return m


def test_flags_default_off_and_env_override(monkeypatch):
    monkeypatch.delenv("AE_STUN_TAX", raising=False)
    m = AEManager()
    assert m.stun_tax_enabled is False
    assert m.stun_tax_mult == pytest.approx(2.0)

    monkeypatch.setenv("AE_STUN_TAX", "1")
    monkeypatch.setenv("AE_STUN_TAX_MULT", "3.5")
    m2 = AEManager()
    assert m2.stun_tax_enabled is True
    assert m2.stun_tax_mult == pytest.approx(3.5)


def _two_item_threat_scenario(m):
    """Item A=(2,0) is closer (static winner) but its forced path cell (1,0) is
    a threat; item B=(0,5) is farther but threat-free. mission value=50,
    DIST_PENALTY=1.15, PATH_THREAT_PENALTY=2.0."""
    m.last_seen_items = {(2, 0): ("mission", 0), (0, 5): ("mission", 0)}
    m.enemy_agents = {}
    m.enemy_bases = set()
    m._enemy_threat_cells = lambda: {(1, 0)}


def test_flag_off_keeps_closer_threatened_item(monkeypatch):
    # A = 50 - 1.15*2 - 2.0*1 = 45.70 ; B = 50 - 1.15*5 = 44.25 -> A wins.
    m = _open_grid_manager(monkeypatch, flag=None)
    _two_item_threat_scenario(m)
    target, _ = m._choose_target((0, 0), set())
    assert target == (2, 0)


def test_mult_one_is_byte_identical(monkeypatch):
    m = _open_grid_manager(monkeypatch, flag="1", AE_STUN_TAX_MULT="1.0")
    _two_item_threat_scenario(m)
    target, _ = m._choose_target((0, 0), set())
    assert target == (2, 0)


def test_stun_tax_avoids_threatened_item(monkeypatch):
    # MULT=2.0: A = 50 - 1.15*2 - (2.0*2.0)*1 = 43.70 ; B = 44.25 -> B wins.
    m = _open_grid_manager(monkeypatch, flag="1", AE_STUN_TAX_MULT="2.0")
    _two_item_threat_scenario(m)
    target, _ = m._choose_target((0, 0), set())
    assert target == (0, 5)
