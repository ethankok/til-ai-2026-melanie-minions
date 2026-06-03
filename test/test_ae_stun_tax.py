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
