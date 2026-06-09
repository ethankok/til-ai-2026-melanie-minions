"""Tests for the time-layered danger map (AE_TIME_DANGER).

Spec: docs/superpowers/specs/2026-06-08-ae-time-layered-danger-map-design.md
Foundation lever #2: per-tick lethality layers with enemy-bomb chain
resolution, wired into the danger set (chain-corrected) and the escape
verifier (chain/arrival-aware). Flag OFF == byte-identical.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "ae" / "src"))

from ae_manager import AEManager  # noqa: E402


def _open_grid_manager(step: int = 5):
    """A manager on a fully-seen open 16x16 grid with no opponents in view."""
    m = AEManager()
    m.seen = {(x, y) for x in range(16) for y in range(16)}
    m.grid_size = 16
    m.last_step = step
    m.health = 60
    m.enemy_agents = {}
    m.enemy_bases = []
    m.base_location = None
    return m


# --- Task 1: flag + cache plumbing ---

def test_time_danger_default_off():
    assert AEManager().time_danger_enabled is False


def test_time_danger_env_on(monkeypatch):
    monkeypatch.setenv("AE_TIME_DANGER", "1")
    assert AEManager().time_danger_enabled is True


def test_danger_horizon_default_six():
    assert AEManager().danger_horizon == 6


def test_danger_horizon_env_override(monkeypatch):
    monkeypatch.setenv("AE_DANGER_HORIZON", "8")
    assert AEManager().danger_horizon == 8


def test_danger_layers_cache_slot_exists():
    assert AEManager()._danger_layers_cache is None
