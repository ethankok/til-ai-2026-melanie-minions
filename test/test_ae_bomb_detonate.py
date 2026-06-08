"""Tests for the offensive bomb-detonation-timing split (AE_BOMB_DETONATE_STEPS).

Background (probe `training/ae/probe_bomb_timer.py`, 6-7 Jun 2026): a freshly
placed bomb actually detonates ~5 of OUR decision-steps after placement (dataclass
`timer=4` + `Bomb.__post_init__` `+1`), but the heuristic hardcoded a single
`BOMB_TIMER = 3` used for BOTH the own-bomb escape window (conservative, safe --
the placer takes zero self-damage on its own tile anyway) AND the *offensive*
"when does my bomb land" reasoning (respawn-camp credit), where 3 UNDER-estimates
the true detonation step by ~2.

Fix: keep `BOMB_TIMER = 3` for escape/survival, add a separate
`BOMB_DETONATE_STEPS` (default 5, env `AE_BOMB_DETONATE_STEPS`) used ONLY for the
offensive `detonation_step = step + ...` window checks. The repeat_kill predictive
bomb now fires when the kill-cell enemy unfreezes at the TRUE detonation step.
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


# --------------------------------------------------------------------------
# Constant: split exists, escape window unchanged, env-overridable
# --------------------------------------------------------------------------
def test_detonate_steps_default_is_five():
    m = AEManager()
    assert m.BOMB_DETONATE_STEPS == 5


def test_escape_bomb_timer_unchanged():
    # The conservative escape/survival window must stay at 3.
    assert AEManager().BOMB_TIMER == 3


def test_detonate_steps_env_override(monkeypatch):
    monkeypatch.setenv("AE_BOMB_DETONATE_STEPS", "6")
    assert AEManager().BOMB_DETONATE_STEPS == 6


# --------------------------------------------------------------------------
# Behaviour: respawn-camp (repeat_kill) bomb fires at the TRUE detonation step
# --------------------------------------------------------------------------
def test_repeat_kill_fires_at_true_detonation():
    """Enemy unfreezes at step+5 (the real detonation) -> bomb should fire.

    Old code (window = step+BOMB_TIMER=step+3) would MISS this (|3-5|=2 > 1).
    """
    step = 5
    m = _open_grid_manager(step=step)
    loc = (8, 8)
    # kill-cell == our cell -> always inside the bomb's own blast.
    m.recent_kills = [(loc, step + m.BOMB_DETONATE_STEPS)]  # unfreeze at +5
    obs = {"action_mask": [1] * 6, "team_bombs": 1}

    assert m._should_place_bomb(obs, loc, None, set()) is True
    assert m.last_bomb_reason == "repeat_kill"


def test_repeat_kill_does_not_fire_at_stale_three_step_window():
    """Enemy unfreezes at step+3 -> with the corrected +5 window this is now a
    miss (|5-3|=2 > 1), so no other tactical target => no bomb.

    Old code would have (wrongly) fired here, so this guards the window move.
    """
    step = 5
    m = _open_grid_manager(step=step)
    loc = (8, 8)
    m.recent_kills = [(loc, step + 3)]  # the stale BOMB_TIMER alignment
    obs = {"action_mask": [1] * 6, "team_bombs": 1}

    assert m._should_place_bomb(obs, loc, None, set()) is False
