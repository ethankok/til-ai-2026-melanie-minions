"""Tests for no-self-damage bomb gating (AE_NO_SELF_DAMAGE).

Spec: docs/superpowers/specs/2026-06-09-ae-no-self-damage-bomb-gate-design.md
The env excludes same-team defenders from a bomb's blast, so a bomb never
damages its placer or the placer's own base. OFF == byte-identical.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "ae" / "src"))

from ae_manager import AEManager  # noqa: E402


def _open_grid_manager(step: int = 5):
    m = AEManager()
    m.seen = {(x, y) for x in range(16) for y in range(16)}
    m.grid_size = 16
    m.last_step = step
    m.health = 60
    m.enemy_agents = {}
    m.enemy_bases = []
    m.base_location = None
    return m


# --- Task 1: flag + predicates ---

def test_no_self_damage_default_off():
    assert AEManager().no_self_damage is False


def test_no_self_damage_env_on(monkeypatch):
    monkeypatch.setenv("AE_NO_SELF_DAMAGE", "1")
    assert AEManager().no_self_damage is True


def test_own_base_veto_off_blocks_when_base_in_blast():
    m = _open_grid_manager()  # OFF
    blast = m._blast_cells((8, 8))
    base = next(iter(blast))
    assert m._own_base_vetoes_bomb(base, blast) is True
    assert m._own_base_vetoes_bomb(None, blast) is False
    assert m._own_base_vetoes_bomb((0, 0), blast) is False  # base not in blast


def test_own_base_veto_on_never_blocks(monkeypatch):
    monkeypatch.setenv("AE_NO_SELF_DAMAGE", "1")
    m = _open_grid_manager()
    blast = m._blast_cells((8, 8))
    base = next(iter(blast))
    assert m._own_base_vetoes_bomb(base, blast) is False


def test_escape_required_off_true_on_false(monkeypatch):
    assert _open_grid_manager()._escape_required_for_bomb() is True
    monkeypatch.setenv("AE_NO_SELF_DAMAGE", "1")
    assert AEManager()._escape_required_for_bomb() is False


def test_env_placer_takes_no_self_damage():
    """Ground truth: a placer STAYing on its own bomb tile takes no damage.

    Drives the real til_environment through detonation and asserts health is
    unchanged. Pins the premise the whole lever rests on.
    """
    import numpy as np
    from til_environment import bomberman_env
    from til_environment.config import default_config
    from til_environment.entities.dynamic import Bomb

    env = bomberman_env.basic_env(env_wrappers=[], cfg=default_config())
    env.reset(seed=42)
    dyn = env.unwrapped.dynamics
    us = env.possible_agents[0]
    placed = False
    healths = []
    fired = False
    for agent in env.agent_iter(max_iter=6000):
        obs, _r, term, trunc, _i = env.last()
        if term or trunc:
            env.step(None)
            continue
        if agent != us:
            env.step(4)  # STAY
            continue
        if placed:
            healths.append(float(np.ravel(np.asarray(obs["health"]))[0]))
            if len(dyn.registry.query().type(Bomb).all()) == 0 or \
                    len(getattr(dyn, "last_explosions", []) or []) > 0:
                fired = True
                break
        mask = [int(x) for x in obs["action_mask"]]
        if not placed and mask[5] == 1:
            env.step(5)  # PLACE_BOMB
            placed = True
        else:
            env.step(4)  # STAY on the bomb tile
    assert placed and fired
    assert healths and min(healths) >= 60.0  # zero self-damage


# --- Task 2: _should_place_bomb ---

def _spb_obs():
    return {"action_mask": [1] * 6, "team_bombs": 1}


def test_spb_off_vetoes_own_base_in_blast():
    m = _open_grid_manager()
    loc = (8, 8)
    m.enemy_bases = [(9, 8)]          # tactical target in blast
    m.base_location = (8, 8)          # OWN base in blast -> legacy veto
    assert m._should_place_bomb(_spb_obs(), loc, None, set()) is False


def test_spb_on_places_with_own_base_in_blast(monkeypatch):
    monkeypatch.setenv("AE_NO_SELF_DAMAGE", "1")
    m = _open_grid_manager()
    loc = (8, 8)
    m.enemy_bases = [(9, 8)]
    m.base_location = (8, 8)
    assert m._should_place_bomb(_spb_obs(), loc, None, set()) is True


def test_spb_off_vetoes_when_trapped():
    m = _open_grid_manager()
    loc = (8, 8)
    m.enemy_bases = [(9, 8)]
    m.seen = set(m._blast_cells(loc))   # no cell outside blast -> no escape
    assert m._should_place_bomb(_spb_obs(), loc, None, set()) is False


def test_spb_on_places_when_trapped(monkeypatch):
    monkeypatch.setenv("AE_NO_SELF_DAMAGE", "1")
    m = _open_grid_manager()
    loc = (8, 8)
    m.enemy_bases = [(9, 8)]
    m.seen = set(m._blast_cells(loc))
    assert m._should_place_bomb(_spb_obs(), loc, None, set()) is True


def test_spb_on_still_vetoes_in_enemy_danger(monkeypatch):
    monkeypatch.setenv("AE_NO_SELF_DAMAGE", "1")
    m = _open_grid_manager()
    loc = (8, 8)
    m.enemy_bases = [(9, 8)]
    assert m._should_place_bomb(_spb_obs(), loc, None, {loc}) is False  # location in danger


def test_spb_on_still_vetoes_low_health(monkeypatch):
    monkeypatch.setenv("AE_NO_SELF_DAMAGE", "1")
    m = _open_grid_manager()
    m.health = 1                        # below LOW_HEALTH_THRESHOLD
    loc = (8, 8)
    m.enemy_bases = [(9, 8)]
    assert m._should_place_bomb(_spb_obs(), loc, None, set()) is False


# --- Task 3: dominant-action shortcut ---

def test_dominant_bomb_on_places_when_trapped(monkeypatch):
    monkeypatch.setenv("AE_NO_SELF_DAMAGE", "1")
    m = _open_grid_manager()
    loc, step = (8, 8), 5
    m.last_step = step
    m.enemy_agents = {(9, 8): step}          # fresh sighting in blast
    m.seen = set(m._blast_cells(loc))         # trapped -> no escape
    obs = {"action_mask": [1] * 6, "team_bombs": 1}
    # direction arg: any legal facing; danger empty; low_health False.
    # Note: method is _try_dominant_action (returns int | None; None = no dominant move)
    action = m._try_dominant_action(obs, loc, m.FORWARD, set(), False)
    assert action == m.PLACE_BOMB


def test_dominant_bomb_off_skips_when_trapped():
    m = _open_grid_manager()
    loc, step = (8, 8), 5
    m.last_step = step
    m.enemy_agents = {(9, 8): step}
    m.seen = set(m._blast_cells(loc))
    obs = {"action_mask": [1] * 6, "team_bombs": 1}
    action = m._try_dominant_action(obs, loc, m.FORWARD, set(), False)
    assert action != m.PLACE_BOMB


# --- Task 4: planner / MCTS paths ---

def test_plan_path_value_on_credits_base_without_escape(monkeypatch):
    target = (8, 8)
    path = [(8, 10), (8, 9), (8, 8)]     # bomb_from = path[-2] = (8,9), base in blast
    monkeypatch.delenv("AE_NO_SELF_DAMAGE", raising=False)
    off = _open_grid_manager(); off.enemy_bases = [target]; off.team_bombs = 1
    off.seen = set(off._blast_cells((8, 9)))
    monkeypatch.setenv("AE_NO_SELF_DAMAGE", "1")
    on = _open_grid_manager(); on.enemy_bases = [target]; on.team_bombs = 1
    on.seen = set(on._blast_cells((8, 9)))
    v_off = off._project_plan_reward((8, 10), target, "enemy_base", path)
    v_on = on._project_plan_reward((8, 10), target, "enemy_base", path)
    assert v_on > v_off   # ON credits the base value despite no escape


# --- Task 5: OFF byte-identical guard ---

def test_off_is_legacy_even_with_base_in_blast_and_trapped():
    """Flag OFF must behave exactly like legacy: own base in blast OR no escape
    both veto placement, even with a real tactical target present."""
    m = _open_grid_manager()  # OFF
    loc = (8, 8)
    m.enemy_bases = [(9, 8)]
    m.base_location = (8, 8)            # base-in-blast veto
    assert m._should_place_bomb(_spb_obs(), loc, None, set()) is False
    m2 = _open_grid_manager()
    m2.enemy_bases = [(9, 8)]
    m2.seen = set(m2._blast_cells(loc))  # escape veto
    assert m2._should_place_bomb(_spb_obs(), loc, None, set()) is False
