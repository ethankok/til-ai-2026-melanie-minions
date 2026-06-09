"""Tests for no-self-damage bomb gating (AE_NO_SELF_DAMAGE).

Spec: docs/superpowers/specs/2026-06-09-ae-no-self-damage-bomb-gate-design.md
The env excludes same-team defenders from a bomb's blast, so a bomb never
damages its placer or the placer's own base. OFF == byte-identical.
"""

import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "ae" / "src"))

from ae_manager import AEManager, _LookaheadState  # noqa: E402


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
    pytest.importorskip("til_environment")  # skip where the submodule isn't installed
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


# --- Task 4 (follow-up): MCTS _lookahead_legal_actions / _lookahead_step ---
# Real tests for B4/E5 and B5/E6 (the spec-review gap: these dormant sites had
# code edits but no focused tests). Each exercises the trapped (escape) veto and
# the own-base-in-blast veto, asserting ON admits PLACE_BOMB and OFF rejects it.

def _bomb_state(m, pos=(8, 8)):
    return _LookaheadState(
        pos=pos, direction=0, bombs=(), bombs_left=1, health=60,
        collected=frozenset(), score=0.0, first_action=None,
        tactical=False, path=(),
    )


def test_lookahead_legal_escape_veto_off_excludes_on_includes(monkeypatch):
    loc = (8, 8)
    off = _open_grid_manager()
    off.base_location = None
    off.seen = set(off._blast_cells(loc))          # trapped -> no escape
    assert off.PLACE_BOMB not in off._lookahead_legal_actions(loc, 0, 1)
    monkeypatch.setenv("AE_NO_SELF_DAMAGE", "1")
    on = _open_grid_manager()
    on.base_location = None
    on.seen = set(on._blast_cells(loc))
    assert on.PLACE_BOMB in on._lookahead_legal_actions(loc, 0, 1)


def test_lookahead_legal_base_veto_off_excludes_on_includes(monkeypatch):
    loc = (8, 8)
    off = _open_grid_manager()                      # open grid -> escape exists
    off.base_location = (9, 8)                       # OWN base in blast
    assert off.PLACE_BOMB not in off._lookahead_legal_actions(loc, 0, 1)
    monkeypatch.setenv("AE_NO_SELF_DAMAGE", "1")
    on = _open_grid_manager()
    on.base_location = (9, 8)
    assert on.PLACE_BOMB in on._lookahead_legal_actions(loc, 0, 1)


def test_lookahead_step_escape_veto_off_none_on_places(monkeypatch):
    loc = (8, 8)
    off = _open_grid_manager()
    off.base_location = None
    off.seen = set(off._blast_cells(loc))           # trapped
    assert off._lookahead_step(_bomb_state(off, loc), off.PLACE_BOMB) is None
    monkeypatch.setenv("AE_NO_SELF_DAMAGE", "1")
    on = _open_grid_manager()
    on.base_location = None
    on.seen = set(on._blast_cells(loc))
    assert on._lookahead_step(_bomb_state(on, loc), on.PLACE_BOMB) is not None


def test_lookahead_step_base_veto_off_none_on_places(monkeypatch):
    loc = (8, 8)
    off = _open_grid_manager()                      # escape exists
    off.base_location = (9, 8)                       # OWN base in blast
    assert off._lookahead_step(_bomb_state(off, loc), off.PLACE_BOMB) is None
    monkeypatch.setenv("AE_NO_SELF_DAMAGE", "1")
    on = _open_grid_manager()
    on.base_location = (9, 8)
    assert on._lookahead_step(_bomb_state(on, loc), on.PLACE_BOMB) is not None


# --- basekill-noescape: flag + predicates ---

def test_basekill_default_off():
    assert AEManager().basekill_noescape is False


def test_basekill_env_on(monkeypatch):
    monkeypatch.setenv("AE_BASEKILL_NOESCAPE", "1")
    assert AEManager().basekill_noescape is True


def test_bomb_hits_enemy_base():
    m = _open_grid_manager()
    blast = m._blast_cells((8, 8))
    m.enemy_bases = [(9, 8)]
    assert m._bomb_hits_enemy_base(blast) is True
    m.enemy_bases = [(0, 0)]
    assert m._bomb_hits_enemy_base(blast) is False


def test_basekill_escape_required_only_for_non_base(monkeypatch):
    monkeypatch.setenv("AE_BASEKILL_NOESCAPE", "1")
    m = _open_grid_manager()
    blast = m._blast_cells((8, 8))
    m.enemy_bases = [(9, 8)]                       # base in blast
    assert m._escape_required_for_bomb(blast) is False   # base kill -> not required
    m.enemy_bases = [(0, 0)]                       # no base in blast
    assert m._escape_required_for_bomb(blast) is True    # speculative -> still required


def test_basekill_own_base_veto_relaxed_only_for_base_kill(monkeypatch):
    monkeypatch.setenv("AE_BASEKILL_NOESCAPE", "1")
    m = _open_grid_manager()
    blast = m._blast_cells((8, 8))
    base = (8, 8)                                  # own base in blast
    m.enemy_bases = [(9, 8)]                       # AND an enemy base in blast
    assert m._own_base_vetoes_bomb(base, blast) is False  # relaxed for the kill
    m.enemy_bases = [(0, 0)]                       # no enemy base in blast
    assert m._own_base_vetoes_bomb(base, blast) is True   # speculative -> still vetoes


def test_no_self_damage_takes_precedence_over_basekill(monkeypatch):
    monkeypatch.setenv("AE_NO_SELF_DAMAGE", "1")
    monkeypatch.setenv("AE_BASEKILL_NOESCAPE", "1")
    m = _open_grid_manager()
    blast = m._blast_cells((8, 8))
    m.enemy_bases = [(0, 0)]                       # no base in blast
    assert m._escape_required_for_bomb(blast) is False    # full relaxation wins
    assert m._own_base_vetoes_bomb((8, 8), blast) is False


# --- basekill-noescape: behaviour ---

def test_spb_basekill_places_base_kill_when_trapped(monkeypatch):
    monkeypatch.setenv("AE_BASEKILL_NOESCAPE", "1")
    m = _open_grid_manager()
    loc = (8, 8)
    m.enemy_bases = [(9, 8)]                       # enemy BASE in blast
    m.seen = set(m._blast_cells(loc))               # trapped
    assert m._should_place_bomb(_spb_obs(), loc, None, set()) is True


def test_spb_basekill_vetoes_speculative_when_trapped(monkeypatch):
    monkeypatch.setenv("AE_BASEKILL_NOESCAPE", "1")
    m = _open_grid_manager()
    loc, step = (8, 8), 5
    m.last_step = step
    m.enemy_bases = []                              # NO enemy base
    m.enemy_agents = {(9, 8): step}                 # only a speculative agent target
    m.seen = set(m._blast_cells(loc))               # trapped
    # speculative bomb still requires escape -> vetoed
    assert m._should_place_bomb(_spb_obs(), loc, None, set()) is False


def test_spb_basekill_places_with_own_base_in_blast_for_kill(monkeypatch):
    monkeypatch.setenv("AE_BASEKILL_NOESCAPE", "1")
    m = _open_grid_manager()
    loc = (8, 8)
    m.enemy_bases = [(9, 8)]                         # enemy base in blast
    m.base_location = (8, 8)                         # own base in blast too
    assert m._should_place_bomb(_spb_obs(), loc, None, set()) is True


def test_spb_basekill_still_vetoes_in_enemy_danger(monkeypatch):
    monkeypatch.setenv("AE_BASEKILL_NOESCAPE", "1")
    m = _open_grid_manager()
    loc = (8, 8)
    m.enemy_bases = [(9, 8)]
    assert m._should_place_bomb(_spb_obs(), loc, None, {loc}) is False  # real guard intact


def test_plan_reward_basekill_credits_base_without_escape(monkeypatch):
    target = (8, 8)
    path = [(8, 10), (8, 9), (8, 8)]
    monkeypatch.delenv("AE_BASEKILL_NOESCAPE", raising=False)
    off = _open_grid_manager(); off.enemy_bases = [target]; off.team_bombs = 1
    off.seen = set(off._blast_cells((8, 9)))
    monkeypatch.setenv("AE_BASEKILL_NOESCAPE", "1")
    on = _open_grid_manager(); on.enemy_bases = [target]; on.team_bombs = 1
    on.seen = set(on._blast_cells((8, 9)))
    v_off = off._project_plan_reward((8, 10), target, "enemy_base", path)
    v_on = on._project_plan_reward((8, 10), target, "enemy_base", path)
    assert v_on > v_off


def test_basekill_off_is_legacy():
    """Both flags OFF: enemy base in blast does NOT relax the vetoes."""
    m = _open_grid_manager()
    loc = (8, 8)
    m.enemy_bases = [(9, 8)]
    m.seen = set(m._blast_cells(loc))               # trapped
    assert m._should_place_bomb(_spb_obs(), loc, None, set()) is False
    m2 = _open_grid_manager()
    m2.enemy_bases = [(9, 8)]
    m2.base_location = (8, 8)                        # own base in blast
    # Legacy: own-base veto fires -> False (verified by running with both flags unset)
    assert m2._should_place_bomb(_spb_obs(), loc, None, set()) is False
