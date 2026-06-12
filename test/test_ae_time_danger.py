"""Tests for the time-layered danger map (AE_TIME_DANGER).

Design spec 2026-06-08-ae-time-layered-danger-map-design (private archive).
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


def test_danger_horizon_floored_at_bomb_timer(monkeypatch):
    # horizon must not drop below BOMB_TIMER, else _on_fire_at misjudges arrival-tick==BOMB_TIMER bombs as safe
    monkeypatch.setenv("AE_DANGER_HORIZON", "1")
    m = AEManager()
    assert m.danger_horizon == m.BOMB_TIMER == 3


def test_danger_layers_cache_slot_exists():
    assert AEManager()._danger_layers_cache is None


# --- Task 2: _danger_layers builder + chain resolution ---

def test_single_bomb_lands_in_its_timer_layer():
    m = _open_grid_manager()
    # bomb at (5,5), timer 2 -> detonates at relative tick 2
    m.known_bombs = {(5, 5): {"timer": 2, "own": False, "last_step": 5}}
    layers = m._danger_layers()
    blast = m._blast_cells((5, 5))
    assert layers[2] == blast
    for t, cells in enumerate(layers):
        if t != 2:
            assert cells.isdisjoint(blast)


def test_chain_makes_later_bomb_fire_early():
    m = _open_grid_manager()
    # A(5,5) timer 1; B(5,6) timer 4 sits in A's blast (Chebyshev<=2) -> B detonates at A's tick (1), not 4
    m.known_bombs = {
        (5, 5): {"timer": 1, "own": False, "last_step": 5},
        (5, 6): {"timer": 4, "own": False, "last_step": 5},
    }
    assert (5, 6) in m._blast_cells((5, 5))  # precondition: B is in A's blast
    layers = m._danger_layers()
    blast_b = m._blast_cells((5, 6))
    assert blast_b <= layers[1]
    assert blast_b.isdisjoint(layers[4])


def test_transitive_three_bomb_chain_collapses_to_fixpoint():
    m = _open_grid_manager()
    # A(5,5) t=1 -> B(5,7) in A's blast (dist 2) -> C(5,9) in B's blast; all collapse to tick 1
    m.known_bombs = {
        (5, 5): {"timer": 1, "own": False, "last_step": 5},
        (5, 7): {"timer": 3, "own": False, "last_step": 5},
        (5, 9): {"timer": 5, "own": False, "last_step": 5},
    }
    assert (5, 7) in m._blast_cells((5, 5))
    assert (5, 9) in m._blast_cells((5, 7))
    layers = m._danger_layers()
    assert m._blast_cells((5, 9)) <= layers[1]


def test_on_fire_at_predicate():
    m = _open_grid_manager()
    m.known_bombs = {(5, 5): {"timer": 2, "own": False, "last_step": 5}}
    assert m._on_fire_at((5, 5), 2) is True
    assert m._on_fire_at((5, 5), 1) is False
    assert m._on_fire_at((5, 5), 99) is False  # beyond horizon -> safe
    assert m._on_fire_at((0, 0), 2) is False


# --- Task 3: _danger_cells flag-gated, chain-corrected ---

def test_danger_cells_off_equals_legacy():
    # flag OFF: only bombs with naive timer <= 2 contribute, no chain logic
    m = _open_grid_manager()
    m.known_bombs = {
        (5, 5): {"timer": 2, "own": False, "last_step": 5},
        (10, 10): {"timer": 4, "own": False, "last_step": 5},  # >2, ignored
    }
    assert m._danger_cells() == m._blast_cells((5, 5))


def test_danger_cells_on_equals_off_when_no_chains(monkeypatch):
    monkeypatch.setenv("AE_TIME_DANGER", "1")
    m = _open_grid_manager()
    m.known_bombs = {
        (5, 5): {"timer": 2, "own": False, "last_step": 5},
        (10, 10): {"timer": 4, "own": False, "last_step": 5},
    }
    # no chains -> d == timer, so <=2 slice matches legacy exactly
    assert m._danger_cells() == m._blast_cells((5, 5))


def test_danger_cells_on_includes_chain_corrected_cell(monkeypatch):
    monkeypatch.setenv("AE_TIME_DANGER", "1")
    m = _open_grid_manager()
    # B's naive timer 4 would be ignored by legacy, but A (timer 1) chains it to tick 1
    m.known_bombs = {
        (5, 5): {"timer": 1, "own": False, "last_step": 5},
        (5, 6): {"timer": 4, "own": False, "last_step": 5},
    }
    danger = m._danger_cells()
    assert m._blast_cells((5, 6)) <= danger
    # sanity: legacy (flag OFF) would NOT include B's blast
    m_off = _open_grid_manager()
    m_off.time_danger_enabled = False  # force OFF despite active monkeypatch
    m_off.known_bombs = dict(m.known_bombs)
    assert not (m._blast_cells((5, 6)) <= m_off._danger_cells())


# --- Task 4: _safe_escape_within chain/arrival-aware ---

def test_escape_off_ignores_enemy_bomb(monkeypatch):
    # flag OFF, danger=None: escape only avoids OWN blast, may return a cell in enemy blast
    m = _open_grid_manager()
    own_blast = m._blast_cells((5, 5))
    # enemy bomb's blast doesn't overlap own blast, but sits on a plausible escape cell (8,5) (own-blast edge x<=7)
    m.known_bombs = {(10, 5): {"timer": 1, "own": False, "last_step": 5}}
    cell = m._safe_escape_within((5, 5), own_blast, m.BOMB_TIMER)
    assert cell is not None
    assert cell not in own_blast  # legacy guarantee only


def test_escape_on_avoids_cell_on_fire_at_arrival(monkeypatch):
    # flag ON: a candidate escape cell on fire at its arrival tick (== BFS dist) must be pruned
    monkeypatch.setenv("AE_TIME_DANGER", "1")
    m = _open_grid_manager()
    own_blast = m._blast_cells((5, 5))
    # enemy bomb's blast covers (8,5), fires at tick 3 == arrival dist going right from (5,5) -> (8,5) must be rejected
    m.known_bombs = {(8, 5): {"timer": 3, "own": False, "last_step": 5}}
    cell = m._safe_escape_within((5, 5), own_blast, m.BOMB_TIMER)
    assert cell is not None
    # arrival tick == Manhattan distance on the open grid
    arrival = abs(cell[0] - 5) + abs(cell[1] - 5)
    assert not m._on_fire_at(cell, arrival)


def test_escape_on_returns_none_when_fully_trapped(monkeypatch):
    # flag ON: if every reachable cell within max_moves is on fire at arrival, verifier returns None
    monkeypatch.setenv("AE_TIME_DANGER", "1")
    m = _open_grid_manager()
    own_blast = m._blast_cells((5, 5))
    # saturate every layer 0..horizon with the whole grid -> nothing safe
    full = {(x, y) for x in range(16) for y in range(16)}
    m._danger_layers_cache = [set(full) for _ in range(m.danger_horizon + 1)]
    cell = m._safe_escape_within((5, 5), own_blast, m.BOMB_TIMER)
    assert cell is None


# --- Task 5: flag-OFF byte-identical guard ---

def test_flag_off_danger_and_escape_match_legacy_with_chains():
    # even with a chain present, flag OFF must behave EXACTLY like legacy:
    # _danger_cells ignores timer>2, escape ignores enemy bombs (danger=None)
    m = _open_grid_manager()
    m.known_bombs = {
        (5, 5): {"timer": 1, "own": False, "last_step": 5},
        (5, 6): {"timer": 4, "own": False, "last_step": 5},  # chained when ON
    }
    legacy = set()
    for pos, d in m.known_bombs.items():
        if int(d["timer"]) <= 2:
            legacy.update(m._blast_cells(pos))
    assert m._danger_cells() == legacy
    own_blast = m._blast_cells((5, 5))
    cell = m._safe_escape_within((5, 5), own_blast, m.BOMB_TIMER)
    assert cell is not None and cell not in own_blast
