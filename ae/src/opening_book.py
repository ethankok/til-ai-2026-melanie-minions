"""Hand-crafted opening-book biases for the fixed Novice map.

STATUS (24 May 2026): NEGATIVE RESULT — not wired into the shipping
path. See ae/NOTES.md "opening-book-v1 experiment" section for the
full write-up. Brief: opening book mechanically works (3x more
enemy-base destructions, ~20 bombs/round vs baseline), but the increased
aggression's gain in low-pressure suites (library +0.026) is exactly
cancelled by its loss in cloudsuite (-0.025). Net aggregate at 24x3
gate: -0.002 vs baseline. Same failure mode as ally-bomb-safe-v2 and
dypm-veto-v1: the cloud evaluator's pressure distribution penalizes
uniform aggression boosts.

Module kept in tree as reusable map-analysis primitive (Dijkstra +
line-of-sight helpers) and reference for the negative result.

To re-enable for ad-hoc experimentation, re-wire AEManager.__init__
to construct the lookup behind AE_OPENING_BOOK=1 env flag and call
get_opening_target() in _choose_target.

ORIGINAL hypothesis: the shipping heuristic re-evaluates targets every
tick and gets distracted by nearby items, so it bombs enemy bases
inconsistently. A fixed-map opening that *commits* the planner to
"race to the nearest enemy base and bomb it" for the first N ticks
may convert more low-EV item picks into +50-reward base destructions.

This module is data-only: it precomputes, at import time, the
following lookup keyed by ``team_idx`` (0..5):

    nearest_enemy_base_pos:  (x, y)
    nearest_enemy_base_slot: int
    bomb_cell:               (x, y)   -- the cell adjacent-or-within-
                                          blast of the enemy base that
                                          we will park on to bomb it
    target_distance:         int      -- Dijkstra cost to bomb_cell

Computed via Dijkstra over the static Novice map using cost=1 per
cardinal step and cost+5 per destructible-wall edge (matches the
existing ``DIJKSTRA_BOMB_COST`` default).

Used by ``ae_manager.AEManager`` behind ``AE_OPENING_BOOK=1`` env flag.
Default OFF — does not affect the 0.638 shipping path.
"""

from __future__ import annotations

import heapq
from typing import NamedTuple

from novice_map_data import (
    BASE_LOCATIONS,
    STARTING_LOCATIONS,
    WALLS,
    DESTRUCTIBLE,
)

# Edge direction codes (match til_environment): 0=right, 1=down, 2=left, 3=up
_DIR_DELTAS = [(1, 0), (0, 1), (-1, 0), (0, -1)]
_OPP_DIR = [2, 3, 0, 1]

_WALL_SET = set(WALLS)
_DESTR_SET = set(DESTRUCTIBLE)

_BOMB_RADIUS = 2
_DESTR_COST = 5  # matches AEManager.DIJKSTRA_BOMB_COST default


class OpeningTarget(NamedTuple):
    enemy_base_pos: tuple[int, int]
    enemy_base_slot: int
    bomb_cell: tuple[int, int]
    distance_cost: int


def _passable(pos: tuple[int, int], direction: int) -> tuple[bool, int]:
    """Return (passable, extra_cost) for the edge from ``pos`` in ``direction``.

    Walls block entirely. Destructibles are passable but add ``_DESTR_COST``
    to model the time the agent would spend bombing them open."""
    x, y = pos
    edge = (x, y, direction)
    if edge in _WALL_SET and edge not in _DESTR_SET:
        return False, 0
    nx, ny = x + _DIR_DELTAS[direction][0], y + _DIR_DELTAS[direction][1]
    if not (0 <= nx < 16 and 0 <= ny < 16):
        return False, 0
    rev = (nx, ny, _OPP_DIR[direction])
    if rev in _WALL_SET and rev not in _DESTR_SET:
        return False, 0
    extra = 0
    if edge in _DESTR_SET or rev in _DESTR_SET:
        extra = _DESTR_COST
    return True, extra


def _dijkstra_costs(start: tuple[int, int]) -> dict[tuple[int, int], int]:
    """Single-source shortest paths from ``start`` over the static map."""
    dist: dict[tuple[int, int], int] = {start: 0}
    pq: list[tuple[int, tuple[int, int]]] = [(0, start)]
    while pq:
        d, p = heapq.heappop(pq)
        if d > dist[p]:
            continue
        for direction in range(4):
            ok, extra = _passable(p, direction)
            if not ok:
                continue
            nx, ny = p[0] + _DIR_DELTAS[direction][0], p[1] + _DIR_DELTAS[direction][1]
            nd = d + 1 + extra
            np_ = (nx, ny)
            if nd < dist.get(np_, 1 << 30):
                dist[np_] = nd
                heapq.heappush(pq, (nd, np_))
    return dist


def _line_of_sight_unblocked(a: tuple[int, int], b: tuple[int, int]) -> bool:
    """True if a bomb at ``a`` would blast cell ``b`` (cardinal-line within
    radius, no full-wall blockers in the way).

    Approximation: we only check that ``a`` and ``b`` share a row or column
    within ``_BOMB_RADIUS``, AND that no edge along the segment is a
    non-destructible wall. Bombs in this env propagate in cardinal lines."""
    ax, ay = a
    bx, by = b
    if ax != bx and ay != by:
        return False
    if max(abs(ax - bx), abs(ay - by)) > _BOMB_RADIUS:
        return False
    if a == b:
        return True
    if ax == bx:
        step = 1 if by > ay else -1
        direction = 1 if step == 1 else 3  # down else up
        cur = (ax, ay)
        while cur != b:
            ok, extra = _passable(cur, direction)
            if not ok:
                return False
            # destructible edges still block bomb propagation in the env's
            # dynamics — be conservative and refuse line-of-sight through
            # any destructible (better to walk one more step than waste a
            # bomb on a wall absorbing the blast).
            if extra > 0:
                return False
            cur = (cur[0], cur[1] + step)
        return True
    # same row
    step = 1 if bx > ax else -1
    direction = 0 if step == 1 else 2  # right else left
    cur = (ax, ay)
    while cur != b:
        ok, extra = _passable(cur, direction)
        if not ok:
            return False
        if extra > 0:
            return False
        cur = (cur[0] + step, cur[1])
    return True


def _find_bomb_cell(
    distances: dict[tuple[int, int], int],
    target_base: tuple[int, int],
) -> tuple[tuple[int, int], int]:
    """Pick the reachable cell with shortest distance from spawn that also
    has clean line-of-sight blast on ``target_base``."""
    best: tuple[int, int] | None = None
    best_cost = 1 << 30
    for cell, cost in distances.items():
        if cell == target_base:
            continue
        if not _line_of_sight_unblocked(cell, target_base):
            continue
        if cost < best_cost:
            best_cost = cost
            best = cell
    if best is None:
        # Fallback: closest cell that is at least cardinally aligned, even
        # through a destructible. The agent will need to clear walls but
        # at least we point in the right direction.
        for cell, cost in sorted(distances.items(), key=lambda kv: kv[1]):
            if cell == target_base:
                continue
            if cell[0] == target_base[0] or cell[1] == target_base[1]:
                if max(abs(cell[0] - target_base[0]), abs(cell[1] - target_base[1])) <= _BOMB_RADIUS:
                    return cell, cost
        # Absolute last resort: just go AT the base.
        return target_base, distances.get(target_base, 1 << 30)
    return best, best_cost


def _compute_openings() -> dict[int, OpeningTarget]:
    """Pre-compute one OpeningTarget per team slot."""
    out: dict[int, OpeningTarget] = {}
    for team_idx, spawn in enumerate(STARTING_LOCATIONS):
        spawn_t = tuple(spawn)
        distances = _dijkstra_costs(spawn_t)
        best_base = None
        best_base_slot = -1
        best_base_cost = 1 << 30
        for slot, base in enumerate(BASE_LOCATIONS):
            if slot == team_idx:
                continue
            base_t = tuple(base)
            cost = distances.get(base_t, 1 << 30)
            if cost < best_base_cost:
                best_base_cost = cost
                best_base = base_t
                best_base_slot = slot
        if best_base is None:
            continue  # shouldn't happen on the real map
        bomb_cell, bomb_cost = _find_bomb_cell(distances, best_base)
        out[team_idx] = OpeningTarget(
            enemy_base_pos=best_base,
            enemy_base_slot=best_base_slot,
            bomb_cell=bomb_cell,
            distance_cost=bomb_cost,
        )
    return out


OPENINGS: dict[int, OpeningTarget] = _compute_openings()


def get_opening_target(team_idx: int) -> OpeningTarget | None:
    """Public lookup. Returns None if team_idx wasn't in the precomputed
    table (e.g. on Advanced maps where this module shouldn't be used)."""
    return OPENINGS.get(team_idx)


if __name__ == "__main__":
    # Diagnostic dump — run with `python ae/src/opening_book.py`.
    for idx in sorted(OPENINGS.keys()):
        t = OPENINGS[idx]
        spawn = STARTING_LOCATIONS[idx]
        print(
            f"slot {idx} spawn={tuple(spawn)} -> "
            f"enemy base {t.enemy_base_pos} (slot {t.enemy_base_slot}), "
            f"bomb from {t.bomb_cell}, dijkstra_cost={t.distance_cost}"
        )
