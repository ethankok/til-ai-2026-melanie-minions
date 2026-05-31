"""Offline opening generator for the fixed Novice map.

`OpeningSim` is a minimal, forkable single-agent forward model of the Novice
map used to search for strong opening move sequences. It is *movement-only*
(FORWARD/BACKWARD/LEFT/RIGHT/STAY — no bombs): the opening's job is to collect
the reachable item value and reach a good hand-off position before enemy
contact; bomb-shortcut openings are a future extension, and the live planner
(which bombs) takes over after the opening.

Physics mirror `til_environment.dynamics.move_agent`:
- Direction RIGHT=0, DOWN=1, LEFT=2, UP=3; deltas below.
- FORWARD moves along facing; BACKWARD moves along (facing+2)%4 but leaves
  facing unchanged; LEFT turns facing (dir+3)%4; RIGHT turns (dir+1)%4; STAY noop.
- A move is blocked by any wall edge (destructible OR indestructible) on the
  current tile in the move direction, or by an out-of-bounds destination.
- Stepping onto a tile with an uncollected item collects it once.

All map data is read from `novice_map_data` (the same source the deployed
detector validates against). The env-parity test in test_opening_sim.py
confirms physics + wall data + item positions + reward values simultaneously.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

# novice_map_data lives in ae/src; ensure importable when used standalone.
_AE_SRC = Path(__file__).resolve().parents[2] / "ae" / "src"
if str(_AE_SRC) not in sys.path:
    sys.path.insert(0, str(_AE_SRC))

from novice_map_data import (  # noqa: E402
    BASE_LOCATIONS,
    DESTRUCTIBLE,
    STARTING_LOCATIONS,
    STATIC_ENTITIES,
    WALLS,
)

# Direction deltas: RIGHT, DOWN, LEFT, UP.
DELTAS = ((1, 0), (0, 1), (-1, 0), (0, -1))
OPPOSITE = (2, 3, 0, 1)
GRID = 16

FORWARD, BACKWARD, LEFT, RIGHT, STAY = 0, 1, 2, 3, 4
MOVE_ACTIONS = (FORWARD, BACKWARD, LEFT, RIGHT, STAY)

# Real game-reward units (validated by the env-parity test).
ITEM_VALUE = {"mission": 5.0, "resource": 2.0, "recon": 1.0}

# Deterministic Novice starting facings per spawn slot (verified across seeds
# 7/42/999/31337 — novice forces a fixed arena RNG). Order matches
# BASE_LOCATIONS / STARTING_LOCATIONS.
STARTING_DIRECTIONS = (0, 1, 2, 1, 3, 1)

# base_location -> (start_pos, start_dir)
SLOT_TABLE: dict[tuple[int, int], tuple[tuple[int, int], int]] = {
    tuple(b): (tuple(s), d)
    for b, s, d in zip(BASE_LOCATIONS, STARTING_LOCATIONS, STARTING_DIRECTIONS)
}


@dataclass(frozen=True)
class State:
    pos: tuple[int, int]
    dir: int
    collected: frozenset
    reward: float
    tick: int


class OpeningSim:
    DELTAS = DELTAS

    def __init__(self) -> None:
        # Both wall types block movement.
        self.blocking_edges: set[tuple[int, int, int]] = set()
        for (x, y, d) in WALLS:
            self.blocking_edges.add((x, y, d))
        for (x, y, d) in DESTRUCTIBLE:
            self.blocking_edges.add((x, y, d))
        self.items: dict[tuple[int, int], str] = {
            (int(x), int(y)): kind for kind, (x, y) in STATIC_ENTITIES
        }

    # ── geometry ──────────────────────────────────────────────────────────
    def _blocked(self, pos: tuple[int, int], direction: int) -> bool:
        x, y = pos
        dx, dy = DELTAS[direction]
        nx, ny = x + dx, y + dy
        if not (0 <= nx < GRID and 0 <= ny < GRID):
            return True
        if (x, y, direction) in self.blocking_edges:
            return True
        # Edge may be recorded from the neighbour's side.
        if (nx, ny, OPPOSITE[direction]) in self.blocking_edges:
            return True
        return False

    # ── transitions ───────────────────────────────────────────────────────
    def initial_state(self, base_location: tuple[int, int]) -> State:
        pos, d = SLOT_TABLE[tuple(base_location)]
        return State(pos=tuple(pos), dir=d, collected=frozenset(), reward=0.0, tick=0)

    def step(self, st: State, action: int) -> State:
        pos, d = st.pos, st.dir
        collected, reward = st.collected, st.reward

        if action == LEFT:
            d = (st.dir + 3) % 4
        elif action == RIGHT:
            d = (st.dir + 1) % 4
        elif action in (FORWARD, BACKWARD):
            move_dir = st.dir if action == FORWARD else (st.dir + 2) % 4
            if not self._blocked(pos, move_dir):
                dx, dy = DELTAS[move_dir]
                npos = (pos[0] + dx, pos[1] + dy)
                pos = npos
                if npos in self.items and npos not in collected:
                    collected = collected | {npos}
                    reward = reward + ITEM_VALUE[self.items[npos]]
        # STAY: no change beyond tick.
        return State(pos=pos, dir=d, collected=collected, reward=reward, tick=st.tick + 1)


def _enemy_bases(base_location: tuple[int, int]) -> list[tuple[int, int]]:
    return [tuple(b) for b in BASE_LOCATIONS if tuple(b) != tuple(base_location)]


def beam_search(
    sim: OpeningSim,
    base_location: tuple[int, int],
    *,
    horizon: int,
    beam_width: int,
    top_k: int,
    position_weight: float = 0.0,
) -> list[dict]:
    """Beam search for high-reward opening sequences from one spawn slot.

    Returns up to ``top_k`` candidates (diverse by collected-item set), ranked
    by terminal score = reward + position_weight * (−dist to nearest enemy base).
    """
    start = sim.initial_state(base_location)
    # beam: list of (state, actions_tuple)
    beam: list[tuple[State, tuple[int, ...]]] = [(start, ())]
    enemy_bases = _enemy_bases(base_location)

    for _ in range(horizon):
        # dedup by (pos, dir, collected); keep the best-reward path per key.
        best: dict[tuple, tuple[State, tuple[int, ...]]] = {}
        for st, acts in beam:
            for a in MOVE_ACTIONS:
                ns = sim.step(st, a)
                key = (ns.pos, ns.dir, ns.collected)
                cur = best.get(key)
                if cur is None or ns.reward > cur[0].reward:
                    best[key] = (ns, acts + (a,))
        # keep top beam_width by reward (search-time ranking).
        beam = sorted(best.values(), key=lambda sa: sa[0].reward, reverse=True)[:beam_width]

    def _pos_bonus(pos: tuple[int, int]) -> float:
        if not enemy_bases or position_weight == 0.0:
            return 0.0
        nearest = min(abs(pos[0] - bx) + abs(pos[1] - by) for bx, by in enemy_bases)
        return -float(nearest)

    def _score(st: State) -> float:
        return st.reward + position_weight * _pos_bonus(st.pos)

    ranked = sorted(beam, key=lambda sa: _score(sa[0]), reverse=True)

    out: list[dict] = []
    seen_collected: set[frozenset] = set()
    for st, acts in ranked:
        if st.collected in seen_collected:
            continue
        seen_collected.add(st.collected)
        out.append(
            {
                "actions": list(acts),
                "reward": st.reward,
                "score": _score(st),
                "end_pos": st.pos,
                "end_dir": st.dir,
                "items": sorted(st.collected),
            }
        )
        if len(out) >= top_k:
            break
    return out
