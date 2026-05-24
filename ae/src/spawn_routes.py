"""Pre-computed greedy item+base routes per fixed-Novice spawn cell.

STATUS (24 May 2026 PM): NEGATIVE RESULT — not part of the shipping
path. Cloud A/B (5 baseline vs 4 cheese samples) returned cheese mean
``0.535`` vs baseline mean ``0.557`` (−0.022, ~1σ). Hypothesis
falsified: committing to a precomputed greedy route loses to per-tick
re-evaluation in the cloud opponent distribution. Same shape failure as
``opening_book.py`` — see ``ae/NOTES.md`` afternoon entry for the full
write-up.

Module kept in tree as a reusable map-analysis primitive (greedy
EV/distance planner over the static map) and reference for the negative
result. To re-enable for ad-hoc experimentation, set
``AE_USE_MEMORIZED_ROUTE=1``; the runtime hook in ``ae_manager.py``
remains wired (default OFF).

ORIGINAL hypothesis: the shipping heuristic re-evaluates targets every
tick and gets distracted by nearby low-value items (same diagnosis as
``opening_book.py``). Committing the planner to a fixed greedy sequence
of high-EV targets per spawn cell should bypass the per-tick re-
evaluation tax.

Guarded at runtime by ``AE_USE_MEMORIZED_ROUTE=1`` env flag. Default OFF
so the shipping path is unaffected. Only loaded after the step-0 fixed-
Novice detection in ``ae_manager.py`` sets ``is_fixed_novice_map=True``.

Algorithm (per spawn cell ``s``):

    1. Run Dijkstra from ``s`` over the static map (same edge/wall/
       destructible rules as opening_book._dijkstra_costs).
    2. Score each remaining item by ``value[kind] / (dist + 1)`` and
       each remaining enemy base by ``ENEMY_BASE_VALUE / (bomb_cost+1)``
       where ``bomb_cost`` is the cost to a line-of-sight cell that
       blasts the base (via opening_book._find_bomb_cell).
    3. Pick the best, append to route, advance virtual position to the
       target (or to the bomb_cell for bases), re-run Dijkstra, repeat.
    4. Stop at ``MAX_WAYPOINTS`` to bound import-time cost.

Item values match ``AEManager.ITEM_VALUES`` defaults (mission=50,
resource=25, recon=10). Enemy base value matches the fixed-Novice
override in ``AEManager.__init__`` step-0 block (130.0).

Output is a module-level lookup ``SPAWN_ROUTES`` keyed by spawn
position (tuple) — the values in ``STARTING_LOCATIONS`` — with a list
of ``(kind, (x, y))`` waypoints in greedy execution order. ``kind`` is
one of ``"mission" | "resource" | "recon" | "enemy_base"``.

Run ``python ae/src/spawn_routes.py`` to dump the precomputed table for
visual inspection before submission.
"""

from __future__ import annotations

from typing import NamedTuple

from novice_map_data import (
    BASE_LOCATIONS,
    STARTING_LOCATIONS,
    STATIC_ENTITIES,
)
from opening_book import (
    _dijkstra_costs,
    _find_bomb_cell,
)


# Match AEManager.ITEM_VALUES defaults. Keep in sync with ae_manager.py
# lines 146 + 260-267 if those defaults are retuned.
ITEM_VALUE: dict[str, float] = {
    "mission": 50.0,
    "resource": 25.0,
    "recon": 10.0,
}

# Match AEManager.__init__ fixed-Novice override (ae_manager.py line 307).
ENEMY_BASE_VALUE: float = 130.0

# Soft cap on route length. 40 waypoints comfortably exceeds the ~30-50
# moves a typical episode allows after subtracting wall-bombing time, so
# the route effectively never runs out under normal play.
MAX_WAYPOINTS: int = 40

# Distance smoothing constant. value / (dist + 1) prevents division by
# zero and discounts unreachable (BIG-distance) candidates uniformly.
_BIG: int = 1 << 30


class Waypoint(NamedTuple):
    kind: str  # "mission" | "resource" | "recon" | "enemy_base"
    pos: tuple[int, int]
    dijkstra_cost: int  # cost from previous waypoint (informational)


def _build_route(
    spawn: tuple[int, int],
    own_base: tuple[int, int],
) -> list[Waypoint]:
    """Greedy EV/distance route from ``spawn``, skipping our own base."""

    # Remaining-candidate sets. Items are by-position; bases by-position too.
    items_remaining: dict[tuple[int, int], str] = {
        tuple(pos): kind for kind, pos in STATIC_ENTITIES
    }
    bases_remaining: set[tuple[int, int]] = {
        tuple(b) for b in BASE_LOCATIONS if tuple(b) != own_base
    }

    cur: tuple[int, int] = tuple(spawn)
    dist = _dijkstra_costs(cur)
    route: list[Waypoint] = []

    while items_remaining or bases_remaining:
        best_ev: float = -1.0
        best: tuple[str, tuple[int, int], int, tuple[int, int]] | None = None
        # best holds (kind, target_pos, cost, next_virtual_pos).
        # next_virtual_pos differs from target_pos for enemy bases (we sit
        # at the bomb_cell, not on top of the base which is enclosed).

        for pos, kind in items_remaining.items():
            d = dist.get(pos, _BIG)
            if d >= _BIG:
                continue
            ev = ITEM_VALUE[kind] / (d + 1)
            if ev > best_ev:
                best_ev = ev
                best = (kind, pos, d, pos)

        for base in bases_remaining:
            bomb_cell, bomb_cost = _find_bomb_cell(dist, base)
            if bomb_cost >= _BIG:
                continue
            ev = ENEMY_BASE_VALUE / (bomb_cost + 1)
            if ev > best_ev:
                best_ev = ev
                best = ("enemy_base", base, bomb_cost, bomb_cell)

        if best is None:
            break
        kind, target_pos, cost, next_pos = best
        route.append(Waypoint(kind=kind, pos=target_pos, dijkstra_cost=cost))
        if kind == "enemy_base":
            bases_remaining.discard(target_pos)
        else:
            items_remaining.pop(target_pos, None)
        cur = next_pos
        dist = _dijkstra_costs(cur)
        if len(route) >= MAX_WAYPOINTS:
            break
    return route


def _build_all_routes() -> dict[tuple[int, int], list[Waypoint]]:
    """One route per spawn cell in STARTING_LOCATIONS."""
    out: dict[tuple[int, int], list[Waypoint]] = {}
    for idx, spawn in enumerate(STARTING_LOCATIONS):
        spawn_t = tuple(spawn)
        own_base_t = tuple(BASE_LOCATIONS[idx])
        out[spawn_t] = _build_route(spawn_t, own_base_t)
    return out


# Precompute at import time. ~6 spawns × Dijkstra-per-waypoint × ~40
# waypoints ≈ 240 Dijkstra runs on a 16×16 grid, total well under a
# second on the deploy container.
SPAWN_ROUTES: dict[tuple[int, int], list[Waypoint]] = _build_all_routes()


def get_route(spawn: tuple[int, int]) -> list[Waypoint] | None:
    """Public lookup. Returns None if spawn isn't in the precomputed
    table (e.g. Advanced map / unexpected spawn)."""
    return SPAWN_ROUTES.get(tuple(spawn))


if __name__ == "__main__":
    # Diagnostic dump — run with `python ae/src/spawn_routes.py`.
    print(f"=== SPAWN_ROUTES ({len(SPAWN_ROUTES)} spawns) ===\n")
    for idx, spawn in enumerate(STARTING_LOCATIONS):
        spawn_t = tuple(spawn)
        route = SPAWN_ROUTES.get(spawn_t, [])
        own_base = tuple(BASE_LOCATIONS[idx])
        kinds = {"mission": 0, "resource": 0, "recon": 0, "enemy_base": 0}
        total_cost = 0
        total_value = 0.0
        for wp in route:
            kinds[wp.kind] += 1
            total_cost += wp.dijkstra_cost
            total_value += (
                ENEMY_BASE_VALUE if wp.kind == "enemy_base"
                else ITEM_VALUE[wp.kind]
            )
        print(
            f"slot {idx} spawn={spawn_t} own_base={own_base} | "
            f"len={len(route)} cost={total_cost} value={total_value:.0f} | "
            f"M={kinds['mission']} Rc={kinds['recon']} Rs={kinds['resource']} B={kinds['enemy_base']}"
        )
        for i, wp in enumerate(route):
            print(f"  {i:2d}. {wp.kind:10s} @ {wp.pos}  (+{wp.dijkstra_cost})")
        print()
