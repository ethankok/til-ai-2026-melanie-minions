"""Spawn-aware first-target ordering, ported from M5 ScriptedBaseAttackPolicy.

The M5 scripted policy used a per-own-base lookup table to prioritize which
enemy base to attack first. Three of the six Novice spawn slots have explicit
priority tuples; the other three fall back to standard route scoring with no
priority boost. See `M5_SUMMARY.md` (FIRST_TARGET_BY_OWN_BASE).

This module is a small library — no AEManager dependency, no env reads. The
manager consumes it via `get_first_target_rank(own_base, enemy_base)` and
boosts the enemy_base candidate value by a rank-decreasing amount.

Default OFF in AEManager — enable with `AE_FIRST_TARGET_TABLE=1`. The boost
magnitude is also env-tunable so we can A/B with multi_seed_eval without code
changes (`AE_FIRST_TARGET_BOOST=60`, etc.).
"""

from __future__ import annotations

from typing import Iterable, Optional


# Only three of the six base spawns get an explicit priority tuple; spawns not
# listed here use route scoring alone, by design.
FIRST_TARGET_BY_OWN_BASE: dict[tuple[int, int], tuple[tuple[int, int], ...]] = {
    (3, 12): ((6, 2), (2, 6), (12, 3), (13, 9), (9, 13)),
    (6, 2):  ((13, 9), (12, 3), (9, 13), (3, 12), (2, 6)),
    (12, 3): ((2, 6), (13, 9), (3, 12), (9, 13), (6, 2)),
}


def get_first_target_rank(
    own_base: tuple[int, int] | None,
    enemy_base: tuple[int, int],
) -> Optional[int]:
    """Return the 0-indexed priority rank for `enemy_base` given our spawn.

    Returns 0 for the first-priority enemy base, 1 for second, etc.; or None
    if our own spawn is not in the table (3 of 6 slots) OR if `enemy_base` is
    not listed for that spawn.
    """
    if own_base is None:
        return None
    key = (int(own_base[0]), int(own_base[1]))
    targets = FIRST_TARGET_BY_OWN_BASE.get(key)
    if targets is None:
        return None
    target = (int(enemy_base[0]), int(enemy_base[1]))
    for i, t in enumerate(targets):
        if t == target:
            return i
    return None


def rank_boost(rank: Optional[int], base_boost: float, decay: float = 0.55) -> float:
    """Convert a rank (0..N-1) to an additive value boost.

    Defaults: rank 0 -> base_boost, rank 1 -> base_boost*0.55, etc.
    None -> 0.0 (no boost).
    """
    if rank is None:
        return 0.0
    return float(base_boost) * (float(decay) ** int(rank))


def known_own_base_spawns() -> Iterable[tuple[int, int]]:
    """For tests / diagnostics."""
    return tuple(FIRST_TARGET_BY_OWN_BASE.keys())
