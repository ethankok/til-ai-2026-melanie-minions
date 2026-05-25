"""Tactical option labels for AE semifinals policy training/runtime.

These labels are intentionally narrower than the older 8-way option set.
They describe the tactical macro the executor should attempt now, not just a
general strategic mood such as "defend" or "collect".
"""

from __future__ import annotations

from typing import Mapping


TACTICAL_ESCAPE_BOMB = 0
TACTICAL_INTERCEPT_BASE_THREAT = 1
TACTICAL_GUARD_BASE_LANE = 2
TACTICAL_BOMB_BASE_THREAT = 3
TACTICAL_RUSH_ENEMY_BASE = 4
TACTICAL_BOMB_ENEMY_BASE = 5
TACTICAL_HUNT_VISIBLE_ENEMY = 6
TACTICAL_COLLECT_MISSION_SAFE = 7
TACTICAL_COLLECT_RESOURCE_SAFE = 8
TACTICAL_DENY_ENEMY_MISSION = 9
TACTICAL_COUNTER_RUSH = 10
TACTICAL_STALL_WHEN_WINNING = 11

TACTICAL_OPTION_NAMES = (
    "escape_bomb",
    "intercept_base_threat",
    "guard_base_lane",
    "bomb_base_threat",
    "rush_enemy_base",
    "bomb_enemy_base",
    "hunt_visible_enemy",
    "collect_mission_safe",
    "collect_resource_safe",
    "deny_enemy_mission",
    "counter_rush",
    "stall_when_winning",
)
NUM_TACTICAL_OPTIONS = len(TACTICAL_OPTION_NAMES)

DEFENSIVE_TACTICAL_OPTIONS = {
    TACTICAL_ESCAPE_BOMB,
    TACTICAL_INTERCEPT_BASE_THREAT,
    TACTICAL_GUARD_BASE_LANE,
    TACTICAL_BOMB_BASE_THREAT,
    TACTICAL_COUNTER_RUSH,
    TACTICAL_STALL_WHEN_WINNING,
}


def tactical_option_name(option: int) -> str:
    if 0 <= int(option) < NUM_TACTICAL_OPTIONS:
        return TACTICAL_OPTION_NAMES[int(option)]
    return f"unknown_{option}"


def _decision(manager) -> str:
    return str(getattr(manager, "last_decision", "") or "")


def _target_kind(manager) -> str:
    return str(getattr(manager, "last_target_kind", "") or "")


def _bomb_reason(manager) -> str:
    return str(getattr(manager, "last_bomb_reason", "") or "")


def _base_threat_active(manager) -> bool:
    base = getattr(manager, "base_location", None)
    if base is None:
        return False
    step = getattr(manager, "last_step", None) or 0
    staleness = max(6, int(getattr(manager, "ENEMY_STALENESS", 4)))
    radius = max(7, int(getattr(manager, "BASE_DEFENSE_RADIUS", 6)) + 2)
    for pos, last_seen in getattr(manager, "enemy_agents", {}).items():
        try:
            if step - int(last_seen) <= staleness and manager._manhattan(pos, base) <= radius:
                return True
        except Exception:
            continue
    return False


def tactical_from_manager(manager, action: int | None = None) -> int:
    """Map the heuristic's last decision to a sharper tactical macro."""

    decision = _decision(manager)
    target_kind = _target_kind(manager)
    bomb_reason = _bomb_reason(manager)
    base_threat = _base_threat_active(manager)

    if decision in {"frozen", "fallback_no_location"}:
        return TACTICAL_ESCAPE_BOMB

    if decision.startswith("bomb_") or decision.startswith("dominant_bomb"):
        reason = decision.split("bomb_", 1)[-1] if "bomb_" in decision else bomb_reason
        if reason in {"enemy_base", "fixed_map_wall", "wall_to_high_value"}:
            return TACTICAL_BOMB_ENEMY_BASE
        if base_threat:
            return TACTICAL_BOMB_BASE_THREAT
        if reason in {"enemy_agent", "enemy_cluster", "predictive_walk", "repeat_kill"}:
            return TACTICAL_HUNT_VISIBLE_ENEMY
        return TACTICAL_BOMB_ENEMY_BASE

    if decision == "tactical_lookahead":
        if action == getattr(manager, "PLACE_BOMB", 5):
            return TACTICAL_BOMB_BASE_THREAT if base_threat else TACTICAL_BOMB_ENEMY_BASE
        return TACTICAL_INTERCEPT_BASE_THREAT if base_threat else TACTICAL_RUSH_ENEMY_BASE

    if decision.startswith("dominant_adjacent_mission"):
        return TACTICAL_COLLECT_MISSION_SAFE

    kind = target_kind
    if decision.startswith("path_"):
        kind = decision.removeprefix("path_")

    if kind in {"defense_emergency", "base_defense"}:
        return TACTICAL_INTERCEPT_BASE_THREAT
    if base_threat and kind in {"enemy_chase", "enemy_base"}:
        return TACTICAL_COUNTER_RUSH
    if kind == "enemy_base":
        return TACTICAL_RUSH_ENEMY_BASE
    if kind == "enemy_chase":
        return TACTICAL_HUNT_VISIBLE_ENEMY
    if kind == "item_mission":
        return TACTICAL_COLLECT_MISSION_SAFE
    if kind in {"item_resource", "item_recon", "respawn_resource", "respawn_recon"}:
        return TACTICAL_COLLECT_RESOURCE_SAFE
    if kind == "respawn_mission":
        return TACTICAL_DENY_ENEMY_MISSION
    if kind in {"frontier", "low_visit", "none"}:
        return TACTICAL_GUARD_BASE_LANE if base_threat else TACTICAL_COLLECT_MISSION_SAFE

    if decision.startswith("fallback") and getattr(manager, "escape_target", None) is not None:
        return TACTICAL_ESCAPE_BOMB

    if base_threat:
        return TACTICAL_GUARD_BASE_LANE
    return TACTICAL_RUSH_ENEMY_BASE


def tactical_counts_to_names(counts: Mapping[int, int]) -> dict[str, int]:
    return {tactical_option_name(option): int(count) for option, count in counts.items()}
