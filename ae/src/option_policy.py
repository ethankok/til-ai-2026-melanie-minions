"""Shared option labels for AE option-policy training and runtime.

The option policy does not choose raw Bomberman actions. It chooses a
high-level intent, then the runtime executes that intent through a tuned
planner variant with the normal action-mask and bomb-safety checks.
"""

from __future__ import annotations

from typing import Mapping


OPTION_ESCAPE = 0
OPTION_RUSH_BASE = 1
OPTION_BASE_BOMB = 2
OPTION_DEFEND_BASE = 3
OPTION_COLLECT_MISSION = 4
OPTION_COLLECT_RESOURCE = 5
OPTION_HUNT_ENEMY = 6
OPTION_EXPLORE = 7

OPTION_NAMES = (
    "escape",
    "rush_base",
    "base_bomb",
    "defend_base",
    "collect_mission",
    "collect_resource",
    "hunt_enemy",
    "explore",
)
NUM_OPTIONS = len(OPTION_NAMES)


def option_name(option: int) -> str:
    if 0 <= int(option) < NUM_OPTIONS:
        return OPTION_NAMES[int(option)]
    return f"unknown_{option}"


def _decision(manager) -> str:
    return str(getattr(manager, "last_decision", "") or "")


def _target_kind(manager) -> str:
    return str(getattr(manager, "last_target_kind", "") or "")


def _bomb_reason(manager) -> str:
    return str(getattr(manager, "last_bomb_reason", "") or "")


def option_from_manager(manager, action: int | None = None) -> int:
    """Map an ``AEManager`` planner decision to one of our option labels.

    This is deliberately planner-derived, not action-derived. A single raw
    action can mean many things (turning toward a base, dodging a bomb,
    grabbing a mission tile), so the option label should follow the reason
    the heuristic recorded while choosing it.
    """

    decision = _decision(manager)
    target_kind = _target_kind(manager)
    bomb_reason = _bomb_reason(manager)

    if decision in {"frozen", "fallback_no_location"}:
        return OPTION_ESCAPE

    if decision.startswith("bomb_") or decision.startswith("dominant_bomb"):
        reason = decision.split("bomb_", 1)[-1] if "bomb_" in decision else bomb_reason
        if reason in {"enemy_base", "fixed_map_wall", "wall_to_high_value", "wall_unstuck"}:
            return OPTION_BASE_BOMB
        if reason in {"enemy_agent", "enemy_cluster", "predictive_walk", "repeat_kill"}:
            return OPTION_HUNT_ENEMY
        return OPTION_BASE_BOMB

    if decision == "tactical_lookahead":
        if action == getattr(manager, "PLACE_BOMB", 5):
            return OPTION_BASE_BOMB
        if getattr(manager, "enemy_agents", None):
            return OPTION_HUNT_ENEMY
        return OPTION_RUSH_BASE

    if decision.startswith("dominant_adjacent_mission"):
        return OPTION_COLLECT_MISSION

    if decision.startswith("path_"):
        kind = decision.removeprefix("path_")
        if kind == "enemy_base":
            return OPTION_RUSH_BASE
        if kind in {"defense_emergency", "base_defense"}:
            return OPTION_DEFEND_BASE
        if kind == "enemy_chase":
            return OPTION_HUNT_ENEMY
        if kind == "item_mission":
            return OPTION_COLLECT_MISSION
        if kind in {"item_resource", "item_recon", "respawn_resource", "respawn_recon"}:
            return OPTION_COLLECT_RESOURCE
        if kind in {"respawn_mission"}:
            return OPTION_COLLECT_MISSION
        if kind in {"frontier", "low_visit", "none"}:
            return OPTION_EXPLORE

    if target_kind == "enemy_base":
        return OPTION_RUSH_BASE
    if target_kind in {"defense_emergency", "base_defense"}:
        return OPTION_DEFEND_BASE
    if target_kind == "enemy_chase":
        return OPTION_HUNT_ENEMY
    if target_kind == "item_mission":
        return OPTION_COLLECT_MISSION
    if target_kind in {"item_resource", "item_recon", "respawn_resource", "respawn_recon"}:
        return OPTION_COLLECT_RESOURCE

    if decision.startswith("fallback") and getattr(manager, "escape_target", None) is not None:
        return OPTION_ESCAPE
    return OPTION_EXPLORE


def option_counts_to_names(counts: Mapping[int, int]) -> dict[str, int]:
    return {option_name(option): int(count) for option, count in counts.items()}
