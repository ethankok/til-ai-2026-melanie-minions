"""Scripted opponents for offline AE training and simulation.

Scripted opponents that take a raw observation dict and return an int
action 0..5. The signature matches the hidden cloud "policy interface"
where the only input is the partial observation. All opponents respect
``action_mask`` and never crash on edge-case observations.

The set is deliberately small and fast — these run inside tight inner
loops in `simulate.py` and `train_ppo.py`. No torch, no numpy heavy
allocations; every choice resolves in microseconds.

Why this set:
  - random       : retained for explicit baseline checks only.
  - greedy       : enemies that move toward visible items / their own base.
  - bomber       : periodically places bombs, otherwise greedy.
  - bomber_fast  : faster periodic bomber for pressure stress tests.
  - defender     : rarely strays far from its own base; bombs intruders.
  - hunter       : chases the nearest sighted enemy and bombs adjacent.
  - hunter_sticky: keeps chasing stale sightings longer.
  - rusher       : fixed-map base pressure; stresses our defense/escape logic.
  - rusher_fast  : more single-minded base rusher.
  - rusher_safe  : base rusher that respects threats more heavily.
  - base_bomber  : frequent-bomb base pressure bot.
  - mixed        : per-game random switch among the non-random above.

Together they span the strategy space the hidden evaluator's NPCs
plausibly occupy. Training PPO and fitting an opponent model against
this library yields policies robust to the *kind* of opponents we'll
actually see, without depending on self-play (which kept failing).

Scripted opponents disable the playbook on their own AEManager instance
(``self.playbook = None``) so they stay pure heuristics during training.
Otherwise PPO rollouts would see opponents whose behavior partially
depends on a learned playbook — defeating the point of training PPO
against them. The opponent model walk_scale is left at the default 1.0
for the same reason (these scripted opponents represent unmodelled
opponents from our perspective).
"""

from __future__ import annotations

import os
import random
import sys
from collections import deque
from pathlib import Path
from typing import Callable

import numpy as np

# Reuse the planner's belief-map machinery for the scripted opponents.
# `AEManager` is already battle-tested for this; we just wrap it with
# different behavior policies.
_AE_SRC = str(Path(__file__).resolve().parents[2] / "ae" / "src")
if _AE_SRC not in sys.path:
    sys.path.insert(0, _AE_SRC)

from ae_manager import AEManager  # noqa: E402


OpponentFn = Callable[[dict], int]


def _strip_aimanager_smarts(m: AEManager) -> None:
    """Remove playbook + opponent-model smarts from an opponent's AEManager.

    Scripted opponents should behave like pure rule-based players regardless
    of any artifacts present in `ae/models/`. Without this, an opponent's
    AEManager would happily consult our learned playbook and use our
    opponent-model walk_scale, which mixes inference-side smarts into
    training data.
    """
    m.playbook = None
    m.opponent_walk_scale = 1.0
    # Also disable the tier-1 opt-ins so opponents play the original v3b
    # heuristic and don't shift the rollout distribution. (Tier-1 toggles
    # were chosen to help OUR policy; we don't want opponents getting them.)
    m.tier1_defense_priority = False
    m.tier1_repeat_kill = False
    m.tier1_shared_credit = False
    m.tier1_no_stay_penalty = False
    m.tier1_predictive_walk = False
    m.planner_mode = "legacy"

    # Force default heuristic parameters for opponents so they don't inherit our tuned settings
    m.ENEMY_BASE_VALUE = 80.0
    m.BASE_DEFENSE_VALUE = 60.0
    m.BASE_DEFENSE_EMERGENCY_VALUE = 150.0
    m.BASE_DEFENSE_RADIUS = 6
    m.DIST_PENALTY = 1.15
    m.PATH_THREAT_PENALTY = 2.0
    m.CELL_THREAT_PENALTY = 5.0
    m.ENEMY_CHASE_VALUE = 0.0


def _legal_actions(obs: dict) -> list[int]:
    mask = obs.get("action_mask")
    if mask is None:
        return [0, 1, 2, 3, 4, 5]
    legal = []
    for i in range(min(6, len(mask))):
        try:
            if int(mask[i]):
                legal.append(i)
        except Exception:
            pass
    return legal or [4]  # fall back to STAY if nothing legal


def _sample_random_action(obs: dict, rng: random.Random) -> int:
    return rng.choice(_legal_actions(obs))


# ---------------------------------------------------------------------------
# Wrapped behaviors — each subclasses AEManager and overrides ae() to alter
# behavior at the right level. We keep the belief-map and BFS infrastructure
# from the parent so opponent decisions stay in-distribution with cloud-
# observed enemy behavior (i.e., they aren't random-walkers; they're at
# least somewhat coherent).
# ---------------------------------------------------------------------------


class RandomOpponent:
    """Uniform random — matches `test/test_ae.py`'s NPC behavior."""

    name = "random"

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    def __call__(self, obs: dict) -> int:
        return _sample_random_action(obs, self.rng)


class GreedyCollector(AEManager):
    """Like the planner but with combat disabled.

    Strips the bomb-placement and lookahead paths, biased toward item
    collection and exploration. Useful as a baseline opponent that
    prioritizes item pickup.
    """

    name = "greedy"

    def __init__(self) -> None:
        super().__init__()
        self.mcts_enabled = False
        _strip_aimanager_smarts(self)

    def __call__(self, obs: dict) -> int:
        return self.ae(obs)

    def ae(self, observation: dict) -> int:
        action = super().ae(observation)
        # Strip place-bomb actions for this opponent — but only when there's
        # a legal alternative.
        if action == self.PLACE_BOMB:
            for alt in (self.FORWARD, self.LEFT, self.RIGHT, self.BACKWARD, self.STAY):
                if self._legal(observation, alt):
                    return alt
        return action


class Bomber(AEManager):
    """Greedy collector that drops a bomb every ``period`` ticks if it can."""

    name = "bomber"

    def __init__(self, period: int = 8) -> None:
        super().__init__()
        self.period = max(2, int(period))
        self._bomb_clock = 0
        _strip_aimanager_smarts(self)

    def __call__(self, obs: dict) -> int:
        return self.ae(obs)

    def ae(self, observation: dict) -> int:
        self._bomb_clock += 1
        if self._bomb_clock >= self.period and self._legal(observation, self.PLACE_BOMB):
            location = self._location(observation.get("location"))
            if location is not None:
                blast = self._blast_cells(location)
                # Don't bomb our own base.
                base = self.base_location or self._location(observation.get("base_location"))
                if base is not None and base in blast:
                    pass
                else:
                    escape = self._safe_escape_within(location, blast, self.BOMB_TIMER)
                    if escape is not None:
                        self._bomb_clock = 0
                        self.escape_target = escape
                        self.escape_until_step = (self.last_step or 0) + self.BOMB_TIMER
                        return self.PLACE_BOMB
        return super().ae(observation)


class FastBomber(Bomber):
    """Bomber with a shorter fuse cycle to stress bomb-escape decisions."""

    name = "bomber_fast"

    def __init__(self) -> None:
        super().__init__(period=4)


class Defender(AEManager):
    """Stays within a small radius of its own base; bombs intruders."""

    name = "defender"
    DEFEND_RADIUS = 5

    def __init__(self) -> None:
        super().__init__()
        # Defender doesn't care about distant enemy bases.
        self.tier1_shared_credit = True
        _strip_aimanager_smarts(self)
        # Re-enable shared-credit since this opponent should treat its base
        # offense as low-value (so it stays home defending).
        self.tier1_shared_credit = True

    def __call__(self, obs: dict) -> int:
        return self.ae(obs)

    def _choose_target(self, start, danger, low_health=False):
        # Override target selection: prefer cells near our base.
        target_pos, path = super()._choose_target(start, danger, low_health)
        if self.base_location is None:
            return target_pos, path
        # Stay close — if heuristic chose a far target, override with
        # nearest visible cell within DEFEND_RADIUS.
        if target_pos is None or self._manhattan(target_pos, self.base_location) > self.DEFEND_RADIUS:
            best = None
            best_dist = None
            for pos in self.seen:
                if pos == start:
                    continue
                d = self._manhattan(pos, self.base_location)
                if d > self.DEFEND_RADIUS:
                    continue
                if best is None or d < best_dist:
                    best = pos
                    best_dist = d
            if best is not None:
                distance, parent = self._bfs_distance_map(start, danger)
                path = self._reconstruct_path(parent, start, best)
                return best, path
        return target_pos, path


class Hunter(AEManager):
    """Chases the nearest sighted enemy and bombs adjacent."""

    name = "hunter"

    def __init__(self) -> None:
        super().__init__()
        _strip_aimanager_smarts(self)

    def __call__(self, obs: dict) -> int:
        return self.ae(obs)

    def _choose_target(self, start, danger, low_health=False):
        if self.enemy_agents and not low_health:
            step = self.last_step if self.last_step is not None else 0
            fresh = [
                pos for pos, seen in self.enemy_agents.items()
                if step - int(seen) <= self.ENEMY_STALENESS
            ]
            if fresh:
                target = min(fresh, key=lambda p: self._manhattan(start, p))
                distance, parent = self._bfs_distance_map(start, danger)
                if target in distance:
                    path = self._reconstruct_path(parent, start, target)
                    return target, path
        return super()._choose_target(start, danger, low_health)


class StickyHunter(Hunter):
    """Hunter that keeps pursuing stale enemy sightings for longer."""

    name = "hunter_sticky"

    def __init__(self) -> None:
        super().__init__()
        self.ENEMY_STALENESS = 8
        self.ENEMY_CHASE_VALUE = 28.0
        self.ENEMY_CHASE_RADIUS = 8

    def _choose_target(self, start, danger, low_health=False):
        if self.enemy_agents and not low_health:
            step = self.last_step if self.last_step is not None else 0
            remembered = [
                pos for pos, seen in self.enemy_agents.items()
                if step - int(seen) <= self.ENEMY_STALENESS
            ]
            if remembered:
                target = min(remembered, key=lambda p: self._manhattan(start, p))
                distance, parent = self._bfs_distance_map(start, danger)
                if target in distance:
                    path = self._reconstruct_path(parent, start, target)
                    return target, path
        return super()._choose_target(start, danger, low_health)


class BaseRusher(AEManager):
    """Pressure opponent that prioritizes enemy bases over item farming."""

    name = "rusher"

    def __init__(self) -> None:
        super().__init__()
        _strip_aimanager_smarts(self)
        self.ENEMY_BASE_VALUE = 150.0
        self.DIST_PENALTY = 0.85
        self.item_mission_value = 12.0
        self.item_resource_value = 5.0
        self.item_recon_value = 2.0
        self.ITEM_VALUES = {
            "mission": self.item_mission_value,
            "resource": self.item_resource_value,
            "recon": self.item_recon_value,
        }
        self.ENEMY_CHASE_VALUE = 18.0
        self.ENEMY_CHASE_RADIUS = 3

    def __call__(self, obs: dict) -> int:
        return self.ae(obs)

    def _choose_target(self, start, danger, low_health=False):
        if self.enemy_bases and not low_health:
            distance, parent = self._dijkstra_distance_map(start, danger)
            best = None
            best_score = float("-inf")
            for base in self.enemy_bases:
                for cell in self._attack_cells_for(base):
                    if cell not in distance:
                        continue
                    score = 170.0 - distance[cell] - 0.25 * self.visit_count.get(cell, 0)
                    if score > best_score:
                        best_score = score
                        best = cell
            if best is not None:
                return best, self._reconstruct_path(parent, start, best)
        return super()._choose_target(start, danger, low_health)

    def _attack_cells_for(self, base: tuple[int, int]) -> list[tuple[int, int]]:
        """Cells adjacent to ``base`` (or the base itself) that are inside the
        grid and not blocked by a known indestructible wall edge.

        This used to live on AEManager as ``_fixed_base_attack_cells`` but was
        removed when the runtime was reverted to the fixed-map-v3 state. Kept
        local to BaseRusher because nothing else needs it."""
        cells: list[tuple[int, int]] = [base]
        for direction, (dx, dy) in self.DIR_DELTAS.items():
            nxt = (base[0] + dx, base[1] + dy)
            if not self._in_bounds(nxt):
                continue
            cells.append(nxt)
        return cells


class FastRusher(BaseRusher):
    """More single-minded base rusher with lower path-distance penalty."""

    name = "rusher_fast"

    def __init__(self) -> None:
        super().__init__()
        self.ENEMY_BASE_VALUE = 190.0
        self.DIST_PENALTY = 0.55
        self.PATH_THREAT_PENALTY = 0.75
        self.CELL_THREAT_PENALTY = 2.0
        self.ITEM_VALUES = {"mission": 6.0, "resource": 2.0, "recon": 1.0}


class SafeRusher(BaseRusher):
    """Base rusher that still pressures bases but avoids threat cells harder."""

    name = "rusher_safe"

    def __init__(self) -> None:
        super().__init__()
        self.ENEMY_BASE_VALUE = 145.0
        self.DIST_PENALTY = 0.95
        self.PATH_THREAT_PENALTY = 4.0
        self.CELL_THREAT_PENALTY = 10.0
        self.ITEM_VALUES = {"mission": 16.0, "resource": 6.0, "recon": 2.0}


class BaseBomber(BaseRusher):
    """Rusher that drops bombs more readily near enemy-base routes."""

    name = "base_bomber"

    def __init__(self) -> None:
        super().__init__()
        self._bomb_clock = 0
        self.period = 5
        self.ENEMY_BASE_VALUE = 175.0
        self.DIST_PENALTY = 0.70
        self.ITEM_VALUES = {"mission": 8.0, "resource": 3.0, "recon": 1.0}

    def _should_place_bomb(self, observation, location, target, danger) -> bool:
        if super()._should_place_bomb(observation, location, target, danger):
            self._bomb_clock = 0
            return True
        self._bomb_clock += 1
        if (
            location is not None
            and self._bomb_clock >= self.period
            and self._legal(observation, self.PLACE_BOMB)
            and self._as_int(observation.get("team_bombs"), default=0) > 0
            and self._as_int(observation.get("health"), default=60) >= self.LOW_HEALTH_THRESHOLD
            and location not in danger
        ):
            blast = self._blast_cells(location)
            base = self.base_location or self._location(observation.get("base_location"))
            enemy_base_hit = any(pos in blast for pos in self.enemy_bases)
            wall_break = False
            target_base = target if target in self.enemy_bases else None
            if target_base is None and self.enemy_bases:
                target_base = min(self.enemy_bases, key=lambda p: self._manhattan(location, p))
            if target_base is not None:
                d = self._rough_direction(location, target_base)
                wall_break = d is not None and (location[0], location[1], d) in self.destructible
            if (enemy_base_hit or wall_break) and (base is None or base not in blast):
                escape = self._safe_escape_within(location, blast, self.BOMB_TIMER, danger)
                if escape is not None:
                    self._bomb_clock = 0
                    self.known_bombs[location] = {
                        "timer": self.BOMB_TIMER,
                        "own": True,
                        "last_step": self.last_step or 0,
                    }
                    self.escape_target = escape
                    self.escape_until_step = (self.last_step or 0) + self.BOMB_TIMER
                    return True
        return False


class ScriptedBaseAttack(BaseBomber):
    """Aggressive scripted-first base attacker.

    This approximates the opponent family we care about for semifinals:
    deterministic base pressure, frequent safe bomb attempts, and only a tiny
    item appetite when the base route is not immediately productive.
    """

    name = "scripted_base_attack"

    def __init__(self) -> None:
        super().__init__()
        self.period = 3
        self.ENEMY_BASE_VALUE = 230.0
        self.DIST_PENALTY = 0.45
        self.PATH_THREAT_PENALTY = 1.0
        self.CELL_THREAT_PENALTY = 3.0
        self.ITEM_VALUES = {"mission": 4.0, "resource": 1.5, "recon": 0.5}
        self.ENEMY_CHASE_VALUE = 10.0
        self.ENEMY_CHASE_RADIUS = 2


class SpawnRusher(BaseBomber):
    """Fixed-map rusher that commits to one nearby enemy base per game."""

    name = "spawn_rusher"

    def __init__(self) -> None:
        super().__init__()
        self.period = 4
        self.ENEMY_BASE_VALUE = 205.0
        self.DIST_PENALTY = 0.55
        self.ITEM_VALUES = {"mission": 5.0, "resource": 2.0, "recon": 1.0}
        self._preferred_base: tuple[int, int] | None = None

    def _choose_target(self, start, danger, low_health=False):
        if self.enemy_bases and not low_health:
            if self._preferred_base not in self.enemy_bases:
                anchor = self.base_location or start
                self._preferred_base = min(
                    self.enemy_bases,
                    key=lambda p: (self._manhattan(anchor, p), self._manhattan(start, p)),
                )
            distance, parent = self._dijkstra_distance_map(start, danger)
            best = None
            best_score = float("-inf")
            for cell in self._attack_cells_for(self._preferred_base):
                if cell not in distance:
                    continue
                score = 210.0 - distance[cell] - 0.15 * self.visit_count.get(cell, 0)
                if score > best_score:
                    best_score = score
                    best = cell
            if best is not None:
                return best, self._reconstruct_path(parent, start, best)
        return super()._choose_target(start, danger, low_health)


class OurBaseSieger(BaseBomber):
    """Stress opponent that targets team 0's Novice base when known."""

    name = "our_base_sieger"
    TEAM0_BASE = (13, 9)

    def __init__(self) -> None:
        super().__init__()
        self.period = 3
        self.ENEMY_BASE_VALUE = 240.0
        self.DIST_PENALTY = 0.50
        self.PATH_THREAT_PENALTY = 0.8
        self.CELL_THREAT_PENALTY = 2.5
        self.ITEM_VALUES = {"mission": 3.0, "resource": 1.0, "recon": 0.5}

    def _choose_target(self, start, danger, low_health=False):
        if self.TEAM0_BASE in self.enemy_bases and not low_health:
            distance, parent = self._dijkstra_distance_map(start, danger)
            best = None
            best_score = float("-inf")
            for cell in self._attack_cells_for(self.TEAM0_BASE):
                if cell not in distance:
                    continue
                score = 240.0 - distance[cell] - 0.10 * self.visit_count.get(cell, 0)
                if score > best_score:
                    best_score = score
                    best = cell
            if best is not None:
                return best, self._reconstruct_path(parent, start, best)
        return super()._choose_target(start, danger, low_health)


class SafeBaseBomber(BaseBomber):
    """Base bomber that keeps strong pressure without ignoring bomb danger."""

    name = "safe_base_bomber"

    def __init__(self) -> None:
        super().__init__()
        self.period = 4
        self.ENEMY_BASE_VALUE = 185.0
        self.DIST_PENALTY = 0.80
        self.PATH_THREAT_PENALTY = 5.0
        self.CELL_THREAT_PENALTY = 12.0
        self.ITEM_VALUES = {"mission": 10.0, "resource": 4.0, "recon": 1.0}


class ClusterHunter(StickyHunter):
    """Sticky hunter biased toward bombing remembered enemy clusters."""

    name = "cluster_hunter"

    def __init__(self) -> None:
        super().__init__()
        self.ENEMY_STALENESS = 12
        self.ENEMY_CHASE_VALUE = 42.0
        self.ENEMY_CHASE_RADIUS = 10
        self.PREDICTIVE_BOMB_RANGE = 2
        self.PREDICTIVE_WALK_HORIZON = 5
        self.opponent_walk_scale = 1.5


class CounterDefender(Defender):
    """Home-base defender that counter-chases nearby attackers."""

    name = "counter_defender"
    DEFEND_RADIUS = 7

    def __init__(self) -> None:
        super().__init__()
        self.BASE_DEFENSE_RADIUS = 8
        self.BASE_DEFENSE_HEALTH = 100
        self.ENEMY_CHASE_VALUE = 35.0
        self.ENEMY_CHASE_RADIUS = 8
        self.PATH_THREAT_PENALTY = 3.0
        self.CELL_THREAT_PENALTY = 8.0

    def _choose_target(self, start, danger, low_health=False):
        if self.base_location is not None and self.enemy_agents and not low_health:
            step = self.last_step if self.last_step is not None else 0
            nearby = [
                pos for pos, seen in self.enemy_agents.items()
                if step - int(seen) <= self.ENEMY_STALENESS
                and self._manhattan(pos, self.base_location) <= self.BASE_DEFENSE_RADIUS
            ]
            if nearby:
                target = min(nearby, key=lambda p: self._manhattan(start, p))
                distance, parent = self._bfs_distance_map(start, danger)
                if target in distance:
                    return target, self._reconstruct_path(parent, start, target)
        return super()._choose_target(start, danger, low_health)


class HybridCollectorAttacker(BaseRusher):
    """Cloud-like hybrid: collect on the route, then pressure bases."""

    name = "hybrid_collector"

    def __init__(self) -> None:
        super().__init__()
        self.ENEMY_BASE_VALUE = 165.0
        self.DIST_PENALTY = 0.90
        self.PATH_THREAT_PENALTY = 2.0
        self.CELL_THREAT_PENALTY = 5.0
        self.item_mission_value = 55.0
        self.item_resource_value = 20.0
        self.item_recon_value = 8.0
        self.ITEM_VALUES = {
            "mission": self.item_mission_value,
            "resource": self.item_resource_value,
            "recon": self.item_recon_value,
        }


class StrongMixedOpponent:
    """Per-game random switch among stronger semifinal proxy opponents."""

    name = "mixed_strong"

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)
        self._inner: object | None = None
        self._inner_name = "scripted_base_attack"

    def reset_for_game(self) -> None:
        choice = self.rng.choice((
            "scripted_base_attack",
            "spawn_rusher",
            "our_base_sieger",
            "safe_base_bomber",
            "cluster_hunter",
            "counter_defender",
            "hybrid_collector",
        ))
        self._inner_name = choice
        self._inner = make_opponent(choice, seed=self.rng.randint(0, 1 << 30))

    def __call__(self, obs: dict) -> int:
        if self._inner is None:
            self.reset_for_game()
        return self._inner(obs)


class MixedOpponent:
    """Picks one non-random scripted opponent at the start of each game."""

    name = "mixed"

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)
        self._inner: object | None = None
        self._inner_name = "greedy"

    def reset_for_game(self) -> None:
        choice = self.rng.choice((
            "greedy",
            "bomber",
            "bomber_fast",
            "defender",
            "hunter",
            "hunter_sticky",
            "rusher",
            "rusher_fast",
            "rusher_safe",
            "base_bomber",
        ))
        self._inner_name = choice
        self._inner = make_opponent(choice, seed=self.rng.randint(0, 1 << 30))

    def __call__(self, obs: dict) -> int:
        if self._inner is None:
            self.reset_for_game()
        return self._inner(obs)


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

OPPONENT_NAMES = (
    "random",
    "greedy",
    "bomber",
    "bomber_fast",
    "defender",
    "hunter",
    "hunter_sticky",
    "rusher",
    "rusher_fast",
    "rusher_safe",
    "base_bomber",
    "scripted_base_attack",
    "spawn_rusher",
    "our_base_sieger",
    "safe_base_bomber",
    "cluster_hunter",
    "counter_defender",
    "hybrid_collector",
    "mixed_strong",
    "mixed",
)


OPPONENT_SUITES = {
    "random": ["random"] * 5,
    "mixed": ["mixed"] * 5,
    "library": ["greedy", "bomber", "defender", "hunter", "rusher"],
    "cloudsuite": ["rusher", "hunter_sticky", "bomber_fast", "defender", "mixed"],
    "pressure2": ["rusher_fast", "rusher_safe", "hunter_sticky", "bomber_fast", "base_bomber"],
    "strong_realistic": [
        "scripted_base_attack",
        "spawn_rusher",
        "safe_base_bomber",
        "cluster_hunter",
        "hybrid_collector",
    ],
    "base_rush_exploit": [
        "our_base_sieger",
        "scripted_base_attack",
        "spawn_rusher",
        "base_bomber",
        "rusher_fast",
    ],
    "bracket_proxy": [
        "scripted_base_attack",
        "safe_base_bomber",
        "counter_defender",
        "hybrid_collector",
        "mixed_strong",
    ],
    "top_seed_proxy": [
        "our_base_sieger",
        "scripted_base_attack",
        "cluster_hunter",
        "safe_base_bomber",
        "hybrid_collector",
    ],
    "defense_trap": [
        "counter_defender",
        "cluster_hunter",
        "hunter_sticky",
        "defender",
        "safe_base_bomber",
    ],
}


def resolve_opponent_spec(spec: str) -> list[str]:
    """Resolve a named suite or explicit comma-separated opponent list."""

    key = spec.lower().strip()
    if key in OPPONENT_SUITES:
        return list(OPPONENT_SUITES[key])
    names = [n.strip().lower() for n in spec.split(",") if n.strip()]
    if len(names) == 1:
        names = names * 5
    if len(names) != 5:
        raise ValueError(f"need 5 opponent names (got {len(names)}): {names}")
    unknown = [n for n in names if n not in OPPONENT_NAMES]
    if unknown:
        raise ValueError(f"unknown opponent names: {unknown}")
    return names


def make_opponent(name: str, seed: int | None = None) -> OpponentFn:
    """Construct one opponent by name. ``seed`` only matters for stochastic types."""

    name = name.lower().strip()
    if name == "random":
        return RandomOpponent(seed)
    if name == "greedy":
        return GreedyCollector()
    if name == "bomber":
        period = int(os.environ.get("AE_BOMBER_PERIOD", "8"))
        return Bomber(period=period)
    if name == "bomber_fast":
        return FastBomber()
    if name == "defender":
        return Defender()
    if name == "hunter":
        return Hunter()
    if name == "hunter_sticky":
        return StickyHunter()
    if name == "rusher":
        return BaseRusher()
    if name == "rusher_fast":
        return FastRusher()
    if name == "rusher_safe":
        return SafeRusher()
    if name == "base_bomber":
        return BaseBomber()
    if name == "scripted_base_attack":
        return ScriptedBaseAttack()
    if name == "spawn_rusher":
        return SpawnRusher()
    if name == "our_base_sieger":
        return OurBaseSieger()
    if name == "safe_base_bomber":
        return SafeBaseBomber()
    if name == "cluster_hunter":
        return ClusterHunter()
    if name == "counter_defender":
        return CounterDefender()
    if name == "hybrid_collector":
        return HybridCollectorAttacker()
    if name == "mixed_strong":
        return StrongMixedOpponent(seed)
    if name == "mixed":
        return MixedOpponent(seed)
    raise ValueError(f"unknown opponent name: {name!r}")


def make_pool(names: list[str], seed: int | None = None) -> list[OpponentFn]:
    """Build a list of opponents in the requested order."""
    rng = random.Random(seed)
    return [make_opponent(n, seed=rng.randint(0, 1 << 30)) for n in names]
