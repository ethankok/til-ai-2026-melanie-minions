"""Scripted opponents for offline AE training and simulation.

Six callable opponents that take a raw observation dict and return an int
action 0..5. The signature matches the hidden cloud "policy interface"
where the only input is the partial observation. All opponents respect
``action_mask`` and never crash on edge-case observations.

The set is deliberately small and fast — these run inside tight inner
loops in `simulate.py` and `train_ppo.py`. No torch, no numpy heavy
allocations; every choice resolves in microseconds.

Why this set:
  - random       : matches `test_ae.py`'s local NPCs (uniform distribution).
  - greedy       : enemies that move toward visible items / their own base.
  - bomber       : periodically places bombs, otherwise greedy.
  - defender     : rarely strays far from its own base; bombs intruders.
  - hunter       : chases the nearest sighted enemy and bombs adjacent.
  - mixed        : per-game random switch among the above.

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

    # Force default heuristic parameters for opponents so they don't inherit our tuned settings
    m.ENEMY_BASE_VALUE = 80.0
    m.BASE_DEFENSE_RADIUS = 6
    m.DIST_PENALTY = 1.15
    m.PATH_THREAT_PENALTY = 2.0
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


class MixedOpponent:
    """Picks one of the above at the start of each game."""

    name = "mixed"

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)
        self._inner: object | None = None
        self._inner_name = "random"

    def reset_for_game(self) -> None:
        choice = self.rng.choice(("random", "greedy", "bomber", "defender", "hunter"))
        self._inner_name = choice
        self._inner = make_opponent(choice, seed=self.rng.randint(0, 1 << 30))

    def __call__(self, obs: dict) -> int:
        if self._inner is None:
            self.reset_for_game()
        return self._inner(obs)


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

OPPONENT_NAMES = ("random", "greedy", "bomber", "defender", "hunter", "mixed")


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
    if name == "defender":
        return Defender()
    if name == "hunter":
        return Hunter()
    if name == "mixed":
        return MixedOpponent(seed)
    raise ValueError(f"unknown opponent name: {name!r}")


def make_pool(names: list[str], seed: int | None = None) -> list[OpponentFn]:
    """Build a list of opponents in the requested order."""
    rng = random.Random(seed)
    return [make_opponent(n, seed=rng.randint(0, 1 << 30)) for n in names]
