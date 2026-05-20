"""Stateful frontier/objective planner for the AE task.

The qualifier evaluator calls this manager once per controlled-agent turn.  The
agent only sees egocentric view tensors, so this keeps a small belief map inside
``AEManager`` and plans over known safe cells.  It intentionally uses no external
runtime dependencies: the Docker image can stay unchanged for this heuristic
baseline.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import inf
import os
import time
from typing import Iterable


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


# Module-level cache for the opponent-model walk scale. PPO training
# instantiates hundreds of AEManager instances per epoch (one per scripted
# opponent per game); without caching we'd reload + re-print every time.
_OPPONENT_WALK_SCALE_CACHE: float | None = None


@dataclass(frozen=True)
class _LookaheadState:
    pos: tuple[int, int]
    direction: int
    bombs: tuple[tuple[int, int, int], ...]
    bombs_left: int
    health: int
    collected: frozenset[tuple[int, int]]
    score: float
    first_action: int | None
    tactical: bool
    path: tuple[int, ...]


class AEManager:
    """Rule-based autonomous-exploration planner.

    Priorities:
    1. Respect ``action_mask`` and frozen state.
    2. Maintain memory from agent/base views.
    3. Prefer scoring objectives: enemy base, mission, resource, recon.
    4. Otherwise explore frontiers and least-visited known cells.
    5. Place bombs only for visible tactical value with an escape path.
    """

    FORWARD = 0
    BACKWARD = 1
    LEFT = 2
    RIGHT = 3
    STAY = 4
    PLACE_BOMB = 5

    DIR_RIGHT = 0
    DIR_DOWN = 1
    DIR_LEFT = 2
    DIR_UP = 3

    DIR_DELTAS = {
        DIR_RIGHT: (1, 0),
        DIR_DOWN: (0, 1),
        DIR_LEFT: (-1, 0),
        DIR_UP: (0, -1),
    }

    OPPOSITE = {DIR_RIGHT: DIR_LEFT, DIR_DOWN: DIR_UP, DIR_LEFT: DIR_RIGHT, DIR_UP: DIR_DOWN}

    # View channels from til_environment.observation.ViewChannel.
    VISIBLE = 0
    WALL_RIGHT = 1
    WALL_DOWN = 2
    WALL_LEFT = 3
    WALL_UP = 4
    TILE_RECON = 6
    TILE_MISSION = 7
    TILE_RESOURCE = 8
    ENEMY_AGENT = 10
    ENEMY_BASE = 12
    DESTR_WALL_RIGHT = 13
    DESTR_WALL_DOWN = 14
    DESTR_WALL_LEFT = 15
    DESTR_WALL_UP = 16
    ALLY_BOMB = 17
    ENEMY_BOMB = 18
    ALLY_BOMB_TIMER = 19
    ENEMY_BOMB_TIMER = 20

    WALL_CHANNELS = {
        DIR_RIGHT: WALL_RIGHT,
        DIR_DOWN: WALL_DOWN,
        DIR_LEFT: WALL_LEFT,
        DIR_UP: WALL_UP,
    }
    DESTR_CHANNELS = {
        DIR_RIGHT: DESTR_WALL_RIGHT,
        DIR_DOWN: DESTR_WALL_DOWN,
        DIR_LEFT: DESTR_WALL_LEFT,
        DIR_UP: DESTR_WALL_UP,
    }
    ITEM_CHANNELS = {
        TILE_MISSION: ("mission", 5.0),
        TILE_RESOURCE: ("resource", 2.0),
        TILE_RECON: ("recon", 1.0),
    }
    ITEM_VALUES = {"mission": 50.0, "resource": 25.0, "recon": 10.0}

    # Game phase boundaries (200-step game)
    PHASE_COLLECT_END = 70    # Steps 0-70: prioritize collection
    PHASE_ATTACK_END = 140    # Steps 70-140: balanced attack + collection
    # Steps 140-200: cleanup / mop up remaining items, avoid risky plays

    GRID_SIZE = 16
    # Bomb timer matches til_environment/bomberman_config.yaml (entities.bomb.timer).
    # Phase order each round is: place → move → detonation → upkeep, so a bomb
    # placed at step N detonates after the agent's movement at step N+timer.
    BOMB_TIMER = 3
    BOMB_RADIUS = 2
    # How long we still consider an enemy-agent sighting "threatening" in steps.
    ENEMY_STALENESS = 3
    # Reachability radius (BFS steps from blast cell) for predictive bomb hits.
    # Range 1 = enemies immediately adjacent to a blast cell. Range 2 was too
    # generous in random-opponent eval (fired almost every turn with bombs).
    PREDICTIVE_BOMB_RANGE = 1
    # Tiles respawn after this many steps per env config (env.tile_respawn_steps).
    # Env config note: 40 is the *max*; actual respawn is randomly generated
    # via perlin noise, so a collected tile is back well before step+40 in
    # expectation. Reconsider at 20 with a 0.5x discount.
    TILE_RESPAWN_STEPS = 20
    # Soft cap on how long we keep an unseen enemy_agent record around. Past
    # this we drop the entry entirely (planning code also has its own
    # staleness filter at ENEMY_STALENESS for threat scoring).
    ENEMY_AGENT_MEMORY_STEPS = 30
    # Below this health the agent prefers safe cells over aggressive plays.
    LOW_HEALTH_THRESHOLD = 20
    # Soft-threat scoring weights. v3 dropped these to 1.0/3.0 and lost ~0.07
    # locally vs v2's 3.0/8.0. v3b splits the difference — penalty is real but
    # not so heavy that we route around harmless random opponents.
    PATH_THREAT_PENALTY = 2.0
    CELL_THREAT_PENALTY = 5.0
    # Distance (Manhattan) within which an enemy near our base becomes a defense target.
    BASE_DEFENSE_RADIUS = 4
    # Tier-1 #3: how long an enemy stays frozen after a kill (matches env config
    # entities.agent.freeze_turns). After unfreezing they respawn at the same
    # cell. We use this to plant follow-up bombs timed for the respawn window.
    ENEMY_FREEZE_DURATION = 3
    # Tier-1 #2: when our base HP drops below this AND an enemy is within
    # BASE_DEFENSE_RADIUS, defense becomes the top objective. Set high enough
    # that we don't preempt every offensive opportunity, low enough that we
    # never let the base ride at <40 HP.
    BASE_DEFENSE_HEALTH = 60
    # Tier-1 #4: realistic shared-credit weights for cloud's 6-team game.
    # destroy_enemy_base raw value is 50 but other teams will share the kill;
    # mean realistic share ~30. attack_kill raw value 30 → realistic ~12.
    SHARED_CREDIT_BASE_VALUE = 30.0
    SHARED_CREDIT_KILL_VALUE = 12.0
    # Tier-1 #7: window over which we credit a random-walk enemy with possibly
    # walking into our blast. Matches BOMB_TIMER + 1; one tick of slack for
    # detonation ordering.
    PREDICTIVE_WALK_HORIZON = 4

    def __init__(self):
        self.grid_size = self.GRID_SIZE
        self.last_step: int | None = None
        self.turn_counter = 0
        self.mcts_enabled = _env_flag("AE_MCTS", False)
        self.mcts_depth = max(1, min(8, _env_int("AE_MCTS_DEPTH", 3)))
        self.mcts_width = max(12, min(512, _env_int("AE_MCTS_WIDTH", 24)))
        self.mcts_min_score = _env_float("AE_MCTS_MIN_SCORE", 12.0)
        # v2: hard per-call latency budget. Cloud killed v1 because DEPTH=5
        # WIDTH=96 with no cap ran 1.2-2.4s/tick; budget is ~600ms/tick on
        # cloud. 80ms gives the rest of the pipeline (encoding, policy
        # forward, hybrid logic) ~500ms headroom.
        self.mcts_time_budget_s = max(0.005, _env_float("AE_MCTS_BUDGET_MS", 80.0) / 1000.0)
        # v2: emit per-call timing to stdout so we can confirm budget
        # compliance in `til test` before submitting.
        self.mcts_log_timing = _env_flag("AE_MCTS_LOG_TIMING", True)
        self.last_lookahead_score = -inf
        self.last_lookahead_path: tuple[int, ...] = ()
        # Tier-1 toggles. Bisect on 100-200 round local sims (17 May 2026)
        # picked the clean winning subset: #3, #6, #7 default ON; #2, #4
        # default OFF. Cloud A/B should validate before flipping more on.
        # Override any individually with `AE_TIER1_*=1` / `=0`.
        self.tier1_defense_priority = _env_flag("AE_TIER1_DEFENSE", False)
        self.tier1_repeat_kill = _env_flag("AE_TIER1_REPEAT_KILL", True)
        self.tier1_shared_credit = _env_flag("AE_TIER1_SHARED_CREDIT", False)
        self.tier1_no_stay_penalty = _env_flag("AE_TIER1_NO_STAY_PENALTY", True)
        self.tier1_predictive_walk = _env_flag("AE_TIER1_PREDICTIVE_WALK", True)
        # Tier-1 #1: load the offline playbook if present. The lookup is a
        # cheap dict access so we hit it on every tick before falling
        # through to the heuristic. Disabled in two cases: env var set, or
        # no .npz on disk. Import is local so heuristic-only paths never
        # pay for a playbook load.
        try:
            from playbook import get_playbook  # noqa: WPS433
            self.playbook = get_playbook()
        except Exception as exc:  # noqa: BLE001
            print(f"[AEManager] playbook import failed: {exc}", flush=True)
            self.playbook = None
        # Tier-2 #8: load opponent model if present so predictive-walk
        # credit reflects measured opponent behavior instead of a hand-tuned
        # constant. The model is a single scalar (mean walk distance per
        # step) — the manager scales its predictive-walk hit probabilities
        # by min(1, walk_distance / 0.5) so values <0.5 (less mobile
        # opponents) get downweighted and >0.5 (more mobile) get upweighted.
        self.opponent_walk_scale = self._load_opponent_walk_scale()
        self.BASE_DEFENSE_HEALTH = _env_float("AE_BASE_DEFENSE_HEALTH", 60.0)
        self.BASE_DEFENSE_RADIUS = _env_int("AE_BASE_DEFENSE_RADIUS", 4)
        self.ENEMY_BASE_VALUE = _env_float("AE_ENEMY_BASE_VALUE", 80.0)
        self.DIST_PENALTY = _env_float("AE_DIST_PENALTY", 1.15)
        self.PATH_THREAT_PENALTY = _env_float("AE_PATH_THREAT_PENALTY", 2.0)
        self.ENEMY_CHASE_VALUE = _env_float("AE_ENEMY_CHASE_VALUE", 0.0)
        self.ENEMY_CHASE_RADIUS = _env_int("AE_ENEMY_CHASE_RADIUS", 4)
        self.item_mission_value = _env_float("AE_ITEM_MISSION_VALUE", 50.0)
        self.item_resource_value = _env_float("AE_ITEM_RESOURCE_VALUE", 25.0)
        self.item_recon_value = _env_float("AE_ITEM_RECON_VALUE", 10.0)
        self.ITEM_VALUES = {
            "mission": self.item_mission_value,
            "resource": self.item_resource_value,
            "recon": self.item_recon_value,
        }
        self._reset_memory()

    # ------------------------------------------------------------------
    # Public policy
    # ------------------------------------------------------------------
    def ae(self, observation: dict) -> int:
        """Choose the next action for the controlled agent."""

        self.turn_counter += 1
        step = self._as_int(observation.get("step"), default=self.turn_counter)
        if self.last_step is None or step == 0 or step < self.last_step:
            self._reset_memory()
        self._age_bombs(step)
        # Per-turn caches; walls/bombs can only change once per turn from new obs.
        self._blast_cache = {}
        self.last_step = step
        self.team_bombs = self._as_int(observation.get("team_bombs"), default=0)

        location = self._location(observation.get("location"))
        direction = self._as_int(observation.get("direction"), default=self.DIR_RIGHT) % 4

        # If step is 0, check if this is the Novice fixed map
        if step == 0 and location is not None:
            base_loc = self._location(observation.get("base_location"))
            if base_loc is not None:
                try:
                    from novice_map_data import BASE_LOCATIONS, STARTING_LOCATIONS, WALLS, DESTRUCTIBLE, STATIC_ENTITIES
                    for i in range(6):
                        if tuple(base_loc) == tuple(BASE_LOCATIONS[i]):
                            self.is_fixed_novice_map = True
                            self.fixed_team_idx = i
                            self.dijkstra_bomb_cost = 5.0
                            self.playbook = None
                            if _env_float("AE_ENEMY_BASE_VALUE", -999.0) == -999.0:
                                self.ENEMY_BASE_VALUE = 130.0
                            if _env_int("AE_BASE_DEFENSE_RADIUS", -999) == -999:
                                self.BASE_DEFENSE_RADIUS = 6
                            if _env_float("AE_ENEMY_CHASE_VALUE", -999.0) == -999.0:
                                self.ENEMY_CHASE_VALUE = 0.0
                            if _env_int("AE_ENEMY_CHASE_RADIUS", -999) == -999:
                                self.ENEMY_CHASE_RADIUS = 4
                            
                            # Pre-populate seen, walls, destructible, enemy bases, and static items
                            self.seen = {(x, y) for x in range(16) for y in range(16)}
                            self.walls = set(WALLS)
                            self.destructible = set(DESTRUCTIBLE)
                            self.enemy_bases = {tuple(BASE_LOCATIONS[j]): 0 for j in range(6) if j != i}
                            self.last_seen_items = {tuple(pos): (kind, 0) for kind, pos in STATIC_ENTITIES}
                            break
                except ImportError:
                    pass

        frozen_ticks = self._as_int(observation.get("frozen_ticks"), default=0)
        self.health = self._as_int(observation.get("health"), default=60)
        self.base_health = self._as_int(observation.get("base_health"), default=100)

        self._update_memory(observation, step, location, direction)

        if location is not None:
            self.visit_count[location] = self.visit_count.get(location, 0) + 1
            self.recent_locations.append(location)
            if len(self.recent_locations) > 10:
                self.recent_locations.pop(0)

        if frozen_ticks > 0:
            return self._first_legal(observation, [self.STAY, self.LEFT, self.RIGHT, self.FORWARD, self.BACKWARD])

        if location is None:
            return self._fallback_action(observation, None, direction, None)

        # Tier-1 #1: playbook override. After belief is updated and frozen-
        # state is handled, before the heuristic does any planning, check
        # whether we have a high-confidence pre-mined action for this
        # (location, direction, step). Only used when the action is legal
        # under the current mask.
        if self.playbook is not None:
            pb_action = self.playbook.lookup(location, direction, step)
            if pb_action is not None and self._legal(observation, pb_action):
                # Bomb-via-playbook still needs a valid escape so we don't
                # walk into a bomb we can't get out of (the offline trace
                # learned an escape path on the same map; we re-verify here
                # cheaply against current belief state).
                if pb_action == self.PLACE_BOMB:
                    if self._as_int(observation.get("team_bombs"), default=0) <= 0:
                        pass
                    else:
                        bomb_blast = self._blast_cells(location)
                        base = self.base_location or self._location(observation.get("base_location"))
                        if base is not None and base in bomb_blast:
                            pass
                        else:
                            escape = self._safe_escape_within(
                                location, bomb_blast, self.BOMB_TIMER
                            )
                            if escape is not None:
                                self.known_bombs[location] = {
                                    "timer": self.BOMB_TIMER,
                                    "own": True,
                                    "last_step": step,
                                }
                                self.escape_target = escape
                                self.escape_until_step = step + self.BOMB_TIMER
                                return self.PLACE_BOMB
                else:
                    return pb_action

        danger = self._danger_cells()
        low_health = self.health < self.LOW_HEALTH_THRESHOLD

        escape_path = None
        if False:
            target, path = self.escape_target, escape_path
        else:
            # Fast path: an obviously-dominant action skips full candidate scoring.
            dominant = self._try_dominant_action(observation, location, direction, danger, low_health)
            if dominant is not None:
                return dominant
            lookahead = self._tactical_lookahead_action(observation, location, direction)
            if lookahead is not None:
                return lookahead
            target, path = self._choose_target(location, danger, low_health)

        if escape_path is None and self._should_place_bomb(observation, location, target, danger):
            return self.PLACE_BOMB

        preferred = self._action_for_path(location, direction, path)
        if preferred is not None and self._legal(observation, preferred):
            return preferred

        return self._fallback_action(observation, location, direction, target)

    # ------------------------------------------------------------------
    # Memory and projection
    # ------------------------------------------------------------------
    @staticmethod
    def _load_opponent_walk_scale() -> float:
        """Return the predictive-walk multiplier from the offline opponent model.

        Tier-2 #8: if `ae/models/opponent_model.json` (or
        `/workspace/models/opponent_model.json`) exists, read its
        ``weighted_walk_distance`` field and convert to a 0..2 scale where
        1.0 == default (matches the manager's hand-tuned formula calibrated
        against random walkers, ~0.5 cells/step). Disabled by setting
        ``AE_USE_OPPONENT_MODEL=0``.

        Result is cached per-process via a module-level guard so PPO training
        loops that instantiate hundreds of AEManager instances don't repeat
        the load message and disk read for every opponent.
        """
        global _OPPONENT_WALK_SCALE_CACHE
        if _OPPONENT_WALK_SCALE_CACHE is not None:
            return _OPPONENT_WALK_SCALE_CACHE

        if os.environ.get("AE_USE_OPPONENT_MODEL", "1").strip().lower() in {"0", "false", "no", "off"}:
            _OPPONENT_WALK_SCALE_CACHE = 1.0
            return 1.0
        candidates = [
            os.environ.get("AE_OPPONENT_MODEL_PATH"),
            os.path.join(os.path.dirname(__file__), "models", "opponent_model.json"),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "opponent_model.json"),
            "/workspace/models/opponent_model.json",
        ]
        for path in candidates:
            if not path or not os.path.exists(path):
                continue
            try:
                import json as _json  # local; cheap on Python 3
                with open(path, "r", encoding="utf-8") as f:
                    data = _json.load(f)
                walk = float(data.get("weighted_walk_distance", 0.5))
                # Normalize: 0.5 cells/step is the "neutral" expected value
                # for a uniform random walk. Cap to [0.4, 2.0] so a
                # weirdly-fit model can't blow the heuristic up.
                scale = max(0.4, min(2.0, walk / 0.5))
                print(
                    f"[AEManager] opponent model loaded ({path}); "
                    f"walk_distance={walk:.3f} scale={scale:.2f}",
                    flush=True,
                )
                _OPPONENT_WALK_SCALE_CACHE = scale
                return scale
            except Exception as exc:  # noqa: BLE001
                print(f"[AEManager] opponent model load failed at {path}: {exc}", flush=True)
        _OPPONENT_WALK_SCALE_CACHE = 1.0
        return 1.0

    def _reset_memory(self) -> None:
        self.grid_size = self.GRID_SIZE
        self.seen: set[tuple[int, int]] = set()
        self.walls: set[tuple[int, int, int]] = set()
        self.destructible: set[tuple[int, int, int]] = set()
        self.last_seen_items: dict[tuple[int, int], tuple[str, int]] = {}
        # Items observed disappearing (presumed collected); reconsider after respawn.
        self.collected_items: dict[tuple[int, int], tuple[str, int]] = {}
        self.enemy_bases: dict[tuple[int, int], int] = {}
        self.enemy_agents: dict[tuple[int, int], int] = {}
        self.known_bombs: dict[tuple[int, int], dict[str, int | bool]] = {}
        self.visit_count: dict[tuple[int, int], int] = {}
        self.recent_locations: list[tuple[int, int]] = []
        self.escape_target: tuple[int, int] | None = None
        self.escape_until_step: int | None = None
        self.base_location: tuple[int, int] | None = None
        self.health: int = 60
        self.base_health: int = 100
        self.last_lookahead_score = -inf
        self.last_lookahead_path = ()
        # Per-turn blast cell cache; cleared at the start of every ae() call.
        self._blast_cache: dict[tuple[int, int], frozenset[tuple[int, int]]] = {}
        # Tier-1 #3: track recent kills so we can plant a follow-up bomb
        # timed for the enemy's respawn. (pos, unfreeze_step).
        self.recent_kills: list[tuple[tuple[int, int], int]] = []
        # Tier-1 #2: per-step record of the last health value we saw on our
        # base so we can detect "took damage this step" reliably.
        self.last_base_health: int = 100
        self.last_step = None
        self.is_fixed_novice_map = False
        self.fixed_team_idx = None
        self.dijkstra_bomb_cost = 5.0
        # Track our item collection for respawn scheduling
        self.items_collected_steps: dict[tuple[int, int], int] = {}

    def _update_memory(
        self,
        observation: dict,
        step: int,
        location: tuple[int, int] | None,
        direction: int,
    ) -> None:
        self.base_damaged_this_step = (self.base_health < getattr(self, "last_base_health", 100))
        self.last_base_health = self.base_health

        if location is not None:
            self._infer_grid_size(location)
            self.seen.add(location)

        agent_view = observation.get("agent_viewcone")
        if location is not None and agent_view is not None:
            self._project_agent_viewcone(agent_view, location, direction, step)

        base_view = observation.get("base_viewcone")
        base_location = self._location(observation.get("base_location"))
        if base_location is not None:
            self.seen.add(base_location)
            self.base_location = base_location
        if base_location is not None and base_view is not None:
            self._infer_grid_size(base_location)
            self._project_centered_view(base_view, base_location, step)

    def _project_agent_viewcone(
        self,
        view: list,
        location: tuple[int, int],
        direction: int,
        step: int,
    ) -> None:
        height = len(view)
        width = len(view[0]) if height else 0
        if not height or not width:
            return
        origin_forward = 2 if height >= 7 else height // 2
        origin_side = 2 if width >= 5 else width // 2

        for row in range(height):
            for col in range(width):
                cell = view[row][col]
                if not self._channel_on(cell, self.VISIBLE):
                    continue
                forward = row - origin_forward
                side = col - origin_side
                world = self._relative_to_world(location, direction, forward, side)
                if self._in_bounds(world):
                    self._ingest_visible_cell(world, cell, step)

    def _project_centered_view(self, view: list, center: tuple[int, int], step: int) -> None:
        height = len(view)
        width = len(view[0]) if height else 0
        if not height or not width:
            return
        center_row = height // 2
        center_col = width // 2
        for row in range(height):
            for col in range(width):
                cell = view[row][col]
                if not self._channel_on(cell, self.VISIBLE):
                    continue
                world = (center[0] + row - center_row, center[1] + col - center_col)
                if self._in_bounds(world):
                    self._ingest_visible_cell(world, cell, step)

    def _ingest_visible_cell(self, world: tuple[int, int], cell: list, step: int) -> None:
        self.seen.add(world)
        self._infer_grid_size(world)

        for d, channel in self.WALL_CHANNELS.items():
            edge = (world[0], world[1], d)
            if self._channel_on(cell, channel):
                self.walls.add(edge)
            else:
                self.walls.discard(edge)
        for d, channel in self.DESTR_CHANNELS.items():
            edge = (world[0], world[1], d)
            if self._channel_on(cell, channel):
                self.destructible.add(edge)
            else:
                self.destructible.discard(edge)

        visible_item = None
        for channel, (kind, _reward) in self.ITEM_CHANNELS.items():
            if self._channel_on(cell, channel):
                visible_item = kind
                break
        if visible_item is None:
            prev = self.last_seen_items.pop(world, None)
            # If an item was here last time and is now gone, it was collected
            # (by us or another agent). Remember the kind + step so we can
            # reconsider it as a target once the respawn timer elapses.
            if prev is not None:
                self.collected_items[world] = (prev[0], step)
                self.items_collected_steps[world] = step
        else:
            self.last_seen_items[world] = (visible_item, step)
            # If we see it again, it must have respawned; clear collected entry.
            self.collected_items.pop(world, None)
            self.items_collected_steps.pop(world, None)

        if self._channel_on(cell, self.ENEMY_BASE):
            self.enemy_bases[world] = step
        else:
            self.enemy_bases.pop(world, None)
        if self._channel_on(cell, self.ENEMY_AGENT):
            self.enemy_agents[world] = step
        else:
            self.enemy_agents.pop(world, None)

        ally_bomb = self._channel_on(cell, self.ALLY_BOMB)
        enemy_bomb = self._channel_on(cell, self.ENEMY_BOMB)
        if ally_bomb or enemy_bomb:
            timer_channel = self.ALLY_BOMB_TIMER if ally_bomb else self.ENEMY_BOMB_TIMER
            timer = max(1, self._as_int(self._channel_value(cell, timer_channel), default=self.BOMB_TIMER))
            self.known_bombs[world] = {"timer": timer, "own": ally_bomb, "last_step": step}
        else:
            # Visible and no bomb means stale bomb memory can be cleared.
            self.known_bombs.pop(world, None)

    def _relative_to_world(
        self,
        location: tuple[int, int],
        direction: int,
        forward: int,
        side: int,
    ) -> tuple[int, int]:
        x, y = location
        if direction == self.DIR_RIGHT:
            return (x + forward, y + side)
        if direction == self.DIR_DOWN:
            return (x - side, y + forward)
        if direction == self.DIR_LEFT:
            return (x - forward, y - side)
        return (x + side, y - forward)

    def _infer_grid_size(self, coord: tuple[int, int]) -> None:
        # Default is 16.  If a hidden map is larger, expand rather than clipping
        # belief-map planning to the novice size.
        self.grid_size = max(self.grid_size, coord[0] + 1, coord[1] + 1)

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------
    def _game_phase(self) -> str:
        """Return the current game phase based on step count."""
        step = self.last_step if self.last_step is not None else 0
        if step < self.PHASE_COLLECT_END:
            return "collect"
        if step < self.PHASE_ATTACK_END:
            return "attack"
        return "cleanup"

    def _choose_target(
        self,
        start: tuple[int, int],
        danger: set[tuple[int, int]],
        low_health: bool = False,
    ) -> tuple[tuple[int, int] | None, list[tuple[int, int]] | None]:
        threats = self._enemy_threat_cells()
        step = self.last_step if self.last_step is not None else 0
        is_fixed = getattr(self, "is_fixed_novice_map", False)
        phase = self._game_phase()

        # Single multi-source BFS gives distance to every reachable cell at
        # roughly the cost of one of the old per-target BFS calls.
        if is_fixed:
            distance, parent = self._dijkstra_distance_map(start, danger)
        else:
            distance, parent = self._bfs_distance_map(start, danger)

        candidates: list[tuple[float, tuple[int, int]]] = []

        # ── Base defense (critical priority) ──
        # On fixed map: absolute defense when base is in danger
        defense_emergency = False
        if self.base_location is not None:
            if is_fixed:
                # Absolute defense when base health is critical
                if self.base_health < 40:
                    for pos, last_seen in self.enemy_agents.items():
                        if step - int(last_seen) > self.ENEMY_STALENESS:
                            continue
                        if self._manhattan(pos, self.base_location) <= self.BASE_DEFENSE_RADIUS + 2:
                            defense_emergency = True
                            candidates.append((200.0, pos))
                # Normal defense: target enemies near base
                elif self.base_damaged_this_step or self.base_health < 80:
                    for pos, last_seen in self.enemy_agents.items():
                        if step - int(last_seen) > self.ENEMY_STALENESS:
                            continue
                        dist_to_base = self._manhattan(pos, self.base_location)
                        if dist_to_base <= 2 or (self.base_damaged_this_step and dist_to_base <= self.BASE_DEFENSE_RADIUS):
                            defense_emergency = True
                            candidates.append((150.0, pos))
            elif (
                self.tier1_defense_priority
                and self.base_health < self.BASE_DEFENSE_HEALTH
            ):
                for pos, last_seen in self.enemy_agents.items():
                    if step - int(last_seen) > self.ENEMY_STALENESS:
                        continue
                    if self._manhattan(pos, self.base_location) <= self.BASE_DEFENSE_RADIUS:
                        defense_emergency = True
                        candidates.append((150.0, pos))

        # ── Combat targets (phase-aware) ──
        if not low_health and not defense_emergency:
            for pos in self.enemy_bases:
                if is_fixed:
                    # Phase-dependent enemy base value
                    if phase == "collect":
                        value = 80.0   # Lower priority during collection phase
                    elif phase == "attack":
                        value = 160.0  # High priority during attack phase
                    else:
                        value = 100.0  # Moderate in cleanup
                else:
                    value = 35.0 if self.tier1_shared_credit else self.ENEMY_BASE_VALUE
                candidates.append((value, pos))
            # Base defense: target enemies near our base (active attack/defense)
            if self.base_location is not None:
                for pos, last_seen in self.enemy_agents.items():
                    if step - int(last_seen) > self.ENEMY_STALENESS:
                        continue
                    if self._manhattan(pos, self.base_location) <= self.BASE_DEFENSE_RADIUS:
                        candidates.append((60.0, pos))
            # Target nearby enemy agents aggressively (chase & kill) if enabled
            if self.ENEMY_CHASE_VALUE > 0.0:
                for pos, last_seen in self.enemy_agents.items():
                    if step - int(last_seen) > 1:
                        continue
                    if self._manhattan(start, pos) <= self.ENEMY_CHASE_RADIUS:
                        candidates.append((self.ENEMY_CHASE_VALUE, pos))

        # ── Items (phase-aware values on fixed map) ──
        for pos, (kind, _step) in self.last_seen_items.items():
            if is_fixed:
                if phase == "collect":
                    # Boost mission value heavily during collection
                    phase_values = {"mission": 65.0, "resource": 30.0, "recon": 12.0}
                elif phase == "attack":
                    phase_values = {"mission": 45.0, "resource": 25.0, "recon": 8.0}
                else:
                    # Cleanup: grab whatever is nearby
                    phase_values = {"mission": 55.0, "resource": 30.0, "recon": 15.0}
                candidates.append((phase_values.get(kind, 10.0), pos))
            else:
                candidates.append((self.ITEM_VALUES.get(kind, 1.0), pos))

        # Respawn awareness: items we saw get collected become candidates
        # again once tile_respawn_steps have elapsed.
        for pos, (kind, collected_step) in self.collected_items.items():
            if step - collected_step >= self.TILE_RESPAWN_STEPS:
                # On fixed map, respawned items are just as valuable as fresh ones
                # (respawn is deterministic, same position, same type).
                if is_fixed:
                    discount = 0.85
                else:
                    discount = 0.5
                candidates.append((discount * self.ITEM_VALUES.get(kind, 1.0), pos))

        # ── Frontier and anti-stall (disabled on fixed map) ──
        if not is_fixed:
            # Weight frontier cells by how much new area they likely reveal.
            for pos in self._frontier_cells():
                unseen_neighbors = sum(1 for n in self._raw_neighbors(pos) if n not in self.seen)
                candidates.append((4.0 + 1.0 * unseen_neighbors, pos))

            # Anti-stall fallback: known safe low-visit cells.
            for pos in self.seen:
                if pos != start:
                    candidates.append((2.0 - 0.08 * self.visit_count.get(pos, 0), pos))
        else:
            # On fixed map, only add a very light anti-stall for truly stuck cases
            if self._stuck_recently():
                for pos in self.seen:
                    if pos != start and self.visit_count.get(pos, 0) < 2:
                        candidates.append((1.0, pos))

        best_target = None
        best_score = -inf
        for base_value, pos in candidates:
            if pos == start or pos not in distance:
                continue
            dist = distance[pos]
            score = base_value - self.DIST_PENALTY * dist - 0.25 * self.visit_count.get(pos, 0)
            if pos in self.recent_locations[-4:]:
                score -= 2.0
            # Threats: penalize paths that brush near recently-seen enemies,
            # but don't penalize when the *target itself* is the enemy (we want
            # to attack them) or an enemy base.
            if pos not in self.enemy_bases and pos not in self.enemy_agents:
                path_threat = 0
                cursor = pos
                while cursor is not None and cursor != start:
                    if cursor in threats:
                        path_threat += 1
                    cursor = parent.get(cursor)
                # On fixed map in cleanup phase, reduce threat penalty
                # (we need to be more aggressive about collecting remaining items)
                threat_mult = 1.0
                if is_fixed and phase == "cleanup":
                    threat_mult = 0.5
                score -= self.PATH_THREAT_PENALTY * path_threat * threat_mult
            if score > best_score:
                best_score = score
                best_target = pos

        if best_target is None:
            self.current_path = None
            return None, None
        path = self._reconstruct_path(parent, start, best_target)
        self.current_path = path
        return best_target, path

    def _bfs(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
        danger: set[tuple[int, int]] | None = None,
        allow_goal_unseen: bool = False,
    ) -> list[tuple[int, int]] | None:
        if start == goal:
            return [start]
        danger = danger or set()
        queue = deque([start])
        parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}

        while queue:
            current = queue.popleft()
            for nxt in self._neighbors(current):
                if nxt in parent:
                    continue
                if nxt in danger and nxt != goal:
                    continue
                if nxt not in self.seen and not (allow_goal_unseen and nxt == goal):
                    continue
                parent[nxt] = current
                if nxt == goal:
                    path = [goal]
                    while path[-1] != start:
                        path.append(parent[path[-1]])  # type: ignore[arg-type]
                    path.reverse()
                    return path
                queue.append(nxt)
        return None

    def _neighbors(self, pos: tuple[int, int]) -> Iterable[tuple[int, int]]:
        for direction, (dx, dy) in self.DIR_DELTAS.items():
            nxt = (pos[0] + dx, pos[1] + dy)
            if self._in_bounds(nxt) and not self._edge_blocked(pos, direction):
                yield nxt

    def _bfs_distance_map(
        self,
        start: tuple[int, int],
        danger: set[tuple[int, int]] | None = None,
    ) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], tuple[int, int] | None]]:
        """Single BFS that returns distances and parents for every reachable cell.

        Replaces N per-target BFS calls with one. Cells are reachable only if
        they're in ``self.seen``; ``danger`` cells are not traversed.
        """

        danger = danger or set()
        distance: dict[tuple[int, int], int] = {start: 0}
        parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
        queue = deque([start])
        while queue:
            current = queue.popleft()
            for nxt in self._neighbors(current):
                if nxt in distance:
                    continue
                if nxt in danger:
                    continue
                if nxt not in self.seen:
                    continue
                distance[nxt] = distance[current] + 1
                parent[nxt] = current
                queue.append(nxt)
        return distance, parent

    def _edge_destructible(self, pos: tuple[int, int], direction: int) -> bool:
        if (pos[0], pos[1], direction) in self.destructible:
            return True
        dx, dy = self.DIR_DELTAS[direction]
        other = (pos[0] + dx, pos[1] + dy)
        opposite = self.OPPOSITE[direction]
        return (other[0], other[1], opposite) in self.destructible

    def _dijkstra_distance_map(
        self,
        start: tuple[int, int],
        danger: set[tuple[int, int]] | None = None,
    ) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int], tuple[int, int] | None]]:
        import heapq
        danger = danger or set()
        distance: dict[tuple[int, int], float] = {start: 0.0}
        parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
        pq = [(0.0, start)]
        while pq:
            cost, current = heapq.heappop(pq)
            if cost > distance.get(current, float('inf')):
                continue
            for direction, (dx, dy) in self.DIR_DELTAS.items():
                nxt = (current[0] + dx, current[1] + dy)
                if not self._in_bounds(nxt):
                    continue
                if nxt in danger:
                    continue
                if nxt not in self.seen:
                    continue
                
                # Check walls
                if not self._edge_blocked(current, direction):
                    step_cost = 1.0
                elif self._edge_destructible(current, direction):
                    step_cost = 1.0 + self.dijkstra_bomb_cost
                else:
                    continue
                
                new_cost = cost + step_cost
                if new_cost < distance.get(nxt, float('inf')):
                    distance[nxt] = new_cost
                    parent[nxt] = current
                    heapq.heappush(pq, (new_cost, nxt))
        return distance, parent

    def _reconstruct_path(
        self,
        parent: dict[tuple[int, int], tuple[int, int] | None],
        start: tuple[int, int],
        goal: tuple[int, int],
    ) -> list[tuple[int, int]] | None:
        if goal not in parent:
            return None
        path = [goal]
        while path[-1] != start:
            prev = parent.get(path[-1])
            if prev is None:
                if path[-1] == start:
                    break
                return None
            path.append(prev)
        path.reverse()
        return path

    def _extended_blast(
        self,
        blast: Iterable[tuple[int, int]],
        radius: int,
    ) -> set[tuple[int, int]]:
        """Cells reachable in BFS distance ≤ radius from any blast cell."""

        distance: dict[tuple[int, int], int] = {pos: 0 for pos in blast}
        queue = deque(distance)
        while queue:
            current = queue.popleft()
            if distance[current] >= radius:
                continue
            for nxt in self._neighbors(current):
                if nxt in distance:
                    continue
                if nxt not in self.seen:
                    continue
                distance[nxt] = distance[current] + 1
                queue.append(nxt)
        return set(distance)

    def _try_dominant_action(
        self,
        observation: dict,
        location: tuple[int, int],
        direction: int,
        danger: set[tuple[int, int]],
        low_health: bool,
    ) -> int | None:
        """Skip full candidate scoring when an obvious move dominates.

        Three short-circuits, all with the same safety constraints as the slow
        path:

        - an enemy base sits in the immediate bomb blast → ``PLACE_BOMB``;
        - an enemy agent we saw THIS step sits in the blast and we have a
          verified escape → ``PLACE_BOMB`` (opportunistic kill; +20 attack
          damage + potential +15 kill bonus is the biggest single reward swing
          in the game outside destroying a base, and the slow path was missing
          these because BFS-targeting routed us toward items first);
        - a mission tile is one step away through a clear edge → step toward it.
        """

        # Hunting bombs are off when health is critically low.
        step = self.last_step if self.last_step is not None else 0
        if (not low_health
                and self._legal(observation, self.PLACE_BOMB)
                and self._as_int(observation.get("team_bombs"), default=0) > 0
                and location not in danger):
            bomb_blast = self._blast_cells(location)
            base_loc = self.base_location
            base_safe = base_loc is None or base_loc not in bomb_blast

            enemy_base_hit = any(pos in bomb_blast for pos in self.enemy_bases)
            # Fresh enemy_agent in blast — only this-step sightings to avoid
            # speculative kills (random opponents wander; a 2-step-old sighting
            # is no longer reliable). Stricter than the slow path's "step-1"
            # tolerance because this is a *dominant-action* shortcut.
            enemy_agent_hit = any(
                int(last_seen) == step and pos in bomb_blast
                for pos, last_seen in self.enemy_agents.items()
            )

            if base_safe and (enemy_base_hit or enemy_agent_hit):
                escape = self._safe_escape_within(location, bomb_blast, self.BOMB_TIMER)
                if escape is not None:
                    # Mirror the side effects of _should_place_bomb so escape
                    # mode kicks in next turn.
                    self.known_bombs[location] = {
                        "timer": self.BOMB_TIMER,
                        "own": True,
                        "last_step": self.last_step or 0,
                    }
                    self.escape_target = escape
                    self.escape_until_step = (self.last_step or 0) + self.BOMB_TIMER
                    # Tier-1 #3: log every enemy_agent currently inside the
                    # blast as a likely kill. They'll respawn in place after
                    # ENEMY_FREEZE_DURATION ticks; remember the cell so a
                    # later bomb can be timed for it.
                    if self.tier1_repeat_kill and enemy_agent_hit:
                        unfreeze = step + self.ENEMY_FREEZE_DURATION
                        for pos, last_seen in self.enemy_agents.items():
                            if int(last_seen) == step and pos in bomb_blast:
                                self.recent_kills.append((pos, unfreeze))
                    return self.PLACE_BOMB

        # Adjacent high-value item grab — purely a speed optimization, not a
        # behavior change; the slow path would pick the same move.
        # On fixed map, also grab resources (2.0 reward + ammo) not just missions.
        grab_kinds = {"mission"}
        if getattr(self, "is_fixed_novice_map", False):
            grab_kinds.add("resource")
        for d, (dx, dy) in self.DIR_DELTAS.items():
            nxt = (location[0] + dx, location[1] + dy)
            if nxt not in self.seen or nxt in danger:
                continue
            if self._edge_blocked(location, d):
                continue
            item = self.last_seen_items.get(nxt)
            if item is None or item[0] not in grab_kinds:
                continue
            preferred = self._action_for_path(location, direction, [location, nxt])
            if preferred is not None and self._legal(observation, preferred):
                return preferred
        return None

    def _wall_break_reveals_high_value(
        self,
        location: tuple[int, int],
        wall_dir: int,
        max_distance: int = 5,
    ) -> bool:
        """Would breaking the destructible wall at (location, wall_dir) open a
        shortest path of length ≤ max_distance to an enemy base or a mission?

        Used to justify proactive bombing when the agent isn't yet stuck.
        """

        dx, dy = self.DIR_DELTAS[wall_dir]
        other = (location[0] + dx, location[1] + dy)
        if not self._in_bounds(other):
            return False
        edge_a = (location[0], location[1], wall_dir)
        edge_b = (other[0], other[1], self.OPPOSITE[wall_dir])
        had_a = edge_a in self.walls
        had_b = edge_b in self.walls
        self.walls.discard(edge_a)
        self.walls.discard(edge_b)
        # Pretend the cell behind the wall is part of the belief map so BFS can
        # reach it even if we never saw it directly.
        added_seen = False
        if other not in self.seen:
            self.seen.add(other)
            added_seen = True
        try:
            distance, _ = self._bfs_distance_map(location, set())
            for pos in self.enemy_bases:
                if distance.get(pos, max_distance + 1) <= max_distance:
                    return True
            for pos, (kind, _step) in self.last_seen_items.items():
                if kind == "mission" and distance.get(pos, max_distance + 1) <= max_distance:
                    return True
        finally:
            if had_a:
                self.walls.add(edge_a)
            if had_b:
                self.walls.add(edge_b)
            if added_seen:
                self.seen.discard(other)
        return False

    def _frontier_cells(self) -> list[tuple[int, int]]:
        frontiers = []
        for pos in self.seen:
            if any(n not in self.seen for n in self._raw_neighbors(pos)):
                frontiers.append(pos)
        return frontiers

    def _raw_neighbors(self, pos: tuple[int, int]) -> Iterable[tuple[int, int]]:
        for dx, dy in self.DIR_DELTAS.values():
            nxt = (pos[0] + dx, pos[1] + dy)
            if self._in_bounds(nxt):
                yield nxt

    def _action_for_path(
        self,
        location: tuple[int, int],
        direction: int,
        path: list[tuple[int, int]] | None,
    ) -> int | None:
        if not path or len(path) < 2:
            return None
        nxt = path[1]
        dx = nxt[0] - location[0]
        dy = nxt[1] - location[1]
        desired_dir = None
        for d, delta in self.DIR_DELTAS.items():
            if delta == (dx, dy):
                desired_dir = d
                break
        if desired_dir is None:
            return None
        if desired_dir == direction:
            return self.FORWARD
        if desired_dir == self.OPPOSITE[direction]:
            return self.BACKWARD
        if desired_dir == (direction + 3) % 4:
            return self.LEFT
        if desired_dir == (direction + 1) % 4:
            return self.RIGHT
        return None

    # ------------------------------------------------------------------
    # Tactical lookahead
    # ------------------------------------------------------------------
    def _tactical_lookahead_action(
        self,
        observation: dict,
        location: tuple[int, int],
        direction: int,
    ) -> int | None:
        """Beam-search a few tactical futures and return the best first move.

        This is deliberately "MCTS-light": the hidden evaluator's opponents are
        not reproducible locally, so we do not pretend to roll them out. Instead
        we search our own legal action sequences against the current belief map,
        score only robust events (safe bombs, known bases/enemies, own-base/self
        danger, and nearby item pickups), and let the normal planner handle
        everything else.
        """

        self.last_lookahead_score = -inf
        self.last_lookahead_path = ()
        if not self.mcts_enabled or self.health < self.LOW_HEALTH_THRESHOLD:
            return None

        # v2: pre-flight gate. Don't pay the cost of building a beam search
        # when there is nothing tactical within reach — the existing
        # frontier/objective planner is already correct for exploration ticks.
        # Expected fire rate ~20-25% of ticks → 4-5× drop in average per-call
        # cost compared to "MCTS on every tick" (which is what timed out v1).
        if not self._should_run_mcts(location):
            return None

        # v2: hard latency budget. If we cross the deadline we bail out and
        # return the best line found so far. Without this guard, depth 5
        # width 96 ran 1.2-2.4 s/tick on cloud and got killed by the
        # evaluator's wall-clock timeout (~600 ms/tick budget).
        t0 = time.monotonic()
        deadline = t0 + self.mcts_time_budget_s
        timeout_hit = False
        depths_reached = 0

        root_actions = self._lookahead_legal_actions(
            location, direction, self._as_int(observation.get("team_bombs"), default=0),
            root_observation=observation,
        )
        if not root_actions:
            return None

        bombs = tuple(
            (pos[0], pos[1], int(data.get("timer", self.BOMB_TIMER)))
            for pos, data in self.known_bombs.items()
        )
        states = [
            _LookaheadState(
                pos=location,
                direction=direction,
                bombs=bombs,
                bombs_left=self._as_int(observation.get("team_bombs"), default=0),
                health=self.health,
                collected=frozenset(),
                score=0.0,
                first_action=None,
                tactical=False,
                path=(),
            )
        ]

        best: _LookaheadState | None = None
        best_tactical: _LookaheadState | None = None
        for _depth in range(self.mcts_depth):
            # v2: outer-ply deadline check. Keeps per-ply work atomic but
            # caps total wall-clock cost. With DEPTH=3 WIDTH=24 we expect
            # to finish in 50-90ms, well under the 80ms budget; this guard
            # is the safety net for the occasional slow tick.
            if time.monotonic() > deadline:
                timeout_hit = True
                break
            depths_reached = _depth + 1
            expanded: list[_LookaheadState] = []
            for state in states:
                actions = root_actions if state.first_action is None else self._lookahead_legal_actions(
                    state.pos, state.direction, state.bombs_left
                )
                for action in actions:
                    nxt = self._lookahead_step(state, action)
                    if nxt is not None:
                        expanded.append(nxt)
            if not expanded:
                break
            expanded.sort(key=self._lookahead_rank, reverse=True)
            # Bomb lines often look bad before detonation because the agent is
            # still paying movement/danger costs. Preserve a tactical side-beam
            # so promising PLACE_BOMB futures are not pruned one ply before the
            # payoff lands.
            tactical_quota = max(8, self.mcts_width // 4)
            tactical = [state for state in expanded if state.tactical]
            tactical.sort(key=self._lookahead_tactical_rank, reverse=True)
            states = []
            seen_paths: set[tuple[int, ...]] = set()
            for candidate in expanded[: self.mcts_width]:
                states.append(candidate)
                seen_paths.add(candidate.path)
            for candidate in tactical[:tactical_quota]:
                if candidate.path in seen_paths:
                    continue
                states.append(candidate)
                seen_paths.add(candidate.path)
            candidate = states[0]
            if best is None or self._lookahead_rank(candidate) > self._lookahead_rank(best):
                best = candidate
            for candidate in states:
                if not candidate.tactical:
                    continue
                if best_tactical is None or self._lookahead_rank(candidate) > self._lookahead_rank(best_tactical):
                    best_tactical = candidate

        if self.mcts_log_timing:
            elapsed_ms = (time.monotonic() - t0) * 1000.0
            print(
                f"mcts {elapsed_ms:.1f}ms depth={depths_reached}/{self.mcts_depth}"
                f" tactical={'y' if best_tactical is not None else 'n'}"
                f"{' TIMEOUT' if timeout_hit else ''}",
                flush=True,
            )

        if best_tactical is None or best_tactical.first_action is None:
            return None

        final_score = self._lookahead_rank(best_tactical)
        self.last_lookahead_score = final_score
        self.last_lookahead_path = best_tactical.path
        if final_score >= self.mcts_min_score:
            return best_tactical.first_action
        return None

    def _should_run_mcts(self, location: tuple[int, int]) -> bool:
        """Cheap pre-flight gate for the tactical lookahead.

        Returns True only when something the search could meaningfully act on
        is within ``mcts_depth + 1`` Manhattan steps: a known bomb (we may
        need to plan around its detonation), an enemy base (we may want to
        bomb it), or a recently-seen enemy agent. Otherwise the existing
        frontier/objective planner handles the tick — and we save the entire
        beam-search cost.

        This is the change that made v2 fit cloud's per-tick budget.
        """
        if not self.mcts_enabled:
            return False
        reach = self.mcts_depth + 1

        for pos in self.known_bombs:
            if self._manhattan(location, pos) <= reach:
                return True

        for base in self.enemy_bases:
            if self._manhattan(location, base) <= reach:
                return True

        step = self.last_step if self.last_step is not None else 0
        for enemy, last_seen in self.enemy_agents.items():
            try:
                seen_step = int(last_seen)
            except (TypeError, ValueError):
                continue
            if step - seen_step > self.ENEMY_STALENESS:
                continue
            if self._manhattan(location, enemy) <= reach:
                return True

        return False

    def _lookahead_rank(self, state: _LookaheadState) -> float:
        return state.score + self._lookahead_terminal_value(state)

    def _lookahead_tactical_rank(self, state: _LookaheadState) -> float:
        """Rank tactical states for beam retention, not final action choice."""

        value = self._lookahead_rank(state)
        step = self.last_step if self.last_step is not None else 0
        for bx, by, timer in state.bombs:
            blast = self._blast_cells((bx, by))
            pending = 0.0
            if any(base in blast for base in self.enemy_bases):
                pending += 55.0
            for enemy, last_seen in self.enemy_agents.items():
                if step - int(last_seen) <= self.ENEMY_STALENESS and enemy in blast:
                    pending += 24.0
            if pending > 0:
                value += (pending / max(1, timer)) + 4.0 * (self.BOMB_TIMER - timer)
        return value

    def _lookahead_terminal_value(self, state: _LookaheadState) -> float:
        value = 0.0
        if state.bombs_left > 0:
            for base in self.enemy_bases:
                dist = self._lookahead_distance(state.pos, base, limit=8)
                if dist is not None:
                    value += max(0.0, 18.0 - 2.2 * dist)
            step = self.last_step if self.last_step is not None else 0
            for enemy, last_seen in self.enemy_agents.items():
                if step - int(last_seen) > self.ENEMY_STALENESS:
                    continue
                dist = self._lookahead_distance(state.pos, enemy, limit=5)
                if dist is not None:
                    value += max(0.0, 8.0 - 1.5 * dist)
        for pos, (kind, _seen_step) in self.last_seen_items.items():
            if pos in state.collected:
                continue
            dist = self._lookahead_distance(state.pos, pos, limit=4)
            if dist is not None:
                reward = {"mission": 5.0, "resource": 2.0, "recon": 1.0}.get(kind, 0.0)
                value += max(0.0, reward - 0.35 * dist)
        return value

    def _lookahead_legal_actions(
        self,
        pos: tuple[int, int],
        direction: int,
        bombs_left: int,
        root_observation: dict | None = None,
    ) -> list[int]:
        actions = [self.FORWARD, self.BACKWARD, self.LEFT, self.RIGHT, self.STAY]
        if bombs_left > 0:
            actions.append(self.PLACE_BOMB)

        legal: list[int] = []
        for action in actions:
            if root_observation is not None and not self._legal(root_observation, action):
                continue
            if action in {self.FORWARD, self.BACKWARD}:
                nxt, _ = self._simulate_action(pos, direction, action)
                move_dir = direction if action == self.FORWARD else self.OPPOSITE[direction]
                if nxt not in self.seen or not self._in_bounds(nxt):
                    continue
                if self._edge_blocked(pos, move_dir):
                    continue
            if action == self.PLACE_BOMB:
                blast = self._blast_cells(pos)
                if self.base_location is not None and self.base_location in blast:
                    continue
                if self._lookahead_escape(pos, blast, self.BOMB_TIMER) is None:
                    continue
            legal.append(action)
        return legal

    def _lookahead_step(self, state: _LookaheadState, action: int) -> _LookaheadState | None:
        pos, direction = self._simulate_action(state.pos, state.direction, action)
        bombs_left = state.bombs_left
        bombs = list(state.bombs)
        newly_placed: tuple[int, int] | None = None
        score = state.score - 0.12
        tactical = state.tactical
        collected = set(state.collected)
        health = state.health

        if action in {self.LEFT, self.RIGHT}:
            score -= 0.08
        elif action == self.STAY:
            score -= 0.45
        elif action == self.PLACE_BOMB:
            if bombs_left <= 0:
                return None
            if any((bx, by) == state.pos for bx, by, _timer in bombs):
                return None
            blast = self._blast_cells(state.pos)
            if self.base_location is not None and self.base_location in blast:
                return None
            if self._lookahead_escape(state.pos, blast, self.BOMB_TIMER) is None:
                return None
            bombs.append((state.pos[0], state.pos[1], self.BOMB_TIMER))
            newly_placed = state.pos
            bombs_left -= 1
            tactical = True
            score -= 0.25

        item = self.last_seen_items.get(pos)
        if item is not None and pos not in collected:
            score += {"mission": 5.0, "resource": 2.0, "recon": 1.0}.get(item[0], 0.0)
            collected.add(pos)

        active_danger = set()
        for bx, by, timer in bombs:
            if timer <= 2:
                active_danger.update(self._blast_cells((bx, by)))
        if pos in active_danger:
            score -= 9.0

        aged_bombs: list[tuple[int, int, int]] = []
        for bx, by, timer in bombs:
            if newly_placed == (bx, by):
                aged_bombs.append((bx, by, timer))
                continue
            timer -= 1
            blast_pos = (bx, by)
            if timer <= 0:
                delta, hit_tactical, health = self._lookahead_detonation_score(
                    blast_pos, pos, health
                )
                score += delta
                tactical = tactical or hit_tactical
            else:
                aged_bombs.append((bx, by, timer))

        return _LookaheadState(
            pos=pos,
            direction=direction,
            bombs=tuple(sorted(aged_bombs)),
            bombs_left=bombs_left,
            health=health,
            collected=frozenset(collected),
            score=score,
            first_action=action if state.first_action is None else state.first_action,
            tactical=tactical,
            path=state.path + (action,),
        )

    def _lookahead_detonation_score(
        self,
        bomb_pos: tuple[int, int],
        our_pos: tuple[int, int],
        health: int,
    ) -> tuple[float, bool, int]:
        blast = self._blast_cells(bomb_pos)
        score = 0.0
        tactical = False

        # Tier-1 #4: shared-credit-aware values. Under cloud's 6-team game
        # the average kill / base-destroy is shared across ~2-3 contributors,
        # so the realistic credit is well below the raw config values. Using
        # the inflated values pulled the planner toward marginal long-range
        # base attacks at the expense of close-range damage and items.
        base_value = (
            self.SHARED_CREDIT_BASE_VALUE if self.tier1_shared_credit else 55.0
        )
        kill_value = (
            self.SHARED_CREDIT_KILL_VALUE if self.tier1_shared_credit else 24.0
        )

        for base in self.enemy_bases:
            if base in blast:
                score += base_value
                tactical = True
        step = self.last_step if self.last_step is not None else 0
        for enemy, last_seen in self.enemy_agents.items():
            if step - int(last_seen) > self.ENEMY_STALENESS:
                continue
            if enemy in blast:
                score += kill_value
                tactical = True
            elif enemy in self._extended_blast(blast, 1):
                score += 6.0
                tactical = True

        # Tier-1 #7: predictive walk credit. Even when no enemy is currently
        # in the blast cone, a recently-seen enemy has nonzero probability of
        # walking into it before detonation. Under a uniform random walker
        # the per-step movement distribution is ~1/5 to each neighbour or
        # STAY; the per-walk-step probability of hitting any specific cell
        # is small but the union over 4 walk steps and 5-13 blast cells is
        # not negligible. We approximate it cheaply: each fresh enemy within
        # PREDICTIVE_WALK_HORIZON Manhattan steps of the blast contributes
        # a small expected-damage term.
        if self.tier1_predictive_walk:
            extended = self._extended_blast(blast, self.PREDICTIVE_WALK_HORIZON)
            for enemy, last_seen in self.enemy_agents.items():
                if enemy in blast:
                    continue
                if step - int(last_seen) > self.ENEMY_STALENESS:
                    continue
                if enemy not in extended:
                    continue
                d = self._manhattan(enemy, our_pos)
                # Probability decays with distance — an enemy 4 steps away
                # has lower P(hit blast) than one 1 step away. Cap at 0.30.
                # Tier-2 #8: scale by measured opponent walk distance so this
                # tracks the actual mobility of cloud opponents instead of
                # the random-walk default.
                p_hit = max(0.0, 0.30 - 0.06 * d) * self.opponent_walk_scale
                score += kill_value * p_hit
                tactical = True

        # Tier-1 #3: respawn-camp credit. If a kill cell from `recent_kills`
        # is in our blast AND the enemy is still frozen / about to unfreeze
        # within the bomb's effective window, count expected hit value.
        if self.tier1_repeat_kill and self.recent_kills:
            for kpos, unfreeze in self.recent_kills:
                if kpos not in blast:
                    continue
                # Detonation arrives roughly BOMB_TIMER ticks after placement.
                # If the unfreeze step lands inside that window, the enemy is
                # at the kill cell exactly when we explode there.
                detonation_step = step + self.BOMB_TIMER
                if abs(detonation_step - unfreeze) <= 1:
                    score += kill_value
                    tactical = True

        if self.base_location is not None and self.base_location in blast:
            score -= 65.0
        if our_pos in blast:
            health -= 20
            score -= 30.0 if health > 0 else 70.0
        return score, tactical, health

    def _lookahead_escape(
        self,
        location: tuple[int, int],
        blast: set[tuple[int, int]],
        max_moves: int,
    ) -> tuple[int, int] | None:
        queue = deque([(location, 0)])
        seen = {location}
        while queue:
            pos, dist = queue.popleft()
            if dist > 0 and pos not in blast and pos in self.seen:
                return pos
            if dist >= max_moves:
                continue
            for nxt in self._neighbors(pos):
                if nxt in seen or nxt not in self.seen:
                    continue
                seen.add(nxt)
                queue.append((nxt, dist + 1))
        return None

    def _lookahead_distance(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
        limit: int,
    ) -> int | None:
        if start == goal:
            return 0
        queue = deque([(start, 0)])
        seen = {start}
        while queue:
            pos, dist = queue.popleft()
            if dist >= limit:
                continue
            for nxt in self._neighbors(pos):
                if nxt in seen or nxt not in self.seen:
                    continue
                if nxt == goal:
                    return dist + 1
                seen.add(nxt)
                queue.append((nxt, dist + 1))
        return None

    # ------------------------------------------------------------------
    # Bombs and danger
    # ------------------------------------------------------------------
    def _age_bombs(self, step: int) -> None:
        if self.last_step is None:
            return
        delta = max(0, step - self.last_step)
        if delta <= 0:
            return
        for pos, data in list(self.known_bombs.items()):
            timer = int(data.get("timer", self.BOMB_TIMER)) - delta
            if timer <= 0:
                self.known_bombs.pop(pos, None)
                pass
            else:
                data["timer"] = timer
                data["last_step"] = step
        # Drop very stale enemy_agent records so old sightings stop being
        # used as base-defense candidates 30+ ticks after the enemy moved on.
        for pos, last_seen in list(self.enemy_agents.items()):
            if step - int(last_seen) > self.ENEMY_AGENT_MEMORY_STEPS:
                self.enemy_agents.pop(pos, None)
        # Tier-1 #3: drop kill records whose respawn window has fully passed.
        # We keep them through ENEMY_FREEZE_DURATION + a little slack for
        # follow-up bomb timing.
        if self.recent_kills:
            self.recent_kills = [
                (pos, unfreeze)
                for pos, unfreeze in self.recent_kills
                if step <= unfreeze + self.BOMB_TIMER + 1
            ]

    def _danger_cells(self) -> set[tuple[int, int]]:
        danger: set[tuple[int, int]] = set()
        for bomb_pos, data in self.known_bombs.items():
            timer = int(data.get("timer", self.BOMB_TIMER))
            if timer > 2:
                continue
            danger.update(self._blast_cells(bomb_pos))
        return danger

    def _enemy_threat_cells(self) -> set[tuple[int, int]]:
        """Recently-seen enemy positions plus their 4-neighbors.

        Enemies move and attack adjacent cells, so a path that passes through
        their immediate vicinity costs us health.  This is a soft signal — the
        BFS still allows these cells, the scorer just penalizes them.
        """

        if not self.enemy_agents:
            return set()
        step = self.last_step if self.last_step is not None else 0
        threats: set[tuple[int, int]] = set()
        for pos, last_seen in self.enemy_agents.items():
            if step - int(last_seen) > self.ENEMY_STALENESS:
                continue
            threats.add(pos)
            threats.update(self._raw_neighbors(pos))
        return threats

    def _blast_cells(self, bomb_pos: tuple[int, int]) -> set[tuple[int, int]]:
        cached = self._blast_cache.get(bomb_pos)
        if cached is not None:
            return set(cached)
        bx, by = bomb_pos
        cells = set()
        for x in range(bx - self.BOMB_RADIUS, bx + self.BOMB_RADIUS + 1):
            for y in range(by - self.BOMB_RADIUS, by + self.BOMB_RADIUS + 1):
                pos = (x, y)
                if not self._in_bounds(pos):
                    continue
                if max(abs(x - bx), abs(y - by)) > self.BOMB_RADIUS:
                    continue
                if self._line_of_sight_clear(bomb_pos, pos):
                    cells.add(pos)
        self._blast_cache[bomb_pos] = frozenset(cells)
        return cells

    def _active_escape_path(self, location: tuple[int, int], step: int) -> list[tuple[int, int]] | None:
        if self.escape_target is None or self.escape_until_step is None:
            return None
        if step > self.escape_until_step:
            self.escape_target = None
            self.escape_until_step = None
            return None

        forced_danger = set()
        for pos, data in self.known_bombs.items():
            # Never step on any bomb cell (own or enemy) because it is solid
            forced_danger.add(pos)
            if not data.get("own"):
                timer = int(data.get("timer", self.BOMB_TIMER))
                if timer <= 3:
                    forced_danger.update(self._blast_cells(pos))
        if location == self.escape_target and location not in forced_danger:
            self.escape_target = None
            self.escape_until_step = None
            return None

        path = self._bfs(location, self.escape_target, forced_danger)
        if path is not None:
            return path

        replacement = self._nearest_escape_cell(location, forced_danger)
        if replacement is None:
            return None
        self.escape_target = replacement
        return self._bfs(location, replacement, forced_danger)

    def _nearest_escape_cell(
        self,
        location: tuple[int, int],
        blast: set[tuple[int, int]],
    ) -> tuple[int, int] | None:
        queue = deque([location])
        seen = {location}
        while queue:
            pos = queue.popleft()
            if pos not in blast and pos in self.seen:
                return pos
            for nxt in self._neighbors(pos):
                if nxt in seen or nxt not in self.seen:
                    continue
                seen.add(nxt)
                queue.append(nxt)
        return None

    def _should_place_bomb(
        self,
        observation: dict,
        location: tuple[int, int],
        target: tuple[int, int] | None,
        danger: set[tuple[int, int]],
    ) -> bool:
        if not self._legal(observation, self.PLACE_BOMB):
            return False
        if self._as_int(observation.get("team_bombs"), default=0) <= 0:
            return False
        if location in danger:
            return False
        # Health-aware retreat: don't initiate fights when one hit kills us.
        if self.health < self.LOW_HEALTH_THRESHOLD:
            return False
        bomb_blast = self._blast_cells(location)
        step = self.last_step if self.last_step is not None else 0
        # Direct hits: enemy base, or a fresh enemy-agent sighting already in blast.
        tactical_target = any(pos in bomb_blast for pos in self.enemy_bases)
        if not tactical_target:
            for pos, last_seen in self.enemy_agents.items():
                if step - int(last_seen) > 1 and pos not in bomb_blast:
                    continue
                if pos in bomb_blast:
                    tactical_target = True
                    break

        base_location = self.base_location or self._location(observation.get("base_location"))
        if base_location is not None and base_location in bomb_blast:
            return False

        # Predictive bombing: bomb when *multiple* enemies are immediately
        # adjacent to the blast cone. Random opponents wander; betting one
        # specific enemy walks into the blast is a coin-flip and a wasted bomb.
        # Betting that one of N≥2 nearby enemies does is much better odds.
        if not tactical_target and len(self.enemy_agents) >= 2:
            extended = self._extended_blast(bomb_blast, self.PREDICTIVE_BOMB_RANGE)
            nearby = 0
            for pos, last_seen in self.enemy_agents.items():
                if step - int(last_seen) > self.ENEMY_STALENESS:
                    continue
                if pos in extended:
                    nearby += 1
            if nearby >= 2:
                tactical_target = True

        # Tier-1 #7: predictive random-walk bomb. If there is at least one
        # fresh enemy sighting within PREDICTIVE_WALK_HORIZON of the blast,
        # the union probability of any one enemy walking into the blast over
        # the bomb timer window is non-trivial. We use a conservative
        # threshold (expected value >= 8 reward, ~one mission's worth) so
        # this only fires on high-EV placements.
        if not tactical_target and self.tier1_predictive_walk and self.enemy_agents:
            extended = self._extended_blast(bomb_blast, self.PREDICTIVE_WALK_HORIZON)
            expected_damage = 0.0
            for pos, last_seen in self.enemy_agents.items():
                if step - int(last_seen) > self.ENEMY_STALENESS:
                    continue
                if pos not in extended:
                    continue
                d = self._manhattan(pos, location)
                # Tier-2 #8: scale by opponent_walk_scale.
                p_hit = max(0.0, 0.30 - 0.06 * d) * self.opponent_walk_scale
                expected_damage += 20.0 * p_hit  # 20 damage per blast hit
            if expected_damage >= 8.0:
                tactical_target = True

        # Tier-1 #3: respawn-camp predictive bomb. If a kill cell falls in
        # our blast and the enemy unfreeze step lines up with this bomb's
        # detonation step, fire even without other targets.
        if not tactical_target and self.tier1_repeat_kill and self.recent_kills:
            detonation_step = step + self.BOMB_TIMER
            for kpos, unfreeze in self.recent_kills:
                if kpos in bomb_blast and abs(detonation_step - unfreeze) <= 1:
                    tactical_target = True
                    break

        wall_to_open = False
        # Bomb conservation: don't waste bombs on walls when we only have 1 left
        # (save it for combat). On fixed map, also check phase.
        bombs_available = self._as_int(observation.get("team_bombs"), default=0)
        allow_wall_bomb = bombs_available >= 2
        if getattr(self, "is_fixed_novice_map", False):
            # In attack phase, be more conservative with wall bombs
            phase = self._game_phase()
            if phase == "attack" and bombs_available < 3:
                allow_wall_bomb = False

        # Proactive wall break: if the target is high-value (enemy base or
        # mission) and a destructible wall sits between us and it, bomb
        # without waiting to be visibly stuck.
        if allow_wall_bomb and target is not None:
            target_dir = self._rough_direction(location, target)
            if target_dir is not None and (location[0], location[1], target_dir) in self.destructible:
                target_kind = self.last_seen_items.get(target, (None, None))[0]
                if target in self.enemy_bases or target_kind == "mission":
                    wall_to_open = True
                elif target not in self.enemy_bases:
                    wall_to_open = self._stuck_recently()

        # Novice fixed map custom wall opening:
        if not wall_to_open and allow_wall_bomb and getattr(self, "is_fixed_novice_map", False) and getattr(self, "current_path", None) is not None:
            path = self.current_path
            if len(path) >= 2:
                nxt = path[1]
                d = self._direction_for_delta(nxt[0] - location[0], nxt[1] - location[1])
                if d is not None and (location[0], location[1], d) in self.destructible:
                    wall_to_open = True

        # Bomb-chain heuristic disabled in v3b: in random-opponent local it
        # was wasting bombs on speculative wall breaks. Helper kept for future
        # use against smarter opponents.

        if not tactical_target and not wall_to_open:
            return False
        escape_target = self._safe_escape_within(location, bomb_blast, self.BOMB_TIMER, danger)
        if escape_target is None:
            return False

        self.known_bombs[location] = {"timer": self.BOMB_TIMER, "own": True, "last_step": self.last_step or 0}
        self.escape_target = escape_target
        self.escape_until_step = (self.last_step or 0) + self.BOMB_TIMER
        return True

    def _safe_escape_within(
        self,
        location: tuple[int, int],
        blast: set[tuple[int, int]],
        max_moves: int,
        danger: set[tuple[int, int]] | None = None,
    ) -> tuple[int, int] | None:
        """Return the closest cell outside ``blast`` reachable in ≤ max_moves.

        The agent gets ``BOMB_TIMER`` movement actions between placing a bomb
        and the detonation phase, so anything beyond that is not actually safe.
        """

        queue = deque([(location, 0)])
        seen = {location}
        while queue:
            pos, dist = queue.popleft()
            if dist > 0 and pos not in blast:
                return pos
            if dist >= max_moves:
                continue
            for nxt in self._neighbors(pos):
                if nxt in seen or nxt not in self.seen:
                    continue
                if danger is not None and nxt in danger:
                    continue
                seen.add(nxt)
                queue.append((nxt, dist + 1))
        return None

    def _stuck_recently(self) -> bool:
        if len(self.recent_locations) < 6:
            return False
        return len(set(self.recent_locations[-6:])) <= 2

    # ------------------------------------------------------------------
    # Fallback action scoring
    # ------------------------------------------------------------------
    def _fallback_action(
        self,
        observation: dict,
        location: tuple[int, int] | None,
        direction: int,
        target: tuple[int, int] | None,
    ) -> int:
        legal_actions = [a for a in [self.FORWARD, self.BACKWARD, self.LEFT, self.RIGHT, self.STAY] if self._legal(observation, a)]
        if not legal_actions:
            return self.STAY
        if location is None:
            return legal_actions[0]

        danger = self._danger_cells()
        threats = self._enemy_threat_cells()
        low_health = self.health < self.LOW_HEALTH_THRESHOLD
        # Softer cell-threat penalty than danger; under low_health, treat
        # threat cells as nearly as bad as bomb-blast cells.
        cell_threat_penalty = 30.0 if low_health else self.CELL_THREAT_PENALTY
        best_action = legal_actions[0]
        best_score = -inf
        for action in legal_actions:
            new_pos, new_dir = self._simulate_action(location, direction, action)
            score = 0.0
            if new_pos in danger:
                score -= 100.0
            if new_pos in threats and new_pos not in self.enemy_bases:
                score -= cell_threat_penalty
            item = self.last_seen_items.get(new_pos)
            if item:
                score += self.ITEM_VALUES.get(item[0], 0.0)
            score += 1.8 * sum(1 for n in self._raw_neighbors(new_pos) if n not in self.seen)
            score -= 0.35 * self.visit_count.get(new_pos, 0)
            if new_pos in self.recent_locations[-3:]:
                score -= 1.5
            if action == self.STAY:
                # Tier-1 #6: cloud reward for STAY is 0 (stationary_penalty=0
                # in the env config). Standing still in safety while a hostile
                # bomb resolves is sometimes optimal. Drop the penalty under
                # the tier-1 toggle; only keep a tiny tie-breaker so MOVE wins
                # all-else-equal.
                if self.tier1_no_stay_penalty:
                    score -= 0.2 if low_health else 0.5
                else:
                    score -= 1.0 if low_health else 4.0
            if target is not None:
                score -= 0.12 * self._manhattan(new_pos, target)
                desired_dir = self._rough_direction(new_pos, target)
                if desired_dir == new_dir:
                    score += 0.6
            if score > best_score:
                best_score = score
                best_action = action
        return best_action

    def _simulate_action(
        self,
        location: tuple[int, int],
        direction: int,
        action: int,
    ) -> tuple[tuple[int, int], int]:
        if action == self.FORWARD:
            dx, dy = self.DIR_DELTAS[direction]
            return (location[0] + dx, location[1] + dy), direction
        if action == self.BACKWARD:
            dx, dy = self.DIR_DELTAS[self.OPPOSITE[direction]]
            return (location[0] + dx, location[1] + dy), direction
        if action == self.LEFT:
            return location, (direction + 3) % 4
        if action == self.RIGHT:
            return location, (direction + 1) % 4
        return location, direction

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _legal(self, observation: dict, action: int) -> bool:
        mask = observation.get("action_mask")
        if mask is None:
            return 0 <= action <= self.PLACE_BOMB
        try:
            return bool(mask[action])
        except Exception:
            return 0 <= action <= self.PLACE_BOMB

    def _first_legal(self, observation: dict, actions: list[int]) -> int:
        for action in actions:
            if self._legal(observation, action):
                return action
        return self.STAY

    def _edge_blocked(self, pos: tuple[int, int], direction: int) -> bool:
        if (pos[0], pos[1], direction) in self.walls:
            return True
        dx, dy = self.DIR_DELTAS[direction]
        other = (pos[0] + dx, pos[1] + dy)
        opposite = self.OPPOSITE[direction]
        return (other[0], other[1], opposite) in self.walls

    def _line_of_sight_clear(self, start: tuple[int, int], end: tuple[int, int]) -> bool:
        if start == end:
            return True
        path = self._line_tiles(start, end)
        for current, nxt in zip(path, path[1:]):
            dx = nxt[0] - current[0]
            dy = nxt[1] - current[1]
            if dx and dy:
                horizontal = self.DIR_RIGHT if dx > 0 else self.DIR_LEFT
                vertical = self.DIR_DOWN if dy > 0 else self.DIR_UP
                # Match the environment's diagonal LOS spirit: a diagonal step is
                # blocked only when both axis alternatives are blocked.
                if self._edge_blocked(current, horizontal) and self._edge_blocked(current, vertical):
                    return False
            else:
                direction = self._direction_for_delta(dx, dy)
                if direction is not None and self._edge_blocked(current, direction):
                    return False
        return True

    @staticmethod
    def _line_tiles(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
        x0, y0 = start
        x1, y1 = end
        dx = x1 - x0
        dy = y1 - y0
        nx = abs(dx)
        ny = abs(dy)
        sx = 1 if dx > 0 else -1 if dx < 0 else 0
        sy = 1 if dy > 0 else -1 if dy < 0 else 0
        x, y = x0, y0
        tiles = [(x, y)]
        ix = iy = 0
        while ix < nx or iy < ny:
            if nx and ny and (1 + 2 * ix) * ny == (1 + 2 * iy) * nx:
                x += sx
                y += sy
                ix += 1
                iy += 1
            elif ny == 0 or ((1 + 2 * ix) * ny < (1 + 2 * iy) * nx):
                x += sx
                ix += 1
            else:
                y += sy
                iy += 1
            tiles.append((x, y))
        return tiles

    def _direction_for_delta(self, dx: int, dy: int) -> int | None:
        for direction, delta in self.DIR_DELTAS.items():
            if delta == (dx, dy):
                return direction
        return None

    def _rough_direction(self, start: tuple[int, int], target: tuple[int, int]) -> int | None:
        dx = target[0] - start[0]
        dy = target[1] - start[1]
        if abs(dx) >= abs(dy) and dx != 0:
            return self.DIR_RIGHT if dx > 0 else self.DIR_LEFT
        if dy != 0:
            return self.DIR_DOWN if dy > 0 else self.DIR_UP
        return None

    def _in_bounds(self, pos: tuple[int, int]) -> bool:
        return 0 <= pos[0] < self.grid_size and 0 <= pos[1] < self.grid_size

    @staticmethod
    def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    @staticmethod
    def _channel_value(cell: list, channel: int):
        try:
            return cell[channel]
        except Exception:
            return 0

    @classmethod
    def _channel_on(cls, cell: list, channel: int) -> bool:
        try:
            return float(cell[channel]) > 0.0
        except Exception:
            return False

    @classmethod
    def _as_int(cls, value, default: int = 0) -> int:
        try:
            if isinstance(value, list):
                if not value:
                    return default
                value = value[0]
            if hasattr(value, "item"):
                value = value.item()
            return int(value)
        except Exception:
            return default

    @classmethod
    def _location(cls, value) -> tuple[int, int] | None:
        try:
            if value is None:
                return None
            if hasattr(value, "tolist"):
                value = value.tolist()
            return (int(value[0]), int(value[1]))
        except Exception:
            return None
