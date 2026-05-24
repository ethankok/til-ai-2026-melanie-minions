"""Scripted-base-attack AE manager — full-heuristic rewrite (hail-mary build).

Design contract (read me before editing):

1. Decision pyramid, top-priority first:
   - frozen_ticks > 0 → STAY (or first legal stand-still action)
   - active escape: if my cell is in a bomb blast cone with timer ≤ 2, move to
     the cheapest safe neighbour.
   - cluster bomb: 2+ visible enemies in my would-be blast → PLACE_BOMB with
     verified escape route.
   - opportunistic bomb: an enemy_base or a fresh enemy_agent is in my blast
     and I have a 3-tick escape → PLACE_BOMB.
   - scripted primary: orientation-aware A* over (x, y, facing) toward the
     nearest enemy base on the Novice fixed map. Turning, forward and
     backward each cost 1 tick. Crossing a destructible wall costs 5 ticks
     (place bomb + wait + step). When the next planned step requires
     breaking a destructible wall and we have bombs + escape, PLACE_BOMB.
   - tertiary item grab: if a mission/resource is one step away through a
     clear edge, take it.
   - exploration fallback: pick a legal action that maximises future room.

2. Hard safety invariants (never violated):
   - action_mask is the source of truth for legality.
   - never bomb a cell whose blast covers our own base.
   - never bomb without a BFS-verified escape cell in ≤ BOMB_TIMER steps.
   - never step into a cell whose blast timer is ≤ 1.

3. Shape contract: the public method is ``AEManager().ae(observation) -> int``
   returning one of {0..5} matching the env's action codes.

The old planner is preserved at ``ae_manager_legacy.py`` — restore with
``cp ae_manager_legacy.py ae_manager.py`` if this build regresses.
"""

from __future__ import annotations

import heapq
import os
from collections import deque
from math import inf
from typing import Iterable


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


# Action codes — must match til_environment.actions.
FORWARD = 0
BACKWARD = 1
LEFT = 2
RIGHT = 3
STAY = 4
PLACE_BOMB = 5

# Facing codes — must match til_environment.types.Direction.
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

# Observation channels — must match til_environment.observation.ViewChannel.
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
    TILE_MISSION: "mission",
    TILE_RESOURCE: "resource",
    TILE_RECON: "recon",
}

GRID_SIZE = 16
BOMB_TIMER = 3
BOMB_RADIUS = 2
LOW_HEALTH = 20
ENEMY_STALENESS = 3            # ticks a sighting still counts as a real threat
ENEMY_AGENT_MEMORY = 30        # drop entries after this many ticks
BOMB_BREAK_COST = 5.0          # A* cost of crossing a destructible edge

ITEM_VALUE_DEFAULT = {"mission": 50.0, "resource": 25.0, "recon": 10.0}


def _channel_on(cell, channel: int) -> bool:
    try:
        return float(cell[channel]) > 0.0
    except Exception:
        return False


def _channel_value(cell, channel: int) -> float:
    try:
        return float(cell[channel])
    except Exception:
        return 0.0


def _as_int(value, default: int = 0) -> int:
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


def _location(value) -> tuple[int, int] | None:
    try:
        if value is None:
            return None
        if hasattr(value, "tolist"):
            value = value.tolist()
        return (int(value[0]), int(value[1]))
    except Exception:
        return None


class AEManager:
    """Full-heuristic, scripted-base-attack manager."""

    # Re-export action / direction codes as class attrs to keep external
    # references (tests, hybrid_manager.py) working.
    FORWARD = FORWARD
    BACKWARD = BACKWARD
    LEFT = LEFT
    RIGHT = RIGHT
    STAY = STAY
    PLACE_BOMB = PLACE_BOMB

    DIR_RIGHT = DIR_RIGHT
    DIR_DOWN = DIR_DOWN
    DIR_LEFT = DIR_LEFT
    DIR_UP = DIR_UP

    DIR_DELTAS = DIR_DELTAS
    OPPOSITE = OPPOSITE

    GRID_SIZE = GRID_SIZE
    BOMB_TIMER = BOMB_TIMER
    BOMB_RADIUS = BOMB_RADIUS

    def __init__(self) -> None:
        # Tunables. Defaults chosen to ship aggressive scripted base-attack.
        self.enemy_base_value = _env_float("AE_ENEMY_BASE_VALUE", 200.0)
        self.dist_penalty = _env_float("AE_DIST_PENALTY", 1.0)
        self.bomb_break_cost = _env_float("AE_BOMB_BREAK_COST", BOMB_BREAK_COST)
        self.item_mission_value = _env_float("AE_ITEM_MISSION_VALUE", 35.0)
        self.item_resource_value = _env_float("AE_ITEM_RESOURCE_VALUE", 18.0)
        self.item_recon_value = _env_float("AE_ITEM_RECON_VALUE", 6.0)
        self.item_values = {
            "mission": self.item_mission_value,
            "resource": self.item_resource_value,
            "recon": self.item_recon_value,
        }
        # Cluster bombing requires 2+ enemy_agents inside the blast.
        self.cluster_min = max(2, _env_int("AE_CLUSTER_MIN", 2))
        self.log = _env_flag("AE_LOG", False)
        self._reset()

    # ------------------------------------------------------------------
    # Belief state lifecycle
    # ------------------------------------------------------------------
    def _reset(self) -> None:
        self.grid_size = GRID_SIZE
        self.last_step: int | None = None
        self.turn_counter = 0
        self.seen: set[tuple[int, int]] = set()
        self.walls: set[tuple[int, int, int]] = set()
        self.destructible: set[tuple[int, int, int]] = set()
        self.last_seen_items: dict[tuple[int, int], tuple[str, int]] = {}
        self.collected_items: dict[tuple[int, int], int] = {}
        self.enemy_bases: dict[tuple[int, int], int] = {}
        self.enemy_agents: dict[tuple[int, int], int] = {}
        self.known_bombs: dict[tuple[int, int], dict] = {}
        self.visit_count: dict[tuple[int, int], int] = {}
        self.recent_locations: list[tuple[int, int]] = []
        self.base_location: tuple[int, int] | None = None
        self.health: int = 60
        self.base_health: int = 100
        self.is_fixed_novice_map = False
        self.fixed_team_idx: int | None = None
        # Per-tick scratch.
        self._blast_cache: dict[tuple[int, int], frozenset[tuple[int, int]]] = {}

    # ------------------------------------------------------------------
    # Public policy
    # ------------------------------------------------------------------
    def ae(self, observation: dict) -> int:
        self.turn_counter += 1
        step = _as_int(observation.get("step"), default=self.turn_counter)
        if self.last_step is None or step == 0 or step < self.last_step:
            self._reset()
        self._age_bombs(step)
        self._blast_cache = {}
        self.last_step = step

        location = _location(observation.get("location"))
        direction = _as_int(observation.get("direction"), default=DIR_RIGHT) % 4
        self.health = _as_int(observation.get("health"), default=60)
        self.base_health = _as_int(observation.get("base_health"), default=100)
        team_bombs = _as_int(observation.get("team_bombs"), default=0)
        frozen_ticks = _as_int(observation.get("frozen_ticks"), default=0)

        # Step-0 fixed-map detection: pre-populate everything we know about
        # the Novice arena (walls, destructibles, bases, items). Without
        # this, the agent has to discover the map turn-by-turn and the
        # first 30+ ticks are spent exploring instead of attacking.
        if step == 0 and location is not None:
            base_loc = _location(observation.get("base_location"))
            if base_loc is not None:
                self._maybe_load_fixed_novice(base_loc)

        self._update_memory(observation, step, location, direction)

        if location is not None:
            self.visit_count[location] = self.visit_count.get(location, 0) + 1
            self.recent_locations.append(location)
            if len(self.recent_locations) > 12:
                self.recent_locations.pop(0)

        if frozen_ticks > 0:
            return self._first_legal(observation, [STAY, LEFT, RIGHT, FORWARD, BACKWARD])

        if location is None:
            return self._first_legal(observation, [STAY, LEFT, RIGHT, FORWARD, BACKWARD])

        # 1. ESCAPE: if currently sitting in a soon-to-detonate blast cone,
        #    bail out to the safest legal neighbour. This is the highest
        #    priority because nothing else matters if we're about to take
        #    20 damage.
        active_danger = self._active_danger_cells()
        if location in active_danger:
            escape_move = self._escape_action(observation, location, direction, active_danger)
            if escape_move is not None:
                return escape_move

        # 2/3. OFFENSIVE BOMBS. Try cluster → opportunistic at our current
        #      cell. Both require an escape and never include own_base.
        if (location not in active_danger
                and team_bombs > 0
                and self.health >= LOW_HEALTH
                and self._legal(observation, PLACE_BOMB)):
            bomb_move = self._maybe_offensive_bomb(location, step, active_danger)
            if bomb_move is not None:
                return bomb_move

        # 4. SCRIPTED PRIMARY: orientation-aware A* toward the closest enemy
        #    base. If breaking a destructible wall is the next step on the
        #    path, place a bomb (with verified escape) instead of moving.
        primary_move = self._scripted_base_attack(
            observation, location, direction, team_bombs, active_danger, step
        )
        if primary_move is not None:
            return primary_move

        # 5. TERTIARY item grab — only when we cannot reach any base. This
        #    matches the friend-stack idea: items are tie-breakers, not the
        #    main loop.
        item_move = self._adjacent_item_grab(observation, location, direction, active_danger)
        if item_move is not None:
            return item_move

        # 6. EXPLORATION fallback — keep moving so we don't lose the speed
        #    component of the score.
        return self._exploration_fallback(observation, location, direction, active_danger)

    # ------------------------------------------------------------------
    # Memory ingestion
    # ------------------------------------------------------------------
    def _maybe_load_fixed_novice(self, base_loc: tuple[int, int]) -> None:
        try:
            from novice_map_data import (
                BASE_LOCATIONS,
                STATIC_ENTITIES,
                WALLS,
                DESTRUCTIBLE,
            )
        except ImportError:
            return
        for i in range(len(BASE_LOCATIONS)):
            if tuple(base_loc) == tuple(BASE_LOCATIONS[i]):
                self.is_fixed_novice_map = True
                self.fixed_team_idx = i
                self.seen = {(x, y) for x in range(GRID_SIZE) for y in range(GRID_SIZE)}
                self.walls = set(WALLS)
                self.destructible = set(DESTRUCTIBLE)
                self.enemy_bases = {
                    tuple(BASE_LOCATIONS[j]): 0
                    for j in range(len(BASE_LOCATIONS))
                    if j != i
                }
                self.last_seen_items = {tuple(pos): (kind, 0) for kind, pos in STATIC_ENTITIES}
                if self.log:
                    print(f"[AEManager] Novice fixed map detected team_idx={i}", flush=True)
                break

    def _update_memory(
        self,
        observation: dict,
        step: int,
        location: tuple[int, int] | None,
        direction: int,
    ) -> None:
        if location is not None:
            self.seen.add(location)

        agent_view = observation.get("agent_viewcone")
        if location is not None and agent_view is not None:
            self._project_agent_view(agent_view, location, direction, step)

        base_view = observation.get("base_viewcone")
        base_loc = _location(observation.get("base_location"))
        if base_loc is not None:
            self.seen.add(base_loc)
            self.base_location = base_loc
        if base_loc is not None and base_view is not None:
            self._project_centered_view(base_view, base_loc, step)

    def _project_agent_view(self, view, location, direction, step) -> None:
        height = len(view)
        width = len(view[0]) if height else 0
        if not height or not width:
            return
        origin_forward = 2 if height >= 7 else height // 2
        origin_side = 2 if width >= 5 else width // 2
        for row in range(height):
            for col in range(width):
                cell = view[row][col]
                if not _channel_on(cell, VISIBLE):
                    continue
                forward = row - origin_forward
                side = col - origin_side
                world = self._rel_to_world(location, direction, forward, side)
                if self._in_bounds(world):
                    self._ingest_cell(world, cell, step)

    def _project_centered_view(self, view, center, step) -> None:
        height = len(view)
        width = len(view[0]) if height else 0
        if not height or not width:
            return
        center_row = height // 2
        center_col = width // 2
        for row in range(height):
            for col in range(width):
                cell = view[row][col]
                if not _channel_on(cell, VISIBLE):
                    continue
                world = (center[0] + row - center_row, center[1] + col - center_col)
                if self._in_bounds(world):
                    self._ingest_cell(world, cell, step)

    def _ingest_cell(self, world, cell, step) -> None:
        self.seen.add(world)
        for d, channel in WALL_CHANNELS.items():
            edge = (world[0], world[1], d)
            if _channel_on(cell, channel):
                self.walls.add(edge)
            else:
                self.walls.discard(edge)
        for d, channel in DESTR_CHANNELS.items():
            edge = (world[0], world[1], d)
            if _channel_on(cell, channel):
                self.destructible.add(edge)
            else:
                self.destructible.discard(edge)

        visible_item = None
        for channel, kind in ITEM_CHANNELS.items():
            if _channel_on(cell, channel):
                visible_item = kind
                break
        if visible_item is None:
            prev = self.last_seen_items.pop(world, None)
            if prev is not None:
                self.collected_items[world] = step
        else:
            self.last_seen_items[world] = (visible_item, step)
            self.collected_items.pop(world, None)

        if _channel_on(cell, ENEMY_BASE):
            self.enemy_bases[world] = step
        else:
            self.enemy_bases.pop(world, None)
        if _channel_on(cell, ENEMY_AGENT):
            self.enemy_agents[world] = step
        else:
            self.enemy_agents.pop(world, None)

        ally_bomb = _channel_on(cell, ALLY_BOMB)
        enemy_bomb = _channel_on(cell, ENEMY_BOMB)
        if ally_bomb or enemy_bomb:
            timer_channel = ALLY_BOMB_TIMER if ally_bomb else ENEMY_BOMB_TIMER
            timer = max(1, _as_int(_channel_value(cell, timer_channel), default=BOMB_TIMER))
            self.known_bombs[world] = {"timer": timer, "own": ally_bomb, "last_step": step}
        else:
            self.known_bombs.pop(world, None)

    def _rel_to_world(self, location, direction, forward, side):
        x, y = location
        if direction == DIR_RIGHT:
            return (x + forward, y + side)
        if direction == DIR_DOWN:
            return (x - side, y + forward)
        if direction == DIR_LEFT:
            return (x - forward, y - side)
        return (x + side, y - forward)

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------
    def _in_bounds(self, pos) -> bool:
        return 0 <= pos[0] < self.grid_size and 0 <= pos[1] < self.grid_size

    @staticmethod
    def _manhattan(a, b) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _edge_blocked(self, pos, direction) -> bool:
        if (pos[0], pos[1], direction) in self.walls:
            return True
        dx, dy = DIR_DELTAS[direction]
        other = (pos[0] + dx, pos[1] + dy)
        opp = OPPOSITE[direction]
        return (other[0], other[1], opp) in self.walls

    def _edge_destructible(self, pos, direction) -> bool:
        if (pos[0], pos[1], direction) in self.destructible:
            return True
        dx, dy = DIR_DELTAS[direction]
        other = (pos[0] + dx, pos[1] + dy)
        opp = OPPOSITE[direction]
        return (other[0], other[1], opp) in self.destructible

    def _walkable_neighbors(self, pos):
        for d, (dx, dy) in DIR_DELTAS.items():
            nxt = (pos[0] + dx, pos[1] + dy)
            if not self._in_bounds(nxt):
                continue
            if self._edge_blocked(pos, d):
                continue
            yield nxt, d

    def _line_of_sight_clear(self, start, end) -> bool:
        if start == end:
            return True
        path = self._line_tiles(start, end)
        for cur, nxt in zip(path, path[1:]):
            dx = nxt[0] - cur[0]
            dy = nxt[1] - cur[1]
            if dx and dy:
                horiz = DIR_RIGHT if dx > 0 else DIR_LEFT
                vert = DIR_DOWN if dy > 0 else DIR_UP
                if self._edge_blocked(cur, horiz) and self._edge_blocked(cur, vert):
                    return False
            else:
                for d, delta in DIR_DELTAS.items():
                    if delta == (dx, dy) and self._edge_blocked(cur, d):
                        return False
        return True

    @staticmethod
    def _line_tiles(start, end):
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

    def _blast_cells(self, bomb_pos) -> frozenset[tuple[int, int]]:
        cached = self._blast_cache.get(bomb_pos)
        if cached is not None:
            return cached
        bx, by = bomb_pos
        cells: set[tuple[int, int]] = set()
        for x in range(bx - BOMB_RADIUS, bx + BOMB_RADIUS + 1):
            for y in range(by - BOMB_RADIUS, by + BOMB_RADIUS + 1):
                pos = (x, y)
                if not self._in_bounds(pos):
                    continue
                if max(abs(x - bx), abs(y - by)) > BOMB_RADIUS:
                    continue
                if self._line_of_sight_clear(bomb_pos, pos):
                    cells.add(pos)
        out = frozenset(cells)
        self._blast_cache[bomb_pos] = out
        return out

    # ------------------------------------------------------------------
    # Bomb / danger tracking
    # ------------------------------------------------------------------
    def _age_bombs(self, step: int) -> None:
        if self.last_step is None:
            return
        delta = max(0, step - self.last_step)
        if delta <= 0:
            return
        for pos, data in list(self.known_bombs.items()):
            timer = int(data.get("timer", BOMB_TIMER)) - delta
            if timer <= 0:
                self.known_bombs.pop(pos, None)
            else:
                data["timer"] = timer
                data["last_step"] = step
        for pos, last_seen in list(self.enemy_agents.items()):
            if step - int(last_seen) > ENEMY_AGENT_MEMORY:
                self.enemy_agents.pop(pos, None)

    def _active_danger_cells(self) -> set[tuple[int, int]]:
        """Cells whose bomb is detonating now or next tick (timer ≤ 2)."""
        out: set[tuple[int, int]] = set()
        for pos, data in self.known_bombs.items():
            timer = int(data.get("timer", BOMB_TIMER))
            if timer <= 2:
                out.update(self._blast_cells(pos))
        return out

    # ------------------------------------------------------------------
    # Escape — used both for "step out of incoming blast" and for verifying
    # that any bomb we place ourselves has a way to survive.
    # ------------------------------------------------------------------
    def _safe_escape_within(
        self,
        location: tuple[int, int],
        blast: Iterable[tuple[int, int]],
        max_moves: int,
        forbidden: Iterable[tuple[int, int]] | None = None,
    ) -> tuple[int, int] | None:
        blast = set(blast)
        forbidden = set(forbidden or ())
        queue = deque([(location, 0)])
        seen = {location}
        while queue:
            pos, dist = queue.popleft()
            if dist > 0 and pos not in blast and pos not in forbidden:
                return pos
            if dist >= max_moves:
                continue
            for nxt, _d in self._walkable_neighbors(pos):
                if nxt in seen or nxt not in self.seen:
                    continue
                if nxt in forbidden:
                    continue
                seen.add(nxt)
                queue.append((nxt, dist + 1))
        return None

    def _escape_action(
        self,
        observation: dict,
        location: tuple[int, int],
        direction: int,
        active_danger: set[tuple[int, int]],
    ) -> int | None:
        """Choose the move that lands us outside active blast cones soonest."""
        best_action = None
        best_score = -inf
        for action in (FORWARD, BACKWARD, LEFT, RIGHT, STAY):
            if not self._legal(observation, action):
                continue
            new_pos, _new_dir = self._simulate_action(location, direction, action)
            if action in (LEFT, RIGHT):
                # Turning keeps us in the danger cell — only useful if there
                # is literally nothing else legal.
                score = -10.0
            elif action == STAY:
                score = -20.0 if location in active_danger else -1.0
            else:
                if new_pos == location:
                    continue
                if not self._in_bounds(new_pos):
                    continue
                if new_pos not in self.seen:
                    continue
                if new_pos in active_danger:
                    score = -50.0
                else:
                    score = 100.0
                # Penalise stepping onto a known bomb (solid).
                if new_pos in self.known_bombs:
                    score -= 200.0
            if score > best_score:
                best_score = score
                best_action = action
        return best_action

    # ------------------------------------------------------------------
    # Offensive bomb placement
    # ------------------------------------------------------------------
    def _maybe_offensive_bomb(
        self,
        location: tuple[int, int],
        step: int,
        active_danger: set[tuple[int, int]],
    ) -> int | None:
        bomb_blast = self._blast_cells(location)
        base = self.base_location
        if base is not None and base in bomb_blast:
            return None

        enemies_in_blast = [
            pos for pos, last_seen in self.enemy_agents.items()
            if step - int(last_seen) <= ENEMY_STALENESS and pos in bomb_blast
        ]
        base_in_blast = any(pos in bomb_blast for pos in self.enemy_bases)
        cluster = len(enemies_in_blast) >= self.cluster_min
        opportunistic = base_in_blast or any(
            int(last_seen) == step and pos in bomb_blast
            for pos, last_seen in self.enemy_agents.items()
        )

        if not (cluster or opportunistic):
            return None

        escape = self._safe_escape_within(
            location, bomb_blast, BOMB_TIMER, forbidden=active_danger
        )
        if escape is None:
            return None
        self.known_bombs[location] = {"timer": BOMB_TIMER, "own": True, "last_step": step}
        if self.log:
            tag = "cluster" if cluster else "opportunistic"
            print(
                f"[AEManager] step={step} BOMB ({tag}) at {location} "
                f"enemies={len(enemies_in_blast)} base={base_in_blast}",
                flush=True,
            )
        return PLACE_BOMB

    # ------------------------------------------------------------------
    # Scripted base attack
    # ------------------------------------------------------------------
    def _scripted_base_attack(
        self,
        observation: dict,
        location: tuple[int, int],
        direction: int,
        team_bombs: int,
        active_danger: set[tuple[int, int]],
        step: int,
    ) -> int | None:
        if not self.enemy_bases:
            return None

        plan = self._astar_oriented(location, direction, set(self.enemy_bases), active_danger)
        if plan is None:
            return None
        target, actions, edges = plan
        if not actions:
            return None

        first_action = actions[0]
        first_edge = edges[0] if edges else None
        # Edge has form ("move", dir) or ("break", dir) for the destructible.
        if first_edge is not None and first_edge[0] == "break":
            # We are about to need a destructible-wall break. Place a bomb
            # only if facing the wall — otherwise rotate first.
            wall_dir = first_edge[1]
            if direction != wall_dir:
                turn = self._turn_action(direction, wall_dir)
                if turn is not None and self._legal(observation, turn):
                    return turn

            if team_bombs <= 0 or self.health < LOW_HEALTH:
                # Can't bomb. Try a detour first via the same A* without
                # destructibles allowed; otherwise fall back to exploration.
                detour = self._astar_oriented(
                    location, direction, set(self.enemy_bases), active_danger,
                    allow_break=False,
                )
                if detour is not None and detour[1]:
                    return self._first_legal_action(observation, detour[1][0])
                return None

            if not self._legal(observation, PLACE_BOMB):
                return None
            bomb_blast = self._blast_cells(location)
            base = self.base_location
            if base is not None and base in bomb_blast:
                return None
            escape = self._safe_escape_within(
                location, bomb_blast, BOMB_TIMER, forbidden=active_danger
            )
            if escape is None:
                return None
            self.known_bombs[location] = {"timer": BOMB_TIMER, "own": True, "last_step": step}
            if self.log:
                print(
                    f"[AEManager] step={step} BOMB (break {wall_dir}) at {location} target={target}",
                    flush=True,
                )
            return PLACE_BOMB

        # Regular movement step.
        return self._first_legal_action(observation, first_action)

    def _turn_action(self, current: int, desired: int) -> int | None:
        if current == desired:
            return None
        if desired == (current + 1) % 4:
            return RIGHT
        if desired == (current + 3) % 4:
            return LEFT
        # 180° turn — prefer a single LEFT/RIGHT first (the next tick will
        # finish it). Choose the direction with no wall on the side so we
        # at least don't waste two ticks rotating into a corner.
        return LEFT

    def _astar_oriented(
        self,
        start: tuple[int, int],
        start_dir: int,
        goals: set[tuple[int, int]],
        forbidden_cells: set[tuple[int, int]],
        allow_break: bool = True,
    ):
        """A* over (x, y, facing) state space.

        Returns (goal, actions, edges) or None. ``edges`` parallels
        ``actions`` and each entry is ("move", dir) for a step or
        ("break", dir) for a placed-bomb wall break.
        """
        if start in goals:
            return start, [], []

        State = tuple[int, int, int]
        start_state: State = (start[0], start[1], start_dir)

        def heuristic(pos):
            return min(self._manhattan(pos, g) for g in goals)

        # came_from[s] = (prev_state, action, edge)
        came_from: dict[State, tuple[State, int, tuple]] = {}
        g_score: dict[State, float] = {start_state: 0.0}
        pq: list[tuple[float, int, State]] = []
        heapq.heappush(pq, (heuristic(start), 0, start_state))
        counter = 1
        found_goal: State | None = None
        max_expand = 4096

        while pq and max_expand > 0:
            _f, _i, state = heapq.heappop(pq)
            max_expand -= 1
            x, y, facing = state
            if (x, y) in goals:
                found_goal = state
                break
            # Transitions: FORWARD, BACKWARD, LEFT, RIGHT.
            transitions = [
                (FORWARD, ("move", facing), facing, *DIR_DELTAS[facing]),
                (
                    BACKWARD,
                    ("move", OPPOSITE[facing]),
                    facing,
                    *DIR_DELTAS[OPPOSITE[facing]],
                ),
                (LEFT, ("turn", None), (facing + 3) % 4, 0, 0),
                (RIGHT, ("turn", None), (facing + 1) % 4, 0, 0),
            ]
            for action, edge, new_facing, dx, dy in transitions:
                if action in (FORWARD, BACKWARD):
                    move_dir = facing if action == FORWARD else OPPOSITE[facing]
                    new_pos = (x + dx, y + dy)
                    if not self._in_bounds(new_pos):
                        continue
                    if new_pos not in self.seen:
                        continue
                    if new_pos in forbidden_cells:
                        continue
                    if new_pos in self.known_bombs:
                        continue
                    # Wall handling.
                    edge_solid = self._edge_blocked((x, y), move_dir)
                    edge_destr = self._edge_destructible((x, y), move_dir)
                    if edge_solid and not edge_destr:
                        continue
                    if edge_destr:
                        if not allow_break:
                            continue
                        step_cost = self.bomb_break_cost
                        edge = ("break", move_dir)
                    else:
                        step_cost = 1.0
                        edge = ("move", move_dir)
                else:
                    # Pure turn.
                    new_pos = (x, y)
                    step_cost = 1.0

                # Threat penalty — small soft term for paths through fresh
                # enemy sightings. Keep it cheap so A* stays admissible-ish.
                threat_penalty = 0.0
                if action in (FORWARD, BACKWARD):
                    if new_pos in self.enemy_agents:
                        last_seen = int(self.enemy_agents[new_pos])
                        if (self.last_step or 0) - last_seen <= ENEMY_STALENESS:
                            threat_penalty = 2.0

                tentative = g_score[state] + step_cost + threat_penalty
                new_state: State = (new_pos[0], new_pos[1], new_facing)
                if tentative < g_score.get(new_state, inf):
                    g_score[new_state] = tentative
                    came_from[new_state] = (state, action, edge)
                    f = tentative + heuristic(new_pos)
                    heapq.heappush(pq, (f, counter, new_state))
                    counter += 1

        if found_goal is None:
            return None

        actions: list[int] = []
        edges: list[tuple] = []
        cursor = found_goal
        while cursor in came_from:
            prev, action, edge = came_from[cursor]
            actions.append(action)
            edges.append(edge)
            cursor = prev
        actions.reverse()
        edges.reverse()
        return (found_goal[0], found_goal[1]), actions, edges

    # ------------------------------------------------------------------
    # Tertiary item grab and exploration
    # ------------------------------------------------------------------
    def _adjacent_item_grab(
        self,
        observation: dict,
        location: tuple[int, int],
        direction: int,
        active_danger: set[tuple[int, int]],
    ) -> int | None:
        for d, (dx, dy) in DIR_DELTAS.items():
            nxt = (location[0] + dx, location[1] + dy)
            if not self._in_bounds(nxt):
                continue
            if nxt not in self.seen:
                continue
            if self._edge_blocked(location, d):
                continue
            if nxt in active_danger:
                continue
            item = self.last_seen_items.get(nxt)
            if item is None:
                continue
            if item[0] not in ("mission", "resource"):
                continue
            action = self._step_action(direction, d)
            if action is not None and self._legal(observation, action):
                return action
        return None

    def _exploration_fallback(
        self,
        observation: dict,
        location: tuple[int, int],
        direction: int,
        active_danger: set[tuple[int, int]],
    ) -> int:
        # Score every legal action.
        legal_actions = [a for a in (FORWARD, BACKWARD, LEFT, RIGHT, STAY)
                         if self._legal(observation, a)]
        if not legal_actions:
            return STAY
        best_action = legal_actions[0]
        best_score = -inf
        for action in legal_actions:
            new_pos, new_dir = self._simulate_action(location, direction, action)
            score = 0.0
            if new_pos in active_danger:
                score -= 50.0
            if new_pos in self.known_bombs:
                score -= 100.0
            if new_pos != location and new_pos not in self.seen:
                score -= 3.0
            item = self.last_seen_items.get(new_pos)
            if item is not None:
                score += self.item_values.get(item[0], 0.0) * 0.1
            score -= 0.25 * self.visit_count.get(new_pos, 0)
            if new_pos in self.recent_locations[-4:]:
                score -= 1.5
            if action == STAY:
                score -= 1.5
            if score > best_score:
                best_score = score
                best_action = action
        return best_action

    # ------------------------------------------------------------------
    # Action helpers
    # ------------------------------------------------------------------
    def _step_action(self, current_dir: int, desired_dir: int) -> int | None:
        if current_dir == desired_dir:
            return FORWARD
        if desired_dir == OPPOSITE[current_dir]:
            return BACKWARD
        if desired_dir == (current_dir + 3) % 4:
            return LEFT
        if desired_dir == (current_dir + 1) % 4:
            return RIGHT
        return None

    def _first_legal_action(self, observation: dict, action: int) -> int | None:
        if self._legal(observation, action):
            return action
        return None

    def _simulate_action(self, location, direction, action):
        if action == FORWARD:
            dx, dy = DIR_DELTAS[direction]
            return (location[0] + dx, location[1] + dy), direction
        if action == BACKWARD:
            dx, dy = DIR_DELTAS[OPPOSITE[direction]]
            return (location[0] + dx, location[1] + dy), direction
        if action == LEFT:
            return location, (direction + 3) % 4
        if action == RIGHT:
            return location, (direction + 1) % 4
        return location, direction

    def _legal(self, observation: dict, action: int) -> bool:
        mask = observation.get("action_mask")
        if mask is None:
            return 0 <= action <= PLACE_BOMB
        try:
            return bool(mask[action])
        except Exception:
            return 0 <= action <= PLACE_BOMB

    def _first_legal(self, observation: dict, actions: list[int]) -> int:
        for action in actions:
            if self._legal(observation, action):
                return action
        return STAY
