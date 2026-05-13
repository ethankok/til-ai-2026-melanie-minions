"""Stateful frontier/objective planner for the AE task.

The qualifier evaluator calls this manager once per controlled-agent turn.  The
agent only sees egocentric view tensors, so this keeps a small belief map inside
``AEManager`` and plans over known safe cells.  It intentionally uses no external
runtime dependencies: the Docker image can stay unchanged for this heuristic
baseline.
"""

from __future__ import annotations

from collections import deque
from math import inf
from typing import Iterable


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

    GRID_SIZE = 16
    # Bomb timer matches til_environment/bomberman_config.yaml (entities.bomb.timer).
    # Phase order each round is: place → move → detonation → upkeep, so a bomb
    # placed at step N detonates after the agent's movement at step N+timer.
    BOMB_TIMER = 3
    BOMB_RADIUS = 2
    # How long we still consider an enemy-agent sighting "threatening" in steps.
    ENEMY_STALENESS = 3
    # Reachability radius (BFS steps from blast cell) for predictive bomb hits.
    PREDICTIVE_BOMB_RANGE = 2
    # Tiles respawn after this many steps per env config (env.tile_respawn_steps).
    TILE_RESPAWN_STEPS = 40
    # Below this health the agent prefers safe cells over aggressive plays.
    LOW_HEALTH_THRESHOLD = 20
    # Soft-threat scoring weights (lower = more aggressive; tuned for planner-v3).
    PATH_THREAT_PENALTY = 1.0
    CELL_THREAT_PENALTY = 3.0
    # Distance (Manhattan) within which an enemy near our base becomes a defense target.
    BASE_DEFENSE_RADIUS = 4

    def __init__(self):
        self.grid_size = self.GRID_SIZE
        self.last_step: int | None = None
        self.turn_counter = 0
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

        location = self._location(observation.get("location"))
        direction = self._as_int(observation.get("direction"), default=self.DIR_RIGHT) % 4
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

        danger = self._danger_cells()
        low_health = self.health < self.LOW_HEALTH_THRESHOLD

        escape_path = self._active_escape_path(location, step)
        if escape_path is not None:
            target, path = self.escape_target, escape_path
        else:
            # Fast path: an obviously-dominant action skips full candidate scoring.
            dominant = self._try_dominant_action(observation, location, direction, danger, low_health)
            if dominant is not None:
                return dominant
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
        # Per-turn blast cell cache; cleared at the start of every ae() call.
        self._blast_cache: dict[tuple[int, int], frozenset[tuple[int, int]]] = {}
        self.last_step = None

    def _update_memory(
        self,
        observation: dict,
        step: int,
        location: tuple[int, int] | None,
        direction: int,
    ) -> None:
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
        else:
            self.last_seen_items[world] = (visible_item, step)
            # If we see it again, it must have respawned; clear collected entry.
            self.collected_items.pop(world, None)

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
    def _choose_target(
        self,
        start: tuple[int, int],
        danger: set[tuple[int, int]],
        low_health: bool = False,
    ) -> tuple[tuple[int, int] | None, list[tuple[int, int]] | None]:
        threats = self._enemy_threat_cells()
        step = self.last_step if self.last_step is not None else 0

        # Single multi-source BFS gives distance to every reachable cell at
        # roughly the cost of one of the old per-target BFS calls.
        distance, parent = self._bfs_distance_map(start, danger)

        candidates: list[tuple[float, tuple[int, int]]] = []

        # When health is low, avoid aggressive targets (enemy bases, base
        # defense) and stick to items/exploration so we don't die in a melee.
        if not low_health:
            for pos in self.enemy_bases:
                candidates.append((80.0, pos))
            # Base defense: enemies near our base become high-priority bomb
            # targets — losing the base is -50, so a 30+ value swing is worth it.
            if self.base_location is not None:
                for pos, last_seen in self.enemy_agents.items():
                    if step - int(last_seen) > self.ENEMY_STALENESS:
                        continue
                    if self._manhattan(pos, self.base_location) <= self.BASE_DEFENSE_RADIUS:
                        candidates.append((60.0, pos))

        for pos, (kind, _step) in self.last_seen_items.items():
            candidates.append((self.ITEM_VALUES.get(kind, 1.0), pos))

        # Respawn awareness: items we saw get collected become candidates
        # again once tile_respawn_steps have elapsed. Slight discount because
        # the respawn is probabilistic, not guaranteed.
        for pos, (kind, collected_step) in self.collected_items.items():
            if step - collected_step >= self.TILE_RESPAWN_STEPS:
                candidates.append((0.7 * self.ITEM_VALUES.get(kind, 1.0), pos))

        # Weight frontier cells by how much new area they likely reveal.
        for pos in self._frontier_cells():
            unseen_neighbors = sum(1 for n in self._raw_neighbors(pos) if n not in self.seen)
            candidates.append((4.0 + 1.0 * unseen_neighbors, pos))

        # Anti-stall fallback: known safe low-visit cells.
        for pos in self.seen:
            if pos != start:
                candidates.append((2.0 - 0.08 * self.visit_count.get(pos, 0), pos))

        best_target = None
        best_score = -inf
        for base_value, pos in candidates:
            if pos == start or pos not in distance:
                continue
            dist = distance[pos]
            score = base_value - 1.15 * dist - 0.25 * self.visit_count.get(pos, 0)
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
                score -= self.PATH_THREAT_PENALTY * path_threat
            if score > best_score:
                best_score = score
                best_target = pos

        if best_target is None:
            return None, None
        return best_target, self._reconstruct_path(parent, start, best_target)

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

        Two short-circuits, both with the same safety constraints as the slow
        path:

        - an enemy base sits in the immediate bomb blast, we have a bomb, the
          cell is safe, and we have a verified escape → ``PLACE_BOMB``;
        - a mission tile is one step away through a clear edge → step toward it.
        """

        # Hunting bombs are off when health is critically low.
        if (not low_health
                and self._legal(observation, self.PLACE_BOMB)
                and self._as_int(observation.get("team_bombs"), default=0) > 0
                and location not in danger):
            bomb_blast = self._blast_cells(location)
            base_loc = self.base_location
            if (base_loc is None or base_loc not in bomb_blast) and any(
                pos in bomb_blast for pos in self.enemy_bases
            ):
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
                    return self.PLACE_BOMB

        # Adjacent mission grab — purely a speed optimization, not a behavior
        # change; the slow path would pick the same move.
        for d, (dx, dy) in self.DIR_DELTAS.items():
            nxt = (location[0] + dx, location[1] + dy)
            if nxt not in self.seen or nxt in danger:
                continue
            if self._edge_blocked(location, d):
                continue
            item = self.last_seen_items.get(nxt)
            if item is None or item[0] != "mission":
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
            else:
                data["timer"] = timer
                data["last_step"] = step

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
            if data.get("own"):
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
        base_location = self.base_location or self._location(observation.get("base_location"))
        if base_location is not None and base_location in bomb_blast:
            return False

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

        # Predictive bombing: enemies near the blast cone may step into it
        # before our bomb expires (fuse is 3 ticks; we conservatively count
        # any recent enemy that is BFS-reachable to a blast cell within
        # PREDICTIVE_BOMB_RANGE moves).
        if not tactical_target and self.enemy_agents:
            extended = self._extended_blast(bomb_blast, self.PREDICTIVE_BOMB_RANGE)
            for pos, last_seen in self.enemy_agents.items():
                if step - int(last_seen) > self.ENEMY_STALENESS:
                    continue
                if pos in extended:
                    tactical_target = True
                    break

        wall_to_open = False
        # Proactive wall break: if the target is high-value (enemy base or
        # mission) and a destructible wall sits between us and it, bomb
        # without waiting to be visibly stuck.
        if target is not None:
            target_dir = self._rough_direction(location, target)
            if target_dir is not None and (location[0], location[1], target_dir) in self.destructible:
                target_kind = self.last_seen_items.get(target, (None, None))[0]
                if target in self.enemy_bases or target_kind == "mission":
                    wall_to_open = True
                elif target not in self.enemy_bases:
                    wall_to_open = self._stuck_recently()

        # Bomb-chain heuristic: even without a current target, if breaking an
        # adjacent destructible wall would reveal a short path to a base or
        # mission, take that bomb now.
        if not tactical_target and not wall_to_open:
            for d in self.DIR_DELTAS:
                if (location[0], location[1], d) not in self.destructible:
                    continue
                if self._wall_break_reveals_high_value(location, d):
                    wall_to_open = True
                    break

        if not tactical_target and not wall_to_open:
            return False
        escape_target = self._safe_escape_within(location, bomb_blast, self.BOMB_TIMER)
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
                # Low-health agent can usefully hide one tick to recover; mild
                # penalty instead of strong avoidance.
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
