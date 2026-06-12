"""ScriptedBaseAttackPolicy — scripted attack-route decision policy.

Decision flow:

  1. Parse location, facing, step, team_bombs (delegated to AEManager).
  2. Sync own/enemy bases (delegated).
  3. Update visible memory from agent view and base view (delegated).
  4. Detect loop/stuck state.
  5. Escape enemy bomb danger if needed.
  6. If no bomb stock, decline to BC/heuristic.
  7. If at planned attack square and blast reaches target base, bomb.
  8. Try tactical visible-enemy bombing.
  9. Try own-base defense.
  10. Continue committed attack route.
  11. If no committed route, choose a new attack plan.
  12. If no confident attack plan, decline to BC/heuristic.

Composition over inheritance: the policy accepts an AEManager and reads its
already-synced state. AEManager's `ae(obs)` method does all the heavy lifting
of observation parsing, bomb tracking, etc. The scripted policy adds:
  - target_base / attack_square commitment
  - orientation-aware route + bomb-from-attack-square decision

Return contract: `act(obs) -> int | None`. None means "decline → fallback".
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ae_manager import AEManager


MAX_ATTACK_TARGET_AGE = 90
MAX_BASE_DEFENSE_ROUTE_COST = 7.5
TACTICAL_AGENT_BOMB_MIN_TARGETS = 2
TACTICAL_SINGLE_BLOCKER_STUCK_TURNS = 4


class ScriptedBaseAttackPolicy:
    """Scripted-first decision policy for the AE task.

    Reads observation-derived state from a host AEManager (call
    `host.ae(obs)` first OR ensure host._sync_step has been called). Selects
    an action based on the decision tree above.
    """

    def __init__(self, host: "AEManager") -> None:
        self.host = host
        # Attack plan commitment
        self.target_base: tuple[int, int] | None = None
        self.attack_square: tuple[int, int] | None = None
        self.attack_target_age: int = 0
        self.last_plan_step: int | None = None
        # Loop/stuck detection state
        self.recent_locations: list[tuple[int, int]] = []
        self.loop_escape_active: bool = False
        # Diagnostics
        self.scripted_decisions: dict[str, int] = {}
        self.decline_reasons: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def act(self, observation: dict) -> int | None:
        """Return a legal action or None to decline.

        The host AEManager must already have processed this observation via
        `host.ae(obs)` so that its memory / state reflects this step. The
        caller (today: the opponent proxies in training/ae/opponents.py)
        handles that ordering.
        """
        host = self.host
        step = host.last_step if host.last_step is not None else 0
        location = host._location(observation.get("location"))
        if location is None:
            return self._decline("no_location")
        direction = host._as_int(observation.get("direction"), default=0) % 4

        self._update_loop_state(location)
        team_bombs = host._as_int(observation.get("team_bombs"), default=0)

        # Enemy-bomb escape supersedes normal action selection, even with the
        # global heuristic flag off.
        escape_action = self._enemy_bomb_escape(observation, location, direction)
        if escape_action is not None:
            self._record("enemy_bomb_escape")
            return escape_action

        # Bomb from attack square if we're committed and at target.
        if self._at_attack_square(location):
            if self._bomb_reaches_target(location) and host._legal(observation, host.PLACE_BOMB):
                if self._bomb_has_escape(location):
                    self._record("bomb_attack_square")
                    return host.PLACE_BOMB

        cluster_action = self._cluster_bomb_action(observation, location)
        if cluster_action is not None:
            self._record("cluster_bomb")
            return cluster_action

        # Bomb a single visible enemy if we're stuck/looping and they're in our blast.
        single_blocker = self._single_blocker_bomb(observation, location)
        if single_blocker is not None:
            self._record("single_blocker_bomb")
            return single_blocker

        defense_action = self._own_base_defense(observation, location, direction)
        if defense_action is not None:
            self._record("own_base_defense")
            return defense_action

        # Continue or pick a new attack plan.
        self._maybe_clear_stale_plan(location, step)
        if self.target_base is None or self.attack_square is None:
            self._pick_attack_plan(location, direction, observation, step)

        if self.target_base is None or self.attack_square is None:
            return self._decline("no_attack_plan")
        if team_bombs <= 0 and not self._at_attack_square(location):
            return self._decline("no_bomb_stock")

        action = self._route_step(location, direction, observation)
        if action is None:
            return self._decline("route_infeasible")
        self._record(f"route_to_{self.target_base}")
        self.attack_target_age += 1
        return action

    def reset_plan(self) -> None:
        self.target_base = None
        self.attack_square = None
        self.attack_target_age = 0
        self.last_plan_step = None
        self.recent_locations = []
        self.loop_escape_active = False

    # ------------------------------------------------------------------
    # Enemy-bomb escape integration
    # ------------------------------------------------------------------
    def _enemy_bomb_escape(
        self,
        observation: dict,
        location: tuple[int, int],
        direction: int,
    ) -> int | None:
        """Enemy bomb (own==False) with timer<=2 in our blast.

        Reuse the host AEManager's _enemy_bomb_only_escape helper, but force
        the flag on for the scripted path even if the global env var is off.
        """
        host = self.host
        prev = host.enemy_bomb_escape_enabled
        host.enemy_bomb_escape_enabled = True
        try:
            return host._enemy_bomb_only_escape(observation, location, direction)
        finally:
            host.enemy_bomb_escape_enabled = prev

    # ------------------------------------------------------------------
    # Tactical cluster bombing
    # ------------------------------------------------------------------
    def _bomb_has_escape(self, location: tuple[int, int]) -> bool:
        """Does PLACE_BOMB at `location` leave us with a non-blast safe cell?"""
        host = self.host
        blast = host._blast_cells(location)
        for n in host._raw_neighbors(location):
            if not host._in_bounds(n):
                continue
            if n in blast:
                continue
            if n in host.known_bombs:
                continue
            if n in host.seen:
                return True
        # Last resort: if the agent's own location is OUT of the blast cells
        # (some line-of-sight geometries leave the placer safe).
        return location not in blast

    def _visible_enemies(self) -> set[tuple[int, int]]:
        host = self.host
        step = host.last_step if host.last_step is not None else 0
        out: set[tuple[int, int]] = set()
        for pos, last_seen in host.enemy_agents.items():
            if step - int(last_seen) > 1:
                continue
            out.add(pos)
        return out

    def _ally_bomb_blast_cells(self) -> set[tuple[int, int]]:
        """Union of blast cells already covered by ally bombs."""
        host = self.host
        out: set[tuple[int, int]] = set()
        for bomb_pos, data in host.known_bombs.items():
            if data.get("own"):
                out.update(host._blast_cells(bomb_pos))
        return out

    def _cluster_bomb_action(
        self,
        observation: dict,
        location: tuple[int, int],
    ) -> int | None:
        host = self.host
        team_bombs = host._as_int(observation.get("team_bombs"), default=0)
        if team_bombs <= 0:
            return None
        if not host._legal(observation, host.PLACE_BOMB):
            return None
        enemies = self._visible_enemies()
        if len(enemies) < TACTICAL_AGENT_BOMB_MIN_TARGETS:
            return None
        blast = host._blast_cells(location)
        # Don't bomb a position that would damage own base.
        if host.base_location is not None and host.base_location in blast:
            return None
        ally_covered = self._ally_bomb_blast_cells()
        # Count visible enemies in our blast, not already covered by an ally bomb.
        in_blast_uncovered = [e for e in enemies if e in blast and e not in ally_covered]
        if len(in_blast_uncovered) < TACTICAL_AGENT_BOMB_MIN_TARGETS:
            return None
        if not self._bomb_has_escape(location):
            return None
        return host.PLACE_BOMB

    # ------------------------------------------------------------------
    # Loop / stuck detection + single-blocker bombing
    # ------------------------------------------------------------------
    def _update_loop_state(self, location: tuple[int, int]) -> None:
        self.recent_locations.append(location)
        if len(self.recent_locations) > 12:
            del self.recent_locations[: len(self.recent_locations) - 12]

    def _is_stuck_or_looping(self, location: tuple[int, int]) -> bool:
        """Stuck/loop triggers.

        - currently in loop escape mode; OR
        - recent tail of 4 locations is all the current location; OR
        - last 8 locations have <= 3 unique cells AND <= 3 physical moves; OR
        - blocker adjacent + current cell visited >= 5 + recent window <= 4 unique.
        """
        host = self.host
        if self.loop_escape_active:
            return True
        if len(self.recent_locations) >= 4:
            tail4 = self.recent_locations[-4:]
            if all(c == location for c in tail4):
                return True
        if len(self.recent_locations) >= 8:
            last8 = self.recent_locations[-8:]
            unique8 = len(set(last8))
            moves = sum(1 for i in range(1, len(last8)) if last8[i] != last8[i - 1])
            if unique8 <= 3 and moves <= 3:
                return True
        if host.visit_count.get(location, 0) >= 5 and len(self.recent_locations) >= 4:
            unique_recent = len(set(self.recent_locations[-4:]))
            if unique_recent <= 4:
                enemies_near = any(
                    host._manhattan(location, e) <= 1
                    for e in self._visible_enemies()
                )
                if enemies_near:
                    return True
        return False

    def _single_blocker_bomb(
        self,
        observation: dict,
        location: tuple[int, int],
    ) -> int | None:
        host = self.host
        team_bombs = host._as_int(observation.get("team_bombs"), default=0)
        if team_bombs <= 0:
            return None
        if not host._legal(observation, host.PLACE_BOMB):
            return None
        if not self._is_stuck_or_looping(location):
            return None
        enemies = self._visible_enemies()
        blast = host._blast_cells(location)
        if host.base_location is not None and host.base_location in blast:
            return None
        in_blast = [e for e in enemies if e in blast]
        if len(in_blast) != 1:
            return None
        # Ignore if already covered by an ally bomb.
        if in_blast[0] in self._ally_bomb_blast_cells():
            return None
        if not self._bomb_has_escape(location):
            return None
        return host.PLACE_BOMB

    # ------------------------------------------------------------------
    # Own-base defense
    # ------------------------------------------------------------------
    def _enemy_base_threats(self) -> list[tuple[int, int]]:
        """Visible enemy agents whose blast (if they bomb in place) reaches our base."""
        host = self.host
        if host.base_location is None:
            return []
        ally_covered = self._ally_bomb_blast_cells()
        threats = []
        for enemy in self._visible_enemies():
            enemy_blast = host._blast_cells(enemy)
            if host.base_location not in enemy_blast:
                continue
            if enemy in ally_covered:
                continue
            threats.append(enemy)
        return threats

    def _own_base_defense(
        self,
        observation: dict,
        location: tuple[int, int],
        direction: int,
    ) -> int | None:
        host = self.host
        if host.base_location is None:
            return None
        threats = self._enemy_base_threats()
        if not threats:
            return None
        # Case A: bomb threat from current cell without hitting own base.
        team_bombs = host._as_int(observation.get("team_bombs"), default=0)
        if team_bombs > 0 and host._legal(observation, host.PLACE_BOMB):
            blast = host._blast_cells(location)
            if host.base_location not in blast and any(t in blast for t in threats):
                if self._bomb_has_escape(location):
                    return host.PLACE_BOMB
        # Case B: route to a defensive bomb square (cost <= 7.5).
        danger = host._danger_cells()
        if (
            getattr(host, "orientation_aware_path_enabled", False)
            and getattr(host, "is_fixed_novice_map", False)
        ):
            distance, parent = host._orientation_aware_distance_map(location, direction, danger)
        elif getattr(host, "is_fixed_novice_map", False):
            distance, parent = host._dijkstra_distance_map(location, danger)
        else:
            distance, parent = host._bfs_distance_map(location, danger)
        best_square: tuple[int, int] | None = None
        best_cost = MAX_BASE_DEFENSE_ROUTE_COST + 1.0
        for cell, cost in distance.items():
            if cost > MAX_BASE_DEFENSE_ROUTE_COST:
                continue
            if cell == host.base_location:
                continue
            cell_blast = host._blast_cells(cell)
            if host.base_location in cell_blast:
                continue
            if not any(t in cell_blast for t in threats):
                continue
            if cost < best_cost:
                best_cost = cost
                best_square = cell
        if best_square is None:
            return None
        path = host._reconstruct_path(parent, location, best_square)
        if path is None:
            return None
        action = host._action_for_path(location, direction, path)
        if action is None or not host._legal(observation, action):
            return None
        return int(action)

    # ------------------------------------------------------------------
    # Plan lifecycle
    # ------------------------------------------------------------------
    def _at_attack_square(self, location: tuple[int, int]) -> bool:
        return self.attack_square is not None and location == self.attack_square

    def _bomb_reaches_target(self, location: tuple[int, int]) -> bool:
        if self.target_base is None:
            return False
        blast = self.host._blast_cells(location)
        return self.target_base in blast

    def _maybe_clear_stale_plan(self, location: tuple[int, int], step: int) -> None:
        if self.target_base is None:
            return
        host = self.host
        if self.target_base not in host.enemy_bases:
            self.reset_plan()
            self._note_decline("target_inactive")
            return
        if self.attack_target_age > MAX_ATTACK_TARGET_AGE:
            self.reset_plan()
            self._note_decline("plan_age_exceeded")
            return
        if self.attack_square is None:
            self.reset_plan()

    def _pick_attack_plan(
        self,
        location: tuple[int, int],
        direction: int,
        observation: dict,
        step: int,
    ) -> None:
        """Score every active enemy base × its valid attack squares.

        Picks the (target_base, attack_square) pair with lowest
        `route_cost + rank_penalty + target_cost`. Rank comes from
        spawn_first_targets if available. Route cost is orientation-aware
        when on fixed novice map.
        """
        host = self.host
        if not host.enemy_bases:
            return
        danger = host._danger_cells()
        if (
            getattr(host, "orientation_aware_path_enabled", False)
            and getattr(host, "is_fixed_novice_map", False)
        ):
            distance, parent = host._orientation_aware_distance_map(location, direction, danger)
        elif getattr(host, "is_fixed_novice_map", False):
            distance, parent = host._dijkstra_distance_map(location, danger)
        else:
            distance, parent = host._bfs_distance_map(location, danger)

        # Rank lookup
        try:
            from spawn_first_targets import get_first_target_rank  # noqa: WPS433
        except Exception:
            def get_first_target_rank(_a, _b):  # type: ignore[no-redef]
                return None

        best: tuple[float, tuple[int, int], tuple[int, int]] | None = None
        for enemy_base in host.enemy_bases:
            rank = get_first_target_rank(host.base_location, enemy_base)
            rank_penalty = 0.0 if rank is None else 2.0 * float(rank)
            # Enumerate every reachable cell whose blast contains this enemy base.
            for cell in distance:
                if cell == enemy_base:
                    continue
                if cell not in host.seen:
                    continue
                blast = host._blast_cells(cell)
                if enemy_base not in blast:
                    continue
                route_cost = float(distance[cell])
                if not math.isfinite(route_cost):
                    continue
                # Revisited attack square + slight preference for attack squares closer to own base.
                target_cost = 0.25 * float(host.visit_count.get(cell, 0))
                if host.base_location is not None:
                    target_cost += 0.05 * float(host._manhattan(cell, host.base_location))
                score = route_cost + rank_penalty + target_cost
                if best is None or score < best[0]:
                    best = (score, enemy_base, cell)
        if best is None:
            return
        _score, enemy_base, attack_square = best
        self.target_base = enemy_base
        self.attack_square = attack_square
        self.attack_target_age = 0
        self.last_plan_step = step
        self._record(f"plan_{enemy_base}_via_{attack_square}")

    def _route_step(
        self,
        location: tuple[int, int],
        direction: int,
        observation: dict,
    ) -> int | None:
        """Compute next legal action toward `self.attack_square`.

        Returns None if no legal forward step exists (caller declines).
        """
        host = self.host
        if self.attack_square is None:
            return None
        danger = host._danger_cells()
        if (
            getattr(host, "orientation_aware_path_enabled", False)
            and getattr(host, "is_fixed_novice_map", False)
        ):
            _dist, parent = host._orientation_aware_distance_map(location, direction, danger)
        elif getattr(host, "is_fixed_novice_map", False):
            _dist, parent = host._dijkstra_distance_map(location, danger)
        else:
            _dist, parent = host._bfs_distance_map(location, danger)
        if self.attack_square not in parent:
            return None
        path = host._reconstruct_path(parent, location, self.attack_square)
        if path is None:
            return None
        action = host._action_for_path(location, direction, path)
        if action is None:
            return None
        if not host._legal(observation, action):
            return None
        return int(action)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def _record(self, key: str) -> None:
        self.scripted_decisions[key] = self.scripted_decisions.get(key, 0) + 1

    def _note_decline(self, key: str) -> None:
        self.decline_reasons[key] = self.decline_reasons.get(key, 0) + 1

    def _decline(self, reason: str) -> None:
        self._note_decline(reason)
        return None
