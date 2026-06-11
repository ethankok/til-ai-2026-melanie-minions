"""Bomb safety — the one module that answers "is this bomb safe?".

Owns every derived bomb computation for the AE planner:

- blast geometry           (``blast_cells``)
- chain-resolved danger    (``danger_layers``, ``on_fire_at``, ``danger_cells``)
- placement vetoes         (``hits_enemy_base``, ``own_base_vetoes``,
                            ``escape_required``)
- escape search            (``safe_escape_within``, ``nearest_escape_cell``,
                            ``active_escape_path``)
- bomb-commit bookkeeping  (``commit_bomb``, ``revert_commit``)

The module operates on the host ``AEManager``'s belief blackboard rather than
copying it. Everything it touches on the host is listed here — this list IS
the host side of the interface:

reads:
    seen, walls (via _line_of_sight_clear), known_bombs, enemy_bases,
    last_step, BOMB_TIMER, BOMB_RADIUS, danger_horizon,
    no_self_damage, basekill_noescape, time_danger_enabled,
    _neighbors, _in_bounds, _line_of_sight_clear, _bfs
reads/writes:
    _blast_cache, _danger_layers_cache   (per-tick caches; the host clears
                                          them at the top of every ae() call)
    escape_target, escape_until_step     (the active escape commitment)
    _tick_bomb_commit                    (this tick's synthetic bomb commit;
                                          the host resets it every tick)

Decision policy — WHAT is worth bombing (enemy base, cluster, predictive
walk, wall break) — stays in ``AEManager._should_place_bomb``; this module
only answers whether a placement is safe and where to run afterwards.

Env flags consumed (through host attributes set in ``AEManager.__init__``):
AE_NO_SELF_DAMAGE, AE_BASEKILL_NOESCAPE, AE_TIME_DANGER, AE_DANGER_HORIZON.
"""

from __future__ import annotations

from collections import deque


class BombSafety:
    """Blast geometry, chain danger, escape search and commit/rollback."""

    def __init__(self, host) -> None:
        self._m = host

    # ------------------------------------------------------------------
    # Blast geometry
    # ------------------------------------------------------------------
    def blast_cells(self, bomb_pos: tuple[int, int]) -> set[tuple[int, int]]:
        m = self._m
        cached = m._blast_cache.get(bomb_pos)
        if cached is not None:
            return set(cached)
        bx, by = bomb_pos
        cells = set()
        for x in range(bx - m.BOMB_RADIUS, bx + m.BOMB_RADIUS + 1):
            for y in range(by - m.BOMB_RADIUS, by + m.BOMB_RADIUS + 1):
                pos = (x, y)
                if not m._in_bounds(pos):
                    continue
                if max(abs(x - bx), abs(y - by)) > m.BOMB_RADIUS:
                    continue
                if m._line_of_sight_clear(bomb_pos, pos):
                    cells.add(pos)
        m._blast_cache[bomb_pos] = frozenset(cells)
        return cells

    # ------------------------------------------------------------------
    # Chain-resolved danger
    # ------------------------------------------------------------------
    def danger_layers(self) -> list[set[tuple[int, int]]]:
        """Per-tick lethality: ``layers[t]`` = cells on fire at relative future
        tick ``t`` (0..danger_horizon), resolving enemy-bomb chains.

        A bomb with ``timer = d`` detonates ``d`` of our decision-steps out.
        If bomb B's cell is inside bomb A's blast and A fires earlier, B
        detonates at A's tick; propagate this to a fixpoint so transitive
        chains collapse to the earliest trigger. Cached per turn.
        """
        m = self._m
        if m._danger_layers_cache is not None:
            return m._danger_layers_cache
        horizon = m.danger_horizon
        # Resolved detonation tick per bomb (start from its own timer).
        fire_tick: dict[tuple[int, int], int] = {
            pos: int(data.get("timer", m.BOMB_TIMER))
            for pos, data in m.known_bombs.items()
        }
        # Min-propagate earlier triggers through blast adjacency to a fixpoint.
        changed = True
        while changed:
            changed = False
            for a_pos, a_tick in list(fire_tick.items()):
                blast_a = self.blast_cells(a_pos)
                for b_pos in fire_tick:
                    if b_pos == a_pos:
                        continue
                    if b_pos in blast_a and a_tick < fire_tick[b_pos]:
                        fire_tick[b_pos] = a_tick
                        changed = True
        layers: list[set[tuple[int, int]]] = [set() for _ in range(horizon + 1)]
        for pos, tick in fire_tick.items():
            if 0 <= tick <= horizon:
                layers[tick].update(self.blast_cells(pos))
        m._danger_layers_cache = layers
        return layers

    def on_fire_at(
        self,
        cell: tuple[int, int],
        tick: int,
        layers: list[set[tuple[int, int]]] | None = None,
    ) -> bool:
        """True if ``cell`` is on fire at relative tick ``tick``. Ticks beyond
        the horizon are treated as safe (the bomb resolves outside our window).
        """
        if layers is None:
            layers = self.danger_layers()
        if 0 <= tick < len(layers):
            return cell in layers[tick]
        return False

    def danger_cells(self) -> set[tuple[int, int]]:
        m = self._m
        if m.time_danger_enabled:
            # Chain-corrected near-window (t <= 2): same reaction horizon as
            # the legacy set, but an enemy bomb chained to fire within it is
            # now included even if its naive timer hid it. Horizon stays <=2
            # deliberately -- a larger flat avoidance set is the over-caution
            # that craters the brackets we already win.
            layers = self.danger_layers()
            danger: set[tuple[int, int]] = set()
            for t in range(0, min(2, m.danger_horizon) + 1):
                danger.update(layers[t])
            return danger
        danger = set()
        for bomb_pos, data in m.known_bombs.items():
            timer = int(data.get("timer", m.BOMB_TIMER))
            if timer > 2:
                continue
            danger.update(self.blast_cells(bomb_pos))
        return danger

    # ------------------------------------------------------------------
    # Placement vetoes
    # ------------------------------------------------------------------
    def hits_enemy_base(self, blast: set[tuple[int, int]]) -> bool:
        """True if any known enemy base lies in this bomb's blast."""
        return any(b in blast for b in self._m.enemy_bases)

    def own_base_vetoes(
        self,
        base: tuple[int, int] | None,
        blast: set[tuple[int, int]],
    ) -> bool:
        """True if our OWN base in ``blast`` should block placing a bomb. A bomb
        never damages its placer's own-team base (env: same-team defenders
        excluded, dynamics.py:695). Relaxed fully under AE_NO_SELF_DAMAGE, and
        under AE_BASEKILL_NOESCAPE only when the bomb also hits an enemy base.
        """
        m = self._m
        if m.no_self_damage:
            return False
        if m.basekill_noescape and self.hits_enemy_base(blast):
            return False
        return base is not None and base in blast

    def escape_required(
        self,
        blast: set[tuple[int, int]] | None = None,
    ) -> bool:
        """Whether a verified own-bomb escape is required to place. The placer
        takes zero self-damage (env-confirmed). Not required under
        AE_NO_SELF_DAMAGE (all bombs); under AE_BASEKILL_NOESCAPE only for a bomb
        whose blast contains an enemy base (speculative bombs still need escape).
        """
        m = self._m
        if m.no_self_damage:
            return False
        if (m.basekill_noescape and blast is not None
                and self.hits_enemy_base(blast)):
            return False
        return True

    # ------------------------------------------------------------------
    # Escape search
    # ------------------------------------------------------------------
    def safe_escape_within(
        self,
        location: tuple[int, int],
        blast: set[tuple[int, int]],
        max_moves: int,
        danger: set[tuple[int, int]] | None = None,
    ) -> tuple[int, int] | None:
        """Return the closest cell outside ``blast`` reachable in ≤ max_moves.

        The agent gets ``BOMB_TIMER`` movement actions between placing a bomb
        and the detonation phase, so anything beyond that is not actually safe.

        Under AE_TIME_DANGER, a step's destination is additionally rejected if
        it is on fire at the relative arrival tick (== BFS distance) per the
        chain-resolved danger layers, so the agent never "escapes" into an
        enemy bomb or chain that lights up exactly when it arrives.
        """
        m = self._m
        layers = self.danger_layers() if m.time_danger_enabled else None
        queue = deque([(location, 0)])
        seen = {location}
        while queue:
            pos, dist = queue.popleft()
            if dist > 0 and pos not in blast:
                return pos
            if dist >= max_moves:
                continue
            for nxt in m._neighbors(pos):
                if nxt in seen or nxt not in m.seen:
                    continue
                if danger is not None and nxt in danger:
                    continue
                if layers is not None and self.on_fire_at(nxt, dist + 1, layers):
                    continue
                seen.add(nxt)
                queue.append((nxt, dist + 1))
        return None

    def nearest_escape_cell(
        self,
        location: tuple[int, int],
        blast: set[tuple[int, int]],
    ) -> tuple[int, int] | None:
        m = self._m
        queue = deque([location])
        seen = {location}
        while queue:
            pos = queue.popleft()
            if pos not in blast and pos in m.seen:
                return pos
            for nxt in m._neighbors(pos):
                if nxt in seen or nxt not in m.seen:
                    continue
                seen.add(nxt)
                queue.append(nxt)
        return None

    def active_escape_path(
        self,
        location: tuple[int, int],
        step: int,
    ) -> list[tuple[int, int]] | None:
        m = self._m
        if m.escape_target is None or m.escape_until_step is None:
            return None
        if step > m.escape_until_step:
            m.escape_target = None
            m.escape_until_step = None
            return None

        forced_danger = set()
        for pos, data in m.known_bombs.items():
            # Never step on any bomb cell (own or enemy) because it is solid
            forced_danger.add(pos)
            if not data.get("own"):
                timer = int(data.get("timer", m.BOMB_TIMER))
                if timer <= 3:
                    forced_danger.update(self.blast_cells(pos))
        if location == m.escape_target and location not in forced_danger:
            m.escape_target = None
            m.escape_until_step = None
            return None

        path = m._bfs(location, m.escape_target, forced_danger)
        if path is not None:
            return path

        replacement = self.nearest_escape_cell(location, forced_danger)
        if replacement is None:
            return None
        m.escape_target = replacement
        return m._bfs(location, replacement, forced_danger)

    # ------------------------------------------------------------------
    # Bomb-commit bookkeeping
    # ------------------------------------------------------------------
    def commit_bomb(
        self,
        location: tuple[int, int],
        escape_target: tuple[int, int] | None,
    ) -> None:
        """Record this tick's own-bomb placement on the belief blackboard.

        Writes the synthetic ``known_bombs`` entry and the escape commitment,
        and snapshots the prior state in ``_tick_bomb_commit`` so a consultant
        wrapper that overrides the PLACE_BOMB can revert the phantom bomb.
        """
        m = self._m
        prior_bomb = m.known_bombs.get(location)
        prior_escape_target = m.escape_target
        prior_escape_until_step = m.escape_until_step
        m.known_bombs[location] = {
            "timer": m.BOMB_TIMER,
            "own": True,
            "last_step": m.last_step or 0,
        }
        m.escape_target = escape_target
        m.escape_until_step = (m.last_step or 0) + m.BOMB_TIMER
        m._tick_bomb_commit = {
            "cell": location,
            "prior_bomb": prior_bomb,
            "prior_escape_target": prior_escape_target,
            "prior_escape_until_step": prior_escape_until_step,
        }

    def revert_commit(self) -> bool:
        """Undo the synthetic own-bomb side effects from this tick's bomb commit.

        Called (via ``AEManager.revert_bomb_commit``) by
        ``ConfidencePolicyHybridAEManager`` when it overrides a heuristic
        ``PLACE_BOMB`` with a policy action: the bomb was never placed, so the
        ``known_bombs`` entry + escape state written by ``commit_bomb`` are
        phantom. Reverting them keeps ``danger_cells`` from routing around a
        bomb that does not exist. No-op (returns False) when no commit was
        recorded this tick.
        """
        m = self._m
        commit = m._tick_bomb_commit
        if commit is None:
            return False
        cell = commit["cell"]
        prior_bomb = commit["prior_bomb"]
        if prior_bomb is None:
            m.known_bombs.pop(cell, None)
        else:
            m.known_bombs[cell] = prior_bomb
        m.escape_target = commit["prior_escape_target"]
        m.escape_until_step = commit["prior_escape_until_step"]
        m._tick_bomb_commit = None
        return True
