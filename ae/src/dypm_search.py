"""dypm-style real-time tree search with rational-killer opponents.

STATUS (24 May 2026): NEGATIVE RESULT — not wired into the shipping path.
See ae/NOTES.md "dypm-veto-v1 experiment" section for the full write-up.
Brief summary:

  * Proposer mode (search picks actions): regressed every suite at n=8
    seed=42 (aggregate -0.096).
  * Veto mode (Skynet-style action filter): looked +0.044 at n=8 seed=42
    but failed the 24x3-seed gate at -0.015 aggregate (seed-42 cloudsuite
    -0.100).
  * Lethal-only veto (ignore base-damage hard fails): recovered to noise
    floor (-0.002 aggregate across 24x3); cloudsuite +0.008 mean but
    within cross-seed sigma. Not a clean win.

Conclusion: pessimistic search adds caution that costs the heuristic
offensive tempo more than it saves from hard fails. The NeurIPS 2018
Pommerman patterns assume a weaker baseline to layer onto than ours.
Code kept in tree for future work / next year's reference. To re-enable
for ad-hoc experimentation, re-wire AEManager.__init__ to construct
DypmSearch behind AE_DYPM=1 and wrap return paths with the veto.


Implements the NeurIPS 2018 Pommerman winner's core idea (Osogami et al.,
"Real-time tree search with pessimistic scenarios") adapted to the TIL-26 AE
environment:

    * Depth-limited minimax search over OUR actions.
    * Each known enemy is advanced under a deterministic "rational killer"
      policy: pick the action that maximises (damage to us) - 0.5 * (damage
      to self), with small heuristics for getting closer to us / our base /
      placing a useful bomb.
    * Bomb dynamics, blast resolution, item pickup, and reward accounting
      use the true reward table from the official AE wiki:

          mission +5, resource +2, recon +1
          damage dealt +1/dmg, attack kill +15
          destroy_enemy_base +50, own_base_destroyed -50, damage taken -1/dmg

    * Leaf value = accumulated true reward over the searched horizon.
      No stochastic rollout in v1; the per-node opponent model already
      injects adversarial pressure.

The search re-uses ``AEManager``'s belief state (``enemy_agents``,
``enemy_bases``, ``last_seen_items``, ``walls``, ``destructibles``) and the
existing ``_blast_cells`` / ``_edge_blocked`` helpers, so we never re-derive
bomb physics.

The module is import-safe even if AEManager is not yet constructed — no
top-level dependencies on manager state.

Enabled via ``AE_DYPM=1`` env flag in ``ae_manager.ae()``. When disabled
(default), the existing 0.638 shipping path is completely untouched.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import inf
import os
import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ae_manager import AEManager


# Reward constants — keep aligned with the official AE wiki + env yaml.
REW_MISSION = 5.0
REW_RESOURCE = 2.0
REW_RECON = 1.0
REW_DESTROY_BASE = 50.0
REW_OWN_BASE_LOST = -50.0
REW_KILL = 15.0
DMG_PER_BOMB_HIT = 20.0  # full-power blast tile damage
AGENT_MAX_HP = 100.0
BASE_MAX_HP = 200.0


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


@dataclass
class Enemy:
    """Pessimistic enemy model. We don't observe enemy facing; we treat
    enemies as omnidirectional movers with actions {N,S,E,W,STAY,BOMB}."""

    pos: tuple[int, int]
    hp: float = AGENT_MAX_HP


@dataclass
class DypmState:
    """Mutable-but-copyable search state.

    Bombs are (x, y, owner, timer) where owner is 'us' or 'them'. We don't
    distinguish between individual enemy owners — for reward credit we
    only need to know whether WE placed the bomb (so destroy_enemy_base
    rewards us) or some enemy placed it (so we lose own_base HP without
    crediting ourselves).
    """

    our_pos: tuple[int, int]
    our_dir: int
    our_hp: float
    our_base_pos: tuple[int, int] | None
    our_base_hp: float
    enemies: list[Enemy] = field(default_factory=list)
    enemy_bases: dict[tuple[int, int], float] = field(default_factory=dict)
    bombs: list[tuple[int, int, str, int]] = field(default_factory=list)
    bombs_left: int = 0
    items_consumed: set[tuple[int, int]] = field(default_factory=set)
    cumulative_reward: float = 0.0
    first_action: int | None = None

    def clone(self) -> "DypmState":
        return DypmState(
            our_pos=self.our_pos,
            our_dir=self.our_dir,
            our_hp=self.our_hp,
            our_base_pos=self.our_base_pos,
            our_base_hp=self.our_base_hp,
            enemies=[Enemy(e.pos, e.hp) for e in self.enemies],
            enemy_bases=dict(self.enemy_bases),
            bombs=list(self.bombs),
            bombs_left=self.bombs_left,
            items_consumed=set(self.items_consumed),
            cumulative_reward=self.cumulative_reward,
            first_action=self.first_action,
        )


# Action codes matching AEManager
FORWARD, BACKWARD, LEFT, RIGHT, STAY, PLACE_BOMB = 0, 1, 2, 3, 4, 5

DIR_RIGHT, DIR_DOWN, DIR_LEFT, DIR_UP = 0, 1, 2, 3
DIR_DELTAS = {
    DIR_RIGHT: (1, 0),
    DIR_DOWN: (0, 1),
    DIR_LEFT: (-1, 0),
    DIR_UP: (0, -1),
}
OPPOSITE = {DIR_RIGHT: DIR_LEFT, DIR_DOWN: DIR_UP, DIR_LEFT: DIR_RIGHT, DIR_UP: DIR_DOWN}
TURN_LEFT = {DIR_RIGHT: DIR_UP, DIR_UP: DIR_LEFT, DIR_LEFT: DIR_DOWN, DIR_DOWN: DIR_RIGHT}
TURN_RIGHT = {DIR_RIGHT: DIR_DOWN, DIR_DOWN: DIR_LEFT, DIR_LEFT: DIR_UP, DIR_UP: DIR_RIGHT}

# Cardinal enemy actions: (dx, dy) — used by the simplified omnidirectional
# enemy model. The 5th action (STAY) and 6th (BOMB) are special-cased.
ENEMY_MOVES = [(1, 0), (-1, 0), (0, 1), (0, -1), (0, 0)]


class DypmSearch:
    """Wrapper bound to a single AEManager instance.

    Build once per AEManager (the manager already lives for the full
    episode + survives /reset thanks to ae_server.py's re-instantiation).
    Call ``select_action(observation)`` each tick; returns either an int
    action that should be played, or ``None`` to defer to the existing
    heuristic.
    """

    def __init__(self, manager: "AEManager") -> None:
        self.mgr = manager
        self.depth = max(1, min(8, _env_int("AE_DYPM_DEPTH", 4)))
        self.time_budget_s = max(0.01, _env_float("AE_DYPM_BUDGET_MS", 200.0) / 1000.0)
        self.min_action_advantage = _env_float("AE_DYPM_MIN_ADV", 1.0)
        # "veto" (default) = Action Filter mode: heuristic proposes, dypm
        # vetos only on hard-fail outcomes (we die or our base is destroyed).
        # "proposer" = dypm picks the action itself (8-round Mac eval shows
        # this regresses every suite; kept for ablation only).
        mode_raw = os.environ.get("AE_DYPM_MODE", "veto").strip().lower()
        self.mode = mode_raw if mode_raw in ("veto", "proposer") else "veto"
        # Veto-mode-only knobs.
        self.veto_horizon = max(2, min(8, _env_int("AE_DYPM_VETO_HORIZON", 5)))
        # Veto fires on HARD fails by default. Set AE_DYPM_VETO_SOFT_HP=1
        # to also veto when projected HP drop > soft_hp_drop within horizon.
        self.veto_soft_hp = bool(_env_int("AE_DYPM_VETO_SOFT_HP", 0))
        self.soft_hp_drop = _env_float("AE_DYPM_SOFT_HP_DROP", 40.0)
        # If True, emit one log line per call (depth reached, nodes, ms,
        # chosen action, root value). Helpful for til test sanity checks.
        self.log_timing = bool(_env_int("AE_DYPM_LOG", 1))
        # If True, search is also active on non-fixed maps. The default
        # (False) keeps it Novice-only since the manager's belief state
        # (enemy_bases especially) is much more accurate there.
        self.run_on_unknown_maps = bool(_env_int("AE_DYPM_ALWAYS", 0))
        self._t_deadline = 0.0
        self._nodes_evaluated = 0
        # Counters surfaced by the validator for diagnostics.
        self.vetoes_fired = 0
        self.veto_calls = 0

    # ------------------------------------------------------------------
    # Public entry
    # ------------------------------------------------------------------
    def select_action(self, observation: dict) -> int | None:
        """Return the best action under dypm search, or ``None`` to defer."""

        mgr = self.mgr
        # Defer when we have no live enemy belief — search degenerates and
        # the existing planner is fine.
        if not mgr.enemy_agents and not mgr.enemy_bases:
            return None
        # Defer on unknown maps unless explicitly forced.
        is_fixed_novice = bool(getattr(mgr, "is_fixed_novice_map", False))
        if not is_fixed_novice and not self.run_on_unknown_maps:
            return None

        # Tactical-urgency gate. With depth=4 and BOMB_TIMER=3, the search
        # horizon only meaningfully reaches a payoff within ~4 Manhattan
        # steps. Outside that radius the existing item-routing planner has
        # better-tuned long-horizon priors. Only fire when SOMETHING
        # actionable (enemy adjacent, enemy base in bomb range, live bomb)
        # is within reach.
        loc = mgr._location(observation.get("location"))
        if loc is None:
            return None
        urgency_radius = self.depth + 2  # one ply slack on each side
        urgent = False
        if mgr.known_bombs:
            urgent = True
        else:
            for pos in mgr.enemy_agents:
                if abs(pos[0] - loc[0]) + abs(pos[1] - loc[1]) <= urgency_radius:
                    urgent = True
                    break
            if not urgent:
                for pos in mgr.enemy_bases:
                    # +bomb_radius because we want to fire when the base is
                    # within bombing reach, not just walking-adjacency.
                    if abs(pos[0] - loc[0]) + abs(pos[1] - loc[1]) <= urgency_radius + mgr.BOMB_RADIUS:
                        urgent = True
                        break
        if not urgent:
            return None

        state = self._build_initial_state(observation)
        if state is None:
            return None

        legal = self._legal_our_actions(state)
        if not legal:
            return None

        t0 = time.monotonic()
        self._t_deadline = t0 + self.time_budget_s
        self._nodes_evaluated = 0

        best_action: int | None = None
        best_value = -inf
        per_action_value: dict[int, float] = {}

        for action in legal:
            child = self._apply_full_tick(state, action)
            if child is None:
                per_action_value[action] = -inf
                continue
            child.first_action = action
            value = self._search(child, self.depth - 1)
            per_action_value[action] = value
            if value > best_value:
                best_value = value
                best_action = action
            if time.monotonic() > self._t_deadline:
                break

        elapsed_ms = (time.monotonic() - t0) * 1000.0
        if self.log_timing:
            print(
                f"dypm depth={self.depth} budget={self.time_budget_s*1000:.0f}ms "
                f"used={elapsed_ms:.1f}ms nodes={self._nodes_evaluated} "
                f"chose={best_action} value={best_value:+.2f} "
                f"per_action={ {k: round(v,2) for k,v in per_action_value.items()} }",
                flush=True,
            )

        if best_action is None:
            return None
        # Only return if the chosen action is meaningfully better than
        # doing nothing — otherwise defer to the planner. This guards
        # against the search spuriously preferring a bomb that scores
        # +0.3 over passing the tick to the (better-tuned) heuristic.
        stay_value = per_action_value.get(STAY, -inf)
        if best_value < stay_value + self.min_action_advantage and best_action != STAY:
            return None
        return best_action

    # ------------------------------------------------------------------
    # Veto: Skynet-style Action Filter
    # ------------------------------------------------------------------
    def veto(self, observation: dict, proposed_action: int) -> int:
        """Return either ``proposed_action`` (pass-through) or a safer
        legal alternative. Never returns None — this method is allowed
        to no-op but must always return an action.

        Hard-fail definition: simulating ``proposed_action`` then K-1
        more ticks of "passive-safe" play under rational-killer enemies
        causes either our agent's HP to drop to 0 or our base HP to drop
        to 0. When that happens, we scan all legal alternatives under
        the same sim and pick the highest-cumulative-reward survivor.
        If no alternative survives either, we pass through.
        """
        self.veto_calls += 1
        mgr = self.mgr

        # Same gates as proposer mode: need a fixed Novice map (unless
        # forced) and at least one known enemy or enemy base.
        is_fixed_novice = bool(getattr(mgr, "is_fixed_novice_map", False))
        if not is_fixed_novice and not self.run_on_unknown_maps:
            return proposed_action
        if not mgr.enemy_agents and not mgr.enemy_bases:
            return proposed_action

        # Only run when there's something actionable nearby — same
        # urgency gate as proposer mode, slightly tighter.
        loc = mgr._location(observation.get("location"))
        if loc is None:
            return proposed_action
        urgency_radius = self.veto_horizon + 2
        urgent = bool(mgr.known_bombs)
        if not urgent:
            for pos in mgr.enemy_agents:
                if abs(pos[0] - loc[0]) + abs(pos[1] - loc[1]) <= urgency_radius:
                    urgent = True
                    break
        if not urgent:
            for pos in mgr.enemy_bases:
                if abs(pos[0] - loc[0]) + abs(pos[1] - loc[1]) <= urgency_radius:
                    urgent = True
                    break
        if not urgent:
            return proposed_action

        state = self._build_initial_state(observation)
        if state is None:
            return proposed_action

        t0 = time.monotonic()
        self._t_deadline = t0 + self.time_budget_s

        proposed_outcome = self._simulate_action_passively(state, proposed_action)
        if proposed_outcome is None:
            return proposed_action  # action was illegal — let the planner deal
        if not self._is_hard_fail(state, proposed_outcome):
            return proposed_action  # safe enough; pass through

        # Proposed action is dangerous. Scan alternatives.
        legal = self._legal_our_actions(state)
        best_alt: int | None = None
        best_alt_reward = -inf
        for alt in legal:
            if alt == proposed_action:
                continue
            outcome = self._simulate_action_passively(state, alt)
            if outcome is None:
                continue
            if self._is_hard_fail(state, outcome):
                continue
            if outcome.cumulative_reward > best_alt_reward:
                best_alt_reward = outcome.cumulative_reward
                best_alt = alt
            if time.monotonic() > self._t_deadline:
                break

        elapsed_ms = (time.monotonic() - t0) * 1000.0
        if best_alt is None:
            # All alternatives are equally dangerous; nothing we can do.
            if self.log_timing:
                print(
                    f"dypm-veto NO-ALT ms={elapsed_ms:.1f} "
                    f"proposed={proposed_action} pass-through",
                    flush=True,
                )
            return proposed_action

        self.vetoes_fired += 1
        if self.log_timing:
            print(
                f"dypm-veto FIRED ms={elapsed_ms:.1f} "
                f"proposed={proposed_action} -> alt={best_alt} "
                f"alt_reward={best_alt_reward:+.2f} "
                f"(fired {self.vetoes_fired}/{self.veto_calls})",
                flush=True,
            )
        return best_alt

    def _simulate_action_passively(
        self,
        root_state: DypmState,
        action: int,
    ) -> DypmState | None:
        """Apply ``action`` then advance ``veto_horizon - 1`` more ticks
        with our agent playing 'safe-passive' (STAY when no immediate
        danger, else a legal cell away from the nearest blast cone).
        Returns the final state or None if the first action was illegal."""
        cur = self._apply_full_tick(root_state, action)
        if cur is None:
            return None
        for _ in range(self.veto_horizon - 1):
            if cur.our_hp <= 0 or cur.our_base_hp <= 0:
                break
            safe_action = self._pick_passive_action(cur)
            nxt = self._apply_full_tick(cur, safe_action)
            if nxt is None:
                # All-illegal fallback: STAY should always be legal.
                nxt = self._apply_full_tick(cur, STAY)
                if nxt is None:
                    break
            cur = nxt
        return cur

    def _pick_passive_action(self, state: DypmState) -> int:
        """Pick the lowest-danger action for the next tick — used inside
        veto rollouts so our future behavior isn't unrealistically
        helpless. We don't search here; just prefer cells outside any
        live blast cone."""
        # Identify dangerous cells (any cell in blast of a bomb with
        # timer<=2).
        danger: set[tuple[int, int]] = set()
        for bx, by, _owner, timer in state.bombs:
            if timer <= 2:
                danger.update(self.mgr._blast_cells((bx, by)))
        if state.our_pos not in danger:
            return STAY  # we're already safe, conserve tempo
        # We're in danger. Try each move; prefer one that lands us OUT
        # of the danger set.
        for action in (FORWARD, BACKWARD, LEFT, RIGHT):
            if action in (FORWARD, BACKWARD):
                move_dir = state.our_dir if action == FORWARD else OPPOSITE[state.our_dir]
                dx, dy = DIR_DELTAS[move_dir]
                nx, ny = state.our_pos[0] + dx, state.our_pos[1] + dy
                if not self.mgr._in_bounds((nx, ny)):
                    continue
                if self.mgr._edge_blocked(state.our_pos, move_dir):
                    continue
                if (nx, ny) not in self.mgr.seen:
                    continue
                if (nx, ny) not in danger:
                    return action
        return STAY

    def _is_hard_fail(self, root: DypmState, outcome: DypmState) -> bool:
        """True if the simulated outcome is a hard fail vs the root."""
        if outcome.our_hp <= 0:
            return True
        if outcome.our_base_hp <= 0 and root.our_base_hp > 0:
            return True
        if self.veto_soft_hp:
            hp_drop = root.our_hp - outcome.our_hp
            base_drop = root.our_base_hp - outcome.our_base_hp
            if hp_drop >= self.soft_hp_drop or base_drop >= self.soft_hp_drop:
                return True
        return False

    # ------------------------------------------------------------------
    # Search core
    # ------------------------------------------------------------------
    def _search(self, state: DypmState, depth_remaining: int) -> float:
        self._nodes_evaluated += 1
        if depth_remaining <= 0:
            return state.cumulative_reward
        if state.our_hp <= 0 or state.our_base_hp <= 0:
            # Game-ending events are already priced into cumulative_reward
            # via the detonation handler; bail early.
            return state.cumulative_reward
        if time.monotonic() > self._t_deadline:
            return state.cumulative_reward

        legal = self._legal_our_actions(state)
        if not legal:
            return state.cumulative_reward

        best = -inf
        for action in legal:
            child = self._apply_full_tick(state, action)
            if child is None:
                continue
            value = self._search(child, depth_remaining - 1)
            if value > best:
                best = value
        if best == -inf:
            return state.cumulative_reward
        return best

    # ------------------------------------------------------------------
    # Initial state from manager belief
    # ------------------------------------------------------------------
    def _build_initial_state(self, observation: dict) -> DypmState | None:
        mgr = self.mgr
        loc = mgr._location(observation.get("location"))
        if loc is None:
            return None
        direction = mgr._as_int(observation.get("direction"), default=DIR_RIGHT) % 4
        base = mgr._location(observation.get("base_location"))
        our_hp = float(mgr._as_int(observation.get("health"), default=int(mgr.health)))
        if isinstance(observation.get("health"), (list, tuple)) and observation["health"]:
            our_hp = float(observation["health"][0])
        base_hp_raw = observation.get("base_health")
        if isinstance(base_hp_raw, (list, tuple)) and base_hp_raw:
            our_base_hp = float(base_hp_raw[0])
        else:
            our_base_hp = float(mgr._as_int(base_hp_raw, default=int(mgr.base_health)))

        # Translate manager bombs (timer counts down each tick). The manager
        # doesn't track ownership, so we treat any bomb in known_bombs as
        # 'them' for reward credit — this is conservative (we won't claim
        # rewards we can't prove are ours). Newly placed bombs during the
        # search ARE marked 'us'.
        bombs: list[tuple[int, int, str, int]] = []
        for (bx, by), data in mgr.known_bombs.items():
            timer = max(1, int(data.get("timer", mgr.BOMB_TIMER)))
            # The manager tags bombs it placed itself with "own": True.
            # Treat those as ours for reward credit; everything else is
            # adversarial (could be allies in the team-of-2 setup, but for
            # damage-credit purposes that's still "not us").
            owner = "us" if data.get("own", False) else "them"
            bombs.append((bx, by, owner, timer))

        enemies = [Enemy(pos=pos) for pos in mgr.enemy_agents.keys()]
        enemy_bases = {pos: BASE_MAX_HP for pos in mgr.enemy_bases.keys()}

        return DypmState(
            our_pos=loc,
            our_dir=direction,
            our_hp=our_hp,
            our_base_pos=base,
            our_base_hp=our_base_hp,
            enemies=enemies,
            enemy_bases=enemy_bases,
            bombs=bombs,
            bombs_left=mgr._as_int(observation.get("team_bombs"), default=0),
        )

    # ------------------------------------------------------------------
    # Our agent: legal actions + transition
    # ------------------------------------------------------------------
    def _legal_our_actions(self, state: DypmState) -> list[int]:
        mgr = self.mgr
        out: list[int] = []
        for action in (FORWARD, BACKWARD, LEFT, RIGHT, STAY):
            if action in (FORWARD, BACKWARD):
                move_dir = state.our_dir if action == FORWARD else OPPOSITE[state.our_dir]
                dx, dy = DIR_DELTAS[move_dir]
                nx, ny = state.our_pos[0] + dx, state.our_pos[1] + dy
                if not mgr._in_bounds((nx, ny)):
                    continue
                if mgr._edge_blocked(state.our_pos, move_dir):
                    continue
                if (nx, ny) not in mgr.seen:
                    # Don't step into terra incognita during search.
                    continue
            out.append(action)
        if state.bombs_left > 0:
            # PLACE_BOMB only if our own base isn't in blast and we have
            # an escape — same rule as the existing planner.
            blast = mgr._blast_cells(state.our_pos)
            if state.our_base_pos is None or state.our_base_pos not in blast:
                if mgr._lookahead_escape(state.our_pos, blast, mgr.BOMB_TIMER) is not None:
                    out.append(PLACE_BOMB)
        return out

    def _apply_our_action(self, state: DypmState, action: int) -> DypmState | None:
        """Apply our action (pos/dir/bomb placement), without advancing
        the world tick. Returns a NEW state."""
        mgr = self.mgr
        nxt = state.clone()

        if action == FORWARD or action == BACKWARD:
            move_dir = state.our_dir if action == FORWARD else OPPOSITE[state.our_dir]
            dx, dy = DIR_DELTAS[move_dir]
            nx, ny = state.our_pos[0] + dx, state.our_pos[1] + dy
            if not mgr._in_bounds((nx, ny)) or mgr._edge_blocked(state.our_pos, move_dir):
                return None
            nxt.our_pos = (nx, ny)
        elif action == LEFT:
            nxt.our_dir = TURN_LEFT[state.our_dir]
        elif action == RIGHT:
            nxt.our_dir = TURN_RIGHT[state.our_dir]
        elif action == STAY:
            pass
        elif action == PLACE_BOMB:
            if state.bombs_left <= 0:
                return None
            if any((bx, by) == state.our_pos for bx, by, _o, _t in state.bombs):
                return None
            nxt.bombs.append((state.our_pos[0], state.our_pos[1], "us", mgr.BOMB_TIMER))
            nxt.bombs_left -= 1
        else:
            return None

        # Item pickup happens on the cell we ended on.
        item = mgr.last_seen_items.get(nxt.our_pos)
        if item is not None and nxt.our_pos not in nxt.items_consumed:
            kind = item[0]
            if kind == "mission":
                nxt.cumulative_reward += REW_MISSION
            elif kind == "resource":
                nxt.cumulative_reward += REW_RESOURCE
            elif kind == "recon":
                nxt.cumulative_reward += REW_RECON
            nxt.items_consumed.add(nxt.our_pos)

        return nxt

    # ------------------------------------------------------------------
    # Enemy: rational-killer policy (deterministic per node)
    # ------------------------------------------------------------------
    def _enemy_action(self, state: DypmState, enemy_idx: int) -> tuple[int, int]:
        """Choose move (dx, dy) for the given enemy under rational-killer.

        We do a 1-ply lookahead: for each candidate enemy move, score
        damage-to-us - 0.5 * damage-to-self based on bombs in flight, and
        prefer moves that close distance to us / our base if no bomb
        action dominates.
        """
        mgr = self.mgr
        enemy = state.enemies[enemy_idx]

        best_move = (0, 0)
        best_score = -inf

        for dx, dy in ENEMY_MOVES:
            nx, ny = enemy.pos[0] + dx, enemy.pos[1] + dy
            if not mgr._in_bounds((nx, ny)):
                continue
            # Block enemy moves through walls when we know about them.
            if (dx, dy) != (0, 0):
                # Use the manager's edge check — best-effort.
                move_dir = None
                for d, delta in DIR_DELTAS.items():
                    if delta == (dx, dy):
                        move_dir = d
                        break
                if move_dir is not None and mgr._edge_blocked(enemy.pos, move_dir):
                    continue
                if (nx, ny) not in mgr.seen:
                    # Don't let enemies teleport through unobserved cells either.
                    continue
            score = self._score_enemy_position(state, (nx, ny), enemy)
            if score > best_score:
                best_score = score
                best_move = (dx, dy)
        return best_move

    def _score_enemy_position(
        self,
        state: DypmState,
        pos: tuple[int, int],
        enemy: Enemy,
    ) -> float:
        """Heuristic value to the enemy of standing at ``pos`` next tick.

        Note on calibration: setting the proximity weights too high makes
        the search assume rational enemies path-track us perfectly, which
        makes US (the searcher) reflexively retreat and lose offensive
        tempo. The empirical sweet spot is *weak* proximity pressure —
        enemies prefer to stay near already-in-flight bomb blasts rather
        than chase, which matches cloudsuite rusher behavior reasonably."""
        score = 0.0
        aggressiveness = _env_float("AE_DYPM_AGGRESSIVENESS", 0.15)
        for bx, by, _owner, timer in state.bombs:
            if timer <= 1:
                blast = self.mgr._blast_cells((bx, by))
                if state.our_pos in blast:
                    score += DMG_PER_BOMB_HIT
                if pos in blast:
                    score -= 0.5 * DMG_PER_BOMB_HIT
        # Weak proximity pressure (tunable via env).
        score -= aggressiveness * (
            abs(pos[0] - state.our_pos[0]) + abs(pos[1] - state.our_pos[1])
        )
        if state.our_base_pos is not None:
            score -= 0.5 * aggressiveness * (
                abs(pos[0] - state.our_base_pos[0]) + abs(pos[1] - state.our_base_pos[1])
            )
        for bx, by, owner, timer in state.bombs:
            if owner == "us" and timer <= 2:
                if pos in self.mgr._blast_cells((bx, by)):
                    score -= 10.0
        return score

    # ------------------------------------------------------------------
    # Full tick = our action + enemy moves + bomb timer + detonations
    # ------------------------------------------------------------------
    def _apply_full_tick(self, state: DypmState, our_action: int) -> DypmState | None:
        nxt = self._apply_our_action(state, our_action)
        if nxt is None:
            return None

        # Advance each enemy under rational-killer.
        for i, enemy in enumerate(nxt.enemies):
            dx, dy = self._enemy_action(nxt, i)
            new_pos = (enemy.pos[0] + dx, enemy.pos[1] + dy)
            if self.mgr._in_bounds(new_pos):
                enemy.pos = new_pos

        # Bomb tick: decrement, then resolve any that hit zero.
        survivors: list[tuple[int, int, str, int]] = []
        to_detonate: list[tuple[int, int, str]] = []
        for bx, by, owner, timer in nxt.bombs:
            timer -= 1
            if timer <= 0:
                to_detonate.append((bx, by, owner))
            else:
                survivors.append((bx, by, owner, timer))
        nxt.bombs = survivors

        for bx, by, owner in to_detonate:
            self._resolve_detonation(nxt, (bx, by), owner)

        return nxt

    def _resolve_detonation(
        self,
        state: DypmState,
        bomb_pos: tuple[int, int],
        owner: str,
    ) -> None:
        """Apply blast effects: damage to us, enemies, bases. Update reward."""
        mgr = self.mgr
        blast = mgr._blast_cells(bomb_pos)

        # Damage to us
        if state.our_pos in blast:
            dmg = min(DMG_PER_BOMB_HIT, state.our_hp)
            state.our_hp -= dmg
            state.cumulative_reward -= dmg  # -1 per damage taken
            if state.our_hp <= 0:
                # Death: no extra penalty beyond the -100 already accrued.
                state.our_hp = 0

        # Damage to our base
        if state.our_base_pos is not None and state.our_base_pos in blast:
            dmg = min(DMG_PER_BOMB_HIT, state.our_base_hp)
            state.our_base_hp -= dmg
            state.cumulative_reward -= dmg  # -1 per base damage
            if state.our_base_hp <= 0:
                state.our_base_hp = 0
                state.cumulative_reward += REW_OWN_BASE_LOST

        # Damage to enemies — only credited if WE placed the bomb.
        for enemy in state.enemies:
            if enemy.pos in blast and enemy.hp > 0:
                dmg = min(DMG_PER_BOMB_HIT, enemy.hp)
                enemy.hp -= dmg
                if owner == "us":
                    state.cumulative_reward += dmg  # +1 per damage dealt
                    if enemy.hp <= 0:
                        state.cumulative_reward += REW_KILL
                        enemy.hp = 0

        # Damage to enemy bases — only credited if WE placed it.
        for base_pos in list(state.enemy_bases.keys()):
            if base_pos in blast:
                hp = state.enemy_bases[base_pos]
                dmg = min(DMG_PER_BOMB_HIT, hp)
                state.enemy_bases[base_pos] = hp - dmg
                if owner == "us":
                    state.cumulative_reward += dmg
                    if state.enemy_bases[base_pos] <= 0:
                        state.cumulative_reward += REW_DESTROY_BASE
                        del state.enemy_bases[base_pos]
