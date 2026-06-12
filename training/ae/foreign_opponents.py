"""Foreign (non-mirror) opponent pool for the AE Semifinals melee eval.

Why this exists
---------------
Every opponent in ``opponents.py`` subclasses our own ``AEManager`` and inherits
the *exact* tuned priority scorer — so scoring our model against them scores it
against mirror images of itself. This module adds opponents built on genuinely
different decision principles, exposed through the existing ``OpponentFn``
interface (``__call__(obs: dict) -> int`` with an optional ``reset_for_game()``)
so ``simulate.py``/``opponents.py`` consume them unchanged:

  * ``curry_aggro`` / ``curry_fortress`` — the vendored competitor heuristic
    (Team A): external A* goal-portfolio + forward-sim plan
    scoring. Two weight-based personas pinned per-instance.
  * ``self_policy``   — our Pandemonium raw CNN-PPO checkpoint in *full control*
    (raw 6-action argmax, NOT the confpol consultant gate).
  * ``self_tactical`` — our learned 12-way tactical-macro selector + planner
    executor (a second, architecturally-distinct learned lineage).
  * ``self_heuristic``— our shipped C+bomb7 ``AEManager`` (the deliberate strong
    mirror baseline, kept for calibration).
  * ``evbot``         — naive nominal-game-reward EV maximizer (the *opposite*
    calibration from our scorer, which under-weights bases ~8x).
  * ``aggressive_proxy`` — hyper-aggressive kill+base focus, no farming
    discipline, over-commits into danger.
  * ``anti_aggro_exploiter`` — farm + survive + punish aggressors near our base.

The purpose-built bots (``evbot`` / ``aggressive_proxy`` / ``anti_aggro_exploiter``)
DO reuse ``AEManager``'s low-level infrastructure (belief map, BFS/Dijkstra,
blast-cell calc, bomb-escape safety) but each fully *replaces* the
target/objective selection with its own scoring loop — that different objective
function, not the utility reuse, is what makes them non-mirror.

Train / eval split (enforce in code, never by convention)
---------------------------------------------------------
The single most important guard against proxy-overfit: some opponents are
EVAL-ONLY and are NEVER trained against (see ``train_ppo.py``). Lift on the
trained opponents that fails to appear on the held-out opponents is the local
early-warning that it won't transfer to the cloud.
"""

from __future__ import annotations

import os
import sys
from math import inf
from pathlib import Path

# Curry: Python-only (no numba/parallel goals); EXPERIMENTAL_VARIANT left unset so
# C.VARIANT stays "balanced" — personas are pinned via set_persona instead.
os.environ.setdefault("EXP_DISABLE_NUMBA", "1")
os.environ.setdefault("USE_PARALLEL_GOALS", "0")

THIS_DIR = Path(__file__).resolve().parent           # training/ae
REPO_ROOT = THIS_DIR.parents[1]
AE_SRC = REPO_ROOT / "ae" / "src"
TIL_AE = REPO_ROOT / "til-26-ae"
CURRY_PATH = THIS_DIR / "foreign" / "curry"           # gitignored vendored snapshot
PEROXIDE_PATH = THIS_DIR / "foreign" / "peroxide"     # gitignored vendored snapshot
CHECKPOINTS = THIS_DIR / "checkpoints"

for _p in (str(AE_SRC), str(THIS_DIR), str(TIL_AE), str(CURRY_PATH), str(PEROXIDE_PATH)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ae_manager import AEManager  # noqa: E402


# Train / eval opponent split — enforced by train_ppo.py's sampler.
FOREIGN_TRAIN_OK = ["curry_aggro", "self_policy", "self_heuristic", "evbot"]
FOREIGN_EVAL_ONLY = ["curry_fortress", "self_tactical",
                     "aggressive_proxy", "anti_aggro_exploiter",
                     "peroxide_astar"]
FOREIGN_NAMES = tuple(FOREIGN_TRAIN_OK + FOREIGN_EVAL_ONLY)


# Checkpoint defaults (env-overridable). Local training layout, not container.
def _self_policy_checkpoint() -> Path:
    override = os.environ.get("AE_SELF_POLICY_CHECKPOINT")
    if override:
        return Path(override)
    return CHECKPOINTS / "pandemonium-v1-best-u860.pt"


def _self_tactical_checkpoint() -> Path:
    override = os.environ.get("AE_SELF_TACTICAL_CHECKPOINT")
    if override:
        return Path(override)
    for cand in (CHECKPOINTS / "tactical_policy.pt", AE_SRC.parent / "models" / "tactical_policy.pt"):
        if cand.exists():
            return cand
    return CHECKPOINTS / "tactical_policy.pt"


# The shipped C+bomb7 heuristic profile (ae/Dockerfile). AEManager reads these
# at construction, so self_heuristic must be built with them set to be faithful.
CBOMB7_ENV = {
    "AE_ITEM_MISSION_VALUE": "80",
    "AE_ITEM_RESOURCE_VALUE": "40",
    "AE_ENEMY_BASE_VALUE": "100",
    "AE_TIER1_DEFENSE": "0",
    "AE_TIER1_REPEAT_KILL": "1",
    "AE_TIER1_SHARED_CREDIT": "0",
    "AE_TIER1_NO_STAY_PENALTY": "1",
    "AE_TIER1_PREDICTIVE_WALK": "1",
    "AE_USE_PLAYBOOK": "0",
    "AE_USE_OPPONENT_MODEL": "0",
    "AE_DIJKSTRA_BOMB_COST": "7.0",
    "AE_ASTAR_TIEBREAK": "0",
    "AE_LEAD_BASE_TETHER": "1",
    "AE_LEAD_TETHER_HEALTH": "60.0",
    "AE_LEAD_TETHER_WEIGHT": "0.5",
    "AE_LEAD_BOMB_GATE_BASE": "0",
    "AE_LEAD_RECON_DISCOUNT": "0",
}


class _EnvOverride:
    """Temporarily set env vars for the duration of a `with` block."""

    def __init__(self, overrides: dict[str, str]) -> None:
        self._overrides = overrides
        self._saved: dict[str, str | None] = {}

    def __enter__(self):
        for key, value in self._overrides.items():
            self._saved[key] = os.environ.get(key)
            os.environ[key] = value
        return self

    def __exit__(self, *exc):
        for key, old in self._saved.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old
        return False


# Purpose-built non-mirror bots: subclass AEManager for infrastructure (belief /
# BFS / blast / escape) but override _choose_target with a different objective.
# Bomb decision and movement stay AEManager's.

def _strip_learned_artifacts(m: AEManager) -> None:
    """Keep the bot a pure standalone policy: no playbook / opponent model."""
    m.playbook = None
    m.opponent_walk_scale = 1.0


def _distance_map(m: AEManager, start, danger):
    """Mirror AEManager's own choice: exact Dijkstra on the fixed novice map,
    BFS otherwise. Returns (distance, parent)."""
    if getattr(m, "is_fixed_novice_map", False):
        return m._dijkstra_distance_map(start, danger)
    return m._bfs_distance_map(start, danger)


def _fresh_enemies(m: AEManager, staleness: int) -> list[tuple[int, int]]:
    step = m.last_step if m.last_step is not None else 0
    return [pos for pos, seen in m.enemy_agents.items()
            if step - int(seen) <= staleness]


class EVBot(AEManager):
    """Naive nominal-game-reward EV maximizer.

    Objective = the raw game reward of the nearest-ish visible target with only
    a light distance discount. Bases (+50) dominate missions (+5) — the exact
    opposite of our calibrated scorer, which deliberately under-weights bases
    ~8x because their *realized* EV is far below nominal. Proxy for a
    reward-calibrated / over-aggressive team that chases nominal points.
    """

    name = "evbot"
    NOMINAL = {"mission": 5.0, "resource": 2.0, "recon": 1.0}
    BASE_REWARD = 50.0
    KILL_REWARD = 30.0
    DIST_COST = 0.3

    def __init__(self) -> None:
        super().__init__()
        _strip_learned_artifacts(self)

    def __call__(self, obs: dict) -> int:
        return self.ae(obs)

    def _choose_target(self, start, danger, low_health=False, direction=None):
        distance, parent = _distance_map(self, start, danger)
        best, best_score, best_kind = None, -inf, "none"
        candidates: list[tuple[float, tuple[int, int], str]] = []
        if not low_health:
            for pos in self.enemy_bases:
                candidates.append((self.BASE_REWARD, pos, "ev_base"))
            for pos in _fresh_enemies(self, max(1, self.ENEMY_STALENESS)):
                candidates.append((self.KILL_REWARD, pos, "ev_kill"))
        for pos, (kind, _step) in self.last_seen_items.items():
            candidates.append((self.NOMINAL.get(kind, 1.0), pos, f"ev_item_{kind}"))
        for value, pos, kind in candidates:
            if pos == start or pos not in distance:
                continue
            score = value - self.DIST_COST * distance[pos]
            if score > best_score:
                best_score, best, best_kind = score, pos, kind
        if best is None:
            # Nothing valuable in view -> fall back to exploration only.
            return super()._choose_target(start, danger, low_health, direction=direction)
        self.last_target_kind = best_kind
        path = self._reconstruct_path(parent, start, best)
        self.current_path = path
        return best, path


class AggressiveProxy(AEManager):
    """Hyper-aggressive kill+base bot with no farming discipline.

    Always commits to the nearest enemy base or visible enemy, ignores items
    entirely, applies no path-threat penalty (it over-commits into danger), and
    bombs readily. Proxy for the org's "aggressive cloud model". EVAL-ONLY.
    """

    name = "aggressive_proxy"
    BASE_VALUE = 100.0
    ENEMY_VALUE = 80.0
    DIST_COST = 0.2

    def __init__(self) -> None:
        super().__init__()
        _strip_learned_artifacts(self)
        # Low threat penalties -> walks into threats / over-commits.
        self.PATH_THREAT_PENALTY = 0.25
        self.CELL_THREAT_PENALTY = 1.0

    def __call__(self, obs: dict) -> int:
        return self.ae(obs)

    def _choose_target(self, start, danger, low_health=False, direction=None):
        distance, parent = _distance_map(self, start, danger)
        best, best_score, best_kind = None, -inf, "none"
        candidates: list[tuple[float, tuple[int, int], str]] = []
        for pos in self.enemy_bases:
            candidates.append((self.BASE_VALUE, pos, "aggro_base"))
        for pos in _fresh_enemies(self, max(2, self.ENEMY_STALENESS)):
            candidates.append((self.ENEMY_VALUE, pos, "aggro_kill"))
        for value, pos, kind in candidates:
            if pos == start or pos not in distance:
                continue
            score = value - self.DIST_COST * distance[pos]   # no threat penalty
            if score > best_score:
                best_score, best, best_kind = score, pos, kind
        if best is None:
            return super()._choose_target(start, danger, low_health, direction=direction)
        self.last_target_kind = best_kind
        path = self._reconstruct_path(parent, start, best)
        self.current_path = path
        return best, path


class AntiAggroExploiter(AEManager):
    """Farm + survive + punish aggressors. The strategy we suspect *wins* the
    melee: maximize own farm rate while keeping distance from aggressors and
    intercepting enemies that approach our base. EVAL-ONLY.
    """

    name = "anti_aggro_exploiter"
    ITEM_VALUE = {"mission": 80.0, "resource": 40.0, "recon": 8.0}
    DIST_COST = 1.0
    ENEMY_AVOID_RADIUS = 4
    ENEMY_AVOID_WEIGHT = 9.0
    INTERCEPT_VALUE = 120.0

    def __init__(self) -> None:
        super().__init__()
        _strip_learned_artifacts(self)
        # High threat penalties -> survives by avoiding danger.
        self.PATH_THREAT_PENALTY = 6.0
        self.CELL_THREAT_PENALTY = 14.0

    def __call__(self, obs: dict) -> int:
        return self.ae(obs)

    def _choose_target(self, start, danger, low_health=False, direction=None):
        distance, parent = _distance_map(self, start, danger)
        fresh = _fresh_enemies(self, max(4, self.ENEMY_STALENESS))

        # An aggressor inside our base radius overrides the farming objective.
        if self.base_location is not None:
            near_base = [p for p in fresh
                         if self._manhattan(p, self.base_location) <= max(6, self.BASE_DEFENSE_RADIUS + 2)]
            if near_base and not low_health:
                target = min(near_base, key=lambda p: self._manhattan(start, p))
                if target in distance:
                    self.last_target_kind = "punish_aggressor"
                    path = self._reconstruct_path(parent, start, target)
                    self.current_path = path
                    return target, path

        # Otherwise farm items, penalized by proximity to aggressors.
        best, best_score, best_kind = None, -inf, "none"
        for pos, (kind, _step) in self.last_seen_items.items():
            if pos == start or pos not in distance:
                continue
            value = self.ITEM_VALUE.get(kind, 1.0)
            score = value - self.DIST_COST * distance[pos]
            if fresh:
                ed = min(self._manhattan(pos, e) for e in fresh)
                if ed < self.ENEMY_AVOID_RADIUS:
                    score -= self.ENEMY_AVOID_WEIGHT * (self.ENEMY_AVOID_RADIUS - ed)
            if score > best_score:
                best_score, best, best_kind = score, pos, f"farm_{kind}"
        if best is None:
            return super()._choose_target(start, danger, low_health, direction=direction)
        self.last_target_kind = best_kind
        path = self._reconstruct_path(parent, start, best)
        self.current_path = path
        return best, path


# --------------------------------------------------------------------------
class CurryOpponent:
    """Adapter around the vendored ``ExperimentalHeuristicAgent``.

    Persona is pinned per-instance via ``dynamic_persona=False`` +
    ``set_persona`` (weight-table swap) so ``curry_aggro`` and ``curry_fortress``
    can coexist in the same process — the env-var ``EXPERIMENTAL_VARIANT`` path
    is process-global and is deliberately left unset (C.VARIANT == "balanced",
    hunter/variant-gated behaviors off). Faithful for weight-based personas
    (aggro_rush / fortress); NOT for C.VARIANT-gated variants.
    """

    def __init__(self, persona: str, seed: int | None = None) -> None:
        self._persona = persona
        self._seed = seed
        self._agent = self._new_agent()

    def _new_agent(self):
        import numpy as np
        from experimental_heuristic.agent import ExperimentalHeuristicAgent
        rng = np.random.default_rng(self._seed) if self._seed is not None else None
        agent = ExperimentalHeuristicAgent(rng=rng)
        agent.dynamic_persona = False     # pin: no runtime persona FSM
        agent.set_persona(self._persona)  # weight-based persona swap
        return agent

    def reset_for_game(self) -> None:
        self._agent = self._new_agent()

    def __call__(self, obs: dict) -> int:
        # act() self-detects round boundaries via step==0 and handles
        # frozen_ticks + action masking internally.
        return int(self._agent.act(obs))


class PeroxideOpponent:
    """Adapter around the vendored Team-B planner: an orientation-aware
    A* over (x,y,facing) state with partial-map memory, time-layered danger
    sets, base-siege with bomb-commitment tracking, and a base-anchor inference
    trick (rotate own base around grid-center by pi/3 to guess the other spawns).

    A genuinely DIFFERENT decision architecture from ours (greedy priority +
    Dijkstra) and from curry (forward-sim plan scoring) — the only SECOND
    foreign architecture we have. **WEAK competitor: qualifier 0.443, did NOT
    reach Semifinals.** Kept EVAL_ONLY as (a) architecture diversity for the
    held-out robustness probe and (b) a realistic mid/low-strength field member
    (we are seeded 15th; the real bracket has weaker teams too). Do NOT read
    beating peroxide as evidence of strength. Self-contained (numpy only); the
    env delivers an already-unpacked (7,5,25) viewcone, which its decoder wants.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._agent = self._new_agent()

    def _new_agent(self):
        from peroxide_planner import AEManager as _PeroxideAE
        return _PeroxideAE()

    def reset_for_game(self) -> None:
        self._agent.reset()

    def __call__(self, obs: dict) -> int:
        return int(self._agent.ae(obs))


def _load_isolated_policy(checkpoint: Path):
    """Construct a PolicyAEManager bound to ``checkpoint`` WITHOUT polluting the
    process-global model cache. policy_manager caches the first model loaded
    keyed on nothing; if the candidate (slot 0) is also a raw policy at a
    different checkpoint, the cache would alias them. We clear+restore the cache
    around the load so each gets its own model."""
    import policy_manager as pm
    saved = (pm._MODEL_CACHE, pm._DEVICE_CACHE, pm._N_FRAMES_CACHE, pm._USE_BELIEF_CACHE)
    old_env = os.environ.get("AE_POLICY_CHECKPOINT")
    pm._MODEL_CACHE = None
    os.environ["AE_POLICY_CHECKPOINT"] = str(checkpoint)
    try:
        mgr = pm.PolicyAEManager()
    finally:
        (pm._MODEL_CACHE, pm._DEVICE_CACHE,
         pm._N_FRAMES_CACHE, pm._USE_BELIEF_CACHE) = saved
        if old_env is None:
            os.environ.pop("AE_POLICY_CHECKPOINT", None)
        else:
            os.environ["AE_POLICY_CHECKPOINT"] = old_env
    return mgr


class SelfPolicyOpponent:
    """Our raw CNN-PPO policy in FULL control (argmax over 6 actions) — not the
    confpol consultant gate. We want the opponent to *be* the learned policy."""

    def __init__(self, checkpoint: Path | str | None = None, seed: int | None = None) -> None:
        self._checkpoint = Path(checkpoint) if checkpoint else _self_policy_checkpoint()
        self._mgr = _load_isolated_policy(self._checkpoint)

    def reset_for_game(self) -> None:
        self._mgr.stacker.reset()
        self._mgr.belief_manager = AEManager()
        self._mgr._last_step = None

    def __call__(self, obs: dict) -> int:
        return int(self._mgr.ae(obs))


class SelfTacticalOpponent:
    """Our learned 12-way tactical-macro selector + planner executor. A second,
    architecturally-distinct learned lineage from the raw policy. (The checkpoint
    is action_dim=12, so it cannot run as a raw 6-action argmax — it needs the
    planner executor, i.e. its deployed TacticalHybridAEManager form.)"""

    def __init__(self, checkpoint: Path | str | None = None, seed: int | None = None) -> None:
        self._checkpoint = Path(checkpoint) if checkpoint else _self_tactical_checkpoint()
        self._mgr = self._new()

    def _new(self):
        # AE_CONTENTION is OUR deployed agent's flag; an opponent proxy must never
        # inherit it (would confound a contention A/B). Forced off here so it
        # holds across reset_for_game() rebuilds.
        with _EnvOverride({"AE_TACTICAL_POLICY_CHECKPOINT": str(self._checkpoint),
                           "AE_CONTENTION": "0", "AE_STUN_TAX": "0",
                           "AE_FORTRESS": "0"}):
            from tactical_hybrid_manager import TacticalHybridAEManager
            return TacticalHybridAEManager()

    def reset_for_game(self) -> None:
        self._mgr = self._new()

    def __call__(self, obs: dict) -> int:
        return int(self._mgr.ae(obs))


class SelfHeuristicOpponent:
    """Our shipped C+bomb7 ``AEManager`` — the deliberate strong mirror baseline,
    kept for calibration against the genuinely-foreign opponents."""

    def __init__(self, seed: int | None = None) -> None:
        self._mgr = self._new()

    def _new(self) -> AEManager:
        # See SelfTacticalOpponent._new: never inherit OUR AE_CONTENTION flag.
        with _EnvOverride({**CBOMB7_ENV, "AE_CONTENTION": "0",
                           "AE_STUN_TAX": "0", "AE_FORTRESS": "0"}):
            return AEManager()

    def reset_for_game(self) -> None:
        self._mgr._reset_memory()

    def __call__(self, obs: dict) -> int:
        return int(self._mgr.ae(obs))


def make_foreign_opponent(name: str, seed: int | None = None):
    """Construct one foreign opponent by name. ``seed`` is used by the stochastic
    / RNG-seeded constructions (curry, self_policy); the heuristic/tactical bots
    are deterministic given the fixed novice map."""
    key = name.lower().strip()
    if key == "curry_aggro":
        return CurryOpponent("aggro_rush", seed=seed)
    if key == "curry_fortress":
        return CurryOpponent("fortress", seed=seed)
    if key == "self_policy":
        return SelfPolicyOpponent(seed=seed)
    if key == "self_tactical":
        return SelfTacticalOpponent(seed=seed)
    if key == "self_heuristic":
        return SelfHeuristicOpponent(seed=seed)
    if key == "peroxide_astar":
        return PeroxideOpponent(seed=seed)
    if key == "evbot":
        return EVBot()
    if key == "aggressive_proxy":
        return AggressiveProxy()
    if key == "anti_aggro_exploiter":
        return AntiAggroExploiter()
    raise ValueError(f"unknown foreign opponent name: {name!r}")
