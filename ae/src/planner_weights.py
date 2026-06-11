"""PlannerWeights — the planner's tunable surface as one typed module.

Every CEM-tunable weight and feature flag the heuristic planner reads from
the environment is declared here, with its type, default and clamp, instead
of being scattered through ``AEManager.__init__`` as ~50 anonymous
``os.environ.get`` calls. The env-var vocabulary (``AE_*``) is unchanged:
``PlannerWeights.from_env()`` reads exactly the same variables with exactly
the same defaults and clamping as the historical inline reads, so deploy
ENV overrides (Dockerfile g02 weights) and the CEM tuner's env-based sweeps
behave identically.

Construction paths:

- ``PlannerWeights.from_env()``  — production / deploy (env wins, as before).
- ``PlannerWeights(field=...)``  — tests and tuners can build a profile
  directly without touching ``os.environ``.

``AEManager.__init__`` accepts an optional ``weights=`` argument and copies
the fields onto its historical attribute names, so all downstream planner
code and the existing test suite are unchanged.

Not covered here (deliberately): per-reset reads (``AE_DIJKSTRA_BOMB_COST``,
``AE_ASTAR_TIEBREAK`` in ``_reset_memory``), the opponent-model path, and
non-planner consumers (wrapper gates, encoder constants).
"""

from __future__ import annotations

from dataclasses import dataclass, fields
import os


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


@dataclass
class PlannerWeights:
    """Typed planner profile. Field order mirrors AEManager.__init__ history."""

    # ── MCTS / tactical lookahead ─────────────────────────────────────
    mcts_enabled: bool = False                      # AE_MCTS
    mcts_depth: int = 3                             # AE_MCTS_DEPTH, clamped [1, 8]
    mcts_width: int = 24                            # AE_MCTS_WIDTH, clamped [12, 512]
    mcts_min_score: float = 12.0                    # AE_MCTS_MIN_SCORE
    mcts_time_budget_s: float = 0.080               # AE_MCTS_BUDGET_MS / 1000, floor 5ms
    mcts_log_timing: bool = True                    # AE_MCTS_LOG_TIMING

    # ── Tier-1 toggles (17 May bisect: #3/#6/#7 ON, #2/#4 OFF) ────────
    tier1_defense_priority: bool = False            # AE_TIER1_DEFENSE
    tier1_repeat_kill: bool = True                  # AE_TIER1_REPEAT_KILL
    tier1_shared_credit: bool = False               # AE_TIER1_SHARED_CREDIT
    tier1_no_stay_penalty: bool = True              # AE_TIER1_NO_STAY_PENALTY
    tier1_predictive_walk: bool = True              # AE_TIER1_PREDICTIVE_WALK

    # ── Bomb timing ───────────────────────────────────────────────────
    bomb_detonate_steps: int = 5                    # AE_BOMB_DETONATE_STEPS, floor 1

    # ── Base defense / target valuation ──────────────────────────────
    base_defense_health: float = 60.0               # AE_BASE_DEFENSE_HEALTH
    base_defense_radius: int = 4                    # AE_BASE_DEFENSE_RADIUS
    enemy_base_value: float = 80.0                  # AE_ENEMY_BASE_VALUE
    dist_penalty: float = 1.15                      # AE_DIST_PENALTY
    path_threat_penalty: float = 2.0                # AE_PATH_THREAT_PENALTY
    enemy_chase_value: float = 0.0                  # AE_ENEMY_CHASE_VALUE
    enemy_chase_radius: int = 4                     # AE_ENEMY_CHASE_RADIUS

    # ── Forward-sim plan re-score (default OFF) ───────────────────────
    plan_rescore_enabled: bool = False              # AE_PLAN_RESCORE
    plan_rescore_k: int = 4                         # AE_PLAN_RESCORE_K, floor 1
    plan_rescore_margin: float = 1.0                # AE_PLAN_RESCORE_MARGIN
    plan_rescore_time_tax: float = 0.35             # AE_PLAN_RESCORE_TIME_TAX
    plan_rescore_base_w: float = 1.0                # AE_PLAN_RESCORE_BASE_W
    plan_rescore_demote_only: bool = True           # AE_PLAN_RESCORE_DEMOTE_ONLY

    # ── Contention-aware item valuation (shipped ON via deploy ENV) ───
    contention_enabled: bool = False                # AE_CONTENTION
    contention_scale: float = 2.5                   # AE_CONTENTION_SCALE, floor 0.1
    contention_pfloor: float = 0.15                 # AE_CONTENTION_PFLOOR, clamped [0, 1]
    contention_topen: int = 40                      # AE_CONTENTION_TOPEN
    contention_tfresh: int = 3                      # AE_CONTENTION_TFRESH

    # ── Time-layered danger map (default OFF == byte-identical) ───────
    time_danger_enabled: bool = False               # AE_TIME_DANGER
    danger_horizon: int = 6                         # AE_DANGER_HORIZON; AEManager floors at BOMB_TIMER

    # ── No-self-damage bomb gating (default OFF; see specs) ───────────
    no_self_damage: bool = False                    # AE_NO_SELF_DAMAGE
    basekill_noescape: bool = False                 # AE_BASEKILL_NOESCAPE

    # ── Farming-race model: stun tax + fortress posture ───────────────
    stun_tax_enabled: bool = False                  # AE_STUN_TAX
    stun_tax_mult: float = 2.0                      # AE_STUN_TAX_MULT, floor 0
    fortress_enabled: bool = False                  # AE_FORTRESS
    fortress_phase: float = 0.6                     # AE_FORTRESS_PHASE
    fortress_base_mult: float = 0.5                 # AE_FORTRESS_BASE_MULT
    fortress_defense_mult: float = 1.5              # AE_FORTRESS_DEFENSE_MULT
    fortress_threat_mult: float = 1.5               # AE_FORTRESS_THREAT_MULT
    fortress_tether_w: float = 0.3                  # AE_FORTRESS_TETHER_W

    # ── Item values (g02 deploy overrides these via ENV) ──────────────
    item_mission_value: float = 50.0                # AE_ITEM_MISSION_VALUE
    item_resource_value: float = 25.0               # AE_ITEM_RESOURCE_VALUE
    item_recon_value: float = 10.0                  # AE_ITEM_RECON_VALUE

    # ── W2.1 leads ─────────────────────────────────────────────────────
    first_target_table_enabled: bool = False        # AE_FIRST_TARGET_TABLE (int > 0)
    first_target_boost: float = 60.0                # AE_FIRST_TARGET_BOOST
    first_target_decay: float = 0.55                # AE_FIRST_TARGET_DECAY
    enemy_bomb_escape_enabled: bool = False         # AE_ENEMY_BOMB_OVERRIDE (int > 0)
    enemy_bomb_escape_turn_penalty: float = 0.2     # AE_ENEMY_BOMB_ESCAPE_TURN_PENALTY
    enemy_bomb_escape_visit_penalty: float = 0.3    # AE_ENEMY_BOMB_ESCAPE_VISIT_PENALTY
    orientation_aware_path_enabled: bool = False    # AE_ORIENTATION_AWARE_PATH (int > 0)
    orientation_aware_turn_cost: float = 1.0        # AE_ORIENTATION_AWARE_TURN_COST

    # ── Rationale-mining leads (29 May; each default OFF) ─────────────
    lead_bomb_gate_base: bool = False               # AE_LEAD_BOMB_GATE_BASE (int > 0)
    lead_base_tether: bool = False                  # AE_LEAD_BASE_TETHER (int > 0)
    lead_tether_health: float = 60.0                # AE_LEAD_TETHER_HEALTH
    lead_tether_weight: float = 0.5                 # AE_LEAD_TETHER_WEIGHT
    lead_recon_discount: bool = False               # AE_LEAD_RECON_DISCOUNT (int > 0)
    lead_recon_dist_mult: float = 1.0               # AE_LEAD_RECON_DIST_MULT

    def __post_init__(self) -> None:
        # Clamps match the historical inline reads exactly, applied on every
        # construction path so a direct profile is as safe as an env one.
        self.mcts_depth = max(1, min(8, int(self.mcts_depth)))
        self.mcts_width = max(12, min(512, int(self.mcts_width)))
        self.mcts_time_budget_s = max(0.005, float(self.mcts_time_budget_s))
        self.bomb_detonate_steps = max(1, int(self.bomb_detonate_steps))
        self.plan_rescore_k = max(1, int(self.plan_rescore_k))
        self.contention_scale = max(0.1, float(self.contention_scale))
        self.contention_pfloor = min(1.0, max(0.0, float(self.contention_pfloor)))
        self.stun_tax_mult = max(0.0, float(self.stun_tax_mult))

    @classmethod
    def from_env(cls) -> "PlannerWeights":
        """Read the full profile from the historical ``AE_*`` env vars."""
        return cls(
            mcts_enabled=_env_flag("AE_MCTS", False),
            mcts_depth=_env_int("AE_MCTS_DEPTH", 3),
            mcts_width=_env_int("AE_MCTS_WIDTH", 24),
            mcts_min_score=_env_float("AE_MCTS_MIN_SCORE", 12.0),
            # Hard per-call latency budget (env var is in ms). Cloud killed
            # MCTS v1 because DEPTH=5 WIDTH=96 with no cap ran 1.2-2.4s/tick.
            mcts_time_budget_s=_env_float("AE_MCTS_BUDGET_MS", 80.0) / 1000.0,
            mcts_log_timing=_env_flag("AE_MCTS_LOG_TIMING", True),
            tier1_defense_priority=_env_flag("AE_TIER1_DEFENSE", False),
            tier1_repeat_kill=_env_flag("AE_TIER1_REPEAT_KILL", True),
            tier1_shared_credit=_env_flag("AE_TIER1_SHARED_CREDIT", False),
            tier1_no_stay_penalty=_env_flag("AE_TIER1_NO_STAY_PENALTY", True),
            tier1_predictive_walk=_env_flag("AE_TIER1_PREDICTIVE_WALK", True),
            bomb_detonate_steps=_env_int("AE_BOMB_DETONATE_STEPS", 5),
            base_defense_health=_env_float("AE_BASE_DEFENSE_HEALTH", 60.0),
            base_defense_radius=_env_int("AE_BASE_DEFENSE_RADIUS", 4),
            enemy_base_value=_env_float("AE_ENEMY_BASE_VALUE", 80.0),
            dist_penalty=_env_float("AE_DIST_PENALTY", 1.15),
            path_threat_penalty=_env_float("AE_PATH_THREAT_PENALTY", 2.0),
            enemy_chase_value=_env_float("AE_ENEMY_CHASE_VALUE", 0.0),
            enemy_chase_radius=_env_int("AE_ENEMY_CHASE_RADIUS", 4),
            plan_rescore_enabled=_env_flag("AE_PLAN_RESCORE", False),
            plan_rescore_k=_env_int("AE_PLAN_RESCORE_K", 4),
            plan_rescore_margin=_env_float("AE_PLAN_RESCORE_MARGIN", 1.0),
            plan_rescore_time_tax=_env_float("AE_PLAN_RESCORE_TIME_TAX", 0.35),
            plan_rescore_base_w=_env_float("AE_PLAN_RESCORE_BASE_W", 1.0),
            plan_rescore_demote_only=_env_flag("AE_PLAN_RESCORE_DEMOTE_ONLY", True),
            contention_enabled=_env_flag("AE_CONTENTION", False),
            contention_scale=_env_float("AE_CONTENTION_SCALE", 2.5),
            contention_pfloor=_env_float("AE_CONTENTION_PFLOOR", 0.15),
            contention_topen=_env_int("AE_CONTENTION_TOPEN", 40),
            contention_tfresh=_env_int("AE_CONTENTION_TFRESH", 3),
            time_danger_enabled=_env_flag("AE_TIME_DANGER", False),
            danger_horizon=_env_int("AE_DANGER_HORIZON", 6),
            no_self_damage=_env_flag("AE_NO_SELF_DAMAGE", False),
            basekill_noescape=_env_flag("AE_BASEKILL_NOESCAPE", False),
            stun_tax_enabled=_env_flag("AE_STUN_TAX", False),
            stun_tax_mult=_env_float("AE_STUN_TAX_MULT", 2.0),
            fortress_enabled=_env_flag("AE_FORTRESS", False),
            fortress_phase=_env_float("AE_FORTRESS_PHASE", 0.6),
            fortress_base_mult=_env_float("AE_FORTRESS_BASE_MULT", 0.5),
            fortress_defense_mult=_env_float("AE_FORTRESS_DEFENSE_MULT", 1.5),
            fortress_threat_mult=_env_float("AE_FORTRESS_THREAT_MULT", 1.5),
            fortress_tether_w=_env_float("AE_FORTRESS_TETHER_W", 0.3),
            item_mission_value=_env_float("AE_ITEM_MISSION_VALUE", 50.0),
            item_resource_value=_env_float("AE_ITEM_RESOURCE_VALUE", 25.0),
            item_recon_value=_env_float("AE_ITEM_RECON_VALUE", 10.0),
            first_target_table_enabled=_env_int("AE_FIRST_TARGET_TABLE", 0) > 0,
            first_target_boost=_env_float("AE_FIRST_TARGET_BOOST", 60.0),
            first_target_decay=_env_float("AE_FIRST_TARGET_DECAY", 0.55),
            enemy_bomb_escape_enabled=_env_int("AE_ENEMY_BOMB_OVERRIDE", 0) > 0,
            enemy_bomb_escape_turn_penalty=_env_float("AE_ENEMY_BOMB_ESCAPE_TURN_PENALTY", 0.2),
            enemy_bomb_escape_visit_penalty=_env_float("AE_ENEMY_BOMB_ESCAPE_VISIT_PENALTY", 0.3),
            orientation_aware_path_enabled=_env_int("AE_ORIENTATION_AWARE_PATH", 0) > 0,
            orientation_aware_turn_cost=_env_float("AE_ORIENTATION_AWARE_TURN_COST", 1.0),
            lead_bomb_gate_base=_env_int("AE_LEAD_BOMB_GATE_BASE", 0) > 0,
            lead_base_tether=_env_int("AE_LEAD_BASE_TETHER", 0) > 0,
            lead_tether_health=_env_float("AE_LEAD_TETHER_HEALTH", 60.0),
            lead_tether_weight=_env_float("AE_LEAD_TETHER_WEIGHT", 0.5),
            lead_recon_discount=_env_int("AE_LEAD_RECON_DISCOUNT", 0) > 0,
            lead_recon_dist_mult=_env_float("AE_LEAD_RECON_DIST_MULT", 1.0),
        )

    def as_dict(self) -> dict:
        """Field name → value, for tuner encode/decode and logging."""
        return {f.name: getattr(self, f.name) for f in fields(self)}
