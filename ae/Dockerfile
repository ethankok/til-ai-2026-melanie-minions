# Dockerfile for building the AE image.

# Regular Python base — AE agents don't need GPU.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_ROOT_USER_ACTION=ignore
# Pin hash order so candidate-scoring tie-breaks are deterministic across cloud runs
# (otherwise byte-identical images can differ by ~0.10 in score).
ENV PYTHONHASHSEED=0
WORKDIR /workspace

# Runtime manager selection, baked in so the cloud evaluator (no shell env) uses
# the right mode. ae/src/.ae_mode is read at startup if AE_MODE is unset.
# DEPLOY = confidence_policy_hybrid (confpol-native-u100): heuristic plays every
# tick; PPO consulted only on low-confidence ticks (AE_CONF_* below).
ENV AE_MODE=confidence_policy_hybrid
# C+bomb7 baseline profile.
ENV AE_ITEM_MISSION_VALUE=80
ENV AE_ITEM_RESOURCE_VALUE=40
ENV AE_ENEMY_BASE_VALUE=100
# Confidence-gate thresholds: PPO consulted on ~29% of ticks at these defaults.
ENV AE_CONF_MARGIN_EPSILON=5.0
ENV AE_CONF_TOP_FLOOR=10.0
ENV AE_CONF_OVERRIDE_TARGET_NONE=1
# Tactical PPO checkpoint path; stage the desired .pt as ae/models/tactical_policy.pt.
ENV AE_TACTICAL_POLICY_CHECKPOINT=/workspace/models/tactical_policy.pt
# Surgical positive-rate gate to block defensive regressions.
ENV AE_TACTICAL_MIN_POSITIVE_RATE=0.25
ENV AE_TACTICAL_MIN_ATTEMPTED=5
# MCTS-light: known-regressed, opt-in only. v2 caps total search via
# AE_MCTS_BUDGET_MS and a cheap pre-flight reachability gate.
ENV AE_MCTS=0
ENV AE_MCTS_DEPTH=3
ENV AE_MCTS_WIDTH=24
ENV AE_MCTS_BUDGET_MS=80
ENV AE_MCTS_LOG_TIMING=0
ENV AE_MCTS_MIN_SCORE=12
ENV AE_HYBRID_TRUST_MCTS=0
ENV AE_HYBRID_MCTS_MIN_SCORE=12

# Forward-sim plan re-score: re-ranks top-K target candidates by projected
# realized reward. DEFAULT OFF — failed the local reject gate.
ENV AE_PLAN_RESCORE=0
ENV AE_PLAN_RESCORE_K=4
ENV AE_PLAN_RESCORE_MARGIN=1.0
ENV AE_PLAN_RESCORE_TIME_TAX=0.35
ENV AE_PLAN_RESCORE_BASE_W=1.0
ENV AE_PLAN_RESCORE_DEMOTE_ONLY=1

# Contention-aware item valuation: discounts items an opponent reaches first.
# Demote-only, flag-off path byte-identical; shipped ON as near-zero-downside.
ENV AE_CONTENTION=1
ENV AE_CONTENTION_SCALE=2.5
ENV AE_CONTENTION_PFLOOR=0.15
ENV AE_CONTENTION_TOPEN=40
ENV AE_CONTENTION_TFRESH=3

# Offensive bomb-detonation timing: true ~5-step landing time for OUR
# respawn-camp bombs (gated behind recent_kills). Escape timing uses BOMB_TIMER=3.
ENV AE_BOMB_DETONATE_STEPS=5

# Tier-1 toggles: winning subset from local bisects (NO_STAY_PENALTY,
# REPEAT_KILL, PREDICTIVE_WALK). DEFENSE/SHARED_CREDIT/playbook/opponent-model
# stay OFF (neutral or regressors).
ENV AE_TIER1_DEFENSE=0
ENV AE_TIER1_REPEAT_KILL=1
ENV AE_TIER1_SHARED_CREDIT=0
ENV AE_TIER1_NO_STAY_PENALTY=1
ENV AE_TIER1_PREDICTIVE_WALK=1
ENV AE_USE_PLAYBOOK=0
ENV AE_USE_OPPONENT_MODEL=0

# AE_DIJKSTRA_BOMB_COST: extra step cost for routing through a destructible wall
# (effective cost = 1.0 + this). AE_ASTAR_TIEBREAK: when 1, break equal-cost ties
# toward enemy territory (optimal distances unchanged).
ENV AE_DIJKSTRA_BOMB_COST=7.0
ENV AE_ASTAR_TIEBREAK=0

# Spawn-aware first-target table: additive value boost for enemy bases per
# FIRST_TARGET_BY_OWN_BASE (3 of 6 Novice spawns). Default OFF.
ENV AE_FIRST_TARGET_TABLE=0
ENV AE_FIRST_TARGET_BOOST=60.0
ENV AE_FIRST_TARGET_DECAY=0.55

# Enemy-bomb-only escape override: if a visible enemy bomb (timer<=2) has us in
# its blast, force an escape action (away from danger, max distance, fewer turns).
ENV AE_ENEMY_BOMB_OVERRIDE=0
ENV AE_ENEMY_BOMB_ESCAPE_TURN_PENALTY=0.2
ENV AE_ENEMY_BOMB_ESCAPE_VISIT_PENALTY=0.3

# Orientation-aware A*: Dijkstra over (x, y, facing) with real turn cost,
# fixed Novice map only. Default OFF (grid Dijkstra).
ENV AE_ORIENTATION_AWARE_PATH=0
ENV AE_ORIENTATION_AWARE_TURN_COST=1.0

# Base-health-conditioned distance tether: below AE_LEAD_TETHER_HEALTH, targets
# are penalized by distance from our base (scaled by base damage) to pull the
# agent home.
ENV AE_LEAD_BASE_TETHER=1
ENV AE_LEAD_TETHER_HEALTH=60.0
ENV AE_LEAD_TETHER_WEIGHT=0.5
ENV AE_LEAD_BOMB_GATE_BASE=0
ENV AE_LEAD_RECON_DISCOUNT=0

# g02-sample-08 CEM planner weights — DEPLOYED (passed all finals-aligned local
# gates + hardware A/B vs incumbent semis2b-u75; balanced profile, retains
# defense). Overrides the C+bomb7 defaults above via Docker last-wins.
ENV AE_ITEM_MISSION_VALUE=62.583312
ENV AE_ITEM_RESOURCE_VALUE=57.370403
ENV AE_ENEMY_BASE_VALUE=80.370903
ENV AE_DIST_PENALTY=1.331559
ENV AE_PATH_THREAT_PENALTY=2.703109
ENV AE_DIJKSTRA_BOMB_COST=8.146925
ENV AE_LEAD_TETHER_HEALTH=52.902480
ENV AE_LEAD_TETHER_WEIGHT=1.314557
ENV AE_CONTENTION_SCALE=2.967429
ENV AE_CONTENTION_PFLOOR=0.093669

RUN pip install -U pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src .

# Build still succeeds with an empty models/ — server falls back to the
# heuristic planner at runtime.
COPY models /workspace/models

# uvloop + httptools for lower per-request latency (called every game tick).
CMD ["uvicorn", "ae_server:app", "--host", "0.0.0.0", "--port", "5005", "--loop", "uvloop", "--http", "httptools"]
