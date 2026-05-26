# Dockerfile for building the AE image.


# The base image. Using a regular Python base because AE agents don't typically
# require GPU. This will speed up your evaluation.
FROM python:3.11-slim

# Configures settings for the image.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_ROOT_USER_ACTION=ignore
# Pin PYTHONHASHSEED so AEManager's hash-order-dependent code paths (set/dict
# iteration in candidate scoring) produce deterministic decisions across cloud
# invocations. Without this, two byte-identical submissions of the same image
# can differ by ~0.10 in cloudsuite-equivalent score because hash order shifts
# which equal-cost candidate wins ties.
ENV PYTHONHASHSEED=0
WORKDIR /workspace

# Runtime manager selection (hybrid | policy | heuristic). Bake it into
# the image so the cloud evaluator (which doesn't inherit our shell env)
# uses the right mode. Edit this line before each `til build` to switch
# modes, OR write the mode into `ae/src/.ae_mode` (the server reads that
# file at startup if AE_MODE is unset in the container env).
ENV AE_MODE=heuristic
# MCTS-light is a known-regressed experiment, so keep it opt-in. See
# ae/NOTES.md for the speed budget math and the v1/v2 cloud failures.
# v1 used DEPTH=5 WIDTH=96 with no gate / no latency cap; cloud killed it after
# 2-4× the per-tick budget. v2 caps total search at AE_MCTS_BUDGET_MS via a
# time.monotonic() guard in AEManager._tactical_lookahead_action, AND only
# invokes MCTS when bombs/enemies/enemy-bases are within mcts_depth+1 reach
# (cheap _should_run_mcts pre-flight gate).
ENV AE_MCTS=0
ENV AE_MCTS_DEPTH=3
ENV AE_MCTS_WIDTH=24
ENV AE_MCTS_BUDGET_MS=80
ENV AE_MCTS_LOG_TIMING=0
ENV AE_MCTS_MIN_SCORE=12
ENV AE_HYBRID_TRUST_MCTS=0
ENV AE_HYBRID_MCTS_MIN_SCORE=12

# Tier-1 / Tier-2 toggles. Bisect on 100-200 round local sims (17 May 2026)
# showed individual changes are net-positive, but the 5-toggle bundle has
# negative interaction (planner thrashes between defense / kill / item).
# Ship the clean winning subset by default:
#   #6 NO_STAY_PENALTY  +0.034 (safe, 1-line change in fallback scoring)
#   #3 REPEAT_KILL      +0.029 (bomb cells where we just killed)
#   #7 PREDICTIVE_WALK  +0.013 (random-walk-aware bomb EV)
# Disabled by default (negative or neutral on random opponents):
#   #2 DEFENSE          +0.004 — neutral; may help on cloud, opt-in only
#   #4 SHARED_CREDIT    -0.044 — confirmed regressor; do not enable
# Tier 1 #1 (playbook) and Tier 2 #8 (opponent model) DEFAULT OFF after
# 200-round A/Bs (17 May 2026): playbook regressed -0.097 standalone on
# random opponents; bomb-only filter still -0.033. Hypothesis falsified
# locally — kept as opt-in for cloud-only A/Bs but not auto-enabled.
# To re-enable any, set the matching env var to 1 here or via `docker run -e ...`.
ENV AE_TIER1_DEFENSE=0
ENV AE_TIER1_REPEAT_KILL=1
ENV AE_TIER1_SHARED_CREDIT=0
ENV AE_TIER1_NO_STAY_PENALTY=1
ENV AE_TIER1_PREDICTIVE_WALK=1
ENV AE_USE_PLAYBOOK=0
ENV AE_USE_OPPONENT_MODEL=0
# Opt-in via `-e AE_USE_PLAYBOOK=1 -e AE_PLAYBOOK_FILTER=bomb_only` for a
# cloud-only A/B if local-cloud distribution gap turns out to favor the
# playbook on real opponents.

# Hail-mary AE knobs (24 May late session). Default values keep behavior
# byte-identical to the 0.638 build; edit before `til build` to A/B.
#   AE_DIJKSTRA_BOMB_COST — step cost added when routing through a
#     destructible wall (effective cost = 1.0 + this). 23 May sweep was
#     monotonic in cloud score for {3.0, 4.0, 5.0} -> {0.544, 0.554, 0.569};
#     values >5.0 are untested. Try 7.0 / 10.0 for variance batches.
#   AE_ASTAR_TIEBREAK — when =1, secondary heap key in _dijkstra_distance_map
#     is Manhattan distance to enemy-base centroid, biasing equal-cost path
#     reconstruction toward enemy territory. Optimal distances unchanged.
ENV AE_ITEM_MISSION_VALUE=80
ENV AE_ITEM_RESOURCE_VALUE=40
ENV AE_ENEMY_BASE_VALUE=100
ENV AE_DIJKSTRA_BOMB_COST=7.0
ENV AE_ASTAR_TIEBREAK=0

# W2.1a — M5 spawn-aware first-target table. When AE_FIRST_TARGET_TABLE=1,
# enemy_base candidate values get an additive boost based on M5's
# FIRST_TARGET_BY_OWN_BASE priority (3 of 6 Novice spawns; others unchanged).
# Default OFF; set to 1 only after multi_seed_eval shows positive lift over
# the protected heuristic-A configuration.
ENV AE_FIRST_TARGET_TABLE=0
ENV AE_FIRST_TARGET_BOOST=60.0
ENV AE_FIRST_TARGET_DECAY=0.55

# W2.1b — M5 enemy-bomb-only escape override. When set, if a visible enemy
# bomb (own==False) with timer <= 2 has us in its blast, force an escape
# action chosen by M5 scoring (leave danger first, max distance from bomb,
# more open neighbors, fewer turns). Default OFF.
ENV AE_ENEMY_BOMB_OVERRIDE=0
ENV AE_ENEMY_BOMB_ESCAPE_TURN_PENALTY=0.2
ENV AE_ENEMY_BOMB_ESCAPE_VISIT_PENALTY=0.3

# W2.1c — orientation-aware A*. When set, _choose_target uses Dijkstra over
# (x, y, facing) states with real turn cost instead of grid-only Dijkstra
# that treats turns as free. Default OFF (grid Dijkstra). On the fixed
# Novice map only; general-map BFS path is untouched.
ENV AE_ORIENTATION_AWARE_PATH=0
ENV AE_ORIENTATION_AWARE_TURN_COST=1.0

# Installs your dependencies.
RUN pip install -U pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copies your source files.
COPY src .

# Copies trained policy weights (if any). Build still succeeds with an empty
# `models/` and the server falls back to the heuristic planner at runtime.
# Models can include:
#   bc.pt              — neural policy checkpoint (used by AE_MODE=hybrid|policy)
#   playbook.npz       — Tier-1 #1 offline playbook (built by training/ae/build_playbook.py)
#   opponent_model.json — Tier-2 #8 opponent action distribution
#   oracle_table.npz   — Tier-2 #10 oracle BC lookup
COPY models /workspace/models

# Starts your model server. uvloop + httptools are already pulled in by
# uvicorn[standard]; setting them explicitly ensures the faster event loop
# and parser are used (gives a small per-request latency win for AE since
# /ae is called every game tick).
CMD ["uvicorn", "ae_server:app", "--host", "0.0.0.0", "--port", "5005", "--loop", "uvloop", "--http", "httptools"]
