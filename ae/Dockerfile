# Dockerfile for building the AE image.


# The base image. Using a regular Python base because AE agents don't typically
# require GPU. This will speed up your evaluation.
FROM python:3.11-slim

# Configures settings for the image.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_ROOT_USER_ACTION=ignore
WORKDIR /workspace

# Runtime manager selection (hybrid | policy | heuristic). Bake it into
# the image so the cloud evaluator (which doesn't inherit our shell env)
# uses the right mode. Edit this line before each `til build` to switch
# modes, OR write the mode into `ae/src/.ae_mode` (the server reads that
# file at startup if AE_MODE is unset in the container env).
ENV AE_MODE=hybrid
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
ENV AE_HYBRID_ALLOW_FIXED_POLICY=0

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
# Fixed-map macro routing is implemented but kept opt-in after local pressure
# suite A/Bs: exact enemy-base attack cells can help a local library run, but
# broad macro routing was unstable against rusher/hunter-style opponents.
ENV AE_FIXED_MACRO=0
ENV AE_FIXED_CENTER=0
ENV AE_FIXED_ATTACK_CELLS=1
ENV AE_ITEM_PRIOR_CONF=0.70
ENV AE_ITEM_PRIOR_FLOOR=0.25
ENV AE_ALLY_BOMB_SAFE=0
ENV AE_TTD_DEFENSE=0
# Deterministic pessimistic mini-search candidate. Unlike MCTS-light, this is
# a tiny safety-only override: it runs only near live bombs / fresh enemy
# pressure, searches our own movement for 3 plies, and requires a large safety
# delta before overriding the normal fixed-map planner. Mac 24-round gate:
# random 0.5613, library 0.5184, cloudsuite 0.3962 (vs reference ~0.3186).
ENV AE_PESSIMISTIC_SEARCH=1
ENV AE_PESSIMISTIC_DEPTH=3
ENV AE_PESSIMISTIC_ENEMY_GATE=1
ENV AE_PESSIMISTIC_FORCE_SCORE=-80
ENV AE_PESSIMISTIC_MIN_DELTA=55
# Opt-in via `-e AE_USE_PLAYBOOK=1 -e AE_PLAYBOOK_FILTER=bomb_only` for a
# cloud-only A/B if local-cloud distribution gap turns out to favor the
# playbook on real opponents.

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
