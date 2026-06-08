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

# Runtime manager selection (hybrid | policy | heuristic | macro_hybrid |
# confidence_hybrid). Bake it into the image so the cloud evaluator (which
# doesn't inherit our shell env) uses the right mode. Edit this line before
# each `til build` to switch modes, OR write the mode into `ae/src/.ae_mode`
# (the server reads that file at startup if AE_MODE is unset in the
# container env).
#
# confidence_hybrid: heuristic-C+bomb7 plays on every tick; the PPO tactical
# macro selector is consulted only when the heuristic itself reports low
# confidence in its top action (top-vs-runner-up margin below
# AE_CONF_MARGIN_EPSILON OR top absolute score below AE_CONF_TOP_FLOOR, or
# the heuristic returned no scorable target). On the 27 May n=5 hash
# multi-seed gate this is the only setup that didn't catastrophically
# collapse defense_trap; pressure2 lifts by +0.10 and the worst single-suite
# regression is top_seed_proxy at -0.115 (smaller than every prior PPO
# attempt). See ae/NOTES.md for the diagnostic.
# 29 May 2026: switched to pure `heuristic` mode for a clean cloud A/B of the
# base-tether lead (AE_LEAD_BASE_TETHER below). This config = heuristic + C+bomb7
# + tether, directly comparable to the prior `heuristic-c-bomb7-v1` cloud result
# (0.590) which was heuristic + C+bomb7 with NO tether. Isolating the tether as
# the single variable. Revert to AE_MODE=confidence_hybrid to restore the prior
# shipped image.
# 1 Jun 2026: shipped opening_hybrid (opening prefix + confpol), then REVERTED.
# Cloud variance-farm of opening+confpol: mean 0.584 (n=9) vs confpol-u860 alone
# 0.634 (n=13) — the opening REGRESSED confpol by ~0.05. The confpol diagnostic
# (training/ae/diagnose_opening_over_confpol.py) explains it: the gate was locked
# over the bare heuristic, but confpol is a *better opener*, so the fixed opening
# overwrites confpol's good early play. Over confpol the gate is net −0.014 local
# (only spawn 9,13 — where confpol opens poorly — still helps); 3 of 4 enabled
# spawns flip negative. Opening code/gate kept in tree (AE_MODE=opening_hybrid +
# AE_OPENING_PLANNER=heuristic still gives the validated +0.067 over the bare
# heuristic if confpol is ever dropped) but NOT the default.
# 1 Jun (later): re-swept the opening gate OVER CONFPOL (sweep_openings.py
# --planner confpol). The confpol-correct gate is DIFFERENT — enables spawns
# 9,13 (H20) / 2,6 (H8) / 6,2 (H12), disables 13,9 / 3,12 / 12,3 — because
# confpol opens 0/2/5 well but opens 3/4 *worse* than the heuristic. Local
# +0.068 over confpol (z 9.7/4.6/2.8, all complete). This confpol-specific gate
# is now baked into ae/src/openings_gate.json. It is the FARM EXPERIMENT, not the
# default: set AE_MODE=opening_hybrid (keep AE_OPENING_PLANNER=confidence_policy_hybrid)
# and variance-farm vs confpol-alone. Promote only if it clears 0.634 on cloud
# (the heuristic-gated version did NOT; this one fixes the planner-mismatch but
# the opponent transfer gap is still untested).
# ⭐ 1 Jun (eval went DETERMINISTIC → 1 submit = true score). Full re-rank of all
# historical images on the new eval found a NEW BEST:
#   confpol-native-u100 = 0.661  (vs confpol-u860 0.626, tether/heuristic 0.605/0.591,
#   opening-v2 0.626, pand-policy 0.525). The EARLY gated-PPO rung wins — it looked
#   worst on the old noisy eval (0.551, a low-tail draw); deterministic eval reveals
#   it's the champion (validates "earlier = less overfit"). The confpol NN is worth
#   it (0.626 > best heuristic 0.605); u100 beats everything by +0.035.
# DEPLOY = confpol-native-u100: stage training/ae/checkpoints/confpol-native-u100.pt
#   -> ae/models/bc.pt on the Workbench (gitignored), AE_MODE=confidence_policy_hybrid.
#   The image melanie-minions-ae:confpol-native-u100 is ALREADY built+submitted (0.661)
#   — `til submit ae confpol-native-u100` re-submits it without a rebuild.
#   Canary: `til test` log "AE policy loaded …" (NOT "falling back to heuristic").
# Follow-up: test even-earlier rungs (u25/u50/u75) — the inverted-U peak may be < u100.
ENV AE_MODE=confidence_policy_hybrid
# C+bomb7 baseline profile.
ENV AE_ITEM_MISSION_VALUE=80
ENV AE_ITEM_RESOURCE_VALUE=40
ENV AE_ENEMY_BASE_VALUE=100
# Confidence-gate thresholds. Defaults validated in the 27 May confidence
# distribution probe: at eps=5.0 / floor=10.0 the wrapper consults PPO on
# ~29% of total ticks (14.7% low_margin + 13.5% target_none + 0.8%
# low_top_score) and passes through on 70.6%.
ENV AE_CONF_MARGIN_EPSILON=5.0
ENV AE_CONF_TOP_FLOOR=10.0
ENV AE_CONF_OVERRIDE_TARGET_NONE=1
# Tactical PPO checkpoint. Copy the desired .pt to ae/models/tactical_policy.pt
# before `til build`. Files in tree right now:
#   tactical_policy_conf_ppo_v2.pt        — update-80 gated save, in-training
#                                            wrapper_delta=+0.078,
#                                            multi-seed n=5 wrapper_delta=-0.013
#   tactical_policy_conf_ppo_v2_latest.pt — update-150 final state,
#                                            multi-seed n=5 wrapper_delta=-0.030
# Either can be staged as `tactical_policy.pt` for submission.
ENV AE_TACTICAL_POLICY_CHECKPOINT=/workspace/models/tactical_policy.pt
# Surgical positive-rate gate variables to block defensive regressions.
ENV AE_TACTICAL_MIN_POSITIVE_RATE=0.25
ENV AE_TACTICAL_MIN_ATTEMPTED=5
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

# Forward-sim plan re-score (3 Jun). Re-ranks the top-K static target
# candidates in _choose_target by projected realized reward. DEFAULT OFF: it
# FAILED the local reject gate (demote-only flat +0.0096 weighted but
# top_seed_proxy -0.137; symmetric -0.080). See ae/NOTES.md "Forward-sim plan
# re-score". Kept here only to run the cloud A/B that the local gate can't
# settle: set AE_PLAN_RESCORE=1, build a SEPARATE tag (e.g. `planrescore`),
# submit it, and compare to the confpol-semis2b-u75 baseline (0.671). The
# demote-only guard is on by default so the cloud test never ships the
# catastrophic base-promotion variant.
ENV AE_PLAN_RESCORE=0
ENV AE_PLAN_RESCORE_K=4
ENV AE_PLAN_RESCORE_MARGIN=1.0
ENV AE_PLAN_RESCORE_TIME_TAX=0.35
ENV AE_PLAN_RESCORE_BASE_W=1.0
ENV AE_PLAN_RESCORE_DEMOTE_ONLY=1

# Contention-aware item valuation (3 Jun). _choose_target discounts item targets
# an opponent reaches first, using free fixed-map spawn geometry + live viewcone
# sightings. Demote-only, item-vs-item; flag-off path byte-identical. **SHIPPED ON
# (AE_CONTENTION=1)** as a near-zero-downside option for the Semis MELEE. The cloud
# A/B proved it NEUTRAL on the qualifier-style eval (tag `contention-aware` = 0.671
# = flag-off exactly, 0/30 err, speed 0.840 ~= baseline) -- but the cloud measures
# absolute reward at a FIXED spawn, NOT melee PLACEMENT vs other teams, which is
# the real Semis objective and the only place this lever can help. The
# de-contaminated melee showed nominal help (all_farmer placement 2.39->2.17,
# small/noisy). Downside is cloud-proven ~0; revert is ONE line (set to 0). See
# ae/NOTES.md "Contention-aware item valuation".
ENV AE_CONTENTION=1
ENV AE_CONTENTION_SCALE=2.5
ENV AE_CONTENTION_PFLOOR=0.15
ENV AE_CONTENTION_TOPEN=40
ENV AE_CONTENTION_TFRESH=3

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

# Lead ① (29 May 2026) — base-health-conditioned distance tether. When base
# health drops below AE_LEAD_TETHER_HEALTH, candidate targets are penalized by
# their Manhattan distance from our base, scaled by how damaged the base is,
# pulling the agent home instead of ranging far while the base is destroyed.
# Local n=5 furnished gate vs C+bomb7 baseline (0.2842 ± 0.0074): 0.2901 ±
# 0.0106, Δ +0.006 (+0.46σ) — within local noise, no suite collapses. ENABLED
# here only as a cloud A/B vs heuristic-c-bomb7-v1 (0.590). NOTE: one cloud
# sample carries ~±0.10 noise (cloud σ≈0.053); a single result is NOT
# conclusive — variance-farm n≥5 before drawing any conclusion. Set
# AE_LEAD_BASE_TETHER=0 to disable.
ENV AE_LEAD_BASE_TETHER=1
ENV AE_LEAD_TETHER_HEALTH=60.0
ENV AE_LEAD_TETHER_WEIGHT=0.5
# Leads ② / ③ stayed below the gate; keep OFF.
ENV AE_LEAD_BOMB_GATE_BASE=0
ENV AE_LEAD_RECON_DISCOUNT=0

# ─────────────────────────────────────────────────────────────────────────────
# g00-fixed-03 CEM planner-weight retune (EXPERIMENT — finals A/B, 8 Jun 2026).
# Overrides the 10 scalar planner weights above with the CEM-tuned values that
# Pareto-dominated semis2b-u75 on the local melee gate (mean placement 3.43→1.89)
# but were REVERTED after a −33% qualifier-style cloud drop (expected for a
# relative-rank retune; see ae/NOTES.md "Planner-weight CEM tuning"). Docker uses
# the LAST ENV for each key, so this block wins. Consultant checkpoint stays
# semis2b-u75 (g00 swaps WEIGHTS, not the .pt) — stage bc.pt=confpol-semis2b-u75.pt.
# Source: training/ae/data/planner-weight-cem-final/candidates.jsonl (candidate
# "g00-fixed-03"). To revert to the incumbent deploy, delete this whole block.
ENV AE_ITEM_MISSION_VALUE=51.157558
ENV AE_ITEM_RESOURCE_VALUE=23.038170
ENV AE_ENEMY_BASE_VALUE=89.398951
ENV AE_DIST_PENALTY=1.060815
ENV AE_PATH_THREAT_PENALTY=2.377860
ENV AE_DIJKSTRA_BOMB_COST=9.076016
ENV AE_LEAD_TETHER_HEALTH=54.267980
ENV AE_LEAD_TETHER_WEIGHT=0.314212
ENV AE_CONTENTION_SCALE=1.607649
ENV AE_CONTENTION_PFLOOR=0.144418
# ─────────────────────────────────────────────────────────────────────────────

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
