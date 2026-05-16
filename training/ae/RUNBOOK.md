# AE Tier-1 + Tier-2 runbook

This is the end-to-end recipe for the AE workstream that targets the
0.8 score. Read top-to-bottom on first run; section headers are jump
points after that.

Quick reference of artifacts produced at each step:

```
training/ae/data/sim-random.npz          1000 rounds vs 5x random       Tier-1 #1, Tier-2 #8/10
training/ae/data/sim-library.npz         1000 rounds vs scripted mix    Tier-1 #1, Tier-2 #8/10
training/ae/data/sim-mixed.npz           1000 rounds vs 5x mixed        Tier-1 #1, Tier-2 #8/10
ae/models/playbook.npz                   state→action lookup            Tier-1 #1
ae/models/opponent_model.json            walk-distance scalar           Tier-2 #8
ae/models/oracle_table.npz               oracle BC table                Tier-2 #10
training/ae/checkpoints/ppo-scripted.pt  PPO trained vs scripted pool   Tier-2 #9
```

---

## 0. Prereqs

You need either the project's `.venv` or anything else with the
`til-26-ae` editable install + numpy. On the Mac dev box:

```bash
.venv/bin/python -c "import til_environment; print('env OK')"
```

On Workbench it should "just work" after `pip install -r requirements-dev.txt`.

Sanity check the AE manager imports cleanly with all toggles:

```bash
.venv/bin/python -c "
import sys; sys.path.insert(0, 'ae/src')
from ae_manager import AEManager
m = AEManager()
print('flags:',
      m.tier1_defense_priority, m.tier1_repeat_kill,
      m.tier1_shared_credit, m.tier1_no_stay_penalty,
      m.tier1_predictive_walk)
print('playbook:', m.playbook is not None)
print('walk_scale:', m.opponent_walk_scale)
"
```

If any toggle prints `False` you forgot the env var; ship-time defaults
should be all-True.

Run the existing AE planner unit tests:

```bash
.venv/bin/python training/ae/_run_tests.py
# expect: 9/9 passed
```

---

## 1. Validate Tier-1 in-code changes (10 min)

Before sinking time into long simulations, verify Tier 1 is positive on
its own.

```bash
# Tier 1 OFF — establishes pre-change baseline.
AE_TIER1_DEFENSE=0 AE_TIER1_REPEAT_KILL=0 AE_TIER1_SHARED_CREDIT=0 \
  AE_TIER1_NO_STAY_PENALTY=0 AE_TIER1_PREDICTIVE_WALK=0 \
  AE_USE_PLAYBOOK=0 AE_USE_OPPONENT_MODEL=0 \
  .venv/bin/python training/ae/simulate.py \
    --rounds 50 --opponents random --our heuristic --no-log-traj \
    --summary-out training/ae/data/baseline-off.json

# Tier 1 ON, no playbook yet.
.venv/bin/python training/ae/simulate.py \
  --rounds 50 --opponents random --our heuristic --no-log-traj \
  --summary-out training/ae/data/baseline-on.json
```

Expected on the Mac (~50 sec total): the ON variant should outscore OFF
by 0.03-0.07. If the ON variant regresses, bisect by toggling one
`AE_TIER1_*` flag at a time to find the bad change, then file a fix.

Cloud submission gate after this step is **not** taken here — Tier 1
ships only after Tier 2's playbook has been built and integrated, since
they're cheaper to validate together than separately.

---

## 2. Run the simulations (1-2 hours)

Three runs against three opponent distributions. Each writes a
trajectory `.npz` file that downstream scripts consume.

```bash
mkdir -p training/ae/data

# Run 1 — vs random (matches `til test`'s NPCs).
.venv/bin/python training/ae/simulate.py \
  --rounds 1000 --opponents random --our heuristic \
  --out training/ae/data/sim-random.npz \
  --summary-out training/ae/data/sim-random-summary.json

# Run 2 — vs scripted library (5 different opponent types per round).
.venv/bin/python training/ae/simulate.py \
  --rounds 1000 --opponents library --our heuristic \
  --out training/ae/data/sim-library.npz \
  --summary-out training/ae/data/sim-library-summary.json

# Run 3 — vs mixed (each enemy slot picks a fresh opponent type per game).
.venv/bin/python training/ae/simulate.py \
  --rounds 1000 --opponents mixed --our heuristic \
  --out training/ae/data/sim-mixed.npz \
  --summary-out training/ae/data/sim-mixed-summary.json
```

Mac timing: ~17 min for 1000 random, ~21 min for 1000 library/mixed.
Workbench is faster.

Tip — run all three in parallel on a multi-core box (the env is mostly
Python-bound so each run is ~1 CPU):

```bash
for spec in random library mixed; do
  .venv/bin/python training/ae/simulate.py \
    --rounds 1000 --opponents $spec --our heuristic \
    --out training/ae/data/sim-$spec.npz \
    --summary-out training/ae/data/sim-$spec-summary.json &
done
wait
```

After each run finishes, look at the summary JSON's `mean_score`. The
random run is comparable to `til test` (~0.62-0.68 expected with Tier 1
on). The library run will be lower because some scripted opponents
actively counter our heuristic.

---

## 3. Build the playbook (Tier 1 #1) — 1 min

Aggregates the trajectories into a state→action lookup. Use all three
trajectory files so the playbook is robust across opponent
distributions.

```bash
.venv/bin/python training/ae/build_playbook.py \
  --inputs training/ae/data/sim-random.npz \
           training/ae/data/sim-library.npz \
           training/ae/data/sim-mixed.npz \
  --out ae/models/playbook.npz \
  --report training/ae/data/playbook-report.json \
  --min-visits 5 \
  --min-margin 0.04
```

What "good" looks like in the printed output:

```
playbook entries: 8000 - 25000     # raw count varies; both bounds OK
action mix: BOMB count is 5-25%    # if BOMB > 40%, --min-margin too low
```

If `playbook entries < 1000`, raise `--min-visits` to 3 and lower
`--min-margin` to 0.02 to retain more states. If `entries > 50000`,
the playbook is overfit to the simulator; tighten with `--min-margin
0.06`.

---

## 4. Fit opponent model (Tier 2 #8) — 10 min

Re-runs simulation but only logs *opponent* per-step actions, not ours.
Fits an action distribution that the AE manager reads at startup to
scale predictive-walk credit.

```bash
.venv/bin/python training/ae/fit_opponent_model.py \
  --rounds 200 --opponents library \
  --out ae/models/opponent_model.json
```

Expected output:

```
weighted_walk_distance = 0.3 - 0.7
by-opponent walk distance:
  random   ~0.40
  greedy   ~0.55
  bomber   ~0.55
  defender ~0.30
  hunter   ~0.55
```

The numbers above are typical; deviations are OK as long as they're
plausible. The AE manager clips the multiplier to [0.4, 2.0] so even
a bad fit can't blow up the heuristic.

---

## 5. Build oracle table (Tier 2 #10) — 1 min

Picks, per state, the action whose mean score across simulated rounds
was highest by at least `--min-action-diff`. Ships as a tabular policy
file with the same shape as the playbook.

```bash
.venv/bin/python training/ae/oracle_bc.py \
  --inputs training/ae/data/sim-random.npz \
           training/ae/data/sim-library.npz \
           training/ae/data/sim-mixed.npz \
  --out ae/models/oracle_table.npz \
  --min-visits 4 \
  --min-action-diff 0.03
```

Currently the manager doesn't directly use `oracle_table.npz` — it's
the same shape as `playbook.npz`. If the oracle table outperforms the
playbook, drop it in by:

```bash
mv ae/models/playbook.npz ae/models/playbook.legacy.npz
mv ae/models/oracle_table.npz ae/models/playbook.npz
```

The differentiation between playbook and oracle table is in the
selection rule (best mean vs best margin). Build both, compare in
local sim, ship the winner.

---

## 6. Local A/B before any submission (5 min)

After steps 1-5 the manager has playbook + opponent_model loaded
automatically (because the files exist in `ae/models/`). Verify the
gain holds on a fresh sim run:

```bash
.venv/bin/python training/ae/simulate.py \
  --rounds 200 --opponents library --our heuristic --no-log-traj \
  --summary-out training/ae/data/post-tier1-tier2.json
```

Expect ~0.05-0.10 gain over the baseline-on number from step 1 (which
had no playbook / opponent model). If it didn't move, the playbook
isn't covering enough states — return to step 3 and lower
`--min-visits`.

---

## 7. Cloud submission (Workbench)

Once local A/B looks healthy, submit:

```bash
echo heuristic > ae/src/.ae_mode  # or 'hybrid' if PPO weights present
til build ae tier1-tier2-v1
til test ae tier1-tier2-v1        # last sanity, still uses random NPCs
til submit ae tier1-tier2-v1
```

The leaderboard keeps the higher score from any submission, so
shipping `tier1-tier2-v1` cannot demote `hybrid-v3` even if it
regresses. Watch the cloud number, not the local one.

---

## 8. Tier 2 #9 (PPO retrain, optional, 4-6 hr GPU)

Workbench-only — local Mac CPU is too slow for 200 PPO updates. Trains
PPO with our agent in slot 0 and the 5-opponent scripted library
(random / greedy / bomber / defender / hunter, one per slot) in slots
1-5. BC-warm-start from `bc.pt` is the recommended base.

**Pre-flight checks (run on Workbench, before kicking off training).**

```bash
# 0. Make sure repo is up to date and the venv has all training deps.
cd ~/til  # or wherever the repo lives on the workbench
git pull origin main
pip install -r requirements-dev.txt

# 1. Confirm BC checkpoint shape — n_frames must match the training arg.
#    train_ppo.py silently SKIPS the warm-start if shapes don't match;
#    we lost ppo-selfplay-v1 to this exact mismatch (--n-frames 1 vs
#    a checkpoint with n_frames=4). Always check first.
python - <<'PY'
import torch
ckpt = torch.load("training/ae/checkpoints/bc.pt", map_location="cpu", weights_only=False)
print("BC n_frames=", ckpt.get("n_frames"), "use_belief=", ckpt.get("use_belief"))
PY

# 2. Smoke the scripted-opponent path with a tiny config first. Should
#    finish in <1 minute and print "✓ best PPO eval ..." once.
python training/ae/train_ppo.py \
    --bc-checkpoint training/ae/checkpoints/bc.pt \
    --out /tmp/ppo_smoke.pt \
    --updates 2 --games-per-update 2 \
    --eval-every 1 --eval-games 2 \
    --opponents scripted --eval-opponents scripted \
    --n-frames 4 --batch-size 64
```

If the smoke run errors out, do not start the long run; debug first.

**Full training run.**

```bash
# Match BC's n_frames; for the shipped bc.pt that's 4.
# 200 updates × 12 games × ~200 ticks = 480k transitions
# Wall clock target: 4-6 hr on T4. Worth running overnight.
python training/ae/train_ppo.py \
    --bc-checkpoint training/ae/checkpoints/bc.pt \
    --out training/ae/checkpoints/ppo-scripted-v1.pt \
    --updates 200 \
    --games-per-update 12 \
    --eval-every 5 \
    --eval-games 12 \
    --opponents scripted \
    --eval-opponents scripted \
    --n-frames 4
```

What "good" looks like during training:

- Best eval climbs over the first 30-60 updates and plateaus by ~120-150.
- `pi_loss` is small and oscillates around 0; `v_loss` decreases steadily.
- Entropy gently decays from ~1.7 → 0.7-1.0 (more confident policy over time).
- Eval score should clear BC's local mixed score (~0.487) by update 30.
- If best eval is still climbing at update 200, rerun with `--updates 300`.

**Local A/B before submission.**

```bash
# Deploy and test against the same scripted library locally — this is
# the proper apples-to-apples comparison vs the heuristic baseline.
cp training/ae/checkpoints/ppo-scripted-v1.pt ae/models/bc.pt
echo hybrid > ae/src/.ae_mode      # keep the safety-veto wrapper

# Container test (same env til test uses on cloud).
til build ae ppo-scripted-v1
til test ae ppo-scripted-v1

# Local sim against the scripted library (faster than docker test).
python training/ae/simulate.py \
    --rounds 200 --opponents library --our heuristic --no-log-traj
# Note: --our heuristic still uses the AEManager planner. PPO weights
# only fire when AE_MODE is hybrid|policy AND the manager goes through
# policy_manager.py. The runbook doesn't yet wire the hybrid wrapper
# into simulate.py, so the canonical local A/B is `til test ae <tag>`.
```

**Submission gate.** Submit only if:

- Local `til test` against the scripted-library cloud-equivalent shows
  PPO ≥ heuristic + 0.05 (i.e. the new training distribution actually
  helps).
- `0 / 30` errors during local test.
- The blended local score (75% acc + 25% speed) clears `hybrid-v3`'s
  blended 0.628 by at least 0.01.

If the run produces a regression vs the heuristic + Tier-1, **do not
submit**. Note the result in `ae/NOTES.md` and pivot to Tier 3.

**Likely failure modes to anticipate.**

- Training collapses to STAY-everywhere — the `entropy` value will
  drop to <0.1. Restart with `--entropy-coef 0.05` to keep exploration.
- v_loss explodes — `--reward-scale` too low. The default 50 is
  calibrated for raw reward range; if raw rewards spike past 50 in
  rollouts (e.g. multi-base destruction games), bump to 100.
- Checkpoint warns "shapes don't match" — your `--n-frames` doesn't
  match the BC checkpoint. Fix the flag, restart.
- Scripted opponents seem too easy — switch eval to `--eval-opponents
  mixed` for a stricter validation distribution.

After PPO finishes, this concludes Tier 2. The full set of artifacts
(playbook, opponent_model, oracle_table, ppo-scripted-v1) are then
available for any combinatorial A/B you want to run before the
submission deadline.

---

## 9. Decision tree for the next 7 days

```
Tier 1 alone improves cloud score → Tier 2 #1 (playbook) should compound.
Tier 1 alone is flat              → bisect Tier 1 toggles, find regressor.
Tier 1 + playbook improves cloud  → ship; iterate on opponent model + oracle.
Tier 1 + playbook is flat cloud   → playbook is overfit to simulator
                                    distribution. Tighten min_margin /
                                    increase min_visits, OR pivot to
                                    Tier 3 (hierarchical agent).
Tier 1 regresses cloud            → toggle off the regressing flag(s)
                                    via Dockerfile ENV vars, rebuild
                                    without retraining anything.
```

Always keep `hybrid-v3` ship-able as a fallback.
