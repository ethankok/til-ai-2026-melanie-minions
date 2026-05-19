# AE training pipeline + Tier-1/Tier-2 runbook

This directory holds scripts for training a learned AE policy and for the
Tier-1/Tier-2 heuristic-augmentation workstream targeting 0.8 score. The
current shipped agent ([../../ae/src/ae_manager.py](../../ae/src/ae_manager.py))
is a hand-coded BFS planner; the learned-policy path imitates it via behavior
cloning then PPO-fine-tunes against mixed opponents.

Official AE spec: <https://github.com/til-ai/til-26/wiki/Challenge-specifications#ae>

Quick reference of artifacts produced at each step:

```
training/ae/checkpoints/bc.pt            BC-imitated planner            base for PPO
training/ae/data/sim-random.npz          1000 rounds vs 5x random       Tier-1 #1, Tier-2 #8/10
training/ae/data/sim-library.npz         1000 rounds vs scripted mix    Tier-1 #1, Tier-2 #8/10
training/ae/data/sim-mixed.npz           1000 rounds vs 5x mixed        Tier-1 #1, Tier-2 #8/10
ae/models/playbook.npz                   state→action lookup            Tier-1 #1
ae/models/opponent_model.json            walk-distance scalar           Tier-2 #8
ae/models/oracle_table.npz               oracle BC table                Tier-2 #10
training/ae/checkpoints/ppo-scripted.pt  PPO trained vs scripted pool   Tier-2 #9
```

---

## Pipeline overview

### File glossary

- `encoder.py` — observation dict → float32 tensors. Single source of truth for the obs/feature shape; reused by training and deployment.
- `model.py` — `PolicyNetwork`: small CNN over each viewcone + MLP over scalars. ~150k params, designed for CPU inference under ~5 ms/call.
- `collect_bc.py` — rolls out the planner in `til_environment.bomberman_env`, logs every `(obs, action)` pair to a compressed `.npz`.
- `train_bc.py` — supervised cross-entropy training of `PolicyNetwork` on the BC dataset. Masks illegal actions in both loss and argmax.
- `eval_policy.py` — runs a checkpoint against the env for N games, reports the same `score = total_reward / games / 1000` as `test/test_ae.py`.
- `train_ppo.py` — pure-PyTorch PPO fine-tune. Warm-starts from `bc.pt`, trains agent 0 against a mixed opponent pool, saves a deployment-compatible actor checkpoint.
- `simulate.py`, `opponents.py` — Tier 1+2 simulation harness with the 5-archetype scripted opponent library (random / greedy / bomber / defender / hunter).
- `build_playbook.py`, `fit_opponent_model.py`, `oracle_bc.py` — Tier 1+2 artifact builders.

`data/` and `checkpoints/` are gitignored. Move/copy weights into a tracked location only at deploy time.

### Status

| Stage                     | Status      |
|---------------------------|-------------|
| Observation encoder       | implemented |
| Policy network            | implemented |
| Behavior-cloning collector| implemented |
| Behavior-cloning trainer  | implemented |
| Local policy evaluator    | implemented |
| PPO fine-tune             | implemented; `ppo-scripted-v1` shipped, REGRESSED cloud |
| Tier 1+2 artifact builders| implemented |
| Deployment into `ae/src/` | implemented; shipped tag = `hybrid-v3` |

### BC training (steps A–C below) is a prerequisite for PPO (step 8) but not for Tier-1/Tier-2 (steps 1-7)

#### A. Collect BC dataset

```bash
# From the repo root (~/til on Workbench)
python training/ae/collect_bc.py --games 200 --out training/ae/data/bc.npz
```

200 games × ~200 of our agent's turns = ~40k samples. With novice mode (default) the map is fixed-seed; pass `--no-novice` to sample varied maps. Random opponents play the other 5 agents. Expected: ~5-10 min on Workbench. Script prints action-class distribution at the end so you can spot collapse (e.g. 90% `FORWARD` would mean the planner barely uses the rest).

#### B. Behavior clone

```bash
python training/ae/train_bc.py \
    --data training/ae/data/bc.npz \
    --out  training/ae/checkpoints/bc.pt \
    --epochs 20
```

Good signal: `val_acc` rises from ~0.4 (random over 6 classes minus illegal-action filter) to **0.85-0.95**. If it plateaus below 0.7, the obs encoder is probably dropping information the planner uses — re-check `encoder.py`. Expected: ~3-5 min on Workbench GPU.

#### C. Sanity-check the cloned policy

```bash
python training/ae/eval_policy.py \
    --checkpoint training/ae/checkpoints/bc.pt \
    --games 6
```

If BC worked, score should land near the planner's local score (~0.65-0.70). If it's far below, the network is failing to imitate — usually means more data or larger net.

### Deployment mechanics (used by step 7 and step 8 below)

The inference path in [../../ae/src/ae_server.py](../../ae/src/ae_server.py) supports three modes selected by `AE_MODE` (env var) or `ae/src/.ae_mode` (file fallback):

- `hybrid` (default, **shipped at `hybrid-v3` 0.555/0.849**) — policy chooses, heuristic vetoes illegal / unsafe-bomb / step-into-blast / frozen-stay actions. See [../../ae/src/hybrid_manager.py](../../ae/src/hybrid_manager.py).
- `policy` — pure `PolicyAEManager`.
- `heuristic` — pure rule-based `AEManager` (no torch needed in the image at all).

Deploy a new policy checkpoint by copying it into the model slot:

```bash
mkdir -p ae/models
cp training/ae/checkpoints/<your>.pt ae/models/bc.pt
echo hybrid > ae/src/.ae_mode        # or 'policy' / 'heuristic'
til build ae <tag>                   # bakes AE_MODE into the image
til test ae <tag>
til submit ae <tag>
```

Keep `ae/models/bc.pt` as the expected filename unless you also set `AE_POLICY_CHECKPOINT`, because `policy_manager.py` searches for that path by default.

**Important**: `AE_MODE=foo til build …` does NOT work — `docker build` doesn't inherit the shell env, so the cloud container would default to hybrid regardless. Either edit the `ENV AE_MODE=…` line in `ae/Dockerfile` or write the mode into `ae/src/.ae_mode` (gitignored) before each build.

### 19 May immediate handoff: speedcheck before more AE code

Latest cloud A/Bs (`no-mcts`, `no-vetofrozen`, `conf-0.3`, heuristic
restores) all lost about `0.25` speed vs the 14 May `hybrid-v3` score. Before
attributing that to code, resubmit the exact old registry image under a fresh
tag:

```bash
cd /home/jupyter/til

docker pull asia-southeast1-docker.pkg.dev/til-ai-2026/repo-til-26-melanie-minions/melanie-minions-ae:hybrid-v3
docker tag asia-southeast1-docker.pkg.dev/til-ai-2026/repo-til-26-melanie-minions/melanie-minions-ae:hybrid-v3 \
           melanie-minions-ae:hybrid-v3-speedcheck
til submit ae hybrid-v3-speedcheck
```

Expected interpretation before the result:

| Speedcheck result | Meaning | Next action |
|---|---|---|
| ~`0.555 / 0.849` | Cloud is fine; current code/runtime additions slowed AE. | Build `hybrid-v3-lean` from the 14 May hot path. |
| ~`0.555 / 0.60` | This week's cloud/evaluator state is slower. | Keep `hybrid-v3`; speed work is low-ROI. |
| In between | Mixed cause. | Lean build is still worth one fast A/B. |

Actual result: `hybrid-v3-speedcheck` scored `0.381 / 0.855` with `0 / 30`
errors. This does not match the expected branches. Speed recovered, so broad
cloud speed congestion is not the explanation; accuracy did not reproduce, so
first verify whether the pulled registry tag is truly the old high-scoring
image. Docker inspection confirmed the retag was clean: both tags point to
image `sha256:5bc18298206d7958de6157019b42a7709891d5e9c3e21d9b51a1d690685aacf0`
and registry digest
`sha256:83c999a6f833ea464fe9742604831cc70c5eb58f4efbd109c3db5ce8d2445fea`.
The remaining missing fact is whether that digest is the original 14 May
`hybrid-v3` digest.

Useful provenance commands on Workbench:

```bash
docker image inspect melanie-minions-ae:hybrid-v3-speedcheck \
  --format '{{.Id}} {{json .RepoDigests}} {{json .RepoTags}}'

docker image inspect asia-southeast1-docker.pkg.dev/til-ai-2026/repo-til-26-melanie-minions/melanie-minions-ae:hybrid-v3 \
  --format '{{.Id}} {{json .RepoDigests}} {{json .RepoTags}}'
```

If old build logs or Discord messages contain an immutable digest for the
original 14 May `hybrid-v3`, compare against that. If not, do not use mutable
tags as proof of same-bytes behavior.

Important training correction: `train_ppo.py` already defaults to Novice
fixed-map mode (`--novice` is true unless `--vary-maps` or `--no-novice` is
used). An explicit fixed-Novice command is still useful for a controlled rerun,
but not a first-ever Novice-map training attempt.

If running the controlled PPO rerun anyway, use the script's real argument names:

```bash
cd /home/jupyter/til
tmux new -s ae-train

python training/ae/train_ppo.py \
  --bc-checkpoint ~/ae-checkpoints-backup/deployed-bc-v1.pt \
  --out ~/ae-checkpoints-backup/novice-fixed-v1.pt \
  --updates 200 \
  --games-per-update 12 \
  --opponents scripted \
  --eval-opponents scripted \
  --novice \
  --eval-every 5 \
  --eval-games 30 \
  --n-frames 4 \
  --seed 42 \
  --eval-seed 42 \
  2>&1 | tee ~/novice-fixed-v1.log
```

Do **not** use `--total-steps`, `--warm-start`, or `--out-dir`; those flags do
not exist here. Detach tmux with `Ctrl-b`, then `d`; reattach with
`tmux attach -t ae-train`.

Workbench idle shutdown has killed previous AE runs. If settings cannot be
changed, leave this Jupyter cell running while the tmux session trains:

```python
import datetime
import time

while True:
    print(f"heartbeat {datetime.datetime.now().isoformat()}", flush=True)
    time.sleep(45)
```

Early kill rule: if eval is still around `0.50-0.55` by update 30, stop and
switch effort back to speedcheck/lean-build or external 0.9-score intelligence.

### Gotchas

- The encoder normalizes scalars by max plausible values (60 hp, 100 base hp, 10 resources, 10 bombs, 200 steps). If the qualifier uses different caps these need to be re-tuned, but the values match the published env config.
- `action_mask` is fed into the loss as `log(mask)` so the network never learns to output an illegal action; if you remove that, expect random illegal-action errors at submission time.
- Don't trust random-opponent local score alone. `bc-v1` looked fine locally and still regressed officially. Treat PPO as ready only if it beats `planner-v3b` under mixed-opponent eval and keeps `0` invalid actions.
- Don't commit `.npz` or `.pt` files — they're large and the dataset is reproducible by re-running `collect_bc.py`.

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

The scripted opponents are kept "pure" (their internal AEManager has
playbook + opponent-model + tier-1 toggles all disabled in
[opponents.py `_strip_aimanager_smarts`](opponents.py)) so PPO learns to
beat raw heuristic policies, not policies amplified by our own learned
artifacts.

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
