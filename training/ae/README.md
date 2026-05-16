# AE training pipeline

This directory holds scripts for training a learned AE policy. The current
shipped agent ([../../ae/src/ae_manager.py](../../ae/src/ae_manager.py)) is a
hand-coded BFS planner; the learned-policy path starts by imitating the planner
(behavior cloning) and then needs reinforcement learning against mixed opponents.
The official AE spec is here: <https://github.com/til-ai/til-26/wiki/Challenge-specifications#ae>

## Status

| Stage                     | Status      |
|---------------------------|-------------|
| Observation encoder       | implemented |
| Policy network            | implemented |
| Behavior-cloning collector| implemented |
| Behavior-cloning trainer  | implemented |
| Local policy evaluator    | implemented |
| PPO fine-tune             | scaffolded in `train_ppo.py`; run on Workbench |
| Deployment into `ae/src/` | implemented; `bc-v1` official regressed |

## Files

- `encoder.py` — observation dict → float32 tensors. Single source of truth for the obs/feature shape; reused by training and deployment.
- `model.py` — `PolicyNetwork`: small CNN over each viewcone + MLP over scalars. ~150k params, designed for CPU inference under ~5 ms/call.
- `collect_bc.py` — rolls out the planner in `til_environment.bomberman_env`, logs every `(obs, action)` pair to a compressed `.npz`.
- `train_bc.py` — supervised cross-entropy training of `PolicyNetwork` on the BC dataset. Masks illegal actions in both loss and argmax.
- `eval_policy.py` — runs a checkpoint against the env for N games, reports the same `score = total_reward / games / 1000` as `test/test_ae.py`.
- `train_ppo.py` — pure-PyTorch PPO fine-tune. Warm-starts from `bc.pt`, trains agent 0 against a mixed opponent pool, and saves a deployment-compatible actor checkpoint.

`data/` and `checkpoints/` are gitignored. Move/copy weights into a tracked location only at deploy time.

## Step 1 — collect BC dataset

```bash
# From the repo root (~/til on Workbench)
python training/ae/collect_bc.py --games 200 --out training/ae/data/bc.npz
```

What this does: 200 games × ~200 of our agent's turns = ~40k samples. With novice mode (default) the map is fixed-seed; pass `--no-novice` to sample varied maps. Random opponents play the other 5 agents.

Expected: ~5–10 minutes on Workbench. At the end the script prints the action-class distribution so you can spot collapse (e.g. 90 % `FORWARD` would mean the planner barely uses the rest).

## Step 2 — behavior clone

```bash
python training/ae/train_bc.py \
    --data training/ae/data/bc.npz \
    --out  training/ae/checkpoints/bc.pt \
    --epochs 20
```

What good looks like: `val_acc` rises from ~0.4 (random over 6 classes minus illegal-action filtering) to **0.85–0.95**. If it plateaus below 0.7, the obs encoder is probably dropping information the planner uses — re-check `encoder.py` first.

Expected: ~3–5 minutes on Workbench GPU.

## Step 3 — sanity-check the cloned policy

```bash
python training/ae/eval_policy.py \
    --checkpoint training/ae/checkpoints/bc.pt \
    --games 6
```

If BC worked, score should land near the planner's local score (~0.65–0.70). If it's far below, the network is failing to imitate — usually means more data or larger net.

## Step 4 — PPO fine-tune

Run this on Workbench after `bc.pt` exists:

```bash
python training/ae/train_ppo.py \
    --bc-checkpoint training/ae/checkpoints/bc.pt \
    --out training/ae/checkpoints/ppo.pt \
    --updates 200 \
    --games-per-update 8 \
    --eval-games 12 \
    --opponents mixed \
    --eval-opponents mixed
```

What this does:
- wraps `bomberman_env` as a single-agent PPO loop controlling `env.possible_agents[0]`;
- warm-starts the actor from BC if `bc.pt` exists;
- trains against a mixed opponent pool: random + frozen planner + frozen self-copies;
- keeps invalid actions masked in both sampling and PPO loss;
- saves a deployment-compatible actor checkpoint at `training/ae/checkpoints/ppo.pt`.

Do not trust random-opponent local score alone. `bc-v1` looked fine locally and still regressed officially. Only treat PPO as ready if it beats `planner-v3b` under mixed-opponent eval and keeps `0` invalid actions.

### Step 4b — PPO with scripted opponents (Tier 2 #9, recommended)

`--opponents scripted` is a new mode (May 2026) that trains PPO against a
fixed library of 5 distinct hand-written opponent types — random, greedy
collector, periodic bomber, defender, and hunter — implemented in
[opponents.py](opponents.py) and shared with [simulate.py](simulate.py).

Why this exists. The previous PPO modes (`mixed`, `league`, `selfplay`)
all leaned on either the planner or self-play snapshots. Self-play
overfit to our policy's idiosyncrasies; planner-only gives a
narrow-distribution opponent. The scripted library presents 5 distinct
strategy archetypes simultaneously (one per enemy slot), forcing the
policy to be robust to each — closer to what cloud's hidden NPC mix
likely looks like.

The scripted opponents are kept "pure" (their internal AEManager has
playbook + opponent-model + tier-1 toggles all disabled in
[opponents.py `_strip_aimanager_smarts`](opponents.py)) so PPO learns to
beat raw heuristic policies, not policies amplified by our own learned
artifacts.

Full Workbench recipe in [RUNBOOK.md](RUNBOOK.md#8-tier-2-9-ppo-retrain-optional-4-6-hr-gpu).

```bash
# Smoke first — should finish in <1 minute.
python training/ae/train_ppo.py \
    --bc-checkpoint training/ae/checkpoints/bc.pt \
    --out /tmp/ppo_smoke.pt \
    --updates 2 --games-per-update 2 \
    --eval-every 1 --eval-games 2 \
    --opponents scripted --eval-opponents scripted \
    --n-frames 4 --batch-size 64

# Full run — ~4-6 hr T4.
python training/ae/train_ppo.py \
    --bc-checkpoint training/ae/checkpoints/bc.pt \
    --out training/ae/checkpoints/ppo-scripted-v1.pt \
    --updates 200 --games-per-update 12 \
    --eval-every 5 --eval-games 12 \
    --opponents scripted --eval-opponents scripted \
    --n-frames 4
```

Submission gate: PPO ≥ heuristic + 0.05 on local `til test`, 0/30
errors. If it doesn't clear that bar, do not submit; document and
pivot.

## Step 5 — deploy

The inference path in [../../ae/src/ae_server.py](../../ae/src/ae_server.py) now supports three modes selected by `AE_MODE` (env var) or `ae/src/.ae_mode` (file fallback):

- `hybrid` (default, **current shipped tag is `hybrid-v2` at 0.545/0.863**) — policy chooses, heuristic vetoes illegal / unsafe-bomb / step-into-blast / frozen-stay actions. See [../../ae/src/hybrid_manager.py](../../ae/src/hybrid_manager.py).
- `policy` — pure `PolicyAEManager`.
- `heuristic` — pure rule-based `AEManager` (no torch needed in the image at all).

Deploy a new policy checkpoint by copying it into the model slot:

```bash
mkdir -p ae/models
cp training/ae/checkpoints/ppo.pt ae/models/bc.pt
echo hybrid > ae/src/.ae_mode        # or 'policy' / 'heuristic'
til build ae <tag>                   # bakes AE_MODE into the image
til test ae <tag>
til submit ae <tag>
```

Keep `ae/models/bc.pt` as the expected filename unless you also set `AE_POLICY_CHECKPOINT`, because `policy_manager.py` searches for that path by default.

**Important**: `AE_MODE=foo til build …` does NOT work — `docker build` doesn't inherit the shell env, so the cloud container would default to hybrid regardless. Either edit the `ENV AE_MODE=…` line in `ae/Dockerfile` or write the mode into `ae/src/.ae_mode` (gitignored) before each build.

## Notes / gotchas

- The encoder normalizes scalars by max plausible values (60 hp, 100 base hp,
  10 resources, 10 bombs, 200 steps). If the qualifier uses different caps
  these need to be re-tuned, but the values match the published env config.
- `action_mask` is fed into the loss as `log(mask)` so the network never
  learns to output an illegal action; if you remove that, expect random
  illegal-action errors at submission time.
- Don't commit `.npz` or `.pt` files — they're large and the dataset is
  reproducible by re-running `collect_bc.py`.
