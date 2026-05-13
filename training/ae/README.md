# AE training pipeline

This directory holds scripts for training a learned AE policy. The current
shipped agent ([../../ae/src/ae_manager.py](../../ae/src/ae_manager.py)) is a
hand-coded BFS planner; the goal of this pipeline is a small neural-network
policy that starts by imitating the planner (behavior cloning) and is then
fine-tuned with reinforcement learning.

## Status

| Stage                     | Status      |
|---------------------------|-------------|
| Observation encoder       | implemented |
| Policy network            | implemented |
| Behavior-cloning collector| implemented |
| Behavior-cloning trainer  | implemented |
| Local policy evaluator    | implemented |
| PPO fine-tune             | TODO (next session) |
| Deployment into `ae/src/` | TODO (after BC works)|

## Files

- `encoder.py` — observation dict → float32 tensors. Single source of truth for the obs/feature shape; reused by training and deployment.
- `model.py` — `PolicyNetwork`: small CNN over each viewcone + MLP over scalars. ~150k params, designed for CPU inference under ~5 ms/call.
- `collect_bc.py` — rolls out the planner in `til_environment.bomberman_env`, logs every `(obs, action)` pair to a compressed `.npz`.
- `train_bc.py` — supervised cross-entropy training of `PolicyNetwork` on the BC dataset. Masks illegal actions in both loss and argmax.
- `eval_policy.py` — runs a checkpoint against the env for N games, reports the same `score = total_reward / games / 1000` as `test/test_ae.py`.

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

## Step 4 — PPO fine-tune (TODO)

Once BC matches the planner locally, the next session adds `train_ppo.py`:
- Wrap `bomberman_env` as a single-agent gym env (us controlling agent 0).
- Initialize PPO with the BC weights.
- Mix opponents: random + frozen planner + frozen self-copies.
- Track local score every N steps, keep the best checkpoint.

## Step 5 — deploy (TODO)

Add inference path to [../../ae/src/ae_manager.py](../../ae/src/ae_manager.py)
that loads the policy if present and falls back to the BFS planner if not.
Copy weights into `ae/models/` and add `torch` to `ae/requirements.txt`.
Test with `til build ae bc-v1 && til test ae bc-v1`.

## Notes / gotchas

- The encoder normalizes scalars by max plausible values (60 hp, 100 base hp,
  10 resources, 10 bombs, 200 steps). If the qualifier uses different caps
  these need to be re-tuned, but the values match the published env config.
- `action_mask` is fed into the loss as `log(mask)` so the network never
  learns to output an illegal action; if you remove that, expect random
  illegal-action errors at submission time.
- Don't commit `.npz` or `.pt` files — they're large and the dataset is
  reproducible by re-running `collect_bc.py`.
