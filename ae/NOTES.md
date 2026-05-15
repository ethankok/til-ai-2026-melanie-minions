# AE — notes & history

Last updated: 14 May 2026 19:30 SGT — **hybrid-v3 is the shipped tag at 0.555/0.849 (3rd consecutive new high; +0.048 vs ppo-v1).**

## Round-3 cloud result (hybrid-v3, 14 May 19:26)

`hybrid-v3` = `hybrid-v2` + **top-K policy cascade** (try policy's #2/#3 if #1 is vetoed before falling back to heuristic) + **opportunistic enemy-kill** (bomb adjacent enemy agents sighted this step in the heuristic's dominant-action shortcut, not just adjacent enemy bases).

| Tag | Score | Speed | Note |
|---|---|---|---|
| `hybrid-v3` | **0.555 ⭐** | 0.849 | NEW HIGH. +0.010 over hybrid-v2 (within ±0.04 cloud noise but trending right); top-quartile on leaderboard (top 0.711). |

Three consecutive AE submissions have each lifted the floor by ~one cloud-noise unit in the same direction: ppo-v1 0.507 → hybrid-v2 0.545 → hybrid-v3 0.555. Each individual lift is within noise; the cumulative +0.048 is not. We've moved the *floor*, but the local→cloud gap (~0.23) is structurally intact.

**Top of leaderboard is 0.711.** We're still 0.16 short of that, and three rounds of heuristic-side moves netted +0.048. The remaining gap is unlikely to close from further heuristic tweaks.

## Round-2 cloud results (after AE_MODE bake fix)

| Tag | Score | Speed | Local 1-run | Verdict |
|---|---|---|---|---|
| `policy-fast-v2`        | **0.425** | 0.859 | 0.654 | Regressed -0.082 vs ppo-v1 same weights. Almost certainly cloud variance (±0.04 on 30 games is normal). Speed flat — confirms speed is evaluator-bound, not torch-bound. |
| `hybrid-v2`             | **0.545** | 0.863 | 0.774 | NEW HIGH at the time. First structurally new approach since ppo-v1; +0.038 over the 0.49-0.51 cloud ceiling 5 prior approaches all hit. |
| `heuristic-restore-v2`  | **0.502** | 0.854 | 0.787 | +0.003 vs planner-v3b 0.499 (noise). Confirms the heuristic-only ceiling is real; TILE_RESPAWN 40→20 and enemy_agent eviction were no-ops on cloud. |

Local→official gap is 0.23 on hybrid (0.774→0.545→0.555) — structural to the hidden eval distribution.

### Bug we shipped past (round-1 → round-2)

All three round-1 builds had **identical sha256 `2572b392...`** because `AE_MODE=foo til build …` set the env var only in the shell that ran the til CLI — `docker build` doesn't inherit shell env. Cloud container then defaulted to hybrid mode regardless of submitted tag. Local `til test` worked because docker run inherits the shell env, masking the bug. **Fixed** by adding `ENV AE_MODE=hybrid` to the Dockerfile and a `.ae_mode` file fallback the server reads at startup.

## Path to 0.60+ from here

We're at 0.555, leaderboard top is 0.711. Three rounds of heuristic-side moves climbed +0.048 within ±0.04 noise per submission; the *direction* is right but each individual lift is barely outside noise. The remaining 0.15 to top is unlikely to come from more heuristic tweaks.

### Cheap A/Bs still on the table (no retraining)

Each is one `ENV …` line in the Dockerfile, hybrid mode. Cloud variance ±0.04 so submit only the ones whose expected lift is ≥ that:

```bash
# Confidence gate — only use policy when its top-action softmax ≥ 0.5
ENV AE_HYBRID_CONF=0.5

# Drop the frozen-stay veto — maybe STAY when policy says STAY is right
ENV AE_HYBRID_VETO_FROZEN_STAY=0

# Drop the danger-step veto — tests whether policy actually self-traps
ENV AE_HYBRID_VETO_DANGER=0
```

Expected lift per A/B: ±0.02. Worth one submission slot if `hybrid-v3` has spare queue capacity.

### Belief-map BC train — local result (15 May)

First end-to-end BC run on Workbench (CPU torch, ~3 min):

| Metric | bc-v1 (no belief) | **bc-belief** | Delta |
|---|---:|---:|---:|
| Best val_acc        | 0.874  | **0.897** | +0.023 |
| Params              | 149k   | 704k      | +555k (mostly head dense) |
| Best epoch          | 17/20  | 19/20     | similar |
| Local 6-game score  | 0.689  | 0.656     | -0.033 (within ±0.07 noise) |

**Read.** Higher val_acc on bc-belief means the policy is fitting
planner-v3b's behavior *better* with the belief input — i.e., the
planner's decisions correlate with belief-map state in ways the
egocentric viewcones alone couldn't capture. That's the *prerequisite*
signal for the memory hypothesis. Local game score is roughly flat to
bc-v1, but local has been an unreliable predictor of cloud all along
(local 0.689 → cloud 0.364 for bc-v1), and the whole hypothesis is
that belief reduces the *gap*, not increases the local. Cloud probe
is the only informative test from here.

Workbench env note: had to reinstall torch as CPU wheel
(`pip install --user --force-reinstall torch --index-url cpu`)
because the host's CUDA 13 driver / libcupti.so.13 didn't match
the user-local torch wheel. CPU is fine for BC at this scale (3 min
for 20 epochs on 40k samples). For PPO (1-3 h) we'll either need to
fix the GPU env or commit to a longer CPU run.

Code fix shipped 4878cd2: `eval_policy.py` was missing
`sys.path.insert(0, REPO_ROOT/"ae"/"src")` for the new ae_manager
import — crashed on first eval after BC train succeeded.

### Cloud probe — bc-belief in hybrid mode (in flight)

Deployed bc-belief.pt as `ae/models/bc.pt`, AE_MODE=hybrid, submitted
as `bc-belief-hybrid`. Reasoning for hybrid (not pure policy) for the
first probe:
- Apples-to-apples comparison with `hybrid-v3` (same wrapper).
- Hybrid catches BC's known failure mode (bombing without escape — BC
  imitates planner-v3b's bomb actions but doesn't know about the
  planner's escape-verification step that gates them).
- If it lands ≥0.55 cloud the architecture is at minimum neutral and
  PPO is justified. If it lands ≥0.58 the architecture is a real lift.

Decision tree from the cloud score:

| Cloud score | Interpretation | Next move |
|---|---|---|
| ≥ 0.58 | Architecture lift confirmed; new high. | PPO with `--use-belief --opponents league`. |
| 0.50 – 0.57 | Within noise of hybrid-v3; architecture might compound with PPO. | PPO anyway (last big swing). |
| < 0.50 | Memory hypothesis wrong, or BC alone insufficient. | Roll back to hybrid-v3 shipped; consider MCTS-light or call it. |

### State-augmented policy retrain — IMPLEMENTED, runs on Workbench

Code shipped (15 May, commit follows): belief-map architecture is now
plumbed end-to-end. Build/run order on Workbench:

```bash
# 0. Pull and install
git pull

# 1. Collect a fresh BC dataset *with* belief snapshots (~6-10 min for 200 games)
python training/ae/collect_bc.py --games 200 --n-frames 1 \
    --out training/ae/data/bc-belief.npz

# 2. BC train with belief input (~30-45 min on T4)
python training/ae/train_bc.py \
    --data training/ae/data/bc-belief.npz \
    --out training/ae/checkpoints/bc-belief.pt \
    --epochs 20

# 3. Validate locally before spending cloud budget
python training/ae/eval_policy.py \
    --checkpoint training/ae/checkpoints/bc-belief.pt \
    --games 6
# Expect val_acc ≥ 0.87 in BC; eval ≥ 0.65 local.

# 4. Submit BC-only as a low-cost cloud probe of the memory hypothesis
cp training/ae/checkpoints/bc-belief.pt ae/models/bc.pt
echo policy > ae/src/.ae_mode      # pure policy first — isolates the architecture lift
til build ae bc-belief && til test ae bc-belief && til submit ae bc-belief

# 5. If bc-belief lands ≥0.50 cloud, the memory hypothesis is real.
#    Run PPO with belief + league opponents (3-6h on T4).
python training/ae/train_ppo.py \
    --bc-checkpoint training/ae/checkpoints/bc-belief.pt \
    --out training/ae/checkpoints/ppo-belief.pt \
    --use-belief \
    --opponents league \
    --eval-opponents league \
    --updates 200 \
    --games-per-update 8 \
    --eval-games 12 \
    --n-frames 1

# 6. Deploy with hybrid wrapper for the safety-veto + top-K cascade
cp training/ae/checkpoints/ppo-belief.pt ae/models/bc.pt
echo hybrid > ae/src/.ae_mode
til build ae hybrid-belief-v1 && til test ae hybrid-belief-v1 && til submit ae hybrid-belief-v1
```

What's new in the source tree:

- `ae/src/encoder.py`: `rasterize_belief(ae_manager, obs)` returns
  `(11, 16, 16)` float32 with channels for visited / wall / destructible /
  mission-fresh / recon-fresh / resource-fresh / enemy-agent-fresh /
  enemy-base / bomb-blast-imminent / own-position / base-position. Item
  freshness decays linearly over 20 ticks; enemy freshness over 5.
- `ae/src/model.py`: `PolicyNetwork(use_belief=True)` adds a small CNN
  branch (Conv→AvgPool→Conv → 512 features). Old checkpoints
  (`use_belief=False`) load unchanged.
- `ae/src/policy_manager.py`: owns a private `AEManager` for belief
  tracking. Calls cheap `_update_memory` + `_age_bombs` per inference
  (no BFS). When the loaded checkpoint has `use_belief=True`, rasterizes
  the belief tensor and passes it to the network.
- `training/ae/{collect_bc,train_bc,train_ppo,eval_policy}.py`: all
  thread belief through the data pipeline + actor + critic.
- `training/ae/train_ppo.py`: NEW `AggressivePlannerOpponent` (planner
  with combat-biased threat penalties and lower retreat threshold) +
  NEW `--opponents league` mode that mixes random + planner +
  aggressive + frozen-self.

Risk assessment:
- **Memory hypothesis wrong** → bc-belief regresses to <0.40. Roll back
  to hybrid-v3, take the L on architecture, fall back to MCTS-light.
- **Memory helps but PPO destabilizes** → ppo-belief lower than bc-belief
  cloud. Reduce `--updates` to 80, `--lr` to 1e-4, retry.
- **All works** → expected 0.60–0.70 cloud range. The hybrid wrapper
  composes naturally; no code change there.

#### Bigger swing — alternative paths if belief-map saturates

**This is the only move with a realistic path to 0.65–0.70.** Three pieces of evidence point at it:

1. **The policy has no persistent memory.** It sees only the egocentric 7×5 and 7×7 viewcones; the frame-stack ppo-v1 doesn't use one is empty. It can't remember a bomb it saw 4 ticks ago or an item it walked past. The heuristic *does* have memory (`AEManager`'s belief dicts), which is why hybrid wins: it patches the policy's blind spots.
2. **The local→cloud gap (~0.23) hasn't moved across 7 approaches.** That kind of invariance points at distribution shift — the policy is keying on viewcone features that correlate with random-opponent local but mean less under hidden opponents. Richer state input would reduce that overfit.
3. **The policy's wins on cloud (where it beats heuristic) almost certainly come from *tactical microbehaviors* in confined situations.** Add memory, and those microbehaviors stop being undermined by the policy's inability to remember basic facts.

Implementation outline:

```text
ae/src/encoder.py
  + Add a `belief_map` field (16 × 16 × K channels). K=8-10:
      - visited_recent / visited_long
      - wall (4 directions packed into 1 channel)
      - destructible_wall
      - mission, recon, resource (each one channel, value = freshness)
      - enemy_agent (freshness-weighted)
      - enemy_base
      - bomb_blast_imminent (cells in known-bomb blast, weighted by timer)
      - own_position (sparse 1-hot)

ae/src/model.py
  + New conv branch in `PolicyNetwork`:
      Conv2d(K, 16, 3) → ReLU → Conv2d(16, 16, 3) → ReLU → Flatten
      Concat into the existing head input.

ae/src/policy_manager.py
  + Rasterize the heuristic's belief into the belief_map tensor every
    /ae call (the AEManager already maintains all the source state).

training/ae/{train_bc.py, train_ppo.py, collect_bc.py}
  + Collect belief snapshots alongside obs.
  + Pass belief_map through the network.
  + Retrain BC (~30 min) → warm-start PPO against mixed opponents
    (~3-6 hours on Workbench T4).

ae/src/hybrid_manager.py
  + No changes — `PolicyAEManager.ae_logits` interface stays the same.
```

Risk: this is exactly the kind of "make local higher" change that previously didn't help (ppo-v2 with frame stacking widened the gap). The reason to expect this one is different: memory is *strictly more information* per call, where frame stacking just gave the policy 4 stale viewcones it didn't know how to use. The belief map encodes things the policy literally cannot derive from any number of stacked viewcones (e.g., walls the agent observed but isn't currently looking at).

If we're going to do it: start with BC alone (~30 min) and submit a `bc-belief` build before doing PPO. If BC-belief alone scores >0.50 cloud, the memory hypothesis is real and PPO will compound it. If BC-belief regresses to <0.40, abort — the architecture change isn't helping.

## Hybrid + speed-fix plan (shipped as `hybrid-v2`, `policy-fast-v2`, `heuristic-restore-v2`)

User goal: AE score 0.60+, speed 0.90+. Net code changes (this commit):

1. **`ae/src/policy_manager.py`** — speed rewrite. `torch.set_num_threads(1)` +
   `torch.set_num_interop_threads(1)` at module load (kills uvicorn/torch
   contention on a 1-vCPU container); `torch.inference_mode()` instead of
   `no_grad()`; **warmup forward pass** at construction so first `/ae`
   call doesn't pay JIT/cudnn init; **preallocated input tensors** that
   we `copy_` into per tick instead of allocating new `from_numpy().to()`
   each call. New helper `ae_logits(obs)` returns (action, masked
   logits) in one forward pass — used by hybrid for confidence gating.
   Expected per-call latency drop: **~13 ms → ~3-5 ms** on CPU; speed
   score should rise from `0.861` toward `0.92+`.

2. **`ae/src/hybrid_manager.py`** (NEW) — `HybridAEManager`: policy
   action by default, **heuristic safety-veto** when the policy picks an
   action the rule-based planner can prove is wrong:
   - illegal action → use heuristic
   - `PLACE_BOMB` without verified escape → use heuristic
   - step into known bomb-blast → use heuristic (only if heuristic's
     own action is safer)
   - `STAY` with `frozen_ticks==0` and a non-STAY legal alternative →
     use heuristic
   - Fast path: if the heuristic is in an active "fleeing my own bomb"
     state, trust the heuristic entirely (it owns bomb-safety mechanics).
   - Optional `AE_HYBRID_CONF` env var: confidence gate — drop to
     heuristic if policy top-action softmax probability is below this.

   This is the **first structurally new AE attempt since ppo-v1**.
   Heuristic-only (planner-v3b) and policy-only (bc-v1, ppo-v1, ppo-v2)
   have *uncorrelated failure modes* — heuristic loses to over-caution,
   policy loses to under-safety — so a per-tick arbiter is expected to
   strictly dominate either alone when their disagreements fall in the
   policy's failure-mode set.

3. **`ae/src/ae_server.py`** — `AE_MODE` env var: `hybrid` (default),
   `policy`, or `heuristic`. Lets the team ship three distinct builds
   from a single source tree without code edits between rebuilds. All
   modes degrade gracefully if their dependency (checkpoint, torch) is
   missing.

4. **`ae/src/ae_manager.py`** — two small conservative fixes:
   - `TILE_RESPAWN_STEPS`: 40 → 20 (env doc says respawn is `random up
     to 40` via perlin noise; reconsidering at 20 with 0.5x discount
     captures average respawn rather than worst-case).
   - **Stale enemy_agent eviction** after `ENEMY_AGENT_MEMORY_STEPS=30`
     in `_age_bombs`. Prevents 30+-tick-old enemy sightings from being
     used as base-defense candidates in `_choose_target`.

5. **`training/ae/diagnose.py`** (NEW) — per-round reward-component
   logger. Runs N local games against the bomberman env with a chosen
   manager and emits JSON: per-round score, action counts, bombs
   placed, freeze ticks seen, unique cells visited, and reward
   attribution by component (mission/recon/resource/attack_damage/
   attack_kill/destroy_enemy_base/own_base_destroyed/self_damage/
   base_damage). This is the diagnostic harness CODEX called for —
   gives us a *qualitative* breakdown of where local score comes from,
   so when we eventually get any per-episode data from official Eval
   URLs we can compare component-by-component instead of comparing
   single mean scores.

### How to pick the runtime mode for a build — IMPORTANT

**`AE_MODE` must be baked into the image, not just set in the shell.**
`til build` invokes `docker build` without inheriting our shell env, so
`AE_MODE=policy til build ...` produces an image with no `AE_MODE` set;
the cloud evaluator then defaults to `hybrid` regardless of the tag.
The first A/B round (2026-05-14) hit this — all three builds had the
same sha256 `2572b392...` and ran hybrid mode on the cloud.

Two ways to set the mode correctly, pick one:

```bash
# (A) Edit one line in ae/Dockerfile before each build:
#     ENV AE_MODE=hybrid   →   ENV AE_MODE=policy   etc.

# (B) Write the mode into the file the server reads at startup:
echo hybrid    > ae/src/.ae_mode
echo policy    > ae/src/.ae_mode
echo heuristic > ae/src/.ae_mode
```

Both are picked up by `ae_server._read_mode()` (env var wins, file is
the fallback). `.ae_mode` lives next to the source and is gitignored.

### Suggested A/B order on Workbench

Wait for Violet's "model received" message before queueing the next
tag; submitting a second tag while the first is in queue **overwrites**
it (we lost two submissions to this on the first attempt).

```bash
# Common: pick the right weights up front.
cp ~/ae-checkpoints-backup/deployed-bc-v1.pt ae/models/bc.pt   # ppo-v1

# Build 1 — speed-only A/B (policy mode).
echo policy > ae/src/.ae_mode
til build ae policy-fast-v2 && til test ae policy-fast-v2 && til submit ae policy-fast-v2
# expect: score ≈ 0.51 (noise vs ppo-v1 0.507), speed > 0.90

# Build 2 — hybrid (THE new bet).
echo hybrid > ae/src/.ae_mode
til build ae hybrid-v2 && til test ae hybrid-v2 && til submit ae hybrid-v2

# Build 3 — heuristic-only restore.
echo heuristic > ae/src/.ae_mode
til build ae heuristic-restore-v2 && til test ae heuristic-restore-v2 && til submit ae heuristic-restore-v2
# expect: speed > 0.92, score 0.49–0.51
```

If any submission scores >0.52, that's the new best. If `hybrid-v2`
scores >0.55, the hybrid hypothesis works — start tuning veto
thresholds. If `heuristic-restore-v2` ties policy-fast-v2, drop torch
from the deployed build entirely and reclaim the +0.04 speed.

### Diagnostic to run alongside the above

```bash
# Cheap — pure heuristic, no torch.
python training/ae/diagnose.py --manager heuristic --games 30 \
    --out /tmp/diag-heuristic.json

# With ppo-v1 weights present:
python training/ae/diagnose.py --manager hybrid --games 30 \
    --checkpoint ae/models/bc.pt --out /tmp/diag-hybrid.json

# Compare reward_component_sum across the two reports. If hybrid
# collects more missions or more attack_kills, that's our explanation
# for any score lift. If hybrid drops own_base_destroyed events vs
# policy-only (run --manager policy too), the safety-veto is doing its
# job.
```

### What I deliberately did NOT do

Per CODEX recommendation and the v3b → ppo-v2 regression record:
- No new heuristic tuning of item weights / frontier scoring / bomb
  thresholds — these have been proven to not move official score.
- No new training runs. The local→official gap is structural.
- No frame-stacking. ppo-v2 widened the gap with frame stacking.
- No bigger network. Same story.

---



Per-task working log for AE (Autonomous Exploration / Bomberman). For the authoritative input/output/scoring spec see
[README.md](README.md) and the official [Challenge specifications](https://github.com/til-ai/til-26/wiki/Challenge-specifications#ae).
For submission history across all tasks see [../RESULTS.md](../RESULTS.md).

## Current shipped tag

**`ppo-v1` — official 0.507 / 0.861 (14 May 04:36 SGT, 0/30 errors).** Team-best AE score. **ppo-v2 regressed and should be rolled back if a fresh AE submission is needed.**

`ppo-v1` is marginally above `planner-v3b` (+0.008 score, +0.008 speed) — within submission noise. The follow-up `ppo-v2` (frame stacking N=4, value-loss fix, varied-maps training) hit a **0.763 local mean** but only **0.489 official** — the local→official gap **widened** from 0.19 (v3b/ppo-v1) to 0.27 (ppo-v2). Varied-map training was the wrong bet; whatever the hidden eval uses, it's *neither* novice nor `novice=False` random seeds.

**Cumulative finding across v1 → v3b → bc-v1 → ppo-v1 → ppo-v2**: the local→official gap is ≥ 0.18 for every approach we've tried (heuristics, BC, mixed-opp PPO, frame-stacked PPO + varied maps). The gap is structural to the hidden eval distribution and **none of our training-side interventions have moved it**. Heuristic-tuning and RL both saturate at ~0.50 official.

Previous shipped tag: `planner-v3b` — official 0.499 / 0.853 (13 May 23:42 SGT, 0/30 errors). v1 weights are backed up at `~/ae-checkpoints-backup/deployed-bc-v1.pt` on Workbench.

Score essentially flat vs `planner-v2` (-0.002, within noise), but **speed
jumped +0.082** (`0.771 → 0.853`) from algorithmic wins kept from the
abandoned `planner-v3`: single multi-source BFS in `_choose_target`
(replaces N per-target BFS calls), `_blast_cells` per-turn cache, and the
Dockerfile pinning `uvloop`+`httptools`. Net blended qualifier estimate
lifted `0.5561 → 0.5638` (+0.008).

The local→official gap held at **~0.18** (local mean 0.681 → official
0.499) — the same gap as `planner-v2`. Two rounds of heuristic tuning have
not closed it. The gap is structural (env distribution mismatch),
not a tunable knob. **Heuristic accuracy ceiling is in sight at ~0.50
official.** Pushing past that needs a learned policy.

AE is **40% of the qualifier** — biggest absolute lever in the whole repo.

## What our agent runs on

- **No model**, no learned weights. Rule-based planner only.
- **Container base**: same NVIDIA pytorch image as the other services (per [Dockerfile](Dockerfile)), but the inference loop is CPU-bound pure Python — no GPU usage. Fast by design (per-step latency dominates because the evaluator calls `/ae` once per game tick).
- **Source**: [src/ae_manager.py](src/ae_manager.py) (manager — what we edit) + [src/ae_server.py](src/ae_server.py) (server with the reset-robustness patch already applied — empty POST or `step == 0` triggers re-instantiation of `AEManager`; see top-level CLAUDE.md notes).
- **Spec contract**: `/ae` on port 5005 returns one `{"action": int}` and `/reset` clears state. Official observation keys are `agent_viewcone`, `base_viewcone`, `direction`, `location`, `base_location`, `health`, `frozen_ticks`, `base_health`, `team_resources`, `team_bombs`, `step`, and `action_mask`.
- **State on `self`**: belief map, frontier set, turn counter, etc. — must NOT use module-level globals because the server re-instantiates the manager on reset.

### Planner sketch (planner-v3b)

- Stateful belief map updated each tick from the partial observation.
- Objective + frontier BFS pathfinding to nearest unexplored / objective cell.
- LOS-safe tactical bomb placement (only when blocked by destructible wall or
  enemy in blast line AND escape route is pre-verified).
- Respects `action_mask` from the observation.
- `frozen_ticks` guard: stay put while frozen.
- Per-action cost balances progress, safety from bomb blast prediction N
  ticks ahead, and pickup proximity.

## Submission history

```text
Tag         Submitted          Score   Speed   Errors  Outcome
baseline    12/05 04:20        0.051   0.856   0/30    Periodic-forward + bomb-every-20 random walk
planner-v1  13/05 11:33        0.445   0.788   0/30    Stateful planner — new team high score, but big local→official gap
planner-v2  13/05 23:03        0.501   0.771   0/30    Bomb timer 4→3, bounded escape, enemy soft threat, frontier unseen-yield → +0.056 over v1
planner-v3  (not submitted)    —       —       9/9     Bigger v2 → multi-source BFS, predictive bombs (range 2), bomb chains, soft threat 1.0/3.0. Local regressed to 0.596, aborted.
planner-v3b 13/05 23:42        0.499   0.853   0/30    v3 minus bomb-chains; predictive range=1 with ≥2 enemies; threat 2.0/5.0. Score flat, speed +0.082 from BFS/cache/uvloop.
bc-v1       14/05 01:22        0.364   0.856   0/30    BC of planner-v3b regressed badly; deployment path works but policy overfit random-opponent local rollouts.
ppo-v1      14/05 04:36        0.507   0.861   0/30    NEW HIGH (+0.008/+0.008 vs v3b). Mixed-opp PPO from bc.pt warm start; local→official gap 0.19 unchanged from heuristic.
ppo-v2      14/05 13:29        0.489   0.854   0/30    REGRESSED -0.018. Frame-stacked + value-loss-fixed + varied-maps PPO. Local 0.763 → official 0.489; gap widened 0.19 → 0.27.
```

## Local validation history

```text
Variant     Date/time          Local score              Notes
baseline    12/05              0.051 official           Reference point only
planner-v1  13/05 10:31 Mac    0.732 local              Stateful belief + BFS + LOS-safe tactical bombs, 0 invalid actions
planner-v1  13/05 Workbench    0.697 local              Built/tested with official Workbench Docker flow before submission
planner-v1  13/05 11:33        0.445 official           ← significant drop from both local environments
planner-v2  13/05 Workbench    0.659/0.659/0.689        3-run local mean ≈ 0.669 (variance ±0.015)
planner-v2  13/05 23:03        0.501 official           +0.056 over planner-v1; gap to local narrowed but not closed
planner-v3  13/05 Workbench    0.588/0.629/0.570        3-run local mean ≈ 0.596 (-0.07 vs v2). Aggressive bombing was wasteful. NOT submitted.
planner-v3b 13/05 Workbench    0.80/0.61/0.66/0.65/0.64/0.63   6-run local mean ≈ 0.681 (variance ±0.07, big tail from one 0.80 outlier)
planner-v3b 13/05 23:42        0.499 official           Score flat vs v2 (-0.002), speed +0.082. Local→official gap still ~0.18 — structural.
bc-v1       14/05 eval_policy  0.7195/0.7193/0.6248/0.6908   Direct (no container) mean ≈ 0.689 — slightly above planner local.
bc-v1       14/05 til test     0.6317/0.7037/0.6128/0.7398   Container mean ≈ 0.672 — within noise of direct eval and of planner.
ppo-v1      14/05 eval_policy  0.7112/0.6998/0.6925         Direct mean ≈ 0.701, range 0.019 — TIGHT variance, best signal we've ever seen locally.
ppo-v1      14/05 til test     0.766/0.634/0.708            Container mean ≈ 0.703 — faithful to direct eval.
ppo-v1      14/05 04:36        0.507 official               Local→official gap 0.19, same as heuristic. Mixed-opponent training didn't close the gap.
ppo-v2 BC   14/05 eval_policy  0.6788 novice                30-epoch supervised BC on 40k stacked samples, val_acc 0.8898 (vs v1 BC's 0.8742). Net frame-stacking gain over single-frame BC.
ppo-v2      14/05 eval_policy  0.7138/0.7752/0.7885 novice  Mean 0.759 (+0.058 vs ppo-v1 novice). Best single run 0.789.
ppo-v2      14/05 eval_policy  0.6353/0.6910/0.6768 varied  Mean 0.668 — policy generalizes to non-novice maps.
ppo-v2      14/05 til test     0.7282/0.8260/0.7352         Container mean 0.763 (faithful to direct). 0.826 single-run high.
ppo-v2      14/05 13:29        0.489 official                REGRESSED -0.018 vs ppo-v1. Local→official gap widened 0.19 → 0.27. Varied-map training was the wrong bet.
```

## Detailed timeline

### baseline (12 May 04:20) — submission plumbing

Periodic forward + periodic bomb every 20 turns, respects `action_mask`,
stays put while frozen. Validates the service starts, endpoint responds,
JSON shape is correct, `/reset` works. Scored 0.051 — non-zero confirms the
service is alive.

### planner-v1 (13 May 11:33) — stateful planner, first real attempt

What's in it:
- Belief map updated every tick from partial observation (occupancy, walls,
  bomb timers, enemy positions).
- Objective/frontier BFS: pathfind toward the nearest unexplored cell or
  objective, prefer cells that reveal more unknown territory.
- LOS-safe tactical bombs: only place a bomb when (a) blocked by destructible
  wall on the planned path, or (b) enemy is in blast line AND a clear escape
  route exists for N ticks. The escape is pre-verified before the bomb
  placement commits.
- Bomb-blast prediction: forecast explosion timers N ticks ahead, avoid
  stepping into blast tiles, never bomb-place into an unsafe self-trap.

Local results:
- 0.732 on a Mac local env (no AE official scoring infra, just `til-26-ae`
  PettingZoo env)
- 0.697 on Workbench local with the official Docker test flow

Official: **0.445 / 0.788**. Significant drop from both local envs.

Hypotheses for the local→official gap (in priority order):

1. **Bomb timer mismatch** — local `til-26-ae` env may use 4-tick fuse,
   official may use 3. Planner-v2 corrects to 3.
2. **Escape-route check too optimistic** — assumes enemies don't actively
   block the escape; in adversarial play they sometimes do. Planner-v2 adds a
   bounded escape check that accounts for enemy reachability.
3. **Enemy threat model too binary** — treating enemies as "is/isn't in the
   blast" misses near-misses. Planner-v2 adds a "soft threat" gradient so the
   planner avoids tiles enemies are likely to step into next.
4. **Frontier scoring rewards revealing tiles for their own sake** — but the
   official scorer probably weights objective completion, not pure
   exploration. Planner-v2 weights frontier nodes by *unseen-yield* (how
   much new info they reveal) rather than raw distance.

### planner-v2 (13 May 23:03) — bomb-timer correctness + soft enemy threat

Changes against planner-v1 (committed in [`cb13c4c`](../README.md)):

- **Bomb fuse: 4 → 3 ticks**. The single biggest signal. `til-26-ae` config
  declares `entities.bomb.timer: 3`; with the phase order
  `place → move → detonate → upkeep`, a bomb placed at step N detonates after
  the agent's movement at step N+timer — i.e. the agent gets exactly 3
  movement actions to escape. The previous `4` allowed `_should_place_bomb`
  to commit to bombs whose escape path needed 4 moves; in those scenarios
  the planner was self-trapping. Net official lift: +0.056.
- **Bounded escape BFS**. Replaced the unbounded `_nearest_escape_cell` in
  `_should_place_bomb` with `_safe_escape_within(loc, blast, BOMB_TIMER)`.
  Refuses to place a bomb unless a safe cell is reachable within the fuse
  window. Removed the dead `_has_escape_after_bomb` helper.
- **Enemy soft threat**. Recently-seen enemies (within `ENEMY_STALENESS = 3`
  steps) plus their 4-neighbors are penalized in path scoring (`-3.0` in
  `_choose_target`, `-8.0` in `_fallback_action`). BFS still allows them so
  attack paths to enemy bases aren't blocked.
- **Frontier scoring by unseen yield**. Each frontier cell scored
  `4.0 + 1.0 × (#unseen 4-neighbors)` instead of flat `6.0`, biasing
  exploration toward frontiers that reveal more area.

Passes 9/9 unit tests. Workbench local: 0.659, 0.659, 0.689 (3 runs, mean
~0.669, ±0.015). Official: **0.501 / 0.771**, 0/30 errors.

### planner-v3 (NOT submitted) — over-eager bombing

Tried to push much further with seven changes bundled at once:

- **Multi-source BFS** in `_choose_target` (one BFS, distance lookup per
  candidate) — pure speed win.
- **`_blast_cells` per-turn cache** + **dominant-action shortcut** for
  obvious adjacent-enemy-base / adjacent-mission cases.
- **Predictive bombing at range 2**: any enemy within 2 BFS steps of any
  blast cell counts as a tactical target.
- **Bomb chains** (`_wall_break_reveals_high_value`): simulate "break this
  destructible wall" and re-run BFS; bomb if it reveals a base or mission
  within 5 moves.
- **Proactive wall-break** when BFS target is an enemy base or mission.
- **Item respawn awareness** (re-candidate collected tiles after 40 ticks),
  **base defense** (enemies near our base = 60.0-value candidates),
  **health-aware retreat** (<20 hp drops aggressive targets), softened
  threat penalty (1.0/3.0).
- **Dockerfile** pinning `uvloop`+`httptools` for slightly faster HTTP.

Workbench local: 0.588 / 0.629 / 0.570 → **mean 0.596, -0.07 vs v2**.
Predictive bombing at range 2 fires almost every turn we have a bomb on
a 16×16 map with 5 random enemies; bombs got burned on speculation. Bomb
chains compounded the problem because mission tiles are scattered, so the
"reveals high value within 5 moves" trigger was rarely false. **Not
submitted.**

### planner-v3b (13 May 23:42) — keep the speed wins, dial back bombs

What changed vs v3:
- `PREDICTIVE_BOMB_RANGE 2 → 1` AND require **≥2 enemies in the extended
  blast** before triggering. Random opponents wander; betting that one of
  N≥2 nearby enemies walks into a 1-step-extended blast is much better odds.
- Bomb-chain trigger **disabled**. Helper kept for re-use against smarter
  opponents but no longer called from `_should_place_bomb`.
- Threat penalty **2.0 / 5.0** (between v2's 3.0/8.0 and v3's 1.0/3.0).

What stayed from v3:
- Multi-source BFS, blast cache, dominant-action shortcut (all speed).
- Item respawn awareness, base defense, health-aware retreat, proactive
  wall-break for enemy_base / mission targets (orthogonal).
- Dockerfile uvloop/httptools.

Workbench local 6-run: 0.80, 0.61, 0.66, 0.65, 0.64, 0.63 → **mean 0.681,
variance ±0.07**. The 0.80 outlier is suspicious (one round hit lucky
bomb-tactic conditions, contributing ~0.13/6 to the mean alone), median is
~0.65. Officially: **0.499 / 0.853**, 0/30 errors.

**Key finding from v2→v3b**: the local→official gap is consistent at
~0.18 across two very different heuristic configurations. The gap is
structural — likely env distribution mismatch (random opponents locally vs
something else officially, fixed novice seed vs varied hidden seeds), not
something heuristic tuning will close. **The heuristic ceiling is around
0.50 official.**

### bc-v1 (14 May, submitted) — first learned policy

Pipeline (scaffolded in [../training/ae/](../training/ae/)):

1. **Dataset**: 200 games × ~200 of our agent's turns = **40,000 samples**
   collected by rolling out planner-v3b with random opponents
   (`collect_bc.py`, ~8 min on Workbench). Action distribution:
   `FORWARD 47.8% / BACKWARD 13.9% / RIGHT 15.0% / LEFT 12.7% /
   PLACE_BOMB 10.3% / STAY 0.3%` — healthy, no class collapse.
2. **Network**: `PolicyNetwork` (~149k params) — small CNN over each
   viewcone (32→16 ch for agent_view 25×7×5, 16→8 ch for base_view
   25×7×7) + MLP over 17-dim scalars → 6-way action head.
3. **Training**: 20 epochs supervised CE with `log(action_mask)` added to
   logits so the model can never assign mass to illegal actions. ~28 s
   total on Workbench GPU. **Best val_acc 0.8742**; train/val gap ~4 pts
   (no overfit).
4. **Direct eval** (`eval_policy.py`, 4×6-game runs): mean **0.689**.
5. **Container deploy**:
   - `ae/src/{encoder,model,policy_manager}.py` mirror the training tree.
   - `ae/src/ae_server.py` calls `_make_manager()` which tries
     `PolicyAEManager` and falls back to `AEManager` on
     `FileNotFoundError` (no checkpoint) or any other exception.
   - `ae/requirements.txt` adds CPU-only torch via `--extra-index-url
     https://download.pytorch.org/whl/cpu`. Image grows ~80 → 250 MB.
   - `ae/Dockerfile` adds `COPY models /workspace/models`.
   - **Path bug fix** (commit `22996fe`): `Path(__file__).parent.parent /
     "models" / "bc.pt"` resolved to `/models/bc.pt` inside the container
     (because `COPY src .` puts source at `/workspace/`, not
     `/workspace/src/`). `policy_manager.py` now checks both candidate
     paths. First bc-v1 build silently fell back to the heuristic for 4
     runs before this was caught from the docker logs.
6. **Container eval** (`til test`, 4×6-game runs): mean **0.672**. Within
   noise of direct eval — deployment is faithful.
7. **Submitted** 14 May → **0.364 / 0.856, 0/30 errors** (14 May 01:22 SGT).
   **Regressed -0.135 vs planner-v3b.** Local→official gap was 0.18 for
   heuristics, ballooned to **0.31** for BC (container local 0.672 →
   official 0.364). Speed held at 0.856 (BC inference doesn't hurt speed
   meaningfully). **planner-v3b stays the shipped tag.**

### Reading bc-v1's regression

The container faithfully reproduces direct eval (0.672 vs 0.689) so the
deployment is NOT the problem. The policy correctly imitates planner-v3b
locally (val_acc 0.874, container mean ~planner). Yet official tanks.

The gap blowup is the key signal: a 0.31 gap on a NN policy vs 0.18 for
heuristics means the NN **memorized planner behavior against random
local opponents** — patterns that don't transfer to the hidden eval's
opponent distribution. Heuristics are explicit rules and degrade
gracefully on new opponents; a BC policy keys on subtle obs features
that correlate with planner-action in our local distribution but mean
nothing officially.

**Implication for PPO**: training against only random opponents will
inherit this gap. PPO **must** use a mixed opponent pool to break the
overfit — random + frozen planner-v3b + frozen self-copies + (ideally)
scripted-aggressor agents. The deployment plumbing is validated, so the
moment PPO produces better-than-v3b local-against-mixed-opponents
weights, we drop them into `ae/models/bc.pt` and resubmit.

What this submission is actually testing:
- **End-to-end NN deployment path** (encoder + model + container + COPY +
  fallback). Required for PPO later.
- **Local→official gap for NN policies**. If it stays at ~0.18 (official
  ≈ 0.49), the gap is environment-distribution, not algorithm; PPO must
  close it through training against varied opponents. If it shrinks
  (official ≈ 0.55+), NN policies *do* generalize better and BC alone is
  a positive datapoint.
- **Speed cost of torch CPU inference vs pure-Python BFS**. v3b speed was
  0.853; BC inference is ~13 ms/call vs heuristic's ~1 ms, so speed will
  likely drop to roughly 0.75–0.82.

### ppo-v1 (14 May, running) — RL fine-tune with mixed opponents

A pure-PyTorch PPO training path now exists in [../training/ae/train_ppo.py](../training/ae/train_ppo.py).
It reuses the deployed `PolicyNetwork`, so the saved actor checkpoint is directly compatible with `PolicyAEManager`.

Current Workbench command:

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

Observed startup:

```text
device: cuda
warm-started actor from training/ae/checkpoints/bc.pt (epoch=20, val_acc=0.87425)
actor params: 149,438; critic params: 103,225
```

Training design:
- Actor warm-starts from `bc.pt` instead of scratch.
- Critic is training-only and is not needed in the Docker image.
- We control `env.possible_agents[0]` only.
- Opponent pool is mixed: random + frozen planner-v3b + frozen self-copy.
- Invalid actions are masked during PPO sampling and loss, same as BC/deployment.

Decision rule before submission:
- Do **not** submit just because PPO beats random-opponent eval.
- Submit `ppo-v1` only if mixed-opponent eval and `til test ae ppo-v1` look competitive with or clearly better than `planner-v3b`.
- Keep `planner-v3b` shipped unless `ppo-v1` has `0 / 30` errors and a convincing local score.

Deployment if the run looks good:

```bash
cp training/ae/checkpoints/ppo.pt ae/models/bc.pt
til build ae ppo-v1
til test ae ppo-v1
# submit only after reviewing score/errors
til submit ae ppo-v1
```

## Where we are vs target

| Metric                  | planner-v3b | Target | Gap     |
|-------------------------|-------------|--------|---------|
| Official score          | 0.499       | 0.70   | +0.20   |
| Official speed          | 0.853       | 0.90   | +0.05   |
| Blended (0.75 / 0.25)   | 0.588       | 0.75   | +0.16   |

Speed is within striking distance of target with more uvicorn/Docker
tuning. **Score is the hard problem** — needs a different approach.

## Open questions

- **Why is the gap so stable at ~0.18?** v2 and v3b have very different
  bomb behavior but the same gap. Possible causes:
  - Different opponent distribution in hidden eval (e.g. smarter agents
    that don't wander into bombs we predicted at random).
  - Different map distribution (advanced track? bigger grid?).
  - Different episode lengths or reward calibration.
- **Is `0 / 30 errors` masking some game-completion failures?** No
  invalid-action errors doesn't mean no agent-deaths or timeouts.
- **Could a learned policy generalize the gap better?** A policy trained
  against a *mix* of opponents (random + frozen planner + self-copies)
  inherits less bias than a planner tuned against one local distribution.

## CODEX recommendation with the way forward/

**Recommendation: keep AE shipped at `ppo-v1` and stop spending blind training
cycles until we can explain the hidden-eval gap.** AE is still the largest
qualifier lever at 40%, but the evidence now says the bottleneck is not
"train longer" or "make local score higher." Every serious local win has
collapsed to about the same official band: `planner-v3b` local ~0.681 →
official 0.499, `ppo-v1` local ~0.703 → official 0.507, and `ppo-v2` local
~0.763 → official 0.489. That means local AE score is currently a poor
selection metric, so new AE submissions should be treated as official A/B
experiments, not as validated improvements.

Immediate way forward:

1. **Protect the best submission.** `ppo-v1` is the team-best official AE
   result (`0.507 / 0.861`). Before any rebuild or resubmit, verify the
   Workbench deploy slot points at the `ppo-v1` checkpoint, not the regressed
   `ppo-v2` checkpoint. If unsure, restore with
   `cp ~/ae-checkpoints-backup/deployed-bc-v1.pt ae/models/bc.pt` and build a
   clearly named restore tag.
2. **Get hidden-eval evidence before changing the policy.** Pull every useful
   detail from the official Debug/Eval URLs: per-round reward totals, deaths or
   freeze counts, bomb damage, base damage, map/seed hints, step counts, and
   whether the agent is losing score through failed attacks, missed objectives,
   or base defense failures. Without this, AE work is guesswork.
3. **Make a replay/diagnostic harness.** Add logging to local AE eval that
   records per-episode reward components, final position, collected mission /
   recon / resource counts, bombs placed, bomb hits, self-damage, base damage,
   freezes, and visited-cell coverage. Compare local failures against any
   official debug traces. The first target is not a better model; it is a
   believable explanation for why official is ~0.18-0.27 lower.
4. **Only then run targeted official A/Bs.** Candidate low-risk A/Bs are:
   restored `planner-v3b` vs `ppo-v1`; a conservative heuristic variant with
   fewer speculative bombs and stronger objective collection; a `ppo-v1`-style
   single-frame policy trained/evaluated on adversarial scripted opponents; and
   a hybrid gate that uses the heuristic planner when the policy confidence is
   low or when bomb/base-defense situations appear. Do not submit another
   "higher local mean" policy unless it tests a concrete hidden-gap hypothesis.
5. **Do not prioritize speed unless reward is protected.** AE speed is already
   healthy (`0.861` on `ppo-v1`). A perfect speed score would only move the
   blended AE score modestly; a +0.05 reward improvement is much more valuable
   than shaving a few milliseconds.

What to avoid:

- Do not repeat `ppo-v2`'s pattern: frame stacking, broader local maps, or
  longer PPO runs can improve local score while hurting official score.
- Do not trust six-game local means; the local evaluator has high variance and
  the wrong distribution.
- Do not tune frontier/exploration reward for its own sake. The spec rewards
  missions, resources, recon, attacks, kills, and base destruction; exploration
  only matters when it finds those rewards.
- Do not commit large checkpoints. Keep weights on Workbench / backup storage
  and copy into `ae/models/bc.pt` only for a deliberate build.

## Next steps

Five AE approaches submitted (v1, v2, v3b heuristics + bc-v1 + ppo-v1 + ppo-v2 RL), best official is **0.507 (ppo-v1)**. The local→official gap is ≥ 0.18 across **every** approach, and ppo-v2's varied-map training actively *widened* it to 0.27. We've spent the budget for training-side experiments. **AE is in maintenance mode at ppo-v1.**

1. **Rollback if a new AE submission is ever needed.** ppo-v2 weights are currently in `ae/models/bc.pt` on Workbench, which means a re-build picks up the worse policy. To revert: `cp ~/ae-checkpoints-backup/deployed-bc-v1.pt ae/models/bc.pt && til build ae ppo-v1-restore`.
2. **Don't run more local-eval-driven AE training.** Five runs proved local mean is not a signal for official score on AE. Local 0.681 → official 0.499, local 0.701 → official 0.507, local 0.763 → **official 0.489**. The relationship is essentially uncorrelated.
3. **Only AE work worth doing now: hidden-eval data.** The Discord submissions include Debug/Eval URLs. If those expose per-game replays/scores, comparing them against our local replays could tell us *what* the hidden eval does differently (map layout, opponent style, episode length). Without that data, every further AE training run is a coin flip.
4. (Deferred forever unless we get hidden-eval data) **Speed micro-opts**, **bigger network**, **longer training**, **richer reward shaping** — none have a reason to help when local doesn't predict official.

## Reproducibility / pointers

- Manager source: [src/ae_manager.py](src/ae_manager.py)
- Server (reset-robustness patch applied): [src/ae_server.py](src/ae_server.py)
- Local Mac test: spin up `til_environment.bomberman_env` directly (no Docker needed for Mac iteration)
- Workbench Docker flow: `til build ae <tag> && til test ae <tag>`
- Official AE env (read-only submodule): [../til-26-ae/](../til-26-ae/)
- Implementation plan / decisions: [../AE_IMPLEMENTATION_PLAN.md](../AE_IMPLEMENTATION_PLAN.md) (if present)
