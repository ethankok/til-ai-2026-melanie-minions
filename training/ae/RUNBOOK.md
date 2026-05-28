# AE training pipeline + Tier-1/Tier-2 runbook

This directory holds scripts for training a learned AE policy and for the
Tier-1/Tier-2 heuristic-augmentation workstream targeting 0.8 score. The
current shipped agent ([../../ae/src/ae_manager.py](../../ae/src/ae_manager.py))
is a hand-coded BFS planner; the learned-policy path imitates it via behavior
cloning then PPO-fine-tunes against mixed opponents.

Official AE spec: <https://github.com/til-ai/til-26/wiki/Challenge-specifications#ae>

> ### ⚠️ Cloud-eval variance (measured 28 May 2026)
>
> Cloud-eval is NOT deterministic on identical image bytes. Variance-farming
> `conf-hybrid-v4` (n=6, same image) yielded **σ ≈ 0.053** on the accuracy
> term (`0.491, 0.508, 0.511, 0.566, 0.605, 0.615` → mean 0.549). Speed
> term σ ≈ 0.004 (stable). 95% CI on a single sample is **±0.10**.
>
> **Implications for any cloud-based decision:**
> - A single submission's score has ±0.10 error bars. Don't promote on n=1
>   unless Δ > 0.10 over the active baseline.
> - For any "is candidate X better than baseline Y?" question on cloud,
>   variance-farm both sides with n≥5 and compare means. Set the
>   promotion bar at **Δ_mean ≥ 0.05 with overlapping CIs disjoint**.
> - All prior single-shot calibration comparisons (local vs cloud, ±0.02
>   "lifts" in the leaderboard log) are inside noise. Treat them as
>   directional, not authoritative.
>
> Variance-farm recipe (same image, fresh tags so the eval queue treats
> each as a new run; build is cache-hit-instant after the first):
>
> ```bash
> BASE=<your-base-tag>
> for i in $(seq 1 5); do
>   TAG="${BASE}-vf${i}"
>   til build  ae "${TAG}"   # cache-hit; near-instant
>   til submit ae "${TAG}"
>   [ "$i" -lt 5 ] && sleep 210   # space out the eval queue
> done
> ```

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
training/ae/checkpoints/ppo-full-rl-v1.pt Full-RL PPO candidate         historical; cloud did not prove PPO lift
training/ae/checkpoints/option_policy.pt Option-selector BC checkpoint  semifinals option path
training/ae/checkpoints/option_policy_ppo.pt Option-selector PPO checkpoint gated candidate
training/ae/checkpoints/tactical_policy.pt Outcome-weighted tactical selector checkpoint
training/ae/checkpoints/tactical_policy_ppo.pt Planner-first 12-way macro PPO candidate
training/ae/data/knob-*.json             heuristic sweep summaries      local-only, no baked winner yet
training/ae/data/option-v2-*.json        option-v2 sweep summaries      local-only, rejected
```

---

## Pipeline overview

### File glossary

- `encoder.py` — observation dict → float32 tensors. Single source of truth for the obs/feature shape; reused by training and deployment.
- `model.py` — `PolicyNetwork`: small CNN over each viewcone + MLP over scalars. ~150k params, designed for CPU inference under ~5 ms/call.
- `collect_bc.py` — rolls out the planner in `til_environment.bomberman_env`, logs every `(obs, action)` pair to a compressed `.npz`.
- `train_bc.py` — supervised cross-entropy training of `PolicyNetwork` on the BC dataset. Masks illegal actions in both loss and argmax.
- `collect_option_bc.py` — logs planner intent labels (`escape`, `rush_base`, `base_bomb`, `defend_base`, `collect_mission`, `collect_resource`, `hunt_enemy`, `explore`) for the option selector.
- `train_option_bc.py` — supervised warm-start for the 8-way option selector.
- `train_option_ppo.py` — PPO fine-tune for option choice only; movement/bomb execution remains planner-backed through `OptionExecutor`.
- `collect_tactical_outcome.py` — full-game terminal-outcome collector for the 12-way tactical selector. It runs heuristic baseline and tactical exploration on the same seed/suite, then labels tactical choices by score delta versus baseline.
- `train_tactical_bc.py` — outcome-weighted BC trainer for the 12-way tactical selector.
- `train_tactical_ppo.py` — planner-first PPO for the 12-way tactical selector. The actor chooses tactical macros only; `TacticalExecutor` converts macros to safe raw actions. Each rollout is paired with a same-seed heuristic baseline and saved checkpoints include harm-aware transition matrices for runtime gates.
- `eval_policy.py` — runs a checkpoint against the env for N games, reports the same `score = total_reward / games / 1000` as `test/test_ae.py`.
- `train_ppo.py` — pure-PyTorch PPO fine-tune. Warm-starts from `bc.pt`/PPO checkpoints, trains agent 0 against scripted/cloudsuite/league/per-game mixed pools, saves a deployment-compatible actor checkpoint.
- `run_full_rl_v1.py` — Mac-first launcher for the current full-RL attempt. Wraps `train_ppo.py --preset full-rl` and streams logs to `training/ae/checkpoints/ppo-full-rl-v1.log`.
- `run_full_rl_belief_v1.py` — Three-stage belief-aware launcher (collect mixed-opp BC -> train belief BC -> PPO with `--preset full-rl --use-belief`). 23 May 2026 attempt did not clear the save floor; kept on disk as the template for any future memory-hypothesis ablation. See `ae/NOTES.md` "23 May session" for the negative-result analysis.
- `collect_bc.py` — supports `--opponents` (vocabulary mirrors `simulate.py`: `random` / `mixed` / `library` / `cloudsuite` / explicit comma list). Default `random` preserves the legacy planner-vs-random distribution; use `library` or `cloudsuite` for belief / future memory-aware experiments to avoid the original `bc-belief-hybrid` overfit cause.
- `simulate.py`, `opponents.py` — Tier 1+2 simulation harness with the scripted opponent library. Current serious gates should use `cloudsuite`, `pressure2`, the furnished proxy suites (`strong_realistic`, `base_rush_exploit`, `bracket_proxy`, `top_seed_proxy`), and/or `mixed`; random is only a weak packaging sanity check.
- `validate_cloud_suite.py` — suite gate wrapper with `legacy`, `furnished`, `bracket`, and `stress` presets, multi-seed support, weighted aggregate reporting, and decision-path diagnostics.
- `rank_opponents.py` — local scripted-opponent ranking harness; scores each archetype both as slot 0 and as a 5x pressure suite against our current agent.
- `compare_candidates.py` — paired checkpoint comparison harness for PPO artifacts.
- `sweep_heuristic_knobs.py` — Mac-first heuristic env-var sweep harness with `broad`, `focused`, and `bridge` candidate generators. The 24 May broad/focused/bridge sweeps found no promotable candidate.
- `sweep_option_v2.py` — controlled sweep for the opt-in `AE_PLANNER=option_v2` decision layer. The 24 May screen and top-3 advance gate rejected option-v2.
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
| PPO fine-tune             | implemented; old scripted/self-play/full-RL runs regressed once policy actually loaded; keep only for analysis |
| Option-selector RL        | implemented as opt-in `option_hybrid`; must beat furnished eval before any submission |
| Tactical macro RL         | implemented as opt-in `macro_hybrid` + `train_tactical_ppo.py`; smoke-tested only, needs real training/eval before submission |
| Tier 1+2 artifact builders| implemented |
| Deployment into `ae/src/` | implemented; current protected path is legacy heuristic / historical heuristic-fallback max |

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

The inference path in [../../ae/src/ae_server.py](../../ae/src/ae_server.py) supports four modes selected by `AE_MODE` (env var) or `ae/src/.ae_mode` (file fallback):

- `hybrid` (current Dockerfile default, matching the `fixed-map-v5` build source shape) — policy chooses, heuristic vetoes illegal / unsafe-bomb / step-into-blast / frozen-stay actions. See [../../ae/src/hybrid_manager.py](../../ae/src/hybrid_manager.py).
- `option_hybrid` — 8-way option policy chooses the strategic intent; planner variants execute the option with legality and bomb-safety guardrails. Requires `AE_OPTION_POLICY_CHECKPOINT` or `ae/models/option_policy.pt`.
- `tactical_hybrid` — 12-way tactical option policy chooses sharper macros such as `intercept_base_threat`, `bomb_base_threat`, and `stall_when_winning`; planner execution remains legality/bomb-safety guarded. Requires `AE_TACTICAL_POLICY_CHECKPOINT` or `ae/models/tactical_policy.pt`.
- `macro_hybrid` — stricter planner-first 12-way macro policy for the "proper hybrid" path. Defaults the fallback planner to the calibrated `heuristic-C + bomb_cost=7.0` profile, tries top-k learned tactical macros, and only accepts deviations that pass confidence/support/harm-aware gates. Requires `AE_TACTICAL_POLICY_CHECKPOINT` or `ae/models/tactical_policy.pt`; falls back to the heuristic if missing.
- `policy` — pure `PolicyAEManager`.
- `heuristic` — pure rule-based `AEManager` (no torch needed in the image at all).

Current AE max-score high: `ppo-full-rl-v1-hybrid` scored `0.638 / 0.847`,
but 24 May forensic review showed this was not evidence for PPO. The deployed
runtime lacked the `legacy-small` policy class, policy load failed, and
`ae_server.py` silently fell back to plain `AEManager`. When legacy checkpoint
support was restored and the policy actually loaded, cloud scores dropped to
about `0.41`. Treat the `0.638` result as a protected leaderboard artifact
with heuristic-fallback provenance, not as proof that PPO/hybrid is better.
The best intentional heuristic cloud tag is `heuristic-A-vf1` at
`0.613 / 0.845`.

External Pandemonium plan note (26 May 2026): the user shared four downloaded
plan files for a user-reported `0.731` AE score. Three of the four are generic
CNN+MLP Stable-Baselines PPO guidance (10M Novice fixed-map steps, self-play,
random maps/ICM for Advanced). The useful detail is `pandemonium3.png`: it
describes "PPO + BFS rule-based fallback", with an immediate BFS manager path
and `/reset` sanity checks. Do not treat this as a new raw-PPO recipe. If
revisiting Pandemonium-style work, prioritize fallback arbitration rules,
fixed-map route tables, bomb/escape safety, and opponent curriculum, then test
through the furnished/bracket gates below.

Deploy a new policy checkpoint by copying it into the model slot:

```bash
mkdir -p ae/models
cp training/ae/checkpoints/<your>.pt ae/models/bc.pt
echo hybrid > ae/src/.ae_mode        # or edit ae/Dockerfile
til build ae <tag>                   # bakes AE_MODE into the image
til test ae <tag>
til submit ae <tag>
```

Keep `ae/models/bc.pt` as the expected filename unless you also set `AE_POLICY_CHECKPOINT`, because `policy_manager.py` searches for that path by default.

For option-hybrid deployment, copy to the option slot and select the mode:

```bash
mkdir -p ae/models
cp training/ae/checkpoints/<your-option>.pt ae/models/option_policy.pt
echo option_hybrid > ae/src/.ae_mode
til build ae <tag>
til test ae <tag>
til submit ae <tag>
```

For macro-hybrid deployment, copy a tactical checkpoint and select the new
mode. This path is intended for PPO-trained 12-way tactical checkpoints from
`train_tactical_ppo.py`, not raw-action PPO weights:

```bash
mkdir -p ae/models
cp training/ae/checkpoints/tactical_policy_ppo.pt ae/models/tactical_policy.pt
echo macro_hybrid > ae/src/.ae_mode

# Optional explicit gate overrides. Defaults match the training gate.
export AE_MACRO_BASELINE_PROFILE=combo_c_bomb7
export AE_MACRO_TOP_K=4
export AE_MACRO_DELTA_MARGIN=-0.01
export AE_TACTICAL_REQUIRE_DELTA_SUPPORT=1
export AE_TACTICAL_MIN_DELTA_SUPPORT=1
export AE_TACTICAL_MIN_ATTEMPTED=0
export AE_TACTICAL_MIN_POSITIVE_RATE=0.0
export AE_TACTICAL_MIN_NET_DELTA=-999.0
export AE_TACTICAL_ALLOWED_DELTA_OPTIONS=rush_enemy_base,bomb_enemy_base,collect_mission_safe,collect_resource_safe,counter_rush

til build ae macro-hybrid-v1
til test  ae macro-hybrid-v1
til submit ae macro-hybrid-v1
```

**Important**: `AE_MODE=foo til build …` does NOT work — `docker build` doesn't inherit the shell env, so the cloud container would default to hybrid regardless. Either edit the `ENV AE_MODE=…` line in `ae/Dockerfile` or write the mode into `ae/src/.ae_mode` (gitignored) before each build.

### 24 May local sweep harnesses and current AE decision

Use these harnesses only for local evidence gathering. The 24 May runs did not
produce a setting worth baking into Docker.

Heuristic sweep smoke:

```bash
.venv/bin/python training/ae/sweep_heuristic_knobs.py \
  --candidates 8 --rounds 1 --suites cloudsuite pressure2 \
  --jobs 4 --summary-out training/ae/data/knob-smoke.json
```

Heuristic bridge screen:

```bash
.venv/bin/python training/ae/sweep_heuristic_knobs.py \
  --mode bridge --candidates 240 --rounds 3 \
  --suites cloudsuite pressure2 \
  --jobs 8 --summary-out training/ae/data/knob-bridge-240.json
```

Promotion gate for named heuristic candidates:

```bash
.venv/bin/python training/ae/sweep_heuristic_knobs.py \
  --mode bridge --candidates 240 \
  --candidate-ids baseline bridge_0211 \
  --rounds 32 --suites library cloudsuite pressure2 mixed \
  --jobs 8 --summary-out training/ae/data/knob-bridge-final.json
```

Option-v2 controlled screen:

```bash
.venv/bin/python training/ae/sweep_option_v2.py \
  --candidates 96 --rounds 2 \
  --suites library cloudsuite pressure2 \
  --jobs 8 --summary-out training/ae/data/option-v2-controlled-96.json
```

Option-v2 advance gate:

```bash
.venv/bin/python training/ae/sweep_option_v2.py \
  --candidates 96 \
  --candidate-ids baseline option_grid_0030 option_grid_0070 option_grid_0086 \
  --rounds 6 --suites library cloudsuite pressure2 mixed \
  --jobs 8 --summary-out training/ae/data/option-v2-advance-top3.json
```

Decision from the 24 May evidence:

- Do not bake `bridge_0211`, `focus_0124`, `focus_0266`, `option_grid_0030`, `option_grid_0070`, or `option_grid_0086`.
- Keep `AE_PLANNER=option_v2` as opt-in experimental scaffolding only; default `AE_PLANNER=legacy` remains the shipping behavior.
- Random-only local score should not promote AE changes. Use paired `cloudsuite` + `pressure2` evidence and reject candidates that trade one pressure suite for the other.
- Future AE work should be a narrow legacy-manager structural patch around visible base-bomb pressure, not another broad knob sweep.

### Semifinals option-selector path

This is the current logical RL route for fixed Novice maps. Do not replace
Dijkstra globally; the manager already has an opt-in `AE_ASTAR_TIEBREAK` that
only changes tied parent reconstruction. Keep Dijkstra as the default for exact
weighted costs, and use the option policy to decide intent.

### Planner-first tactical macro PPO path

This is the current "proper hybrid" implementation. It does not train a raw
Bomberman action policy. The model chooses one of the 12 tactical macros from
`ae/src/tactical_policy.py`; `TacticalExecutor` handles pathing, action-mask
legality, bomb safety, own-base blast checks, and escape mechanics.

As of 27 May 2026, `train_tactical_ppo.py` gates checkpoints on the actual
deployed `macro_hybrid` wrapper, not the standalone policy/executor score.
Rollouts still explore sampled tactical macros to gather PPO signal and
transition evidence, but every eval/save pass runs the current in-memory actor
through `MacroHybridAEManager` with the cumulative harm-aware matrices that
will be written into the checkpoint. The old standalone score can be printed
with `--eval-ungated-policy`, but it is diagnostic only and must not decide
promotion.

Important correction from the first repaired run: a gate with
`support=8`, `attempted=20`, `positive_rate=0.50`, `mean_delta>=0`, and
`margin=0.02` deadlocked training. `wrapper_eval` stayed equal to heuristic
fallback with `accept_rate=0.000`, so PPO had no way to affect the promoted
metric. There was also a top-k bug: if the BC-warm-start policy ranked the
heuristic option first, the wrapper returned immediately and never tried
runner-up deviations. Executor-only admission was also tested and was too
broad. The current default candidate gate uses `top_k=4`, `margin=-0.01`,
requires at least one positive same-seed episode sample for the prior->option
transition, and only admits scoring/collection/counter-rush deltas
(`rush_enemy_base`, `bomb_enemy_base`, `collect_mission_safe`,
`collect_resource_safe`, `counter_rush`). The outer save gate below is still
the real protection.

The default fallback planner profile is the latest calibrated heuristic combo:

```bash
AE_ITEM_MISSION_VALUE=80
AE_ITEM_RESOURCE_VALUE=40
AE_ENEMY_BASE_VALUE=100
AE_DIJKSTRA_BOMB_COST=7.0
```

One-update smoke test:

```bash
.venv/bin/python training/ae/train_tactical_ppo.py \
  --updates 1 --games-per-update 1 --eval-every 1 --eval-games 1 \
  --device cpu \
  --out /tmp/tactical_macro_ppo_smoke.pt \
  --latest-out /tmp/tactical_macro_ppo_latest.pt \
  --save-floor -999 --min-eval-delta -999 --min-required-suite-score -999 \
  --opponent-suites cloudsuite --eval-suites cloudsuite \
  --batch-size 128 --ppo-epochs 1
```

Real training starter command:

```bash
PYTHONHASHSEED=0 .venv/bin/python training/ae/train_tactical_ppo.py \
  --bc-checkpoint training/ae/checkpoints/tactical_policy.pt \
  --out training/ae/checkpoints/tactical_policy_ppo.pt \
  --latest-out training/ae/checkpoints/tactical_policy_ppo_latest.pt \
  --baseline-profile combo_c_bomb7 \
  --updates 80 --games-per-update 8 --eval-every 10 --eval-games 8 \
  --opponent-suites base_rush_exploit top_seed_proxy bracket_proxy strong_realistic cloudsuite pressure2 defense_trap mixed \
  --eval-suites cloudsuite pressure2 strong_realistic base_rush_exploit bracket_proxy top_seed_proxy defense_trap mixed \
  --save-floor 0.285 --min-eval-delta 0.005 \
  --required-suites top_seed_proxy base_rush_exploit bracket_proxy \
  --min-required-suite-score 0.16
```

For debugging only, append `--eval-ungated-policy` to see the old standalone
macro-controller score next to `wrapper_eval`. A low ungated score is not a
reject by itself; promotion is based on `wrapper_eval`, `wrapper_delta`,
required-suite floors, and the post-training `multi_seed_eval.py` gate below.
Wrapper eval is intentionally skipped at update 1 for real PPO runs; add
`--eval-first` only for short smoke tests.

27 May diagnostic result: the repaired trainer completes, but the current
warm-started tactical policy is not yet useful. Broad executor-only admission
accepted too many deltas and collapsed score. A scoring-only allowed set
bounded acceptance, but still lost to fallback on the short MPS screen. Treat
`tactical_policy_ppo_latest.pt` as analysis-only unless a later run creates
`tactical_policy_ppo.pt` and then clears the full multi-seed gate below.

Evaluate the deployed wrapper with the produced checkpoint:

```bash
AE_TACTICAL_POLICY_CHECKPOINT=training/ae/checkpoints/tactical_policy_ppo.pt \
.venv/bin/python training/ae/multi_seed_eval.py \
  --rounds 6 --our macro_hybrid --preset furnished \
  --hash-seeds 0 1 2 3 4 --sim-seeds 42 \
  --summary-out training/ae/data/macro_hybrid_n5.json
```

Only consider cloud if `macro_hybrid` beats the direct
`heuristic-C + bomb_cost=7.0` gate. If it only beats the old baseline, it is
not enough.

Confidence-gated continuation:

```bash
PYTHONHASHSEED=0 .venv/bin/python training/ae/train_tactical_ppo.py \
  --bc-checkpoint training/ae/checkpoints/tactical_policy_pos_only.pt \
  --out training/ae/checkpoints/tactical_policy_conf_ppo_v3.pt \
  --latest-out training/ae/checkpoints/tactical_policy_conf_ppo_v3_latest.pt \
  --baseline-profile combo_c_bomb7 \
  --confidence-gated --eval-wrapper confidence_hybrid \
  --eval-multi-seed --eval-hash-seeds 0 1 2 \
  --eval-rounds-per-hash-seed 6 \
  --baseline-json training/ae/data/w3_2_C_bomb7_n5.json \
  --updates 80 --games-per-update 16 --eval-every 5 \
  --save-floor 0.260 --min-eval-delta 0.005
```

This path is for PPO runs that should only act on low-confidence heuristic
ticks, then save only when the candidate clears a multi-`PYTHONHASHSEED`
gate against the calibrated `C + bomb=7.0` baseline. If the multi-seed
subprocess fails, the trainer may still print a single-seed diagnostic, but
it must not save a checkpoint from that fallback path.

Collect supervised option labels against the furnished semifinal proxy suites:

```bash
PYTHONHASHSEED=0 .venv/bin/python training/ae/collect_option_bc.py \
  --games 400 \
  --suite-cycle strong_realistic base_rush_exploit bracket_proxy top_seed_proxy cloudsuite pressure2 \
  --out training/ae/data/option_bc_furnished.npz
```

Train the warm-start option selector:

```bash
.venv/bin/python training/ae/train_option_bc.py \
  --data training/ae/data/option_bc_furnished.npz \
  --out training/ae/checkpoints/option_policy.pt \
  --epochs 20
```

Fine-tune option choice with PPO:

```bash
PYTHONHASHSEED=0 .venv/bin/python training/ae/train_option_ppo.py \
  --bc-checkpoint training/ae/checkpoints/option_policy_defense.pt \
  --out training/ae/checkpoints/option_policy_ppo.pt \
  --latest-out training/ae/checkpoints/option_policy_ppo_latest.pt \
  --updates 50 --games-per-update 8 --eval-every 5 --eval-games 16 \
  --opponent-suites base_rush_exploit top_seed_proxy defense_trap bracket_proxy base_rush_exploit top_seed_proxy strong_realistic cloudsuite pressure2 \
  --eval-suites cloudsuite pressure2 strong_realistic base_rush_exploit bracket_proxy top_seed_proxy defense_trap mixed \
  --option-temperature 1.35 --option-epsilon 0.05 \
  --base-health-delta-coef 1.25 --base-survival-bonus 0.35 --base-destroyed-penalty 0.75 \
  --required-suites top_seed_proxy base_rush_exploit --min-required-suite-score 0.16 \
  --save-floor 0.285
```

If you did not collect the defense-weighted warm start, replace
`option_policy_defense.pt` with `option_policy.pt`. The PPO saver now uses the
weighted furnished eval, restores critic state on resume, keeps `latest` for
analysis, and only promotes `out` when the weak bracket-critical suites clear
the required-suite floor. The floor is intentionally above the current
heuristic furnished baseline (`0.2806` weighted in
`training/ae/data/ae-furnished-baseline-8r.json`); option PPO checkpoints below
that are analysis artifacts, not deployment candidates.

Gate it against the furnished eval before touching deploy artifacts:

```bash
AE_OPTION_POLICY_CHECKPOINT=training/ae/checkpoints/option_policy_ppo.pt \
PYTHONHASHSEED=0 .venv/bin/python training/ae/validate_cloud_suite.py \
  --preset furnished --rounds 12 --seeds 42 137 --our option_hybrid \
  --summary-out training/ae/data/option_hybrid_furnished_eval.json
```

Promotion rule: only consider submission if the weighted furnished mean clears
the current heuristic baseline by a visible margin and does not trade away
`base_rush_exploit`/`top_seed_proxy`.

### Semifinals tactical outcome path

This is the preferred current AE route after the 8-way option PPO failed to
beat the furnished heuristic baseline. It uses terminal score delta as the main
learning signal, but keeps the executor tactical enough that a learned policy
can express anti-rush behavior. BC here means behavior cloning: the network is
still a supervised selector trained on outcome-weighted labels, not an online
RL loop that directly maximizes the furnished eval.

Collect full-game outcome-weighted tactical data:

```bash
PYTHONHASHSEED=0 .venv/bin/python training/ae/collect_tactical_outcome.py \
  --games 400 \
  --suite-cycle base_rush_exploit top_seed_proxy defense_trap bracket_proxy base_rush_exploit top_seed_proxy strong_realistic cloudsuite pressure2 \
  --out training/ae/data/tactical_outcome_400.npz \
  2>&1 | tee training/ae/logs/tactical_outcome_collect_400.log
```

Train the tactical selector:

```bash
.venv/bin/python training/ae/train_tactical_bc.py \
  --data training/ae/data/tactical_outcome_400.npz \
  --out training/ae/checkpoints/tactical_policy.pt \
  --epochs 20 --num-workers 0 \
  2>&1 | tee training/ae/logs/tactical_bc_train.log
```

The runtime is conservative by default. `AE_MODE=tactical_hybrid` loads the
policy, but learned deviations only execute when the support/confidence gates
allow them. Leave the profile unset for a safe heuristic-equivalent shadow run.
For the current 400-game checkpoint, the best local gate found so far is
explicitly stricter than the `bracket` profile defaults:

```bash
AE_TACTICAL_PROFILE=bracket
AE_TACTICAL_DELTA_CONF=0.85
AE_TACTICAL_MIN_DELTA_SUPPORT=50
AE_TACTICAL_POLICY_CHECKPOINT=training/ae/checkpoints/tactical_policy.pt
```

Quick bracket-only probe:

```bash
AE_TACTICAL_PROFILE=bracket \
AE_TACTICAL_DELTA_CONF=0.85 \
AE_TACTICAL_MIN_DELTA_SUPPORT=50 \
AE_TACTICAL_POLICY_CHECKPOINT=training/ae/checkpoints/tactical_policy.pt \
PYTHONHASHSEED=0 .venv/bin/python training/ae/validate_cloud_suite.py \
  --suites bracket_proxy --rounds 12 --seeds 42 137 --our tactical_hybrid
```

Gate it before touching deploy artifacts:

```bash
AE_TACTICAL_PROFILE=bracket \
AE_TACTICAL_DELTA_CONF=0.85 \
AE_TACTICAL_MIN_DELTA_SUPPORT=50 \
AE_TACTICAL_POLICY_CHECKPOINT=training/ae/checkpoints/tactical_policy.pt \
PYTHONHASHSEED=0 .venv/bin/python training/ae/validate_cloud_suite.py \
  --preset furnished --rounds 12 --seeds 42 137 --our tactical_hybrid \
  --summary-out training/ae/data/gate_sweep_c085_s050_12r_2seed.json
```

Promotion rule for global deployment is still strict: weighted furnished mean
must clear the heuristic baseline by a visible margin, with no
`base_rush_exploit`/`top_seed_proxy` regression. Current evidence:

- Heuristic baseline, furnished 12-round x seeds 42/137:
  `training/ae/data/heuristic_furnished_12r_2seed.json`, weighted `0.2825`,
  worst-suite mean `0.1633`.
- 400-game checkpoint, broad `AE_TACTICAL_PROFILE=bracket` on furnished
  4-round seed-42: weighted `0.2339`; rejected.
- 400-game checkpoint with the explicit `0.85/50` gate above:
  `training/ae/data/gate_sweep_c085_s050_12r_2seed.json`, weighted `0.2932`,
  mean-of-means `0.3139`, median-of-medians `0.2880`, worst-suite mean
  `0.2305`. Suite means: `cloudsuite=0.4233`, `pressure2=0.2305`,
  `strong_realistic=0.2613`, `base_rush_exploit=0.2362`,
  `bracket_proxy=0.2664`, `top_seed_proxy=0.2850`,
  `defense_trap=0.5080`, `mixed=0.3009`.
- Stricter `0.95/50` and hand-filtered transition/distance gates
  (`bombonly`, `cr_bomb_guard`, `far_only`) were rejected because they either
  lost the bracket/cloud lift or dropped below heuristic.

The 800-game follow-up is useful analysis data, not the current candidate:

```bash
PYTHONHASHSEED=0 .venv/bin/python training/ae/collect_tactical_outcome.py \
  --games 800 \
  --suite-cycle base_rush_exploit top_seed_proxy bracket_proxy base_rush_exploit top_seed_proxy defense_trap bracket_proxy strong_realistic cloudsuite pressure2 mixed \
  --out training/ae/data/tactical_outcome_800_more.npz \
  2>&1 | tee training/ae/logs/tactical_outcome_collect_800_more.log

.venv/bin/python training/ae/train_tactical_bc.py \
  --data training/ae/data/tactical_outcome_800_more.npz \
  --out training/ae/checkpoints/tactical_policy_800_more.pt \
  --epochs 30 --num-workers 0 \
  2>&1 | tee training/ae/logs/tactical_bc_800_more_train.log
```

The 800-game run collected `172,000` samples and saved best checkpoint around
epoch 5. Under the old `0.85/50` gate it scored furnished 12-round x 2-seed
weighted `0.2593`, with `base_rush_exploit=0.1246`,
`bracket_proxy=0.1887`, and `top_seed_proxy=0.1180`. Raising support to `100`
on a 4-round x 2-seed screen restored the catastrophic suites
(`pressure2=0.2305`, `top_seed_proxy=0.2850`) and reached weighted `0.2798`,
but still did not beat the 400-game candidate. Diagnosis: current
`AE_TACTICAL_MIN_DELTA_SUPPORT` is a raw positive-support count, so doubling
data changes the meaning of the gate. The next implementation should record
attempted transition counts and gate on positive rate/net delta/harm rate,
possibly bucketed by distance or suite-like pressure features, before training
more BC.

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

Workbench caveat: `gcloud artifacts docker images list ... --include-tags`
failed for `svc-melanie-minions@til-ai-2026.iam.gserviceaccount.com` with
`artifactregistry.versions.list` denied. If provenance is needed later, use
local shell/log search, Docker history, old Discord/build logs, or ask someone
with Artifact Registry version-list permission.

Important training correction: `train_ppo.py` already defaults to Novice
fixed-map mode (`--novice` is true unless `--vary-maps` or `--no-novice` is
used). An explicit fixed-Novice command is still useful for a controlled rerun,
but not a first-ever Novice-map training attempt.

To restore the exact fixed-map-v3 training base on Workbench, use the helper
below. The source code/config is in git, but the `.pt` checkpoint is ignored by
git; it should come from `~/ae-checkpoints-backup/deployed-bc-v1.pt` or from the
original `ae-fixed-map-v3` Docker image. The helper copies it into both
`ae/models/bc.pt` for deployment and
`training/ae/checkpoints/fixed-map-v3-base.pt` for fine-tuning, then prints a
seed-88 fine-tune command with the checkpoint's actual frame-stack size.

```bash
cd /home/jupyter/til
git pull origin main
bash training/ae/restore_fixed_map_v3_base.sh
```

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
  --n-frames 1 \
  --seed 42 \
  --eval-seed 42 \
  2>&1 | tee ~/novice-fixed-v1.log
```

### PPO retry path: `qualifier-best` preset

If we intentionally retry PPO despite the older negative results, use the new
`qualifier-best` preset rather than the plain scripted rerun. It keeps the
lightweight in-repo PyTorch policy/deployment path, but borrows the useful
parts of public MaskablePPO/Pommerman recipes: action masking, progressive
opponent curriculum, adaptive exploration shaping, KL early stop, and
checkpoint selection on a pressure-weighted validation suite.

```bash
cd /home/jupyter/til
git pull origin main

tmux new -s ae-ppo-best

python training/ae/train_ppo.py \
  --preset qualifier-best \
  --bc-checkpoint ~/ae-checkpoints-backup/deployed-bc-v1.pt \
  --out ~/ae-checkpoints-backup/ppo-qualifier-best-v1.pt \
  --updates 300 \
  --n-frames 1 \
  --eval-every 5 \
  --seed 88 \
  --eval-seed 8800 \
  2>&1 | tee ~/ppo-qualifier-best-v1.log
```

The saved checkpoint is selected by weighted validation:

```text
0.20 * random + 0.30 * scripted + 0.50 * cloudsuite
```

Observed 22 May results:

| Tag | Local evidence | Cloud |
|---|---|---|
| `ppo-qualifier-best-v1` | best `epoch=65`, weighted eval `0.6547`, local Docker around `0.7345` | `0.610 / 0.857` |
| `ppo-qualifier-best-v2` | cloudsuite-focused continuation, weighted eval `0.6576`, local Docker `0.7185` | `0.598 / 0.845` |
| `ppo-qualifier-best-v3` | scripted/balanced continuation, weighted eval `0.6199` | not submitted |
| `ppo-qualifier-best-v4-balanced` | best `epoch=75`, weighted eval `0.68808125`, local Docker `0.7506667` | `0.602 / 0.846`, then duplicate `0.578 / 0.845` |

Do **not** submit only because the trainer's eval rises. These runs prove the
local proxy can climb while hidden eval stays below `fixed-map-v5`. First deploy
to `ae/models/bc.pt`, build a tag, run `til test ae <tag>`, and require a large
margin over the v1/v4 local profiles before spending another cloud submission.

### Belief-aware PPO launcher (`run_full_rl_belief_v1.py`)

Built on 23 May 2026 to test whether memory (the existing belief branch in
`model.py` + `encoder.py`) closes the local→cloud gap. Three stages, each
skippable:

```bash
.venv/bin/python -u training/ae/run_full_rl_belief_v1.py                 # all stages
.venv/bin/python -u training/ae/run_full_rl_belief_v1.py --skip-ppo      # BC only (preview)
.venv/bin/python -u training/ae/run_full_rl_belief_v1.py --skip-collect  # rerun BC + PPO
.venv/bin/python -u training/ae/run_full_rl_belief_v1.py --skip-collect --skip-bc  # PPO only
```

Outputs (all gitignored):

```text
training/ae/data/bc-belief-mixed-v1.npz
training/ae/checkpoints/bc-belief-mixed-v1.pt
training/ae/checkpoints/ppo-full-rl-belief-v1.pt          gated candidate only
training/ae/checkpoints/ppo-full-rl-belief-v1-latest.pt   latest evaluated, for analysis
training/ae/checkpoints/ppo-full-rl-belief-v1.log
```

23 May result: BC val_acc `0.9527` on 600-game `library` data (better than
the 15 May `bc-belief`'s `0.897`). PPO stopped early at update 200/240 with
weighted eval `0.4925` (parts random `0.7909`, scripted `0.5553`, cloudsuite
`0.2805`); save floor was `0.6236`. No candidate cleared the gate. The
comparison was confounded by warm-start (BC at `0.13` vs no-belief reference
at `~0.60`), belief overhead (`+20-30%` per-update wall-clock), and BC-vs-PPO
distribution shift. Conclusion is "not validatable in time", not
"memory hypothesis killed". See [../../ae/NOTES.md](../../ae/NOTES.md)
"23 May session" for the per-suite breakdown.

### `AE_DIJKSTRA_BOMB_COST` lever

Plumbed 23 May 2026 as an env var read once in `AEManager.__init__`,
propagated into the fixed-Novice detection block. Default `5.0` preserves
shipped behavior. Only affects the heuristic path (used by `AE_MODE=heuristic`
and the heuristic fallback inside `hybrid` when the policy is vetoed; pure
`policy` mode ignores it).

Mac sweep at 24 rounds × {random, library, cloudsuite}, deterministic seed=42:

| Cost | random | library | cloudsuite | mean-of-means |
|---|---:|---:|---:|---:|
| 5.0 (shipped) | 0.821 | 0.567 | 0.319 | 0.569 |
| 4.0 | 0.783 | 0.544 | 0.335 | 0.554 |
| 3.0 | 0.742 | 0.535 | 0.354 | 0.544 |

Direction is monotonic: lower cost trades random tempo for cloudsuite
pressure score. Aggregate falls because random regression dominates the
cloudsuite gain. Did not ship. The lever stays available for cloud A/B work
without code changes — set `ENV AE_DIJKSTRA_BOMB_COST=4.0` in `ae/Dockerfile`
and rebuild. A heuristic-mode A/B at `4.0` would be the cheapest comparison
to the `fixed-map-v5` lineage (cloud `0.630 / 0.858`).

### Full fixed-Novice RL launcher

For the next serious RL attempt, use the `full-rl` preset through the launcher
instead of hand-copying a long command. This keeps fixed Novice geometry but
rotates rollout seeds and stratifies each PPO update across a wider opponent
mix:

```text
10% random, 35% scripted, 35% cloudsuite,
5% planner, 5% aggressive, 10% league/self-play snapshots
```

Run it on the Mac, which benchmarked faster than Workbench for this rollout
loop (`~3:52` vs `~7:51` for the same 3-update v1 checkpoint benchmark):

```bash
cd /Users/ethankok/projects/TIL
.venv/bin/python -u training/ae/run_full_rl_v1.py
```

The launcher expects `training/ae/checkpoints/ppo-qualifier-best-v1.pt` and
writes:

```text
training/ae/checkpoints/ppo-full-rl-v1.pt         gated candidate only
training/ae/checkpoints/ppo-full-rl-v1-latest.pt  latest evaluated checkpoint
training/ae/checkpoints/ppo-full-rl-v1.log
```

Checkpoint files are gitignored; do not commit them. The main candidate file is
save-gated: checkpoint selection runs on CPU in pure `policy` mode, pins
`PYTHONHASHSEED=0` in the trainer subprocess, uses held-out
`random,scripted,cloudsuite` seeds, compares against the starting
`ppo-qualifier-best-v1.pt`, and also compares against
`ppo-qualifier-best-v4-balanced.pt` when that reference checkpoint is present.
It only writes `ppo-full-rl-v1.pt` if the trained model beats the strongest
baseline/reference by at least `0.015` on the same selection suite. The
`latest` file is for inspection/recovery, not automatic deployment.

Final 22 May result: the pure-policy run finished and saved the gated candidate
at epoch 230:

```text
training/ae/checkpoints/ppo-full-rl-v1.pt
sha256: 1f30da4ebbfa7bd8d6fd131df5d5dcb9ee9a103fb57b1092d77c9beb3c827b43
weighted eval: 0.7428583333333334
random: 0.916625
scripted: 0.6468333333333334
cloudsuite: 0.752
```

The final/latest epoch 240 checkpoint regressed to weighted eval `0.7096958333`,
so do not deploy `ppo-full-rl-v1-latest.pt`.

The first Workbench A/B tested the checkpoint in pure `policy` mode as
`ppo-full-rl-v1-policy`. It submitted cleanly three times: `0.550 / 0.847`,
`0.579 / 0.848`, then `0.625 / 0.848`. The next A/B used the same checkpoint
inside the hybrid wrapper with `AE_HYBRID_FIXED_MAP_SHORTCUT=0`, so PPO stayed
active on detected Novice maps; duplicate submits landed `0.564 / 0.843`,
`0.638 / 0.847`, `0.599 / 0.848`, and `0.552 / 0.841`. A final shortcut-on
A/B used `AE_HYBRID_FIXED_MAP_SHORTCUT=1`, which mostly returns the heuristic
before querying PPO on detected Novice maps; it landed `0.521 / 0.849`,
`0.637 / 0.845`, and `0.582 / 0.845`.

Conclusion: protect `ppo-full-rl-v1-hybrid` as the max-score high, but do not
read too much into the architecture comparison. The cloud variance within one
config is larger than the mean difference between pure policy, shortcut-off
hybrid, and shortcut-on hybrid.

Continue from the log, not from memory:

```bash
cd /Users/ethankok/projects/TIL
tail -n 80 training/ae/checkpoints/ppo-full-rl-v1.log
ls -lh training/ae/checkpoints/ppo-full-rl-v1.pt \
       training/ae/checkpoints/ppo-full-rl-v1-latest.pt
```

If `ppo-full-rl-v1.pt` does not exist after training completes, the run did not
produce a deployable candidate. Do not copy `ppo-full-rl-v1-latest.pt` into
`ae/models/bc.pt` except for explicit analysis.

If the gated candidate exists and the run is promising, copy
`ppo-full-rl-v1.pt` to Workbench manually or through the browser/download path,
then deploy a clearly tagged A/B. The default protected code path remains
shortcut-off hybrid:

```bash
cd /home/jupyter/til
git pull --ff-only
mkdir -p ae/models
# Skip this copy if the gated epoch-230 checkpoint is already present there.
cp /path/to/ppo-full-rl-v1.pt ae/models/bc.pt
rm -f ae/src/.ae_mode
grep -n "ENV AE_MODE" ae/Dockerfile                    # must print: ENV AE_MODE=hybrid
grep -n "ENV AE_HYBRID_FIXED_MAP_SHORTCUT" ae/Dockerfile # must print: ENV AE_HYBRID_FIXED_MAP_SHORTCUT=0
sha256sum ae/models/bc.pt                              # must start with 1f30da4e...
til build ae ppo-full-rl-v1-hybrid
til test ae ppo-full-rl-v1-hybrid
til submit ae ppo-full-rl-v1-hybrid
```

For an explicit shortcut-on variance A/B, flip only
`ENV AE_HYBRID_FIXED_MAP_SHORTCUT=1`, use a separate tag such as
`ppo-full-rl-v1-hybrid-shortcut`, and remember that PPO is mostly bypassed on
detected Novice maps in that mode.

Deployment risk here means mismatch risk: wrong checkpoint architecture or
`n_frames`, wrong mode baked into `.ae_mode`/Docker, copying the `latest` file
instead of the gated candidate, missing Docker dependency, or local proxy
overfit. The code can train correctly and still deploy a weaker hybrid if any of
those pieces are off, so always inspect checkpoint metadata and run `til test`
before submit.

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

## 6. Local A/B before any push/build/submission (Mac first)

For future AE behavior changes, run the furnished validation suite locally on
the Mac before pushing to Workbench, building an image, or submitting. This is
the first gate, not an optional afterthought. Launch it with `PYTHONHASHSEED=0`
so AEManager-bearing opponents are comparable across runs:

```bash
PYTHONHASHSEED=0 .venv/bin/python training/ae/validate_cloud_suite.py \
  --preset furnished \
  --rounds 24 \
  --seeds 42 137 \
  --our heuristic \
  --summary-out training/ae/data/ae-furnished-gate.json
```

`furnished` means `cloudsuite`, `pressure2`, `strong_realistic`,
`base_rush_exploit`, `bracket_proxy`, `top_seed_proxy`, `defense_trap`, and
`mixed`. The weighted aggregate intentionally puts most mass on pressure and
bracket-like suites. Promote only if the weighted mean improves, the
worst-suite mean does not collapse, and the decision counters explain the
change. Use the stress preset when testing base-survival patches:

```bash
PYTHONHASHSEED=0 .venv/bin/python training/ae/validate_cloud_suite.py \
  --preset stress \
  --rounds 24 \
  --seeds 42 137 \
  --our heuristic \
  --summary-out training/ae/data/ae-stress-gate.json
```

Before freezing a new furnished suite, rank the opponent archetypes:

```bash
PYTHONHASHSEED=0 .venv/bin/python training/ae/rank_opponents.py \
  --rounds 8 \
  --summary-out training/ae/data/opponent-rank-furnished.json
```

The ranker reports two signals: how well an archetype scores as slot 0, and
how much five copies suppress our current heuristic score. Prefer opponents
that are high on both axes when composing a match-realistic local suite.

The older legacy gate is still available for historical comparison:

```bash
PYTHONHASHSEED=0 python training/ae/validate_cloud_suite.py \
  --preset legacy \
  --rounds 24 \
  --our heuristic \
  --summary-out training/ae/data/ae-candidate-cloudsuite.json
```

Use `.venv/bin/python` if that is the active Mac environment for the checkout.
If this gate clearly fails, stop there: do not push the code, do not run
`til build`, and do not spend a cloud submission. Workbench `til test` still
matters as a packaging/sanity check, but it is too random-opponent-heavy to be
the behavioral judge.

`cloudsuite` means `rusher,hunter,bomber,defender,mixed`: it is intentionally
pressure-heavy and should expose base-defense / escape weaknesses that random
NPCs miss. The validator defaults to shipping Docker settings
(`AE_USE_PLAYBOOK=0`, `AE_USE_OPPONENT_MODEL=0`) unless explicitly overridden.
It also prints diagnostics: mean bombs placed, cells visited, final health/base
health, early-end rate, and the largest inferred reward components. These are
best-effort local attributions, but they are good enough to identify whether a
candidate is losing by self-damage, base damage, low coverage, or low attack
pressure.

The simulator now resets each round with `--seed + round_idx`, so A/Bs are more
repeatable. Prefer comparing candidates on the same `--rounds` and `--seed`.

Fixed-map route/strategy sweeps should use the local-only route harness before
any source change is promoted into Docker defaults:

```bash
# Fast harness check; proves the runner works.
.venv/bin/python training/ae/sweep_fixed_routes.py \
  --profiles smoke \
  --rounds 4 \
  --suites cloudsuite \
  --summary-out training/ae/data/fixed-route-suite-smoke.json

# Main cloudsuite search. This is the first useful ranking pass.
.venv/bin/python training/ae/sweep_fixed_routes.py \
  --profiles core \
  --rounds 8 \
  --suites cloudsuite \
  --summary-out training/ae/data/fixed-route-suite-core.json

# Promotion gate before Workbench build/submit.
.venv/bin/python training/ae/sweep_fixed_routes.py \
  --profiles core \
  --rounds 24 \
  --suites random library cloudsuite \
  --summary-out training/ae/data/fixed-route-suite-gate.json
```

The runner instantiates `AEManager` with fixed-Novice-map strategy profiles
such as `attack_cells_close`, `center_then_attack`, `base_leash`, and
`low_ammo_base_race`. It ranks profiles by `cloudsuite` by default and prints
the same base-failure / reward-component diagnostics as the validator. Treat it
as a search tool: if a profile only wins `cloudsuite` by noise, rerun with more
rounds and the same seed before changing production defaults.

For the current fixed-map candidate family, do **not** require the optional
playbook/opponent-model artifacts. The shipping Dockerfile keeps both disabled;
the extra pressure suite is mainly a regression screen before Docker build:

```bash
.venv/bin/python training/ae/simulate.py \
  --rounds 200 --opponents library --our heuristic --no-log-traj \
  --summary-out training/ae/data/ae-item-prior-strong-v1-library.json
```

For `ae-item-confidence-v1`, cloud beat the 0.5 recovery line despite a weak
`cloudsuite` mean, so treat this as advisory rather than a hard gate. For
`ae-item-prior-strong-v1`, stronger priors moved library up but random,
cloudsuite, and Docker down; do not blindly retry that canceled submission.
The 23:45 diagnostic pass rejected follow-up behavior probes, and the 21 May
base-defense A/Bs reinforced the rule. `candidate-b` narrowed the Docker-cloud
gap (`0.538` local Docker -> `0.500 / 0.858` cloud) but did not clear the best
AE tags. `ally-bomb-safe-v2` was the sharper warning: random Docker rose to
`0.6395`, but the pressure gate dropped to `cloudsuite 0.2870` and cloud scored
only `0.369 / 0.847`. Do not submit from random `til test` alone; require the
candidate to improve `cloudsuite` or have a very explicit reason to distrust the
pressure gate.

Latest Workbench reference run after commit `725c097`:

```bash
python training/ae/validate_cloud_suite.py \
  --rounds 24 \
  --suites random library cloudsuite \
  --our heuristic \
  --summary-out training/ae/data/ae-diagnostic-check.json
```

Scores: random `0.7462`, library `0.5564`, cloudsuite `0.3186`, aggregate
mean-of-means `0.5404`. The key diagnostic signal is `cloudsuite` final base
health `0.0` with large base/own-base damage, while attack components remain
positive. Next AE work should improve base survival under pressure without
trading away attack tempo.

Latest cloud-rejected pressure A/B, `pessimistic-mini-search-v1`:

```bash
AE_PESSIMISTIC_SEARCH=1 AE_PESSIMISTIC_ENEMY_GATE=1 \
  .venv/bin/python training/ae/validate_cloud_suite.py \
  --rounds 24 \
  --suites random library cloudsuite \
  --our heuristic \
  --summary-out training/ae/data/ae-pessimistic-search-thresh-24.json
```

Scores: random `0.5613`, library `0.5184`, cloudsuite `0.3962`, aggregate
mean-of-means `0.4920`. This is a pressure-specific win over the `0.3186`
cloudsuite reference, not a broad local-suite win. The search is deterministic
and safety-only: depth-3 own-movement search near live bombs / fresh enemy
pressure, with `AE_PESSIMISTIC_FORCE_SCORE=-80` and
`AE_PESSIMISTIC_MIN_DELTA=55` so it does not behave like the old MCTS planner.
It reduced visible-bomb failures to `8` but traded away random/library farming
tempo. Workbench `til test` scored only `0.456`, and cloud returned
`0.396 / 0.847` with 0/30 errors, effectively matching the Mac cloudsuite
score but far below `ae-fixed-map-v3`. Keep `AE_PESSIMISTIC_SEARCH=0` in the
Dockerfile if replaying that branch; in the restored best-model tree, the
pessimistic-search Docker knobs are removed entirely.

After this cloud result, the shipping AE runtime files were restored to the
`ae-fixed-map-v3` source state (`4c00f92`): `ae/src/ae_manager.py`,
`ae/src/hybrid_manager.py`, and `ae/Dockerfile`. Build from the current tree to
preserve the best known AE model rather than a later diagnostic branch.

Latest rejected A/B reference, `ally-bomb-safe-v2`:

```bash
python training/ae/validate_cloud_suite.py \
  --rounds 24 \
  --suites random library cloudsuite \
  --our heuristic \
  --summary-out training/ae/data/ae-ally-bomb-safe-v2.json
```

Scores: random `0.6650`, library `0.5415`, cloudsuite `0.2870`, aggregate
mean-of-means `0.4978`; cloud `0.369 / 0.847`. The result explains why a
simple mechanics correction can be a large behavioral regression: allied bombs
do not damage same-team defenders, but bombs still consume scarce team bomb
budget, occupy cells, change pathing, and can make the hybrid/policy layer place
bombs in situations where the old escape/base guards were acting as useful
positioning constraints.

Latest Mac-stopped candidate, `ttd-defense-v1`:

```bash
AE_USE_PLAYBOOK=0 AE_USE_OPPONENT_MODEL=0 AE_ALLY_BOMB_SAFE=0 AE_TTD_DEFENSE=1 \
  .venv/bin/python training/ae/validate_cloud_suite.py \
  --rounds 24 \
  --suites random library cloudsuite \
  --our heuristic \
  --summary-out training/ae/data/ae-ttd-defense-v1.json
```

Scores: random `0.6446`, library `0.5475`, cloudsuite `0.2714`, aggregate
mean-of-means `0.4878`. This candidate added a narrow time-to-base-damage /
time-to-intercept override, but it fired too often (`ttd≈22.6` on cloudsuite)
and still did not stop visible enemy bombs (`base_failures={visible_enemy_bomb:75}`).
Keep `AE_TTD_DEFENSE=0` by default. The useful takeaway is diagnostic, not
behavioral: future base-defense work should react to visible enemy bombs that
already threaten the base, not just to enemies near base-hit cells.

---

## 7. Cloud submission (Workbench)

Once local A/B looks healthy, submit:

```bash
echo heuristic > ae/src/.ae_mode  # or 'hybrid' if PPO weights present
til build ae <candidate-tag>
til test ae <candidate-tag>        # last sanity, still uses random NPCs
til submit ae <candidate-tag>
```

The leaderboard keeps the higher score from any submission, so
shipping a new candidate cannot demote `ae-fixed-map-v3` even if it
regresses. Watch the cloud number, not the local one.

If `til submit` ends with `context canceled` / `Terminated` during Artifact
Registry upload, that is not a cloud evaluation. Retry only if the local result
is still worth spending a submission on.

---

## 8. Tier 2 #9 (PPO retrain, legacy scripted recipe)

This section is the older scripted-PPO recipe. Keep it for provenance, but use
the full-RL launcher above for new work. Contrary to the original assumption,
the rollout-heavy PPO loop benchmarked faster on the Mac than on Workbench for
the current lightweight policy.

The legacy recipe trains PPO with our agent in slot 0 and the 5-opponent scripted library
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
