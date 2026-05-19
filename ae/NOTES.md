# AE — notes & history

Last updated: 17 May 2026 ~12:30 SGT — **Parked at `hybrid-v3` (0.555/0.849, 0/30 errors).** Team-best AE score; top-quartile (leaderboard top 0.711). 11 submissions total, 4 post-hybrid-v3 attempts all regressed (mcts-light timeout/regress, ppo-selfplay-v1/v2 regress, ppo-scripted-v1 regress, bc-belief regress, hybrid-conf50 regress). Both **training-side** (BC, BC+belief, PPO mixed/league/selfplay/scripted) and **inference-side** (mcts-light at two configs) AE hypotheses are now exhausted. Tier 3 (hierarchical goal selector) is the only structurally different remaining option (2-3 days work, uncertain outcome); otherwise AE is final at hybrid-v3.

**Architecture**: PPO policy (`ppo-v1` weights) wrapped by `HybridAEManager` (heuristic safety-veto + top-K policy cascade) + `AEManager`'s heuristic dominant-action shortcut firing on adjacent enemy *agents* (not just bases). Mode selection via `AE_MODE` env in `ae/Dockerfile` or `ae/src/.ae_mode` fallback (`hybrid` | `policy` | `heuristic`). Currently `hybrid`. Weights on Workbench: `ae/models/bc.pt` ← `~/ae-checkpoints-backup/deployed-bc-v1.pt`.

**Central finding**: the local→cloud gap is ~0.23 for every approach we tried — heuristics, BC, mixed-opp PPO, frame-stacked PPO + varied maps, hybrid wrappers with various gates, state-augmented BC, league self-play, scripted PPO, MCTS-light inference. **The gap is structural to the hidden eval distribution; none of our training-side or inference-side interventions have moved it.** Hybrid-style wrappers moved the floor by +0.048 but the gap stayed constant; bc-belief actually *widened* it.

Per-task working log for AE (Autonomous Exploration / Bomberman). For the authoritative input/output/scoring spec see [README.md](README.md) and the official [Challenge specifications](https://github.com/til-ai/til-26/wiki/Challenge-specifications#ae). For submission history across all tasks see [../RESULTS.md](../RESULTS.md). For training pipeline see [../training/ae/RUNBOOK.md](../training/ae/RUNBOOK.md).

## Full AE submission ledger (cloud scores)

| Tag | Cloud | Speed | Notes |
|---|---:|---:|---|
| baseline | 0.051 | 0.856 | Periodic FORWARD + bomb-every-20 |
| planner-v1 | 0.445 | 0.788 | First stateful planner |
| planner-v2 | 0.501 | 0.771 | Bomb timer 4→3 + bounded escape + soft threat |
| planner-v3b | 0.499 | 0.853 | Multi-source BFS + blast cache + uvloop |
| bc-v1 | 0.364 | 0.856 | BC of planner-v3b — regressed |
| ppo-v1 | 0.507 | 0.861 | Mixed-opp PPO from BC warm start |
| ppo-v2 | 0.489 | 0.854 | Frame-stacked + varied-maps — regressed |
| hybrid-v2 | 0.545 | 0.863 | Hybrid wrapper (policy + safety veto) — new high |
| **hybrid-v3** | **0.555** | **0.849** | **+ top-K cascade + opportunistic kill — SHIPPED** |
| bc-belief-hybrid | 0.287 | 0.846 | 16×16×11 belief-map (704k params) — memory hypothesis REJECTED |
| hybrid-conf50 | 0.504 | 0.857 | `AE_HYBRID_CONF=0.5` confidence gate — too aggressive |
| ppo-selfplay-v1 | 0.305 | 0.851 | BC warm-start silently SKIPPED (n_frames mismatch), trained from random init |
| ppo-selfplay-v2 | 0.436 | 0.857 | Proper BC warm-start + snapshot pool league |
| mcts-light-v1 | TIMEOUT | — | DEPTH=5 WIDTH=96 no latency cap → 1.2-2.4 s/tick vs ~600 ms budget |
| mcts-light-v2 | 0.487 | 0.595 | 80 ms cap + pre-flight gate + DEPTH=3 WIDTH=24 — speed regressed -0.254, accuracy -0.068 |
| ppo-scripted-v1 | 0.450 | 0.607 | Tier 2 #9 — PPO with `--opponents scripted` (5 archetypes). Gap 0.291. |

## Cloud gap by submission (the structural pattern)

| Submission | Local (apples-to-apples) | Cloud | Gap |
|---|---:|---:|---:|
| planner-v1 | 0.732 | 0.445 | 0.287 |
| planner-v2 | 0.669 | 0.501 | 0.168 |
| planner-v3b | 0.681 | 0.499 | 0.182 |
| bc-v1 | 0.672 | 0.364 | 0.308 |
| ppo-v1 | 0.703 | 0.507 | 0.196 |
| ppo-v2 | 0.763 | 0.489 | 0.274 |
| hybrid-v2 | 0.774 | 0.545 | 0.229 |
| hybrid-v3 | 0.774 | 0.555 | 0.219 |
| bc-belief-hybrid | 0.646 | 0.287 | 0.359 |
| ppo-selfplay-v2 (hybrid) | 0.7068 | 0.436 | 0.271 |
| mcts-light-v2 | 0.6618 | 0.487 | 0.175 |
| ppo-scripted-v1 | 0.741 | 0.450 | 0.291 |

Pattern: ~0.19-0.30 gap for everything. Wrapper changes the gap because they change the local floor more than the cloud floor; the underlying transfer problem is invariant.

## Why every post-hybrid-v3 attempt failed

### `bc-belief-hybrid` (15/05, 0.287) — memory hypothesis REJECTED

BC train results (CPU torch, ~3 min on Workbench):

| Metric | bc-v1 (no belief) | **bc-belief** | Delta |
|---|---:|---:|---:|
| Best val_acc | 0.874 | **0.897** | +0.023 |
| Params | 149k | 704k | +555k |
| Local 6-game | 0.689 | 0.656 | -0.033 (within noise) |

Higher val_acc means the policy fit planner-v3b's behavior *better* with belief input — i.e., planner decisions correlate with belief-map state in ways viewcones alone couldn't capture. That's the prerequisite signal for the memory hypothesis.

Cloud probe — `bc-belief-hybrid` 15/05 11:46:

| Model | Local | Cloud | Gap |
|---|---:|---:|---:|
| bc-v1 (no belief) | 0.689 | 0.364 | 0.325 |
| bc-belief | 0.656 | 0.287 | 0.369 |

Belief input **widened** the local→cloud gap by +0.044. The 704k-param network with belief input found *more* ways to overfit the random-opponent local distribution than the 149k-param bc-v1 did. The belief tensor encodes "where have I been, what's visible, where are enemies recently" — all features that depend heavily on opponent behavior. Locally we trained against random opponents whose movement patterns produced specific belief-tensor distributions; cloud opponents produce different ones, and the policy's belief-conditioned responses fire wrong.

**Lessons**: memory-augmented BC against random opponents amplifies the gap. Belief-map architecture itself is not categorically broken (it fit val_acc 0.897 vs 0.874 — learns better against local). The failure is transfer. **Do not revisit**: bigger BC networks with rich state inputs against random opponents; larger BC datasets of same opponent distribution.

What's tracked in the source tree (still present for future use):

- `ae/src/encoder.py`: `rasterize_belief(ae_manager, obs)` returns `(11, 16, 16)` float32 with channels for visited / wall / destructible / mission-fresh / recon-fresh / resource-fresh / enemy-agent-fresh / enemy-base / bomb-blast-imminent / own-position / base-position. Item freshness decays linearly over 20 ticks; enemy freshness over 5.
- `ae/src/model.py`: `PolicyNetwork(use_belief=True)` adds a small CNN branch (Conv→AvgPool→Conv → 512 features). Old checkpoints (`use_belief=False`) load unchanged.
- `ae/src/policy_manager.py`: owns a private `AEManager` for belief tracking. Calls cheap `_update_memory` + `_age_bombs` per inference. When loaded checkpoint has `use_belief=True`, rasterizes belief and passes to network.
- `training/ae/{collect_bc,train_bc,train_ppo,eval_policy}.py`: thread belief through data pipeline + actor + critic.
- `training/ae/train_ppo.py`: `AggressivePlannerOpponent` (combat-biased threat penalties + lower retreat threshold) + `--opponents league` mode mixing random + planner + aggressive + frozen-self.

### `hybrid-conf50` (15/05, 0.504) — confidence gate too aggressive

ppo-v1 weights + hybrid wrapper + `ENV AE_HYBRID_CONF=0.5` (use policy only when softmax top-action ≥ 0.5; else heuristic).

| Metric | hybrid-v3 | hybrid-conf50 | Δ |
|---|---:|---:|---:|
| Cloud score | 0.555 | **0.504** | **-0.051** |
| Cloud speed | 0.849 | 0.857 | +0.008 |
| Local (6-game) | 0.774 / 0.646 | 0.719 | within band |

Local 0.719 was in upper half of hybrid-v3 distribution, which mis-suggested positive direction. Cloud regression was -0.051 (just outside ±0.04 noise). Gate at 0.5 was too aggressive — kicked the policy out of decisions where its top-2 actions were close (softmax 0.4-0.5 band) but the policy's choice was actually correct on cloud distribution. Heuristic fallback in those moments dragged down.

### `ppo-selfplay-v1` (16/05, 0.305) — n_frames mismatch silently skipped BC warm-start

Training command had `--n-frames 1` (mistakenly copied from bc-belief example), but BC checkpoint is `n_frames=4`. `load_actor` prints a WARN and skips state_dict load when shapes mismatch. PPO trained from random init for 200 updates against league opponents.

| Eval | Score | Notes |
|---|---:|---|
| BC checkpoint vs mixed (pure policy) | 0.4866 | Starting point we *should* have warm-started from |
| ppo-selfplay-v1 vs mixed (pure policy) | 0.4040 | -0.083 vs BC. From-scratch. |
| ppo-selfplay-v1 vs league (training peak) | 0.4362 | Misleading: harder opponent distribution |
| **ppo-selfplay-v1 CLOUD (16/05 13:18)** | **0.305 / 0.851** | -0.250 vs hybrid-v3, 0/30 errors |

That ppo-selfplay-v1 reached 0.404 against mixed from random init is actually a positive signal for the self-play loop *itself* — it produced a policy that holds its own against league. But below BC starting point, so cloud regressed as expected.

**Apparent gap tightening was an artifact**. v1 local-cloud gap was 0.099 (0.404 local → 0.305 cloud), vs ~0.23 structural. That looked like the first intervention to move the gap. v2 retry (same setup with proper warm-start) showed why.

### `ppo-selfplay-v2` (16/05, 0.436) — confirmed v1 tightening was illusion

Proper warm-start + snapshot pool league. Training summary:
- `warm-started actor from training/ae/checkpoints/bc.pt (n_frames=4, use_belief=False)` ✓
- 648,342 actor params (n_frames=4 arch).
- Best PPO eval against league saved 7 times, monotonically: `0.4908 → 0.5038 → 0.5327 → 0.5333 → 0.5982 → 0.6428 → **0.6601**` at epoch 120.
- Workbench auto-restart killed tmux at update 162/200. Best weights at epoch 120 preserved.
- Healthy signals: `pi_loss` consistently negative, `v_loss` dropping (0.37 → 0.13-0.20), `entropy` 0.10 → 0.25, `pool=5` reached at update 50.

| Eval setup | Score | Notes |
|---|---:|---|
| BC checkpoint vs mixed (pure policy) | 0.4866 | Starting point |
| ppo-selfplay-v1 vs mixed (pure policy) | 0.4040 | From-scratch |
| **ppo-selfplay-v2 vs mixed (pure policy)** | **0.5747** | +0.088 over BC, +0.171 over v1 |
| ppo-selfplay-v2 `til test` (hybrid wrapper) | **0.7068** | Wrapper adds ~0.13 |
| **ppo-selfplay-v2 CLOUD (16/05 18:09)** | **0.436 / 0.857** | -0.119 vs hybrid-v3, +0.131 over v1, 0/30 errors |

**v2 vs v1 (+0.131 cloud)**: BC warm-start + self-play opponent curriculum is strictly better than self-play from random init. Workshop's prescription was correct.

**v2 vs hybrid-v3 (-0.119 cloud)**: but doesn't clear the hybrid-v3 ceiling. Self-play improves the policy but not enough to compensate for the gap.

**v1 "gap tightening" was an artifact**. Apples-to-apples (both hybrid mode):

| Submission | Local | Cloud | Gap |
|---|---|---:|---:|
| v1 | 0.404 (pure policy) | 0.305 | 0.099 |
| v2 | 0.575 (pure policy) | 0.436 | 0.139 |
| v2 | 0.7068 (til test, hybrid) | 0.436 | **0.271** |
| hybrid-v3 | 0.774 (til test, hybrid) | 0.555 | 0.219 |

v2's apples-to-apples gap (0.27) is *larger* than hybrid-v3's 0.22. v1's tightening was an illusion: v1's pure-policy raw score (0.404) happened to be closer to its hybrid-wrapped cloud score (0.305) because v1's weak policy meant the hybrid wrapper was carrying more of the cloud performance proportionally. With v2's stronger raw policy, wrapper contributes less relatively, structural gap re-emerges.

### `mcts-light-v1` (16/05 ~23:00, TIMEOUT) — speed budget math was wrong

Returned *"An error occurred while evaluating your AE model … Your model took too long to evaluate."* No score, no leaderboard delta.

Speed math we should have done before shipping:

| Quantity | Value |
|---|---:|
| Cloud wall-clock for whole AE eval | ~30 min |
| Games per eval | 30 |
| Ticks per game (typical) | ~100 |
| → Budget per `/ae` call | **~600 ms** |
| v1 `AE_MCTS_DEPTH × WIDTH` | 5 × 96 |
| Per-tick state expansions | ~2,400 |
| Realistic per-expansion cost in Python | 0.5-1 ms |
| → v1 per-tick MCTS cost | **1.2-2.4 s** |
| Over budget by | **2-4×** |

Local `til test` didn't catch it — no wall-clock cap, just runs to completion.

### `mcts-light-v2` (16/05 23:52, 0.487/0.595) — fix worked, accuracy still regressed

v2 fixes:

1. **Hard latency budget per `/ae` call** — 80 ms hard cap; lookahead returns best-so-far on timeout instead of blocking.
2. **Cheap pre-flight gate** — only run MCTS when `known_bombs / last_seen_enemies / enemy_bases` exist and nearest is within `mcts_depth + 1` cells (gate-fire rate ~20-25% of ticks).
3. **Shrunk Dockerfile defaults**: `AE_MCTS_DEPTH=3` (was 5), `AE_MCTS_WIDTH=24` (was 96). Combined with gate, avg per-tick MCTS cost ~50-90 ms.

| Eval | Score | Speed |
|---|---:|---:|
| hybrid-v3 (shipped) | 0.555 | 0.849 |
| mcts-light-v2 cloud | **0.487** | **0.595** |
| Delta vs hybrid-v3 | **−0.068** | **−0.254** |
| Blended (75% acc + 25% speed) v2 | **0.514** | |
| Blended hybrid-v3 | **0.628** | |
| Blended delta | **−0.114** | |

Wall-clock: hybrid-v3 finished cloud eval in ~4.5 min; v2 took ~12.2 min. Local `til test` already foreshadowed this (17-25 s/round vs ~3-5 s for hybrid-v3, total ~2 min vs ~30 s).

**Why accuracy regressed (not just speed)**: MCTS commits to simulated tactical lines using a stationary-opponent assumption. Cloud opponents move on their own logic, so simulated detonations sometimes miss while we pay danger/escape cost. We're paying *cost of simulated combat* without earning *reward of simulated combat* — same distribution-shift failure as bc-belief and ppo-selfplay, this time at inference.

**Conclusion on MCTS as a class**: as a *primary planner* that overrides hybrid actions on positive scores, MCTS loses to hybrid-v3 on this cloud. Two cheap variants worth one submission each before declaring AE done:

- **A. Conservative-MCTS** (high-confidence override only). Raise `AE_MCTS_MIN_SCORE` 12→22; shrink `DEPTH=2 WIDTH=16`. Speed back to 0.80+, accuracy closer to hybrid-v3-ish.
- **B. Heuristic-only MCTS A/B**. `AE_MODE=heuristic`. Diagnostic — isolates MCTS contribution from hybrid wrapper.

### `ppo-scripted-v1` (17/05 12:19, 0.450/0.607) — Tier 2 #9, REGRESSED

Tier 2 #9: PPO with `--opponents scripted` (5-archetype scripted library: random/greedy/bomber/defender/hunter, no self-play), BC warm-start. Scripted opponents kept "pure" via `_strip_aimanager_smarts` (playbook + opponent-model + tier-1 toggles OFF on themselves) so PPO trains against pure rule-based archetypes, not policies amplified by our own artifacts.

Workbench T4 training reached best eval 0.6322 at update 45/200 before Workbench idle-shutdown killed the run at ~update 54. Even post-shutdown weights regressed cloud −0.105 vs hybrid-v3 — issue is structural, not "needed more training".

Local `til test` 0.741 → cloud 0.450 = gap **0.291**, same pattern that killed every prior PPO/BC submission. Speed regressed -0.242 due to policy net forward + top-K cascade per tick. Blended 0.489 vs hybrid-v3 0.628 = -0.139.

**Operational gotcha for any future long AE run**: if Workbench idle-shutdown settings cannot be changed, start training inside `tmux` and keep a Jupyter notebook kernel active with a one-minute heartbeat cell. GPU utilization alone did not reliably prevent idle shutdown.

Full recipe in [training/ae/RUNBOOK.md §8](../training/ae/RUNBOOK.md).

## Tier 1 / Tier 2 — wired in but local-falsified before cloud submission (17/05 ~03:30)

200-round A/Bs vs three opponent distributions (avg lift over OFF baseline: -0.005):

| Opponent | OFF baseline | Tier 1 shipping (#3+#6+#7) | Δ |
|---|---:|---:|---:|
| random (n=200) | 0.6523 | 0.6369 | −0.015 |
| library (n=200) | 0.5251 | 0.5265 | +0.001 |
| mixed (n=200) | 0.3747 | 0.3728 | −0.002 |

1000-round baseline (heuristic only, all Tier 1 OFF): random 0.6625 ± 0.137, library 0.5292 ± 0.124, mixed 0.3747 ± 0.165.

### Tier 1 single-toggle bisect (vs OFF baseline, 100-round random opponents)

- **#6 no-STAY-penalty**: +0.034 — safe, 1-line. **KEPT default ON**.
- **#3 repeat-kill**: +0.029 — bombs cells where we just killed. **KEPT default ON**.
- **#7 predictive-walk bombing**: +0.013 — random-walk-aware bomb EV. **KEPT default ON**.
- **#2 defense priority**: +0.004 — neutral on random. **DEFAULT OFF**; opt-in for cloud A/B.
- **#4 shared-credit weights**: −0.044 — clear regressor; lookahead values dropped too aggressively. **DEFAULT OFF, do not re-enable without redesign**.

### Tier 1 #1 (offline playbook) — falsified locally

600K trajectory steps yielded a 15K-entry playbook covering 19% of visited states. Standalone (all Tier 1 OFF, only playbook ON) regressed −0.097 vs OFF baseline on random (0.6523 → 0.5556). Tightening to 1186 entries: −0.040. Tightening to 58 entries: −0.018. Bomb-only filter (only override when playbook says BOMB): −0.033. On harder distributions (mixed): bomb-only +0.002 (within noise).

**Diagnosis**: the (x, y, dir, step) state key is too coarse — it ignores belief state (visible enemies, base health, items collected). Different belief states share the same playbook bucket but want different actions. A single chosen action averaged across all belief states picks one that matches no specific situation well.

**Decision**: NOT shipping Tier 1 / Tier 2 bundle to cloud — local lift within noise. Cloud variance ±0.04 means submitting is a coin flip and would consume submission slots without learning anything. Hybrid-v3 stays live.

### Tier 1 in `ae_manager.py` — final defaults

- **#1** offline playbook *(default OFF, falsified)*: kept as opt-in via `AE_USE_PLAYBOOK=1` for future cloud-only A/Bs. Tighten with `AE_PLAYBOOK_FILTER=bomb_only`.
- **#2** defense priority *(default OFF, neutral)*.
- **#3** repeat-kill camping *(default ON, +0.029)*.
- **#4** shared-credit-aware lookahead values *(default OFF, regressor)*.
- **#6** drop the STAY penalty *(default ON, +0.034)*.
- **#7** predictive random-walk bombing *(default ON, +0.013)*.

### What stays valuable from Tier 1+2 work

- `training/ae/simulate.py` runs N rounds against any opponent mix at ~1 round/sec on Mac. Use for any future code A/B before paying for cloud submission.
- `training/ae/opponents.py` — 6 scripted opponent types (`random`, `greedy`, `bomber`, `defender`, `hunter`, `mixed`).
- `training/ae/build_playbook.py`, `fit_opponent_model.py`, `oracle_bc.py` reusable if retrying offline-aggregation with belief-state-aware keys.
- `train_ppo.py` has `--opponents scripted` for scripted-library training.
- `training/ae/RUNBOOK.md` — full end-to-end procedure.

## Hybrid-v3 architecture detail (round 3 of 3, 14 May 19:26)

`hybrid-v3` = `hybrid-v2` + **top-K policy cascade** (try policy's #2/#3 actions if #1 is vetoed before falling back to heuristic) + **opportunistic enemy-kill** (bomb adjacent enemy *agents* sighted this step in the heuristic's dominant-action shortcut, not just adjacent enemy *bases*).

Three consecutive AE submissions each lifted the floor by ~one cloud-noise unit in the same direction: ppo-v1 0.507 → hybrid-v2 0.545 → hybrid-v3 0.555. Each individual lift within noise; cumulative +0.048 is not. We moved the floor; the gap (~0.23) is structurally intact.

### Round 2 cloud results (after AE_MODE bake fix, 14 May)

| Tag | Score | Speed | Local 1-run | Verdict |
|---|---|---|---|---|
| `policy-fast-v2` | **0.425** | 0.859 | 0.654 | Regressed -0.082 vs ppo-v1 same weights. Almost certainly cloud variance (±0.04 on 30 games is normal). Speed flat — confirms speed is evaluator-bound, not torch-bound. |
| `hybrid-v2` | **0.545** | 0.863 | 0.774 | NEW HIGH at the time. First structurally new approach since ppo-v1; +0.038 over the 0.49-0.51 cloud ceiling 5 prior approaches all hit. |
| `heuristic-restore-v2` | **0.502** | 0.854 | 0.787 | +0.003 vs planner-v3b 0.499 (noise). Confirms heuristic-only ceiling is real; TILE_RESPAWN 40→20 and enemy_agent eviction were no-ops on cloud. |

### Code changes that produced this generation

1. **`ae/src/policy_manager.py`** — speed rewrite. `torch.set_num_threads(1)` + `torch.set_num_interop_threads(1)` at module load (kills uvicorn/torch contention on 1-vCPU container); `torch.inference_mode()`; **warmup forward pass** at construction so first `/ae` call doesn't pay JIT/cudnn init; **preallocated input tensors** that we `copy_` into per tick. New `ae_logits(obs)` returns (action, masked logits) in one forward pass — used by hybrid for confidence gating. Per-call latency: ~13 ms → ~3-5 ms on CPU.
2. **`ae/src/hybrid_manager.py`** (NEW) — `HybridAEManager`: policy action by default, heuristic safety-veto when policy picks an action the rule-based planner can prove is wrong (illegal action; `PLACE_BOMB` without verified escape; step into known bomb-blast if heuristic's own action is safer; `STAY` with `frozen_ticks==0` and a non-STAY legal alternative). Fast path: if heuristic is in active "fleeing my own bomb" state, trust heuristic entirely. Optional `AE_HYBRID_CONF` env var for confidence gating.
3. **`ae/src/ae_server.py`** — `AE_MODE` env var: `hybrid` (default), `policy`, or `heuristic`. All modes degrade gracefully if dependency (checkpoint, torch) missing.
4. **`ae/src/ae_manager.py`** — `TILE_RESPAWN_STEPS` 40→20; stale enemy_agent eviction after `ENEMY_AGENT_MEMORY_STEPS=30` in `_age_bombs`.
5. **`training/ae/diagnose.py`** (NEW) — per-round reward-component logger.

### Critical gotcha — `AE_MODE` must be baked into the image

**`AE_MODE=foo til build …` does NOT work.** `docker build` doesn't inherit shell env. Cloud container then defaults to hybrid regardless of submitted tag. First A/B round (2026-05-14) hit this — all three builds had identical sha256 `2572b392...` and ran hybrid mode on cloud.

Two correct ways:

```bash
# (A) Edit Dockerfile before each build: ENV AE_MODE=hybrid → ENV AE_MODE=policy etc.

# (B) Write the mode into the file the server reads at startup:
echo hybrid    > ae/src/.ae_mode
echo policy    > ae/src/.ae_mode
echo heuristic > ae/src/.ae_mode
```

Both picked up by `ae_server._read_mode()` (env wins, file fallback). `.ae_mode` lives next to source, gitignored.

## Pre-hybrid history

### `baseline` (12 May 04:20) — plumbing only

Periodic FORWARD + periodic bomb every 20 turns, respects `action_mask`, stays put while frozen. Validates service starts, endpoint responds, JSON shape correct, `/reset` works. Scored 0.051.

### `planner-v1` (13 May 11:33) — first real attempt, 0.445/0.788

Stateful belief map updated every tick from partial observation. Objective/frontier BFS pathfinding. LOS-safe tactical bombs (place only if blocked by destructible wall on planned path, or enemy in blast line AND clear escape route exists for N ticks; escape pre-verified). Bomb-blast prediction N ticks ahead.

Local: 0.732 (Mac), 0.697 (Workbench Docker). Official: 0.445/0.788.

Hypotheses for the local→official gap (in priority order):
1. **Bomb timer mismatch** — local `til-26-ae` 4-tick fuse vs official maybe 3.
2. **Escape-route check too optimistic** — assumes enemies don't actively block; in adversarial play sometimes they do.
3. **Enemy threat model too binary** — "is/isn't in blast" misses near-misses.
4. **Frontier scoring rewards revealing tiles for own sake** — official scorer probably weights objective completion, not pure exploration.

### `planner-v2` (13 May 23:03) — bomb-timer correctness + soft enemy threat, 0.501/0.771

Changes (committed in [`cb13c4c`](../README.md)):

- **Bomb fuse 4→3 ticks**. Biggest signal. `til-26-ae` config declares `entities.bomb.timer: 3`; with phase order `place → move → detonate → upkeep`, bomb placed at step N detonates after agent movement at step N+timer — agent gets exactly 3 movement actions to escape. Previous `4` allowed `_should_place_bomb` to commit to self-trapping bombs. Net official lift: +0.056.
- **Bounded escape BFS**: replaced unbounded `_nearest_escape_cell` with `_safe_escape_within(loc, blast, BOMB_TIMER)`. Refuses bomb unless safe cell reachable within fuse.
- **Enemy soft threat**: recently-seen enemies (within `ENEMY_STALENESS = 3` steps) + 4-neighbors penalized in path scoring (-3.0 in `_choose_target`, -8.0 in `_fallback_action`). BFS still allows them so attack paths to enemy bases aren't blocked.
- **Frontier scoring by unseen yield**: each frontier cell scored `4.0 + 1.0 × (#unseen 4-neighbors)` instead of flat `6.0`.

Workbench local 3-run: 0.659/0.659/0.689 (mean ~0.669, ±0.015). Official: 0.501/0.771, 0/30 errors.

### `planner-v3` (NOT submitted) — over-eager bombing, local regressed

Bundled 7 changes (multi-source BFS, blast cache, dominant-action shortcut, predictive bomb at range 2, bomb chains, proactive wall-break, item respawn awareness, base defense, health-aware retreat, softened threat 1.0/3.0, Dockerfile uvloop/httptools).

Workbench local: 0.588/0.629/0.570 (mean 0.596, **-0.07 vs v2**). Predictive bombing at range 2 fires almost every turn on 16×16 map with 5 random enemies; bombs got burned on speculation. Bomb chains compounded the problem. NOT submitted.

### `planner-v3b` (13 May 23:42) — keep speed wins, dial back bombs, 0.499/0.853

vs v3:
- `PREDICTIVE_BOMB_RANGE 2 → 1` AND require **≥2 enemies in the extended blast**.
- Bomb-chain trigger **disabled** (helper kept for reuse).
- Threat penalty **2.0 / 5.0** (between v2's 3.0/8.0 and v3's 1.0/3.0).

Kept from v3: multi-source BFS, blast cache, dominant-action shortcut (all speed); item respawn awareness, base defense, health-aware retreat, proactive wall-break for enemy_base/mission (orthogonal); Dockerfile uvloop/httptools.

Workbench local 6-run: 0.80, 0.61, 0.66, 0.65, 0.64, 0.63 → mean 0.681 (±0.07; 0.80 outlier inflates mean ~0.13/6). Median ~0.65. Official: 0.499/0.853, 0/30 errors. Score flat vs v2 (-0.002), speed +0.082 from BFS/cache/uvloop.

**Key finding from v2→v3b**: local→official gap is consistent at ~0.18 across very different heuristic configurations. Gap is structural — likely env distribution mismatch (random opponents locally vs something else officially, fixed novice seed vs varied hidden), not heuristic tuning will close. **Heuristic ceiling around 0.50 official.**

### `bc-v1` (14 May 01:22) — first learned policy, 0.364/0.856 — REGRESSED

Pipeline:

1. **Dataset**: 200 games × ~200 turns = 40K samples from `collect_bc.py` rolling out planner-v3b vs random (~8 min Workbench). Action distribution: FORWARD 47.8% / BACKWARD 13.9% / RIGHT 15.0% / LEFT 12.7% / PLACE_BOMB 10.3% / STAY 0.3% — healthy.
2. **Network**: `PolicyNetwork` (~149k params) — small CNN over each viewcone (32→16 ch for agent_view 25×7×5, 16→8 ch for base_view 25×7×7) + MLP over 17-dim scalars → 6-way action head.
3. **Training**: 20 epochs supervised CE with `log(action_mask)` added to logits so model can never assign mass to illegal actions. ~28 s on Workbench GPU. Best val_acc 0.8742.
4. **Direct eval** (`eval_policy.py`, 4×6-game): mean 0.689.
5. **Container deploy**: `policy_manager.py` falls back to `AEManager` on FileNotFoundError or any exception. `ae/requirements.txt` adds CPU-only torch via `--extra-index-url https://download.pytorch.org/whl/cpu`. Image grows ~80 → 250 MB.
6. **Path bug fix** (commit `22996fe`): `Path(__file__).parent.parent / "models" / "bc.pt"` resolved to `/models/bc.pt` inside container (because `COPY src .` puts source at `/workspace/`, not `/workspace/src/`). `policy_manager.py` now checks both candidate paths. First bc-v1 build silently fell back to heuristic for 4 runs before this was caught.
7. **Container eval** (`til test`, 4×6-game): mean 0.672. Within noise of direct eval — deployment faithful.
8. **Submitted** 14 May → 0.364/0.856. Regressed -0.135 vs planner-v3b. Local→official gap was 0.18 for heuristics, ballooned to **0.31** for BC.

**Reading**: container faithfully reproduces direct eval (0.672 vs 0.689) so deployment is NOT the problem. Policy correctly imitates planner-v3b locally (val_acc 0.874). Yet official tanks. The 0.31 gap means the NN **memorized planner behavior against random local opponents** — patterns that don't transfer. Heuristics are explicit rules and degrade gracefully; BC keys on subtle obs features correlating with planner-action in our local distribution but meaning nothing officially.

**Implication for PPO**: training against random only inherits this gap. PPO **must** use mixed opponent pool — random + frozen planner-v3b + frozen self-copies + (ideally) scripted-aggressor agents.

### `ppo-v1` (14 May 04:36) — RL fine-tune with mixed opponents, 0.507/0.861 — NEW HIGH at the time

Pure-PyTorch PPO in [`training/ae/train_ppo.py`](../training/ae/train_ppo.py). Reuses deployed `PolicyNetwork`, so saved actor checkpoint directly compatible with `PolicyAEManager`.

```bash
python training/ae/train_ppo.py \
  --bc-checkpoint training/ae/checkpoints/bc.pt \
  --out training/ae/checkpoints/ppo.pt \
  --updates 200 --games-per-update 8 --eval-games 12 \
  --opponents mixed --eval-opponents mixed
```

Actor warm-starts from `bc.pt` (not scratch). Critic training-only (not in Docker image). Control `env.possible_agents[0]` only. Opponent pool mixed: random + frozen planner-v3b + frozen self-copy. Invalid actions masked during PPO sampling and loss.

Local: eval_policy 4-run mean ~0.701 (range 0.019 — tight variance, best signal we've ever seen locally); `til test` 4-run mean ~0.703. Official: 0.507/0.861. Local→official gap 0.19, same as heuristic — mixed-opponent training didn't close the gap.

### `ppo-v2` (14 May 13:29) — frame-stacked + varied-maps, 0.489/0.854 — REGRESSED

- 30-epoch supervised BC on 40k stacked samples, val_acc 0.8898 (vs v1's 0.8742). Net frame-stacking gain over single-frame.
- eval_policy on novice: 0.7138/0.7752/0.7885 mean 0.759 (+0.058 vs v1 novice). Best 0.789.
- eval_policy on varied maps: 0.6353/0.6910/0.6768 mean 0.668 — policy generalizes to non-novice.
- til test: 0.7282/0.8260/0.7352 mean 0.763. 0.826 single-run high.
- Official: 0.489/0.854. **Regressed -0.018 vs ppo-v1**. Gap widened 0.19 → 0.27. Varied-map training was the wrong bet.

### `[bake bug]` (14 May 14:30) — AE_MODE silently not baked

Three v1 builds (policy-fast-v1, hybrid-v1, heuristic-restore) shipped identical images because `AE_MODE=foo til build` doesn't bake AE_MODE. Fixed via `ENV AE_MODE` in Dockerfile + `.ae_mode` file fallback. (See "Critical gotcha" above.)

## Local validation history

```text
Variant     Date/time            Local score              Notes
baseline    12/05                0.051 official           Reference point only
planner-v1  13/05 10:31 Mac      0.732 local              Stateful belief + BFS + LOS-safe tactical bombs, 0 invalid actions
planner-v1  13/05 Workbench      0.697 local              Built/tested with official Workbench Docker flow before submission
planner-v1  13/05 11:33          0.445 official           ← significant drop from both local environments
planner-v2  13/05 Workbench      0.659/0.659/0.689        3-run local mean ≈ 0.669 (variance ±0.015)
planner-v2  13/05 23:03          0.501 official           +0.056 over v1; gap narrowed not closed
planner-v3  13/05 Workbench      0.588/0.629/0.570        3-run mean ≈ 0.596 (-0.07 vs v2). NOT submitted.
planner-v3b 13/05 Workbench      0.80/0.61/0.66/0.65/0.64/0.63   6-run mean ≈ 0.681 (±0.07)
planner-v3b 13/05 23:42          0.499 official           Score flat vs v2 (-0.002), speed +0.082. Gap ~0.18 structural.
bc-v1       14/05 eval_policy    0.7195/0.7193/0.6248/0.6908   Direct mean ≈ 0.689
bc-v1       14/05 til test       0.6317/0.7037/0.6128/0.7398   Container mean ≈ 0.672
ppo-v1      14/05 eval_policy    0.7112/0.6998/0.6925         Direct mean ≈ 0.701 (range 0.019)
ppo-v1      14/05 til test       0.766/0.634/0.708            Container mean ≈ 0.703
ppo-v1      14/05 04:36          0.507 official               Gap 0.19, same as heuristic
ppo-v2 BC   14/05 eval_policy    0.6788 novice                30-epoch BC, val_acc 0.8898
ppo-v2      14/05 eval_policy    0.7138/0.7752/0.7885 novice  Mean 0.759 (+0.058 vs ppo-v1 novice)
ppo-v2      14/05 eval_policy    0.6353/0.6910/0.6768 varied  Mean 0.668 — generalizes to non-novice
ppo-v2      14/05 til test       0.7282/0.8260/0.7352         Container mean 0.763
ppo-v2      14/05 13:29          0.489 official               REGRESSED -0.018 vs v1. Gap widened 0.19 → 0.27
policy-fast-v2 14/05 til test    0.654 (1 run)                ppo-v1 weights + speed fixes. Cloud 0.425 — variance
hybrid-v2   14/05 til test       0.774 (1 run)                Hybrid wrapper around ppo-v1. Cloud 0.545 = +0.038
heuristic-restore-v2 14/05 til test 0.787 (1 run)            Pure heuristic + v2 tweaks. Cloud 0.502
hybrid-v3   14/05 til test       —                            +top-K + opportunistic kill. Cloud 0.555 (shipped)
bc-belief   15/05 BC train       val_acc 0.897 (+0.023 vs bc-v1); local 0.656/0.663 (hybrid/policy). 704k params, 20 ep, ~3 min CPU
bc-belief-hybrid 15/05 til test  0.646 (1 run)                Cloud 0.287 — memory hypothesis rejected
hybrid-conf50 15/05 til test     0.719 (1 run)                ppo-v1 + hybrid + AE_HYBRID_CONF=0.5. Cloud 0.504 (-0.051)
ppo-selfplay-v1 16/05            local mixed 0.404            n_frames=1 vs n_frames=4 BC ckpt → warm-start skipped. Cloud 0.305
ppo-selfplay-v2 16/05 til test   0.7068 (hybrid wrapper)      BC warm-start ✓; pure policy 0.575. Cloud 0.436
mcts-light-v2 16/05 til test     0.6618                       Cloud 0.487/0.595
ppo-scripted-v1 17/05 til test   0.741                        Cloud 0.450/0.607
```

## What our agent runs on

- **No GPU usage** — CPU-bound pure Python in inference loop. Container base same as other services per [Dockerfile](Dockerfile). Per-step latency dominates (`/ae` called once per game tick).
- **Source**: [src/ae_manager.py](src/ae_manager.py) (manager) + [src/ae_server.py](src/ae_server.py) (server with reset-robustness patch — empty POST or `step == 0` triggers re-instantiation; see CLAUDE.md).
- **Spec contract**: `/ae` on port 5005 returns `{"action": int}`; `/reset` clears state. Observation keys: `agent_viewcone`, `base_viewcone`, `direction`, `location`, `base_location`, `health`, `frozen_ticks`, `base_health`, `team_resources`, `team_bombs`, `step`, `action_mask`.
- **State on `self`**: belief map, frontier set, turn counter — must NOT use module-level globals because server re-instantiates manager on reset.

### Planner sketch (planner-v3b)

- Stateful belief map updated each tick from partial observation.
- Objective + frontier BFS pathfinding to nearest unexplored / objective cell.
- LOS-safe tactical bomb placement (only when blocked by destructible wall or enemy in blast line AND escape route pre-verified).
- Respects `action_mask`.
- `frozen_ticks` guard: stay put while frozen.
- Per-action cost balances progress, bomb-blast safety N ticks ahead, pickup proximity.

## Operational notes

### `n_frames` MUST match BC checkpoint when running PPO

`train_ppo.py` silently SKIPS warm-start if shapes don't match (lines 656-661). We lost `ppo-selfplay-v1` to this exact mismatch (`--n-frames 1` vs `n_frames=4` checkpoint). Always check first:

```bash
python - <<'PY'
import torch
ckpt = torch.load("training/ae/checkpoints/bc.pt", map_location="cpu", weights_only=False)
print("BC n_frames=", ckpt.get("n_frames"), "use_belief=", ckpt.get("use_belief"))
PY
```

### SSH access from Mac (16 May 14:25 SGT)

`ssh workbench` configured on Mac dev box. Lets Claude run read-only checks (`tail` training log, `nvidia-smi`, file listings) without copy-paste round-trips. Writes still go through user. Public key in `~/.ssh/authorized_keys` on Workbench. Setup matches upstream wiki ("Power users" → SSH).

### Workshop reading

TIL workshop notebook 05 ("Multi-Agent Introduction") prescription was:

> "Self-play trains an agent by having it compete against a copy of itself. Periodically, the opponent is updated to a checkpoint of the current policy. This produces a curriculum: the opponent always provides a challenge at the current skill level."

We implemented this in `train_ppo.py` via `SnapshotPool` (bounded FIFO ring buffer of historical actor snapshots, kept on CPU; deepcopy at insert + at opponent construction). Late-training rollouts face opponents from updates 10/30/50/80/100 ago. The prescription was directionally correct (v2 beat v1 by +0.131 cloud) but doesn't clear the hybrid-v3 ceiling.

Workshop contains zero MCTS / model-based search content (only BFS/DFS/A* in notebook 03 for fully-observed grids). Top teams almost certainly aren't doing MCTS — they're doing what the workshop teaches. MCTS attempt confirmed this: MCTS as primary planner lost to hybrid-v3.

## Where we are vs target

| Metric | Currently | Target | Gap |
|---|---:|---:|---:|
| Official score | 0.555 | 0.70 | +0.145 |
| Official speed | 0.849 | 0.90 | +0.05 |
| Blended (0.75/0.25) | 0.628 | 0.75 | +0.122 |

Speed is within striking distance with uvicorn/Docker tuning. **Score is the hard problem.**

## Open questions

- **Why is the gap so stable at ~0.18-0.30?** v2 and v3b have very different bomb behavior but the same gap. Self-play, scripted PPO, MCTS, belief-map — all converge near the same gap. Possible causes:
  - Different opponent distribution in hidden eval (smarter agents that don't wander into bombs we predicted).
  - Different map distribution (advanced track? bigger grid?).
  - Different episode lengths or reward calibration.
- **Is `0/30 errors` masking some game-completion failures?** No invalid-action errors ≠ no agent-deaths or timeouts.
- **Could a fundamentally different *evaluation* lens help?** Without hidden-eval per-game replays, we're guessing.

## What remains worth doing (Tier 3 candidate)

**Hierarchical goal selector** is the only structurally different remaining option. Operates on coarser state (mission/recon/resource zones, base proximity, bomb-vs-explore phase) — coarse enough to plausibly be invariant to the local-cloud opponent distribution shift that killed everything else. 2-3 days work, uncertain outcome.

If not Tier 3, AE is final at hybrid-v3.

What to avoid (lessons learned):
- Bigger BC networks with rich state inputs against random opponents (overfits transfer).
- Larger BC datasets of same opponent distribution (won't help transfer).
- Frame stacking (ppo-v2 widened gap).
- Frontier/exploration tuning for its own sake (spec rewards missions/resources/recon/attacks/kills/base destruction; exploration only matters when it finds those).
- Six-game local means as a selection metric (high variance + wrong distribution; uncorrelated with cloud).
- Committing large checkpoints to git (keep on Workbench + backup, copy into `ae/models/bc.pt` only for deliberate build).
- Tuning frontier/exploration reward without hidden-eval data — every further tweak is a coin flip.

## Reproducibility / pointers

- Manager source: [src/ae_manager.py](src/ae_manager.py)
- Server (reset-robustness patch applied): [src/ae_server.py](src/ae_server.py)
- Local Mac test: spin up `til_environment.bomberman_env` directly (no Docker needed for Mac iteration)
- Workbench Docker flow: `til build ae <tag> && til test ae <tag>`
- Official AE env (read-only submodule): [../til-26-ae/](../til-26-ae/)
- Training pipeline + Tier 1+2 runbook: [../training/ae/RUNBOOK.md](../training/ae/RUNBOOK.md)
