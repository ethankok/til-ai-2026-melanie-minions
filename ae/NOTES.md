# AE — notes & history

Last updated: 17 May 2026 ~03:30 SGT — **Tier 1 + Tier 2 LOCAL FALSIFIED, NOT SUBMITTING.** 200-round A/Bs against three opponent distributions show no meaningful lift over hybrid-v3 baseline:

| Opponent | OFF baseline | Tier 1 shipping (#3+#6+#7) | Δ |
|---|---:|---:|---:|
| random (n=200) | 0.6523 | 0.6369 | −0.015 |
| library (n=200) | 0.5251 | 0.5265 | +0.001 |
| mixed (n=200) | 0.3747 | 0.3728 | −0.002 |

1000-round baseline (heuristic only, all Tier 1 OFF):
- vs 5× random: 0.6625 ± 0.137
- vs library (random/greedy/bomber/defender/hunter): 0.5292 ± 0.124
- vs 5× mixed: 0.3747 ± 0.165

**Tier 1 #1 (offline playbook) hypothesis falsified locally.** 600K trajectory steps yielded a 15K-entry playbook covering 19% of visited states. Standalone (all Tier 1 OFF, only playbook ON) regressed −0.097 vs OFF baseline on random (0.6523 → 0.5556). Tightening to 1186 entries: −0.040. Tightening to 58 entries: −0.018. Bomb-only filter (only override when playbook says BOMB): −0.033. Even the narrowest, highest-confidence filter is net negative on local. On harder distributions (mixed): bomb-only +0.002 (within noise).

Diagnosis. The (x, y, dir, step) state key is too coarse — it ignores the belief state (visible enemies, base health, items collected so far). Different belief states share the same playbook bucket but want different actions. A single chosen action averaged across all belief states picks one that matches no specific situation well.

**Decision. NOT shipping the Tier 1 / Tier 2 bundle to cloud — local lift is within noise (avg −0.005). Cloud variance ±0.04 means submitting is a coin flip and we'd consume submission slots without learning anything. Hybrid-v3 (0.555/0.849) stays the live entry.**

Bisect data (single-toggle vs OFF baseline, 100-round random opponents):
- **#6 no-STAY-penalty**: +0.034 — safe, 1-line change. KEPT in defaults.
- **#3 repeat-kill**: +0.029 — bombs cells where we just killed. KEPT.
- **#7 predictive-walk bombing**: +0.013 — random-walk-aware bomb EV. KEPT.
- **#2 defense priority**: +0.004 — neutral on random. DEFAULT OFF; opt-in for cloud A/B.
- **#4 shared-credit weights**: −0.044 — clear regressor; lookahead values dropped too aggressively. DEFAULT OFF, do not re-enable without redesign.

Tier-1 in `ae_manager.py` (defaults: #3, #6, #7 ON; #2, #4 OFF; playbook OFF; opponent model OFF):
- **#1** offline playbook *(default OFF, falsified)*: kept as opt-in via `AE_USE_PLAYBOOK=1` for future cloud-only A/Bs. Tighten with `AE_PLAYBOOK_FILTER=bomb_only`.
- **#2** defense priority *(default OFF, neutral)*.
- **#3** repeat-kill camping *(default ON, +0.029)*.
- **#4** shared-credit-aware lookahead values *(default OFF, regressor)*.
- **#6** drop the STAY penalty *(default ON, +0.034)*.
- **#7** predictive random-walk bombing *(default ON, +0.013)*.

What stays valuable from this work:
- `training/ae/simulate.py` runs N rounds against any opponent mix at ~1 round/sec on Mac. Use for any future code A/B before paying for a cloud submission slot.
- `training/ae/opponents.py` 6 scripted opponent types (`random`, `greedy`, `bomber`, `defender`, `hunter`, `mixed`).
- `training/ae/build_playbook.py`, `fit_opponent_model.py`, `oracle_bc.py` are reusable if we want to retry the offline-aggregation approach with belief-state-aware keys.
- `train_ppo.py` now has `--opponents scripted` for training against the library on Workbench (Tier 2 #9, **READY for Workbench, NOT yet run**). The scripted opponents force playbook + opponent-model + Tier-1 toggles OFF on themselves via `_strip_aimanager_smarts` so PPO trains against pure rule-based archetypes instead of policies amplified by our own artifacts. Full recipe in [training/ae/RUNBOOK.md §8](../training/ae/RUNBOOK.md). Local smoke test passed (2 updates, eval climbed 0.000 → 0.139 with no warm-start).
- `training/ae/RUNBOOK.md` is the end-to-end procedure if anyone wants to re-run the full pipeline.

Earlier today: **`mcts-light-v2` SHIPPED 23:52 at 0.487/0.595 — REGRESSED.** Blended 0.514 vs hybrid-v3's 0.628 (−0.114). The speed cap + pre-flight gate prevented the v1 timeout, but MCTS still cost +9 minutes of cloud wall-clock (4.5 → 12.2 min); speed score dropped 0.849 → 0.595. Accuracy ALSO regressed (−0.068 vs hybrid-v3), suggesting MCTS is *replacing* hybrid-v3 actions with cloud-worse choices, not just being slower. Best read: MCTS commits to simulated combat lines based on a stationary-opponent assumption that cloud opponents don't honor — same local-cloud distribution-shift failure mode that killed bc-belief and ppo-selfplay, this time at inference instead of training. Hybrid-v3 (0.555/0.849) stays leaderboard-shipped via highest-score retention. **Both training-side (bc-belief, ppo-selfplay) and inference-side (mcts-light at two configs) hypotheses are now falsified.** Remaining cheap iterations: (A) conservative-MCTS variant (raised `AE_MCTS_MIN_SCORE` + smaller `DEPTH/WIDTH`) so MCTS is a high-confidence override layer only; (B) heuristic-only MCTS A/B to isolate the MCTS contribution from the hybrid wrapper. After that, AE is genuinely exhausted at hybrid-v3.

Earlier today: `mcts-light-v1` TIMED OUT on cloud ("Your model took too long to evaluate"; no score, no leaderboard impact). MCTS budget arithmetic was wrong: `AE_MCTS_DEPTH=5 × AE_MCTS_WIDTH=96` ran on **every** tick with no latency cap → ~2400 state expansions/tick × ~0.5-1 ms each = **1.2-2.4 s/tick** vs cloud's ~600 ms/tick budget. v2 fixed the timeout (latency cap + gate + smaller defaults) but introduced the speed/accuracy regression above.

## CURRENT CANDIDATE — `mcts-light-v2` (timeout fix on v1)

Why this exists: reaching 0.7 needs roughly +145 reward/round over
`hybrid-v3`'s 0.555. Mission/resource/recon routing is too small for that;
the only credible source is more combat/base reward without increasing
self/base damage. Training has repeatedly moved local scores without moving
cloud, so this candidate stops trying to learn hidden opponents and instead
searches tactical futures from the live belief map.

### What killed v1 (16/05 ~23:00 cloud timeout)

Submission returned: *"An error occurred while evaluating your AE model … Your
model took too long to evaluate."* No score, no leaderboard delta (hybrid-v3
still on board at 0.555/0.849).

Speed budget math we should have done before shipping v1:

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

Why local `til test` didn't catch it: no wall-clock cap, just runs to
completion (6 games × ~3-5 min = 15-30 min was probably fine in absolute
terms even at 1-2 s/tick; cloud has a hard 30-min cap across 30 games and
killed the container when crossed).

### v2 fixes (must all land before next submission)

1. **Hard latency budget per `/ae` call** (the non-negotiable one)

   ```python
   import time
   deadline = time.monotonic() + 0.080   # 80 ms hard cap
   for _depth in range(self.mcts_depth):
       if time.monotonic() > deadline:
           break  # return best-so-far instead of timing out
       # ... existing expansion loop ...
   ```

   Even if depth/width targets aren't reached, the function must return.

2. **Cheap pre-flight gate** — only run MCTS when something tactical is
   plausibly within reach:

   ```python
   def _should_run_mcts(self, location):
       if not self.known_bombs and not self.last_seen_enemies and not self.enemy_bases:
           return False
       targets = list(self.known_bombs) + list(self.enemy_bases) + list(self.last_seen_enemies)
       if not targets:
           return False
       return min(self._manhattan(location, t) for t in targets) <= self.mcts_depth + 1
   ```

   Expected gate-fire rate: ~20-25% of ticks (the rest are pure exploration
   where the existing frontier planner already does the right thing). That
   alone is a 4-5× drop in average cost.

3. **Shrunk Dockerfile defaults**

   ```dockerfile
   ENV AE_MCTS_DEPTH=3      # was 5
   ENV AE_MCTS_WIDTH=24     # was 96
   ```

   24 × 5 actions × 3 plies = ~360 expansions worst case = ~180-360 ms when
   the gate fires. Combined with the gate, average per-tick MCTS cost ~50-90
   ms. Comfortably in budget with headroom for the rest of inference.

Local validation before submitting v2 — must measure, not just trust:

```bash
# Add a temporary timing print inside the lookahead:
#   t0 = time.monotonic()
#   ... search ...
#   print(f"mcts {(time.monotonic()-t0)*1000:.1f}ms", flush=True)
# Then:
til test ae mcts-light-v2 2>&1 | tee /tmp/mcts-v2-times.log
grep mcts /tmp/mcts-v2-times.log | awk '{print $2}' | sort -n | tail -20
# Worst-case tick should be ≤ 100 ms. If it's 200+, lower DEPTH or WIDTH.
```

Only submit once `til test`'s worst tick is comfortably below 100 ms.

Code changes:

- `ae/src/ae_manager.py`
  - Adds `AE_MCTS` gated tactical lookahead.
  - Searches our own action sequences for `AE_MCTS_DEPTH` steps (default 5)
    with beam width `AE_MCTS_WIDTH` (default 96).
  - Preserves a tactical side-beam so low-immediate-score bomb lines are not
    pruned before their detonation payoff lands.
  - Scores robust events only: known enemy bases in blast, recently seen
    enemies in/near blast, own-base/self blast penalties, and nearby item
    pickup as small tie-breakers.
  - Returns the lookahead action only when a tactical sequence clears
    `AE_MCTS_MIN_SCORE` (default 12), so ordinary movement still falls back to
    the existing frontier/objective planner.
- `ae/src/hybrid_manager.py`
  - Adds `AE_HYBRID_TRUST_MCTS=1` behavior: if the heuristic's lookahead finds
    a tactical line above `AE_HYBRID_MCTS_MIN_SCORE`, take it before asking the
    neural policy. This prevents the ppo-v1 policy from overriding the only
    remaining 0.7-upside path.
- `ae/Dockerfile`
  - Bakes the MCTS-light candidate knobs:
    `AE_MCTS=1`, `AE_MCTS_DEPTH=5`, `AE_MCTS_WIDTH=96`,
    `AE_MCTS_MIN_SCORE=12`, `AE_HYBRID_TRUST_MCTS=1`,
    `AE_HYBRID_MCTS_MIN_SCORE=12`.

Workbench runbook:

```bash
# Use the current ppo-v1 checkpoint slot if testing hybrid+mcts.
cp ~/ae-checkpoints-backup/deployed-bc-v1.pt ae/models/bc.pt
echo hybrid > ae/src/.ae_mode

til build ae mcts-light-v1
til test ae mcts-light-v1
til submit ae mcts-light-v1
```

Optional pure-heuristic A/B if hybrid+mcts regresses and we need attribution:

```bash
echo heuristic > ae/src/.ae_mode
til build ae mcts-light-heuristic-v1
til test ae mcts-light-heuristic-v1
til submit ae mcts-light-heuristic-v1
```

Decision rule (was set before v2 had a score):

- Cloud `> 0.555`: keep iterating MCTS thresholds/depth; this is the first
  inference-side evidence that combat lookahead transfers.
- Cloud `0.52-0.555`: inspect local diagnostics; try pure heuristic or
  lower/higher `AE_MCTS_MIN_SCORE` once.
- Cloud `< 0.52`: MCTS-light as implemented is too speculative; roll back to
  hybrid-v3 and stop AE unless hidden-eval traces become available.
- Cloud `TIMEOUT`: speed budget blown — shrink depth/width further, tighten
  the latency cap, and re-test locally with timing instrumentation before
  re-submitting. (This is what happened to v1.)

### v2 result — SHIPPED 16/05 23:52 SGT, REGRESSED 0.487/0.595

| Eval | Score | Speed |
|---|---:|---:|
| hybrid-v3 (shipped) | 0.555 | 0.849 |
| mcts-light-v2 cloud | **0.487** | **0.595** |
| Delta vs hybrid-v3 | **−0.068** | **−0.254** |
| Blended (75% acc + 25% speed) — v2 | **0.514** | |
| Blended — hybrid-v3 | **0.628** | |
| Blended delta | **−0.114** | |

Speed cost in wall-clock: hybrid-v3 finished cloud eval in ~4.5 min;
v2 took ~12.2 min. The 80 ms per-call cap + pre-flight gate prevented
the v1 timeout, but MCTS still added real compute. Local `til test`
already foreshadowed this: each round took 17-25 s (vs ~3-5 s for
hybrid-v3), total local test ~2 min vs ~30 s.

Local-cloud gap on v2 was 0.175 (local til test 0.662 → cloud 0.487),
*narrower* than hybrid-v3's 0.219. Same pattern as the
ppo-selfplay-v1-vs-v2 finding: changing the inference layer changes
the gap, but the local floor dropped enough that the tighter gap
didn't help.

Best read on why accuracy regressed (not just speed):
- MCTS commits to simulated tactical lines using a stationary-opponent
  assumption. Cloud opponents move on their own logic, so simulated
  detonations sometimes miss while we still pay the danger/escape cost.
- We're paying the *cost of simulated combat* without earning the
  *reward of simulated combat*. Same distribution-shift failure that
  killed bc-belief (training-side) and ppo-selfplay (training-side),
  this time at inference.

**Conclusion on MCTS as a class**: as a *primary* planner that
overrides hybrid actions on positive scores, MCTS loses to hybrid-v3
on this cloud. Two cheap variants are still worth one submission
each before declaring AE done:

### Cheap remaining iterations (each ~30 min)

**A. Conservative-MCTS** — make MCTS a high-confidence override layer
only, not a primary planner. Raise `AE_MCTS_MIN_SCORE` from 12 → 22
so MCTS only commits to *clearly* high-value lines; shrink
`DEPTH=2 WIDTH=16` to recover speed.

```dockerfile
ENV AE_MCTS_DEPTH=2
ENV AE_MCTS_WIDTH=16
ENV AE_MCTS_MIN_SCORE=22
ENV AE_HYBRID_MCTS_MIN_SCORE=22
```

Tag: `mcts-light-v3-conservative`. Expected: speed back to 0.80+,
accuracy hopefully closer to hybrid-v3-ish since MCTS only fires
on the high-confidence subset of tactical ticks.

**B. Heuristic-only MCTS A/B** — `AE_MODE=heuristic`. Bypasses the
neural policy entirely; tells us whether MCTS adds value over plain
planner-v3b (0.499 cloud baseline) or just adds cost.

```bash
echo heuristic > ae/src/.ae_mode
til build ae mcts-light-heuristic-v1
til test  ae mcts-light-heuristic-v1
til submit ae mcts-light-heuristic-v1
echo hybrid > ae/src/.ae_mode   # restore for future builds
```

Tag: `mcts-light-heuristic-v1`. Diagnostic-only — isolates the MCTS
contribution from the hybrid wrapper interaction.

After A and B (or just A if speed remains the dominant blocker), AE
is exhausted at hybrid-v3 across both training-side and inference-side
interventions. The next-highest EV remaining qualifier lever is the
NLP `v13b-deberta` retune already wired up.

## NEXT EXPERIMENT — self-play PPO retrain (workshop-recommended)

The TIL workshop notebook 05 ("Multi-Agent Introduction") explicitly diagnoses our problem and prescribes the fix. Quoting directly:

> "It's easy to overfit to a weak fixed opponent and regress when the opponent improves."

This is exactly what `bc-belief-hybrid` (0.287 cloud, -0.268 vs hybrid-v3) did. Our prior PPO runs all trained against `random + planner-v3b`, and the bigger belief network amplified the overfit.

The workshop's prescription:

> "Self-play trains an agent by having it compete against a copy of itself. Periodically, the opponent is updated to a checkpoint of the current policy. This produces a curriculum: the opponent always provides a challenge at the current skill level."

### What we changed in `training/ae/train_ppo.py`

1. **`SnapshotPool` class** — bounded FIFO ring buffer of historical actor snapshots, kept on CPU (deepcopy at insert, deepcopy + `.to(device)` at opponent construction).
2. **`_make_opponents`** now accepts an optional `snapshot_pool`; when non-empty, `FrozenPolicyOpponent` is built from a *sampled snapshot* instead of the live actor.
3. **Training loop** seeds the pool with the initial actor and adds a fresh snapshot every `--snapshot-interval` updates (default 10), capped at `--snapshot-pool-size` (default 5).
4. **New `selfplay` opponent mode** — pool-only (no heuristic mix). Available as an A/B against `league`.

Critical: the prior `FrozenPolicyOpponent` deepcopied the *live* actor at rollout construction. In late training that meant "play yourself" — almost no gradient signal. The snapshot pool fixes this: late-training rollouts face opponents from updates 10, 30, 50, 80, 100 ago, preserving curriculum diversity.

### Recommended run command (Workbench) — `--n-frames` MUST match BC checkpoint

```bash
# 1. Check the BC checkpoint's n_frames first — train_ppo.py silently
#    SKIPS the warm-start if --n-frames doesn't match (lines 656-661).
#    This cost us ppo-selfplay-v1: ran with --n-frames 1 against a
#    n_frames=4 BC ckpt, trained 200 updates from random init, regressed
#    -0.083 vs BC baseline. Don't repeat.
python - <<'PY'
import torch
ckpt = torch.load("training/ae/checkpoints/bc.pt", map_location="cpu", weights_only=False)
print("BC n_frames=", ckpt.get("n_frames"), "use_belief=", ckpt.get("use_belief"))
PY

# 2. Launch self-play PPO with league opponents (random + planner +
#    aggressive + snapshot pool). 200 updates × 12 games ≈ 4-6 h on T4.
#    The shipped bc.pt is n_frames=4 → pass --n-frames 4.
python training/ae/train_ppo.py \
    --bc-checkpoint training/ae/checkpoints/bc.pt \
    --out training/ae/checkpoints/ppo-selfplay-v2.pt \
    --opponents league \
    --eval-opponents league \
    --snapshot-interval 10 \
    --snapshot-pool-size 5 \
    --updates 200 \
    --games-per-update 12 \
    --eval-games 12 \
    --n-frames 4

# 3. Deploy (only if local eval beats BC baseline 0.4866 against mixed)
cp training/ae/checkpoints/ppo-selfplay-v2.pt ae/models/bc.pt
echo hybrid > ae/src/.ae_mode      # keep the safety-veto wrapper
til build ae ppo-selfplay-v2
til test ae ppo-selfplay-v2
til submit ae ppo-selfplay-v2
```

### ppo-selfplay-v1 result (16 May, 4h GPU) — BC WARM-START WAS SKIPPED

| Eval | Score | Notes |
|---|---:|---|
| BC checkpoint (training/ae/checkpoints/bc.pt, n_frames=4) | **0.4866** | Starting point we should have warm-started from (local, mixed, 12 games) |
| ppo-selfplay-v1 (epoch=150, n_frames=1) — local mixed | **0.4040** | -0.083 vs BC. Local. |
| ppo-selfplay-v1 vs league (training-time peak) | 0.4362 | Misleading: harder opponent distribution |
| **ppo-selfplay-v1 — CLOUD (16/05 13:18)** | **0.305 / 0.851** | -0.250 vs hybrid-v3, 0/30 errors |

Training command had `--n-frames 1` (my recommendation in NOTES.md, copied
incorrectly from the bc-belief example), but BC checkpoint is `n_frames=4`.
`load_actor` prints a WARN and skips state_dict load when shapes mismatch.
PPO trained from random init for 200 updates against league opponents.

That ppo-selfplay-v1 reached 0.4040 against mixed from random init is
actually a positive signal for the self-play *training loop itself* — it
produced a policy that holds its own against league. But it's below the BC
starting point, so we expected cloud to regress vs hybrid-v3 (which it did,
0.305).

**CRITICAL POSITIVE FINDING — gap tightening**: local-cloud gap on v1 was
**0.099** (0.404 local → 0.305 cloud), vs **the structural ~0.23 gap across
all 9 prior AE submissions**. This is the first intervention that has
materially moved the gap. The training distribution (league self-play
opponents instead of random+planner) appears to be the lever. This makes the
v2 retry — same setup with proper BC warm-start — meaningfully more
interesting:

- v2 starts from BC's 0.4866 local-mixed baseline (not random init).
- Self-play training adds whatever lift it added in v1 (~0.40 over random init).
- If the gap stays at ~0.10, projected v2 cloud is **0.45-0.55** — first
  AE submission to plausibly beat hybrid-v3 (0.555) since 14 May.

### ppo-selfplay-v2 — SHIPPED 16 May 18:09 SGT, REGRESSED to 0.436/0.857

Training summary:
- `warm-started actor from training/ae/checkpoints/bc.pt (n_frames=4, use_belief=False)` ✓
- 648,342 actor params (n_frames=4 arch).
- Best PPO eval against league saved 7 times, monotonically:
  `0.4908 → 0.5038 → 0.5327 → 0.5333 → 0.5982 → 0.6428 → **0.6601**` at epoch 120.
- Workbench auto-restart killed tmux at update 162/200 (no fault of training).
  Best weights at epoch 120 preserved on disk.
- Healthy signals throughout: `pi_loss` consistently negative, `v_loss`
  dropping (0.37 → 0.13-0.20), `entropy` 0.10 → 0.25 (gentle exploration
  pickup), `pool=5` reached at update 50 and stayed full.

Full results table:

| Eval setup | Score | Notes |
|---|---:|---|
| BC checkpoint vs mixed (pure policy) | 0.4866 | Starting point |
| ppo-selfplay-v1 vs mixed (pure policy) | 0.4040 | From-scratch (no warm-start) |
| **ppo-selfplay-v2 vs mixed (pure policy)** | **0.5747** | +0.088 over BC, +0.171 over v1 |
| ppo-selfplay-v2 `til test` (hybrid wrapper) | **0.7068** | Wrapper adds ~0.13 |
| **ppo-selfplay-v2 — CLOUD (16/05 18:09)** | **0.436 / 0.857** | -0.119 vs hybrid-v3, but +0.131 over v1, 0/30 errors |

### What we learned (negative result, but informative)

**v2 vs v1 (+0.131 cloud, 0.305 → 0.436)**: BC warm-start + self-play
opponent curriculum is strictly better than self-play from random init. This
piece of the workshop's prescription was correct.

**v2 vs hybrid-v3 (-0.119 cloud, 0.555 → 0.436)**: but it doesn't clear the
hybrid-v3 ceiling. Self-play training improves the policy, but not enough
to compensate for the local-cloud gap.

**The v1 "gap tightening" was an artifact**. v1's local-cloud gap of 0.099
looked like a breakthrough, but the actual local→cloud transformation was:

| Submission | Local (mode) | Cloud (hybrid wrapper) | Gap |
|---|---|---:|---:|
| v1 | 0.404 (pure policy) | 0.305 | 0.099 |
| v2 | 0.575 (pure policy) | 0.436 | 0.139 |
| v2 | 0.7068 (til test, hybrid) | 0.436 | **0.271** |
| hybrid-v3 | 0.774 (til test, hybrid) | 0.555 | 0.219 |

The apples-to-apples comparison (both in hybrid mode) shows the gap is
**0.27 for v2**, *larger* than hybrid-v3's 0.22. The v1 "tightening" was
an illusion: v1's pure-policy raw score (0.404) happened to be closer to
its hybrid-wrapped cloud score (0.305) because v1's weak policy meant the
hybrid wrapper was carrying more of the cloud performance proportionally.
With v2's stronger raw policy, the wrapper contributes less in relative
terms, and the structural gap re-emerges.

**Conclusion**: training-side interventions (BC, league opponents, self-play
snapshots) can move the *local* score by significant amounts but the
local-cloud gap is structural to the hidden eval distribution, not to our
training opponent distribution. Memory-augmented BC (bc-belief) tried to
close it through state representation and failed. League + self-play tried
to close it through opponent diversity and failed. The gap survives both.

### AE re-parked at hybrid-v3 (0.555/0.849)

Remaining AE moves and realistic ceilings:

| Move | Effort | Realistic cloud | Notes |
|---|---|---:|---|
| Longer v2 training (300-500 updates) | 4-8h | 0.45-0.50 | Best v2 eval was at epoch 120 of 200; further training likely drifts. |
| Larger snapshot pool (10-20) | 4-6h | 0.45-0.55 | More opponent diversity. Might shift gap a little. |
| MCTS-light at inference | 1-2 days | 0.55-0.65 | Only approach that *doesn't* depend on opponent distribution. Speed budget risk. |
| Re-introduce belief-map + self-play | 4-6h training | 0.40-0.55 | bc-belief was killed by random opponents; self-play might save it. Untested. |

None of these has a strong path to clearing hybrid-v3 by a margin worth a
submission slot. **Recommend pivoting to CV Phase C** (active workstream,
hypothesis-aligned recipe TBD, +0.03-0.07 cloud lift band) — higher
EV-per-hour than any remaining AE move with 7 days left.

### Workflow improvement — SSH access from Mac

`ssh workbench` is now configured on the Mac dev box (16 May 14:25 SGT). Lets
Claude run read-only checks (`tail` the training log, `nvidia-smi`, file
listings) on Workbench without copy-paste round-trips. Writes still go
through the user. IP is `34.124.176.110` (changes on instance restart;
`@Tech` can grant static IP if SSH proves persistent value).

Public key in `~/.ssh/authorized_keys` on Workbench. Setup matches the doc
in the upstream wiki ("Power users" → SSH).

### Decision tree

- **Cloud ≥ 0.58**: self-play hypothesis validated. Iterate: longer training, larger pool, or selfplay-only mode.
- **Cloud 0.54-0.58**: marginal — within ±0.04 noise of hybrid-v3. Run a second seed before deciding.
- **Cloud < 0.54**: self-play didn't help. Roll back to hybrid-v3 and pivot remaining hours to NLP/CV.

### Realistic expected lift

Honest range based on our 9-point local→cloud history (~0.23 structural gap): **0.58-0.65 cloud**. Probably not 0.7 (leader 0.711) — top teams may have access to demonstrations or hidden-eval replays we don't. Self-play is still our highest-EV remaining AE move because it's the only intervention that materially changes the training opponent distribution.

### Why we did NOT do MCTS-light / inference-time search

Considered. The workshop materials contain zero MCTS / model-based search content (only BFS/DFS/A* in notebook 03 for fully-observed grids). Combined with no Discord hints, the 0.7+ teams almost certainly aren't doing MCTS — they're doing what the workshop teaches. Self-play is also strictly cheaper (no inference-time speed budget risk; existing hybrid wrapper composes unchanged).

Lookahead search remains a v2 idea if self-play hits a hard ceiling.

### v2 ideas if v1 self-play lands well

- **Parameter sharing**: train all 4 agents with one shared policy at training time. 4× the data per game, deployed inference unchanged. Requires refactoring `collect_rollouts` to collect transitions from all controlled agents, not just `our_agent`. Workshop notebook 05 MD 19 is the reference.
- **Larger snapshot pool with weighted sampling** — bias toward older snapshots to prevent cycle-collapse.
- **Belief-map architecture + self-play**: the bc-belief failure was diagnosed as "memory + weak opponents = overfit". Memory + self-play opponents is structurally different and untested.

---



## TL;DR for the next person reading this

- **Shipped tag**: `hybrid-v3` at **0.555 / 0.849**. Team-best, top-quartile (leaderboard top is 0.711).
- **Architecture**: PPO policy (`ppo-v1` weights) wrapped by `HybridAEManager` (heuristic safety-veto + top-K policy cascade) + heuristic dominant-action shortcut (opportunistic enemy-kill).
- **Mode selection**: `AE_MODE` env var in `ae/Dockerfile` *or* `ae/src/.ae_mode` file fallback (`hybrid` | `policy` | `heuristic`). Currently `hybrid`.
- **Weights on Workbench**: `ae/models/bc.pt` ← `~/ae-checkpoints-backup/deployed-bc-v1.pt` (the ppo-v1 actor).
- **Local→cloud gap**: ~0.23, structural across 11 submissions. Never closed.
- **Status**: parked. Four post-hybrid-v3 attempts all regressed (-0.051 to -0.268). AE-side return is now far below NLP-side return per remaining qualifier hour.

### Full AE submission ledger (cloud scores)

| Tag | Cloud score | Cloud speed | Note |
|---|---:|---:|---|
| baseline | 0.051 | 0.856 | Periodic FORWARD + bomb-every-20 |
| planner-v1 | 0.445 | 0.788 | First stateful planner |
| planner-v2 | 0.501 | 0.771 | Bomb timer 4→3 + bounded escape + soft threat |
| planner-v3b | 0.499 | 0.853 | Multi-source BFS + blast cache + uvloop |
| bc-v1 | 0.364 | 0.856 | BC of planner-v3b — regressed |
| ppo-v1 | 0.507 | 0.861 | Mixed-opp PPO from BC warm start |
| ppo-v2 | 0.489 | 0.854 | Frame-stacked + varied-maps — regressed |
| hybrid-v2 | 0.545 | 0.863 | Hybrid wrapper (policy + safety veto) — new high |
| hybrid-v3 | **0.555** | **0.849** | **+ top-K cascade + opportunistic kill — SHIPPED** |
| bc-belief-hybrid | 0.287 | 0.846 | 16×16×11 belief-map architecture (704k params) — memory hypothesis REJECTED |
| hybrid-conf50 | 0.504 | 0.857 | `AE_HYBRID_CONF=0.5` confidence gate — too aggressive |

---

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

### Cloud probe — bc-belief in hybrid mode (RESULT: 0.287, MEMORY HYPOTHESIS REJECTED)

Submitted as `bc-belief-hybrid` 15/05 11:46. Result: **0.287 / 0.846**
— regressed -0.268 vs hybrid-v3 (far outside ±0.04 cloud noise; not
explainable as variance).

Gap analysis:
| Model | Local | Cloud | Gap |
|---|---:|---:|---:|
| bc-v1 (no belief)     | 0.689 | 0.364 | 0.325 |
| bc-belief             | 0.656 | 0.287 | 0.369 |

Belief input WIDENED the local→cloud gap by +0.044, the opposite of
what the hypothesis predicted. Cleanest explanation: the 704k-param
network with belief input found *more* ways to overfit the
random-opponent local distribution than the 149k-param bc-v1 did. The
belief tensor encodes "where have I been, what's visible, where are
enemies recently" — all features that depend heavily on opponent
behavior. Locally we trained against random opponents whose movement
patterns produced specific belief-tensor distributions; cloud
opponents produce different ones, and the policy's belief-conditioned
responses fire wrong.

Local hybrid-bc-belief ≈ bc-belief solo (0.646 vs 0.656) had already
shown the safety vetoes were firing very often — the policy was
proposing actions the heuristic kept overriding. Cloud confirmed:
the policy's contribution was net-negative when it *did* get through.

Decision: **roll back to `hybrid-v3` shipped** (0.555/0.849, still
the team-best). Don't run PPO with belief — the BC-stage data shows
the architecture is making transfer worse, and PPO can't reverse
that without retraining the BC backbone on a different opponent mix.

The pre-set decision tree (<0.45 → roll back) made this call cleanly
before any time was sunk on the 3-6h PPO run.

### What we learned (and what to NOT revisit)

- **Memory-augmented BC against random opponents amplifies the
  local→cloud gap.** This is the opposite of the prediction. The
  bigger model and richer state input give BC more ways to overfit.
- **Belief-map architecture itself is not categorically broken.** It
  fit val_acc 0.897 vs bc-v1's 0.874 — it learns *better* against the
  local opponent distribution. The failure is the transfer.
- **Local game score remains an unreliable cloud predictor.** Local
  bc-belief-hybrid (0.646) and local hybrid-v3 (0.774) differ by
  0.128; cloud-wise they differ by -0.268. ~2× amplified.

Do not revisit:
- Bigger BC networks with rich state inputs against random opponents.
- Larger BC datasets *of the same opponent distribution* — won't help
  the transfer problem.

### Remaining options (post-rollback)

| Option | Effort | Expected cloud lift | Notes |
|---|---|---|---|
| Veto-tuning A/B on hybrid-v3 (`AE_HYBRID_CONF=0.5`, etc.) | 30 min | ±0.02 | Cheap; bounded upside |
| Bigger BC with **league** opponents only (not random) | 3-4 h | unknown, possibly +0.05 | The right fix per the diagnosis: if random opponents are the overfit source, train without them. Plumbing already in place (`AggressivePlannerOpponent`, `--opponents league`). |
| MCTS-light at inference | 1-2 days | +0.05 to +0.15 | Non-learned; doesn't suffer from opponent distribution shift. Real engineering — needs partial env forward model. |
| **Pivot to NLP v7-finetune** | 1 day | +0.03 to +0.06 *qualifier* (NLP is 20%) | Higher qualifier-impact-per-hour than any remaining AE move. Recommended. |

### `hybrid-conf50` (15/05 17:59) — confidence-gate A/B, REGRESSED

Same ppo-v1 weights as hybrid-v3, hybrid wrapper, plus
`ENV AE_HYBRID_CONF=0.5` so the policy is only used when its softmax
top-action probability ≥ 0.5; otherwise the heuristic action is taken.

| Metric | hybrid-v3 | hybrid-conf50 | Δ |
|---|---:|---:|---:|
| Cloud score | 0.555 | **0.504** | **-0.051** |
| Cloud speed | 0.849 | 0.857 | +0.008 |
| Local (6-game) | 0.774 / 0.646 | 0.719 | within band |
| Per-call latency (local) | ~3.1 s/round | 3.08 s/round | flat |

Local 0.719 was solidly in hybrid-v3's local distribution upper half,
which mis-suggested a positive direction. Cloud regression was -0.051
(just outside ±0.04 noise) — small but real. Read: the confidence
gate at 0.5 was too aggressive; it kicked the policy out of decisions
where its top-2 actions were close (softmax 0.4-0.5 band) but the
policy's choice was actually correct on the cloud distribution. The
heuristic fallback in those moments dragged us down.

**AE is parked at `hybrid-v3` (0.555 / 0.849).** Four
post-hybrid-v3 attempts in total — bc-belief-hybrid (-0.268),
bc-belief-policy (not shipped after the bc-belief diagnosis),
hybrid-conf50 (-0.051), and ppo-v2 from earlier — all regressed.
The 0.555 ceiling is firm for this approach. Top of leaderboard is
0.711; the remaining 0.16 gap is unlikely to close without either
the MCTS-light path (1-2 days new engineering, no guarantees) or a
fundamentally different training pipeline (already tried league-style
plumbing, the architecture-level fix was the failed one).

For the qualifier-deadline timeline, AE-side returns are now
diminishing far below NLP-side returns (v7-finetune lifting accuracy
by +0.034 per submission). Recommend redirecting remaining effort to
NLP / CV polish.

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

**`hybrid-v3` — official 0.555 / 0.849 (14 May 19:26 SGT, 0/30 errors).** Team-best AE score; top-quartile (leaderboard top is 0.711).

Architecture: `PolicyAEManager` running ppo-v1 actor weights, wrapped by `HybridAEManager` (heuristic safety-veto + top-K policy cascade), with `AEManager`'s heuristic dominant-action shortcut now firing on adjacent enemy *agents* (not just bases). Mode selection is via `AE_MODE` env var in `ae/Dockerfile` (or fallback `ae/src/.ae_mode` file): `hybrid` | `policy` | `heuristic`. Currently `hybrid`.

Cloud-deployed weights: `ae/models/bc.pt` ← `~/ae-checkpoints-backup/deployed-bc-v1.pt` (ppo-v1 actor). Restored after `bc-belief-hybrid` was tested.

**Three consecutive AE highs lifted the floor by +0.048 across submissions** (ppo-v1 0.507 → hybrid-v2 0.545 → hybrid-v3 0.555). After hybrid-v3, four follow-up attempts all regressed:
- `bc-belief-hybrid` (15/05, 0.287): 16×16×11 belief-map architecture — bigger network amplified the local→cloud gap rather than reducing it. Memory hypothesis as implemented through BC is dead.
- `bc-belief-policy` (local-only, 0.663): same checkpoint in pure-policy mode. NOT submitted after `bc-belief-hybrid` failed.
- `hybrid-conf50` (15/05, 0.504): `AE_HYBRID_CONF=0.5` confidence gate. Too aggressive — kicked the policy out of cloud-correct 0.4-0.5 softmax decisions.
- `ppo-v2` (14/05, 0.489): frame stacking + varied maps. Widened the gap.

**Cumulative finding across v1 → v3b → bc-v1 → ppo-v1 → ppo-v2 → hybrid-v2 → hybrid-v3 → bc-belief-hybrid → hybrid-conf50**: the local→cloud gap is ~0.23 for every approach we've tried — heuristics, BC, mixed-opp PPO, frame-stacked PPO + varied maps, hybrid wrappers with various gates, state-augmented BC. **The gap is structural to the hidden eval distribution and none of our training-side or inference-side interventions have moved it.** The hybrid-style wrapper *moved the floor* by +0.048 but the gap stayed constant; the bc-belief attempt actually *widened* it.

Previous shipped tags: `hybrid-v2` (0.545/0.863, 14/05 14:55) → `ppo-v1` (0.507/0.861, 14/05 04:36) → `planner-v3b` (0.499/0.853, 13/05 23:42).

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
[bake bug]  14/05 14:30        —       —       —       AE_MODE=foo til build doesn't bake AE_MODE; the three v1 builds (policy-fast-v1, hybrid-v1, heuristic-restore) shipped identical images. Fixed via ENV AE_MODE in Dockerfile + .ae_mode file fallback.
policy-fast-v2     14/05 14:42 0.425   0.859   0/30    ppo-v1 weights + speed fixes (single-thread torch, inference_mode, warmup, preallocated tensors). Regressed -0.082 from ppo-v1 (cloud variance on 30-game sample). Speed flat — confirmed evaluator-bound at ~0.86.
hybrid-v2          14/05 14:55 0.545   0.863   0/30    NEW HIGH. First structurally new approach since ppo-v1. HybridAEManager wraps PolicyAEManager with safety vetoes (illegal / unsafe-bomb / step-into-blast / frozen-stay). +0.038 over ppo-v1.
heuristic-restore-v2 14/05 15:02 0.502 0.854   0/30    Pure heuristic (planner-v3b + TILE_RESPAWN 40→20 + enemy_agent eviction). Noise vs planner-v3b; confirms heuristic-only ceiling.
hybrid-v3          14/05 19:26 0.555   0.849   0/30    NEW HIGH (still shipped). + top-K policy cascade (try policy's #2/#3 if vetoed before falling back to heuristic) + opportunistic enemy-kill in heuristic dominant-action shortcut. +0.010 over hybrid-v2.
bc-belief-hybrid   15/05 11:46 0.287   0.846   0/30    MEMORY HYPOTHESIS REJECTED. 16×16×11 belief-map architecture (704k params), bc-belief weights, hybrid wrapper. Regressed -0.268 vs hybrid-v3. Bigger network amplified local→cloud gap (0.36 vs 0.33) by overfitting random-opponent local belief distributions that don't transfer.
hybrid-conf50      15/05 17:59 0.504   0.857   0/30    ppo-v1 weights + hybrid + AE_HYBRID_CONF=0.5 (only use policy when softmax top ≥ 0.5). Regressed -0.051 (just outside ±0.04 noise). Gate was too aggressive — kicked policy out of cloud-correct 0.4-0.5 softmax decisions.
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
policy-fast-v2 14/05 til test  0.654 (1 run)                ppo-v1 weights + speed-fixed PolicyAEManager. Cloud 0.425 — variance.
hybrid-v2   14/05 til test     0.774 (1 run)                Hybrid wrapper around ppo-v1 weights. Cloud 0.545 = +0.038 over ppo-v1.
heuristic-restore-v2 14/05 til test 0.787 (1 run)          Pure heuristic + the v2 tweaks. Cloud 0.502.
hybrid-v3   14/05 til test     —                            Same family as hybrid-v2 + top-K + opportunistic kill. Cloud 0.555 (current shipped tag).
bc-belief   15/05 BC train     val_acc 0.897 (vs bc-v1 0.874, +0.023); local 0.656 / 0.663 (hybrid / policy). 704k params, 20 epochs, ~3 min CPU.
bc-belief-hybrid 15/05 til test 0.646 (1 run)               Hybrid wrapper around bc-belief. Cloud 0.287 — memory hypothesis rejected.
hybrid-conf50 15/05 til test   0.719 (1 run)                ppo-v1 + hybrid + AE_HYBRID_CONF=0.5. Local in upper half of hybrid-v3 distribution; cloud 0.504 (-0.051) — gate too aggressive.
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
