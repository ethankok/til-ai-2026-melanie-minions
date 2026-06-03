# AE — notes & history

> **How to read this file.** This is the AE working memory: what we've tried,
> what worked, what failed, and *why*, so we don't burn time re-running dead
> ends. **Read the `## Read this first` digest below before doing any AE work** —
> it is the current, distilled state. The reverse-chronological per-session log
> (raw numbers, reproduction commands) lives in a separate file,
> [`ae/NOTES-ARCHIVE.md`](NOTES-ARCHIVE.md). The archive is intentionally
> redundant with the digest; you only need it when you want the exact
> per-submission detail behind a claim here.
>
> _Digest last refreshed: 4 June 2026 — ⚠ CLOUD EVAL CHANGED (org opponent swap; all cloud
> numbers re-based, byte-identical champion 0.671→0.414; see "Current state (4 June 2026)").
> Also this session: respawn-loophole fix + pandemonium-v2 run launched + strategic pivot to a
> stronger planner. Prior 2 June: Stage-B foreign-curriculum line run end to
> end: NEW BEST `confpol-semis2b-u75` = 0.671 cloud, melee-gate-selected + cloud
> A/B-validated vs native-u100 0.661; Stage B CONCLUDED — peak found, later rungs
> overfit, training stopped. Also added: foreign melee eval (Stage A), peroxide
> vendored EVAL_ONLY, public-repo scrape, AE_FORCE_CPU. Prior 1 Jun: eval went
> DETERMINISTIC → variance farming dead; opening-book closed; competitor teardown)._

---

## Read this first

### Current state (4 June 2026) — ⚠ CLOUD EVAL CHANGED; RL washed out; pivot to a stronger PLANNER

- **🛑 THE CLOUD EVAL CHANGED ~4 Jun (org's pre-announced Semis opponent swap). ALL prior
  cloud numbers (0.671 / 0.661 / 0.634 / 0.626 …) are STALE.** Proof: re-submitting the
  BYTE-IDENTICAL `confpol-semis2b-u75` image (no rebuild) scored **0.414** vs its **0.671**
  on 2–3 Jun (**−0.257**). Speed also dropped uniformly (~0.84 → ~0.74) across heterogeneous
  models including the unchanged-inference incumbent — accuracy *and* speed dropping together
  = an environment change, not a per-model regression. Still deterministic (double-submits
  return identical scores → new opponents are fixed/seeded). **Action: re-rank the whole field
  on the new eval (1 submit each — still deterministic).** Possibly infra-wide — other tasks
  should re-check a known image too.
- **New-eval re-rank (4 Jun, 0/30 err each):**
  | image | NEW score | OLD score |
  |---|---|---|
  | confpol-semis2b-u75 (deploy champion) | **0.414** | 0.671 |
  | confpol-u860 (rebuild) | **0.414** | 0.626 |
  | respawnfix-u400 | 0.312 | — |
  | respawnfix-u720 | 0.286 | — |
  | respawnfix-u580 | 0.259 | — |
  - **⚠ CORRECTED (4 Jun, bare-heuristic submit): the RL consultant did NOT wash out — it
    adds +0.152.** `AE_MODE=heuristic` (bare planner, same flags) = **0.262**, so the two
    champions at 0.414 are **two STRONG consultants coinciding**, NOT a collapse to the
    heuristic. The earlier "washed-out / tie-at-backbone" read was an inference from the exact
    tie and is FALSIFIED. (The submit comparison is also the canary: 0.414 ≠ 0.262 proves the
    policy loaded + helps; the `til test` log grep was empty/moot.) **Field on the new eval:**
    bare heuristic 0.262 (weak) < respawnfix rungs 0.259–0.312 (weak consultants, ~heuristic
    level — the from-scratch run made INFERIOR policies) < the two foreign-curriculum champions
    **0.414 (+0.152)**. **KEEP the policy; do NOT ship bare heuristic (−0.15). Deploy stays
    `confpol-semis2b-u75` = 0.414.** "Heuristic alone is weak on the new opponents" = confirmed.
  - **Semis is melee PLACEMENT, not this absolute number** → keep gating on melee regardless
    (and our foreign pool is now arguably a *better* proxy than the retired BenBots).
- **This session also shipped: the respawn-loophole fix** — `train_ppo.py`
  `AdaptiveRewardShaper._health_delta_bonus` now clamps the agent-health delta negative-only
  (`min(0.0, Δhealth)`) so the 0→100 respawn jump no longer refunds a life's damage penalty
  (6 TDD tests, `test/test_reward_shaper_respawn.py`). Launched `pandemonium-v2-respawnfix`
  (fresh from-scratch CNN-PPO, CPU-forced; patched `run_pandemonium_v1.py` to add
  `--checkpoint-every 25` for a dense `-u<N>` ladder). Run is HEALTHY (eval −0.22→0.66,
  plateaued; entropy in the 0.06–0.15 sweet spot ~u500–600) but the new-eval rungs above
  already say it won't beat the heuristic on cloud. Gate phase-1 rungs on melee placement
  before the final call.
- **🧭 STRATEGIC PIVOT (4 Jun, agreed): the ceiling-raiser is a STRONGER PLANNER, not more RL.**
  Evidence: curry's **0.715 is a hand-coded forward-sim planner, not RL** (their RL failed
  transfer too); our RL has been the LOWER-ceiling / fragile path (pure ~0.43–0.51; consultant
  +0.045 that just washed out on the new eval). The headroom above our **greedy one-step** scorer
  is *multi-step value*, best captured by **forward-sim SEARCH** (transfers — reasons about the
  *game*, not the *opponents*), not opponent-overfitting RL. **Three directions for the next
  session (1 is the foundation for 2):**
  1. **Forward-sim + respawn/stun ("farming-race") plan scorer** — curry's 0.715 mechanism. We
     built the self-plan half (`AE_PLAN_RESCORE`, cloud-dead ALONE) but OMITTED the stun-downtime
     pricing that is curry's actual edge. Highest evidence.
  2. **RL anchored to the planner** (heuristic/planner as BC teacher + RL residual/opponent —
     curry's own semis plan): "some form of RL" but with a real answer to "why won't it overfit
     like the last ten times" — pure-RL's target (the cloud opponents) is moving/unlearnable;
     a planner has no such dependency.
  3. **Finish `pandemonium-v2-respawnfix` → gate on melee placement** before discarding it; do
     NOT judge it by the now-negative cloud number.
- **Phase-0 farming-race diagnosis (4 Jun, `diagnose_farming_race.py`, all_farmer n=24):** verdict =
  **hypothesis (ii) base-rush dominant** — `mean_final_base_health=0.0` (base dead EVERY game),
  `own_base_destroyed −1680`/`base_damage −480`, placement 5.875 / margin −406.7. Freeze moderate
  (9.7 ticks, secondary) and offense is actually high (`destroy_enemy_base +3150`, but
  `self_damage −7798`) → **not** out-collected (hypothesis-iii STOP does NOT fire). → Build Phase B
  with the **base-threat trigger as the dominant `_posture` lever** (defense+tether); de-emphasize A.
- **Phase-A `AE_STUN_TAX` gate verdict (4 Jun, melee A/B, deploy confpol-semis2b-u75, n=3×1, 3 brackets):**
  **FLAT — not shipped, `AE_STUN_TAX=0`** (as predicted, redundant with the calibrated
  `PATH_THREAT_PENALTY=2.0`). `all_farmer` placement **2.17→2.17 (unchanged)**; adversarial 3.19→3.03,
  semis_mixed 1.03→1.06; margins nudge ~+15 on loss brackets but inside n=3 noise. Smoke not cratered
  (`top_seed_proxy` 0.472). Code stays (default-OFF, byte-identical); the lever folds into Phase B's
  `fortress_threat_mult` anyway. → Phase B is the real bet.
- **Phase-B `AE_FORTRESS` gate verdict (4 Jun, melee on-vs-flag-off, deploy confpol-semis2b-u75, n=3×1):**
  **KILLED — NOT shipped, `AE_FORTRESS=0`.** Fortress regresses EVERY bracket: all_farmer **2.17→4.69**,
  adversarial **3.19→4.25**, semis_mixed **1.03→1.50** (worst-bracket placement 3.19→4.69); the reject smoke
  also cratered the cloud-predictive suite (`top_seed_proxy` 0.472→0.120, while defensive suites held).
  **Why the Phase-0 lever didn't transfer:** base-rush was real on the BARE heuristic (placement 5.875) but
  the DEPLOYED confpol consultant already handles all_farmer (2.17 flag-off); forcing a defensive posture on
  top makes the strong agent passive → out-farmed, placement tanks. Same action-policy wall as
  contention/plan-rescore (helps the wrong regime). Default-OFF ships nothing; deploy stays
  `confpol-semis2b-u75`. Code retained (flag-off byte-identical, TDD-guarded) but both farming-race levers
  (`AE_STUN_TAX`, `AE_FORTRESS`) are DEAD on the melee. **Next ceiling-raiser must be a different shape** —
  a planner that improves WITHOUT a global posture switch, or Dir-2 (RL anchored to the planner) — not more
  heuristic posture levers.

### Current state (2 June 2026)

- **🎯 TL;DR for the next session:**
  1. **The cloud AE eval is now DETERMINISTIC** (org seeded it 1 Jun). 1 submit =
     true score; **variance farming is dead** (see *Measurement reality*). Re-rank
     by submitting old image tags — they persist on the Workbench
     (`docker images | grep ae`; `til submit ae <tag>` needs no rebuild).
  2. **🏆 NEW BEST = `confpol-semis2b-u75` = 0.671** (cloud, 2 Jun A/B; beats
     prior best `confpol-native-u100` = 0.661, which RE-submitted to 0.661 exactly
     same-day → same-conditions, deterministic eval intact, **+0.010 is a real
     delta**). This is the **Stage-B foreign-curriculum** candidate (confpol-native
     line fine-tuned vs the FOREIGN non-mirror pool) — **first Stage-B rung to
     validate on cloud, and the melee gate correctly predicted it** (2nd
     melee→cloud agreement). Deploy: stage
     `training/ae/checkpoints/confpol-semis2b-u75.pt` → `ae/models/bc.pt`,
     `AE_MODE=confidence_policy_hybrid` (image `melanie-minions-ae:confpol-semis2b-u75`
     built+submitted; staged at `gs://melanie-minions-bucket-til-26/handoff/confpol-semis2b-u75.pt`).
     Magnitude is modest (+0.010); the real result is that the foreign curriculum
     TRANSFERS. **Stage B is now CONCLUDED (2 Jun): semis2b-u75 is the PEAK** —
     continued training (semis2c, real u273/u348) DEGRADED it (worst-bracket melee
     placement 3.39→3.67→5.78 on held-out brackets; over-specialized into
     all_aggressive). Training stopped. `confpol-native-u100` (0.661) is the
     fallback floor. **To beat 0.671 toward curryfarmer's 0.715 needs a NEW lever**
     (different warm-start / foreign mix / peroxide base-inference trick), not more
     of this curriculum — see the Stage B "CONCLUDED" bullet for the full trajectory.
  3. **Opening-book line is CLOSED** for the eval: v1 (heuristic gate) regressed
     to 0.584; v2 (confpol-correct gate) = 0.626 = confpol exactly → opening never
     fires on our eval spawn (∈ {13,9/3,12/12,3}). All infra kept; not the default.
  4. **Re-rank DONE (1 Jun PM):** confpol-native-u100 **0.661** » confpol-u860
     0.626 = opening-v2 0.626 > tether-v1 / confpol-native-u360 / pand-hybrid 0.605
     > heuristic-c-bomb7 0.591 > confpol-native-u200 0.559 > pand-policy 0.525 >
     heuristic-a-bomb7 0.515. Reads: the NN IS worth it (0.626 > best heuristic
     0.605); the **early gated-PPO rung wins** (u100 looked *worst* on the old
     noisy eval at 0.551 — a low-tail draw — but is the champion on the
     deterministic eval → "earlier = less overfit"). **Next: test even-earlier
     rungs (u25/u50/u75) — the inverted-U peak may be < u100** (needs a rebuild
     each: stage the .pt → `til build/submit`).
- **Phase:** Qualifiers closed. Semifinals prep runs through **2026-06-10**. We
  placed 15th on the Novice path, so the expected Semifinals Match-1 bracket is
  seeds ≈ 3/8/9/14/15/20. AE is 40% of the score and our single biggest lever —
  this is where a match is won or lost. **Semis = Novice path = the fixed
  seed-42 map** (see the determinism section below — this is now the top lever).
- **Shipped Docker default** ([Dockerfile](Dockerfile)) — **`AE_MODE=confidence_policy_hybrid`
  = confpol-u860** (heuristic-first C+bomb7 + PPO consultant; farmed mean 0.634).
  - **Still requires staging `pandemonium-v1-best-u860.pt` → `ae/models/bc.pt` on
    the Workbench** (gitignored). If bc.pt is absent → degrades to bare C+bomb7.
  - C+bomb7 profile + base-tether + `PYTHONHASHSEED=0` baked in.
  - **Canary (`til test`):** `AE policy loaded … epoch=860` (NOT "falling back to
    heuristic").
- **❌ Opening book (opening_hybrid) — SHIPPED then REVERTED 1 Jun (cloud-negative
  over confpol).** Cloud farm of opening+confpol: **mean 0.584 (n=9, CI
  [0.564,0.604]) vs confpol-u860 0.634** — the opening *regressed* confpol by
  ~0.05 (non-overlapping CIs). The confpol diagnostic
  ([../training/ae/diagnose_opening_over_confpol.py](../training/ae/diagnose_opening_over_confpol.py))
  nailed the mechanism: the gate was locked over the **bare heuristic**, but
  confpol is a **better opener**, so the fixed opening overwrites confpol's good
  early play. Over confpol the gate is **net −0.014 local** — 3 of 4 enabled
  spawns flip negative (only `9,13`, where confpol still opens poorly at 0.211,
  helps +0.086). Local sign matches cloud sign → not a transfer gap, a
  planner-mismatch in the gate. **The +0.067 lift was real but only over the
  *bare heuristic*** (`AE_MODE=opening_hybrid AE_OPENING_PLANNER=heuristic` still
  gives it; kept in tree, not default).
- **🔬 Opening book v2 — confpol-correct gate, BUILT, TO FARM (1 Jun).** Re-swept
  the gate OVER CONFPOL across all horizons (`sweep_openings.py --planner confpol`).
  The confpol-correct gate is **almost disjoint** from the heuristic one: enables
  **9,13 (H20, +0.174 z9.7) / 2,6 (H8, +0.119 z4.6) / 6,2 (H12, +0.117 z2.8)**,
  disables 13,9 / 3,12 / 12,3. Why different: confpol opens 0/2/5 *well* (no room)
  but opens **3/4 *worse* than the heuristic** (baselines 0.33/0.35 vs the
  heuristic's 0.34/0.52), so a *short* opening fills that gap. **Local +0.068 over
  confpol**, all enabled spawns complete 100%, strong z. Baked into
  [ae/src/openings_gate.json](src/openings_gate.json) (now the confpol-specific
  gate). **This is the clean farm experiment** — it fixes the planner-mismatch
  that sank v1; the only remaining risk is the local→cloud *opponent* gap. Farm
  `AE_MODE=opening_hybrid` (confpol planner) vs confpol-alone; promote only if it
  clears 0.634. If a *correctly-specified* gate also fails on cloud, fixed
  openings simply don't transfer → close the line. **Lesson: lock opening gates
  against the actual deployed planner; openings only pay where it opens poorly.**
- **Honest cloud performance: no AE config has a variance-farmed mean
  meaningfully above ~0.59**, except confpol-u860. The famous heuristic "highs"
  are all upper-tail single draws, not means:
  - `heuristic-A-vf1` **0.613** — true farmed mean ≈ 0.575–0.599.
  - `ppo-full-rl-v1-hybrid` **0.638** (protected leaderboard max) — forensically
    the **heuristic via silent fallback**, not real PPO.
  - Treat ~0.59–0.60 as the real incumbent bar. Do not chase 0.613/0.638 as
    stable means.
  - **`pandemonium-confpol`@u860 farmed mean 0.634 (n=13, 95%CI [0.609,0.659])
    — the best AE artifact.** Pandemonium policy as a low-confidence *consultant*
    on the heuristic. First AE config with a FARMED mean above the bar.
    **Use the u860 checkpoint, NOT the final one:** the phase-1-final u1400
    checkpoint (local eval 0.79) farmed only 0.600 — more training OVERFIT the
    local opponents and lost the cloud edge. Never select an AE checkpoint by
    local eval.
- **confpol-native (gated PPO, train==deploy): MATCHED u860, no lift (1 Jun).**
  A confidence-gated raw-policy PPO trained only on the heuristic's low-conf
  ticks, warm-started from u860. n=1 cloud: u100 0.551 / u200 0.621 / u360 0.624
  — same ~0.60–0.63 band, no rung clears the floor. **Deploy stays confpol-u860
  (0.634).** Run was still training at u507/2976 when shelved — **fine to
  `pkill -f run_confpol_native.py`** (verdict in; eval flat ~0.56, entropy
  collapsed). Ladder `confpol-native-u*.pt` on disk if a deeper farm is wanted;
  prior says it ties/loses.
- **🔭 TOP LEVER — first exploitation BUILT + SHIPPED (1 Jun): divergence-gated
  opening book.** Inspired by competitor `curryfarmer`/royal-recruits (0.715/0.807,
  who exploits novice determinism deeply). Offline beam search generates per-spawn
  item-farming openings ([../training/ae/opening_sim.py](../training/ae/opening_sim.py),
  `gen_openings.py`); a divergence-gated wrapper replays them then hands to the
  planner ([src/opening_hybrid_manager.py](src/opening_hybrid_manager.py)).
  **Key finding: the opening is a regression-to-the-mean operator** — it lifts
  spawns where the planner opens weakly and *hurts* spawns where it already opens
  well, so it MUST be per-slot gated. Per-slot sweep (`sweep_openings.py` →
  `lock_gate.py`) locked 4/6 spawns ON (+0.067 over the *bare heuristic*).
  **Shipped then REVERTED (1 Jun): over confpol it's cloud-negative (0.584 vs
  0.634) — the gate was validated against the wrong planner.** See the
  ❌ bullet under *Current state* and archive §4. The +0.067-over-heuristic result
  stands; the determinism *lever* is still under-exploited vs the competitor
  (distance/opponent LUTs, forward-sim planner) — but a naive item-farm opening
  doesn't help a planner that already opens well.

### Semifinals melee eval — BUILT (1 Jun, Stage A complete)

Implemented the approved redesign (`docs/superpowers/specs/2026-06-01-ae-semis-eval-design.md`):
a head-to-head **6-team melee** gate built on a **foreign (non-mirror) opponent
pool**, replacing the absolute-reward-vs-our-own-mirrors suites that mis-predict
Semis. **The old `validate_cloud_suite.py`/`multi_seed_eval.py` suites still
exist and remain valid as a cheap local pre-filter; the melee gate is the new
Semis-realistic selector.**

- **Foreign pool** ([../training/ae/foreign_opponents.py](../training/ae/foreign_opponents.py)) —
  `curry_aggro`/`curry_fortress` (vendored competitor A*+forward-sim heuristic,
  two weight-personas), `self_policy` (our raw u860 CNN-PPO in FULL control),
  `self_tactical` (our 12-way tactical-macro hybrid), `self_heuristic` (shipped
  C+bomb7), and 3 purpose-built non-mirror bots `evbot` (nominal-reward EV
  maximizer — the *opposite* of our base-underweighting calibration),
  `aggressive_proxy`, `anti_aggro_exploiter`. Curry is **vendored gitignored** at
  `training/ae/foreign/curry/` (`EXP_DISABLE_NUMBA=1 USE_PARALLEL_GOALS=0`,
  personas pinned per-instance via `set_persona`+`dynamic_persona=False`, NOT the
  process-global `EXPERIMENTAL_VARIANT` env). **Do not commit competitor code.**
- **Hard train/eval split (enforced in code):** `FOREIGN_TRAIN_OK = [curry_aggro,
  self_policy, self_heuristic, evbot]`; `FOREIGN_EVAL_ONLY = [curry_fortress,
  self_tactical, aggressive_proxy, anti_aggro_exploiter]`. Stage-B training may
  draw ONLY from TRAIN_OK (`train_ppo.py` mode `foreign_train` / preset
  `semis-foreign`; `_foreign_train_blend()` aborts if an EVAL_ONLY name leaks).
  Held-out lift that fails to appear on EVAL_ONLY = the local proxy-overfit
  early-warning (the transfer-gap proxy we never had).
- **Melee metric** (`simulate.py`): per-round `placement` (rank of 6) + `margin`
  (us − best opponent) alongside `total_reward`; aggregated to `mean_placement`,
  `win_rate`, `mean_margin`, `placement_hist`. **`total_reward` unchanged** so
  the old local-reject filter still works.
- **4 brackets** (`opponents.MELEE_BRACKETS`): `semis_mixed`, `all_aggressive`,
  `all_farmer`, `adversarial` (worst-case probe). `make_opponent` now delegates
  unknown names to the foreign factory, so `--suites semis_mixed` "just works".
- **Gate** ([../training/ae/melee_eval.py](../training/ae/melee_eval.py)) —
  per-(hash_seed,sim_seed) subprocess pattern (one bracket/worker, isolates the
  policy_manager model cache). **Minimax-plus-margin promotion (all 3 must hold):**
  (1) worst-bracket `mean_placement` ≤ incumbent's; (2) `mean_margin ≥ 0` every
  bracket; (3) `semis_mixed` `mean_score` ≥ incumbent. Run:
  `.venv/bin/python training/ae/melee_eval.py --rounds 12 --hash-seeds 0 1 2 --sim-seeds 42 137 --summary-out training/ae/data/melee-rerank.json`
  (add ladder rungs with `--confpol-ckpt LABEL=PATH`). **confpol-u860 (0.626
  cloud) stays the deploy floor regardless.**
- **First full re-rank (1 Jun, n=3 hash × 6 rounds/bracket, `melee-rerank.json`).**
  Worst-bracket `mean_placement` (lower=less exploitable) / min-margin / semis_mixed score:
  | candidate | semis_mixed | all_aggr | all_farmer | adversarial | worst-place | verdict |
  |---|---|---|---|---|---|---|
  | confpol-u860 (incumbent) | 2.33 (−60) | 2.89 (−34) | **1.00 (+14)** | **5.78 (−259)** | 5.78 | — |
  | heuristic-cbomb7 | 3.94 (−162) | 1.00 (+29) | 4.56 (−388) | 1.78 (−46) | 4.56 | not promotable |
  | self-policy-u860 | **1.00 (+165)** | **1.17 (+107)** | **1.11 (+142)** | 3.28 (−108) | **3.28** | not promotable |

  **⚠ READ THIS, do NOT over-react:** raw `self_policy` (full control) *dominates
  the local melee* (1st in 3/4 brackets, best worst-place 3.28) — **but this is a
  textbook FALSE POSITIVE of the exact overfit disease this whole project fights.**
  Cloud ground truth already says raw-policy-full-control = **0.507** (NOTES
  "Pandemonium RESULTS"), far below confpol 0.626. The melee inflates it because
  (a) `self_policy` is partly IN the opponent pool (semis_mixed/adversarial), so
  it's beating near-copies of itself, and (b) the slot-0 fixed-Novice spawn it
  trained on. **Lesson: the melee is a better RELATIVE/robustness gate and a great
  weakness map, but it is still LOCAL — never read absolute local dominance as a
  cloud promotion; the cloud submit (deterministic, 1=truth) is the arbiter.** The
  minimax-plus-margin conjunction *correctly withheld promotion* from self_policy
  (fails margin≥0 in adversarial), which is the gate doing its job.
- **Durable findings from the weakness map** (these are robust, not overfit):
  (1) the **`adversarial` bracket is everyone's worst** — confpol 5.78th/−259,
  even self_policy only 3.28th; base-siege+counter-aggressors+strong-farmer is the
  hole to close. (2) The **bare heuristic loses the farming race** (`all_farmer`
  4.56th, −388) — exactly the curryfarmer "immortality/defense-wins" dynamic.
  (3) confpol is the most *balanced* deployable (no bracket worse than ~3 except
  adversarial). **No candidate is promotable over confpol-u860 under the full
  conjunction → confpol stays deployed.** Re-run with the new cloud-best
  `confpol-native-u100` (0.661) as incumbent via
  `--confpol-ckpt native-u100=training/ae/checkpoints/confpol-native-u100.pt`.
- **Second re-rank DONE (2 Jun, `melee-rerank-u100.json`, incumbent =
  `confpol-native-u100`, + early rungs u25/u50/u75).** Same config (n=3 hash ×
  6 rounds, sim 42). Worst-bracket `mean_placement` / min-margin / semis_mixed:
  | candidate | semis_mixed | all_aggr | all_farmer | adversarial | worst-place | min-margin | verdict |
  |---|---|---|---|---|---|---|---|
  | native-u100 (incumbent) | 2.72 | 4.39 | 4.17 | 4.83 | **4.83** | −291.8 | — |
  | confpol-u860 (old deploy) | 2.22 | 3.00 | **1.33** | 5.83 | 5.83 | −275.2 | not promotable |
  | heuristic-cbomb7 | 4.06 | **1.22** | 4.78 | 2.11 | 4.78 | −418.3 | not promotable |
  | self-policy-u860 | **1.00** | 1.67 | **1.06** | 3.28 | **3.28** | −129.3 | not promotable (false-pos) |
  | native-u25 | 2.00 | 1.83 | 4.00 | 4.61 | 4.61 | −181.8 | not promotable |
  | native-u50 | 2.67 | 6.00 | 3.94 | 5.00 | 6.00 | −236.0 | not promotable |
  | native-u75 | 1.22 | 3.89 | 4.61 | 4.83 | 4.83 | −267.9 | not promotable |

  **Two real takeaways:**
  1. **The melee gate AGREES with the deterministic cloud re-rank on the one
     comparison where we have ground truth:** native-u100 (cloud 0.661) is *not
     beaten* by the old deploy confpol-u860 (cloud 0.626) — u860 fails all three
     gate criteria (worst-place 5.83 > 4.83, margin, semis_mixed). First time the
     melee gate has been cross-checked against cloud truth and it picked the same
     direction. (Caveat: n=1 comparison; self_policy still false-positives — see
     below — so this is a *weak* validation, not a license to trust local melee
     for absolute rank.)
  2. **The "inverted-U peak might be < u100" hypothesis is NOT supported locally
     → do NOT spend cloud rebuilds on u25/u50/u75.** None of the earlier rungs
     improves the worst-bracket minimax over u100; u50 is actively bad (last in
     `all_aggressive`, worst-place 6.00); u75 has a strong `semis_mixed` (1.22,
     0.89 win) but collapses in `all_farmer`/`adversarial` (4.61/4.83). u100 is
     the most *balanced* native rung. The melee says the earlier rungs are not
     worth a Workbench rebuild+submit — closes that "Next" item.
  - **self-policy false-positive reproduced exactly** (dominates 3/4 brackets,
    best worst-place 3.28) and the conjunction **correctly withheld promotion**
    (margin ✗ in adversarial −129.3). Cloud truth for raw-policy-full-control is
    0.507 ≪ confpol — the gate doing its job.
  - Durable holes unchanged: `adversarial` + `all_farmer` are everyone's worst;
    native-u100's deepest loss is the farming race (`all_farmer` margin −291.8).
  - **Net: confpol-native-u100 (0.661) stays the ship pick; nothing local
    dethrones it.** Effort should go to Stage B (close the farming-race /
    adversarial holes via the foreign curriculum), not to re-submitting old rungs.
- **Stage B — LAUNCHED 2 Jun (tag `confpol-semis2`, warm-start
  `confpol-native-u100` = the cloud-best 0.661, ~4-day CPU budget).** Command:
  `EXP_DISABLE_NUMBA=1 USE_PARALLEL_GOALS=0 PYTHONHASHSEED=0 PYTORCH_ENABLE_MPS_FALLBACK=1 nohup caffeinate -is .venv/bin/python -u training/ae/run_confpol_native.py --opponent-mix-preset semis-foreign --tag confpol-semis2 --warmstart training/ae/checkpoints/confpol-native-u100.pt > training/ae/checkpoints/confpol-semis2.run.log 2>&1 &`
  Watch `training/ae/checkpoints/confpol-semis2.log`; kill `pkill -f run_confpol_native.py`.
  Verified at launch: `warm-started actor from confpol-native-u100.pt` +
  `config: preset=full-rl ... opponent_mix=semis-foreign` (the preset no longer
  clobbers the explicit mix — `apply_preset` fix + launcher passthrough, 4995ca7).
  **Decision (2 Jun): build on the current best, not a clean A/B.** We warm-start
  from the cloud-best u100 (not u860) and fine-tune on the foreign curriculum —
  goal is the single best deployable, not a controlled comparison. (A short
  u860-warm-started `confpol-semis` run was launched first then killed before any
  rung — superseded.) Dense `confpol-semis2-u<N>.pt` ladder every 25 updates.
  - **⚠ CRASHED + RESUMED 2 Jun (~u80).** The MacBook hard-restarted at 07:24
    (MPS/Metal failure: `Unable to reach MTLCompilerService` — broken pipe).
    Surviving ladder: `confpol-semis2-u25/u50/u75.pt` + `-latest.pt` (epoch 80).
    **Resumed** warm-started from `confpol-semis2-latest.pt` under a NEW tag
    **`confpol-semis2b`** (so the original u25/u50/u75 rungs are NOT clobbered) —
    so `confpol-semis2b-u<N>` ≈ original **u80 + N** in real training terms.
    **Forced CPU (`AE_FORCE_CPU=1`)** to avoid another Metal crash on the long
    unattended run (added to `train_ppo.py`; the net is tiny + the bottleneck is
    the opponent planners, so MPS barely helped anyway). Relaunch:
    `AE_FORCE_CPU=1 EXP_DISABLE_NUMBA=1 USE_PARALLEL_GOALS=0 PYTHONHASHSEED=0 nohup caffeinate -is .venv/bin/python -u training/ae/run_confpol_native.py --opponent-mix-preset semis-foreign --tag confpol-semis2b --warmstart training/ae/checkpoints/confpol-semis2-latest.pt > training/ae/checkpoints/confpol-semis2b.run.log 2>&1 &`
    Gate rungs from BOTH ladders (semis2 u25/u50/u75 + semis2b-u*) together.
  - **🟢 STOP-CHECK 2 Jun (`melee-stopcheck.json`) → VERDICT: KEEP TRAINING, the
    foreign curriculum is WORKING.** Gated native-u100 (incumbent) vs semis2-u25
    (real u25) vs semis2b-u75 (real ~u155), 5 brackets, n=3 hash × 6 rounds.
    Worst-bracket `mean_placement` (lower=less exploitable) **improves
    monotonically with training: u100 4.61 → u25 3.89 → u155 3.33**, and the
    gains appear on the **EVAL_ONLY held-out brackets**, not just trained
    opponents — the generalization check passes:
    | bracket | u100 | →u155 | held-out content |
    |---|---|---|---|
    | all_farmer | 4.17 | **2.17** | curry_fortress/self_tactical/anti_aggro (NO self_policy → not self-similarity) |
    | real_field | 2.00 | **1.06** (0.94 win) | peroxide + curry |
    | adversarial | 4.61 | **3.33** | curry_fortress |
    | semis_mixed | 1.56 | **1.33** (+60 margin) | mixed |
    Neither is "promotable" under the strict conjunction (both still margin<0 in
    all_farmer/adversarial — but so is u100, which is net-negative in 4/5
    brackets), yet semis2b-u75 is unambiguously the best of the three.
    **`semis2b-u75` is the leading CANDIDATE** for the eventual cloud
    cross-check; `native-u100` (0.661) stays the cloud-verified deploy floor
    until a finalist earns its submit.
    - **Why the entropy-collapse "stop" prior was WRONG here:** entropy fell
      0.20→0.03-0.06, which historically = mirror-overfit + cloud regression.
      But that pattern was on the LOCAL full-rl (mirror) curriculum. On the
      FOREIGN curriculum measured against a FOREIGN held-out gate, the collapse
      is the policy *specializing against real-competitor-like opponents* and it
      TRANSFERS (held-out brackets improve). Lesson: judge Stage B by the melee +
      EVAL_ONLY holdout, NOT by entropy or the historical inverted-U intuition.
    - **RESUMED again (tag `confpol-semis2c`, warm-start `confpol-semis2b-latest`
      ≈ real u173, CPU).** semis2c-uN ≈ real u173+N. Keep gating new rungs; stop
      only when worst-bracket placement stops improving OR held-out gains stall.
  - **🟢🟢 CLOUD A/B 2 Jun 16:33 → semis2b-u75 = 0.671 vs native-u100 = 0.661
    (+0.010, 0/30 errors, speed 0.844/0.849).** native-u100 RE-submitted to
    **0.661 exactly** same session → eval opponents unchanged + still
    deterministic ⇒ +0.010 is a REAL resolvable delta (the old "sub-0.05
    unresolvable" caveat was the noisy eval; retired). **The melee gate's verdict
    (semis2b-u75 > u100) TRANSFERRED to cloud** — 2nd melee→cloud agreement
    (u100>u860, now semis2b-u75>u100) and the FIRST Stage-B foreign-curriculum
    candidate to validate on the independent signal. Magnitude modest but the
    method works end-to-end → keep training + gating; later semis2c rungs may go
    higher. **`confpol-semis2b-u75` (0.671) is the new deploy pick; u100 (0.661)
    the fallback floor.**
  - **🛑 STAGE B CONCLUDED 2 Jun (`melee-semis2c.json`) → PEAK FOUND, STOPPED.**
    Gated the later semis2c rungs vs incumbent semis2b-u75. Worst-bracket
    placement gets monotonically WORSE past the peak: **semis2b-u75 (≈u155) 3.39
    → semis2c-u100 (≈u273) 3.67 → semis2c-u175 (≈u348) 5.78** — and the decline
    is on the HELD-OUT brackets (adversarial 3.39→5.78 / −392 margin; real_field
    1.17→2.89; semis_mixed 1.72→2.94). Mechanism: the policy over-specializes
    into `all_aggressive` (2.89→1.00→1.00, wins everything there) and loses
    balanced robustness everywhere else — the classic inverted-U, now on held-out
    opponents. **The melee peak (semis2b-u75, worst-bracket ~3.3–3.4) coincides
    with the cloud peak (0.671)** — both independent signals agree. **TRAINING
    STOPPED; `confpol-semis2b-u75` = 0.671 is the final Stage-B deploy pick.** Not
    worth submitting untested intermediates (real u198–248): they'd each need a
    cloud submit to beat a validated 0.671, and the melee trend says they won't.
    To push past 0.671 toward curryfarmer's 0.715 needs a DIFFERENT lever (new
    warm-start / foreign mix / the peroxide base-inference trick), not more of
    this curriculum — it has peaked.
  **NOTE the overfit risk this raises:** u100 is already the cloud peak and the
  whole line shows an inverted-U (more training past the peak LOSES cloud value).
  Fine-tuning *from* the peak means the FIRST few rungs (u25/u50) are the most
  likely winners; later rungs probably regress. Gate aggressively and early.
  **WHAT TO WATCH (in priority order):**
  1. **The EVAL_ONLY holdout is the kill-switch.** Gate ladder rungs with
     `melee_eval.py --confpol-ckpt semis2-u<N>=...checkpoints/confpol-semis2-u<N>.pt`
     and compare placement/margin on TRAIN_OK-heavy brackets vs the EVAL_ONLY
     opponents (`curry_fortress`/`self_tactical`/`aggressive_proxy`/`anti_aggro_exploiter`,
     concentrated in `all_farmer`/`adversarial`). If TRAIN-side improves but
     held-out does NOT → proxy-overfit, STOP (this is the failure that killed
     every prior line; the holdout is the early-warning we never had).
  2. **In-training entropy** (`confpol-semis.log`): u860 sits ~0.06–0.15.
     Gentle lr 1e-4 should keep it from collapsing to ~0.02 fast; a sudden
     entropy crash = over-confident overfit (the inverted-U) — favor EARLY rungs.
  3. **Do NOT promote on the in-training gated eval** — it's anti-correlated with
     cloud-consultant value past ~u860. Selection = melee gate + a cloud submit.
  4. **The two holes Stage B must close** (from the re-rank weakness map):
     `all_farmer` (native-u100 margin −291.8) and `adversarial`. Improvement
     there on the EVAL_ONLY opponents is the win condition; anywhere else is
     likely proxy memorization.
  - **Promotion bar (REVISED 2 Jun — Semis objective is competitive placement,
    not cloud reward):** the **melee gate is now the PRIMARY selector** — a rung
    is promotable if it beats `confpol-native-u100` on worst-bracket
    `mean_placement` + margin AND that gain shows up on the **EVAL_ONLY holdout**
    (not just trained opponents). **Cloud is demoted to a cheap final
    cross-check on the 1–2 finalists** (catch local self-deception; we do NOT
    submit every rung). Rationale: Semis is a 6-team placement race vs other
    teams' models, so the melee metric models the real goal; cloud measures
    absolute reward (possibly still the stale qualifier eval). Caveat that keeps
    cloud in the loop: the melee runs on OUR proxies and has false-positived
    hard (self_policy melee-1st, cloud 0.507) — cloud is the only INDEPENDENT
    signal, so a finalist must not crater on it. `confpol-native-u100` stays the
    deploy floor regardless. Cloud rebuild+submit is Workbench-only (not this Mac).

### The novice-determinism lever (the fixed seed-42 map) — LIVE, under-exploited

**The Semis Novice map is identical every game, every seed, and our detector
fires on it.** Proven 1 Jun via [../training/ae/probe_fixed_map.py](../training/ae/probe_fixed_map.py)
against the real `til_environment`: `is_fixed_novice_map=True` on all 6 agents
across seeds 7/999/31337; `base_location` (`array([13,9])` → native `[13,9]` →
`(13,9)`) matches our hardcoded [src/novice_map_data.py](src/novice_map_data.py)
exactly. Our `BASE_LOCATIONS` / `STARTING_LOCATIONS` / item table are
**byte-identical** to the competitor's `HUNTER_*` tables.

- **DELETED stale claim:** earlier NOTES said "the fixed-Novice map detector
  doesn't fire on the cloud eval." That was a **misdiagnosis** — it was inferred
  from a hybrid-PPO experiment (where the policy was in control and heuristic
  fast-paths were subordinate), never tested directly. The detector firing is a
  pure function of the observation format, which is defined by the `til_environment`
  package the cloud runs — so fires-locally ⇒ fires-on-cloud. Confirmed. (Cloud
  `til submit` gives no logs; the local env probe is the decisive test. Optional
  belt-and-suspenders: add a `[FIXEDMAP] fired` print and grep the `til test`
  log on the Workbench — `til test` runs the official evaluator locally.)
- **What we do with it today = SHALLOW:** preload walls/destructibles/bases/items
  at step 0, switch BFS→Dijkstra for exact pathing, bump `ENEMY_BASE_VALUE` to
  130. That's it. ([ae_manager.py:329](src/ae_manager.py))
- **What the competitor does = DEEP** (and scores 0.715): exact pathing **+ a
  divergence-gated opening book + an all-pairs distance LUT + an opponent-position
  LUT + a "defense wins / immortality" strategy**. Same trigger, far more
  leverage. This depth gap is the most plausible single explanation for 0.715 vs
  our 0.634. See the improvement plan in **"Using curryfarmer's work"** below.

### Competitor intel: curryfarmer / "royal-recruits" (public repo, 0.715/0.807)

Public GitHub repo `curryfarmer/til-26-ae` (team royal-recruits), cloud
**0.715 reward / 0.807 speed**. Cloned + fully reviewed 1 Jun. Key facts:

- **Same meta-conclusion as us:** their RL failed local→cloud transfer at
  qualifiers → they ship a hand-coded heuristic and plan to use it as a BC
  teacher/opponent for fresh RL at semis. Convergent with our entire arc.
- **Their heuristic is a different machine: portfolio-A* + forward-sim plan
  scoring**, not greedy target-picking. Per tick: enumerate K=4–6 goals → A*
  each over `(x,y,facing,t,bombs,placed)` (facing/time/bomb-inventory aware,
  horizon 8–10) → **project the world forward and score each plan by estimated
  raw game reward** (`eval.score_plan`) → pick best. Time-layered danger map
  (`is_lethal(x,y,t)` per future tick). ~12 personas (aggro/greedy/fortress/…)
  + a runtime persona-FSM. **Rust (PyO3) A* kernel** for speed; full IS-MCTS
  built but shipped OFF (A* portfolio is the live path).
- **Their hunter scores:** `greedy_hunter` local bench qual **0.929** / semis
  **0.747** composite; they note til-server ≈ −0.20 reward vs til-test, landing
  cloud at ~0.71–0.75. The 0.715 the user found is one of these hunters.
- **Strategic insight we lack: "immortality → farming race → defense wins."**
  They reverse-engineered that kills only freeze 3 turns then respawn full-HP,
  so a match is a 200-tick points-farming race; they price stun downtime in
  `score_plan` (`FREEZE_TURNS × W_STUN_PER_TURN`) and built a `fortress` persona
  (tops their roster). Our scorer has no respawn/stun model.
- **Where we already match or beat them:** same RL verdict; both built search and
  ship it OFF; both maintain a belief/world model; both pin `PYTHONHASHSEED=0`
  for the same tie-break-determinism reason. **We're slightly FASTER** (~0.84 vs
  their 0.807) — speed is not our problem; accuracy is.

### Public-repo scrape (2 Jun) — full field of TIL-26 AE repos

Scraped GitHub (forks of `til-ai/til-26{,-ae,-finals}` + keyword/code search).
Of ~12 candidate repos, **4 had real AE work** beyond the 663-byte stock
template; the rest are the untouched scaffold. Findings + vendoring decisions:

| Repo | Track | Architecture | Weights | Strength | Decision |
|---|---|---|---|---|---|
| `curryfarmer` (royal-recruits) | — | portfolio-A* + forward-sim | (heuristic) | **0.715, semifinalist** | already vendored (curry_aggro/fortress) |
| `peroxide-dev/til-26` | — | orientation-aware A* + base-anchor inference | none needed | **0.443, NON-semifinalist (weak)** | **VENDORED `peroxide_astar` (EVAL_ONLY)** |
| `AndreLiu1225/til-26` | general (`novice:false`) | CNN+MLP masked PPO | 3 committed | team did poorly | **rejected** (user call) |
| `Pushkaltoocool/til-26` | — | **DQN** (value-based) | none | unknown | not runnable (no weights) |
| `Parachuters/til-26` | **Advanced** | masked **recurrent** PPO | none | off our Novice bracket | not runnable / off-bracket |

- **Why vendor a *weak* agent (peroxide):** it is the only **second genuinely
  foreign decision architecture** we can obtain (A* portfolio, distinct from
  curry's forward-sim and our greedy-Dijkstra). For the EVAL_ONLY holdout,
  *architecture diversity > strength*; and as a realistic mid/low-strength field
  member it fits our 15th seed (the real bracket has weaker teams too). It's
  numpy-only, self-contained, and the env's already-unpacked (7,5,25) viewcone
  matches its decoder (verified it decodes + acts, not blind). **Do NOT read
  beating peroxide as signal** — it's a field-filler, not a bar. New bracket
  `real_field` (curry + peroxide + self_policy + anti_aggro + self_heuristic)
  added to `MELEE_BRACKETS`; the original 4 brackets are unchanged so the prior
  weakness map stays comparable. Vendored gitignored at
  `training/ae/foreign/peroxide/` (do NOT commit competitor code).
- **Why NOT restore an old model of ours as an opponent:** it would be a 4th
  MIRROR (we already have self_policy/self_tactical/self_heuristic), trained
  against our own opponents so it shares our blind spots — the opposite of what
  the foreign pool is for. The spec warns against over-weighting one source.
- **Intel worth keeping:** (1) **two teams went DQN self-play** (peroxide,
  Pushkaltoocool) — a value-based line we never tried (no evidence it scored
  well, though). (2) **peroxide's base-inference trick** — guess the other 5
  spawns by rotating own base around grid-center by π/3 on the symmetric layout;
  borrowable into our own planner. (3) Most public repos are stock template;
  the strong semifinalists' code is private, so curry remains our only *strong*
  real-competitor proxy.

### Reward calibration: why we DON'T use nominal game rewards as candidate values

Recurring question — settled here so we don't re-litigate. Real game rewards:
mission +5, resource +2, recon +1, destroy_base +50, kill +30/15, own_base −50,
damage +1/HP. Our candidate values: mission 80, resource 40, base 100/130.

- **Our scorer is a greedy *priority* function, not an EV estimate.** `score =
  base_value − DIST_PENALTY·dist − …` ranks one target cell to walk toward; the
  value lives in the same arbitrary units as `DIST_PENALTY=1.15` and the
  visit/threat penalties. Only *ratios* matter; rescaling to "game units" changes
  nothing unless you change ratios.
- **Nominal reward ≠ realized EV.** Game values base:mission at **10:1**; we use
  **~1.25:1** — i.e. we deliberately UNDER-weight bases ~8× vs nominal. That's
  correct: a base is nominally +50 but hard/contested/slow (realized EV ≪ 50),
  a mission is +5 but near-certain. In a 200-tick farming race, steady certain
  mission income beats risky base attempts. Our C+bomb7 ratios (survivors of the
  +1.13σ `multi_seed_eval` sweep) already encode this realized-EV correction.
- **Real-reward calibration is correct-by-construction ONLY when paired with
  forward-sim plan projection** (the competitor's `score_plan` discounts a base
  to ~0 when the bomb won't land). Bolting nominal rewards onto our greedy
  one-step scorer with no projection would over-chase bases it can't finish →
  almost certainly regress (same failure mode as our dead aggressive/scripted
  lines). Calibration is a *consequence* of adopting plan-projection scoring, not
  a standalone win.
- **The RL side is already real-reward-aligned** (PPO trains on the env's actual
  reward + shaping). The abstract constants are a heuristic-only object.
- Cheap to falsify: `AE_ITEM_MISSION_VALUE=5 AE_ITEM_RESOURCE_VALUE=2
  AE_ENEMY_BASE_VALUE=50` + `multi_seed_eval.py`. Strong prior: regresses unless
  `DIST_PENALTY` is also retuned.

### Forward-sim plan re-score — BUILT, CLOUD-CONFIRMED DEAD (0.590 vs 0.671), default-OFF (3 Jun)

The one structural lever curry (0.715) has and we don't: re-score plans by
*projected realized reward* instead of static `value − dist`. Adopted the
**principle, not curry's code** (no Rust kernel, no persona-FSM, no respawn/stun
model). **Contained:** reuses the manager's existing forward-sim primitives
(`_lookahead_step`/`_blast_cells`/`_lookahead_escape`, real env reward units) to
re-rank **only the top-K static target candidates** — NOT the dead `mcts-light`
full-width beam (that timed out / −0.254 speed). **Opponent-light:** projects our
own plan on the known map only — no opponent rollout, no safety veto (the line
that keeps it out of the dypm/pessimistic-search grave).

- **Code** ([src/ae_manager.py](src/ae_manager.py)): `_project_plan_reward`
  (items en route + base value *only if the bomb from `path[-2]` reaches the base
  and an escape exists* + time tax) and `_rescore_top_k` (margin-gated override,
  ties broken on `pos` so it's `PYTHONHASHSEED`-independent), spliced into
  `_choose_target` behind **`AE_PLAN_RESCORE` (default 0)**. The `scored` list is
  only built when the flag is on ⇒ **flag-off path is byte-identical**.
  `last_decision_confidence` stays in static units ⇒ the deployed
  `confidence_policy_hybrid` gate is unperturbed; only the *executed* target
  changes. Knobs: `_K`(4) `_MARGIN`(1.0) `_TIME_TAX`(0.35) `_BASE_W`(1.0)
  `_DEMOTE_ONLY`(1). 9 TDD tests in
  [test/test_ae_plan_rescore.py](../test/test_ae_plan_rescore.py).
- **⚠ KEY FINDING — promotion regresses, demotion is the only viable half.**
  First smoke (n=2 hash × 6 rounds, `--our heuristic`, furnished) with the naive
  *symmetric* re-score (credit landable bases at full 55): weighted **0.1935 vs
  off 0.273 (−0.080)**; `top_seed_proxy` 0.472→0.190, `pressure2` went *negative*.
  This is exactly the calibration-section prediction: crediting a landable base at
  full realized value makes the planner abandon steady item income to chase bases.
  Fix = **asymmetric `AE_PLAN_RESCORE_DEMOTE_ONLY` (now default ON):** projection
  may DEMOTE a static-winner phantom base toward a realizable alternative, but may
  never PROMOTE a base over a non-base static winner. That is curry's actual
  documented benefit ("discount a base to ~0 when the bomb won't land"), without
  the over-aggression. Demote-only smoke: weighted **0.2795 vs off 0.273
  (+0.0065)** — recovers the regression but is **flat at the noise floor**, and it
  REDISTRIBUTES (wins pressure2 +0.072 / strong_realistic +0.055 / cloudsuite
  +0.049; loses top_seed_proxy −0.137 / defense_trap −0.072 / base_rush −0.062) —
  the classic "helps low-pressure, hurts high-pressure, cancels" AE pattern.
- **Verdict (3 Jun): CLOUD-CONFIRMED DEAD → stays OFF.** Cloud A/B settled it
  conclusively (deterministic eval, 1 submit = truth): tag **`planrescore`
  (AE_PLAN_RESCORE=1, demote-only, bc.pt=confpol-semis2b-u75) = 0.590 / speed
  0.833, 0/30 errors** vs the same image flag-off **0.671** → **−0.081 cloud
  regression.** So the local reject gate was directionally RIGHT (don't promote)
  but, if anything, too generous: local demote-only read **flat (+0.0096)** while
  cloud is **−0.081** — yet another local-flat/cloud-negative gap. Speed 0.833
  confirms the per-tick top-K projection never threatened the latency budget; the
  loss is pure accuracy. **Possibility that "the local gate mis-predicted" is now
  closed — it did not; the lever genuinely hurts.**
- **Local gate trail (kept for the mechanism):** proper n=3 hash × 12-round gate
  (`plan-rescore-gate-{off,demote}.json`): demote-only weighted **0.2757 vs off
  0.2661 (+0.0096, flat)** but **worst-per-run −0.012** and **top_seed_proxy
  0.335 vs 0.472 (−0.137)** — the most cloud-predictive suite, which correctly
  foreshadowed the cloud loss. The SYMMETRIC variant (promote landable bases) was
  −0.080 weighted locally (over-aggression, as the calibration section predicted).
- **Why it's dead and what it means:** same "helps low-pressure, hurts
  high-pressure, cancels (and then loses on cloud)" wall as every prior
  action-policy lever (opening book, dypm-veto, aggression sweeps). **The
  contained top-K-projection mechanism works as engineered and stays in-tree
  (default-OFF, Dockerfile `AE_PLAN_RESCORE=0`) as reusable infra, but self-plan
  projection alone does NOT close the gap to curry's 0.715** — curry's edge also
  prices respawn/stun downtime in `score_plan` (a 200-tick farming-race model our
  projection omits) and runs a persona-FSM. **Do NOT re-run this or base-value
  calibration on the greedy scorer.** A genuinely different mechanism is needed.
  Reproduce local: `.venv/bin/python training/ae/multi_seed_eval.py --rounds 12
  --our heuristic --preset furnished --hash-seeds 0 1 2 --sim-seeds 42
  --extra-env AE_PLAN_RESCORE=1 --summary-out <out>.json`.

### Contention-aware item valuation — CLOUD-NEUTRAL, SHIPPED ON for the Semis melee (3 Jun)

The **first AE lever that adds the opponent dimension.** Every dead action-policy
lever (opening book, dypm-veto, plan-rescore) only re-weighted *our own* plan with
no opponent model; our heuristic `_choose_target` scorer is **opponent-blind**
(`value − DIST_PENALTY·dist − …`). But Semis = the fixed seed-42 Novice map, so
`novice_map_data.STARTING_LOCATIONS` hands us all 6 spawns for free, and
`self.enemy_agents` already tracks live viewcone sightings. This lever uses that to
discount items an opponent reaches first — aimed squarely at the **`all_farmer`
hole** (the −388/−291 farming-race margins). Design/plan:
`docs/superpowers/specs/2026-06-03-ae-contention-aware-valuation-design.md` +
`docs/superpowers/plans/2026-06-03-ae-contention-aware-valuation.md`.

- **Code** ([src/ae_manager.py](src/ae_manager.py), behind **`AE_CONTENTION`
  (default 0)**): `_believed_opponents` (fixed-map spawns in the opening +
  fresh sightings), `_opponent_distance_map` (one multi-source BFS, reuses
  `_neighbors`), `_apply_contention` (demote-only, **item-vs-item**: scales each
  item by `mult=max(PFLOOR, sigmoid((d_opp−d_us)/SCALE))` via the delta
  `adj=score−base_value·(1−mult)`; never promotes a non-item). Spliced into
  `_choose_target` **after** the static loop, parallel to the plan-rescore block —
  **flag-off path byte-identical**, `last_decision_confidence` stays in static
  units (confpol gate unperturbed). `scored` tuple + `_rescore_top_k` untouched
  (base_value recovered from `kind`). Knobs: `AE_CONTENTION_SCALE`(2.5)
  `_PFLOOR`(0.15, =1.0 disables) `_TOPEN`(40) `_TFRESH`(3). 18 TDD tests in
  [test/test_ae_contention.py](../test/test_ae_contention.py); plan-rescore suite
  still 9/9.
- **Why it might differ from the graveyard:** new *information* (opponent geometry,
  ground-truth not learned), demote-only + soft + floored, and it **no-ops whenever
  we have no credible opponent position** (post-opening, nobody in view) — that
  conservatism is the transfer-safety. Still an action-policy lever though; could
  still hit the "helps low-pressure / hurts high-pressure / cancels" wall.
- **Local gate RUN 3 Jun → NEUTRAL/inconclusive (not a reject).**
  - **Step 1 (multi_seed reject, n=3 hash):** weighted **−0.009** (flat); all 8
    suites within ±0.07; `defense_trap` exactly 0.000 (item-vs-item doesn't touch
    defense). **The cloud-predictive `top_seed_proxy` = +0.037** — the same suite
    that was −0.137 for plan-rescore and correctly foreshadowed its cloud death;
    here it points the *opposite* way. Did NOT reject.
  - **Step 2 (melee gate) — first run was CONFOUNDED.** Every opponent
    subclasses/contains our `AEManager`, so with `AE_CONTENTION=1` in the shared
    env the self-mirror opponents (`self_heuristic`, `self_tactical`) and the
    purpose-built bots' `super()._choose_target` fallback ALSO got contention,
    buffing them. **Fixed the harness** (commit): `AE_CONTENTION=0` baked into
    `self_heuristic`/`self_tactical` `_EnvOverride` (durable across their per-round
    rebuilds) + a `contention_enabled=False` guard on direct-subclass opponents in
    `simulate.run_simulation`. our_agent builds first and resets in-place, so it
    keeps contention; probe-verified (ours True, every opponent False). No-op when
    the flag is off (OFF baseline unchanged). **General lesson: an experimental
    AE_* flag must be scoped to our agent in the melee or it buffs the opponent
    mirrors too.**
  - **Clean melee (de-contaminated, n=3) = NEUTRAL.** `all_farmer` placement
    nominally improved (2.39→2.17 — the lever's target) but its **score is flat**
    (0.278→0.279). `semis_mixed` looked like a −0.204 score crater at n=3, **but
    that was chaotic-cascade VARIANCE** — re-run at **n=5 it is −0.006** (OFF 0.407
    vs ON 0.401). Contention only fires mid-game (opening targets aren't contested
    enough to flip), so a single mid-game target swap cascades unpredictably →
    per-seed `semis_mixed` ON ranged ~0.20–0.70 while OFF stayed ~0.40. **The
    melee at n=3 simply cannot resolve a lever this small.**
- Local verdict was NEUTRAL (no clear win, no reject); only the `top_seed_proxy`
  canary (+0.037) favored it, so it earned the cloud A/B.
- **🛑 CLOUD A/B DONE 3 Jun → INERT.** Tag `contention-aware` (`AE_CONTENTION=1`,
  bc.pt=`confpol-semis2b-u75`) = **0.671 / speed 0.840, 0/30 errors** =
  **byte-identical to the flag-off 0.671.** The image really had the flag baked
  (`docker run --rm melanie-minions-ae:contention-aware env | grep AE_CONTENTION`
  → `AE_CONTENTION=1`), so this is a valid test, NOT a build miss.
- **Why inert — diagnosed locally (`PYTHONHASHSEED=0`, confpol+semis2b-u75,
  AE_CONTENTION=1, 3 novice games):** `contention_enabled=True` ✓,
  `is_fixed_novice_map` fires ✓, `_apply_contention` actually RAN **21×**, but it
  **CHANGED the executed target only 4× across 3 whole games (~1.3/game).** The gate
  (best target is an item AND believed-opponents non-empty) rarely co-occurs:
  spawn-seeding only covers the opening (`step ≤ TOPEN=40`), live sightings are
  sparse, and when it does run the contested item rarely loses to an alternative.
  On a single deterministic cloud game that's ~0–2 inconsequential target swaps →
  0.671 unchanged. **The lever's logic is sound; it's starved of opponent-position
  "fuel."**
- **Durable lesson (matters for the whole "deeper fixed-map opponent modeling"
  family):** the bottleneck is *opponent-position availability mid-game*, not the
  scoring logic. We only reliably know opponent SPAWNS (opening); we cannot get
  trustworthy mid/late-game opponent positions vs unknown Semis teams without the
  exact overfit the project fights (curry hardcodes an opponent-position LUT, but
  only because their qualifier opponents were the deterministic BenBots). So
  contention — and any opponent-aware action lever — is inert-or-overfit-risky
  until we have a credible source of live opponent positions. **Don't re-pursue
  this family expecting it to fire; a different lever is needed.**
- **STATUS: SHIPPED ON (user call, 3 Jun) — Dockerfile default flipped to
  `AE_CONTENTION=1`.** Rationale: the cloud A/B is NEUTRAL (0.671 = flag-off, 0/30
  err, speed 0.840 ≈ baseline → downside cloud-proven ~0), but the cloud measures
  absolute reward at a fixed spawn, **not melee placement vs other teams — the real
  Semis objective and the only thing this lever targets** (de-contaminated melee:
  `all_farmer` 2.39→2.17, small/noisy). So we ship it as a near-zero-downside bet on
  the melee. **Caveat (honest):** the melee upside is from noisy local proxies, not
  proven; if anything looks off in-person, revert is ONE line (`AE_CONTENTION=0`).
  18 TDD tests pass; flag-off path byte-identical (instant rollback). The deployed
  image is the same one validated at 0.671 (tag `contention-aware`). `confpol-semis2b-u75`
  is still the policy/checkpoint (0.671). The **melee de-contamination fix**
  (`foreign_opponents`/`simulate`) is a permanent keeper — experimental `AE_*` flags
  no longer leak to the opponent mirrors.

### The one problem that dominates AE: the local→cloud transfer gap

Every learned-policy line (BC, PPO, self-play, belief-map, MCTS, macro/tactical
hybrid) has died the **same death**: local reward improves while hidden cloud
eval does not. The gap between optimistic local scores and cloud is a stable
**~0.19–0.30**, and it does not close with more training, bigger nets, frame
stacking, or richer state. Root cause: our local opponents ≠ the cloud NPC
distribution, so the policy overfits local opponent behavior. **The rule-based
heuristic transfers best precisely because it does not learn local-opponent
quirks.** Any new learning attempt must have a credible answer to "why won't
this overfit the local opponents like the last ten attempts did?"

### Measurement reality — ⚠ CHANGED 1 Jun: DETERMINISTIC; ⚠⚠ CHANGED AGAIN 4 Jun: OPPONENTS SWAPPED (all numbers below re-based — see "Current state (4 June 2026)")

- **As of 1 Jun the organisers SEEDED the AE eval and removed non-determinism.**
  Proven empirically: confpol-u860 farmed **0.626 ×3 byte-identical**, opening-v2
  **0.626 ×5 byte-identical** — zero variance. **Same image → same score.**
- **Consequence: 1 submission = the true score. Variance farming is OBSOLETE.**
  The whole `variance_farm.py` / σ=0.053 / "n≥18 to resolve +0.05" apparatus
  below was built for the OLD noisy eval and **no longer applies** — do NOT
  re-submit the same image n times; do NOT discount a single score as noise.
  One submit per distinct image now cleanly ranks everything.
- **The eval appears to evaluate us at a FIXED spawn ∈ {13,9 / 3,12 / 12,3}**
  (inferred 1 Jun: opening-v1, which enabled those spawns, scored 0.584 ≠ confpol;
  opening-v2, which disables them, scored 0.626 = confpol *exactly* → the opening
  never fires ⇒ our eval spawn is one v2 disables, i.e. one confpol opens well).
  Not 100% confirmed; semis *bracket* matches may assign other spawns.
- **Org also said they "may change the deployed opponent models"** (away from the
  BenBots, toward other teams' models). If/when that happens the whole ranking
  can shift — re-rank the field again (cheap now, 1 submit each).
- ⤵ **HISTORICAL (pre-1-Jun noisy eval, kept for context):** cloud σ ≈ 0.053 on a
  byte-identical image (±0.10 per single submit); +0.05 needed ~18 submits/arm to
  resolve; `variance_farm.py` + the `cloud_samples.json` ledger existed to farm
  n≥5 and compare. All the farmed means in this file (0.634, 0.599, 0.584 …) are
  old-eval numbers; the new deterministic eval supersedes them (confpol = 0.626).

### How to evaluate properly (the gate)

1. **Local gate = `training/ae/multi_seed_eval.py`** (still valid as a *local*
   pre-filter), n≥5 hash seeds × sim seeds × 6 rounds vs the `heuristic-C+bomb7`
   baseline (`w3_2_C_bomb7_n5.json`, 0.2842 ± 0.0074). `PYTHONHASHSEED` auto-pinned
   to 0 everywhere. BUT: local lift has repeatedly NOT transferred to cloud — use
   local only to reject obviously-bad candidates, not to predict cloud rank.
2. **Cloud is now the cheap ground truth: 1 `til submit` = the score.** Re-rank by
   submitting one tag per distinct image (see `training/ae/resubmit_rerank.sh`).
   `variance_farm.py` is retained only as a tidy ledger/ranker (`add`/`summary`);
   its variance/power math is moot under the deterministic eval.
   - **Resubmit old models WITHOUT rebuilding:** `til build` images persist in the
     Workbench Docker daemon as `melanie-minions-ae:<tag>`; `til submit ae <tag>`
     re-uploads the existing image. `docker images | grep ae` lists them. Lets us
     re-score every historical build on the new eval for free.

### What works / keep doing

- **The rule-based heuristic is the strongest deployable agent we have.** Ship
  it. The `confidence_hybrid` wrapper safely falls back to it and adds no
  inference cost.
- The **C+bomb7** profile is the best-ranked heuristic config at the calibrated
  local gate (+1.13σ over baseline; wins defense_trap/top_seed_proxy/
  bracket_proxy — every semifinals-relevant suite; only loses pressure2).
- **Measurement loop (post-1-Jun):** local multi-seed eval to *reject* bad
  candidates, then **a single `til submit` per config = the true cloud score**
  (eval is deterministic now). Local lift has repeatedly failed to transfer, so
  trust cloud for the rank; local only filters the obviously-bad.

### Dead ends — DO NOT REDO (each cost a session; all confirmed negative)

| Line | What was tried | Result / why it failed |
|---|---|---|
| **Plain PPO over raw actions** (`ppo-v1/v2`, selfplay, full-RL) | BC warm-start + PPO, frame-stacking, self-play snapshot league | Cloud ceiling ~0.43–0.51; transfer gap never closed. `ppo-full-rl-v1` farmed mean ~0.577. |
| **Confidence-gated PPO** (`confidence_hybrid`, v2/v3/v4) | PPO consulted only on low-confidence heuristic ticks; multi-seed save gate | 3 independent retrains all fail local multi-seed (best Δ -0.008, never positive). v4 cloud-farmed **0.549** mean — below heuristic. **PPO-on-heuristic line is empirically closed.** |
| **Macro/tactical-hybrid PPO** (12-way macro selector) | Learn *when* to pick macros, planner owns movement | Harm-aware data on 800 games: **zero positive-EV transitions across all 90 prior/option pairs.** Heuristic beats random macro exploration everywhere. |
| **Tactical BC** (400/800-game outcome-weighted) | Behavior-clone good macros | 400-game "win" was legacy-gate variance; 800-game overfit and collapsed base/top/bracket suites. |
| **Belief-map / memory BC** (`bc-belief-hybrid`) | 704k-param CNN belief input | Fit local *better* (val_acc 0.897) but **widened** cloud gap by +0.044. Rich state against random opponents = more ways to overfit. |
| **MCTS as primary planner** (`mcts-light`) | Depth/width search per tick | Either times out (no latency cap) or, when capped, regresses accuracy −0.068 and speed −0.254. Workshop teaches no MCTS; top teams aren't doing it. |
| **Forward-sim plan re-score** (`AE_PLAN_RESCORE`, top-K projection, 3 Jun) | Re-rank top-K target candidates by projected realized reward (curry's `score_plan` principle, contained — not the dead full-width beam) | **CLOUD-CONFIRMED DEAD: tag `planrescore` = 0.590 vs champion 0.671 (−0.081, deterministic, 0/30 err, speed 0.833).** Demote-only read flat locally (+0.0096 n=3×12) but cloud −0.081; symmetric −0.080 local. top_seed_proxy −0.137 correctly foreshadowed it. Self-plan projection alone doesn't close the gap to curry's 0.715 (curry also prices respawn/stun). Code in-tree default-OFF. Do NOT re-run. See *Forward-sim plan re-score* above. |
| **Contention-aware item valuation** (`AE_CONTENTION`, opponent-aware scorer, 3 Jun) | Discount item targets an opponent reaches first, using free fixed-map spawns + live viewcone sightings (demote-only, item-vs-item) | **CLOUD-NEUTRAL, not negative: tag `contention-aware` = 0.671 = flag-off exactly (0/30 err, speed 0.840; image confirmed AE_CONTENTION=1).** Flag fires but changes the target only ~1.3×/game (`_apply_contention` ran 21× / changed 4× over 3 games). **SHIPPED ON anyway** (Dockerfile `AE_CONTENTION=1`, user call) as a near-zero-downside bet on melee PLACEMENT — which the cloud can't measure (de-contam melee `all_farmer` 2.39→2.17, small/noisy); revert = one line. **LESSON (why it's in this table):** an opponent-aware ACTION lever can't move the cloud/reward number — bottlenecked by no trustworthy mid-game opponent positions vs unknown teams (only spawns known); don't build another expecting cloud gains. 18 TDD tests; melee de-contamination fix kept. See *Contention-aware item valuation* above. |
| **Scripted M5 port** (`scripted_hybrid`, full ScriptedBaseAttackPolicy) | Port the 0.731 team's full decision tree | **−3.35σ LOSS** at the local gate. M5's 0.731 is codebase-specific, not primitive-additive. |
| **Three M5 primitives** (spawn-first-target table, enemy-bomb-only escape, orientation-aware A*) | Bolt-on env flags | All noise-to-catastrophic (orientation-aware A* −0.142; our DIST_PENALTY is tuned for grid, not orientation distance). All default-OFF. |
| **Memorized-route / rusher / camping cheese** (RIGID) | Precompute a greedy route per fixed-Novice spawn, replay blindly | RIGID replay loses to per-tick re-evaluation. **⚠ But see below: a DIVERGENCE-GATED opening book is NOT this** — the competitor ships one and scores 0.715. Re-open with abort-on-divergence (the rigid version is what failed, not the concept). |
| **LLM-as-player** (`llm_manager.py`) | Sonnet/Gemini play the game; mine rationales | LLM play ≤ heuristic even with belief-memory; a *worse* teacher than the heuristic itself. |
| **Rationale-mined heuristic leads** (bomb-gate-base, recon-discount) | Tweaks suggested by LLM rationale mining | bomb-gate-base collapses defense_trap −0.213; recon-discount is a no-op. Only base-tether survived (and it's neutral, see above). |
| **Heuristic-A (160/0.9) aggression**, `heuristic-a-bomb7` combos | Untested heuristic-parameter combos | heuristic-A ranks 7th (−0.28σ) at the calibrated gate; its 0.613 cloud was tail variance. Untested combos regress. |

**General anti-patterns (from the table above):** bigger BC nets with rich
state, larger same-distribution BC datasets, frame stacking, frontier/
exploration tuning for its own sake, six-game local means as a selection metric,
and committing large checkpoints to git. All burned.

### Active line: Pandemonium-v1 (from-scratch CNN-PPO)

The **one untried recipe**: the 0.731 team's CNN-over-viewcone + MLP PPO trained
**from scratch at ~15M steps** (~10M Novice mix + ~5M self-play). Note this is
NOT a new architecture — `PolicyNetwork` (model.py) already matches the spec and
`train_ppo.py` has the loop. The only new variables are **scale** (~15M vs our
historical ~1M steps), from-scratch init, and their hyperparams.

- Launcher: [../training/ae/run_pandemonium_v1.py](../training/ae/run_pandemonium_v1.py)
  (two-phase: full-rl mix from scratch → self-play fine-tune; γ0.99/λ0.95/
  ent0.01→0.001/lr-decay; orthogonal init; saves `-latest` every update).
- **CPU-bound, ~3.5–4 days wall-clock** (cost is 3 opponent-AEManager Dijkstra
  plans per tick; the NN is trivial, MPS barely helps). Phase1 ~60h, phase2 ~30h.
- Launch (run manually so it owns the machine):
  ```bash
  cd /Users/ethankok/projects/TIL
  PYTHONHASHSEED=0 PYTORCH_ENABLE_MPS_FALLBACK=1 \
    nohup caffeinate -is .venv/bin/python -u training/ae/run_pandemonium_v1.py \
    --tag pandemonium-v1 --games-per-update 12 --eval-every 20 \
    > training/ae/checkpoints/pandemonium-v1.run.log 2>&1 &
  tail -f training/ae/checkpoints/pandemonium-v1.log   # watch training
  # resume after phase 1 crash:  add --skip-phase1
  # kill:  pkill -f run_pandemonium_v1.py
  ```
  Gotchas: orthogonal-init's `linalg_qr` needs `PYTORCH_ENABLE_MPS_FALLBACK=1`;
  from-scratch via a non-existent `--bc-checkpoint`; `caffeinate -is` keeps the
  Mac awake but a **closed lid still clamshell-sleeps**.
- **Early-abort signal:** eval prints every 20 updates (~45 min). From scratch
  starts at eval ≈ −0.35. If it isn't trending up by **update ~100–200**, it's
  the same transfer-gap failure — kill it.
- **Promotion gate (critical):** do NOT trust the single-seed in-training gate
  (that's what burned the elo line: predicted +0.028 → cloud −0.21). Real gate =
  n≥3 `multi_seed_eval.py` under the hybrid wrapper vs C+bomb7, **then** cloud
  variance-farm via `variance_farm.py`. Deployment fallback stays the existing
  heuristic-veto (already stronger than Pandemonium's BFS), so no separate BFS
  manager was built.
- **Concurrent early-read submissions (30 May, u860 snapshot).** While training
  runs, the current-best checkpoint (`pandemonium-v1-best-u860.pt`, local
  ppo_eval 0.7415; in bucket `handoff/`) is submitted in three deploy modes to
  get an early transfer read (mid-training, no self-play yet — read the *sign*,
  not the exact value; cloud n=1 = ±0.10):
  - `AE_MODE=policy` — pure PPO in full control.
  - `AE_MODE=hybrid` — policy-first, heuristic safety veto.
  - `AE_MODE=confidence_policy_hybrid` — **new wrapper**
    ([src/confidence_policy_hybrid_manager.py](src/confidence_policy_hybrid_manager.py)):
    heuristic-first, raw 6-action policy consulted only on low-confidence ticks.
    Built because the existing `confidence_hybrid` is **macro-only** and would
    silently fall back to heuristic on a raw-action checkpoint (the 0.638 bug).
  All three verified to load the real policy via `_make_manager()` (no silent
  fallback). Workbench: `git pull` → copy the ckpt to `ae/models/bc.pt` → set
  `ENV AE_MODE` per build → `til build/test/submit`. **Canary:** the `til test`
  log must say `AE policy loaded … epoch=860`, NOT "falling back to heuristic".

  **RESULTS (30–31 May, 0/30 errors). policy/hybrid n=1; confpol VARIANCE-FARMED n=13:**
  | mode | cloud acc | speed | read |
  |---|---:|---:|---|
  | `policy` (pure PPO) | **0.507** (n=1) | 0.848 | transfer gap holds (local 0.74 → cloud 0.51) |
  | `hybrid` (policy-first + veto) | **0.508** (n=1) | 0.852 | ≈ policy; the safety veto rarely changes the outcome |
  | `confpol` (heuristic-first) | **mean 0.634** (n=13, σ0.046, 95%CI [0.609,0.659]) | 0.84 | **first farmed AE config above the incumbent bar** |

  **Read:** the raw policy is a *worse global controller* than the heuristic
  (0.51 < 0.59 — Pandemonium scale did NOT close the transfer gap) but a *useful
  local specialist*. `confpol` gates it to only the ~20% of ticks where the
  heuristic is unsure, so two ≤0.59 components combine because their errors are
  uncorrelated and the confidence router sends each tick to the stronger one.
  This validates the long-standing "planner-first arbitration, learned policy as
  consultant — NOT policy-as-primary" thesis with a policy finally good enough to pay.

  **The farm (n=13): the 0.667 first draw was the 73rd percentile; true mean is
  0.634.** Range 0.563–0.710, sample σ 0.046 (consistent with the 0.053 noise
  model). This is the **first AE config in project history with a *farmed* mean
  above the ~0.59–0.60 incumbent bar** — every prior "high" (0.613, 0.638) was a
  single tail draw that regressed on re-sampling; this one holds at n=13.
  - vs the established ~0.59–0.60 bar (one-sample): **+0.04, z≈2.7–3.4 — clears it.**
  - vs `heuristic-A` (n=3, 0.599) two-sample: Δ+0.035, p=0.30 **NOT yet significant
    — only because the incumbent is under-sampled (n=3)**, not because confpol is weak.
  **Next steps (in order):**
  1. **Pin the incumbent: re-farm the shipped heuristic (tether/C+bomb7) to n≥5**
     so the comparison is apples-to-apples (n=13 vs n=13), not vs n=3. This is the
     one clean experiment that converts "+0.035, p=0.30" into a real promotion call.
  2. **Re-farm `confpol` with the FINAL post-self-play checkpoint** — the farm used
     the *u860* snapshot; training is now at u1140 with eval 0.772 / cloudsuite 0.702
     (vs 0.74/0.65 at u860), so the final policy likely lifts confpol further.
  3. Sweep the gate (`AE_CONFPOL_MARGIN_EPSILON`/`_TOP_FLOOR`) — how often the policy
     is consulted is the contribution lever.
  - Drop `policy`/`hybrid` as deploy candidates (transfer-gapped at 0.51).

  **It's WHICH ticks, not HOW MANY (measured 31 May, local 12-game tally).**
  `confpol` consults the policy ~46% of ticks / changes the action ~25.5%;
  `hybrid` (fixed-map shortcut off) lets the policy serve ~39.5% / changes ~23.7%
  — **nearly the same policy-influence rate, yet confpol=0.634 vs hybrid=0.508.**
  The difference is routing: `confpol` gives the policy only the heuristic's
  *low-confidence* ticks (its competent ticks → helpful); `hybrid` lets it
  override even confident-and-correct heuristic ticks (→ harmful, since the
  policy is a bad global controller). Same policy, +0.13 from routing alone.
  Note: local `hybrid` heuristic fast-paths fire heavily (fixed-map 42% + escape
  60%), but on cloud hybrid scored 0.508 ≈ pure-policy 0.507. This was originally
  read as "those fast-paths (esp. the fixed-Novice map detector) don't fire on the
  cloud eval." **⚠ CORRECTED 1 Jun: that inference was WRONG.** The fixed-map
  detector provably DOES fire (see *The novice-determinism lever* above). What
  actually happened in `hybrid` mode is that the *policy* was the controller and
  the heuristic's fast-paths were subordinate/overridden — so their firing didn't
  show up in the score. The detector itself is healthy; do not cite this line as
  evidence the map exploit is dead.

  **⚠ CRITICAL (31 May): more local training made the policy a WORSE cloud
  consultant — the cloud-best checkpoint is EARLIER than the local-best.**
  Phase 1 finished at u1563 (best-local checkpoint = u1400, eval 0.790). Farmed
  `confpol` with it (n=10): **mean 0.600** — vs the u860 checkpoint's **0.634**
  (n=13). The u1400 policy was better on EVERY local metric (eval 0.742→0.790,
  cloudsuite 0.647→0.747) and far more deterministic (entropy 0.064→0.025), yet
  its cloud-consultant value DROPPED and its edge over the heuristic vanished
  (0.600 ≈ heuristic 0.599; Δ vs u860 = −0.034, p=0.12 — not conclusive but
  zero gain from +0.05 local eval). **Mechanism = overfitting / inverted-U
  generalization:** the local eval suites are the same opponents the policy
  trains on, so past ~u860 rising local eval = memorizing those opponents +
  entropy-collapsing into over-confident locally-optimal moves that give worse
  advice on the cloud's unseen hard ticks. **local eval and cloud transfer
  decoupled, then went anti-correlated.** Consequences:
  - **Deploy/promote `confpol`-u860 (0.634)** — the EARLIER checkpoint, not the
    final one. Frozen at `pandemonium-v1-best-u860.pt` / bucket `handoff/`.
  - **Never select an AE checkpoint by local eval** — it picks the overfit one.
    Save periodic checkpoints next run; the cloud-optimum is likely even earlier
    than u860 (we didn't keep intermediates to test).
  - **Phase 2 (self-play) — KILLED 31 May at u224/782, flat.** Warm-started from
    phase1-`latest` (u1563, baseline greedy eval **0.7242**, loaded fine — NOT a
    load failure). One update at the reset lr (2.5e-5→**2.5e-4**, 10×) + re-armed
    entropy bonus drove entropy 0.030→0.20 and the GREEDY eval 0.7242→0.3412 in a
    single update — the argmax flipped on many near-tie states. Nothing was
    "destroyed" (features intact, 0.7242 confirmed loaded); it was *deliberately
    re-stochasticized* and then **never recovered**: best froze at 0.3743 (u60),
    eval bounced 0.19–0.37 flat for 160 updates. Right death (re-exploring,
    finding nothing better); its cloudsuite gate is the wrong selector anyway.
    Superseded by confpol-native below.

  **NEXT BIG BET — "confpol-native" gated PPO — BUILT + LAUNCHED 31 May.**
  Smoking-gun evidence: the SAME extra training (u860→u1400) moved PURE policy UP
  (0.507→0.550, n=4) but CONFPOL DOWN (0.634→0.600). Opposite responses ⇒ the
  policy is trained as a global *controller* but deployed as a *consultant* on
  the heuristic's hard ticks — a train/deploy mismatch. Fix = train the raw CNN
  policy ON the deployed distribution. **Implemented** in
  [../training/ae/train_ppo.py](../training/ae/train_ppo.py):
  - `--confidence-gated` → `collect_rollouts_gated`: heuristic (`planner.ae`)
    runs every tick (owns confidence + bomb-safety); on **confident** ticks it
    plays the heuristic action, does NOT advance the frame-stacker, logs no
    transition (reward accrues to the last logged low-conf transition, semi-MDP);
    on **low-conf** ticks the policy acts AND logs. Critical match: the deploy
    `PolicyAEManager.ae()` only `observe()`s its stacker when called (= on
    low-conf ticks), so training must too — that's why the stacker is skipped on
    confident ticks. Gate replicates the deploy wrapper exactly (eps 5.0 / floor
    10.0 / override_target_none) — see `_confpol_low_confidence`.
  - `evaluate()` gains a gated path (scores the live actor through the same
    heuristic-first gate, not the misleading pure-policy score).
  - `--checkpoint-every` writes an unconditional `-u<N>` ladder — fixes last
    run's fatal "kept no intermediates" (local eval is anti-correlated with
    cloud-consultant value past ~u860, so we FARM the ladder, never trust the
    best-by-eval).
  - Launcher [../training/ae/run_confpol_native.py](../training/ae/run_confpol_native.py):
    warm-start u860 (actor+critic), **gentle lr 1e-4** (NOT 2.5e-4 — phase-2
    proved high lr+entropy on a warm policy is destructive), full-rl opponent
    mix, eval-every 20, checkpoint-every 25, ~2976 updates (~2.7-day budget).
    Run: `nohup caffeinate -is .venv/bin/python -u training/ae/run_confpol_native.py`
    → `confpol-native.log`; kill `pkill -f run_confpol_native.py`. Smoke-verified
    31 May: warm-start ok, gate fires consult%≈33–42 (matches the ~46% deploy
    measurement), gated eval > heuristic baseline, ladder checkpoints write.
  - Caveat (still trains on local opponents): the macro version of this failed
    27–28 May (conf-hybrid-v3/v4, cloud 0.507–0.549) — but our RAW policy already
    beats that as a consultant (0.634), so gated-specializing it should help.
  - **Promotion gate:** farm 3–4 ladder rungs (early/mid/late) under
    `AE_MODE=confidence_policy_hybrid` on cloud (n≥5), re-farm the incumbent
    heuristic n≥5 in the same window, promote a rung only if it clears 0.634 by a
    visible margin. **confpol-u860 (0.634) remains the deploy floor regardless.**

  **RESULTS (1 Jun, early n=1 reads — VERDICT: matched u860, no lift).** Three
  single cloud submissions (0/30 errors each), logged in the ledger as
  `confpol-native-u100/u200/u360`:
  | rung | cloud acc | speed | local eval / entropy at save |
  |---|---:|---:|---|
  | u100 | 0.551 | 0.842 | 0.520 / ent 0.147 (still exploring) |
  | u200 | 0.621 | 0.844 | 0.535 / ent 0.109 |
  | u360 | 0.624 | 0.838 | 0.561 / ent 0.028 (collapsed) |

  **Read:** at n=1 (±0.10) the three are statistically indistinguishable from
  each other (pairwise gaps ≤1σ) — cannot rank the rungs. More important, the
  cluster (avg ~0.599, best single 0.624) does **NOT** beat the **farmed**
  `pandemonium-confpol` u860 (0.6342, n=13) — a single 0.624 draw is below a
  farmed 0.634 and single highs reliably regress here. confpol-native landed in
  the **same ~0.60–0.63 consultant band as u860**: the train/deploy-match
  hypothesis did **not** produce a visible cloud lift. (The "early-is-better"
  inverted-U prediction also didn't appear — u100 was *lowest* — but that's
  within noise too.) The local run plateaued (flat 0.56 eval, entropy collapsed
  to ~0.01 from ~u280) and cloud confirms later rungs don't separate, so more
  training is low-prior. **Decision is now farming, not steps.** If anyone wants
  to truly settle it: farm the best rung (u200 or u360) to n≥5 and check the
  *mean* vs 0.634 — but the strong prior is it ties/loses. **Ship/keep
  confpol-u860 (0.634).** The confpol-native run (`run_confpol_native.py`) was
  still running at u502/2976 when this was written — low value, fine to
  `pkill -f run_confpol_native.py`; the ladder `confpol-native-u*.pt` is on disk
  if a deeper farm is ever wanted.

### Key files

- Manager (the thing we edit): [src/ae_manager.py](src/ae_manager.py)
- Server (reset-robustness patch applied): [src/ae_server.py](src/ae_server.py)
- Wrappers: [src/confidence_policy_hybrid_manager.py](src/confidence_policy_hybrid_manager.py)
  (raw-policy confidence gate, active Pandemonium deploy),
  [src/confidence_hybrid_manager.py](src/confidence_hybrid_manager.py) (macro-only),
  [src/macro_hybrid_manager.py](src/macro_hybrid_manager.py),
  [src/scripted_hybrid_manager.py](src/scripted_hybrid_manager.py) (last two are
  closed lines; kept for reference)
- Fixed-map detector probe: [../training/ae/probe_fixed_map.py](../training/ae/probe_fixed_map.py)
  (proves `is_fixed_novice_map` fires against the real env; runs Mac or Workbench)
- Hardcoded novice map data: [src/novice_map_data.py](src/novice_map_data.py)
  (walls/destructibles/bases/spawns/items — byte-identical to the competitor's)
- Local gate: [../training/ae/multi_seed_eval.py](../training/ae/multi_seed_eval.py)
- Cloud decision tool: [../training/ae/variance_farm.py](../training/ae/variance_farm.py)
  + ledger `training/ae/data/cloud_samples.json`
- Active experiment launcher:
  [../training/ae/run_pandemonium_v1.py](../training/ae/run_pandemonium_v1.py)
- Training runbook: [../training/ae/RUNBOOK.md](../training/ae/RUNBOOK.md)
- **Competitor reference** (cloned for analysis, not vendored):
  `github.com/curryfarmer/til-26-ae` (royal-recruits, 0.715/0.807). Key files to
  port-study: `experimental_heuristic/agent.py` (`_try_opening_playbook` +
  `act()` pipeline), `experimental_heuristic/novice_state.py`
  (`HUNTER_OPENING_SEQUENCES`, dist/opponent LUTs), `experimental_heuristic/eval.py`
  (`score_plan` forward-sim), `experimental_heuristic/search.py` (A* over
  `(x,y,facing,t,bombs,placed)`).
- Deployment contract & full submission ledger: see
  [`ae/NOTES-ARCHIVE.md`](NOTES-ARCHIVE.md)
  (*Full AE submission ledger*, *What our agent runs on*).

---
## Detailed history (archive)

The reverse-chronological per-session log (raw numbers, reproduction commands,
negative results from 19 May–2 June) now lives in
[`ae/NOTES-ARCHIVE.md`](NOTES-ARCHIVE.md). The digest above is the current
distilled state; open the archive only when you need the exact detail behind a
claim here.
