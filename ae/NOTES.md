# AE — notes & history

Last updated: 21 May 2026 — **Code-only PPO retry path added. `training/ae/train_ppo.py` now has a `--preset qualifier-best` recipe for a stronger PPO attempt: pressure curriculum (`scripted → cloudsuite → league`), action-masked PPO with target-KL early stopping, clip/entropy schedules, adaptive visit-count exploration shaping, health/base-health shaping, small bomb-use shaping, and weighted checkpoint selection across `random,scripted,cloudsuite` instead of one friendly local eval. This has only passed a 1-update smoke test on the Mac; no new PPO checkpoint has been trained, built, submitted, or promoted. Current AE high remains `fixed-map-v5` (`0.630 / 0.858`).**

Prior update: 21 May 2026 17:45 SGT — **Current AE high is `fixed-map-v5` (`0.630 / 0.858`). The repo runtime files are restored to the source shape that built that tag: fixed-map-v3-era `ae/src` runtime code with `AE_MODE=hybrid`, plus the restored `deployed-bc-v1.pt` artifact supplied locally on Workbench (`sha256 746bbe8198e77666d45ab9d9c6b4bb322a8f4de8faec1dab7343fc63ff4b73aa`). `ae/src/ae_manager.py` itself was already unchanged from the v5 build; the important restore was reverting later `model.py` / `policy_manager.py` legacy-checkpoint support and the temporary Dockerfile heuristic lock.**

Prior update: 21 May 2026 17:15 SGT — **`fixed-map-v5-finetune-v1` reached local Docker `0.85025` and checkpoint eval `0.6678`, then cloud regressed to `0.587 / 0.848`. Treat this as local/scripted PPO overfit; do not continue PPO as the default path.**

Prior update: 21 May 2026 05:15 SGT — **Restored the current AE runtime to the `ae-fixed-map-v3` best-model source state (`4c00f92`): `ae/src/ae_manager.py`, `ae/src/hybrid_manager.py`, and `ae/Dockerfile` now match the winning fixed-map-v3 code/config again. This removes later item-prior/macro/base-defense/pessimistic-search runtime changes from the shipping path. Use this state for the next Workbench build if the goal is to preserve the current high (`0.614 / 0.860`).**

Prior update: 21 May 2026 05:05 SGT — **Cloud rejected `pessimistic-mini-search-v1`: Workbench `til test` scored only `0.456`, then cloud returned `0.396 / 0.847` with 0/30 errors. The local Mac gate had predicted pressure-suite `cloudsuite 0.3962` almost exactly, but that did not transfer upward and is far below `ae-fixed-map-v3` (`0.614 / 0.860`) and `ae-item-confidence-v1` (`0.593 / 0.844`). Remove/disable `AE_PESSIMISTIC_SEARCH` in the Docker runtime; keep the code only as evidence that deterministic safety search mostly trades away farming/attack tempo for base preservation.**

Prior update: 21 May 2026 04:05 SGT — **Added and baked `pessimistic-mini-search-v1`, a deterministic safety-only override behind `AE_PESSIMISTIC_SEARCH=1`, `AE_PESSIMISTIC_DEPTH=3`, `AE_PESSIMISTIC_ENEMY_GATE=1`, `AE_PESSIMISTIC_FORCE_SCORE=-80`, and `AE_PESSIMISTIC_MIN_DELTA=55`. It is not MCTS: it searches only our own 3-ply movement near live bombs / fresh enemy pressure and requires a large safety delta before overriding the fixed-map planner. Mac 24-round gate: random `0.5613`, library `0.5184`, cloudsuite `0.3962`, aggregate `0.4920`. This improves the pressure/cloudsuite target over the `0.3186` reference and cuts visible-bomb failures to `8`, but it sacrifices random/library farming tempo, so submit as an explicit pressure A/B rather than treating it as a guaranteed cloud high.**

Prior update: 21 May 2026 03:10 SGT — **Added a Mac-only fixed-route sweep harness at `training/ae/sweep_fixed_routes.py`. It instantiates `AEManager` with fixed-Novice-map strategy profiles (`attack_cells_close`, `center_then_attack`, `base_leash`, `low_ammo_base_race`, etc.), ranks them by `cloudsuite`, and writes JSON reports under ignored `training/ae/data/`. Smoke command verified end-to-end: `.venv/bin/python training/ae/sweep_fixed_routes.py --profiles smoke --rounds 1 --suites cloudsuite --summary-out training/ae/data/fixed-route-suite-smoke.json`. This is a search/gate tool only; no production behavior changed.**

Prior update: 21 May 2026 02:20 SGT — **Mac-first gate rejected `ttd-defense-v1` before any Workbench build/submit. The candidate added a narrow base-threat TTD/TTI override plus failure-class counters, then ran `python training/ae/validate_cloud_suite.py --rounds 24 --suites random library cloudsuite --our heuristic --summary-out training/ae/data/ae-ttd-defense-v1.json` locally on the Mac. Results: random `0.6446`, library `0.5475`, cloudsuite `0.2714`, aggregate `0.4878`. Because cloudsuite regressed below Candidate B's `~0.318`, the behavior is disabled by default (`AE_TTD_DEFENSE=0`) and should not be pushed as a submit candidate. The useful keep is the diagnostic: base failures are mostly `visible_enemy_bomb`, so future work should react to already-visible bombs rather than broad enemy proximity or global own-bomb changes.**

Prior update: 21 May 2026 02:05 SGT — **Workflow rule after the `ally-bomb-safe-v2` failure: for future AE changes, run `python training/ae/validate_cloud_suite.py --rounds 24 --suites random library cloudsuite --our heuristic --summary-out training/ae/data/ae-candidate-cloudsuite.json` locally on the Mac before pushing to Workbench, building, or submitting. If `cloudsuite` clearly regresses, stop locally. Workbench `til test` is only a packaging/random-NPC sanity check, not the behavioral gate.**

Prior update: 21 May 2026 02:00 SGT — **Submitted the surgical `ally-bomb-safe-v2` A/B and it failed hard on cloud: local Docker `til test` improved to `0.6395`, but the 24-round pressure gate already warned against it (`cloudsuite 0.2870`, down from Candidate B's `~0.318`, with own-base destroyed still large), and cloud returned only `0.369 / 0.847` with 0/30 errors. Lesson: allied bombs are harmless for same-team damage in the public env, but treating them as strategically "free" is wrong. Bombs still consume the shared team bomb budget, occupy/reshape space, invite policy over-bombing, and remove useful flee/leash behavior; random `til test` rewarded extra bombing while hidden pressure punished the lost positioning. Do not continue this branch.**

Prior update: 21 May 2026 early SGT — **Completed trial and finalized Candidate B ("base-minefield-v1"). Proactive defense triggers (base health < 100 or enemy in proximity) and close-quarters tactical bombing near own base improved local `cloudsuite` mean from `0.2643` to `0.3182` (final base health up from 0 to 30) without degrading attack metrics. Trialed Candidate C ("fixed-map defensive chokepoints") but it regressed `cloudsuite` to `0.2044` due to blocking/pathing conflicts, and was reverted. `candidate-b` was submitted after local Docker `0.538` and landed cloud `0.500 / 0.858`; it narrowed the local-cloud gap but did not beat `ae-fixed-map-v3` or `ae-item-confidence-v1`.**

Prior update: 20 May 2026 late-night SGT — **Workbench pulled commit `725c097` and ran the new diagnostic gate for 24 rounds per suite. Current defaults scored random `0.7462`, library `0.5564`, cloudsuite `0.3186`, aggregate `0.5404`; report saved on Workbench at `training/ae/data/ae-diagnostic-check.json`. Diagnostics confirm the current failure mode: under `cloudsuite`, our attack output is decent (`attack_kill_or_multi=+4759.5`, `destroy_enemy_base=+4297`) but base survival is still broken (`base=0.0`, `base_damage=-2098`, `own_base_destroyed=-1680`). No submit candidate tonight; next AE work should target base survival under pressure without killing attack tempo.**


Prior update: 20 May 2026 ~23:45 SGT — **Added diagnostics to the local AE validation suite and tightened simulator seeding. No new behavior candidate was accepted. The gate now reports bombs placed, visited cells, final health/base health, early-end rate, action counts, and inferred reward components. Tested the five proposed directions one by one: stronger/conditional item-prior variants, opening-book bonuses, pressure-mode switching, and base-defense value bumps were all rejected or left off after `cloudsuite` gates. The strongest safe outcome is diagnostics-only; keep runtime behavior at `ae-item-prior-strong-v1` defaults (`AE_ITEM_PRIOR_CONF=0.70`, `AE_ITEM_PRIOR_FLOOR=0.25`) until a candidate beats it robustly.**

Prior update: 20 May 2026 ~22:35 SGT — **`ae-item-prior-strong-v1` was built/tested but not cloud-evaluated: `til submit` reached Artifact Registry upload, then ended with `context canceled` / `Terminated`. Local evidence does not justify an automatic retry: 12-round suite random `0.7012`, library `0.5712`, cloudsuite `0.2848`, aggregate `0.5190`; Docker `til test` `0.7245`. This is weaker than `ae-item-confidence-v1` on aggregate (`0.5501`) and Docker (`0.7435`), even though library improved. Diagnosis: stronger fixed-item priors likely over-commit to stale/static item routes. Keep `ae-fixed-map-v3` as high score and `ae-item-confidence-v1` as the best confirmed new run.**

Prior update: 20 May 2026 ~22:05 SGT — **Submitted `ae-item-confidence-v1`: cloud `0.593 / 0.844`, 0/30 errors. This beats `heuristic-tweaks` by `+0.055` and `hybrid-v3` by `+0.038`, but stays `-0.021` below the `ae-fixed-map-v3` high (`0.614 / 0.860`). Workbench Docker `til test` was `0.7435`; the 12-round validation suite was random `0.7959`, library `0.5499`, cloudsuite `0.3045`, aggregate `0.5501`. Diagnosis: item-confidence/respawn logic transferred well enough to clear 0.5, but it likely under-trusted the fixed Novice initial item priors. Next candidate `ae-item-prior-strong-v1` raised default fixed-item prior confidence from `0.58` to `0.70` and prior floor from `0.18` to `0.25`; macro routing stayed off.**

Prior update: 20 May 2026 ~21:31 SGT — **Implemented the local-cloud-gap response: item targets now use confidence/respawn priors instead of assuming every fixed-map item is currently live; fixed-map macro attack cells exist but are opt-in after local pressure-suite A/Bs; added a `cloudsuite` rusher/hunter validation pool plus `training/ae/validate_cloud_suite.py`; wired `cloudsuite` into simulator, PPO training/eval, and opponent-model fitting. Local 6-round smoke with shipping defaults: random `0.741`, library `0.555`, cloudsuite `0.306`, aggregate mean-of-means `0.534`. Next Workbench candidate tag: `ae-item-confidence-v1`; submit only if Docker `til test` is healthy.**

Prior update: 20 May 2026 ~21:00 SGT — **Submitted parameter-swept optimized heuristics (`heuristic-tweaks` tag, cloud score: 0.538 / 0.851, local mixed score: 0.5373). While parameter-swept optimal_combo parameters (Dijkstra soft penalty, low-ammo scaling, base panic defense, enemy chase) beat the baseline locally, they underperformed against the cloud's hidden opponent distribution compared to the 0.614 all-time high set by `ae-fixed-map-v3` (baseline). Revert to `ae-fixed-map-v3` as our shipped candidate.**

Prior update: 20 May 2026 ~18:25 SGT — **Submitted `ae-fixed-map-v3` and scored cloud 0.614 / 0.860 (new all-time high score!). Resolved escape thrashing, relaxed map detection to base-only for 100% activation consistency across randomized slots/spawn-points, and fixed base-defense camping loops.**
**Architecture**: PPO policy (`ppo-v1` weights) wrapped by `HybridAEManager` (heuristic safety-veto + top-K policy cascade) + `AEManager`'s heuristic dominant-action shortcut firing on adjacent enemy *agents* (not just bases). Mode selection via `AE_MODE` env in `ae/Dockerfile` or `ae/src/.ae_mode` fallback (`hybrid` | `policy` | `heuristic`). Currently `hybrid`. MCTS-light is off by default again (`AE_MCTS=0`, `AE_HYBRID_TRUST_MCTS=0`) because v1 timed out and v2 regressed; opt in only for explicit A/Bs. Weights on Workbench: `ae/models/bc.pt` ← `~/ae-checkpoints-backup/deployed-bc-v1.pt`.

**Central finding, revised 19 May**: the hybrid wrapper has very little slack. Removing MCTS cost `-0.073` cloud accuracy, removing the frozen-STAY veto cost `-0.101`, and lowering the policy confidence gate to `0.3` cost `-0.144`. The most important ablation is `AE_HYBRID_VETO_FROZEN_STAY`: local suggested it barely mattered (`0.6866`, worse than baseline by only ~0.013), while cloud said it was more important than MCTS. That is the first clean local→cloud direction flip at component level, so local A/Bs are not just noisy in aggregate; they can mis-rank individual wrapper parts.

**Novice-map correction**: Claude's suggested "train Novice-specific PPO" direction had one wrong premise. `training/ae/train_ppo.py` already defaults to Novice mode: `config.env.novice = (not args.vary_maps) and args.novice`, and `--novice` defaults true. `ppo-v1` and `ppo-scripted-v1` were therefore already fixed-Novice unless their command used `--vary-maps` or `--no-novice`; only `ppo-v2` explicitly explored varied maps. A new explicit `--novice --seed 42 --eval-seed 42` run can still be useful as a controlled rerun with better logging/keepalive, but it is **not** a brand-new structural lever by itself. Treat it as lower-EV than the speedcheck and 0.9-team intelligence.

**Controlled Novice PPO rerun result (19 May)**: command used `--bc-checkpoint ~/ae-checkpoints-backup/deployed-bc-v1.pt --out ~/ae-checkpoints-backup/novice-fixed-v1 --updates 200 --opponents scripted --eval-opponents scripted --novice --eval-every 5 --eval-games 30 --seed 42 --eval-seed 42`. Training finished cleanly after `223.2m`; best eval was `0.5915` at update 200, with prior peaks `0.5497` at update 105 and `0.5677` at update 140. Docker validation then copied the checkpoint to `ae/models/bc.pt`, set `ae/src/.ae_mode` to `hybrid`, built `novice-fixed-v1`, and `til test ae novice-fixed-v1` scored `3094.0` total reward / `0.5156666667` over 6 rounds. This is below even the shipped hybrid-v3 cloud score and far below prior local candidates that failed to transfer (`ppo-scripted-v1` local `0.741` -> cloud `0.450`; hybrid-v3 local `~0.774` -> cloud `0.555`). Decision: preserve checkpoint for analysis, **do not submit**.

**`hybrid-v3-speedcheck` result (19 May 14:14 SGT)**: pulled `melanie-minions-ae:hybrid-v3` from Artifact Registry, retagged/submitted as `hybrid-v3-speedcheck`, and cloud returned `0 / 30` errors, score `0.381`, speed `0.855`. Docker inspection showed both tags point to identical local image `sha256:5bc18298206d7958de6157019b42a7709891d5e9c3e21d9b51a1d690685aacf0` and registry digest `sha256:83c999a6f833ea464fe9742604831cc70c5eb58f4efbd109c3db5ce8d2445fea`. This proves the retag was clean, but not that `83c999...` is the original 14 May high-scoring image. Speed recovery proves the broad 0.60 speed issue was not global evaluator congestion; the severe accuracy drop means we need the original immutable digest/build log before making a same-bytes variance claim.

**Provenance blocker (19 May)**: Workbench command `gcloud artifacts docker images list asia-southeast1-docker.pkg.dev/til-ai-2026/repo-til-26-melanie-minions/melanie-minions-ae --include-tags --format="table(version.basename(), updateTime, tags)" --sort-by="~updateTime"` failed with `PERMISSION_DENIED: Permission 'artifactregistry.versions.list' denied` for active account `svc-melanie-minions@til-ai-2026.iam.gserviceaccount.com`. This means we can pull/submit but cannot list old Artifact Registry versions from Workbench. Provenance hunt is deferred; later options are local shell/log search, Docker history, Discord/leaderboard build logs, or someone with Artifact Registry version-list permission.

Per-task working log for AE (Autonomous Exploration / Bomberman). For the authoritative input/output/scoring spec see [README.md](README.md) and the official [Challenge specifications](https://github.com/til-ai/til-26/wiki/Challenge-specifications#ae). For submission history across all tasks see [../RESULTS.md](../RESULTS.md). For training pipeline see [../training/ae/RUNBOOK.md](../training/ae/RUNBOOK.md).

## Full AE submission ledger (cloud scores)

| Tag | Cloud | Speed | Notes |
|---|---:|---:|---|
| **fixed-map-v5** | **0.630** | **0.858** | **Current high. Fixed-map-v3-era runtime source plus restored `deployed-bc-v1.pt`; current repo runtime restored to this source shape.** |
| fixed-map-v5-finetune-v1 | 0.587 | 0.848 | FAILED. True legacy-policy fine-tune; local Docker `0.85025`, checkpoint eval `0.6678`, cloud regressed. Do not continue PPO as default. |
| **ae-item-confidence-v1** | **0.593** | **0.844** | **Item-confidence/respawn priors; second-best AE cloud score, `-0.021` vs `ae-fixed-map-v3`.** |
| pessimistic-mini-search-v1 | 0.396 | 0.847 | FAILED. Local Docker `0.456`; local cloudsuite `0.3962` predicted cloud almost exactly. Safety search preserved base better but lost too much attack/farming tempo. Disabled by default. |
| ttd-defense-v1 | local rejected | — | Mac gate failed before Workbench: random `0.6446`, library `0.5475`, cloudsuite `0.2714`; disabled by default. |
| ally-bomb-safe-v2 | 0.369 | 0.847 | FAILED. Local Docker rose to `0.6395`, but pressure gate regressed (`cloudsuite 0.2870`) and cloud collapsed. Removing own-bomb escape/base safety globally caused over-bombing and worse hidden-pressure positioning. |
| candidate-b | 0.500 | 0.858 | Base-minefield-v1. Local Docker `0.538`; narrowest local-cloud gap so far, but lower absolute cloud than current high and second-best. |
| ae-item-prior-strong-v1 | local only | — | Built/tested; submit push canceled before cloud eval. Local suite aggregate `0.5190`, Docker `0.7245`; weaker than `ae-item-confidence-v1`, so do not blindly retry. |
| **heuristic-tweaks** | **0.538** | **0.851** | **Swept optimal_combo parameters (Dijkstra soft penalty, low-ammo scaling, base panic defense, enemy chase).** |
| **ae-fixed-map-v3** | **0.614** | **0.860** | **Fixed-map exploitation with Dijkstra pathfinding + relaxed map detection (current high score).** |
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
| hybrid-v3-no-mcts | 0.482 | 0.598 | Removing MCTS/trust path regressed accuracy `-0.073`; local direction matched. |
| hybrid-v3-no-vetofrozen | 0.454 | 0.556 | Removing frozen-STAY veto regressed accuracy `-0.101`; local direction flipped. |
| hybrid-v3-conf-0.3 | 0.411 | 0.591 | Lower confidence gate was even worse than `conf50`; policy gating is cloud-bad. |
| heur-restore-v3 | — | 0.602 | Speed-only note from cloud result set; broad ~0.25 speed loss vs 14 May baseline. |
| heur-restore-v3-bombfix | — | 0.616 | Speed-only note from cloud result set; still broad speed regression. |
| novice-fixed-v1 | local only | — | Scripted-opponent Novice PPO rerun best eval `0.5915`; Docker `til test` scored `0.5157`, so do not submit. |
| hybrid-v3-speedcheck | 0.381 | 0.855 | Pulled registry `hybrid-v3`, retagged, resubmitted. Both tags point to image `5bc182...` / digest `83c999...`; speed healthy, accuracy collapsed. Need original 14 May digest to prove same-image variance. |
| **ae-fixed-map-v1** | **local 0.7466** | **—** | **Option A fixed-map exploitation (Dijkstra cost-aware path planner + proactive wall bombing + pre-populated map layout).** |

## Cloud gap by submission (the structural pattern)

| Submission | Local (apples-to-apples) | Cloud | Gap |
|---|---:|---:|---:|
| ae-item-confidence-v1 | 0.7435 Docker / 0.5501 suite mean | 0.593 | 0.1505 vs Docker |
| pessimistic-mini-search-v1 | 0.456 Docker / 0.3962 cloudsuite | 0.396 | 0.060 vs Docker; cloudsuite predicted cloud but absolute score was too low |
| ttd-defense-v1 | 0.4878 suite mean / 0.2714 cloudsuite | not submitted | Mac gate failed; behavior disabled |
| ally-bomb-safe-v2 | 0.6395 Docker / 0.4978 suite mean / 0.2870 cloudsuite | 0.369 | 0.2705 vs Docker; failed despite better random local |
| candidate-b | 0.538 Docker / ~0.318 cloudsuite | 0.500 | 0.038 vs Docker; narrow gap but low ceiling |
| ae-item-prior-strong-v1 | 0.7245 Docker / 0.5190 suite mean | not evaluated | submit push canceled |
| ae-diagnostic-check | random 0.7462 / library 0.5564 / cloudsuite 0.3186 | not submitted | Workbench 24-round diagnostic gate after `725c097`; confirms base survival failure under cloudsuite |
| diagnostic-gated local A/Bs | 24-round cloudsuite probes | not submitted | behavior changes rejected; diagnostics-only accepted |
| heuristic-tweaks | 0.5373 | 0.538 | -0.0007 |
| ae-fixed-map-v3 | 0.5225 | 0.614 | -0.0915 |
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
| hybrid-v3-no-mcts | 0.6767 | 0.482 | 0.195 |
| hybrid-v3-no-vetofrozen | 0.6866 | 0.454 | 0.233 |
| ae-fixed-map-v1 | 0.7466 | — | — |

Pattern: ~0.19-0.30 gap for most optimistic random-local candidates, with two instructive exceptions. `candidate-b` had a tiny Docker-cloud gap because it lowered random-local performance toward the cloud floor, not because it beat the best cloud tags. `ally-bomb-safe-v2` looked better on random Docker but worse on the pressure gate and then collapsed on cloud, confirming that random `til test` can reward bomb tempo that hidden-pressure opponents exploit.

## 19 May handoff — paused state

1. **Do not submit `novice-fixed-v1` or the latest ablations.** `novice-fixed-v1` failed local Docker (`0.5157`); `no-mcts`, `no-vetofrozen`, and `conf-0.3` all hurt cloud accuracy.
2. **Do not build `hybrid-v3-lean` just because speed recovered.** The speedcheck result was `0.381 / 0.855`, so speed is fine but accuracy is not. The retag was clean (`hybrid-v3` and `hybrid-v3-speedcheck` both resolve to digest `83c999...`), but we still need proof that this digest is the original 14 May high-scoring image.
3. **Provenance hunt is deferred.** Workbench service account cannot list Artifact Registry versions (`artifactregistry.versions.list` denied). Later, search local logs/history or ask someone with registry-list permission to confirm whether digest `83c999...` was the original 14 May `hybrid-v3`.
4. **If AE reopens, start from external intelligence.** A public 0.9 AE score implies a structural unlock. Given repeated local-cloud reversals, prioritize clues about map memorization, hand-coded route tables, reward farming, or opponent assumptions before another blind PPO day.

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

## What remains worth doing after 19 May

**Immediate**: paused. The Workbench service account cannot list Artifact Registry versions, so proving whether registry digest `sha256:83c999a6f833ea464fe9742604831cc70c5eb58f4efbd109c3db5ce8d2445fea` is the original 14 May `hybrid-v3` high-scoring image is deferred. The retag itself was clean: both local tags resolve to image `sha256:5bc18298206d7958de6157019b42a7709891d5e9c3e21d9b51a1d690685aacf0`.

**If the tag is not original**: stop using mutable registry tags for same-bytes conclusions. Only compare immutable digests or freshly built, documented images.

**If the tag is original**: keep `hybrid-v3` as shipped via leaderboard retention and treat AE accuracy as hidden-eval unstable/changed. Spend AE time on structural intelligence. A public 0.9 score means someone found a different frame; likely candidates are fixed-map exploitation, hand-coded route/waypoint tables, or reward-farming behavior we have not observed locally.

**Tier 3 hierarchical goal selector** remains the code-heavy option. It operates on coarser state (mission/recon/resource zones, base proximity, bomb-vs-explore phase), which could be more invariant to the local-cloud opponent distribution shift that killed BC/PPO/MCTS. Cost: 2-3 days, uncertain outcome.

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
