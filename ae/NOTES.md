# AE — notes & history

Last updated: 13 May 2026 23:55 SGT

Per-task working log for AE (Autonomous Exploration / Bomberman). For
input/output spec see [README.md](README.md). For submission history across
all tasks see [../RESULTS.md](../RESULTS.md).

## Current shipped tag

**`planner-v3b` — official 0.499 / 0.853 (13 May 23:42 SGT, 0/30 errors).**

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

## Next steps

1. **Behavior cloning + PPO** (started in [../training/ae/](../training/ae/)).
   Scaffolded as of 13 May 23:55 SGT:
   - `encoder.py` — observation → tensors (CNN-ready)
   - `model.py` — `PolicyNetwork` (~150k params, designed for <5 ms CPU
     inference so we don't lose the speed gains)
   - `collect_bc.py` — rolls out the planner in `til_environment.bomberman_env`,
     dumps (obs, action) pairs to `.npz`
   - `train_bc.py` — supervised CE with `log(action_mask)` in the logits
   - `eval_policy.py` — local eval harness reusing `test/test_ae.py` math
   - **TODO**: `train_ppo.py` (single-agent gym wrapper around PettingZoo
     env, BC-initialized PPO with mixed opponents), deployment path in
     `ae/src/ae_manager.py` that loads weights if present and falls back
     to the BFS planner.
2. **Per-game logging during `til test`** if PPO doesn't help. Capture
   step-by-step (obs, action, reward) JSONL during a local run, then
   compare against post-mortem reasoning about what the official eval
   probably did differently. Cheap and cumulative.
3. **Speed micro-optimizations** if blended score becomes the constraint:
   move from python:3.11-slim to a slimmer base, ahead-of-time-compile the
   manager with mypyc, or split the BFS into native C with cython. Only
   worth it after PPO either lands or fails — current speed (0.853) is
   already close to target.

## Reproducibility / pointers

- Manager source: [src/ae_manager.py](src/ae_manager.py)
- Server (reset-robustness patch applied): [src/ae_server.py](src/ae_server.py)
- Local Mac test: spin up `til_environment.bomberman_env` directly (no Docker needed for Mac iteration)
- Workbench Docker flow: `til build ae <tag> && til test ae <tag>`
- Official AE env (read-only submodule): [../til-26-ae/](../til-26-ae/)
- Implementation plan / decisions: [../AE_IMPLEMENTATION_PLAN.md](../AE_IMPLEMENTATION_PLAN.md) (if present)
