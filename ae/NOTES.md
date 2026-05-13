# AE — notes & history

Last updated: 13 May 2026 23:10 SGT

Per-task working log for AE (Autonomous Exploration / Bomberman). For
input/output spec see [README.md](README.md). For submission history across
all tasks see [../RESULTS.md](../RESULTS.md).

## Current shipped tag

**`planner-v2` — official 0.501 / 0.771 (13 May 23:03 SGT, 0/30 errors).**

+0.056 over `planner-v1` (0.445) from a single round of correctness fixes —
most notably correcting `BOMB_TIMER = 4 → 3` so the env's true placement →
detonation budget is respected. Speed dipped slightly (`0.788 → 0.771`) from
the bounded-escape and threat-aware BFS — small price for the accuracy jump.

The local→official gap narrowed from ~0.25 (planner-v1: 0.697 → 0.445) to
~0.17 (planner-v2: ~0.669 → 0.501). Better, but **not closed**. The residual
gap is likely a mix of fixed-seed novice map being easier than hidden eval,
random opponents locally vs whatever the hidden eval uses, and 6-round
local variance.

AE is **40% of the qualifier** — biggest absolute lever in the whole repo.

## What our agent runs on

- **No model**, no learned weights. Rule-based planner only.
- **Container base**: same NVIDIA pytorch image as the other services (per [Dockerfile](Dockerfile)), but the inference loop is CPU-bound pure Python — no GPU usage. Fast by design (per-step latency dominates because the evaluator calls `/ae` once per game tick).
- **Source**: [src/ae_manager.py](src/ae_manager.py) (manager — what we edit) + [src/ae_server.py](src/ae_server.py) (server with the reset-robustness patch already applied — empty POST or `step == 0` triggers re-instantiation of `AEManager`; see top-level CLAUDE.md notes).
- **State on `self`**: belief map, frontier set, turn counter, etc. — must NOT use module-level globals because the server re-instantiates the manager on reset.

### Planner sketch (planner-v2)

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
planner-v3  pending            —       —       9/9     Multi-source BFS, predictive bombs, proactive wall break, respawn awareness, base defense, bomb chains, health retreat
planner-v3  13/05 Workbench    0.588/0.629/0.570 (local mean ≈0.596, -0.07 vs v2). Aggressive bombing wasted bombs in random-opponent eval; not submitted.
planner-v3b pending            —       —       9/9     v3 minus bomb-chains; predictive bomb requires ≥2 enemies in range-1 extended blast; threat penalty restored to 2.0/5.0
```

## Local validation history

```text
Variant     Date/time          Local score        Notes
baseline    12/05              0.051 official     Reference point only
planner-v1  13/05 10:31 Mac    0.732 local        Stateful belief + BFS + LOS-safe tactical bombs, 0 invalid actions
planner-v1  13/05 Workbench    0.697 local        Built/tested with official Workbench Docker flow before submission
planner-v1  13/05 11:33        0.445 official     ← significant drop from both local environments
planner-v2  13/05 Workbench    0.659/0.659/0.689  3-run local mean ≈ 0.669 (variance ±0.015)
planner-v2  13/05 23:03        0.501 official     +0.056 over planner-v1; gap to local narrowed but not closed
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

## Open questions

- **What's still in the residual ~0.17 local→official gap?** The novice
  local map is fixed-seed (seed 88) with random opponents over only 6 rounds.
  Hidden eval probably has either non-novice maps or smarter/different
  opponents. We don't know how the official 30 scenarios are drawn.
- **Are the threat-penalty weights too conservative?** The `-3.0` per
  threat-cell on path and `-8.0` for stepping into a threat were untuned.
  In 6-random-opponent novice the agent may be running past harmless
  enemies and avoiding items it could safely grab.
- **Is wall-bombing too gated?** Currently `wall_to_open` requires
  `_stuck_recently()`, which means we never proactively break a destructible
  wall on the shortest path to a mission/base unless we're already stuck.
  Could be leaving easy +5/+50 rewards on the table.

## Next steps

1. **planner-v3 (cheap heuristic tuning)** — diminishing returns, but two
   experiments are nearly free:
   - **(a) Softer threat penalty**: drop `-3.0`/`-8.0` to `-1.0`/`-3.0` or
     set `ENEMY_STALENESS = 1`. Workbench local should tell us if the
     planner was being too cautious.
   - **(b) Proactive wall-break**: in `_should_place_bomb`, drop the
     `_stuck_recently()` gate when `target in self.enemy_bases` or
     `target in self.last_seen_items` for missions specifically. Risk: more
     bombs placed → more self-blast if escape check is wrong. Mitigate with
     the now-bounded `_safe_escape_within`.
2. **Per-game logging** if (1) plateaus. Capture step-by-step decisions to a
   JSONL trail during `til test` — compare local-passing scenarios against
   failure modes inferred from official low-score runs.
3. **RL policy** (deferred). PPO/DQN against `til_environment.bomberman_env`
   if rule-based hits a ceiling. Per [../SUMMARY.md "Phase 3"](../SUMMARY.md),
   inference must stay fast — tiny MLP/CNN, not a transformer.

## Reproducibility / pointers

- Manager source: [src/ae_manager.py](src/ae_manager.py)
- Server (reset-robustness patch applied): [src/ae_server.py](src/ae_server.py)
- Local Mac test: spin up `til_environment.bomberman_env` directly (no Docker needed for Mac iteration)
- Workbench Docker flow: `til build ae <tag> && til test ae <tag>`
- Official AE env (read-only submodule): [../til-26-ae/](../til-26-ae/)
- Implementation plan / decisions: [../AE_IMPLEMENTATION_PLAN.md](../AE_IMPLEMENTATION_PLAN.md) (if present)
