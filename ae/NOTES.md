# AE — notes & history

Last updated: 13 May 2026

Per-task working log for AE (Autonomous Exploration / Bomberman). For
input/output spec see [README.md](README.md). For submission history across
all tasks see [../RESULTS.md](../RESULTS.md).

## Current shipped tag

**`planner-v1` — official 0.445 / 0.788 (13 May 11:33 SGT, 0/30 errors).**

A team high score and a large jump from the `0.051` random-walk baseline, but
**meaningfully worse than Workbench local 0.697 / Mac local 0.732**. The
local→official gap (~0.25) is the dominant mystery and the main work item.

AE is **40% of the qualifier** — biggest absolute lever in the whole repo.

## What our agent runs on

- **No model**, no learned weights. Rule-based planner only.
- **Container base**: same NVIDIA pytorch image as the other services (per [Dockerfile](Dockerfile)), but the inference loop is CPU-bound pure Python — no GPU usage. Fast by design (per-step latency dominates because the evaluator calls `/ae` once per game tick).
- **Source**: [src/ae_manager.py](src/ae_manager.py) (manager — what we edit) + [src/ae_server.py](src/ae_server.py) (server with the reset-robustness patch already applied — empty POST or `step == 0` triggers re-instantiation of `AEManager`; see top-level CLAUDE.md notes).
- **State on `self`**: belief map, frontier set, turn counter, etc. — must NOT use module-level globals because the server re-instantiates the manager on reset.

### Planner sketch (planner-v1)

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
planner-v2  pending            —       —       —       Bomb timer fix (4→3), bounded escape, enemy soft threat, frontier scoring
```

## Local validation history

```text
Variant     Date/time          Local score        Notes
baseline    12/05              0.051 official     Reference point only
planner-v1  13/05 10:31 Mac    0.732 local        Stateful belief + BFS + LOS-safe tactical bombs, 0 invalid actions
planner-v1  13/05 Workbench    0.697 local        Built/tested with official Workbench Docker flow before submission
planner-v1  13/05 11:33        0.445 official     ← significant drop from both local environments
planner-v2  13/05 pending      9/9 unit tests     Bomb timer 4→3 (matches env), bounded escape check, enemy soft threat, frontier scoring by unseen-yield
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

### planner-v2 (work in progress as of 13 May)

Changes against planner-v1:

- **Bomb fuse: 4 → 3 ticks**. Matches what's actually in the env. Previous
  value was guessed from documentation.
- **Bounded escape check**. Verifies that the planned escape path is
  reachable WITHIN the fuse window AND accounts for at least one enemy
  potentially blocking it.
- **Enemy soft threat**. Tiles adjacent to enemies get a positive cost
  contribution proportional to enemy step-probability into them, not a hard
  block. This lets the planner traverse near enemies when necessary but
  prefer wider berths when possible.
- **Frontier scoring by unseen yield**. Each frontier cell scored by how
  many new map cells it would reveal if visited, not just BFS distance.

Passing 9/9 local unit tests. Pending Workbench full-game test and official
submission.

## Open questions

- **Is the local env actually representative?** AE doesn't have a per-game
  WER-equivalent — the 0.25 local→official gap is large. Either the local
  scenarios are systematically easier (random seed bias?) or the official
  evaluator weights different things (e.g. objective completion > exploration).
- **How many official runs do we have?** The shipped evaluator ran 30 games
  with `0 / 30` errors. We don't know which 30 scenarios or whether they're
  randomly drawn from the same distribution as our local tests.
- **Should we instrument planner-v1 with detailed per-game logging** before
  shipping planner-v2, so we can compare failure modes? The Workbench `til
  test` does run a real game loop — could capture step-by-step decisions to
  a JSONL trail for offline analysis. This is the cheapest diagnostic if
  planner-v2 also under-delivers vs local.

## Next steps

1. **Submit planner-v2** once Workbench `til test` confirms no regression
   vs planner-v1 local (≥0.697).
2. **If planner-v2 also drops to <0.55 official**: the local env is systematically misrepresenting hidden eval. Need to:
   - Capture per-game logs from `til test` on Workbench
   - Diff against the same scenarios on the Mac local env to find local-vs-Workbench scoring discrepancies
   - Look at the official evaluator code in [../til-26-ae/](../til-26-ae/) (it's pulled in as a submodule) — submodule is read-only but reading it for understanding is fine
3. **Eventually**: a learned policy (PPO/DQN) trained against `til_environment.bomberman_env` if rule-based hits a ceiling. Per [../SUMMARY.md "Phase 3"](../SUMMARY.md), inference must stay fast — tiny MLP/CNN, not a transformer.

## Reproducibility / pointers

- Manager source: [src/ae_manager.py](src/ae_manager.py)
- Server (reset-robustness patch applied): [src/ae_server.py](src/ae_server.py)
- Local Mac test: spin up `til_environment.bomberman_env` directly (no Docker needed for Mac iteration)
- Workbench Docker flow: `til build ae <tag> && til test ae <tag>`
- Official AE env (read-only submodule): [../til-26-ae/](../til-26-ae/)
- Implementation plan / decisions: [../AE_IMPLEMENTATION_PLAN.md](../AE_IMPLEMENTATION_PLAN.md) (if present)
