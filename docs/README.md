# docs/ — orientation

This directory holds the design notes, planning artifacts, and research that sat behind the
five model services in this repo. The code lives in the task directories (`ae/`, `asr/`, `cv/`,
`nlp/`, `noise/`); this folder is the "why" and the "what we tried" — including the dead-ends.

## How we worked

Most non-trivial changes went through a gated loop: **brainstorm → spec → plan → gated eval → ship**.
A change only became the deployed default after clearing its eval gate; everything else stayed
behind a default-OFF flag or was recorded as a dead-end. The specs and plans below are the
written trail of that loop. (Agentic tooling helped author them, but the substance is the
engineering — see the specs themselves for the reasoning.)

### `superpowers/specs/` — design specs

Each spec is a design doc: problem, proposed approach, risks, non-goals. Representative ones:

- `2026-06-01-ae-semis-eval-design.md` — foreign-opponent pool + train/eval split for the AE melee gate
- `2026-06-03-ae-contention-aware-valuation.md` — discount items an opponent reaches first
- `2026-06-04-ae-farming-race-model-design.md` — two-phase stun-tax / fortress posture
- `2026-06-08-ae-time-layered-danger-map-design.md` — per-tick, chain-resolved bomb-danger map
- `2026-06-09-ae-no-self-damage-bomb-gate-design.md` — relax bomb vetoes built on a false premise
- `2026-06-02-asr-ngram-lm-fusion-design.md` — GPU-resident n-gram shallow fusion (later a dead-end)
- `2026-06-10-surprise-agent-design.md` — hybrid algorithmic + LLM agent for the 20-player FFA side challenge

### `superpowers/plans/` — implementation plans

TDD checklists (tasks → steps → commits) that execute a spec. They pair with the specs above,
e.g. `2026-06-06-ae-planner-weight-cem-tuning.md` (the CEM search over planner weights) and
`2026-06-09-ae-finals-aligned-eval-revamp.md` (the relative-rank melee gate that selected the
shipped AE checkpoint). Not every plan shipped — some are honest records of inconclusive or
reverted work.

## Competitor learnings (anonymized)

Summaries of approaches other teams documented, which we studied to inform our own. Team names
are anonymized:

- [`competitor-learnings/team-a-learnings.md`](competitor-learnings/team-a-learnings.md) — **Team A**: Whisper 4-bit quantization, CNN-DQN, P2-layer small-object detection
- [`competitor-learnings/team-b-learnings.md`](competitor-learnings/team-b-learnings.md) — **Team B**: TensorRT-compiled RT-DETRv2, PPO+LSTM, imitation seeding
- [`competitor-learnings/team-c-learnings.md`](competitor-learnings/team-c-learnings.md) — **Team C**: faster-whisper FP16, Dueling DQN with attention/residuals
- [`competitor-learnings/team-d-learnings.md`](competitor-learnings/team-d-learnings.md) — **Team D**: faster-whisper + beam search, SAHI slicing, frame stacking

The cross-cutting takeaway we acted on: hand-coded/rule-based cores transferred to the hidden
eval better than learned policies, and inference-time tricks (resolution, quantization) often
beat raw model capacity.

## Per-task `NOTES.md`

The running engineering notebook for each task lives next to its code, not here:

- [`ae/NOTES.md`](../ae/NOTES.md) · [`asr/NOTES.md`](../asr/NOTES.md) · [`cv/NOTES.md`](../cv/NOTES.md) · [`nlp/NOTES.md`](../nlp/NOTES.md) · [`noise/NOTES.md`](../noise/NOTES.md)

Each tracks current state, shipped config, levers tried, and gotchas for that task. Start there
for task-specific decisions; use [`../RESULTS.md`](../RESULTS.md) for the cross-task scoreboard
and submission history.
