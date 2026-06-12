# docs/ — orientation

This directory holds supplementary write-ups that sat alongside the five model services in this
repo. The code lives in the task directories (`ae/`, `asr/`, `cv/`, `nlp/`, `noise/`); each task's
`NOTES.md` (next to its code) is the running engineering notebook — decisions, gotchas, and
dead-ends. This folder holds [`RESULTS.md`](RESULTS.md), the cross-task scoreboard and
submission history.

> The dated design specs and implementation plans behind shipped features (written as part of a
> gated brainstorm → spec → plan → eval → ship workflow), along with anonymized notes on other
> teams' documented approaches, are kept in a private archive and are not part of this public
> repo. The cross-cutting takeaway from studying other teams, which we acted on: hand-coded /
> rule-based cores transferred to the hidden eval better than learned policies, and
> inference-time tricks (resolution, quantization) often beat raw model capacity.

## Per-task `NOTES.md`

The running engineering notebook for each task lives next to its code, not here:

- [`ae/NOTES.md`](../ae/NOTES.md) · [`asr/NOTES.md`](../asr/NOTES.md) · [`cv/NOTES.md`](../cv/NOTES.md) · [`nlp/NOTES.md`](../nlp/NOTES.md) · [`noise/NOTES.md`](../noise/NOTES.md)

Each tracks current state, shipped config, levers tried, and gotchas for that task. Start there
for task-specific decisions; use [`RESULTS.md`](RESULTS.md) for the cross-task scoreboard
and submission history.
