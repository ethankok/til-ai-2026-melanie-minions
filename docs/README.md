# docs/ — orientation

This directory holds supplementary write-ups that sat alongside the five model services in this
repo. The code lives in the task directories (`ae/`, `asr/`, `cv/`, `nlp/`, `noise/`); each task's
`NOTES.md` (next to its code) is the running engineering notebook — decisions, gotchas, and
dead-ends. This folder currently holds the **competitor learnings**.

> The dated design specs and implementation plans behind shipped features (written as part of a
> gated brainstorm → spec → plan → eval → ship workflow) are kept in a private archive and are
> not part of this public repo.

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
