# TIL-AI 2026 Submission Results

Team: `melanie-minions`
Last updated: 13 May 2026 23:55 SGT

## Latest submitted scores

```text
Task   Image                    Tag         Submitted             Errors        Score   Speed
NLP    melanie-minions-nlp      latest      12/05/2026 03:23:35   0 / 700       0.301   0.971
ASR    melanie-minions-asr      ft-lora32-v1 13/05/2026 11:22:30  0 / 400       0.957   0.849
CV     melanie-minions-cv       latest      12/05/2026 03:52:32   4 / 500       0.000   0.981
Noise  melanie-minions-noise    latest      12/05/2026 03:54:55   0 / 500       1.000   0.970
AE     melanie-minions-ae       planner-v3b 13/05/2026 23:42:57   0 / 30        0.499   0.853
AE (bc) melanie-minions-ae      bc-v1       14/05/2026 01:22:13   0 / 30        0.364   0.856  ← regressed; planner-v3b still shipped
```

## ASR submission history

```text
Tag           Submitted          Score   Speed   Local Eng-WER   Notes
v1            12/05 03:42        0.000   0.993   —               Empty-string baseline
norm-v1       12/05 16:23        0.877   0.864   0.0759          + digit verbalization + silence guard
vad-off-v1    12/05 20:00        0.938   0.859   0.0554          + VAD off + hallucination guards + ordinals + decimal-safe
ft-lora32-v1  13/05 11:22        0.957   0.849   0.0299*         + LoRA rank-32 decoder fine-tune (3 epochs, lr 1e-4)
```

*`ft-lora32-v1` local Eng-WER is **leaky** (trained on 90% of the 4110-clip test
set; the bare 0.0299 includes memorization). The held-out 10% val WER at step
1000 was **0.04662** — that's the cleaner proxy. Official WER 0.043 means the
generalization gap turned out *negative* (val 0.04662 → official 0.043), i.e.
distil-large-v3 + LoRA generalized **better** than leaky-val suggested.

## AE local validation history

```text
Variant     Date/time          Local novice score   Errors/action validity   Notes
baseline    12/05              0.051 official       0 / 30 official errors   Periodic-forward + periodic bomb baseline
planner-v1  13/05 10:31 +08    0.732 local Mac      0 invalid actions        Stateful belief map + objective/frontier BFS + LOS-safe tactical bombs
planner-v1  13/05 Workbench    0.697 local          til test completed       Built/tested with official Workbench Docker flow before submission
planner-v1  13/05 11:33        0.445 official       0 / 30 official errors   New AE high score, but hidden evaluation underperformed local test
planner-v2  13/05 Workbench    0.659/0.659/0.689    3-run mean ≈ 0.669       Bomb timer 4→3 (matches env), bounded escape check, enemy soft threat, frontier scoring by unseen yield
planner-v2  13/05 23:03        0.501 official       0 / 30 official errors   New AE high score (+0.056 vs planner-v1); local→official gap narrowed from 0.25 → 0.17
planner-v3  13/05 Workbench    0.588/0.629/0.570    3-run mean ≈ 0.596       Aggressive bombing (predictive range 2, bomb chains) + soft threat 1.0/3.0; regressed locally, NOT submitted
planner-v3b 13/05 Workbench    0.80/0.61/0.66/0.65/0.64/0.63  6-run mean ≈ 0.681  v3 minus bomb-chains; predictive bomb requires ≥2 enemies in range-1 blast; threat 2.0/5.0
planner-v3b 13/05 23:42        0.499/0.853 official 0 / 30 official errors   Score essentially flat vs v2 (-0.002), but speed +0.082 from multi-source BFS + blast cache + uvloop. Blended +0.018.
bc-v1       14/05 Workbench    0.689 direct / 0.672 container    149k-param CNN BC of planner-v3b, val_acc 0.8742; container mean within noise of direct eval and planner.
bc-v1       14/05 01:22        0.364/0.856 official 0 / 30 official errors   REGRESSED -0.135 vs planner-v3b. Local→official gap ballooned 0.18 → 0.31. BC overfit to random-opponent local distribution.
```

## Qualifier weighted score estimate

Qualifier weights from the handbook:

```text
AE   40%
NLP  20%
ASR  20%
CV   20%
Noise has no direct qualifier weight observed
```

Using raw task scores only:

```text
0.40 * AE 0.499  = 0.1996
0.20 * NLP 0.301 = 0.0602
0.20 * ASR 0.957 = 0.1914
0.20 * CV 0.000  = 0.0000
--------------------------------
Estimated weighted qualifier score = 0.4512
```

`planner-v3b` keeps AE accuracy flat at `0.499` (vs `planner-v2`'s `0.501`) — within noise — but speed jumped `0.771 → 0.853`. Local→official gap is unchanged at `~0.17` (local mean `0.681` → official `0.499`); heuristic ceiling is in sight. Next swing is a learned policy (see Next priority).

Using the observed ~75% score / 25% speed blend:

```text
AE   contribution = 0.2350   (0.75*0.499 + 0.25*0.853 = 0.5876)
NLP  contribution = 0.0937
ASR  contribution = 0.1860
CV   contribution = 0.0491
--------------------------------
Estimated blended qualifier score = 0.5638
```

## Notes

- All 5 tasks have been submitted successfully at least once.
- **ASR crossed the 0.95 accuracy target**: `ft-lora32-v1` officially scored `0.957 / 0.849` (errors `0 / 400`). That's **+0.019 absolute accuracy** over `vad-off-v1`, achieved by a single 3-epoch LoRA-rank-32 decoder fine-tune of `distil-whisper/distil-large-v3` on the full 4110-clip novice manifest. Speed dipped by `0.010` (CT2 file size noise; recoverable via int8_float16 re-export, see Next priority). Generalization gap turned out **negative**: held-out val WER `0.04662` → official WER `~0.043`, i.e. the official 400-clip distribution is slightly easier than the local held-out slice — a useful piece of leaderboard intuition for future runs.
- **`vad-off-v1` (the prior peak)**: inference-only fixes (`vad_filter=False`, hallucination guards, spoken-form ordinals, coordinate-safe decimal regex, tighter silence guard) took accuracy from `norm-v1`'s `0.877` to `0.938` with speed barely changed (-0.005). Local-official WER gap on that run was +0.007 absolute. Those fixes stayed in `ft-lora32-v1` and compounded with the LoRA gains.
- The local `1 - MER` number is a **scoring artifact** of the local manifest being English-only (three other language buckets contribute 0 to the divide-by-4 mean). Track the bare `english error rate (WER)` line instead.
- `vad-off-v2` REGRESSED to local WER 0.0604 after reversing the slang-prompt order (intent was to survive Whisper's truncation but reversing over-primed the decoder). Reverted; not submitted.
- Noise scored `1.000`, but appears required/useful rather than directly weighted for qualifiers.
- CV submitted but had `4 / 500` errors and score `0.000`; fix robustness/schema edge cases before improving model quality.
- AE `planner-v1` officially scored `0.445 / 0.788` with `0 / 30` errors — a new team high score and a large jump from `0.051`, but much worse than Workbench local `0.697`, so the planner likely overfit/easy-rolled local scenarios or hidden eval punished tactical choices differently.
- **AE `planner-v2` lifts official to `0.501 / 0.771`** (`0 / 30` errors, 13/05 23:03 SGT). The single highest-value change was correcting `BOMB_TIMER = 4 → 3` to match the env config (placement→detonation budget is 3 movements, not 4); planner-v1 was almost certainly self-trapping in tactical bomb scenarios. Speed dipped slightly (`0.788 → 0.771`) from the bounded-escape and threat-aware BFS — well worth the +0.056 accuracy. The local→official gap narrowed from ~0.25 to ~0.17 but isn't closed; the remaining gap is likely (a) novice local map being fixed-seed and easier than hidden eval, (b) random opponents locally vs whatever the hidden eval uses, and (c) 6-round local variance.
- **AE `planner-v3` (NOT submitted)** — bundled predictive-bombing at range 2, bomb-chain wall-break heuristic, soft threat penalty (1.0/3.0). Locally regressed to mean `0.596` (vs v2's `0.669`) — in 6-team random-opponent local, aggressive bombing burns the team's bomb budget on speculation. Reverted before submission.
- **AE `planner-v3b` (submitted)**: `0.499 / 0.853` officially (`0 / 30` errors, 13/05 23:42 SGT). v3 minus the bomb-chain trigger, with predictive bombing tightened to `range=1` + `≥2 enemies in extended blast`, and threat penalty restored to 2.0/5.0. **Score essentially flat vs v2 (-0.002)** — heuristic accuracy ceiling is in sight. **Speed jumped +0.082** from the algorithmic wins kept from v3: single multi-source BFS in `_choose_target` (replaces N per-target BFS calls), `_blast_cells` per-turn cache, and the Dockerfile pinning `uvloop`+`httptools`. Net blended score lifted `0.5561 → 0.5638` (+0.008). The 6-run local mean was `0.681` and the local→official gap held at `~0.18` — the same gap as v2, which suggests it's structural (env distribution mismatch), not a tunable knob.

## Next priority

1. **AE PPO with mixed opponents** — bc-v1 shipped, scored `0.364/0.856` and **regressed -0.135 vs planner-v3b**. Local→official gap blew up from 0.18 (heuristic) to 0.31 (BC), confirming the gap is **environment-distribution mismatch**, not heuristic-specific: the BC policy memorized planner behavior against random opponents, which doesn't generalize to whatever the hidden eval uses. **Planner-v3b stays shipped** (`0.499` > `0.364`). Way forward: `train_ppo.py` initialized from `bc.pt`, trained against a **mixed opponent pool** (random + frozen planner-v3b + frozen self-copies + scripted aggressors) so the policy can't overfit to one distribution. Deployment plumbing already validated end-to-end through bc-v1. If PPO still loses to v3b official, fall back to v3b and pursue per-game logging.
2. **ASR re-export `int8_float16`** — same LoRA-merged checkpoint, no retrain. Target: speed `0.849 → 0.90+` with accuracy delta `≤ 0.005`. Tag `ft-lora32-int8f16`.
3. **NLP retrieval / chunking** upgrade — currently 0.301.
4. **CV** — fix the `4 / 500` errors first, then drop in a pretrained detector.
5. **ASR rank-64 escalation** is deferred — `ft-lora32-v1` already crosses 0.95, so the marginal lift isn't worth the extra training time unless int8 quantization unexpectedly regresses accuracy.
