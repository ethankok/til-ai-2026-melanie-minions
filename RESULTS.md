# TIL-AI 2026 Submission Results

Team: `melanie-minions`
Last updated: 14 May 2026 17:20 SGT

## Latest submitted scores

```text
Task   Image                    Tag         Submitted             Errors        Score   Speed
NLP    melanie-minions-nlp      v4-dict-id  14/05/2026 13:29:56   0 / 700       0.483   0.888  ← NEW HIGH; new-eval recovery from 0.000
ASR    melanie-minions-asr      ft-lora32-v1 13/05/2026 11:22:30  0 / 400       0.957   0.849
CV     melanie-minions-cv       cv-yolo-v2-tier1-best 14/05/2026 17:10:09 0 / 500 0.556 0.956  ← NEW HIGH; +0.007 vs v2-best (TTA + imgsz=896 + score field)
Noise  melanie-minions-noise    latest      12/05/2026 03:54:55   0 / 500       1.000   0.970
AE     melanie-minions-ae       ppo-v1      14/05/2026 04:36:51   0 / 30        0.507   0.861  ← team BEST stands; rollback recommended
AE (v2)  melanie-minions-ae     ppo-v2      14/05/2026 13:29:54   0 / 30        0.489   0.854  ← regressed -0.018; local→official gap widened 0.19 → 0.27
AE (v3b) melanie-minions-ae     planner-v3b 13/05/2026 23:42:57   0 / 30        0.499   0.853
AE (bc)  melanie-minions-ae     bc-v1       14/05/2026 01:22:13   0 / 30        0.364   0.856
```

## NLP submission history

```text
Tag           Submitted          Score   Speed   Errors    Local        Notes
latest        12/05 03:23        0.301   0.971   0 / 700   —            OLD EVAL; pre-wipe; lexical token-overlap baseline (no longer on leaderboard)
v2-hybrid-rag 14/05 ~04:00       0.000   ~       0 / 700   —            New eval. Hybrid BM25+BGE+rerank+RoBERTa-SQuAD2 + positional DOC-{i+1:04d}. 0.0 because cloud was buggy: organisers' eval server was sending plain strings while the spec called for dicts (see v4-dict-id)
v3-id-parse   14/05 05:33        0.000   0.888   0 / 700   0.678 (1)    Same pipeline + defensive parser (DOC-XXXX prefix / dict / positional). Cloud 0.000 caused by Ryan's eval bug (still sending plain strings); local 0.678 with prefix patch proved the model itself was sound
v4-dict-id    14/05 13:29        0.483   0.888   0 / 700   0.678        After Ryan FIXED the eval to send {"id":"DOC-XXXX","document":"..."}. Same image as v3-id-parse (just re-tagged); defensive parser's dict-shape branch caught the format immediately. NEW HIGH on the post-wipe leaderboard
```

(1) Local was patched to prepend `DOC-XXXX\n` to each plain string for local verification before Ryan confirmed the cloud format. Same image produced the same local 0.678 once the upstream test was updated to send dicts — proving the pipeline was correct all along; the 0.000 was purely Ryan's eval-server bug.

## ASR submission history

```text
Tag           Submitted          Score   Speed   Local Eng-WER   Notes
v1            12/05 03:42        0.000   0.993   —               Empty-string baseline
norm-v1       12/05 16:23        0.877   0.864   0.0759          + digit verbalization + silence guard
vad-off-v1    12/05 20:00        0.938   0.859   0.0554          + VAD off + hallucination guards + ordinals + decimal-safe
ft-lora32-v1  13/05 11:22        0.957   0.849   0.0299*         + LoRA rank-32 decoder fine-tune (3 epochs, lr 1e-4)
```

## CV submission history

```text
Tag                    Submitted          Score   Speed   Errors    Local mAP50-95         Notes
latest                 12/05 03:52        0.000   0.981   4 / 500   —                      Empty-detection baseline, 4 inputs erroring
yolo-til-map-v2        14/05 01:56        0.044   0.961   0 / 500   —                      YOLOv8n + sparse COCO→TIL map; clean serving, weak domain fit
cv-yolo-ft-v1          14/05 03:53        0.402   0.963   0 / 500   0.885                  YOLOv8s fine-tuned on official 18-class annotations
cv-yolo-v2-best        14/05 14:00        0.549   0.960   0 / 500   0.884 / 0.859          YOLOv8s 768 hard-split retrain + tuned inference
cv-yolo-v2-tier1-best  14/05 17:10        0.556   0.956   0 / 500   0.851 / 0.905          NEW HIGH; same v2-best weights with TTA + imgsz=896 + iou=0.60 + emit score field; hard held-out lifted +0.046 (0.859→0.905), full-local dipped -0.034 (0.884→0.851)
```

Notes on local mAP columns:
- `cv-yolo-v2-best`: full local `til test` `0.8839`; hard held-out HTTP eval
  `0.8589` (`CV_CONF=0.25 CV_IOU=0.50 CV_IMGSZ=768`).
- `cv-yolo-v2-tier1-best`: full local `til test` `0.8505`; hard held-out HTTP
  eval `0.9049` (`CV_CONF=0.20 CV_IOU=0.60 CV_IMGSZ=896 CV_AUGMENT=1
  CV_HALF=1`). The hard split is heavily weighted toward small/dense/rare
  scenes; TTA at imgsz=896 helps those but hurts easy full-local images. Hidden
  eval correlated with the hard held-out (`0.9049` → `0.556`), confirming the
  selection metric.
- The hard held-out split was intentionally harder and non-leaky
  (`4000/500/500` train/val/test; test had `3334` boxes).

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
ppo-v1      14/05 Workbench    0.711/0.700/0.693 eval_policy + 0.766/0.634/0.708 container.  Tight variance vs prior runs; mixed-opponent PPO from bc.pt warm start, best @update 75 of 200 before idle shutdown.
ppo-v1      14/05 04:36        0.507/0.861 official 0 / 30 official errors   NEW HIGH (+0.008 score, +0.008 speed vs v3b). Gap stayed at 0.19 — mixed-opponent training did NOT close the local→official gap.
ppo-v2      14/05 eval_policy  0.7138/0.7752/0.7885 novice + 0.6353/0.6910/0.6768 varied   PPO from BC warm start with frame-stacking (N=4), reward-scale 50, value-loss clip, vary-maps, 162/200 updates before idle shutdown. Novice mean 0.759, varied mean 0.668.
ppo-v2      14/05 til test     0.7282/0.8260/0.7352   Container mean ≈ 0.763 (highest single run 0.826 — best AE local ever).
ppo-v2      14/05 13:29        0.489/0.854 official 0 / 30 official errors   REGRESSED -0.018 vs ppo-v1. Local 0.763 → official 0.489: gap WIDENED 0.19 → 0.27. Frame-stacking + varied-map training did NOT generalize; the policy overfit the varied-maps distribution.
```

## Qualifier weighted score estimate

Official Challenge-spec weights:

```text
AE   40%
NLP  20%
ASR  20%
CV   20%
Noise has no direct Qualifier reward; it is a Finals CV-disruption component
```

Each scored challenge blends `75%` accuracy/reward and `25%` speed. Qualifier speed is `1 - min(t_elapsed, 30 minutes) / 30 minutes` over the full test set.

Using raw task scores only (best-ever shipped per task on the CURRENT
leaderboard — old NLP score of 0.301 was wiped when organisers rolled out
the new eval; `v4-dict-id` 0.483 is our recovery):

```text
0.40 * AE 0.507  = 0.2028
0.20 * NLP 0.483 = 0.0966   ← NEW HIGH; recovered from 0.000 after Ryan fixed eval server bug
0.20 * ASR 0.957 = 0.1914
0.20 * CV 0.549  = 0.1098   ← NEW HIGH; YOLOv8s 768 hard-split retrain
--------------------------------
Estimated weighted qualifier score = 0.6006
```

`planner-v3b` keeps AE accuracy flat at `0.499` (vs `planner-v2`'s `0.501`) — within noise — but speed jumped `0.771 → 0.853`. Local→official gap is unchanged at `~0.17` (local mean `0.681` → official `0.499`); heuristic ceiling is in sight. Next swing is a learned policy (see Next priority).

Using the observed ~75% score / 25% speed blend:

```text
AE   contribution = 0.2381   (0.75*0.507 + 0.25*0.861 = 0.5953)
NLP  contribution = 0.1168   (0.75*0.483 + 0.25*0.888 = 0.5843)  ← +0.0724 vs the 0.000 / speed-only state
ASR  contribution = 0.1860
CV   contribution = 0.1304   (0.75*0.549 + 0.25*0.960 = 0.6518)
--------------------------------
Estimated blended qualifier score = 0.6714
```

## Notes

- All 5 tasks have been submitted successfully at least once.
- **NLP recovered to `0.483` on `v4-dict-id`** (14/05 13:29 SGT). The new eval was rolled out 14 May with a strict top-3-docs schema and 0.9 equivalence threshold; the old `0.301` was wiped. Two prior new-eval attempts (`v2-hybrid-rag`, `v3-id-parse`) both scored `0.000` — root cause was Ryan's eval server bug (sending plain strings instead of `{"id": "DOC-XXXX", "document": "..."}` dicts). Our manager's defensive parser was already correct; resubmitting the *same image* under `v4-dict-id` after Ryan's fix landed `0.483/0.888`. Local→cloud gap of `~0.20` (local 0.678) is consistent with AE/CV gaps on this competition. Pipeline: 3-sentence sliding chunks + BM25 ⊕ BGE-small dense ⊕ BGE-reranker-base + RoBERTa-SQuAD2 extractive QA, all GPU fp16.
- **ASR crossed the 0.95 accuracy target**: `ft-lora32-v1` officially scored `0.957 / 0.849` (errors `0 / 400`). That's **+0.019 absolute accuracy** over `vad-off-v1`, achieved by a single 3-epoch LoRA-rank-32 decoder fine-tune of `distil-whisper/distil-large-v3` on the full 4110-clip novice manifest. Speed dipped by `0.010` (CT2 file size noise; recoverable via int8_float16 re-export, see Next priority). Generalization gap turned out **negative**: held-out val WER `0.04662` → official WER `~0.043`, i.e. the official 400-clip distribution is slightly easier than the local held-out slice — a useful piece of leaderboard intuition for future runs.
- **`vad-off-v1` (the prior peak)**: inference-only fixes (`vad_filter=False`, hallucination guards, spoken-form ordinals, coordinate-safe decimal regex, tighter silence guard) took accuracy from `norm-v1`'s `0.877` to `0.938` with speed barely changed (-0.005). Local-official WER gap on that run was +0.007 absolute. Those fixes stayed in `ft-lora32-v1` and compounded with the LoRA gains.
- The local `1 - MER` number is a **scoring artifact** of the local manifest being English-only (three other language buckets contribute 0 to the divide-by-4 mean). Track the bare `english error rate (WER)` line instead.
- `vad-off-v2` REGRESSED to local WER 0.0604 after reversing the slang-prompt order (intent was to survive Whisper's truncation but reversing over-primed the decoder). Reverted; not submitted.
- Noise scored `1.000`, but the official Challenge spec says Noise is not directly rewarded in Qualifiers; it matters in Finals by degrading opponents' CV inputs within SSIM/RMSE limits.
- **CV `cv-yolo-v2-best` is the new high score**: official `0.549 / 0.960` with `0 / 500` errors (14/05 14:00 SGT), up from `cv-yolo-ft-v1`'s `0.402 / 0.963`. The v2 path fixed the local-eval blind spot by creating a hard held-out split (`4000/500/500`, test `3334` boxes), training YOLOv8s at `imgsz=768` for 80 epochs, and selecting inference defaults through HTTP-container sweeps. Best held-out full eval was `0.8589` mAP50-95 with small-object AP `0.5596`; full local `til test` scored `0.8839`. The official hidden gap remains large (`0.8589 → 0.549`) but the direction was right and worth +0.147 official accuracy at almost unchanged speed.
- AE `planner-v1` officially scored `0.445 / 0.788` with `0 / 30` errors — a new team high score and a large jump from `0.051`, but much worse than Workbench local `0.697`, so the planner likely overfit/easy-rolled local scenarios or hidden eval punished tactical choices differently.
- **AE `planner-v2` lifts official to `0.501 / 0.771`** (`0 / 30` errors, 13/05 23:03 SGT). The single highest-value change was correcting `BOMB_TIMER = 4 → 3` to match the env config (placement→detonation budget is 3 movements, not 4); planner-v1 was almost certainly self-trapping in tactical bomb scenarios. Speed dipped slightly (`0.788 → 0.771`) from the bounded-escape and threat-aware BFS — well worth the +0.056 accuracy. The local→official gap narrowed from ~0.25 to ~0.17 but isn't closed; the remaining gap is likely (a) novice local map being fixed-seed and easier than hidden eval, (b) random opponents locally vs whatever the hidden eval uses, and (c) 6-round local variance.
- **AE `planner-v3` (NOT submitted)** — bundled predictive-bombing at range 2, bomb-chain wall-break heuristic, soft threat penalty (1.0/3.0). Locally regressed to mean `0.596` (vs v2's `0.669`) — in 6-team random-opponent local, aggressive bombing burns the team's bomb budget on speculation. Reverted before submission.
- **AE `planner-v3b` (submitted)**: `0.499 / 0.853` officially (`0 / 30` errors, 13/05 23:42 SGT). v3 minus the bomb-chain trigger, with predictive bombing tightened to `range=1` + `≥2 enemies in extended blast`, and threat penalty restored to 2.0/5.0. **Score essentially flat vs v2 (-0.002)** — heuristic accuracy ceiling is in sight. **Speed jumped +0.082** from the algorithmic wins kept from v3: single multi-source BFS in `_choose_target` (replaces N per-target BFS calls), `_blast_cells` per-turn cache, and the Dockerfile pinning `uvloop`+`httptools`. Net blended score lifted `0.5561 → 0.5638` (+0.008). The 6-run local mean was `0.681` and the local→official gap held at `~0.18` — the same gap as v2, which suggests it's structural (env distribution mismatch), not a tunable knob.

## Next priority

1. **NLP quality push (now unblocked)** — `v4-dict-id` recovered to `0.483 / 0.888` on the cloud, local 0.678 → cloud 0.483 gap of `~0.20`. Pipeline is hybrid BM25+BGE-small + BGE-reranker-base + RoBERTa-SQuAD2; same proportional gap we see on AE/CV, so it's likely distribution shift on the held-out corpus rather than a bug. Levers in increasing cost/impact: (a) **batch the SQuAD2 forward pass** (currently sequential over top-10 chunks in `_extract_answer`; `QA_BATCH=16` is defined but unused — should free 50–80% of QA latency), (b) **paragraph-aware chunking** (split on `\n\n` before sentence-windowing, preserves longer answer spans), (c) **bigger embedder/reranker** (`bge-base-en-v1.5`, `bge-reranker-large` — costs latency budget, possibly worth trading against the saved time from QA batching), (d) **sentence-of-best-chunk fallback** for when SQuAD2 confidence is low. Estimated reachable: 0.55–0.65 cloud.
2. **AE PPO regression analysis** — `ppo-v2` (`0.489/0.854`) regressed -0.018 vs `ppo-v1` (`0.507/0.861`), with the local→official gap widening from `~0.19` to `~0.27`. Frame-stacking + varied-map training did not generalize. `ppo-v1` stays shipped. Path forward: per-game logging on official eval to characterise the gap (which scenarios punish the policy most), then targeted curriculum. Alternative: dial back to a leaner PPO variant without frame-stacking.
3. **CV next A/B is optional** — `cv-yolo-v2-best` is clean and materially better (`0.549/0.960`, `0 / 500` errors). Remaining gap is hidden distribution/small-object/aircraft-subclass generalization, not schema. Next CV swings if time allows: `yolov8m` at `imgsz=768`, class-balanced or aircraft-heavy sampling, and stronger small-object augmentation. Keep the non-leaky hard held-out eval and HTTP sweep loop; do not return to full-set `til test` as the selection metric.
4. **ASR beam/prompt tweaks only if idle** — `ft-lora32-v1` already crosses 0.95 and `int8_float16` regressed to `0.923/0.856`, so speed quantization is off the table for this checkpoint.
