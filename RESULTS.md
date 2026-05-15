# TIL-AI 2026 Submission Results

Team: `melanie-minions`
Last updated: 15 May 2026 21:20 SGT — NLP `v9-doc-ensemble` remains best submitted blend at 0.683/0.883; `v11-canonical-answer` is staged for Workbench test.

## Latest submitted scores

```text
Task   Image                    Tag         Submitted             Errors        Score   Speed
NLP    melanie-minions-nlp      v9-doc-ensemble 15/05/2026 19:46:01 0 / 700 0.683   0.883  ← BEST BLEND; same image resubmit, speed variance helped
NLP (v10 neutral) melanie-minions-nlp v10-template-lite 15/05/2026 20:16:17 0 / 700 0.683 0.882
NLP (v9 first) melanie-minions-nlp v9-doc-ensemble 15/05/2026 19:25:50 0 / 700 0.683 0.868
NLP (prior v8b) melanie-minions-nlp v8b-chunked-context 15/05/2026 18:35:19 0 / 700 0.679 0.872
NLP (prior v7) melanie-minions-nlp v7-finetuned-v1 15/05/2026 11:39:09 0 / 700 0.517 0.880
NLP (prior v5c) melanie-minions-nlp v5c-no-para 14/05/2026 19:44:08 0 / 700     0.483   0.912
ASR    melanie-minions-asr      nemo-zs     14/05/2026 20:33:36   0 / 400       0.956   0.946  ← NEW BLENDED HIGH; Parakeet-TDT-0.6B-v2 zero-shot, +0.097 speed vs ft-lora32-v1
CV     melanie-minions-cv       cv-yolo-v2-tier1-best 14/05/2026 17:10:09 0 / 500 0.556 0.956
CV (v3-pre) melanie-minions-cv  cv-yolo11m-v3-pre 15/05/2026 11:34:42 0 / 500   0.376   0.955  ← REGRESSED -0.180; v11m fully trained but matched-imgsz lost to v8s+upscaled. Tier1 stays on leaderboard.
Noise  melanie-minions-noise    latest      12/05/2026 03:54:55   0 / 500       1.000   0.970
AE     melanie-minions-ae       hybrid-v3   14/05/2026 19:26:06   0 / 30        0.555   0.849
ASR (ft-lora32-v1) melanie-minions-asr ft-lora32-v1 13/05/2026 11:22:30 0 / 400  0.957   0.849  ← prior ASR high (still on leaderboard via highest-score retention)
AE (hybrid-v2) melanie-minions-ae hybrid-v2 14/05/2026 14:55:23   0 / 30        0.545   0.863
AE (heur-restore-v2) melanie-minions-ae heuristic-restore-v2 14/05/2026 15:02:13 0 / 30   0.502   0.854
AE (policy-fast-v2)  melanie-minions-ae policy-fast-v2       14/05/2026 14:42:05 0 / 30   0.425   0.859
AE (ppo-v1) melanie-minions-ae   ppo-v1      14/05/2026 04:36:51   0 / 30        0.507   0.861
AE (ppo-v2) melanie-minions-ae   ppo-v2      14/05/2026 13:29:54   0 / 30        0.489   0.854
AE (v3b)  melanie-minions-ae    planner-v3b 13/05/2026 23:42:57   0 / 30        0.499   0.853
```

## NLP submission history

```text
Tag           Submitted          Score   Speed   Errors    Local        Notes
latest        12/05 03:23        0.301   0.971   0 / 700   —            OLD EVAL; pre-wipe; lexical token-overlap baseline (no longer on leaderboard)
v2-hybrid-rag 14/05 ~04:00       0.000   ~       0 / 700   —            New eval. Hybrid BM25+BGE+rerank+RoBERTa-SQuAD2 + positional DOC-{i+1:04d}. 0.0 because cloud was buggy: organisers' eval server was sending plain strings while the spec called for dicts (see v4-dict-id)
v3-id-parse   14/05 05:33        0.000   0.888   0 / 700   0.678 (1)    Same pipeline + defensive parser (DOC-XXXX prefix / dict / positional). Cloud 0.000 caused by Ryan's eval bug (still sending plain strings); local 0.678 with prefix patch proved the model itself was sound
v4-dict-id    14/05 13:29        0.483   0.888   0 / 700   0.678        After Ryan FIXED the eval to send {"id":"DOC-XXXX","document":"..."}. Same image as v3-id-parse (just re-tagged); defensive parser's dict-shape branch caught the format immediately
v5-multi      (not shipped)      —       —       —         0.628        Para-aware chunking + batched SQuAD2 + BM25 doc backfill + low-conf sentence fallback. Local REGRESSED -0.050 vs v4; error-bucket report showed fallback was firing on every single-word answer ("Velez", "1992", ...) and replacing correct-but-short SQuAD2 spans. NOT submitted
v5b-no-fallback 14/05 19:10      0.456   0.916   0 / 700   0.674        Dropped fallback; kept the other three. Local OK (within 0.005 of v4) but cloud REGRESSED -0.027 vs v4. Local→cloud gap widened from 0.195 → 0.218 — clear signal that one of the remaining changes hurt on the held-out corpus
v5c-no-para   14/05 19:44        0.483   0.912   0 / 700   0.678        Reverted paragraph chunking; kept batched SQuAD2 + BM25 backfill. Cloud RECOVERED to v4's 0.483 with v5b's speed gain (+0.024) retained → blended 0.590 (vs v4 0.584). Confirmed paragraph chunking was the v5b regressor
v7-finetuned-v1 15/05 11:39      0.517   0.880   0 / 700   0.709        NEW HIGH (+0.034 cloud vs v5c). Fine-tuned roberta-large-squad2 on local nlp.jsonl (353/883 examples retained via exact + case-insensitive matching; trained 3 epochs, load_best_model_at_end picked epoch 1 with eval_loss 0.614). Local-cloud gap held at 0.19 — fine-tune transferred 1:1. Speed dipped -0.032 from roberta-large's 2.5x latency. Blended 0.608 (vs v5c 0.590). Error-bucket shift: L1 exact-match 29.7% → 39.7%, retrieval_hit_exact 183 → 242
v7-finetuned-v2 (not shipped)    —       —       —         0.698        v2 data-prep: variants + flexible regex + rapidfuzz fuzzy fallback. Retained 431/883 (+78 vs v1) but local regressed from v1; fuzzy-matched spans were noisy
v8b-chunked-context 15/05 18:35  0.679   0.872   0 / 700   0.708        NEW HIGH (+0.162 cloud vs v7-v1). --use-answer-chunk + rapidfuzz off, 353/883 retained; span realignment bug fixed in 627c9ce before retrain. Local aggregate looked flat (0.708 vs 0.709), but cloud strongly rewarded chunked-context training. Blended 0.727
v9-doc-ensemble 15/05 19:25      0.683   0.868   0 / 700   0.711        NEW HIGH (+0.004 cloud vs v8b). Whole-doc BM25+BGE prior/seeding reduced local retrieval misses 40→37 and nudged cloud accuracy. Small speed cost (-0.004); blended 0.729
v9-doc-ensemble 15/05 19:46      0.683   0.883   0 / 700   0.711        Same image resubmitted. Accuracy unchanged, speed +0.015; best NLP blend ~0.733. Do not over-interpret speed deltas at this scale.
v10-template-lite 15/05 20:16     0.683   0.882   0 / 700   0.711        NEUTRAL. Narrow deterministic answer layer for elapsed days/years and percentage-point deltas. Local substr +1 / diff -1, retrieval unchanged; cloud accuracy unchanged.
v11-canonical-answer (candidate) —       —       —         replay +10    Staged. Full-document canonicalizer over top returned docs: codenames, penalties, event-year differences, unit/name aliases. Replay exact/substr 451→461, diff 395→385 on v9 predictions.
```

(1) Local was patched to prepend `DOC-XXXX\n` to each plain string for local verification before Ryan confirmed the cloud format. Same image produced the same local 0.678 once the upstream test was updated to send dicts — proving the pipeline was correct all along; the 0.000 was purely Ryan's eval-server bug.

## ASR submission history

```text
Tag           Submitted          Score   Speed   Local Eng-WER   Notes
v1            12/05 03:42        0.000   0.993   —               Empty-string baseline
norm-v1       12/05 16:23        0.877   0.864   0.0759          + digit verbalization + silence guard
vad-off-v1    12/05 20:00        0.938   0.859   0.0554          + VAD off + hallucination guards + ordinals + decimal-safe
ft-lora32-v1  13/05 11:22        0.957   0.849   0.0299*         + LoRA rank-32 decoder fine-tune (3 epochs, lr 1e-4)
nemo-zs       14/05 20:33        0.956   0.946   0.0429          BACKBONE SWITCH: Parakeet-TDT-0.6B-v2 zero-shot. Accuracy flat (-0.001), speed +0.097, blended +0.025
nemo-zs-v2    14/05 22:07        0.956   0.946   0.0429          + cuda-python CUDA-graph fast path. Local wall clock -7% (37:28→34:42), cloud unchanged. Cloud speed bottleneck is now HTTP / audio I/O / Python overhead, NOT the TDT decoder. Speed parked at 0.946; next lever is accuracy (Parakeet FT).
```

## CV submission history

```text
Tag                    Submitted          Score   Speed   Errors    Local mAP50-95         Notes
latest                 12/05 03:52        0.000   0.981   4 / 500   —                      Empty-detection baseline, 4 inputs erroring
yolo-til-map-v2        14/05 01:56        0.044   0.961   0 / 500   —                      YOLOv8n + sparse COCO→TIL map; clean serving, weak domain fit
cv-yolo-ft-v1          14/05 03:53        0.402   0.963   0 / 500   0.885                  YOLOv8s fine-tuned on official 18-class annotations
cv-yolo-v2-best        14/05 14:00        0.549   0.960   0 / 500   0.884 / 0.859          YOLOv8s 768 hard-split retrain + tuned inference
cv-yolo-v2-tier1-best  14/05 17:10        0.556   0.956   0 / 500   0.851 / 0.905          NEW HIGH (still on leaderboard); v2-best weights + TTA + imgsz=896 + iou=0.60 + score field
cv-yolo11m-v3-pre      15/05 11:34        0.376   0.955   0 / 500   0.937 / 0.867          REGRESSED -0.180; YOLOv11m@1024 fully trained 120ep. Local val 0.937, hard held-out 0.867 (-0.038 vs tier1) — bigger model + matched-imgsz lost to v8s + upscaled inference. Tier1 stays on leaderboard.
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
- `cv-yolo11m-v3-pre`: Ultralytics val mAP50-95 `0.937` on the 500-image val
  split; hard held-out HTTP eval top sweep row `0.8673` at `CV_CONF=0.001
  CV_IOU=0.70 CV_IMGSZ=1024 CV_AUGMENT=1`. Hard small AP `0.587` (vs tier1's
  `0.746`) — the regression is driven by small-object bbox precision in the
  high-IoU bins, almost certainly a resolution-mismatch story (trained at
  1024, never tested at 1280 inference). Hidden eval correlated tightly with
  hard held-out (`0.8673` → `0.376`), so the gap diagnosis is confirmed.
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
[round-1 AE_MODE bake bug] All three of policy-fast-v1 / hybrid-v1 / heuristic-restore shipped the SAME image sha256:2572b392… because AE_MODE was only set in the shell, not baked into the Docker image. Each new tag overwrote the previous in the eval queue (we only got one score back per round). Fixed by ENV AE_MODE= line in Dockerfile + .ae_mode fallback file. Resubmitted as v2 below.
policy-fast-v2 14/05 14:42       0.425/0.859 official 0 / 30 official errors   ppo-v1 weights + speed fixes (single-thread torch, inference_mode, warmup, preallocated tensors). Local mean 0.654 (1 run). Score regressed -0.082 from ppo-v1 — almost certainly cloud variance on 30-game sample (we've seen ±0.04 between identical runs). Speed flat at 0.859.
hybrid-v2    14/05 14:55         0.545/0.863 official 0 / 30 official errors   *** NEW HIGH ***. Hybrid manager: policy chooses, heuristic safety-veto on illegal / no-escape-bomb / step-into-blast / frozen-stay. First structurally new approach since ppo-v1. Local 0.774 (1 run, 6 rounds). Local→official gap 0.23 — same band as everything else, but the *floor* lifted by 0.038. Speed 0.863 (+0.002 vs ppo-v1).
heuristic-restore-v2 14/05 15:02 0.502/0.854 official 0 / 30 official errors   Pure heuristic (planner-v3b + TILE_RESPAWN 40→20 + enemy_agent eviction). Local 0.787 (1 run). +0.003 vs planner-v3b 0.499 — confirms heuristic-only ceiling and that the 40→20 / eviction tweaks were noise on cloud. Speed flat at 0.854.
hybrid-v3    14/05 19:26         0.555/0.849 official 0 / 30 official errors   *** NEW HIGH (STILL SHIPPED) ***. Hybrid + top-K policy cascade (try policy's #2/#3 actions when #1 is vetoed before falling back to heuristic) + opportunistic enemy-kill in heuristic dominant-action shortcut. +0.010 score vs hybrid-v2 (within ±0.04 cloud noise but trending right); -0.014 speed (likely more bomb-escape work or noise). Now top-quartile on the leaderboard (top is 0.711).
bc-belief        15/05 train         BC val_acc 0.897 (vs bc-v1's 0.874, +0.023); local 6-game 0.656 (vs bc-v1's 0.689). 704k params, 16x16x11 belief-map CNN branch. CPU torch (3 min / 20 epochs / 40k samples). Architecture's prerequisite signal positive (better fit to planner-v3b actions); cloud was the real test.
bc-belief-hybrid 15/05 11:46         0.287/0.846 official 0 / 30 official errors   *** MEMORY HYPOTHESIS REJECTED ***. Regressed -0.268 vs hybrid-v3 (huge — far outside ±0.04 noise). Local 6-game 0.646 → cloud 0.287 = gap 0.36 (wider than bc-v1's 0.33). Best read: the 704k-param model with belief input has *more ways to overfit* to planner-v3b's random-opponent local behavior; belief tensor encodes spurious local-distribution correlations that don't transfer. Local hybrid ≈ bc-belief solo (0.646 vs 0.656) showed vetoes were firing so often that hybrid wrapper added nothing, so the cloud regression is the policy's own. PPO would lift this maybe +0.10-0.14 (bc→ppo scale from prior runs) but still below hybrid-v3 0.555. Memory-augmented BC is a dead end for this opponent distribution.
bc-belief-policy 15/05 local-only    local 0.663 (1 run, 6 games). Tested in pure policy mode for comparison; ~0.02 above hybrid-wrapped locally. NOT submitted — bc-belief-hybrid's cloud regression made pure-policy unlikely to be better.
hybrid-conf50    15/05 17:59         0.504/0.857 official 0 / 30 official errors   Regressed -0.051 vs hybrid-v3 (just outside ±0.04 noise; small but real). ppo-v1 weights + hybrid + `AE_HYBRID_CONF=0.5` (only use policy when softmax top ≥ 0.5; otherwise heuristic). Local 0.719 → cloud 0.504 = gap 0.22 (same as hybrid-v3) — confidence gate did real local work but threw out cloud-correct policy actions in the 0.4-0.5 softmax band. AE PARKED at hybrid-v3 (0.555/0.849). Four post-hybrid-v3 attempts (bc-belief-hybrid -0.268, bc-belief-policy not-shipped, hybrid-conf50 -0.051) → heuristic-side ceiling confirmed at ~0.555.
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
the new eval):

```text
0.40 * AE 0.555  = 0.2220
0.20 * NLP 0.683 = 0.1366   ← v9-doc-ensemble / v10-template-lite tied on accuracy
0.20 * ASR 0.957 = 0.1914   ← raw accuracy held by leaderboard's max policy
0.20 * CV 0.556  = 0.1112
--------------------------------
Estimated weighted qualifier score = 0.6612
```

AE has now incrementally climbed in three consecutive submissions (ppo-v1 0.507 → hybrid-v2 0.545 → hybrid-v3 0.555), each one moving the floor up by the size of cloud noise but in the same direction. The structural local→cloud gap (~0.23) is intact, but the floor itself has moved +0.048. Top of leaderboard is 0.711; we're now top-quartile.

The belief-map state-augmentation attempt (`bc-belief-hybrid`, 15/05) regressed to 0.287 — far outside cloud noise and a clean rejection of the memory hypothesis as implemented through BC. The bigger network (704k params) with belief input found *more* spurious correlations to the local random-opponent distribution rather than fewer. AE rolled back to `hybrid-v3`. Remaining AE moves (veto-tuning A/Bs, MCTS-light) have realistic ceiling ~0.58; the higher-EV qualifier lift has now come from NLP `v8b/v9`.

ASR `nemo-zs` (Parakeet-TDT-0.6B-v2 zero-shot) just shipped at `0.956/0.946` — a -0.001 accuracy nudge but +0.097 speed. The leaderboard keeps the higher score for raw accuracy, but the BLENDED score per challenge is what feeds the qualifier total via the 75/25 weighting below.

Using the observed ~75% score / 25% speed blend:

```text
AE   contribution = 0.2515   (0.40 * (0.75*0.555 + 0.25*0.849) = 0.40 * 0.6285 = 0.2514)
NLP  contribution = 0.1466   (0.75*0.683 + 0.25*0.883 = 0.7330)  ← best blend from same-image v9 resubmit
ASR  contribution = 0.1907   (0.75*0.956 + 0.25*0.946 = 0.9535)
CV   contribution = 0.1312   (0.75*0.556 + 0.25*0.956 = 0.6560)
--------------------------------
Estimated blended qualifier score = 0.7199  (+0.0007 from v9 resubmit speed noise vs first v9 run)
```

## Notes

- All 5 tasks have been submitted successfully at least once.
- **NLP `v9-doc-ensemble` is the best current blend at `0.683/0.883`** after a same-image resubmit (first v9 was `0.683/0.868`). It keeps the `v8b` chunked-context RoBERTa answerer and adds whole-document BM25+BGE retrieval as a light prior/reranker seeder. Local retrieval misses dropped `40→37`; exact stayed 273, substr 176→177, diff 394→396. `v10-template-lite` tied accuracy at `0.683/0.882` with only a local substr +1 shift, so the gain is still from v9 retrieval plus speed variance; answer syntax/equivalence remains the blocker.
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

1. **ASR `parakeet-ft-v1` (Parakeet decoder-only fine-tune)** — `nemo-zs-v2` proved cloud speed is parked at 0.946 (HTTP / audio I/O / Python overhead, not the TDT decoder). The remaining ASR lever is accuracy. Pipeline is wired end-to-end: `prepare_data_nemo.py` → `train_parakeet.py` (encoder frozen, lr 5e-5, 5 epochs, ~3-4 hr T4) → `export_parakeet.py`. Decision gate before submission: local Eng-WER ≤ 0.035 (from zero-shot 0.0429). Expected official: 0.965-0.975. Leaderboard keeps the higher blended score so regression cannot demote `nemo-zs`.
2. **NLP next: Workbench-test `v11-canonical-answer`.** `v9` is best submitted at `0.683/0.883` after same-image resubmit; retrieval hit is `95.8%` locally, but the bigger headroom is not retrieval. Local upper bound is `0.958` if every retrieved answer were accepted, while actual local is `0.711`. The downloaded local corpus confirms the problem: 481/883 gold answers are not literal source substrings, including 225 L1 answers. `v10-template-lite` was safe but neutral (`0.683/0.882`, substr +1 only). `v11-canonical-answer` is the broader deterministic pass: replay exact/substr 451→461 and diff 395→385 on v9 predictions, no proxy regressions. The bigger swing remains `v8a-genqa`, but only if outputs stay short/canonical enough for the AE 0.9 threshold and speed remains acceptable.
3. **AE hybrid is the new shipped tag at `0.555 / 0.849`.** The 0.49-0.51 cloud ceiling was real for individual approaches but broke under the policy + heuristic safety-veto combo (+0.048 over ppo-v1). To push toward 0.60 the cheap next moves all reuse the existing `HybridAEManager` plumbing — no retraining needed. Each is one env-var toggle + one rebuild:
   - **`AE_HYBRID_CONF=0.5`**: only use policy when its softmax top-action ≥ 0.5. Below that, fall back to heuristic. Tests whether the policy's *uncertain* outputs are the ones costing us score.
   - **`AE_HYBRID_VETO_BOMBS=0`** *or* **`AE_HYBRID_VETO_DANGER=0`**: turn one veto off at a time to find which one is actually doing work. If hybrid still scores >0.5 without bomb-vetoes, the heuristic's bomb-escape check was wrong and we can simplify.
   - **`AE_HYBRID_VETO_FROZEN_STAY=0`**: cheapest A/B; if STAY was the right call sometimes, we recover that.
   Submit at most 2-3 of these — cloud variance is ±0.04 per run so we want big effect sizes, not micro-tunes. Speed is already evaluator-bound at ~0.86, no point optimizing further. If none beat 0.555, retraining the policy *knowing it has a heuristic safety net* (e.g. PPO with veto-aware rollouts) is the longer path.
4. **CV next A/B is optional** — `cv-yolo-v2-tier1-best` is clean and materially better (`0.556/0.956`, `0 / 500` errors). Remaining gap is hidden distribution/small-object/aircraft-subclass generalization, not schema. Next CV swings if time allows: `yolov8m` at `imgsz=768`, class-balanced or aircraft-heavy sampling, and stronger small-object augmentation.
