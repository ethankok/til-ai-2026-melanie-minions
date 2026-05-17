# TIL-AI 2026 Submission Results

Team: `melanie-minions`
Last updated: 17 May 2026 ~16:15 SGT — **NLP UN-FROZEN. `v14-llm-rag`
shipped at `0.734 / 0.286` — new NLP accuracy high (+0.051 vs v9's 0.683,
above the previous public leaderboard top of 0.711)**. Architecture: kept
v9's BM25+BGE+rerank retrieval, replaced RoBERTa-large extractive head with
Qwen2.5-7B-Instruct-AWQ served by vLLM. Local 0.754 → cloud 0.734, gap 0.020
(consistent with v9's 0.028). The non-extractive +5pp came from the model
class change exactly as predicted — 481/883 local gold answers were
non-literal and v13a oracle said the candidate pool had +0.10 of headroom
extractive heads couldn't surface. **Blended cost: -0.112** (v14 blended
0.622 vs v9 0.734) because cloud wall-clock went 3:50 → ~21 min → speed
score 0.886 → 0.286. v14b-speed (few-shots 6→3, max_new_tokens 48→32) is
the immediate iteration; v14c-3b (swap to Qwen2.5-3B-AWQ) is the bigger
fallback if v14b can't recover blended above v9.

Earlier 17 May state: NLP was FROZEN at `v9-doc-ensemble` (0.683/0.886
official, 0.711 local) after every post-v9 architecture-internal swing
regressed. `v13b-deberta` failed the local gate: 0.667 vs v9's 0.711, with
inference 9:06 vs v9's 3:48. The freeze held until v14-llm-rag changed the
answerer model class entirely.

Earlier 17 May state: **`mcts-light-v2` SHIPPED 23:52 at
0.487/0.595 — REGRESSED.** Blended 0.514 vs hybrid-v3's 0.628 (−0.114).
Speed cap + pre-flight gate prevented the v1 timeout but MCTS still cost
+7.7 min of cloud wall-clock (4.5 → 12.2 min); speed score dropped
0.849 → 0.595. Accuracy ALSO regressed (−0.068 vs hybrid-v3): MCTS is
*replacing* hybrid-v3 actions with cloud-worse choices, not just being
slower. Best read: MCTS commits to simulated combat lines using a
stationary-opponent assumption that cloud opponents don't honor — same
local-cloud distribution-shift failure that killed bc-belief (training)
and ppo-selfplay (training), this time at inference. **Both training-
side AND inference-side hypotheses are now falsified.** Remaining cheap
iterations: (A) conservative-MCTS variant (raise `AE_MCTS_MIN_SCORE`
12 → 22 + shrink `DEPTH=2 WIDTH=16`) so MCTS is a high-confidence
override layer only; (B) heuristic-only MCTS A/B to isolate MCTS
contribution. After A or A+B, AE is exhausted at hybrid-v3. Earlier
today: `mcts-light-v1` TIMED OUT (speed budget math wrong; fixed in
v2 by latency cap + gate + smaller defaults but the resulting v2 still
underperformed). Hybrid-v3 (0.555/0.849) stays leaderboard-shipped
throughout via highest-score retention.
**CV RE-PARKED at tier1 0.556/0.956
after Phase C.1 didn't transfer.** `cv-augc1-v4` shipped 15:28 at 0.553/0.962
(mismatched Dockerfile config) and `cv-augc1-v4-1280` shipped 18:28 at
0.553/0.959 (matched: imgsz=1280 conf=0.001 iou=0.7 aug=0). The augmented
training (JPEG aug q=40-85 + 1024×1024 native-res tile crops + scale=0.80)
lifted hard held-out **+0.04 mAP / +0.03 small AP** (0.948 vs 0.905) — training
worked on its target. But cloud was flat and the local→cloud gap **widened
from 0.349 to 0.395**: the augmentation specialized the model further from
cloud's mixed distribution. v8s/v11m family confirmed at-ceiling near cloud
0.556. Tier1 stays via highest-score retention.
**AE is UN-PARKED for `mcts-light-v1` code-only inference A/B.**
`ppo-selfplay-v2` shipped 18:09 at 0.436/0.857 and REGRESSED -0.119 vs
hybrid-v3 (0.555). It still beat v1 (0.305) by +0.131, confirming BC
warm-start + self-play > self-play from scratch, but the v1 gap-tightening
(local-cloud 0.10) was an artifact: v2's gap is 0.271 (til test 0.7068 →
cloud 0.436), bigger than hybrid-v3's 0.219. Training-side AE hypotheses
are now falsified: league/self-play helped over from-scratch, but does not
clear the hybrid-v3 ceiling; memory via bc-belief also regressed hard.
The current AE push is therefore inference-side: `mcts-light-v1` adds
bounded tactical lookahead in `AEManager` and lets `HybridAEManager` trust
high-value bomb/base/enemy lines before the neural policy. **No MCTS-light
official score yet; hybrid-v3 remains shipped via highest-score retention
at 0.555/0.849.**
**NLP FROZEN at `v9-doc-ensemble`.** Candidate-ranker arch was eliminated by
v13a: AE-trained logistic ranker val top-1 0.384 (not built), heuristic-only
local 0.663 = v12 (not submitted). The remaining QA-head retune (`v13b-deberta`)
also failed: DeBERTa-v3-large trained successfully, but local `til test` landed
0.667 vs v9's 0.711 and slowed to 9:06 vs 3:48. Every post-v9 swing has now
regressed or failed gate (v10 neutral/slightly worse, v11 -0.003, v8a -0.031,
v12 -0.041, v13a local 0.663, v13b local 0.667). No more NLP training.

## Latest submitted scores

```text
Task   Image                    Tag         Submitted             Errors        Score   Speed
NLP    melanie-minions-nlp      v14-llm-rag 17/05/2026 16:15:15 0 / 700 0.734   0.286  ← NEW NLP ACCURACY HIGH (+0.051 vs v9). Qwen2.5-7B-Instruct-AWQ via vLLM as the answerer, v9 retrieval kept. Blended 0.622 vs v9 0.734 (-0.112) — speed regressed from 21-min wall-clock. v14b-speed iterates with fewer few-shots.
NLP (prior best blend) melanie-minions-nlp v9-doc-ensemble 16/05/2026 05:21:57 0 / 700 0.683   0.886  ← BEST BLENDED until v14b/v14c can recover; 3rd v9 resubmit, speed 0.886. Blended ~0.734.
NLP (v12 regressed) melanie-minions-nlp v12-candidate-ranker 16/05/2026 13:48:51 0 / 700 0.642 0.829  ← REGRESSED -0.041 acc, -0.057 speed vs v9. Candidate-ranker promoted doc-mined short tokens that passed exact/substr proxy but failed 0.9 AE threshold; -69 exact +54 substr +15 diff in local buckets. Worst since v5b.
NLP (v8a regressed) melanie-minions-nlp v8a-genqa 16/05/2026 05:10:19 0 / 700 0.652 0.836  ← REGRESSED -0.031 acc, -0.047 speed vs v9. Flan-T5-base generative; landed at low end of predicted band. Generative confirmed dead lever.
NLP (v9 prior) melanie-minions-nlp v9-doc-ensemble 15/05/2026 19:46:01 0 / 700 0.683 0.883
NLP (v11 regressed) melanie-minions-nlp v11-canonical-answer 15/05/2026 21:26:38 0 / 700 0.680 0.881  ← REGRESSED -0.003 vs v9; canonicalizer rewrites didn't pass 0.9 AE threshold
NLP (v11 resubmit) melanie-minions-nlp v11-canonical-answer 15/05/2026 21:39:55 0 / 700 0.680 0.873  ← Same image resubmit confirms accuracy is real -0.003, speed within variance
NLP (v10 neutral) melanie-minions-nlp v10-template-lite 15/05/2026 20:16:17 0 / 700 0.683 0.882
NLP (v9 first) melanie-minions-nlp v9-doc-ensemble 15/05/2026 19:25:50 0 / 700 0.683 0.868
NLP (prior v8b) melanie-minions-nlp v8b-chunked-context 15/05/2026 18:35:19 0 / 700 0.679 0.872
NLP (prior v7) melanie-minions-nlp v7-finetuned-v1 15/05/2026 11:39:09 0 / 700 0.517 0.880
NLP (prior v5c) melanie-minions-nlp v5c-no-para 14/05/2026 19:44:08 0 / 700     0.483   0.912
ASR    melanie-minions-asr      nemo-zs     14/05/2026 20:33:36   0 / 400       0.956   0.946  ← NEW BLENDED HIGH; Parakeet-TDT-0.6B-v2 zero-shot, +0.097 speed vs ft-lora32-v1
CV     melanie-minions-cv       cv-yolo-v2-tier1-best 14/05/2026 17:10:09 0 / 500 0.556 0.956
CV (v3-pre) melanie-minions-cv  cv-yolo11m-v3-pre 15/05/2026 11:34:42 0 / 500   0.376   0.955  ← REGRESSED -0.180; v11m fully trained but matched-imgsz lost to v8s+upscaled. Tier1 stays on leaderboard.
CV (v11m-1280-noaug-v1) melanie-minions-cv  v11m-1280-noaug-v1 16/05/2026 04:04:59 0 / 500 0.474 0.949  ← REGRESSED -0.082; v11m at 1280 aug=0 looked great locally (hard held-out 0.9088, +0.014 vs tier1) but v11m's local→cloud gap is structurally ~0.44 vs v8s's ~0.35. Tier1 stays on leaderboard.
CV (cv-augc1-v4) melanie-minions-cv  cv-augc1-v4  16/05/2026 15:28:35 0 / 500 0.553 0.962  ← Phase C.1 augmented training (JPEG aug + 1024×1024 native-res tile crops, v8s @ imgsz=1024 scale=0.80), shipped with mismatched Dockerfile config (CV_IMGSZ=768 default). Essentially tied with tier1; informative only as a config-mismatch demonstration.
CV (cv-augc1-v4-1280) melanie-minions-cv  cv-augc1-v4-1280 16/05/2026 18:28:27 0 / 500 0.553 0.959  ← Same model rebuilt with matched Dockerfile config (CV_IMGSZ=1280 CV_CONF=0.001 CV_IOU=0.7 CV_AUGMENT=0). Hard held-out lifted +0.04 to 0.948 / small AP 0.779. Cloud flat at 0.553 — gap WIDENED 0.349 → 0.395. Phase C confirmed dead as a path to 0.7; v8s/v11m family at-ceiling. Tier1 stays on leaderboard.
Noise  melanie-minions-noise    latest      12/05/2026 03:54:55   0 / 500       1.000   0.970
AE     melanie-minions-ae       hybrid-v3   14/05/2026 19:26:06   0 / 30        0.555   0.849  ← STILL SHIPPED via highest-score retention
AE (ppo-scripted-v1) melanie-minions-ae ppo-scripted-v1 17/05/2026 12:19:21 0 / 30  0.450  0.607  ← REGRESSED -0.105 acc, -0.242 speed vs hybrid-v3; blended 0.489 (-0.139). Tier-2 #9 PPO with --opponents scripted (5-archetype scripted library, no self-play) trained from BC warm-start; best eval at update 45 was local 0.6322 (cloud only achieved 0.450). Local til test 0.741 → cloud 0.450 = gap 0.291, the same structural local→cloud gap that killed every prior PPO/BC submission. Speed regressed because policy net forward + top-K cascade adds ~3-5x per-tick wall clock vs heuristic-only path. **Confirms training-side AE hypotheses are exhausted** — even the most informative training distribution (5 distinct scripted opponents) doesn't close the cloud gap.
AE (mcts-light-v2) melanie-minions-ae  mcts-light-v2  16/05/2026 23:52:21  0 / 30  0.487  0.595  ← REGRESSED. Blended 0.514 vs hybrid-v3's 0.628 (−0.114). Latency cap (80ms) + pre-flight gate prevented v1's timeout, but MCTS still cost +7.7 min cloud wall-clock vs hybrid-v3, AND accuracy dropped 0.068 (MCTS replacing hybrid actions with cloud-worse choices — stationary-opponent assumption doesn't transfer). Local til test 0.6618. Local-cloud gap 0.175 (tighter than hybrid-v3's 0.219 but local floor was lower). MCTS as a primary planner is dead; remaining variants: conservative-MCTS (high-confidence override only) + heuristic-only A/B.
AE (mcts-light-v1) melanie-minions-ae  mcts-light-v1  16/05/2026 ~23:00  TIMEOUT  —  —  ← "Your model took too long to evaluate". MCTS DEPTH=5 WIDTH=96 ran every tick with no latency cap → ~2400 expansions/tick × ~0.5-1 ms = 1.2-2.4 s/tick vs cloud's ~600 ms/tick budget. Local `til test` (no wall-clock cap) didn't catch it. No score, no leaderboard impact. v2 fixed the timeout but introduced score+speed regression.
AE (ppo-selfplay-v2)  melanie-minions-ae  ppo-selfplay-v2  16/05/2026 18:09:15  0 / 30  0.436  0.857  ← REGRESSED -0.119 vs hybrid-v3 (but +0.131 over v1, confirming BC warm-start + self-play beats from-scratch). til test (hybrid wrapper) 0.7068 → cloud 0.436 = gap 0.271 (BIGGER than hybrid-v3's 0.219); v1's apparent "gap tightening" to 0.10 was an artifact of v1 being weak in pure-policy mode → hybrid wrapper added more relatively. Confirmed: self-play helped some, but doesn't clear hybrid-v3 ceiling; current follow-up is code-only mcts-light-v1.
AE (ppo-selfplay-v1)  melanie-minions-ae  ppo-selfplay-v1  16/05/2026 13:18:52  0 / 30  0.305  0.851  ← REGRESSED -0.250; BC warm-start was silently skipped (--n-frames 1 vs ckpt's 4), PPO trained from random init for 200 updates. But local-cloud gap 0.10 (vs structural 0.23) → first signal that league self-play tightens the gap. v2 retry capitalized on this.
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
v11-canonical-answer 15/05 21:26  0.680   0.881   0 / 700   0.711(old)   REGRESSED -0.003 vs v9. Full-doc canonicalizer's +10 replay (exact/substr 451→461 on OLD-eval proxy) did not transfer through the 0.9 AE threshold; rewrites either weren't sampled on the hidden corpus, or were canonicalized into forms that the ModernBERT equivalence model rated below 0.9. Blended 0.730 vs v9 0.733.
v11-canonical-answer 15/05 21:39  0.680   0.873   0 / 700   0.711(old)   Same image resubmit. Accuracy stable at 0.680 (confirms the -0.003 is real, not variance); speed dropped slightly. v9-doc-ensemble retains the leaderboard slot.
v8a-genqa            16/05 05:10  0.652   0.836   0 / 700   0.682        REGRESSED -0.031 acc, -0.047 speed vs v9. Flan-T5-base fine-tuned on all 883 (q,ctx,ans) triples with --use-chunk-context, fp32 inference (T5 fp16 NaN trap), beam=4 generation on top reranked chunk. Local→cloud gap (0.030) was consistent with v9 (0.028) → transferred predictably. Generative answers either failed 0.9 AE threshold or paraphrased away. Blended 0.694 vs v9 0.733. **Generative QA confirmed dead lever on this corpus.**
v9-doc-ensemble      16/05 05:21  0.683   0.886   0 / 700   0.711        Third v9 resubmit. Accuracy unchanged; speed bumped to new high 0.883→0.886 (+0.003). Best NLP blend now 0.734. Marginal speed variance.
v12-candidate-ranker 16/05 13:48  0.642   0.829   0 / 700   0.663        REGRESSED -0.041 acc, -0.057 speed vs v9. Candidate-answer reranker (RoBERTa top-12 spans + rule/canon/doc-mined literals, heuristic + optional logistic ranker). Local buckets shifted exact 273→204 (-69), substr 178→232 (+54), diff 395→410 (+15) — confirms the predicted failure: ranker promoted doc-mined short tokens (e.g. "37" over "37 days") that passed the exact/substr training proxy but failed the 0.9 ModernBERT AE threshold. Speed -0.057 from candidate mining over 18 sentences × 6 regex types per question. Blended 0.689 vs v9 0.734. **Candidate-ranker confirmed dead lever; NLP architecture exhausted.**
v13b-deberta         16/05 local  —       —       —         0.667        NOT SUBMITTED. DeBERTa-v3-large QA retune built and tested locally after OOM fix (`--batch-size 2 --gradient-accumulation-steps 4 --gradient-checkpointing`). Baseline v9 local 0.711 in 3:48; v13b local 0.667 in 9:06. Failed gate by -0.044 and 2.4x slower. NLP frozen at v9-doc-ensemble.
v14-llm-rag          17/05 16:15  0.734   0.286   0 / 700   0.754        ★ NEW NLP ACCURACY HIGH ★ (+0.051 cloud vs v9). Architecture change: kept v9 BM25+BGE+rerank retrieval, swapped RoBERTa-large extractive head for Qwen2.5-7B-Instruct-AWQ via vLLM. T4 ABI gauntlet survived: vLLM downgraded torch which broke both pre-installed torchao (`torch.int1`) and flash_attn (undefined C++ symbol); fix was pin transformers==4.46.3 (gated imports) and uninstall both broken NGC `.so`s. Runtime knobs: `quantization=awq` (Marlin needs sm_80+, T4 is sm_75), `enforce_eager=True` (skip ~60-120s CUDA-graph capture during the 5-min corpus-load gate). Local-cloud gap 0.020 (vs v9 0.028) — confirms hypothesis that the +5pp was capped by extractor class, not retrieval. **Blended REGRESSED -0.112 vs v9** (0.622 vs 0.734) because cloud wall-clock 3:50 → ~21 min (speed 0.886 → 0.286). Highest-score retention keeps v14 on the accuracy slot; v9 remains the better-blended NLP contribution to the qualifier total until v14b/v14c recover speed. Top of leaderboard now belongs to us on raw NLP score.
v14b-speed           17/05 build  —       —       —         —            SKIPPED. Was prompt-trim only; superseded by v14c which makes the bigger swap.
v14c-qwen3-4b        17/05 local  —       —       —         0.659        LOCAL ONLY, NOT SUBMITTED. Two-axis change: (1) model Qwen2.5-7B-AWQ → cpatonn/Qwen3-4B-Instruct-2507-AWQ-4bit, (2) base image NGC pytorch → vllm/vllm-openai:v0.9.0 (fixes NGC torch/flash_attn/torchao ABI fight + unlocks transformers ≥4.51 for Qwen3 model_type). Local wall-clock 27:29 → **5:10 (5.3× speedup)** — far above the projected 1.7×. BUT accuracy regressed -0.095 vs v14 (0.754 → 0.659) and -0.052 vs v9 (0.711 → 0.659). Projected cloud blended ~0.684 — better than v14's 0.622 but worse than v9's 0.734, so v14c is not a ship. Three hypotheses for the drop, ordered by likelihood: (1) 4B is below the QA capacity threshold for this corpus (especially L2 cross-fact composition); (2) Qwen3-Instruct-2507 paraphrases more than Qwen2.5-Instruct under the same "quote verbatim" prompt; (3) trimmed 3-shot prompt under-anchors the smaller model. Next: v14d-qwen3-8b tests whether the regression was 4B capacity vs Qwen3 paraphrase tendency.
v14d-qwen3-8b        17/05 build  —       —       —         —            IN PROGRESS. Same architecture/base-image as v14c. Swap LLM repo to Qwen/Qwen3-8B-AWQ (or cpatonn 2507 variant if available). Bigger model, expected accuracy 0.72-0.75 / speed 0.50-0.60 → blended target 0.69-0.73. If accuracy recovers, the v14c drop was capacity. If accuracy stays flat at ~0.66, Qwen3 paraphrase tendency is the structural issue and we need Qwen2.5 family or a prompt rework.
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
v11m-1280-noaug-v1     16/05 04:04        0.474   0.949   0 / 500   0.812 / 0.909          REGRESSED -0.082; same v11m weights, inferenced at imgsz=1280 aug=0. Hard held-out 0.9088 (+0.014 vs tier1, small AP 0.7168) looked great locally. Cloud landed -0.082 because v11m's local→cloud gap is structurally ~0.44 vs v8s's ~0.35. Speed beat T4-based projection (cloud 0.949 vs projected 0.85-0.92) — cloud GPU is faster, no-TTA at 1280 isn't a speed bottleneck. Tier1 stays on leaderboard.
```

## CV local A/Bs (16 May)

```text
Variant                                    Hard held-out mAP / small AP   Notes
tier1-debug off mode                       0.8947 / 0.6434                Bit-identical to shipped tier1 (sanity).
tier1 + CV_CONF=0.40                       0.8929 / 0.6434                -0.0018 mAP. Raising conf does NOT clean up FPs profitably.
tier1 + CV_CONF=0.60                       0.8909 / 0.6434                -0.0038.
tier1 + CV_CONF=0.70                       0.8886 / 0.6434                -0.0061.
tier1 + CV_CONF=0.80                       0.8854 / 0.6434                -0.0093. mAP drops monotonically with conf — PR-curve tail is doing real work even at 2.8% precision.
tier1 + CV_TILE_MODE=2x2                   0.8993 / 0.6139                Total +0.0046; small AP REGRESSED -0.030 (tile-edge filter drops legit small detections). 5 forward passes per image.
tier1 + CV_TILE_MODE=2x1                   0.8771 / 0.5929                Regressed everywhere. 3 forward passes.
tier1 + CV_TILE_MODE=3x2                   0.9009 / 0.6255                Total +0.0062 driven by medium AP +0.022 (0.8506 → 0.8728). Small AP still down. 7 forward passes (≈3× compute) — speed math: blended ≈ 0.638-0.653 vs shipped 0.656. Not shipping.
tier1 + 3x2 EM=0 OV=0.30                   ≈ 0.90 / 0.62                  Edge-margin off + 30% overlap; small AP did NOT recover, confirming the regression is model behavior on tile crops, not the edge filter.
v8s-1024 imgsz=1024 aug=1                  0.8217 / 0.5327                v8s-1024 retrain (`copy_paste=0.40`). REGRESSED -0.073 vs tier1; small AP -0.110. Toxic copy_paste was the regressor.
v8s-1024 imgsz=1280 aug=0                  0.8370 / 0.5168                Same weights, upscaled inference. Still -0.058 vs tier1. NOT submitted.
v8s-1024 imgsz=1024 + tile=3x2             0.8361 / 0.5058                Same weights, tiled inference. Still -0.059 vs tier1. NOT submitted.
v11m@1280 conf=0.001 iou=0.70 aug=0        0.9088 / 0.7168                BEST hard-held-out across all 16/05 sweeps. +0.014 vs tier1. Small AP +0.073. Submitted as v11m-1280-noaug-v1; cloud regressed -0.082 (local→cloud gap was 0.44, structurally wider than v8s's 0.35).
v11m@1280 conf=0.001-0.20 iou=0.50/0.70    0.9075-0.9088                  Sweep was tightly clustered; conf knob is essentially flat at imgsz=1280 aug=0 for v11m.
```

Failure-analysis breakdown of the tier1 baseline (Pass A on hard held-out):
- 3334 ground-truth boxes; class-correct matches `3317`, class-wrong `16`
  (0.48%), unmatched `1`, false-positives `176`. Aircraft-subclass confusion
  is dead as a hypothesis.
- FP/TP by score bucket showed 97% of TPs are conf ≥ 0.80; conf 0.20-0.70
  collectively contributed `15 TP / 128 FP` but their PR-curve contribution
  is what makes the integrated mAP higher at low conf, hence the
  `CV_CONF` sweep regressing.
- Per-area AP `small=0.6434 / medium=0.8506 / large=0.9128` — small bucket
  is the entire bottleneck. At inference imgsz=896 the dataset's p25 box
  becomes ~29×29 pixels in the model's input.
- Worst 25 images (`3919`, `4853`, `1992`, `249`, `4780`, `2260`, `3510`,
  `1293`, `1851`, ...) all have 5-11 boxes per image with mixed scales on
  photo-composited backgrounds; recall 0.66-0.86. No systematic class or
  scene bias.

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
[AE UN-PARKED 16/05] After reviewing the TIL workshop materials (notebook 05 explicitly diagnoses bc-belief's failure as "overfit to a weak fixed opponent" and prescribes self-play as the fix), added a `SnapshotPool` to `training/ae/train_ppo.py` that holds historical actor snapshots and feeds them into `_make_opponents`. Previous `FrozenPolicyOpponent` deepcopied the live actor → effectively "play your shadow", not true self-play. New `--opponents league` + `--snapshot-interval 10` + `--snapshot-pool-size 5` gives a proper opponent curriculum: random + planner + aggressive + frozen-self-from-K-updates-ago.
ppo-selfplay-v1  16/05 13:18         0.305/0.851 official 0 / 30 official errors   FIRST SHIP OF NEW ARCH (n_frames=4 model.py + new weights). REGRESSED -0.250 vs hybrid-v3. Root cause: `--n-frames 1` (recommended by claude in NOTES, copied from old bc-belief example) didn't match the BC checkpoint's `n_frames=4` and `load_actor` silently skipped the warm-start (train_ppo.py:656-661). PPO trained 200 updates from random init against league opponents. Best PPO eval climbed monotonically -0.088 → +0.436 across 7 saves. Local eval against `mixed`: 0.4040 (-0.083 vs BC ckpt's 0.4866 baseline). Cloud 0.305. **Critical positive finding**: local-cloud gap was 0.099 (0.404→0.305) vs the structural ~0.23 across all 9 prior AE submissions. First evidence that league/self-play training distribution materially tightens the gap. Speed 0.851 ≈ hybrid-v3 0.849 (new arch + hybrid wrapper is fine speed-wise).
ppo-selfplay-v2  16/05 18:09         0.436/0.857 official 0 / 30 official errors   REGRESSED -0.119 vs hybrid-v3 (0.555), but +0.131 over v1 (0.305) — confirms BC warm-start + self-play > self-play from random init. Training peaked at update 120/200; 7 monotonic best-saves: 0.4908 → 0.5038 → 0.5327 → 0.5333 → 0.5982 → 0.6428 → 0.6601 against league. Local evals: pure-policy vs mixed 0.5747 (+0.088 over BC, +0.171 over v1); til test (hybrid wrapper) 0.7068. Local-cloud gap **0.271** (0.7068 - 0.436), BIGGER than hybrid-v3's 0.219 — the v1 gap-tightening to 0.10 was an artifact (v1 was weak in pure-policy so the hybrid wrapper added more in relative terms; v2's stronger raw policy makes the wrapper contribute less relatively, exposing the structural gap). **Self-play opponent curriculum + warm-start was a real win over from-scratch training, but the local-cloud gap is structural to the hidden eval distribution, not to our training distribution.** This is the cleanest negative result on the league/self-play hypothesis we could get. Superseded as current workstream by code-only `mcts-light-v1`; hybrid-v3 remains the shipped best until MCTS-light has an official score.
mcts-light-v1    16/05 ~23:00       TIMEOUT  0 / 30 cloud errors (model never finished). Violet bot: "Your model took too long to evaluate." No score returned, leaderboard unaffected. Inference-side candidate: bounded tactical beam search in `AEManager` (depth 5, width 96, side-beam for tactical lines) + `HybridAEManager` trust-MCTS gate. Implementation focused entirely on score, none on speed — no per-call latency cap, no pre-flight gate, MCTS fired every tick. Math: `AE_MCTS_DEPTH=5 × AE_MCTS_WIDTH=96` → ~2,400 expansions per tick × ~0.5-1 ms each = **1.2-2.4 s/tick**. Cloud budget is ~30 min / 30 games / ~100 ticks ≈ **~600 ms/tick**. Over by 2-4×. Local `til test` (no wall-clock cap) didn't catch it because it just runs to completion regardless of speed.
mcts-light-v2    16/05 23:52        0.487/0.595 official 0 / 30 official errors   REGRESSED -0.068 acc, -0.254 speed vs hybrid-v3 → blended 0.514 vs hybrid-v3's 0.628 (-0.114). v2 added: hard `time.monotonic()` 80 ms latency cap with best-so-far fallback inside `_tactical_lookahead_action`; cheap pre-flight gate `_should_run_mcts(location)` that returns False when no bombs/enemies/enemy-bases are within `depth+1` Manhattan reach; shrunk Dockerfile defaults DEPTH=5→3, WIDTH=96→24, BUDGET_MS=80. The cap + gate prevented timeout (v1 → v2 fix), but MCTS still cost +7.7 min cloud wall-clock (~4.5 min → ~12.2 min) AND accuracy regressed −0.068 vs hybrid-v3. Local `til test` 0.6618 with 17-25 s/round (~2 min total). Local-cloud gap was 0.175 — narrower than hybrid-v3's 0.219, but local floor was lower. Best read on the accuracy regression: MCTS commits to simulated tactical lines using a stationary-opponent assumption; cloud opponents move on their own logic, so we pay simulated-combat cost without earning simulated-combat reward. Same distribution-shift failure that killed bc-belief and ppo-selfplay, this time at inference. **MCTS as a *primary planner* is dead.** Remaining cheap variants: (A) conservative-MCTS (raise `AE_MCTS_MIN_SCORE` 12→22, shrink DEPTH=2 WIDTH=16) so MCTS is a high-confidence override layer only; (B) heuristic-only MCTS A/B (AE_MODE=heuristic) to isolate MCTS contribution vs the hybrid wrapper. After A or A+B, AE is exhausted at hybrid-v3 across both training-side AND inference-side intervention classes.
```

## AE current code candidate

`mcts-light-v1` is a code-only inference A/B after the training-side AE
failures. It adds bounded tactical lookahead in `AEManager`, scoring safe
short-horizon bomb/base/enemy lines from the live belief map. Hybrid mode
trusts the heuristic action before policy only when the projected tactical
value clears threshold. Built knobs: `AE_MCTS=1`, `AE_MCTS_DEPTH=5`,
`AE_MCTS_WIDTH=96`, `AE_MCTS_MIN_SCORE=12`, and
`AE_HYBRID_TRUST_MCTS=1`. Workbench target: restore ppo-v1 weights, then
build/test/submit tag `mcts-light-v1`. No official score yet.

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
4. **CV is PARKED at tier1 (0.556/0.956).** Three more avenues tested 16/05
   on top of the prior failure analysis: tiled inference patch (local
   `+0.006` only on medium AP at 3× compute), v8s-1024 retrain (regressed
   locally; `copy_paste=0.40` toxic on this dataset, NOT submitted),
   v11m@1280 aug=0 submitted as `v11m-1280-noaug-v1` (cloud `0.474/0.949`,
   regressed `-0.082`). Calibration learned: v11m's local→cloud gap is
   structurally `~0.44` vs v8s's `~0.35`, so a v11m hard-held-out would
   need `≥ 0.99` to beat tier1 cloud, out of reach. Cloud GPU is faster
   than expected — speed is no longer a CV constraint, but accuracy is
   the bottleneck and we don't have a remaining lever. Tier1 stays on
   leaderboard via highest-score retention. Remaining qualifier lift has
   to come from NLP / ASR / AE.
