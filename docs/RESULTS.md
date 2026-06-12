# RESULTS — Team melanie-minions, DSTA BrainHack TIL-AI 2026

A public record of what Team **melanie-minions** built, shipped, and abandoned across the five tasks of the DSTA BrainHack TIL-AI 2026 competition — five separately Dockerized model services (AE, ASR, CV, NLP, Noise), scored on a blend of accuracy/reward and speed. This document is honest about dead-ends: most of the value here is in the long list of things that *didn't* work and why.

> Competitor teams are anonymized as Team A/B/C/D. A brief note on our development process is in the footer.

---

## Final scores

| Task  | Weight | Shipped solution | Final score / placement |
|-------|--------|------------------|--------------------------|
| **AE** (autonomous exploration, RL bomb-game) | 40% | `confpol-semis2b-u75` heuristic+gated-PPO consultant hybrid + `g02-sample-08` CEM planner weights | 0.382 cloud (non-crater) / 0.746 speed; Semifinals worst-bracket placement ~1.46; 1st in live hardware runs |
| **ASR** (speech recognition) | 20% | `nemo-ft-v3` — Parakeet-TDT-0.6B-v2 fine-tune + post-processing | 0.970 acc / 0.947 speed (blended 0.964) |
| **CV** (object detection, 18-class LTWH) | 20% | `yolo11l-1024-alldata-final-v1` served at 1280px (Semis: 1408px) | 0.671 acc / 0.950 speed (blended 0.741); Semis config 0.684 acc (blended 0.747) |
| **NLP** (retrieval QA) | 20% | `v28-optimized-bm25` — pure tuned BM25 retrieval | 0.984 acc / 0.985 speed (blended 0.984) |
| **Noise** (adversarial image perturbation) | Finals tool only | `level10-detector-stress` — AdvGAN + detector-stress overlay | 1.000 fairness / 0.947 speed; ~6% mAP@50 disruption on victim |

Estimated blended Qualifier total ≈ **0.72** (AE 0.25, NLP 0.15, ASR 0.19, CV 0.13).

---

## AE — Autonomous Exploration (40%)

A partially-observed grid wargame: farm resources → fuel → bombs → destroy enemy bases, scored on relative placement against other agents. The single most important and most contested task.

### Shipped

A **hand-coded Dijkstra heuristic core** (greedy single-target scorer over item/mission/base values) wrapped in a **confidence-gated PPO consultant**: the learned policy only overrides the heuristic when its margin and confidence floor are both high. Checkpoint `confpol-semis2b-u75` was warm-started from a native consultant (0.661 cloud) and fine-tuned on a **foreign-opponent curriculum** (an external A\* agent, self-play snapshots, and aggressive-proxy bots). Planner weights `g02-sample-08` (tether 1.31, base 80, 8 other scalar knobs) came from a CEM search and were the first candidate to clear *all four* promotion gates: local placement, local raw-AE, held-out composition gap, and cloud non-crater. Cloud improved 0.661 → 0.671 on the rebased opponent set; on-hardware runs placed 1st with ~1.5× reward margins.

### What we tried

**Heuristic / planner line**
- Greedy Dijkstra single-target scorer (item/mission/base values, bomb-cost tuned) — **shipped** (the core)
- Base-tether (defend-while-farming bias, LLM-rationale-mined) — **shipped**
- C+bomb7 profile (item 80/40, base 100, bomb_cost 7.0) via multi-seed calibration — **shipped**
- Bomb-timer offensive split (detonate at 5 decision-steps, escape window 3) — **shipped** (correctness fix)
- CEM planner-weight tuning over 10 scalar knobs (`g02-sample-08`) — **shipped**
- Stun/respawn downtime pricing, option-mode planner (defend/destroy/farm/hunt/escape), Dijkstra bomb-cost sweeps, broad heuristic-knob grids (224/288/240-candidate screens) — **dead-end** (flat or noise)
- Forward-sim plan re-score (project bomb landing, demote landable bases) — **dead-end** (local flat, cloud −0.081)
- Opening book (per-spawn farming openings, divergence-gated) v1/v2 — **dead-end** (never fires on the eval spawn; gate validated against the wrong planner)
- CEM `g00-fixed-03` / `g03-sample-08` — **dead-end** (byte-identical/inert on the live seed, or cloud-rejected with search bounds pegged)

**Learned-policy line — every variant died the same death: local-opponent overfit**
- From-scratch CNN-PPO vs mixed opponents — **dead-end** (local 0.54–0.56, cloud collapsed to ~0.41–0.51)
- Behavior cloning of the heuristic + PPO fine-tune — **dead-end** (local val-acc 0.87–0.90, cloud ~0.36–0.41)
- Self-play RL (BC warm-start + snapshot pool) — **dead-end** (local 0.575, cloud 0.436)
- Belief-map state augmentation (704k-param CNN, learned belief tensor) — **dead-end** (more params = more overfit; cloud 0.287)
- Tactical / 12-way macro-policy selectors — **dead-end** (flat vs baseline)
- MCTS-light bounded beam search — **dead-end** (v1 timed out; v2 regressed on speed *and* accuracy)
- Confidence-gated PPO consultant (native + foreign-curriculum rungs) — **shipped** (`semis2b-u75`; later rungs overfit into a single opponent — an inverted-U)
- Dir-2 BC-clone + self-play league — **dead-end** (flat)
- LLM-as-player (Sonnet/Gemini vs heuristic) — **dead-end** (LLM *rationale-mining* for heuristic tuning was the real win)

**Bomb/safety flags (all default-OFF, gated)**
- Contention-aware item valuation (discount items an opponent reaches first) — **shipped** ON, but measured inert (lacks mid-game opponent positions)
- Time-layered danger map (chain-resolved per-tick lethality) — **shipped** (default-OFF, byte-identical)
- No-self-damage bomb gate (own bombs deal zero self-damage, env-confirmed) — **parked** OFF (high-variance, inverts under the shipped planner weights)
- Base-kill no-escape surgical variant — **dead-end** (worst of both worlds)
- Phantom-bomb rollback (latent-state consistency) — **dead-end** (noise-floor)
- Resource-reward env fix (org confirmed pickups score nothing) — **shipped** (pin bumped; placement-neutral)

**Key learnings:** local eval ≠ cloud — a structural ~0.18–0.30 gap that no amount of training, bigger nets, frame-stacking, or richer state closed; the bottleneck is opponent-distribution mismatch, not a tunable knob. Hand-coded heuristics transfer; learned policies overfit the local opponent. Finals scoring is *relative rank*, so a retune can earn less absolute reward yet place better — which makes single-agent cloud reward the wrong shape. A relative-rank **melee gate** (head-to-head across opponent brackets, with held-out compositions) was the first eval that agreed with cloud ranking. On-hardware A/B on the deterministic Finals seed was the final arbiter, and it once *inverted* a synthetic gate's pick.

---

## ASR — Speech Recognition (20%)

### Shipped

`nemo-ft-v3`: **NVIDIA Parakeet-TDT-0.6B-v2** fine-tuned (step 713, val WER 0.0856) plus a layered post-processing stack — spelling normalization, regex fixes, in-world proper-noun corrections, currency/ordinal/percentage rules. **0.970 acc / 0.947 speed.**

### What we tried
- Empty-string baseline, digit-verbalization + silence guard, VAD-off + hallucination guards — **dead-end** (calibration baselines)
- LoRA rank-32 fine-tune of the original backbone (3 epochs) — **shipped** (early winner)
- Parakeet-TDT-0.6B-v2 backbone swap, zero-shot — **shipped**, then iteratively post-processed (v3→v7: phonetic corrections, slang prompter, proper-noun and currency rules) — **shipped**
- Parakeet-TDT fine-tune (nemo-ft-v1/v2/v3) — **shipped** (final line)
- N-gram LM shallow fusion for in-world proper nouns — **dead-end / parked** (real local WER gain, but the GPU decode path crashed on the cloud GPU, so no net gain)

**Key learnings:** post-processing rules on a strong zero-shot backbone matched the blended score of full fine-tuning. The n-gram LM was a big *local* win that didn't survive the cloud GPU — a recurring "validate on the actual hardware" lesson.

---

## CV — Object Detection (20%)

18-class LTWH bounding-box detection on a hidden distribution with heavy small-object bias.

### Shipped

`yolo11l-1024-alldata-final-v1`: **YOLOv11l** trained 70 epochs at 1024px on all data (val+test folded back in), then served at **upscaled 1280px** inference. **0.671 acc / 0.950 speed.** Post-Qualifier Semis prep found 1408px is the empirical peak: **0.684 acc** (blended 0.747).

### What we tried
- YOLOv8s 768→896 with TTA (tier1 baseline) — **shipped** (Qualifier leaderboard entry)
- YOLOv11m native 1024, and 1024→1280 upscale — **dead-end** (bigger model + matched resolution *regressed* to 0.376; wider local→cloud gap)
- Augmented training (JPEG re-compression + tile crops) — **dead-end** (+0.04 held-out but the local→cloud gap *widened* — specialized away from cloud)
- Inference-knob sweeps (CONF/IOU/IMGSZ/AUGMENT), tiled inference (2×2/2×1/3×2) — **dead-end** (CONF regressed monotonically; tiling hurt small-AP)
- YOLOv8s retrain at 1024 with aggressive copy-paste — **dead-end** (toxic on an already-composited dataset)
- YOLOv11l plusval recipe (fold val back, fix ship-class imbalance) — **shipped** (stepping stone)
- YOLOv11l **all-data @ 1024, served at 1280** — **shipped** (Qualifier champion)
- YOLOv11l native 1280 training — **dead-end** (lost to smaller-trained + upscale, same paradox again)
- YOLOv11l TTA (optimized-v3) — **dead-end** (raw +0.001, blended −0.002 from speed cost)
- YOLOv11l upscale sweep 1280→1408→1536 — **parked** at **1408** (peak; 1536 regressed)
- **RF-DETR** (DINOv2 ViT-DETR) at 728 / native-952 / 952-upscaled-1064 — **dead-end** (ties YOLO acc with a *narrower* local→cloud gap, but hits the same ~0.665 ceiling and loses on speed; pos-embeds are tied to resolution so the upscale trick doesn't transfer)
- Input purification + multi-model ensembling scaffolding — **parked** (merged to main but OFF by default; unvalidated)

**Key learnings:** the ~0.67 cloud ceiling is **content-shift, not backbone capacity** — RF-DETR disproved that a fancier foundation backbone could break it. The one reliable lever was **upscale-at-inference** (train small, serve big), and it's counterintuitive: a bigger model at matched resolution loses to a smaller one upscaled. ~80% of the local→cloud gap is scene-content shift, not a fixable inference knob.

---

## NLP — Retrieval-Augmented QA (20%)

### Shipped

`v28-optimized-bm25`: **pure BM25 retrieval, no neural models** — document-level BM25 (k1=2.05, b=1.0) fused with passage-level BM25 (k1=1.5, b=0.75, 3-sentence windows) via z-score weighting (doc + 0.6 × max-passage). Local retrieval hit-rate 0.9853. **0.984 acc / 0.985 speed**, 0/700 errors. This beat every neural and adversarial path on the blended frontier.

### What we tried
- Hybrid BM25+BGE+rerank+RoBERTa-SQuAD2 (initial) — **dead-end** (scored 0.000 on cloud due to an org eval-server bug, then recovered)
- Defensive dict-ID parser (`v4-dict-id`) — **shipped** (recovery after the org fix)
- Paragraph chunking + batched SQuAD2 + BM25 backfill variants — **dead-end / shipped** (mixed; fallback fired too aggressively)
- RoBERTa-large-squad2 fine-tune on local data — **shipped** (+0.034 cloud)
- **Chunked-context fine-tune** (train on answer-containing 3-sentence chunks) — **shipped** (+0.162 cloud; inductive bias matched the retrieval pipeline)
- Doc-ensemble (whole-doc BM25+BGE prior + reranker seeding) — **shipped** (cloud 0.683; blended high before the BM25 line)
- Template-lite canonicalizer — **shipped** (neutral); broader canonicalizer / candidate-ranker / doc-mined entities — **dead-end** (passed weak local proxies, failed the strict 0.9 answer-equivalence threshold)
- DeBERTa-v3-large, ModernBERT (stock + fine-tuned), Qwen3-Reranker swap, hybrid router — **dead-end** (missed the RoBERTa gate and/or timed out on cloud)
- Flan-T5 generative QA — **dead-end** (paraphrases failed the 0.9 threshold)
- LLM RAG: Qwen2.5-7B-AWQ via vLLM — **shipped** (highest raw accuracy 0.734, but speed cost killed the blended score); Qwen3-4B/8B on a different base — **dead-end** (cloud-incompatible base image)
- QLoRA fine-tune of Qwen3-8B — **parked** (trained cleanly but unshippable: vLLM LoRA kernel crashed on the T4, AWQ/quant export blocked)
- **Adversarial trigger** (HotFlip universal trigger vs the ModernBERT answer-equivalence classifier), `v20`/`v21` — **shipped** (0.951 acc via the trigger; proved the eval classifier is exploitable, but the speed penalty lost to pure BM25)
- Speed line: vectorized retrieval, large reranker, conditional model bypass, torch.compile, **pure BM25**, **tuned hybrid BM25** — **shipped** (the final climb to 0.984/0.985)

**Key learnings:** on this corpus, **tuned BM25 beat every neural and LLM path** on the blended frontier. The org's 0.9 answer-equivalence threshold killed an entire class of post-processing (canonicalizers, rankers, paraphrasing LLMs) that passed weaker local proxies. The most transferable training trick was matching the fine-tune distribution to the retrieval pipeline (chunked-context, +0.162). The HotFlip adversarial trigger was a genuinely impressive exploit that simply lost on speed.

---

## Noise — Adversarial Image Perturbation (Finals tool only)

No direct Qualifier reward; an optional Finals capability to degrade an opponent's CV inputs while staying inside fairness bounds (L2 / SSIM caps).

### Shipped

`level10-detector-stress`: a two-stage pipeline — an **AdvGAN generator** (trained on ResNet18/Imagenette, ~14 KB weights) produces base noise in one forward pass, then a **detector-stress overlay** (multi-phase checkerboards + edge-aware masking targeting CNN receptive fields) saturates the remaining fairness budget. Clamped to ε=32/255, JPEG-re-encoded. **Passes 500/500 fairness, 0.947 speed**, ~6% mAP@50 drop on a YOLO11l victim.

### What we tried
- PGD with a classifier ensemble (ResNet18/MobileNetV3/SqueezeNet/ViT) — **dead-end** (per-query optimization too slow for the speed wall)
- Single-pass AdvGAN generator (Level 9) — **shipped** (sub-ms inference)
- Strength sweep 1→2→3 on a YOLO victim — **shipped** (measured real impact: ~6% at deployed strength, flat to strength 2, 15% only at strength 3 which breaks fairness)
- Detector-stress overlay (Level 10) — **shipped**
- Retrain the generator against a *detector* objective (not classification) — **parked** (identified as the real lever; high-effort, deferred)

**Key learnings:** noise trained on a classification surrogate transfers poorly to detection victims (different gradient structure) — hence only ~6% disruption despite passing fairness. The genuine fix is detector-gradient training; what shipped is a structural workaround.

---

## Timeline (oldest → newest)

- **14 May 2026** — NLP eval overhaul: answer-equivalence threshold 0.5→0.9, leaderboard wiped, doc-ID dict format standardized. Early NLP submissions reset to 0.000 by an org eval-server bug, then recovered (`v4-dict-id`).
- **22 May 2026** — CV Qualifier champion locked: `yolo11l-1024-alldata-final-v1` at upscaled 1280px (0.671 acc).
- **24 May 2026** — NLP final `v28-optimized-bm25` (0.984/0.985); ASR `nemo-ft-v3` (0.970/0.947); Noise `level10-detector-stress`. **Qualifier deadline.**
- **30 May 2026** — Post-lock CV Semis config found: `yolo11l-1408` (0.684 acc, new blended high); RF-DETR architecture bet resolved as a dead-end.
- **01 Jun 2026** — Cloud eval went deterministic (one submit = true score), retiring variance-farming.
- **~04 Jun 2026** — Cloud AE opponent set swapped; all prior cloud numbers became stale. AE `confpol-semis2b-u75` validated at 0.671 on the new set.
- **06 Jun 2026** — Org disclosed resource pickups score nothing; AE env pin bumped to match (placement-neutral). First on-hardware Finals validation on Blackwell RTX 5070 Ti passed.
- **08 Jun 2026** — On-hardware AE A/B on the deterministic Finals seed: `semis2b-u75` won 516/1st, *inverting* the synthetic gate's pick of `semis2c`.
- **10 Jun 2026** — AE `g02-sample-08` CEM planner weights adopted (first to clear all four gates). Surprise hex-strategy agent built. Live Semis hardware runs: 1st in both, ~1.5× margins.
- **11 Jun 2026** — Final Finals submission validated clean on hardware (0 timeouts/crashes). Deploy frozen for Semis/Finals.

---

*The full internal day-by-day submission log and the per-task strategic notes (decisions, gotchas, dead-end post-mortems) live in each task directory's `NOTES.md` and in `docs/`. Much of the experimentation above was driven through an AI-assisted, gated brainstorm → spec → plan → eval → ship workflow; the engineering and results are the headline.*
