# TIL-AI 2026 — Team *melanie-minions*

Our competition entry for **DSTA BrainHack TIL-AI 2026**. The challenge ships as five independent ML problems, each packaged as its own Dockerized FastAPI microservice. We competed in the online **Qualifiers**, advanced to the in-person **Semifinals at Marina Bay Sands**, and were unfortunately eliminated there.

This repo is a public showcase of the engineering: what we shipped, what we tried, and — honestly — what didn't work. Most of the depth went into **AE** (40% of the score), an autonomous-agent bomb-and-resource game where hand-coded heuristics beat every learned policy we threw at it.

> Each task service follows the same shape: `<task>/src/<task>_manager.py` holds *our* logic, wrapped by a thin `<task>_server.py` FastAPI layer. We edit the manager; the server and Dockerfile are mostly fixed.

---

<!-- TOC -->
- [TIL-AI 2026 — Team *melanie-minions*](#til-ai-2026--team-melanie-minions)
  - [The team](#the-team)
  - [Results](#results)
  - [The competition](#the-competition)
  - [How the repo is organised](#how-the-repo-is-organised)
  - [AE — Autonomous Exploration (40%, the deep end)](#ae--autonomous-exploration-40-the-deep-end)
  - [NLP — Retrieval-Augmented QA (20%)](#nlp--retrieval-augmented-qa-20)
  - [ASR — Speech Recognition (20%)](#asr--speech-recognition-20)
  - [CV — Object Detection (20%)](#cv--object-detection-20)
  - [Noise — Adversarial Perturbation (Finals tool)](#noise--adversarial-perturbation-finals-tool)
  - [Engineering notes \& docs](#engineering-notes--docs)
  - [Final words](#final-words)
  - [License](#license)
<!-- TOC -->

---

## The team

- [@kushmics](https://github.com/kushmics)
- [@crispie-bit](https://github.com/crispie-bit)
- [@daaavidsim](https://github.com/daaavidsim)
- [@GeeRuiYang](https://github.com/GeeRuiYang)
- [@TehBingLessSugar](https://github.com/TehBingLessSugar)

---

## Results

- Placed **top-20 out of 80+ novice teams** in the Qualifiers and advanced to the Semifinals.
- Placed **1st in both pre-Semifinals hardware test matches** on competition hardware.
- Finished **top-10 in the Semifinals** and did not advance to the Finals.

### Results at a glance

| Task | Weight | What we shipped | Score / placement |
|------|--------|-----------------|-------------------|
| **AE** (autonomous exploration / RL) | 40% | Dijkstra heuristic core + confidence-gated PPO consultant (`confpol-semis2b-u75`) + CEM-tuned planner weights (`g02-sample-08`) | Qualifiers 0.638 acc-axis / 0.847 speed; **1st in both live hardware test matches** (≈1.5× margins) |
| **NLP** (retrieval QA) | 20% | `v28-optimized-bm25` — tuned BM25 retrieval + universal adversarial trigger as the answer; zero neural models at inference | **0.984 acc / 0.985 speed** (blended 0.984), 0/700 errors |
| **ASR** (speech recognition) | 20% | `nemo-ft-v3` — fine-tuned Parakeet-TDT-0.6B-v2 + post-processing rules | **0.970 acc / 0.947 speed** (blended 0.964) |
| **CV** (object detection, 18-class LTWH) | 20% | `yolo11l-1408` — YOLOv11l trained at 1024px, served at 1408px | **0.684 acc / 0.937 speed** (blended 0.747) |
| **Noise** (adversarial perturbation) | Finals tool | `level10-detector-stress` — AdvGAN + detector-stress overlay | **1.000 fairness / 0.947 speed**; Finals-only CV sabotage |

*Each task is scored 75% accuracy/reward + 25% speed. Blended Qualifier total = 0.814.*

---

## The competition

DSTA BrainHack **TIL-AI 2026** poses five independent ML tasks, each delivered as its own Dockerized FastAPI microservice with a fixed request/response schema:

| Task   | What it is | Weight |
|--------|-----------|--------|
| **AE** | Autonomous-agent bomb-and-resource game, Bomberman-style (RL/heuristics) | 40% |
| **ASR** | English speech-to-text transcription | 20% |
| **CV** | 18-class object detection (LTWH boxes) | 20% |
| **NLP** | Retrieval-augmented question answering | 20% |
| **Noise** | Adversarial image perturbation | Finals disruption tool, no direct Qualifier reward |

Each scored task is graded **75% accuracy/reward + 25% speed**. In the **Qualifiers**, an official evaluator POSTs each task's inputs directly to its container over HTTP. In the **Semifinals/Finals**, all six containers — the five task services plus a WebSocket orchestration server — run together in one Docker Compose stack on competition desktops (Blackwell RTX 5070 Ti hardware).

The official challenge specifications live on the [TIL-AI wiki](https://github.com/til-ai/til-26/wiki/Challenge-specifications); this repo started from the [official template repo](https://github.com/til-ai/til-26).

---

## How the repo is organised

```
TIL/
├── ae/        AE — autonomous-exploration agent (heuristic + PPO consultant)
├── asr/       ASR — Parakeet-TDT fine-tune
├── cv/        CV  — YOLOv11l detector
├── nlp/       NLP — BM25 retrieval QA
├── noise/     Noise — AdvGAN adversarial perturbation
│
│   (every task dir above has the same shape:)
│   <task>/src/
│       ├── <task>_manager.py   ← our inference / game logic (what we edit)
│       └── <task>_server.py    ← thin FastAPI wrapper (fixed schema)
│
├── docs/      orientation + RESULTS.md (cross-task scoreboard & submission history)
├── training/  offline training & evaluation scripts (not shipped in containers)
├── test/      pytest-style per-task tests (test_<task>.py), hit the running container
│
├── til-26-ae/       submodule — official AE environment (til_environment package)
└── til-26-finals/   submodule — Finals WebSocket orchestration + local test server
```

Each task container speaks the same JSON contract used in Qualifiers, so the Finals orchestration server fans out to all five over plain HTTP with zero manager changes.

---

## AE — Autonomous Exploration (40%, the deep end)

**The task.** A partially-observable grid wargame: farm resources → build bombs → destroy enemy bases, scored on **relative placement** against other teams in a melee, not absolute reward. Basically Bomberman with bases!

**Key technologies**
- Python
- Hand-coded Dijkstra-based greedy planning core (item / mission / base-destruction scoring, base-tether defense, calibrated bomb cost)
- PyTorch CNN-PPO confidence-gated consultant policy
- CEM (cross-entropy method) black-box search over planner weights

**What shipped.** The core is a hand-coded **Dijkstra-based greedy scorer** (item / mission / base-destruction values, base-tether defense, calibrated bomb cost) — heuristics. On top sits a **confidence-gated PPO consultant** (`confpol-semis2b-u75`): a policy trained only on the heuristic's low-confidence ticks, consulted only when its margin clears a threshold. It was warm-started from a native checkpoint and fine-tuned on a **foreign-opponent curriculum** (an external A\* bot, self-play snapshots, aggressive/anti-aggressive proxies) to learn robustness rather than mirror-match quirks. Final tuning came from a **CEM black-box search over 10 scalar planner weights** (`g02-sample-08`, e.g. tether 1.31 / base 80) — the first candidate to clear *all* of: local placement, held-out opponent-composition gap, and a cloud non-crater check. On Blackwell competition hardware the deploy went **1st place in both live hardware test matches at ~1.5× margin against 2nd**.

**Dead-ends (there were so very many, literally 500+ commits and attempts).**
- **Every learned-policy line died the same death — local-opponent overfit.** Behavior cloning (val_acc 0.90 local → 0.36 cloud), self-play RL, a 704k-param belief-map CNN (local +0.023 → cloud **crashed −0.27**), tactical macro-policies — all improved local eval while cloud stayed flat at ~0.41. More parameters meant more overfit surface, not less. I was so stuck that I went to scour til-25 public repos to see if they had takeways from their RL task, all that echoed was: **hand-coded heuristics ship; RL overfits.**
- **MCTS-light lookahead** timed out (2–4s/tick vs budget); even with an 80ms hard cap it regressed both speed and accuracy — the stationary-opponent assumption costs without paying.
- A key meta-lesson: **synthetic gates misrank.** Our melee gate once promoted `semis2c` over `semis2b`; an on-hardware A/B on the deterministic Finals seed reversed it (516 vs 411). On-hardware runs became the only trusted arbiter.

---

## NLP — Retrieval-Augmented QA (20%)

**The task.** Retrieval-augmented question answering over a ~300-document corpus: return the top-3 source document IDs plus an answer string. Retrieval gates everything (a miss scores 0), and answers are judged by a **ModernBERT answer-equivalence classifier** at a 0.9 acceptance threshold.

**Key technologies**
- Python
- `rank_bm25` — tuned document- + passage-level BM25 (the only retrieval in the shipped image)
- HotFlip universal adversarial trigger (Wallace 2019), optimized white-box against the bundled ModernBERT judge

**What we tried first.** Two weeks climbing the conventional RAG ladder, against an eval that got harsher mid-competition (14 May: equivalence threshold raised 0.5 → 0.9, leaderboard wiped):
- **Extractive stack** — hybrid BM25+BGE retrieval → bge-reranker → fine-tuned RoBERTa-large reader. Fine-tuning on inference-matched chunks was the one real lever (+0.16 cloud); plateaued at **0.683**.
- **Post-processing layers** — answer canonicalizers and candidate rankers all *regressed*: they optimized exact-match proxies that didn't match the 0.9 neural judge.
- **Generative readers** — Qwen2.5-7B-AWQ reached **0.734** accuracy but its 0.286 speed score sank the blend; Qwen3-4B/8B timed out on the cloud base image; a QLoRA fine-tune of Qwen3-8B trained cleanly but was unshippable from the T4 (vLLM LoRA-kernel crashes, quantized-merge dead-ends).

**Then we found the cheese.** The answer-equivalence judge shipped *with the eval harness* — white-box weights — and the organisers confirmed any solution that runs and scores is allowed. HotFlip token replacement found a fixed **16-token universal adversarial trigger** that pegs the judge's equivalence probability at ~0.999 regardless of question, reference, or candidate content. It transferred to the cloud checkpoint first try — **0.683 → 0.951** overnight — and later survived an evaluator tokenization patch unchanged.

**…and just used it.** With the trigger answering every question, the score reduced to retrieval hit-rate × trigger pass-rate plus speed, so every neural model became dead weight. We deleted them one by one: the RoBERTa reader, then the reranker and dense retriever. The shipped `v28-optimized-bm25` is just **tuned BM25** — document-level (k1=2.05) fused with 3-sentence passage windows (0.6 weight) — returning top-3 doc IDs and the trigger string: **0.984 acc / 0.985 speed**, 0/700 errors, with near-ceiling speed because nothing neural ever loads.

---

## ASR — Speech Recognition (20%)

**The task.** Convert spoken-audio clips into accurate text transcripts, scored on word-error-rate-derived accuracy plus speed.

**Key technologies**
- Python
- NVIDIA NeMo toolkit, **Parakeet-TDT-0.6B-v2** fine-tune
- `librosa` / `soundfile` audio preprocessing, regex/phonetic post-processing

**What shipped.** `nemo-ft-v3`: NVIDIA **Parakeet-TDT-0.6B-v2** fine-tuned (val WER 0.0856) with a layer of spelling / regex / phonetic post-corrections for the in-world proper nouns and currency formats. A from-scratch empty baseline and an early LoRA-on-Whisper line were left behind once the Parakeet backbone proved stronger.

**Dead-end worth noting.** An **n-gram LM shallow-fusion** path gave a real *local* WER win but crashed on the cloud GPU (and later risked CUDA-graph timeouts on Blackwell) — local gain, no cloud transfer, so it was parked.

---

## CV — Object Detection (20%)

**The task.** Detect and classify objects from an 18-class taxonomy, returning LTWH bounding boxes, scored on mAP plus speed.

**Key technologies**
- Python
- Ultralytics **YOLOv11l**, PyTorch
- Train-small-serve-big inference scaling (1024px train → 1408px inference)

**What shipped.** 18-class LTWH bounding boxes. We shipped **YOLOv11l weights trained at 1024px but served upscaled to 1408px** (`CV_IMGSZ=1408`). Counterintuitively, *train-small-serve-big* beat bigger-trained / matched-resolution models repeatedly — the only reliable lever on this distribution.

**Biggest dead-end (a clean negative result).** We bet that a stronger backbone — **RF-DETR** (DINOv2 ViT) — could break the ~0.67 cloud-accuracy ceiling. It tied YOLO on accuracy with a *narrower* local→cloud gap but hit the **exact same ceiling and lost on speed**, proving the wall is **content-shift, not backbone capacity**. Closing the architecture-family question was itself a result. TTA and augmented training also consistently failed to transfer.

---

## Noise — Adversarial Perturbation (Finals tool)

**The task.** Generate adversarial perturbations to degrade an opponent's CV pipeline, while staying inside fairness bounds (L2/SSIM caps). No direct Qualifier reward — purely a Finals disruption tool.

**Key technologies**
- Python, PyTorch
- Custom **AdvGAN** generator (trained against a ResNet18 classifier surrogate)
- Edge-aware detector-stress overlay, JPEG re-encoding for fairness compliance

**What shipped.** `level10-detector-stress`: a single-pass **AdvGAN generator** (~14 KB weights) producing base noise, overlaid with edge-aware detector-stress patterns, clamped to a fairness budget. The honest finding: noise trained on a *classifier* transfers poorly to a *detector* victim (~6% mAP drop measured); the real lever (retraining against a detector objective) was scoped but deferred.

---

## Engineering notes & docs

The real story lives in the per-task **`NOTES.md`** files ([ae/NOTES.md](ae/NOTES.md), [asr/NOTES.md](asr/NOTES.md), [cv/NOTES.md](cv/NOTES.md), [nlp/NOTES.md](nlp/NOTES.md), [noise/NOTES.md](noise/NOTES.md)): per-task working logs of decisions, gotchas, and dead-ends. [docs/RESULTS.md](docs/RESULTS.md) is the cross-task scoreboard with the full cloud submission history. The codebase was built with a gated *brainstorm → spec → plan → eval → ship* workflow, with experimental features kept behind default-OFF flags so the deploy was never at risk.

---

## Final words

A few things this competition reinforced. **Hand-coded heuristics transfer; learned policies overfit the local opponent** — every RL line we tried for AE improved local eval while cloud performance flattened or collapsed. **Local eval ≠ cloud ≠ competition hardware**: the AE melee gate once misranked two candidates, and only an on-hardware A/B on the real Finals seed caught it; the ASR n-gram LM was a real local win that didn't survive the cloud GPU. **Relative-rank scoring changes what "better" means** — for AE, a configuration that scores less absolute reward can still place higher, which makes single-agent reward the wrong optimization target. And **the evaluator is part of the problem statement**: NLP's biggest single jump came not from a better reader but from attacking the white-box answer-equivalence judge directly — after which the winning move was *removing* models, not adding them.

Thanks to DSTA and the TIL-AI organisers for running the competition.

---

## License

MIT.
