# TIL-AI 2026 — Team *melanie-minions*

Our competition entry for **DSTA BrainHack TIL-AI 2026**. The challenge ships as five independent ML problems, each packaged as its own Dockerized FastAPI microservice. An official evaluator feeds inputs over the wire; our containers return scored JSON. We took this stack from the online **Qualifiers** through to the in-person **Semifinals and Finals at Marina Bay Sands**, where all six services (the five tasks plus a WebSocket orchestration server) run together in one Docker Compose stack on competition hardware.

This repo is a public showcase of the engineering: what we shipped, what we tried, and — honestly — what didn't work. Most of the depth went into **AE** (40% of the score), an autonomous-agent bomb-and-resource game where hand-coded heuristics beat every learned policy we threw at it.

> Each task service follows the same shape: `<task>/src/<task>_manager.py` holds *our* logic, wrapped by a thin `<task>_server.py` FastAPI layer. We edit the manager; the server and Dockerfile are mostly fixed.

---

## Results at a glance

| Task | Weight | What we shipped | Score / placement |
|------|--------|-----------------|-------------------|
| **AE** (autonomous exploration / RL) | 40% | Dijkstra heuristic core + confidence-gated PPO consultant (`confpol-semis2b-u75`) + CEM-tuned planner weights (`g02-sample-08`) | Cloud 0.382 acc-axis / 0.746 speed; **1st place in live Semifinals matches** (≈1.5× margins on hardware) |
| **NLP** (retrieval QA) | 20% | `v28-optimized-bm25` — pure tuned BM25 hybrid retrieval, no neural models | **0.984 acc / 0.985 speed** (blended 0.984), 0/700 errors |
| **ASR** (speech recognition) | 20% | `nemo-ft-v3` — fine-tuned Parakeet-TDT-0.6B-v2 + post-processing rules | 0.970 acc / 0.947 speed (blended 0.964) |
| **CV** (object detection, 18-class LTWH) | 20% | `yolo11l-1408` — YOLOv11l, upscale-at-inference | 0.684 acc / 0.937 speed (blended 0.747) |
| **Noise** (adversarial perturbation) | Finals tool | `level10-detector-stress` — AdvGAN + detector-stress overlay | 1.000 / 0.947 speed; Finals-only CV sabotage |

*Each task is scored 75% accuracy/reward + 25% speed. Estimated blended Qualifier total ≈ 0.72.*

---

## What's in each task

### AE — Autonomous Exploration (40%, the deep end)

A partially-observable grid wargame: farm resources → build bombs → destroy enemy bases, scored on **relative placement** against other teams in a melee, not absolute reward.

**What shipped** is a hybrid. The core is a hand-coded **Dijkstra-based greedy scorer** (item / mission / base-destruction values, base-tether defense, calibrated bomb cost) — heuristics, because they *transfer*. On top sits a **confidence-gated PPO consultant** (`confpol-semis2b-u75`): a policy trained only on the heuristic's low-confidence ticks, consulted only when its margin clears a threshold. It was warm-started from a native checkpoint and fine-tuned on a **foreign-opponent curriculum** (an external A\* bot, self-play snapshots, aggressive/anti-aggressive proxies) to learn robustness rather than mirror-match quirks. Final tuning came from a **CEM black-box search over 10 scalar planner weights** (`g02-sample-08`, e.g. tether 1.31 / base 80) — the first candidate to clear *all* of: local placement, held-out opponent-composition gap, and a cloud non-crater check. On Blackwell competition hardware the deploy went **1st place in both live Semifinals matches at ~1.5× margin**.

**Notable dead-ends (there were many):**
- **Every learned-policy line died the same death — local-opponent overfit.** Behavior cloning (val_acc 0.90 local → 0.36 cloud), self-play RL, a 704k-param belief-map CNN (local +0.023 → cloud **crashed −0.27**), tactical macro-policies — all improved local eval while cloud stayed flat at ~0.41. More parameters meant more overfit surface, not less. The lesson, echoed by competitors and the prior year's postmortem: **hand-coded heuristics ship; RL overfits.**
- **MCTS-light lookahead** timed out (2–4s/tick vs budget); even with an 80ms hard cap it regressed both speed and accuracy — the stationary-opponent assumption costs without paying.
- A key meta-lesson: **synthetic gates misrank.** Our melee gate once promoted `semis2c` over `semis2b`; an on-hardware A/B on the deterministic Finals seed reversed it (516 vs 411). On-hardware runs became the only trusted arbiter.

### NLP — Retrieval-Augmented QA (20%)

The winning move was *removing* models, not adding them. After a 14-May eval change (answer-equivalence threshold raised 0.5 → 0.9, leaderboard wiped), every fancy reader and reranker we tried hit a wall. The shipped `v28-optimized-bm25` is **pure BM25** — document-level (k1=2.05) fused with passage-level (k1=1.5, 3-sentence windows, 0.6 weight) — no neural retrieval at all. Result: **0.984 accuracy and 0.985 speed**, a Pareto win, because skipping model load gave near-ceiling speed.

**What we tried first:** a full hybrid BM25+BGE + fine-tuned RoBERTa reader (peaked ~0.68), then generative LLMs (Qwen2.5-7B hit 0.734 accuracy but **speed cost sank the blended score**), and a HotFlip **adversarial trigger** against the ModernBERT answer-equivalence judge (0.951 acc — proving the judge had exploitable features — but slower than pure retrieval). A QLoRA fine-tune of Qwen3-8B trained cleanly but was unshippable from the T4 (vLLM LoRA-kernel crashes).

### ASR — Speech Recognition (20%)

Shipped `nemo-ft-v3`: NVIDIA **Parakeet-TDT-0.6B-v2** fine-tuned (val WER 0.0856) with a layer of spelling / regex / phonetic post-corrections for the in-world proper nouns and currency formats. A from-scratch empty baseline and an early LoRA-on-Whisper line were left behind once the Parakeet backbone proved stronger.

**Dead-end worth noting:** an **n-gram LM shallow-fusion** path gave a real *local* WER win but crashed on the cloud GPU (and later risked CUDA-graph timeouts on Blackwell) — local gain, no cloud transfer, so it was parked.

### CV — Object Detection (20%)

18-class LTWH bounding boxes. Shipped **YOLOv11l weights trained at 1024px but served upscaled to 1408px** (`CV_IMGSZ=1408`). Counterintuitively, *train-small-serve-big* beat bigger-trained / matched-resolution models repeatedly — the only reliable lever on this distribution.

**Biggest dead-end (a clean negative result):** we bet that a stronger backbone — **RF-DETR** (DINOv2 ViT) — could break the ~0.67 cloud-accuracy ceiling. It tied YOLO on accuracy with a *narrower* local→cloud gap but hit the **exact same ceiling and lost on speed**, proving the wall is **content-shift, not backbone capacity**. Closing the architecture-family question was itself a result. TTA and augmented training also consistently failed to transfer.

### Noise — Adversarial Perturbation (Finals tool)

Shipped `level10-detector-stress`: a single-pass **AdvGAN generator** (~14 KB weights) producing base noise, overlaid with edge-aware detector-stress patterns, clamped to a fairness budget. No direct Qualifier reward — it's a Finals-only capability to degrade an opponent's CV inputs (~6% mAP drop measured). The honest finding: noise trained on a *classifier* transfers poorly to a *detector* victim; the real lever (retraining against a detector objective) was scoped but deferred.

---

## How the repo is organised

```
TIL/
├── ae/        AE — autonomous-exploration agent (heuristic + PPO consultant)
├── asr/       ASR — Parakeet-TDT fine-tune
├── cv/        CV  — YOLOv11l detector
├── nlp/       NLP — BM25 retrieval QA
├── noise/     Noise — AdvGAN adversarial perturbation
│   └── src/
│       ├── <task>_manager.py   ← our inference / game logic (what we edit)
│       └── <task>_server.py    ← thin FastAPI wrapper (fixed schema)
│
├── docs/      per-task NOTES.md, design specs & implementation plans, competitor learnings
├── training/  offline training & evaluation scripts (not shipped in containers)
├── test/      pytest-style per-task tests (test_<task>.py), hit the running container
│
├── til-26-ae/       submodule — official AE environment (til_environment package)
└── til-26-finals/   submodule — Finals WebSocket orchestration + local test server
```

Each task container speaks the same JSON contract used in Qualifiers, so the Finals orchestration server fans out to all five over plain HTTP with zero manager changes.

---

## Running it

```bash
# 1. Pull submodules (the --recursive matters — there's a nested one)
git submodule update --init --recursive

# 2. Python 3.13 environment
conda create --name til python=3.13 && conda activate til
pip install -r requirements-dev.txt

# 3. Sanity-check that all five managers import
python - <<'PY'
import sys; sys.path.extend(['asr/src','cv/src','nlp/src','noise/src','ae/src'])
from asr_manager import ASRManager
from cv_manager import CVManager
from nlp_manager import NLPManager
from noise_manager import NoiseManager
from ae_manager import AEManager
print('ok')
PY
```

**Build / test / submit** uses the `til` CLI (per-task) and `finals.sh` (the six-container Finals stack), both run on a GPU host with Docker:

```bash
til build <task> && til test <task> && til submit <task>   # per-task, Qualifier-style
bash finals.sh build finals --build_all                    # full Finals stack
bash finals.sh test                                        # local end-to-end match
bash finals.sh submit finals --submit_all                  # push server + all task images (:finals tag)
```

See `docs/` for the full toolchain, hardware notes (Blackwell RTX 5070 Ti vs T4), and per-batch timing details.

---

## Engineering notes & design docs

The real story lives in **`docs/`**: per-task `NOTES.md` (decisions, gotchas, dead-ends), the design specs and implementation plans behind each shipped feature, and our extracted **competitor-learnings** writeups. The codebase was built with a gated *brainstorm → spec → plan → eval → ship* workflow, with experimental features kept behind default-OFF flags so the deploy was never at risk — notes on that are in `docs/` too.

---

## License

MIT.
