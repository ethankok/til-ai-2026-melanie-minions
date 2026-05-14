# PLEASEEEREAD — TIL-AI 2026

We need to submit **5 separate Docker services** by:

**24 May 2026, 11:59:59 PM SGT**

The repo looks huge, but most of it is official scaffolding. For now, we mainly edit **5 manager files**.

This file is cross-cutting orientation. **Per-task history, decisions, and current state live in `<task>/NOTES.md`** — read that for the task you're working on. Submission scores live in [RESULTS.md](RESULTS.md).

---

## The only files that matter right now

```text
SUMMARY.md                   read this (orientation + strategy)
RESULTS.md                   leaderboard / submission scores
asr/NOTES.md                 ASR working log
cv/NOTES.md                  CV working log
nlp/NOTES.md                 NLP working log
ae/NOTES.md                  AE working log
noise/NOTES.md               Noise working log

asr/src/asr_manager.py       ASR inference code
cv/src/cv_manager.py         CV inference code
noise/src/noise_manager.py   Noise code
nlp/src/nlp_manager.py       NLP retrieval code
ae/src/ae_manager.py         AE agent code
```

I also patched this once for reset robustness:

```text
ae/src/ae_server.py
```

Don't edit the rest unless we specifically need to.

---

## What the important folders/files mean

```text
asr/       speech recognition task
cv/        object detection task
noise/     adversarial image noising task
nlp/       RAG question-answering task
ae/        autonomous exploration / Bomberman agent task

test/      official local evaluators used by til test
README.md  official starter repo instructions
requirements-dev.txt  dev dependencies
.gitmodules  submodule config, don't touch

til-26-ae/      official AE environment, don't edit
til-26-finals/  official finals code, don't edit
```

Inside each task folder:

```text
NOTES.md          OUR working log — history, decisions, gotchas
src/*_manager.py  OUR model/inference logic — edit this
src/*_server.py   HTTP wrapper — usually don't edit
Dockerfile        packaging for submission
requirements.txt  task-specific Docker deps
README.md         task input/output spec — reference only
```

Mental model:

```text
server.py receives request -> manager.py returns prediction
```

Plain English:

- The evaluator sends data to our Docker container using HTTP.
- `*_server.py` handles the web/API part.
- `*_manager.py` is where our actual AI/model/rules go.
- The server calls the manager once per input and wraps the answer into JSON.
- If the JSON shape is wrong, we can get 0 even if the model is smart.

What a "baseline" means:

- A baseline is a simple working version.
- It is not meant to win.
- It proves the service starts, accepts input, and returns valid output.
- Once baselines work, we replace the dumb logic with better models one task at a time.

---

## Scoring

Authoritative source: [official challenge specifications](https://github.com/til-ai/til-26/wiki/Challenge-specifications). If local READMEs conflict with that page, the Wiki wins.

Overall qualifier weights:

```text
AE     40%
ASR    20%
CV     20%
NLP    20%
Noise  not directly rewarded in Qualifiers; relevant for Finals CV disruption
```

Each scored challenge is:

```text
75% accuracy/reward
25% speed
speed = 1 - min(t_elapsed, t_max) / t_max
Qualifier t_max = 30 minutes for the full test set
```

Task ports/routes:

```text
ASR    /asr    port 5001
CV     /cv     port 5002
Noise  /noise  port 5003
NLP    /nlp    port 5004
AE     /ae     port 5005, plus /reset
```

Important terms:

- **Docker service**: a packaged mini-app for one task. Runs a web server, waits for inputs.
- **Endpoint/route**: URL path the evaluator calls, e.g. `/asr` or `/nlp`.
- **Port**: network number the service listens on (see table above).
- **Schema**: exact input/output JSON format. Matching schema is non-negotiable.
- **GCP Workbench**: official Google Cloud machine where data + submission tools live.
- **`til build` / `til test` / `til submit`**: build, locally evaluate, and submit a Docker image.

---

## Where each task stands (14 May 2026)

For task-specific history, decisions, gotchas, and next-step plans, open the task's NOTES.md.

| Task | Latest shipped tag | Acc/reward | Speed | Status | Working log |
|---|---|---:|---:|---|---|
| ASR | `ft-lora32-v1` | 0.957 | 0.849 | Crossed 0.95 target. Parked; int8 speed re-export regressed accuracy. | [asr/NOTES.md](asr/NOTES.md) |
| AE | `hybrid-v2` | 0.545 | 0.863 | **NEW HIGH.** Hybrid manager (policy + heuristic safety-veto) broke the 0.49–0.51 cloud ceiling 5 prior approaches hit; +0.038 over ppo-v1. | [ae/NOTES.md](ae/NOTES.md) |
| NLP | `v4-dict-id` | 0.483 | 0.888 | New-eval recovery after organiser format bug; next gains are answer quality and QA latency. | [nlp/NOTES.md](nlp/NOTES.md) |
| CV | `cv-yolo-v2-tier1-best` | 0.556 | 0.956 | NEW HIGH (+0.007 vs v2-best). Tier 1 inference-only sweep landed; Tier 2 (YOLOv11m@1024 retrain) in flight to target 0.70. | [cv/NOTES.md](cv/NOTES.md) |
| Noise | `latest` | 1.000 | 0.970 | Qualifier-safe. Official spec says no direct Qualifier reward. | [noise/NOTES.md](noise/NOTES.md) |

---

## Rough outline (what to do, in order)

**Week 0 — plumbing (done)**

1. Workbench access for everyone ✅
2. All 5 services build, `til test` runs clean, JSON shape correct ✅
3. All 5 baselines submitted ✅

**Week 1 — first real models (in progress)**

4. **AE** (40%) — `hybrid-v2` is now team best at **0.545/0.863** (+0.038 over ppo-v1). The hybrid manager combines policy action selection with a heuristic safety-veto on illegal / unsafe-bomb / step-into-blast / frozen-stay actions; this is the first AE approach to break the 0.49–0.51 cloud ceiling that ppo-v1, ppo-v2, planner-v3b, and bc-v1 all hit. Local→cloud gap is still ~0.23 but the floor moved up. Next moves are env-var toggles on the same hybrid (`AE_HYBRID_CONF` / `AE_HYBRID_VETO_*`).
5. **NLP** (20%) — `v4-dict-id` recovered to 0.483/0.888 after organiser eval-format fix; next work is quality/latency rather than schema.
6. **CV** (20%) — `cv-yolo-v2-tier1-best` shipped at 0.556/0.956 (new high, +0.007 vs v2-best) using v2-best weights with TTA + imgsz=896 + score field. Tier 2 YOLOv11m@1024 retrain in flight to target 0.70.
7. **ASR** (20%) — `ft-lora32-v1` shipped at 0.957. Parked; int8 re-export already proved low-ROI.

**Week 2 — push scores**

8. **AE**: hybrid-v2 shipped at 0.545/0.863; further lift via veto-tuning A/Bs (`AE_HYBRID_CONF=0.5`, drop one veto at a time), or — longer path — retrain the policy *aware* of the heuristic safety net (PPO with vetoes in the rollout loop). Speed is evaluator-bound at ~0.86, not worth further work.
9. **NLP**: hybrid retrieval (BM25 + dense embeddings) + extractive QA model on top-k.
10. **CV**: Tier 1 SHIPPED at 0.556/0.956 (+0.007 vs v2-best) — same weights, env-only changes (`CV_IMGSZ=896`, `CV_IOU=0.60`, `CV_AUGMENT=1`, `CV_HALF=1`, emit real `score`). Tier 2 = YOLOv11m@1024 retrain with copy-paste/mosaic for small-object/aircraft-subclass generalization (`training/cv/train_v3.sh`, in flight ~4-5h on T4). `imgsz=1024` confirmed bad with v2-best 768-trained weights (resolution mismatch).
11. **Noise**: bounded FGSM perturbation if time and the score actually matters.

**Always**

- Re-run `til test` after every change.
- Don't commit model weights to git — bake into the Docker image build.
- If a change makes a container fail to start, **revert immediately**. A working bad model > broken good model.
- Leaderboard keeps the **higher** score, so a regressing submission can't demote a peak. But a regressing submission *for free* is still a waste of time and submission slot.

Why this order:

- AE is 40% — every hour spent there is worth ~2× the same hour on a 20% task, but naive BC already failed, so PPO must solve distribution mismatch instead of just imitate the planner.
- NLP's lexical baseline should jump with retrieval upgrades alone, and the official schema gives partial credit (`0.4`) for successful retrieval even when answer equivalence fails.
- CV's v2 YOLOv8s path is now clean and materially better; remaining CV gap is likely hidden distribution/small-object/aircraft-subclass generalization, not LTWH formatting.
- ASR has crossed its target; marginal hour is low-ROI now.
- Noise has no direct qualifier reward per the official spec; do it last unless preparing for Finals.

What "done for now" means for a task:

```text
1. service starts
2. endpoint responds
3. output JSON has correct shape
4. Docker builds on GCP
5. til test runs without crashing
6. til submit succeeds
```

---

## Who edits what

```text
ASR owner:   asr/src/asr_manager.py
CV owner:    cv/src/cv_manager.py
Noise owner: noise/src/noise_manager.py
NLP owner:   nlp/src/nlp_manager.py
AE owner:    ae/src/ae_manager.py
```

If unsure, don't edit server/Docker/submodule files yet.

---

## Local setup

Check imports:

```bash
python - <<'PY'
import sys
sys.path.extend(['asr/src','cv/src','nlp/src','noise/src','ae/src'])
from asr_manager import ASRManager
from cv_manager import CVManager
from nlp_manager import NLPManager
from noise_manager import NoiseManager
from ae_manager import AEManager
print('ok')
PY
```

---

## GCP setup (Workbench-only)

Official loop:

```bash
til build <task> <tag>
til test  <task> <tag>
til submit <task> <tag>
```

Repeat for: `asr cv noise nlp ae`.

If `til` errors with "Could not find directory for task ...", set `TIL_FOLDER`:

```bash
echo 'export TIL_FOLDER=/home/jupyter/til' >> ~/.bash_profile
export TIL_FOLDER=/home/jupyter/til
```

If `til submit` errors with "Unauthenticated request" on the Docker push, register the gcloud credential helper once:

```bash
gcloud auth configure-docker asia-southeast1-docker.pkg.dev
```

---

## Do not touch unless needed

```text
til-26-ae/
til-26-finals/
.gitmodules
<task>/src/<task>_server.py
<task>/Dockerfile
<task>/requirements.txt
```

Do not commit large model weights to Git.
