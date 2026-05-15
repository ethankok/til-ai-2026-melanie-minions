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
| ASR | `nemo-zs` | 0.956 | 0.946 | **NEW BLENDED HIGH.** Backbone switch from distil-whisper LoRA to NVIDIA Parakeet-TDT-0.6B-v2 zero-shot. Same accuracy, +0.097 speed → blended +0.025. CUDA-graph fast path (`cuda-python>=12.3`) staged in for `nemo-zs-v2`. | [asr/NOTES.md](asr/NOTES.md) |
| AE | `hybrid-v3` | 0.555 | 0.849 | Team best (top-quartile; top is 0.711). Belief-map BC retrain (`bc-belief-hybrid`, 15/05) regressed to 0.287 — memory hypothesis rejected, rolled back. Realistic remaining AE ceiling ~0.58. | [ae/NOTES.md](ae/NOTES.md) |
| NLP | `v7-finetuned-v1` | 0.517 | 0.880 | **NEW HIGH (+0.034 vs v5c).** Fine-tuned roberta-large-squad2 on local nlp.jsonl (353/883 examples). Local 0.709 → cloud 0.517 transferred ~1:1. Blended 0.608. v2 data-prep (variants + rapidfuzz, 431/883 retained) testing locally; v8-genqa / chunked-context if v2 plateaus. | [nlp/NOTES.md](nlp/NOTES.md) |
| CV | `cv-yolo-v2-tier1-best` | 0.556 | 0.956 | Tier1 STILL ON LEADERBOARD. v11m@1024 regressed to 0.376. imgsz=1280 sweep: hard held-out 0.9141 (+0.009 vs tier1) but TTA too slow. aug=0 at 1280 is the next test. | [cv/NOTES.md](cv/NOTES.md) |
| Noise | `latest` | 1.000 | 0.970 | Qualifier-safe. Official spec says no direct Qualifier reward. | [noise/NOTES.md](noise/NOTES.md) |

---

## Rough outline (what to do, in order)

**Week 0 — plumbing (done)**

1. Workbench access for everyone ✅
2. All 5 services build, `til test` runs clean, JSON shape correct ✅
3. All 5 baselines submitted ✅

**Week 1 — first real models (in progress)**

4. **AE** (40%) — `hybrid-v3` is now team best at **0.555/0.849** (third consecutive AE high; +0.048 over ppo-v1). Builds on hybrid-v2's policy + heuristic safety-veto with two more lifts: a top-K policy cascade (try policy's #2/#3 actions when #1 is vetoed before falling back to heuristic) and an opportunistic-kill shortcut in the heuristic's dominant-action path (bomb adjacent enemy *agents* sighted this step, not just adjacent enemy *bases*). Top of leaderboard is 0.711; the heuristic-side ceiling is probably near. Next big swing if needed: state-augmented policy retrain (pass full 16×16 belief map as additional CNN input).
5. **NLP** (20%) — `v7-finetuned-v1` is now shipped at 0.517/0.880 (+0.034 vs v5c, NEW HIGH). Fine-tuned `roberta-large-squad2` on local `nlp.jsonl`; pipeline-side bisect from v4 → v5c established the floor before the model swap. v2 data-prep + retrain in flight.
6. **CV** (20%) — `cv-yolo-v2-tier1-best` (0.556/0.956) STILL ON LEADERBOARD. Tier 2 `cv-yolo11m-v3-pre` (YOLOv11m@1024 fully trained 120ep) submitted 15/05 → 0.376/0.955 REGRESSED. Local val 0.937 looked great but hard held-out only 0.8673 (-0.038 vs tier1's 0.9049); small-AP collapsed (0.587 vs 0.746). Lesson: bigger model + matched-imgsz lost to smaller model + upscaled-imgsz inference on this dataset. Untested lever: v11m@1280 inference sweep.
7. **ASR** (20%) — `nemo-zs` shipped at 0.956/0.946 (Parakeet-TDT-0.6B-v2 zero-shot, +0.097 speed over `ft-lora32-v1`). cuda-python pinned for `nemo-zs-v2`.

**Week 2 — push scores**

8. **AE**: hybrid-v3 shipped at 0.555/0.849. Belief-map state augmentation (16×16×11 CNN branch) was tested — BC val_acc 0.897 (+0.023 vs bc-v1), but `bc-belief-hybrid` cloud crashed to 0.287 (-0.268 vs hybrid-v3). Memory hypothesis rejected: bigger model + richer state input found *more* spurious correlations to the random-opponent local distribution, widening the gap. Rolled back to hybrid-v3. Remaining AE moves all have realistic ceiling ~0.58; the higher-EV qualifier lever is NLP v7-finetune (already shipped at 0.517/0.880).
9. **NLP**: `v7-finetuned-v1` shipped at 0.517/0.880, fine-tuned `roberta-large-squad2` on local `nlp.jsonl`. v2 data-prep (variants + flexible regex + rapidfuzz) retrained, 431/883 retained vs v1's 353 — testing locally before submit. If v2 plateaus, next paths are `v8-genqa` (Flan-T5 generative for full-coverage of paraphrased answers) or `v8-chunked-context` (train QA on top-BGE-chunk instead of full doc). Confirmed regressors: paragraph chunking, low-confidence sentence fallback.
10. **CV**: Tier 1 `cv-yolo-v2-tier1-best` STILL on leaderboard at 0.556/0.956. Tier 2 `cv-yolo11m-v3-pre` (YOLOv11m@1024, 120 epochs) regressed to 0.376/0.955 — matched-imgsz inference lost to v8s+upscaled. imgsz=1280 sweep showed hard held-out 0.9141 (+0.009 vs tier1) but TTA at 1280 is too slow (speed ~0.71). **Next: aug=0 sweep at 1280** — if mAP holds ≥0.88 without TTA, blended beats tier1 and we ship.
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
