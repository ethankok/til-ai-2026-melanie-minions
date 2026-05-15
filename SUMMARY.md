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

## Where each task stands (15 May 2026)

For task-specific history, decisions, gotchas, and next-step plans, open the task's NOTES.md.

| Task | Latest shipped tag | Acc/reward | Speed | Status | Working log |
|---|---|---:|---:|---|---|
| ASR | `nemo-zs` | 0.956 | 0.946 | **NEW BLENDED HIGH.** Backbone switch from distil-whisper LoRA to NVIDIA Parakeet-TDT-0.6B-v2 zero-shot. Same accuracy, +0.097 speed → blended +0.025. CUDA-graph fast path (`cuda-python>=12.3`) staged in for `nemo-zs-v2`. | [asr/NOTES.md](asr/NOTES.md) |
| AE | `hybrid-v3` | 0.555 | 0.849 | Team best (top-quartile; top is 0.711). Belief-map BC retrain (`bc-belief-hybrid`, 15/05) regressed to 0.287 — memory hypothesis rejected, rolled back. Realistic remaining AE ceiling ~0.58. | [ae/NOTES.md](ae/NOTES.md) |
| NLP | `v9-doc-ensemble` | 0.683 | 0.883 | **BEST SUBMITTED BLEND.** `v11-canonical-answer` submitted 15/05 21:26+21:39, both `0.680` (speeds 0.881/0.873) — REGRESSED -0.003; replay +10 didn't transfer through 0.9 AE threshold. Replay had been against the OLD (pre-14-May) local rubric; upstream test_nlp.py synced 16/05 is now the proper ship gate. | [nlp/NOTES.md](nlp/NOTES.md) |
| CV | `cv-yolo-v2-tier1-best` | 0.556 | 0.956 | Pass A on hard held-out: gap is small/medium-AP localization, **not** class confusion (16/3334 wrong, 0.48%), and **not** low-conf FP cleanup (raising `CV_CONF` regresses mAP monotonically). Tiled-inference A/B: `3x2` lifts hard held-out to 0.9009 (+0.006) but only on medium AP, and the 3× compute breaks blended. **`v8s-1024` retrain in progress** (16/05) — same backbone as tier1 but trained at 1024 with `copy_paste=0.40` to attack the small/medium-AP gap directly. | [cv/NOTES.md](cv/NOTES.md) |
| Noise | `latest` | 1.000 | 0.970 | Qualifier-safe. Official spec says no direct Qualifier reward. | [noise/NOTES.md](noise/NOTES.md) |

---

## Rough outline (what to do, in order)

**Week 0 — plumbing (done)**

1. Workbench access for everyone ✅
2. All 5 services build, `til test` runs clean, JSON shape correct ✅
3. All 5 baselines submitted ✅

**Week 1 — first real models (in progress)**

4. **AE** (40%) — `hybrid-v3` is now team best at **0.555/0.849** (third consecutive AE high; +0.048 over ppo-v1). Builds on hybrid-v2's policy + heuristic safety-veto with two more lifts: a top-K policy cascade (try policy's #2/#3 actions when #1 is vetoed before falling back to heuristic) and an opportunistic-kill shortcut in the heuristic's dominant-action path (bomb adjacent enemy *agents* sighted this step, not just adjacent enemy *bases*). Top of leaderboard is 0.711; the heuristic-side ceiling is probably near. Next big swing if needed: state-augmented policy retrain (pass full 16×16 belief map as additional CNN input).
5. **NLP** (20%) — `v9-doc-ensemble` is shipped at 0.683/0.883. Whole-document retrieval helped a little (local misses 40→37), but the persistent gap is answer syntax/equivalence, not raw retrieval: retrieved-but-different answers stayed ~395-396 cases. `v10-template-lite` was safe but neutral; `v11-canonical-answer` was submitted and REGRESSED to 0.680/0.881 — the +10 replay was scored against the OLD pre-14-May rubric so it did not transfer through the 0.9 AE threshold. Deterministic regex/canonicalization is exhausted as a lever; the next swing is `v8a-genqa` (Flan-T5 generative) if at all.
6. **CV** (20%) — `cv-yolo-v2-tier1-best` (0.556/0.956) STILL ON LEADERBOARD. Pass A failure analysis on hard held-out (16/05) confirmed the gap is structural, not plumbing: small AP `0.6434` is the bottleneck, class confusion is only `16/3334` boxes (0.48%), and raising `CV_CONF` regresses mAP monotonically (the PR-curve tail is doing real work). Tiled inference A/B (`2x2`/`2x1`/`3x2`) shipped behind `CV_TILE_MODE` env in `cv_manager.py`; `3x2` lifted hard held-out to 0.9009 (+0.006) but only on medium AP, and 3× compute breaks blended. **`v8s-1024` retrain running (16/05)** — same v8s backbone as tier1, imgsz 768→1024, `copy_paste` 0.10→0.40. Hypothesis: training at 2× pixel area gives the model the resolution to localize medium/small targets at IoU ≥ 0.75, the exact gap Pass A identified.
7. **ASR** (20%) — `nemo-zs` shipped at 0.956/0.946 (Parakeet-TDT-0.6B-v2 zero-shot, +0.097 speed over `ft-lora32-v1`). cuda-python pinned for `nemo-zs-v2`.

**Week 2 — push scores**

8. **AE**: hybrid-v3 shipped at 0.555/0.849. Belief-map state augmentation (16×16×11 CNN branch) was tested — BC val_acc 0.897 (+0.023 vs bc-v1), but `bc-belief-hybrid` cloud crashed to 0.287 (-0.268 vs hybrid-v3). Memory hypothesis rejected: bigger model + richer state input found *more* spurious correlations to the random-opponent local distribution, widening the gap. Rolled back to hybrid-v3. Remaining AE moves all have realistic ceiling ~0.58; the higher-EV qualifier lift has now come from NLP `v8b/v9`.
9. **NLP**: `v9-doc-ensemble` shipped at 0.683/0.883. Protect this as the submitted baseline. `v11-canonical-answer` submitted twice (15/05 21:26 + 21:39) and REGRESSED to 0.680/0.881 + 0.680/0.873 — accuracy stable across both runs so the -0.003 is real. The +10 replay was against the OLD pre-14-May rubric (threshold 0.5, no partial credit); under the actual 0.9 ModernBERT threshold the canonicalized rewrites either weren't sampled or didn't pass equivalence. test_nlp.py has been synced to upstream (16/05) so future replays will use the correct rubric. Deterministic answer rewriting at top-3 doc scope is now confirmed too narrow. Next swing if any: `v8a-genqa` (Flan-T5 generative on all 883 triples), with speed and 0.9-threshold risk front and center. Confirmed regressors: paragraph chunking, broad low-confidence sentence fallback, noisy rapidfuzz span training, full-doc deterministic canonicalization.
10. **CV**: Tier 1 `cv-yolo-v2-tier1-best` STILL on leaderboard at 0.556/0.956. Pass A failure analysis (16/05) ruled out class confusion (16/3334 boxes wrong) and low-conf FP cleanup (mAP regressed monotonically with higher `CV_CONF`); the gap is small/medium-AP at high IoU. Tiled inference (new `CV_TILE_MODE` env in `cv_manager.py`) tested locally — `3x2` lifts hard held-out by +0.006 total but only on medium AP, with 3× the compute cost, so it loses on blended. **`v8s-1024` retrain running**: same backbone as tier1, imgsz 768→1024, `copy_paste=0.40`. Eval all four modes (`1024 aug=1`, `1280 aug=0`, `tile=3x2`, `tile=2x2`) against the new weights when training finishes; ship the best-blended candidate. Tier1 stays on leaderboard regardless via highest-score retention.
11. **Noise**: bounded FGSM perturbation if time and the score actually matters.

**Always**

- Re-run `til test` after every change.
- Don't commit model weights to git — bake into the Docker image build.
- If a change makes a container fail to start, **revert immediately**. A working bad model > broken good model.
- Leaderboard keeps the **higher** score, so a regressing submission can't demote a peak. But a regressing submission *for free* is still a waste of time and submission slot.

Why this order:

- AE is 40% — every hour spent there is worth ~2× the same hour on a 20% task, but naive BC already failed, so PPO must solve distribution mismatch instead of just imitate the planner.
- NLP's lexical baseline should jump with retrieval upgrades alone, and the official schema gives partial credit (`0.4`) for successful retrieval even when answer equivalence fails.
- CV's v2 YOLOv8s path is the right backbone; Pass A on hard held-out (16/05) showed remaining CV gap is small/medium-AP localization at IoU ≥ 0.75. Aircraft-subclass confusion and low-conf FP cleanup are confirmed dead as hypotheses. The v8s-1024 retrain attacks the resolution axis directly.
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
