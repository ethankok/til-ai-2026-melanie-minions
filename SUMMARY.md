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

## Where each task stands (16 May 2026)

For task-specific history, decisions, gotchas, and next-step plans, open the task's NOTES.md.

| Task | Latest shipped tag | Acc/reward | Speed | Status | Working log |
|---|---|---:|---:|---|---|
| ASR | `nemo-zs` | 0.956 | 0.946 | **NEW BLENDED HIGH.** Backbone switch from distil-whisper LoRA to NVIDIA Parakeet-TDT-0.6B-v2 zero-shot. Same accuracy, +0.097 speed → blended +0.025. CUDA-graph fast path (`cuda-python>=12.3`) staged in for `nemo-zs-v2`. | [asr/NOTES.md](asr/NOTES.md) |
| AE | `hybrid-v3` | 0.555 | 0.849 | **UN-PARKED for inference-side `mcts-light-v1`.** `ppo-selfplay-v2` shipped 18:09 at **0.436/0.857** (REGRESSED -0.119 vs hybrid-v3), falsifying the opponent-diversity training hypothesis. New candidate avoids more training: bounded tactical lookahead inside `AEManager`, with hybrid mode trusting it before the neural policy when it finds high-value bomb/base lines. Build tag: `mcts-light-v1`; hybrid-v3 remains protected by leaderboard highest-score retention. | [ae/NOTES.md](ae/NOTES.md) |
| NLP | `v9-doc-ensemble` | 0.683 | 0.886 | **UN-PARKED for `v13b-deberta` (QA retune).** Candidate-ranker arch eliminated 16/05: v13a-ae-ranker val top-1 0.384, v13a-heuristic local 0.663 — both lose to v9's 0.711. But the v13a oracle was 0.814 → the answer-form problem is QA-head not post-processing. Post-v9 swings (v10/v11/v8a/v12) all regressed; the unbroken path is extractor retrain. Same v8b chunked-context recipe, swap base to `deepset/deberta-v3-large-squad2` (structurally +1-2% over roberta-large on extractive QA). Manager + Dockerfile patched to prefer DeBERTa weights when present, fall back to v9 weights otherwise. | [nlp/NOTES.md](nlp/NOTES.md) |
| CV | `cv-yolo-v2-tier1-best` | 0.556 | 0.956 | **RE-PARKED 16/05 19:00.** Phase C.1 augmented training (JPEG aug + native tile crops) shipped twice: `cv-augc1-v4` 0.553/0.962 (mismatched config), `cv-augc1-v4-1280` 0.553/0.959 (matched). Hard held-out lifted +0.04 (0.948 vs 0.905) but cloud flat — gap WIDENED from 0.349 to 0.395 — augmentation overfit to small-object-dense distribution. v8s/v11m family confirmed at-ceiling near 0.556 cloud. 0.7 needs backbone change (RT-DETR/YOLOv9) or wholesale distribution recipe; neither high-EV at deadline. Tier1 stays. | [cv/NOTES.md](cv/NOTES.md) |
| Noise | `latest` | 1.000 | 0.970 | Qualifier-safe. Official spec says no direct Qualifier reward. | [noise/NOTES.md](noise/NOTES.md) |

---

## Rough outline (what to do, in order)

**Week 0 — plumbing (done)**

1. Workbench access for everyone ✅
2. All 5 services build, `til test` runs clean, JSON shape correct ✅
3. All 5 baselines submitted ✅

**Week 1 — first real models (in progress)**

4. **AE** (40%) — `hybrid-v3` is still the shipped best at **0.555/0.849**. Training-side swings after it failed (`bc-belief-hybrid`, `hybrid-conf50`, `ppo-selfplay-v1/v2`), so the current AE push is **`mcts-light-v1`**: inference-time tactical lookahead for known bomb/base/enemy lines, enabled in `ae/Dockerfile` with `AE_MCTS=1`. Ignore speed for this A/B; the question is whether explicit combat lookahead can move cloud reward above 0.555.
5. **NLP** (20%) — `v9-doc-ensemble` HARD-PARKED at 0.683/0.886. `v12-candidate-ranker` was submitted 16/05 13:48 and REGRESSED to 0.642/0.829 (-0.041 acc, -0.057 speed) — worst since v5b. Local was already -0.048 vs v9; the bucket shift exact -69 / substr +54 / diff +15 confirmed the ranker promoted doc-mined short tokens that passed the exact/substr training proxy but failed the 0.9 ModernBERT AE threshold on cloud. Same root cause as v11 (training on a proxy that doesn't match the cloud scorer) at 10× the surface area, hence 10× the regression. Every post-v9 swing has regressed monotonically (v10 -0.001, v11 -0.004, v8a -0.040, v12 -0.045). **NLP architecture is at its ceiling on this corpus; v9 is final.**
6. **CV** (20%) — **RE-PARKED 16/05 19:00 after Phase C.1 didn't transfer.** `cv-augc1-v4` and `cv-augc1-v4-1280` both shipped at **0.553/0.96** (essentially tied with tier1's 0.556). Augmented training (JPEG aug + native-res tile crops) lifted hard held-out by +0.04 (0.948 vs 0.905) but cloud was flat and the local→cloud gap **widened** from 0.349 to 0.395. Augmentation overfit to the small-object-dense distribution rather than generalizing. v8s/v11m family is confirmed at-ceiling near cloud 0.556; 0.7 requires a backbone change (RT-DETR / YOLOv9) or wholesale distribution-shift recipe, neither high-EV at deadline. Tier1 stays shipped.
7. **ASR** (20%) — `nemo-zs` shipped at 0.956/0.946 (Parakeet-TDT-0.6B-v2 zero-shot, +0.097 speed over `ft-lora32-v1`). cuda-python pinned for `nemo-zs-v2`.

**Week 2 — push scores**

8. **AE**: `mcts-light-v1` is the current code candidate. It is not another local-score policy; it searches short tactical futures from the live belief map and only overrides when projected bomb/base combat value clears threshold. Workbench run: restore ppo-v1 weights, `echo hybrid > ae/src/.ae_mode`, `til build ae mcts-light-v1`, `til test ae mcts-light-v1`, `til submit ae mcts-light-v1`.
9. **NLP**: `v9-doc-ensemble` remains the shipped baseline at 0.683/0.886 (blended 0.734). The next attempt is `v12-candidate-ranker`: top RoBERTa spans + rules + canonicalized/doc-mined candidates, with optional `nlp/models/answer_ranker.json`. Submit only if local equiv_rate beats v9 or exact/substr proxy improves with retrieval misses flat. Full generative QA remains dead as a primary path.
10. **CV**: RE-PARKED at tier1 (0.556/0.956) after Phase C.1 ran and didn't transfer. The augmented training (JPEG aug + 1024×1024 native-res tile crops, scale=0.80, copy_paste=0.10, 80 epochs) lifted hard held-out small AP from 0.746 → 0.779 and total mAP from 0.905 → 0.948 — the training worked on its target. But cloud landed at 0.553 both at mismatched config (imgsz=768 Dockerfile default) and matched config (imgsz=1280 baked in) — gap widened from 0.349 to 0.395. The augmentation specialized the model further from cloud's mixed distribution, not closer. v8s/v11m family is at-ceiling. Pivot options if revisited: RT-DETR-L backbone transfer (unknown gap, ~6h GPU, high variance), conservative re-augmentation (JPEG only, no tile crops; realistic cap 0.55-0.58), or accept the ceiling. Default: tier1 stays, hours go to AE / NLP / ASR.
11. **Noise**: bounded FGSM perturbation if time and the score actually matters.

**Always**

- Re-run `til test` after every change.
- Don't commit model weights to git — bake into the Docker image build.
- If a change makes a container fail to start, **revert immediately**. A working bad model > broken good model.
- Leaderboard keeps the **higher** score, so a regressing submission can't demote a peak. Submission slots aren't capped, so when in doubt — ship and observe; the cloud signal is the only one that pays.

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
