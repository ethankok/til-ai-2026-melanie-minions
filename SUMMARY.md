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

## Where each task stands (18 May 2026)

For task-specific history, decisions, gotchas, and next-step plans, open the task's NOTES.md.

| Task | Latest shipped tag | Acc/reward | Speed | Status | Working log |
|---|---|---:|---:|---|---|
| ASR | `nemo-zs` | 0.956 | 0.946 | **NEW BLENDED HIGH.** Backbone switch from distil-whisper LoRA to NVIDIA Parakeet-TDT-0.6B-v2 zero-shot. Same accuracy, +0.097 speed → blended +0.025. CUDA-graph fast path (`cuda-python>=12.3`) staged in for `nemo-zs-v2`. | [asr/NOTES.md](asr/NOTES.md) |
| AE | `hybrid-v3` | 0.555 | 0.849 | **Tier 2 #9 (`ppo-scripted-v1`) SHIPPED 17/05 12:19, REGRESSED to 0.450/0.607** (blended 0.489 vs hybrid-v3 0.628 = −0.139). Local 0.741 → cloud 0.450, gap 0.291 — same structural pattern that killed every prior PPO/BC submission. Tier 1 + Tier 2 (offline playbook + opponent model + scripted PPO mode) all locally and now cloud-falsified. **Both training-side (BC, BC+belief, PPO mixed/league/selfplay/scripted) AND inference-side (mcts-light) AE hypotheses are exhausted.** Hybrid-v3 stays leaderboard-shipped throughout via highest-score retention. Only structurally different option remaining: Tier 3 #11 hierarchical goal selector (uncertain outcome, 2-3 days work). | [ae/NOTES.md](ae/NOTES.md) |
| NLP | `v14-llm-rag` (acc) / `v9-doc-ensemble` (blended) | 0.734 / 0.683 | 0.286 / 0.886 | **v15-family BLOCKED.** v9 still wins blended; v14 holds raw accuracy. The Qwen3 LoRA adapter trained cleanly (8h T4, eval_loss 0.559, mean_token_acc 87.6%), but vLLM LoRA crashes on T4/cloud and the merged-AWQ route is blocked on Workbench T4 (`autoawq` deprecated; `llm-compressor` OOM or Qwen3-GQA `NoneType` failure). v14c/v14d/v15-lora cloud failures all share the `vllm/vllm-openai` base; NGC v14 is the only cloud-proven LLM stack. | [nlp/NOTES.md](nlp/NOTES.md) |
| CV | `cv-yolo-v2-tier1-best` | 0.556 | 0.956 | **RE-PARKED 16/05 19:00.** Phase C.1 augmented training (JPEG aug + native tile crops) shipped twice: `cv-augc1-v4` 0.553/0.962 (mismatched config), `cv-augc1-v4-1280` 0.553/0.959 (matched). Hard held-out lifted +0.04 (0.948 vs 0.905) but cloud flat — gap WIDENED from 0.349 to 0.395 — augmentation overfit to small-object-dense distribution. v8s/v11m family confirmed at-ceiling near 0.556 cloud. 0.7 needs backbone change (RT-DETR/YOLOv9) or wholesale distribution recipe; neither high-EV at deadline. Tier1 stays. | [cv/NOTES.md](cv/NOTES.md) |
| Noise | `latest` | 1.000 | 0.970 | Qualifier-safe. Official spec says no direct Qualifier reward. | [noise/NOTES.md](noise/NOTES.md) |

---

## Rough outline (what to do, in order)

**Week 0 — plumbing (done)**

1. Workbench access for everyone ✅
2. All 5 services build, `til test` runs clean, JSON shape correct ✅
3. All 5 baselines submitted ✅

**Week 1 — first real models (in progress)**

4. **AE** (40%) — `hybrid-v3` is still the shipped best at **0.555/0.849**. Tier 1 + Tier 2 (offline playbook, opponent model, oracle BC) wired in 17/05 but local-falsified at 200 rounds (avg ‑0.005 vs OFF baseline). Tier 2 #9 PPO scripted SHIPPED 17/05 12:19 and **REGRESSED** to 0.450/0.607 (blended −0.139). Local→cloud gap 0.291, same structural pattern as every prior PPO/BC submission. Both training-side AND inference-side AE hypotheses are now exhausted. Tier-1 in-code keepers (#3, #6, #7) stay default ON in code; #2, #4, playbook, opponent_model default OFF. Only structurally different remaining option: Tier 3 #11 hierarchical goal selector. Otherwise AE is final at hybrid-v3.
5. **NLP** (20%) — **v15-family now blocked.** v9's extractive ceiling was structural; v14 broke it by replacing the RoBERTa-large answerer with Qwen2.5-7B-AWQ via vLLM (0.734/0.286, new accuracy high), but speed kept v9 as the blended winner. The Qwen3 route looked promising locally (`v14d` 0.755/18:33; v15 LoRA trained cleanly), then failed cloud/runtime: `v14c` and `v14d` timed out on the `vllm/vllm-openai` image, `v15-lora` returned 700/700 errors from the broken Triton LoRA kernel, and `v15-merged` cannot be AWQ-quantized on T4 with current tooling. Further NLP only makes sense via A100/H100 quantization of the merged Qwen3 model or a Qwen2.5-7B LoRA retrain on the NGC stack.
6. **CV** (20%) — **RE-PARKED 16/05 19:00 after Phase C.1 didn't transfer.** `cv-augc1-v4` and `cv-augc1-v4-1280` both shipped at **0.553/0.96** (essentially tied with tier1's 0.556). Augmented training (JPEG aug + native-res tile crops) lifted hard held-out by +0.04 (0.948 vs 0.905) but cloud was flat and the local→cloud gap **widened** from 0.349 to 0.395. Augmentation overfit to the small-object-dense distribution rather than generalizing. v8s/v11m family is confirmed at-ceiling near cloud 0.556; 0.7 requires a backbone change (RT-DETR / YOLOv9) or wholesale distribution-shift recipe, neither high-EV at deadline. Tier1 stays shipped.
7. **ASR** (20%) — `nemo-zs` shipped at 0.956/0.946 (Parakeet-TDT-0.6B-v2 zero-shot, +0.097 speed over `ft-lora32-v1`). cuda-python pinned for `nemo-zs-v2`.

**Week 2 — push scores**

8. **AE**: Tier 2 #9 PPO scripted **SHIPPED 17/05 12:19 and REGRESSED** to 0.450/0.607 (blended 0.489 vs hybrid-v3 0.628 = −0.139). Local 0.741 → cloud 0.450 = gap 0.291, same structural pattern as every prior PPO/BC submission. Hybrid-v3 (0.555/0.849) stays shipped via highest-score retention. **Tier 1 + Tier 2 are exhausted across both local and cloud.** Tier 1 in-code keepers (#3 repeat-kill, #6 no-STAY, #7 predictive-walk) stay default ON; #2, #4, playbook, opponent_model default OFF. Only structurally different remaining option: Tier 3 #11 hierarchical goal selector (uncertain outcome, 2-3 days). Otherwise AE is final at hybrid-v3 — redirect remaining hours to NLP/CV/ASR if any have headroom (per their NOTES.md, mostly they don't).
9. **NLP**: v9 still wins blended (0.734) and v14-llm-rag holds the accuracy slot (0.734 cloud). Do **not** treat this as a format issue: v14c/v14d served locally before cloud timeout, and v15-lora used the same JSON path as v14. The common failure is infrastructure: `vllm/vllm-openai` cloud startup/throughput plus vLLM's Turing LoRA kernel. The shipped path remains v9 for blended. If NLP is reopened, choose one of two clean routes: quantize the merged Qwen3 LoRA on bigger hardware and then solve the base-image issue, or retrain LoRA on Qwen2.5-7B so it can run on the NGC stack that already survived cloud.
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

## GCP setup

### On the Workbench (only place `til` + Docker live)

Official loop:

```bash
til build <task> <tag>
til test  <task> <tag>
til submit <task> <tag>
```

Repeat for: `asr cv noise nlp ae`.

For long training jobs, do not rely on GPU activity alone to keep Workbench
alive. If you cannot edit Workbench idle-shutdown settings, run the job inside
`tmux` and keep a Jupyter notebook kernel active with a tiny heartbeat cell
that prints/sleeps every minute. We saw AE PPO and NLP LoRA runs get killed
or threatened by UI-idle behavior even while useful work was happening.

If `til` errors with "Could not find directory for task ...", set `TIL_FOLDER`:

```bash
echo 'export TIL_FOLDER=/home/jupyter/til' >> ~/.bash_profile
export TIL_FOLDER=/home/jupyter/til
```

If `til submit` errors with "Unauthenticated request" on the Docker push, register the gcloud credential helper once:

```bash
gcloud auth configure-docker asia-southeast1-docker.pkg.dev
```

### On the local Mac (gcloud SDK is installed)

The Mac now has `gcloud`, `gsutil`, and `bq` (auth: `ethankok@gmail.com`
impersonating `svc-melanie-minions`, default project `til-ai-2026`). Use it
for things that don't need Docker or `til`:

```bash
gsutil ls gs://melanie-minions-bucket-til-26/             # see team bucket
gsutil cp gs://melanie-minions-bucket-til-26/nlp_results.json .   # pull artifacts
```

Useful for: fetching `*_results.json` / failure packs onto the Mac for
offline triage with `nlp/error_report.py`, `training/cv/analyze_cv_failures.py`,
etc., without round-tripping through the Workbench. Build/test/submit still
need the Workbench — `docker` and `til` are not installed locally.

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
