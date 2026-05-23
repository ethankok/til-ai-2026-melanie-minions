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

## Where each task stands (23 May 2026)

For task-specific history, decisions, gotchas, and next-step plans, open the task's NOTES.md.

| Task | Latest shipped tag | Acc/reward | Speed | Status | Working log |
|---|---|---:|---:|---|---|
| ASR | `nemo-zs-v5` | 0.966 | 0.944 | **NEW BLENDED & ACCURACY HIGH.** Blended score `0.9605` (Accuracy: `0.966`, Speed: `0.944`). Includes slang prompter fix, space-eating bugfix, Zonnon/Caulfield rules, refined v5 proper nouns (Canian, Hegemony, Sharpsea, Nyari, Dreamer, Fullwalker, Edgedancer, Floodwall, TEC, CYPHER, Bloc), and context-specific Phi currency rules. | [asr/NOTES.md](asr/NOTES.md) |
| AE | `ppo-full-rl-v1-hybrid` | 0.638 | 0.847 | **CURRENT HIGH BY MAX SCORE.** Unchanged after 23 May session. Full-RL epoch-230 checkpoint in hybrid shortcut-off mode hit `0.638/0.847`; shortcut-on nearly tied at `0.637/0.845`; old `fixed-map-v5` is `0.630/0.858`. Pure-policy variance farm now n=8 (mean ~0.577, max still 0.625; right tail did not repeat). Belief-aware PPO (`ppo-full-rl-belief-v1`) trained but did not clear the `0.6236` save floor (final weighted eval `0.4925`); not submitted. New env lever `AE_DIJKSTRA_BOMB_COST` (default `5.0`) is available but did not lift aggregate locally. See [ae/NOTES.md](ae/NOTES.md) "23 May session" for full detail. | [ae/NOTES.md](ae/NOTES.md) |
| NLP | `v24-speed-optimized` | 0.959 | 0.950 | **NEW BLENDED HIGH.** Blended score `0.95675` (Accuracy: `0.959`, Speed: `0.950`). Shipped with optimized retrieval parameters (`tkr=15`, `dpw=0.6`, `bw=0.8`, `dw=0.5`) yielding `0.9807` hit rate, reduced sequence length (128), and batched rerank (128) to recover speed to `0.950`. | [nlp/NOTES.md](nlp/NOTES.md) |
| CV | `yolo11l-1280-alldata-final-v2` | [pending] | [pending] | **NEW ALL-DATA 1280px candidate.** Fine-tuned YOLO11l on all data natively at imgsz=1280 for 36 epochs. Served at imgsz=1280. Local validation mAP is 0.988 (small AP 0.793). | [cv/NOTES.md](cv/NOTES.md) |
| Noise | `latest` | 1.000 | 0.970 | Last shipped is still the JPEG re-encode baseline. Level 9 AdvGAN generator landed on `main` 24 May (`83051b9`) but has not been built/submitted. Official spec: no direct Qualifier reward. | [noise/NOTES.md](noise/NOTES.md) |

---

## Rough outline (what to do, in order)

**Week 0 — plumbing (done)**

1. Workbench access for everyone ✅
2. All 5 services build, `til test` runs clean, JSON shape correct ✅
3. All 5 baselines submitted ✅

**Week 1 — first real models (in progress)**

4. **AE** (40%) — **Direction change as of 24 May**: forensic git-diff revealed the entire `ppo-full-rl-v1-hybrid`/`-policy`/`-shortcut` cluster (0.55-0.64 cloud scores) was actually pure-heuristic behavior all along due to silent fallback in `ae_server.py` (deployed `ae/src/model.py` couldn't load the `legacy-small` checkpoint arch). Confirmed today when fixing the load path made the policy actually engage on cloud and scores immediately dropped to ~0.41. Path forward is heuristic parameter tuning, not learned policies. Current cloud high is `heuristic-A-vf1` at **0.613/0.845** (`AE_MODE=heuristic, AE_ENEMY_BASE_VALUE=160, AE_DIST_PENALTY=0.9`), 3-submit mean 0.599. Five structural experiments today (dypm tree search, opening book, rusher, camping, Elo population self-play) were all falsified — none beat the tuned heuristic. Tonight's heuristic-knob sweep is running for further uplift; aggressive-offense direction wins clearly over defense-first (mean 0.563) and item-farming (mean 0.553).
5. **NLP** (20%) — **NEW ACCURACY HIGH (0.971 / 0.880).** `v23-large-reranker-v2` correctly baked in the large reranker model (`BAAI/bge-reranker-large`) along with swept retrieval parameters (`dpw=0.45, tkr=30, bw=1.0, dw=1.0`) to resolve a weight mismatch and hit a new accuracy high of `0.971`. Shipped `v24-speed-optimized` to optimize speed by halving `TOP_K_RETRIEVE` to 15 (sweep hit rate: `0.9807`), shrinking sequence length to 128, and increasing batching to 128.
6. **CV** (20%) — **NEW ALL-DATA 1280px candidate (shipped yolo11l-1280-alldata-final-v2, pending score).** This fine-tunes YOLO11l on all available training data natively at 1280px resolution for 36 epochs. Served at imgsz=1280 to maximize detection and localization precision (local validation mAP 0.988, small AP 0.793). Prior high: `yolo11l-896-plusval-v1` at **0.640/0.954**.
7. **ASR** (20%) — `nemo-zs-v5` reached accuracy `0.966` and speed `0.944` (blended high of `0.9605`) on the cloud. This includes refined proper noun/spacing rules (Canian, Hegemony, Sharpsea, Nyari, Dreamer, Fullwalker, Edgedancer, Floodwall, TEC, CYPHER, Bloc) and refined context-specific Phi rules.

**Week 2 — push scores**

8. **AE**: pivot to heuristic parameter tuning (`AE_MODE=heuristic`, no PPO checkpoint loaded). `heuristic-A` aggressive-offense variant at `0.613/0.845` is the live high; mean 0.599 across 3 submits. Tonight's parameter sweep is running for further uplift — top candidates to try: `AE_ENEMY_BASE_VALUE` ∈ {180, 200}, `AE_DIST_PENALTY` ∈ {0.7, 0.8}, `AE_DIJKSTRA_BOMB_COST` ∈ {3, 4}, in combination. Defense-first knobs (`AE_TIER1_DEFENSE=1`, `AE_BASE_DEFENSE_HEALTH=80`) hurt and are off the table — same wall as historical `ttd-defense-v1`, `ally-bomb-safe-v2`, dypm-veto, and camping experiments. Item-farming knobs (high mission value, low base value) also hurt — `+50/destroy_enemy_base` outweighs marginal +30 from extra missions. The legacy-arch support added today stays in tree (dormant under `AE_MODE=heuristic`) in case a genuinely better policy ever materializes; `compare_candidates.py` is available as a 24×3 local gate for checkpoint comparison. The session's 5 falsified structural experiments are documented in `ae/NOTES.md`.
9. **NLP**: `v23-large-reranker-v2` is our new accuracy high at **0.971 / 0.880**. We have built and submitted `v24-speed-optimized` which halves `TOP_K_RETRIEVE` to 15 (improving sweep hit rate to `0.9807`), restricts `RERANK_MAX_LEN` to 128, and increases `RERANK_BATCH` to 128. This yields `2.24 it/s` locally (1.82x speedup over v22 base-reranker) and should recover cloud speed to >0.95.
10. **CV**: now active at the YOLO11l all-data native 1280px lineage. We trained and submitted `yolo11l-1280-alldata-final-v2` (fine-tuned YOLO11l on all data natively at imgsz=1280 for 36 epochs, local validation mAP 0.988). Prior high is `yolo11l-896-plusval-v1` (**0.640/0.954**). If the new all-data 1280px model exceeds the plusval-v1 score, protect it; otherwise plusval-v1 remains the active high.
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
