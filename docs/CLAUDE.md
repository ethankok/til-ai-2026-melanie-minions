# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

DSTA BrainHack TIL-AI 2026 competition entry. Five separate Dockerized model services. Qualifier submission deadline (24 May 2026 23:59:59 SGT) has passed; team is now in **Semifinals prep through 2026-06-10** (in-person testing 10 Jun, Semis/Finals 11 Jun, at Marina Bay Sands). See [RESULTS.md](RESULTS.md) for final Qualifier scores and the live status table.

**Qualifiers vs Finals — how the evaluator reaches us changed.** In Qualifiers the official evaluator POSTed each task container directly over HTTP. In **Finals**, all six containers (the five tasks **plus a new `finals` orchestration server**) run simultaneously in one Docker Compose stack on a competition desktop. The competition server speaks **WebSocket** to *our* `finals` server (`ws://<server>/ws/<TEAM_NAME>`), which then calls our five task containers over plain HTTP on their usual ports (5001–5005) using the **same request/response JSON as Qualifiers** — so all existing `*_manager.py` work carries straight over. See the [Finals orchestration](#finals-orchestration) section below.

The five tasks (and official scoring weight):

| Task   | Dir       | Endpoint  | Port  | Weight |
|--------|-----------|-----------|-------|--------|
| AE     | `ae/`     | `/ae` + `/reset` | 5005 | 40% |
| ASR    | `asr/`    | `/asr`    | 5001  | 20% |
| CV     | `cv/`     | `/cv`     | 5002  | 20% |
| NLP    | `nlp/`    | `/nlp`    | 5004  | 20% |
| Noise  | `noise/`  | `/noise`  | 5003  | no direct Qualifier reward; Finals disruption tool |

For ASR/CV/NLP/AE, each challenge is 75% accuracy/reward + 25% speed. Qualifier speed score is `1 - min(t_elapsed, 30min) / 30min` over the whole test set. **Finals speed is per-batch, not whole-set**: `time_score = 1 - min(batch_elapsed, 5s) / 5s`, weighted 0.75 perf / 0.25 speed (`MAX_TIME_PER_TEST_CASE=5s`). A batch of 4 that takes ≥5s still scores accuracy but speed 0; a batch that misses the **10s `MISSION_BATCH_TIMEOUT_SEC`** is abandoned and all 4 items score 0. AE is per-step with a **2s `AE_TIME_CUTOFF`** (slow → forced `STAY`). Corpus ingest deadline is **60s**.

Official challenge spec source of truth: <https://github.com/til-ai/til-26/wiki/Challenge-specifications>

## Architecture

Every task directory has the same shape:

```
<task>/
  Dockerfile              packaging — usually don't edit
  requirements.txt        deps that ship inside the container
  src/
    <task>_server.py      FastAPI HTTP wrapper — usually don't edit
    <task>_manager.py     OUR inference/logic — this is what we edit
```

Request flow: evaluator → `*_server.py` (FastAPI) → `*_manager.py` (one call per input) → JSON response. The server wraps `manager.<task>(...)` results into the schema the evaluator expects; **wrong JSON shape scores 0 even with a perfect model.** See each task's `README.md` and the official Challenge specifications for its input/output spec.

Critical schema notes from the Wiki:
- CV boxes are LTWH `[l, t, w, h]`, not center-XYWH or x1/y1/x2/y2; category IDs are exactly 0-17 from the official target list.
- NLP corpus-load response is `{"predictions": [{"status": "loaded"}]}`; question responses are dicts `{"documents": [...], "answer": "..."}`, not bare answer strings.
- AE observations include `agent_viewcone`, `base_viewcone`, scalar state, and `action_mask`; always obey the mask.

AE is special: its server treats empty POSTs and `step == 0` observations as a reset signal, and `/reset` re-instantiates `AEManager` (no persistent state outside the manager — see [ae/src/ae_server.py](ae/src/ae_server.py)).

## Submodules

```
til-26-ae/      official AE environment (provides the `til_environment` Python package)
til-26-finals/  Finals orchestration scaffold + local test competition server
                (pinned at upstream `6c81ca8`, "hotfix: update nlp eval")
```

Both are git submodules — **do not edit, do not commit changes inside them, do not touch `.gitmodules`**. Bumping the submodule *pointer* (checking out a newer upstream commit and staging the gitlink in the parent repo) is fine and expected; editing files *inside* a submodule is not. Initialize with `git submodule update --init`. `til-26-ae` is installed as an editable pip package via `requirements-dev.txt`.

`til-26-finals/` contains: `finals/` (the orchestration server we deploy — `participant_server.py` + `models_manager.py`), `test/` (stub participants), `test_competition_server/` (a local replica of the competition server, incl. the real `nlp_eval.py`/`scoring.py`/`constants.py`), `finals.sh` (build/test/submit), and `docker-compose-test.yml`.

## Setup

```bash
git submodule update --init
conda create --name til-asr python=3.13 && conda activate til-asr   # or venv/uv/poetry
pip install -r requirements-dev.txt
```

Sanity-check that all five managers import:

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

## Build / test / submit

`til build / til test / til submit` need both Docker and the `til` CLI, neither
of which is installed on the local Mac — run them on the GCP Workbench
instance, which has them pre-installed and mounts the dataset at
`/home/jupyter/novice` or `/home/jupyter/advanced`:

```bash
til build <task> [tag]    # docker build -t TEAM_ID-CHALLENGE:TAG .
til test  <task> [tag]    # runs official evaluator against your image on a no-internet docker network
til submit <task> [tag]   # uploads for scoring
```

What the **local Mac** *can* do (gcloud SDK is installed, authenticated as the
team GCP user impersonating the team service account `<TEAM_SERVICE_ACCOUNT>`):

```bash
gsutil ls gs://<TEAM_BUCKET>/     # team bucket: results + failure packs
gsutil cp gs://<TEAM_BUCKET>/nlp_results.json .   # pull artifacts for analysis
gcloud auth list                                  # check active account
bq query ...                                      # if you ever need BQ
```

Use this to pull failure packs / results JSONs for offline analysis on the
Mac without needing the Workbench. Don't try `til build` / `docker build`
here — they will fail.

Local pytest-style tests live in `test/` (`test_<task>.py`) and are what `til test` invokes; they hit the running container at `localhost:<port>`.

## Finals orchestration

The Finals submission is built/tested/submitted from inside `til-26-finals/` via `finals.sh` (run on the Workbench — needs Docker), **not** `til`:

```bash
bash finals.sh build  finals --build_all   # build all six finals-tagged images locally
bash finals.sh test                        # full local end-to-end match vs test_competition_server
bash finals.sh submit finals               # build+push ONLY the orchestration server
bash finals.sh submit finals --submit_all  # build+push server + all five task images
```

**The image tag MUST be `finals`** — the organisers only pull what is tagged `finals` (`{TEAM_NAME}-server:finals`, `{TEAM_NAME}-{task}:finals`). Per-task qualifier commands (`bash finals.sh submit asr`, etc.) still work. To run a non-standard stack (e.g. one model serving two tasks to save VRAM), drop a `custom-compose.yml` at the root of the team GCS bucket and they run that instead.

How our pieces fit:
- **`finals/participant_server.py`** — connects out over WebSocket to `ws://$COMPETITION_SERVER_IP:$PORT/ws/$TEAM_NAME`, plus a local `GET /health` on port 5000. Handles message types `task=ae` (per-step), `mission_batch`/`noise` (batch of 4), `corpus`, `done`.
- **`finals/models_manager.py`** — HTTP client that fans out to our containers. Exact contracts it relies on (all unchanged from Qualifiers, so don't break them):
  - AE → `POST :5005/ae {"instances":[{"observation": obs}]}` → reads `predictions[0]["action"]`. Our reset-on-`step==0` still fires; there is **no** `/reset` call between matches, so each match must self-reset on step 0.
  - ASR → `:5001/asr {"instances":[{"b64"}]}` → `predictions` = list of answer strings.
  - CV → `:5002/cv {"instances":[{"b64"}]}` → `predictions` = list of detection-lists (`{bbox(LTWH), category_id}`).
  - NLP question → `:5004/nlp {"instances":[{"question"}]}` → `predictions` = list of `{"answer","documents"}`.
  - NLP corpus → `:5004/nlp {"instances":[{"documents":[{"doc_id","content"}]}]}` then polls with `{"poll":"true"}`. Our `_parse_doc_payload` already accepts `doc_id`/`content`. **Caveat:** on load failure our `nlp_server.py` returns status `"failed"`, but the orchestrator only special-cases `"error"` — a true load failure would make it poll until the 60s deadline rather than fail fast. Harmless unless corpus load actually breaks.
  - Noise → `:5003/noise {"instances":[{"key","b64"}]}` → `predictions` = list of b64 (null entries dropped).

**Hardware change — the real risk.** MBS desktops run a **Blackwell RTX 5070 Ti (sm_120)**, not the Turing **T4 (sm_75)** on the GCP eval. CUDA/PyTorch/wheel builds that work on the T4 can fail on Blackwell (this is exactly the class of failure that sank the ASR NGPU-LM `malsd_batch` path on cloud GPU). **Test on the actual hardware via the automated system in `til-26-finals` before 10 Jun** — don't assume a T4-validated image runs on sm_120.

### Eval changes for Finals (from the wiki)
- **NLP:** the answer-equivalence evaluator now tokenises the candidate with `add_special_tokens=False` (in `test_competition_server/src/nlp_eval.py`). Scoring-side only — no change to our container's I/O, but candidate truncation/scoring shifts slightly. Eval model is `nlp_eval_512` (`NLP_EVAL_MAX_LENGTH=512`).
- **AE:** AE *advanced* is now **seeded and de-randomised** ("some sources of non-determinism removed"). This is why the cloud eval is now deterministic (1 submit = true score — see memory). Lives in the competition server, not the `til_environment` package, so our `til-26-ae` pin is unaffected.
- **AE resource reward (6 Jun org disclosure):** picking up **resource tiles gives NO score** — `collect_resource:2.0` was a stray reward, never paid in finals. Resources only yield fuel (→ bombs → enemy-base destruction, which scores). This one IS in `til_environment`: our pin was `beb81f8` (paid +2.0/pickup via the `default_config()` dataclass we use). **Bumped the `til-26-ae` submodule pin to `b32af97`** (resource reward removed; same-seed A/B showed the change is placement-neutral and the deploy holds). On the Workbench, `git pull && git submodule update --init` to match. See [ae/NOTES.md](ae/NOTES.md) "Resource-reward env fix". (A separate, unverified bomb-timer discrepancy — heuristic `BOMB_TIMER=3` vs true ~5-step detonation — is flagged there too; not yet changed.)

## What to edit, what not to

Edit:
- `<task>/src/<task>_manager.py` — the only files that should change for normal task work.

Avoid editing unless you know why:
- `<task>/src/<task>_server.py`, `<task>/Dockerfile`, `<task>/requirements.txt` — the one current exception is the reset-robustness patch already applied to [ae/src/ae_server.py](ae/src/ae_server.py).
- Anything in `til-26-ae/`, `til-26-finals/`, or `test/` (those are official scaffolding). The default `finals/` orchestration server in `til-26-finals/` works as-is **because our task containers already match the schemas it expects** — verify that before changing managers, and prefer not to edit inside the submodule. If the orchestrator genuinely needs customizing (e.g. a `custom-compose.yml` collapsing two tasks into one container), do it via the bucket `custom-compose.yml` route rather than editing submodule files.

Don't commit large model weights to git.

## Container deps vs. dev deps

- Need it inside the deployed Docker image? Add to `<task>/requirements.txt`.
- Only needed for local training/testing? Add to root `requirements-dev.txt`.

## Strategic context

Per-task current state, decisions, and gotchas live in `<task>/NOTES.md`.
Submission history and the final cross-task status table live in [RESULTS.md](RESULTS.md).

Read the relevant task's `NOTES.md` before making strategic decisions. Use
[RESULTS.md](RESULTS.md) for official scores, shipped tags, and leaderboard
history.

One quirk worth knowing here: the NLP leaderboard was wiped 14 May 2026 when organisers rolled out the new eval (top-3 docs, 0.9 ModernBERT-AE threshold, dict response shape). The old `latest` score of `0.301` is gone; only new-eval submissions count. `v4-dict-id` was the recovery from earlier 0.000 submissions caused by a bug in the org eval server (was sending plain strings instead of `{"id": "DOC-XXXX", "document": "..."}` dicts; fixed 14 May 09:33 SGT).
