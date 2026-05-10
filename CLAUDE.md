# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

DSTA BrainHack TIL-AI 2026 competition entry. Five separate Dockerized model services that the official evaluator queries over HTTP. Submission deadline: 24 May 2026, 11:59:59 PM SGT.

The five tasks (and scoring weight):

| Task   | Dir       | Endpoint  | Port  | Weight |
|--------|-----------|-----------|-------|--------|
| AE     | `ae/`     | `/ae` + `/reset` | 5005 | 40% |
| ASR    | `asr/`    | `/asr`    | 5001  | 20% |
| CV     | `cv/`     | `/cv`     | 5002  | 20% |
| NLP    | `nlp/`    | `/nlp`    | 5004  | 20% |
| Noise  | `noise/`  | `/noise`  | 5003  | required, no qualifier weight |

Each scored task is roughly 75% accuracy/reward + 25% speed.

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

Request flow: evaluator → `*_server.py` (FastAPI) → `*_manager.py` (one call per input) → JSON response. The server wraps `manager.<task>(...)` results into the schema the evaluator expects; **wrong JSON shape scores 0 even with a perfect model.** See each task's `README.md` for its input/output spec.

AE is special: its server treats empty POSTs and `step == 0` observations as a reset signal, and `/reset` re-instantiates `AEManager` (no persistent state outside the manager — see [ae/src/ae_server.py](ae/src/ae_server.py)).

## Submodules

```
til-26-ae/      official AE environment (provides the `til_environment` Python package)
til-26-finals/  pulled in for Semifinals/Finals
```

Both are git submodules — **do not edit, do not commit changes inside them, do not touch `.gitmodules`**. Initialize with `git submodule update --init`. `til-26-ae` is installed as an editable pip package via `requirements-dev.txt`.

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

These commands only work on the GCP Workbench instance (the `til` CLI is pre-installed there, and `til test` mounts the dataset from `/home/jupyter/novice` or `/home/jupyter/advanced`):

```bash
til build <task> [tag]    # docker build -t TEAM_ID-CHALLENGE:TAG .
til test  <task> [tag]    # runs official evaluator against your image on a no-internet docker network
til submit <task> [tag]   # uploads for scoring
```

Local equivalents if needed: `cd <task> && docker build -t TEAM_ID-<task>:<tag> .`

Local pytest-style tests live in `test/` (`test_<task>.py`) and are what `til test` invokes; they hit the running container at `localhost:<port>`.

## What to edit, what not to

Edit:
- `<task>/src/<task>_manager.py` — the only files that should change for normal task work.

Avoid editing unless you know why:
- `<task>/src/<task>_server.py`, `<task>/Dockerfile`, `<task>/requirements.txt` — the one current exception is the reset-robustness patch already applied to [ae/src/ae_server.py](ae/src/ae_server.py).
- Anything in `til-26-ae/`, `til-26-finals/`, or `test/` (those are official scaffolding).

Don't commit large model weights to git.

## Container deps vs. dev deps

- Need it inside the deployed Docker image? Add to `<task>/requirements.txt`.
- Only needed for local training/testing? Add to root `requirements-dev.txt`.

## Strategic notes (from SUMMARY.md)

Goal order: get all 5 baselines submitting valid output first, **then** improve in this order: AE → NLP → CV → ASR → Noise. AE is worth 40% so early wins there matter most; NLP's keyword-match baseline is the easiest to upgrade with retrieval. See [SUMMARY.md](SUMMARY.md) for current baseline behavior, target improvements, and per-task ownership.
