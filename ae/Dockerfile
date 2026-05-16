# Dockerfile for building the AE image.


# The base image. Using a regular Python base because AE agents don't typically
# require GPU. This will speed up your evaluation.
FROM python:3.11-slim

# Configures settings for the image.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_ROOT_USER_ACTION=ignore
WORKDIR /workspace

# Runtime manager selection (hybrid | policy | heuristic). Bake it into
# the image so the cloud evaluator (which doesn't inherit our shell env)
# uses the right mode. Edit this line before each `til build` to switch
# modes, OR write the mode into `ae/src/.ae_mode` (the server reads that
# file at startup if AE_MODE is unset in the container env).
ENV AE_MODE=hybrid
# v2 (post-timeout) MCTS-light config — see ae/NOTES.md for the speed budget math.
# v1 used DEPTH=5 WIDTH=96 with no gate / no latency cap; cloud killed it after
# 2-4× the per-tick budget. v2 caps total search at AE_MCTS_BUDGET_MS via a
# time.monotonic() guard in AEManager._tactical_lookahead_action, AND only
# invokes MCTS when bombs/enemies/enemy-bases are within mcts_depth+1 reach
# (cheap _should_run_mcts pre-flight gate).
ENV AE_MCTS=1
ENV AE_MCTS_DEPTH=3
ENV AE_MCTS_WIDTH=24
ENV AE_MCTS_BUDGET_MS=80
ENV AE_MCTS_LOG_TIMING=1
ENV AE_MCTS_MIN_SCORE=12
ENV AE_HYBRID_TRUST_MCTS=1
ENV AE_HYBRID_MCTS_MIN_SCORE=12

# Installs your dependencies.
RUN pip install -U pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copies your source files.
COPY src .

# Copies trained policy weights (if any). Build still succeeds with an empty
# `models/` and the server falls back to the heuristic planner at runtime.
COPY models /workspace/models

# Starts your model server. uvloop + httptools are already pulled in by
# uvicorn[standard]; setting them explicitly ensures the faster event loop
# and parser are used (gives a small per-request latency win for AE since
# /ae is called every game tick).
CMD ["uvicorn", "ae_server:app", "--host", "0.0.0.0", "--port", "5005", "--loop", "uvloop", "--http", "httptools"]
