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
