# Dockerfile for building the ASR image on the NeMo / Parakeet backend.
#
# Current A/B target:
#   til build asr parakeet-unified-zs
#
# This stages nvidia/parakeet-unified-en-0.6b as the default ASR build. The
# prior faster-whisper LoRA path remains recoverable from git history; the
# parallel Dockerfile.nemo mirrors this NeMo build shape for explicit docker
# builds.

FROM nvcr.io/nvidia/pytorch:25.11-py3

# Configures settings for the image.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_ROOT_USER_ACTION=ignore
# Keep model loaders fully offline at runtime; eval container has no internet.
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
# Switch the server to the NeMo backend and select the staged unified model.
ENV ASR_BACKEND=nemo
ENV ASR_NEMO_MODEL=parakeet-unified-en-0.6b.nemo
WORKDIR /workspace

# Install dependencies. NeMo is a heavy install; this is the longest layer.
# Git is required because requirements-nemo.txt pins NeMo to a GitHub commit
# newer than the latest PyPI release.
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
RUN pip install -U pip
COPY requirements-nemo.txt .
RUN pip install --no-cache-dir -r requirements-nemo.txt
# cuda-python is needed for the TDT/RNNT CUDA-graph fast path where supported.
# Pin to 12.x because numba 0.65.x can't import cuda-python 13.x (package
# restructure broke `from cuda import cuda`). Use --no-deps so pip cannot
# downgrade torch / torchvision / cuDNN to satisfy cuda-bindings's nvidia-*
# requirements; the base image already ships a consistent CUDA stack.
RUN pip install --no-cache-dir --no-deps "cuda-python>=12.3,<13"

# Fail during `til build`, before copying multi-GB model weights, if the NeMo
# runtime cannot instantiate nvidia/parakeet-unified-en-0.6b.
RUN python -c "import inspect, nemo; from nemo.collections.asr.modules.conformer_encoder import ConformerEncoder; params = inspect.signature(ConformerEncoder.__init__).parameters; print('NeMo runtime', getattr(nemo, '__version__', 'unknown'), nemo.__file__, flush=True); print('ConformerEncoder supports att_chunk_context_size:', 'att_chunk_context_size' in params, flush=True); assert 'att_chunk_context_size' in params, 'NeMo runtime cannot load parakeet-unified-en-0.6b: missing ConformerEncoder(att_chunk_context_size)'"

# Copy source.
COPY src .

# Bake in the Parakeet `.nemo` weights and the slang prompt. Populated on the
# GCP Workbench instance via training/asr/download_models_nemo.py before
# `til build asr parakeet-unified-zs`. Layout:
#   models/parakeet-unified-en-0.6b.nemo
#   models/slang_prompt.txt
COPY models /workspace/models/asr

# Starts your model server.
CMD uvicorn asr_server:app --port 5001 --host 0.0.0.0
