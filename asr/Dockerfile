# Dockerfile for building the ASR image on the NeMo / Parakeet backend.
#
# Current A/B target:
#   til build asr nemo-zs-v3
#
# This stages nvidia/parakeet-tdt-0.6b-v2 as the default ASR build. The
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
# Switch the server to the NeMo backend and select the staged TDT model.
ENV ASR_BACKEND=nemo
ENV ASR_NEMO_MODEL=parakeet-tdt-0.6b-v2.nemo
# NGPU-LM n-gram shallow fusion. DEFAULT-OFF (PARKED 02 Jun 2026): with
# ASR_NGRAM_LM unset the decoder stays greedy and this image is behaviourally
# identical to nemo-ft-v3 (the shipped high). Fusion was built + tested this
# session: it gives a large LOCAL gain (T4 WER 0.0209 -> 0.0124) but malsd_batch
# RAISES on the cloud eval GPU for every batch, so on cloud the manager's
# protective fallback reverts to greedy (cloud score == greedy 0.970, never the
# 0.000 we saw before the fallback existed). Net blended 0.9633 < nemo-ft-v3
# 0.96425 -> not promotable; left OFF. See asr/NOTES.md "Result" for the full
# postmortem. To re-enable for further work, uncomment (path is the .arpa, NOT
# .nemo). The protective fallback in asr_manager_nemo.py makes ON safe to test.
#   ENV ASR_NGRAM_LM=/workspace/models/asr/ngram_lm.arpa
#   ENV ASR_NGRAM_LM_ALPHA=0.3
#   ENV ASR_BEAM_SIZE=4
#   ENV ASR_LM_STRATEGY=malsd_batch
#   ENV ASR_LM_PRUNING=late
#   ENV ASR_LM_BLANK_MODE=lm_weighted_full
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
# runtime cannot be imported.
RUN python -c "import nemo; import nemo.collections.asr as nemo_asr; print('NeMo runtime OK', getattr(nemo, '__version__', 'unknown'), nemo.__file__, flush=True)"

# Copy source.
COPY src .

# Bake in the Parakeet `.nemo` weights and the slang prompt. Populated on the
# GCP Workbench instance via training/asr/download_models_nemo.py before
# `til build asr nemo-zs-v3`. Layout:
#   models/parakeet-tdt-0.6b-v2.nemo
#   models/slang_prompt.txt
COPY models /workspace/models/asr

# Starts your model server.
CMD uvicorn asr_server:app --port 5001 --host 0.0.0.0
