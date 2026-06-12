# Dockerfile for building the ASR image on the NeMo / Parakeet backend.
#
# Stages nvidia/parakeet-tdt-0.6b-v2 as the default ASR build. The prior
# faster-whisper LoRA path remains recoverable from git history; the parallel
# Dockerfile.nemo mirrors this build shape for explicit docker builds.

FROM nvcr.io/nvidia/pytorch:25.11-py3

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_ROOT_USER_ACTION=ignore
# Keep model loaders fully offline at runtime; eval container has no internet.
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1
ENV ASR_BACKEND=nemo
ENV ASR_NEMO_MODEL=parakeet-tdt-0.6b-v2.nemo
# NGPU-LM n-gram shallow fusion: DEFAULT OFF (parked, see asr/NOTES.md). Large
# local WER gain but malsd_batch raises on the cloud eval GPU; manager's
# fallback reverts to greedy so it's safe to test. Path is the .arpa, NOT .nemo.
#   ENV ASR_NGRAM_LM=/workspace/models/asr/ngram_lm.arpa
#   ENV ASR_NGRAM_LM_ALPHA=0.3
#   ENV ASR_BEAM_SIZE=4
#   ENV ASR_LM_STRATEGY=malsd_batch
#   ENV ASR_LM_PRUNING=late
#   ENV ASR_LM_BLANK_MODE=lm_weighted_full
WORKDIR /workspace

# NeMo is a heavy install; this is the longest layer. Git is required because
# requirements-nemo.txt pins NeMo to a GitHub commit newer than PyPI.
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
RUN pip install -U pip
COPY requirements-nemo.txt .
RUN pip install --no-cache-dir -r requirements-nemo.txt
# cuda-python for the TDT/RNNT CUDA-graph fast path. Pin to 12.x — numba 0.65.x
# can't import cuda-python 13.x. --no-deps avoids pip downgrading torch/torchvision/cuDNN.
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
