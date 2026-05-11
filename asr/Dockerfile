# Dockerfile for building the ASR image.


# The base image, an example deep learning VM.
# For a full list, see https://us-docker.pkg.dev/deeplearning-platform-release/gcr.io/
# For info, see https://cloud.google.com/deep-learning-vm/docs/images#supported-frameworks
FROM nvcr.io/nvidia/pytorch:25.11-py3

# Configures settings for the image.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_ROOT_USER_ACTION=ignore
# Make pip-installed CUDA 12 libraries discoverable by CTranslate2.
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cublas/lib:/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH}
WORKDIR /workspace

# Installs your dependencies.
RUN pip install -U pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copies your source files.
COPY src .

# Bakes in CT2-converted model weights + slang prompt. Populated on the GCP
# Workbench instance via `gsutil cp -r gs://<bucket>/asr-models models` before
# `til build asr`. The bucket layout mirrors a faster-whisper model dir:
#   models/{model.bin, config.json, tokenizer.json, ..., slang_prompt.txt}
COPY models /workspace/models/asr

# Starts your model server.
CMD uvicorn asr_server:app --port 5001 --host 0.0.0.0
