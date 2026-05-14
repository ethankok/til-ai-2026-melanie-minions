# Dockerfile for building the NLP image.


# The base image, an example deep learning VM.
# For a full list, see https://us-docker.pkg.dev/deeplearning-platform-release/gcr.io/
# For info, see https://cloud.google.com/deep-learning-vm/docs/images#supported-frameworks
FROM nvcr.io/nvidia/pytorch:25.11-py3

# Configures settings for the image.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_ROOT_USER_ACTION=ignore
ENV HF_HUB_DISABLE_TELEMETRY=1
ENV TRANSFORMERS_OFFLINE=1
ENV NLP_MODEL_DIR=/workspace/models
WORKDIR /workspace

# Installs your dependencies.
RUN pip install -U pip
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake model weights into the image so inference runs fully offline.
# This must run BEFORE setting TRANSFORMERS_OFFLINE=1 takes effect for the
# downloader — ENV directives apply to subsequent layers, so we override
# locally with `TRANSFORMERS_OFFLINE=0`.
COPY download_models.py .
RUN TRANSFORMERS_OFFLINE=0 python download_models.py

# Bundle the fine-tuned QA model if it's been built locally (see
# training/nlp/finetune_qa.py). The .gitkeep keeps the directory tracked
# in git; COPY accepts an empty/contents-only dir fine.
COPY models /workspace/models-local
RUN if [ -d /workspace/models-local/roberta-finetuned-squad2 ]; then \
        cp -r /workspace/models-local/roberta-finetuned-squad2 /workspace/models/; \
        echo "[nlp build] bundled fine-tuned QA weights"; \
    else \
        echo "[nlp build] no fine-tuned QA weights found; falling back to base"; \
    fi && rm -rf /workspace/models-local

# Copies your source files.
COPY src .

# Starts your model server.
CMD uvicorn nlp_server:app --port 5004 --host 0.0.0.0
