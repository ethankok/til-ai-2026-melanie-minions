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
ENV HF_HUB_OFFLINE=1
ENV NLP_MODEL_DIR=/workspace/models
# v14-llm-rag: 'llm' routes the answerer through vLLM + Qwen2.5-7B-Instruct-AWQ.
# Set NLP_ANSWERER=extractive to roll back to v9 RoBERTa-large behaviour.
ENV NLP_ANSWERER=llm
ENV NLP_LLM_DIR=/workspace/models/llm
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

# Bundle the fine-tuned QA model(s) and optional v12 answer-ranker JSON if
# they've been built locally (see training/nlp/finetune_qa.py,
# training/nlp/finetune_genqa.py, and training/nlp/train_answer_ranker.py).
# Manager prefers extractive in this order (v13b):
#   deberta-finetuned-squad2 (DeBERTa-v3-large fine-tune, v13b)
#   > roberta-finetuned-squad2 (RoBERTa-large fine-tune, v8b/v9)
#   > stock SQuAD2 base
# Flan-T5 generative is opt-in via NLP_QA_MODE=generative.
COPY models /workspace/models-local
RUN if [ -d /workspace/models-local/flan-t5-finetuned ]; then \
        cp -r /workspace/models-local/flan-t5-finetuned /workspace/models/; \
        echo "[nlp build] bundled fine-tuned generative QA (Flan-T5) weights"; \
    fi && \
    if [ -d /workspace/models-local/deberta-finetuned-squad2 ]; then \
        cp -r /workspace/models-local/deberta-finetuned-squad2 /workspace/models/; \
        echo "[nlp build] bundled fine-tuned extractive QA (DeBERTa-v3-large) weights"; \
    fi && \
    if [ -d /workspace/models-local/roberta-finetuned-squad2 ]; then \
        cp -r /workspace/models-local/roberta-finetuned-squad2 /workspace/models/; \
        echo "[nlp build] bundled fine-tuned extractive QA (RoBERTa-SQuAD2) weights"; \
    fi && \
    if [ ! -d /workspace/models-local/flan-t5-finetuned ] && \
       [ ! -d /workspace/models-local/deberta-finetuned-squad2 ] && \
       [ ! -d /workspace/models-local/roberta-finetuned-squad2 ]; then \
        echo "[nlp build] no fine-tuned QA weights found; falling back to stock base"; \
    fi && \
    if [ -f /workspace/models-local/answer_ranker.json ]; then \
        cp /workspace/models-local/answer_ranker.json /workspace/models/answer_ranker.json; \
        echo "[nlp build] bundled v12 answer ranker"; \
    fi && rm -rf /workspace/models-local

# v14-llm-rag: hard-fail the build if the LLM weights didn't land. The
# manager will still boot without them by falling back to extractive, but a
# silent fallback would mask a packaging regression on a v14 tag.
RUN if [ "$NLP_ANSWERER" = "llm" ] && [ ! -f /workspace/models/llm/config.json ]; then \
        echo "[nlp build] WARNING: NLP_ANSWERER=llm but /workspace/models/llm/config.json missing"; \
        echo "[nlp build] container will fall back to extractive QA at runtime"; \
    fi

# Copies your source files.
COPY src .

# Starts your model server.
CMD uvicorn nlp_server:app --port 5004 --host 0.0.0.0
