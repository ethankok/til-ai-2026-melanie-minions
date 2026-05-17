"""Pre-download model weights into the image so /nlp can run offline.

Run at build time via the Dockerfile. Outputs to /workspace/models, which
matches NLP_MODEL_DIR in nlp_manager.py.

v14-llm-rag adds Qwen2.5-7B-Instruct-AWQ as the answerer. We fetch it via
huggingface_hub.snapshot_download because vLLM expects the AWQ weights/config
on disk, not the transformers cache layout. Set NLP_SKIP_LLM_DOWNLOAD=1 (or
build with a different target) to skip the ~5 GB LLM fetch when iterating on
the retrieval-only path.
"""

import os
from pathlib import Path

from transformers import (
    AutoModel,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

OUT = Path("/workspace/models")

SPECS = [
    ("BAAI/bge-small-en-v1.5", AutoModel, "bge-small-en-v1.5"),
    (
        "BAAI/bge-reranker-base",
        AutoModelForSequenceClassification,
        "bge-reranker-base",
    ),
    (
        "deepset/roberta-base-squad2",
        AutoModelForQuestionAnswering,
        "roberta-base-squad2",
    ),
]

LLM_REPO = os.getenv("NLP_LLM_REPO", "Qwen/Qwen2.5-7B-Instruct-AWQ")
LLM_LOCAL = "llm"


def _download_retrievers() -> None:
    for repo, model_cls, local_name in SPECS:
        dest = OUT / local_name
        print(f"--> {repo} -> {dest}")
        tok = AutoTokenizer.from_pretrained(repo)
        mdl = model_cls.from_pretrained(repo)
        tok.save_pretrained(dest)
        mdl.save_pretrained(dest)


def _download_llm() -> None:
    if os.getenv("NLP_SKIP_LLM_DOWNLOAD", "").lower() in {"1", "true", "yes"}:
        print(f"[download_models] skipping LLM download (NLP_SKIP_LLM_DOWNLOAD set)")
        return
    from huggingface_hub import snapshot_download

    dest = OUT / LLM_LOCAL
    print(f"--> {LLM_REPO} -> {dest}")
    # allow_patterns avoids pulling random extras; AWQ weights are .safetensors.
    snapshot_download(
        repo_id=LLM_REPO,
        local_dir=str(dest),
        allow_patterns=[
            "*.safetensors",
            "*.json",
            "*.txt",
            "tokenizer*",
            "*.model",
            "*.tiktoken",
        ],
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _download_retrievers()
    _download_llm()


if __name__ == "__main__":
    main()
