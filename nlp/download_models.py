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
RERANKER_REPO = os.getenv("NLP_RERANKER_REPO", "BAAI/bge-reranker-base")
RERANKER_LOCAL_NAME = os.getenv("NLP_RERANKER_LOCAL_NAME", "bge-reranker-base")
DOWNLOAD_MODERNBERT = os.getenv("NLP_DOWNLOAD_MODERNBERT", "0").lower() in {
    "1",
    "true",
    "yes",
}

SPECS = [
    ("BAAI/bge-small-en-v1.5", AutoModel, "bge-small-en-v1.5"),
    (RERANKER_REPO, AutoModelForSequenceClassification, RERANKER_LOCAL_NAME),
    (
        "deepset/roberta-base-squad2",
        AutoModelForQuestionAnswering,
        "roberta-base-squad2",
    ),
]

if DOWNLOAD_MODERNBERT:
    SPECS.append(
        (
            "kiddothe2b/ModernBERT-base-squad2",
            AutoModelForQuestionAnswering,
            "modernbert-base-squad2",
        )
    )

# v19 hybrid uses the cloud-survived v14 answerer checkpoint.
LLM_REPO = os.getenv("NLP_LLM_REPO", "Qwen/Qwen2.5-7B-Instruct-AWQ")
LLM_LOCAL = "llm"


def _download_retrievers() -> None:
    for repo, model_cls, local_name in SPECS:
        dest = OUT / local_name
        print(f"--> {repo} -> {dest}")
        tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
        mdl = model_cls.from_pretrained(repo, trust_remote_code=True)
        if local_name == "modernbert-base-squad2":
            arch = getattr(mdl.config, "architectures", None)
            print(f"[download_models] ModernBERT QA architectures={arch}")
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
