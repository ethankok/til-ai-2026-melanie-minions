"""v14-llm-rag: vLLM-backed answerer using Qwen2.5-7B-Instruct-AWQ.

Replaces the v9 extractive RoBERTa head. Retrieval and rerank are unchanged
upstream — this module only takes (question, ordered list of chunks) tuples
and returns a short answer string per question.

Why this exists: v9's extractive head is structurally capped by the corpus —
481/883 local gold answers are not literal source substrings, and v13a's
oracle showed the candidate pool has +0.10 of headroom we couldn't pick.
A strong instruction-tuned LLM can return canonical answer forms (dates,
codenames, units) that the ModernBERT AE @ 0.9 evaluator accepts.

Why this is not v8a-genqa: v8a used Flan-T5-base (250M params), which both
paraphrased away from source phrasing and lacked the instruction-following
needed to obey 'quote the exact wording'. Qwen2.5-7B-Instruct-AWQ is a
different model class — 7B params, AWQ-quantised to ~5 GB, with strong
multi-turn instruction following.

Speed notes:
- vLLM's continuous batching is the speed lever. The server batches all
  questions in a single /nlp request into one .generate() call.
- gpu_memory_utilization is tuned conservatively so the BGE retriever and
  bge-reranker-base still fit on the same T4 alongside the LLM.
- A warmup generation runs inside load_corpus() so the first real question
  doesn't pay CUDA-graph capture cost during the timed phase.
"""

from __future__ import annotations

import json
import os
import string
from pathlib import Path
from typing import Sequence

# Lazy-loaded heavy deps: vllm is multi-second to import; defer until first use.
_VLLM_IMPORTED = False
_LLM = None
_SamplingParams = None


def _import_vllm() -> None:
    global _VLLM_IMPORTED, _LLM, _SamplingParams
    if _VLLM_IMPORTED:
        return
    from vllm import LLM, SamplingParams  # type: ignore

    _LLM = LLM
    _SamplingParams = SamplingParams
    _VLLM_IMPORTED = True


_PRINTABLE = set(string.printable)
_MAX_ANSWER_TOKENS = 64


def _clean_answer(text: str) -> str:
    return "".join(c for c in (text or "") if c in _PRINTABLE).strip()


def _clip_words(text: str, limit: int = _MAX_ANSWER_TOKENS) -> str:
    words = (text or "").strip().split()
    if len(words) <= limit:
        return (text or "").strip()
    return " ".join(words[:limit]).strip()


_DEFAULT_SYSTEM_PROMPT = (
    "You are an extractive question-answering assistant for the world of "
    "Clairos.\n"
    "Rules:\n"
    "1. Answer ONLY from the provided context. If the context does not "
    "contain the answer, return an empty string.\n"
    "2. Quote the answer using the EXACT wording, dates, numbers, and units "
    "from the context. Do not paraphrase. Do not add explanation. Do not "
    "wrap the answer in quotes.\n"
    "3. Keep the answer as short as possible — typically 1 to 8 words, "
    "never more than a single short sentence.\n"
    "4. For dates use the source format (e.g. 76-07-19, Q4 78 PCE). For "
    "money use the source units (e.g. 4.5 million Phi Credits). For "
    "codenames return the uppercase token only (e.g. SEASTITCH).\n"
    "5. Output only the answer, nothing else."
)


class LLMAnswerer:
    """vLLM wrapper for Qwen2.5-7B-Instruct-AWQ (or compatible)."""

    def __init__(
        self,
        model_dir: str | os.PathLike,
        few_shots_path: str | os.PathLike | None = None,
        max_context_chunks: int = 3,
        gpu_memory_utilization: float | None = None,
        max_model_len: int = 4096,
        max_new_tokens: int = 48,
    ) -> None:
        self.model_dir = str(model_dir)
        self.max_context_chunks = max_context_chunks
        self.max_new_tokens = max_new_tokens

        self._few_shots: list[dict] = []
        if few_shots_path is None:
            few_shots_path = Path(__file__).parent / "few_shots.json"
        few_shots_path = Path(few_shots_path)
        if few_shots_path.exists():
            blob = json.loads(few_shots_path.read_text())
            self._few_shots = list(blob.get("examples", []))
            print(
                f"[llm_answerer] loaded {len(self._few_shots)} few-shots "
                f"from {few_shots_path}",
                flush=True,
            )
        else:
            print(
                f"[llm_answerer] no few-shots file at {few_shots_path}; "
                f"running zero-shot",
                flush=True,
            )

        _import_vllm()

        gmu = (
            gpu_memory_utilization
            if gpu_memory_utilization is not None
            else float(os.getenv("NLP_LLM_GPU_MEM_FRACTION", "0.78"))
        )
        print(
            f"[llm_answerer] loading vLLM from {self.model_dir} "
            f"(gpu_memory_utilization={gmu}, max_model_len={max_model_len})",
            flush=True,
        )
        # awq_marlin gives the fastest AWQ kernels on Ampere/Ada/Hopper.
        # vLLM auto-falls-back to the standard AWQ kernel on older GPUs.
        self.llm = _LLM(  # type: ignore[misc]
            model=self.model_dir,
            quantization="awq_marlin",
            dtype="float16",
            gpu_memory_utilization=gmu,
            max_model_len=max_model_len,
            enforce_eager=False,
            disable_log_stats=True,
            trust_remote_code=False,
        )
        self.tokenizer = self.llm.get_tokenizer()
        # Greedy decoding + short cap is what prevents paraphrase past the
        # ModernBERT AE 0.9 threshold (the v8a failure mode).
        self.sampling = _SamplingParams(  # type: ignore[misc]
            temperature=0.0,
            top_p=1.0,
            max_tokens=self.max_new_tokens,
            stop=["\n\n", "\nQuestion:", "\nContext:"],
        )

    # ----------------------------------------------------------------- prompt

    def _build_messages(self, question: str, chunks: Sequence[str]) -> list[dict]:
        messages: list[dict] = [
            {"role": "system", "content": _DEFAULT_SYSTEM_PROMPT}
        ]
        for ex in self._few_shots:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Context:\n{ex['context']}\n\nQuestion: {ex['question']}"
                    ),
                }
            )
            messages.append({"role": "assistant", "content": ex["answer"]})

        ctx_chunks = list(chunks)[: self.max_context_chunks]
        ctx_block = "\n\n".join(ctx_chunks) if ctx_chunks else "(no context)"
        messages.append(
            {
                "role": "user",
                "content": f"Context:\n{ctx_block}\n\nQuestion: {question}",
            }
        )
        return messages

    def _build_prompt(self, question: str, chunks: Sequence[str]) -> str:
        messages = self._build_messages(question, chunks)
        return self.tokenizer.apply_chat_template(  # type: ignore[no-any-return]
            messages, tokenize=False, add_generation_prompt=True
        )

    # ---------------------------------------------------------------- inference

    def answer_batch(
        self,
        questions: Sequence[str],
        chunks_per_question: Sequence[Sequence[str]],
    ) -> list[str]:
        if not questions:
            return []
        assert len(questions) == len(chunks_per_question)
        prompts = [
            self._build_prompt(q, c)
            for q, c in zip(questions, chunks_per_question)
        ]
        # vLLM continuously batches whatever we hand it.
        outputs = self.llm.generate(prompts, self.sampling, use_tqdm=False)
        answers: list[str] = []
        for out in outputs:
            text = out.outputs[0].text if out.outputs else ""
            answers.append(_clean_answer(_clip_words(text)))
        return answers

    def answer(self, question: str, chunks: Sequence[str]) -> str:
        return self.answer_batch([question], [chunks])[0]

    # -------------------------------------------------------------------- ops

    def warmup(self) -> None:
        """Prime the engine so the first real /nlp question doesn't pay
        CUDA-graph capture cost during the timed phase. Runs inside
        load_corpus(), which the evaluator polls until 'loaded' — so the
        cost lands in the untimed phase."""
        try:
            _ = self.answer_batch(
                ["What is the codename?"],
                [["The classified annex codename is WARMUP."]],
            )
            print("[llm_answerer] warmup complete", flush=True)
        except Exception as exc:  # warmup is best-effort
            print(f"[llm_answerer] warmup failed (non-fatal): {exc}", flush=True)


def llm_dir_is_ready(path: str | os.PathLike) -> bool:
    p = Path(path)
    return p.is_dir() and (p / "config.json").exists()
