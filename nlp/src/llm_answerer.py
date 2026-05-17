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
_LoRARequest = None


def _import_vllm() -> None:
    global _VLLM_IMPORTED, _LLM, _SamplingParams, _LoRARequest
    if _VLLM_IMPORTED:
        return
    from vllm import LLM, SamplingParams  # type: ignore
    from vllm.lora.request import LoRARequest  # type: ignore

    _LLM = LLM
    _SamplingParams = SamplingParams
    _LoRARequest = LoRARequest
    _VLLM_IMPORTED = True


def _lora_dir_is_ready(path: str | os.PathLike) -> bool:
    p = Path(path)
    if not p.is_dir():
        return False
    # peft saves adapter_config.json + adapter_model.safetensors
    return (p / "adapter_config.json").exists()


_PRINTABLE = set(string.printable)
_MAX_ANSWER_TOKENS = 64
# Prefixes the model sometimes emits even when the prompt forbids them.
# Stripped case-insensitively from the start of the output, once.
_ANSWER_PREFIXES = (
    "answer:",
    "the answer is",
    "answer is",
    "a:",
)


def _clean_answer(text: str) -> str:
    return "".join(c for c in (text or "") if c in _PRINTABLE).strip()


def _clip_words(text: str, limit: int = _MAX_ANSWER_TOKENS) -> str:
    words = (text or "").strip().split()
    if len(words) <= limit:
        return (text or "").strip()
    return " ".join(words[:limit]).strip()


def _strip_boilerplate(text: str) -> str:
    """Light post-processing for the LLM's raw output. Keep this minimal —
    over-aggressive normalisation has historically regressed cloud AE @ 0.9
    (see v11-canonical-answer in NOTES). All we do here:
      - Strip a single leading "Answer:" / "The answer is" prefix
      - Strip matched surrounding quotes/parens
      - Strip a single trailing period if the body has no internal period
    """
    s = (text or "").strip()
    if not s:
        return s

    lower = s.lower()
    for prefix in _ANSWER_PREFIXES:
        if lower.startswith(prefix):
            s = s[len(prefix):].lstrip(" \t:-")
            break

    if len(s) >= 2 and s[0] == s[-1] and s[0] in {'"', "'", "`"}:
        s = s[1:-1].strip()
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1].strip()

    # Trim trailing period only if the answer is a single statement.
    if s.endswith(".") and "." not in s[:-1]:
        s = s[:-1].strip()
    return s


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
    """vLLM wrapper for an instruction-tuned chat model (Qwen2.5/Qwen3/etc).

    Default expectation as of v14c is Qwen3-4B-Instruct-2507 (AWQ-4bit) —
    the "2507" suffix is the non-thinking instruct release, so the chat
    template doesn't emit <think>...</think> blocks. apply_chat_template
    is called with enable_thinking=False as a defensive belt-and-braces
    measure; tokenizers that don't know the kwarg ignore it silently.
    """

    def __init__(
        self,
        model_dir: str | os.PathLike,
        few_shots_path: str | os.PathLike | None = None,
        lora_dir: str | os.PathLike | None = None,
        max_context_chunks: int = 3,
        gpu_memory_utilization: float | None = None,
        max_model_len: int = 4096,
        max_new_tokens: int = 32,
        max_lora_rank: int = 16,
    ) -> None:
        self.model_dir = str(model_dir)
        self.max_context_chunks = max_context_chunks
        self.max_new_tokens = max_new_tokens
        self.max_lora_rank = max_lora_rank

        # v15-lora: detect a bundled adapter. If NLP_LLM_LORA_DIR is set and
        # points at a directory with adapter_config.json, we'll enable LoRA in
        # the vLLM engine and apply this adapter to every generate call.
        if lora_dir is None:
            lora_dir = os.getenv("NLP_LLM_LORA_DIR", "")
        self.lora_dir: str | None = (
            str(lora_dir)
            if lora_dir and _lora_dir_is_ready(lora_dir)
            else None
        )
        if self.lora_dir:
            print(
                f"[llm_answerer] LoRA adapter detected at {self.lora_dir} "
                f"(max_rank={self.max_lora_rank})",
                flush=True,
            )
        else:
            print(
                "[llm_answerer] no LoRA adapter (running base model only)",
                flush=True,
            )

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
        # Quantization kernel selection. Marlin AWQ is fastest on Ampere+
        # (sm_80+) but requires sm_80; T4 (sm_75 / Turing) hangs or falls
        # back slowly. Default to plain "awq" which works on every CUDA arch
        # vLLM supports. Override with NLP_LLM_QUANT=awq_marlin on Ampere.
        quant = os.getenv("NLP_LLM_QUANT", "awq").strip()
        # CUDA-graph capture (~60-120s for a 7B on T4) is the dominant
        # corpus-load cost. enforce_eager=True skips capture entirely; we
        # pay ~10-20% per-token latency in exchange. With 700 questions and
        # 48-token outputs, eager is still well inside the speed budget.
        # Override with NLP_LLM_ENFORCE_EAGER=0 to enable graph capture once
        # we know the rest of the pipeline boots.
        enforce_eager = os.getenv("NLP_LLM_ENFORCE_EAGER", "1").lower() not in {
            "0",
            "false",
            "no",
        }
        print(
            f"[llm_answerer] loading vLLM from {self.model_dir} "
            f"(quantization={quant}, gpu_memory_utilization={gmu}, "
            f"max_model_len={max_model_len}, enforce_eager={enforce_eager})",
            flush=True,
        )
        # v15-lora: enable LoRA in the engine if we have an adapter on disk.
        # max_loras=1 + max_lora_rank=16 matches our QLoRA training config.
        # Enabling LoRA adds a small per-request overhead (~5-10%) on every
        # request — only do it if we actually have an adapter.
        llm_kwargs: dict = dict(
            model=self.model_dir,
            quantization=quant,
            dtype="float16",
            gpu_memory_utilization=gmu,
            max_model_len=max_model_len,
            enforce_eager=enforce_eager,
            disable_log_stats=True,
            trust_remote_code=False,
        )
        if self.lora_dir:
            llm_kwargs.update(
                enable_lora=True,
                max_loras=1,
                max_lora_rank=self.max_lora_rank,
            )
        self.llm = _LLM(**llm_kwargs)  # type: ignore[misc]
        self.tokenizer = self.llm.get_tokenizer()

        # Pre-build the LoRARequest once. Numeric ID is arbitrary but must
        # be stable across calls. vLLM caches loaded adapters by ID so this
        # path-load cost is amortised.
        self._lora_request = None
        if self.lora_dir:
            self._lora_request = _LoRARequest(  # type: ignore[misc]
                lora_name="v15-lora",
                lora_int_id=1,
                lora_path=self.lora_dir,
            )
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
        # enable_thinking=False is a Qwen3-specific kwarg that suppresses
        # the <think>...</think> reasoning trace. Models that don't know
        # the kwarg silently ignore it. Belt-and-braces in case a future
        # repo flip puts us on a thinking-by-default variant.
        try:
            return self.tokenizer.apply_chat_template(  # type: ignore[no-any-return]
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
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
        # vLLM continuously batches whatever we hand it. Pass the LoRA
        # adapter (if any) so the engine applies it to every prompt.
        gen_kwargs: dict = {"use_tqdm": False}
        if self._lora_request is not None:
            gen_kwargs["lora_request"] = self._lora_request
        outputs = self.llm.generate(prompts, self.sampling, **gen_kwargs)
        answers: list[str] = []
        for out in outputs:
            text = out.outputs[0].text if out.outputs else ""
            answers.append(_clean_answer(_clip_words(_strip_boilerplate(text))))
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
