"""Merge a trained LoRA adapter into the Qwen3-8B base, then quantize the
merged model for T4 inference. v15.1-lora-merged.

Why this script exists: vLLM 0.9.0's Punica LoRA Triton kernels crash on T4
(sm_75) with `LLVM ERROR: Unsupported rounding mode for conversion` when
JIT-compiling `_lora_shrink_kernel`. The base+adapter at runtime path is
therefore broken on our hardware. Merging the adapter into the base weights
offline produces a model that's mathematically equivalent to base+LoRA,
needs no runtime LoRA application, and runs through vLLM's vanilla int4 path.

Pipeline:
  1. CPU merge:    Load Qwen3-8B BF16 on CPU, apply LoRA, merge_and_unload(),
                   save merged BF16 to nlp/models/qwen3-8b-merged-bf16/.
                   ~10 min, ~32 GB system RAM peak.
  2. GPU quantize: Load merged BF16, calibrate using nlp.jsonl prompts, save
                   an int4 W4A16 model to nlp/models/llm-merged/. GPTQ is the
                   default because llm-compressor's AWQ smoothing path hits a
                   Qwen3/T4 failure in this environment.
  3. Verify:       Quick sanity load + generate via vLLM (separate from main
                   container) to confirm the quantized output looks reasonable.

After this completes, build the v15.1 image: the Dockerfile will COPY
nlp/models/llm-merged/ into /workspace/models/llm/ (overrides the
downloaded Qwen3-8B-AWQ), and the nlp/models/lora/ dir is left out of the
bundle (renamed or NLP_LLM_LORA_DIR=""), so vLLM runs as v14d would but
with the fine-tuned weights baked in.

Usage on Workbench:

    pip install autoawq>=0.2.6        # one-time

    python training/nlp/merge_lora_and_quantize.py \\
        --base       Qwen/Qwen3-8B \\
        --lora-dir   nlp/models/lora \\
        --data       /home/jupyter/novice/nlp/nlp.jsonl \\
        --docs       /home/jupyter/novice/nlp/documents \\
        --out-merged nlp/models/qwen3-8b-merged-bf16 \\
        --out-awq    nlp/models/llm-merged \\
        --calib-n    64
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
from pathlib import Path
from typing import Iterable


# ----------------------------------------------------------------- chunking
# (kept in sync with training/nlp/finetune_lora.py — same chunks used to
# build calibration prompts as were used at training time, so AWQ scales
# the same activation distribution.)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
CHUNK_SENTENCES = 3
CHUNK_OVERLAP = 1
MAX_CONTEXT_CHUNKS = 3

SYSTEM_PROMPT = (
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


def _split_sentences(text: str) -> list[str]:
    sents = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    return sents if sents else ([text.strip()] if text.strip() else [])


def _chunk_doc(text: str) -> list[str]:
    sents = _split_sentences(text)
    if not sents:
        return []
    if len(sents) <= CHUNK_SENTENCES:
        return [" ".join(sents)]
    step = max(1, CHUNK_SENTENCES - CHUNK_OVERLAP)
    out: list[str] = []
    for i in range(0, len(sents), step):
        window = sents[i : i + CHUNK_SENTENCES]
        if not window:
            break
        out.append(" ".join(window))
        if i + CHUNK_SENTENCES >= len(sents):
            break
    return out


def _pick_calib_chunks(
    answer: str, source_docs: list[str], docs_dir: Path
) -> list[str]:
    answer_lc = (answer or "").lower().strip()
    chosen: list[str] = []
    fallback: list[str] = []
    for did in source_docs or []:
        doc_path = docs_dir / f"{did}.txt"
        if not doc_path.exists():
            continue
        chunks = _chunk_doc(doc_path.read_text(errors="ignore"))
        if not chunks:
            continue
        added = 0
        if answer_lc:
            for c in chunks:
                if answer_lc in c.lower() and c not in chosen:
                    chosen.append(c)
                    added += 1
                    if added >= 2:
                        break
        if added == 0:
            fallback.append(chunks[0])
        if len(chosen) >= MAX_CONTEXT_CHUNKS:
            return chosen
    for c in fallback:
        if c not in chosen:
            chosen.append(c)
        if len(chosen) >= MAX_CONTEXT_CHUNKS:
            break
    return chosen


def _load_few_shots(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    return list(json.loads(path.read_text()).get("examples", []))


def _build_calib_prompt(
    question: str,
    chunks: list[str],
    few_shots: list[dict[str, str]],
    tokenizer,
) -> str:
    """Render the same chat template the manager uses at inference. Important
    for AWQ — calibration activations must match the inference distribution
    or the quantization scales will be wrong."""
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in few_shots:
        msgs.append(
            {
                "role": "user",
                "content": f"Context:\n{ex['context']}\n\nQuestion: {ex['question']}",
            }
        )
        msgs.append({"role": "assistant", "content": ex["answer"]})
    ctx = "\n\n".join(chunks) if chunks else "(no context)"
    msgs.append(
        {"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {question}"}
    )
    try:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )


def _build_calibration_set(
    data_path: Path,
    docs_dir: Path,
    few_shots_path: Path,
    tokenizer,
    n: int,
) -> list[str]:
    few_shots = _load_few_shots(few_shots_path)
    prompts: list[str] = []
    for line in data_path.read_text().splitlines():
        if not line.strip():
            continue
        ex = json.loads(line)
        if not ex.get("answerable", True):
            continue
        ans = (ex.get("answer") or "").strip()
        docs = ex.get("source_docs") or []
        if not ans or not docs:
            continue
        chunks = _pick_calib_chunks(ans, docs, docs_dir)
        if not chunks:
            continue
        prompts.append(_build_calib_prompt(ex["question"], chunks, few_shots, tokenizer))
        if len(prompts) >= n:
            break
    return prompts


def _late_down_proj_ignores(start_layer: int) -> list[str]:
    if start_layer < 0:
        return []
    # Qwen3-8B has 36 decoder layers. This helper intentionally returns
    # explicit module names instead of regexes because llm-compressor's ignore
    # handling has been more predictable with exact paths across releases.
    return [f"model.layers.{i}.mlp.down_proj" for i in range(start_layer, 36)]


# ----------------------------------------------------------------- pipeline


def step1_merge(args) -> None:
    print(
        f"\n[merge] STEP 1 — load base + adapter on CPU, merge to BF16\n"
        f"        base:    {args.base}\n"
        f"        adapter: {args.lora_dir}\n"
        f"        out:     {args.out_merged}\n",
        flush=True,
    )

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # CPU-only load. low_cpu_mem_usage=True streams shards to avoid 2× RAM peak.
    base = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    base.eval()

    print("[merge] applying LoRA adapter...", flush=True)
    model = PeftModel.from_pretrained(base, args.lora_dir)
    print("[merge] merge_and_unload()...", flush=True)
    merged = model.merge_and_unload()

    out_merged = Path(args.out_merged)
    out_merged.mkdir(parents=True, exist_ok=True)
    print(f"[merge] saving merged BF16 to {out_merged}...", flush=True)
    merged.save_pretrained(out_merged, safe_serialization=True)

    # The adapter dir has the tokenizer (saved by SFTTrainer). Fall back to
    # base if it's missing for some reason.
    tok_src = args.lora_dir if (Path(args.lora_dir) / "tokenizer.json").exists() else args.base
    tok = AutoTokenizer.from_pretrained(tok_src, use_fast=True)
    tok.save_pretrained(out_merged)
    print(f"[merge] step 1 done. Merged BF16 at {out_merged}\n", flush=True)

    # Free CPU RAM before step 2 spins up the quantization stack.
    del base, model, merged
    gc.collect()


def step2_quantize(args) -> None:
    """Quantize the merged BF16 model to int4 using llm-compressor.

    AutoAWQ is officially deprecated as of 2025 — its `__init__.py` prints
    a farewell message and `from awq import AutoAWQForCausalLM` raises
    ImportError in the final dev release. The deprecation message itself
    points users at `vllm-project/llm-compressor` as the successor; that
    package produces `compressed-tensors`-format weights which vLLM 0.9
    loads natively.

    Memory budget on T4 (16 GB) is tight: the merged Qwen3-8B is ~16 GB BF16.
    We use device_map="auto" to offload most of the model to CPU, default to
    GPTQ W4A16 to avoid AWQ's smoothing/propagation path, and cap calibration
    length at 512 tokens so the attention path stays comfortably below T4 RAM.
    """
    # Reduce CUDA-allocator fragmentation; suppress the FastTokenizer
    # threading warning that fires during calibration.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    print(
        f"\n[merge] STEP 2 — {args.quant_method.upper()}-quantize merged model "
        f"on GPU (llm-compressor)\n"
        f"        in:        {args.out_merged}\n"
        f"        out:       {args.out_awq}\n"
        f"        calib-n:   {args.calib_n}\n"
        f"        seq-len:   {args.max_seq_length}\n",
        flush=True,
    )

    try:
        from llmcompressor import oneshot  # type: ignore
        from llmcompressor.modifiers.awq import AWQModifier  # type: ignore
        try:
            from llmcompressor.modifiers.gptq import GPTQModifier  # type: ignore
        except ImportError:
            from llmcompressor.modifiers.quantization import GPTQModifier  # type: ignore
    except ImportError as exc:
        sys.exit(
            f"llm-compressor not available ({exc}).\n"
            "Install: pip install llmcompressor\n"
            "If pip resolves a torch version that breaks your env, use:\n"
            "  pip install llmcompressor --no-deps\n"
            "then install missing top-level deps individually."
        )

    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(args.out_merged), use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print("[merge] building calibration set...", flush=True)
    calib_prompts = _build_calibration_set(
        Path(args.data),
        Path(args.docs),
        Path(args.few_shots),
        tok,
        n=args.calib_n,
    )
    print(f"[merge] calibration set: {len(calib_prompts)} prompts", flush=True)
    if not calib_prompts:
        sys.exit("no calibration prompts built; check --data and --docs paths")

    # llm-compressor expects a HF dataset with a `text` column it can tokenize.
    calib_ds = Dataset.from_list([{"text": p} for p in calib_prompts])

    if args.quant_method == "awq":
        # Closest to autoawq's GEMM-W4A16 config, but this currently fails on
        # Qwen3/T4 after a few calibration groups. Keep it for non-T4 retries.
        recipe = [
            AWQModifier(
                targets="Linear",
                scheme="W4A16_ASYM",
                ignore=["lm_head"],
            ),
        ]
        sequential_targets = ["Linear"]
        quantizer_label = "llmcompressor AWQModifier W4A16_ASYM g128"
    else:
        # Official llm-compressor W4A16 path. It avoids AWQ's smoothing /
        # propagation pass, which is where the Qwen3 GQA NoneType failure
        # happens. offload_hessians=True trades runtime for lower VRAM.
        ignore = ["lm_head", *_late_down_proj_ignores(args.gptq_ignore_down_proj_from_layer)]
        recipe = [
            GPTQModifier(
                targets="Linear",
                scheme="W4A16",
                ignore=ignore,
                block_size=64,
                dampening_frac=0.01,
                offload_hessians=True,
            ),
        ]
        sequential_targets = ["Linear"]
        if args.gptq_ignore_down_proj_from_layer >= 0:
            skipped = f"; ignored down_proj from layer {args.gptq_ignore_down_proj_from_layer}"
        else:
            skipped = ""
        quantizer_label = f"llmcompressor GPTQModifier W4A16 g128 block64{skipped}"

    out_awq = Path(args.out_awq)
    out_awq.mkdir(parents=True, exist_ok=True)

    # llm-compressor 0.10's `oneshot()` parses its kwargs through
    # HfArgumentParser, which rejects unknown keys like `model_kwargs`.
    # The supported pattern is: pre-load the model with the desired
    # device_map / dtype, then pass the loaded instance to `oneshot`.
    # That gives us the CPU-offloading we need for an 8B BF16 on a 16 GB
    # T4 without fighting the argparser.
    print("[merge] loading merged model for quantization...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(args.out_merged),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    print("[merge] running calibration + quantization...", flush=True)
    oneshot_kwargs = dict(
        model=model,
        dataset=calib_ds,
        recipe=recipe,
        output_dir=str(out_awq),
        max_seq_length=args.max_seq_length,
        num_calibration_samples=args.calib_n,
        batch_size=1,
    )
    if sequential_targets is not None:
        oneshot_kwargs["sequential_targets"] = sequential_targets
    oneshot(**oneshot_kwargs)

    tok.save_pretrained(str(out_awq))

    # Write a marker noting what produced this dir.
    (out_awq / "MERGED_FROM").write_text(
        f"base={args.base}\nadapter={args.lora_dir}\ncalib_n={args.calib_n}\n"
        f"max_seq_length={args.max_seq_length}\nquant_method={args.quant_method}\n"
        f"gptq_ignore_down_proj_from_layer={args.gptq_ignore_down_proj_from_layer}\n"
        f"quantizer={quantizer_label}\n"
    )
    print(f"[merge] step 2 done. Quantized merged model at {out_awq}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="Qwen/Qwen3-8B")
    parser.add_argument("--lora-dir", default="nlp/models/lora")
    parser.add_argument(
        "--data", default="/home/jupyter/novice/nlp/nlp.jsonl",
        help="nlp.jsonl used to build calibration prompts (same as training).",
    )
    parser.add_argument(
        "--docs", default="/home/jupyter/novice/nlp/documents",
        help="Corpus document dir, used to look up source chunks for calibration prompts.",
    )
    parser.add_argument(
        "--few-shots",
        default=str(Path(__file__).resolve().parents[2] / "nlp/src/few_shots.json"),
    )
    parser.add_argument("--out-merged", default="nlp/models/qwen3-8b-merged-bf16")
    parser.add_argument("--out-awq", default="nlp/models/llm-merged")
    parser.add_argument(
        "--calib-n", type=int, default=32,
        help="Number of calibration prompts. 32 is the T4-safe default; more "
        "than ~128 has diminishing returns and slows quantization.",
    )
    parser.add_argument(
        "--max-seq-length", type=int, default=256,
        help="Calibration token length. 256 is the T4-safe default for the "
        "merged 8B BF16 model; raise only on larger GPUs.",
    )
    parser.add_argument(
        "--quant-method",
        choices=["gptq", "awq"],
        default="gptq",
        help="Quantizer to use. gptq is the T4-safe default; awq is kept for "
        "larger GPUs because llm-compressor AWQ fails on Qwen3/T4 here.",
    )
    parser.add_argument(
        "--gptq-ignore-down-proj-from-layer",
        type=int,
        default=-1,
        help="Emergency T4 escape hatch. If GPTQ OOMs late in mlp.down_proj, "
        "set this to that layer index (for example 29) to leave remaining "
        "down_proj layers unquantized while still producing a bootable model.",
    )
    parser.add_argument(
        "--skip-merge", action="store_true",
        help="Skip step 1 (use existing --out-merged dir, e.g. for re-quantizing).",
    )
    parser.add_argument(
        "--skip-quantize", action="store_true",
        help="Skip step 2 (produce merged BF16 only, e.g. for inspection).",
    )
    args = parser.parse_args()

    if not Path(args.lora_dir).is_dir():
        sys.exit(f"adapter dir not found: {args.lora_dir}")
    if not (Path(args.lora_dir) / "adapter_config.json").exists():
        sys.exit(f"no adapter_config.json in {args.lora_dir}")

    if not args.skip_merge:
        step1_merge(args)
    else:
        print(f"[merge] step 1 skipped; using existing {args.out_merged}", flush=True)
        if not Path(args.out_merged).is_dir():
            sys.exit(f"--skip-merge but {args.out_merged} doesn't exist")

    if not args.skip_quantize:
        step2_quantize(args)
    else:
        print("[merge] step 2 skipped", flush=True)

    print("\n[merge] all done.\n", flush=True)
    print("Next:")
    print(f"  1. Verify the AWQ model looks right:")
    print(f"     ls -la {args.out_awq}/")
    print(f"  2. Move the now-stale runtime-LoRA dir out of the build context:")
    print(f"     mv nlp/models/lora nlp/models/lora-archive")
    print(f"  3. Update Dockerfile to copy {args.out_awq}/ over /workspace/models/llm/")
    print(f"  4. til build nlp v15-merged-qwen3-8b && til test nlp v15-merged-qwen3-8b")


if __name__ == "__main__":
    main()
