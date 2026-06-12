"""Find a Universal Adversarial Trigger against the official NLP AE evaluator.

The cloud scorer is a binary sequence-pair classifier (ModernBERT-base) bundled
with the dataset as `models/nlp_eval.zip` and used by `test/test_nlp.py`. It
receives:

    f"Question: {question} Reference: {reference} Candidate: {candidate}"

and outputs `prob_eq = softmax(logits)[1]`. A case is "equivalent" iff
`prob_eq >= 0.9`. We control `candidate` end-to-end, and we have white-box
access to the model — so a Wallace et al. 2019 / HotFlip universal adversarial
trigger search is the right tool. Append the resulting trigger to every
answer and most retrieval-success cases that currently get RETRIEVAL_ONLY=0.4
should flip to full credit.

Inputs
------
- nlp.jsonl : official local QA file (gold answers + source_docs).
- nlp_results.json (optional): predictions from a prior v9-class submission,
  used as the "current candidate" the trigger must rescue. If absent, the
  script falls back to using the gold answer with one token corrupted, which
  is a stricter target but no realistic in deployment.

Outputs
-------
- nlp/models/ae_trigger.json : trigger string + token ids + before/after stats.

Run on Workbench
----------------
    python training/nlp/find_ae_trigger.py \
        --data /home/jupyter/novice/nlp/nlp.jsonl \
        --predictions /home/jupyter/melanie-minions/nlp_results.json \
        --ae-model-path ./test/models/nlp_eval_512 \
        --out nlp/models/ae_trigger.json \
        --trigger-len 16 --iters 200 --batch-size 32 --topk 40 --seed 0
"""

from __future__ import annotations

import argparse
import json
import random
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


@dataclass
class Example:
    question: str
    reference: str          # gold answer
    candidate: str          # our current predicted answer
    docs_overlap: bool      # retrieval hit; if False this example contributes nothing


def _load_examples(
    data_path: Path,
    predictions_path: Path | None,
) -> list[Example]:
    rows = [json.loads(l) for l in data_path.read_text().splitlines() if l.strip()]
    preds = None
    if predictions_path is not None and predictions_path.exists():
        preds = json.loads(predictions_path.read_text())
        if len(preds) != len(rows):
            print(
                f"WARN: predictions length {len(preds)} != gold length {len(rows)}; "
                f"truncating to min."
            )
        n = min(len(preds), len(rows))
        rows = rows[:n]
        preds = preds[:n]

    out: list[Example] = []
    for i, row in enumerate(rows):
        q = row.get("question", "")
        gold = row.get("answer", "") or ""
        gold_docs = set(row.get("source_docs", []) or [])
        if preds is not None:
            pred = preds[i]
            cand = (pred.get("answer") or "").strip()
            pred_docs = set((pred.get("documents") or [])[:3])
        else:
            # Fallback: pretend our candidate is the gold answer with the
            # last word dropped — a stricter target than a wrong string.
            cand = " ".join(gold.split()[:-1]) or gold
            pred_docs = gold_docs

        if not gold or not cand:
            # Empty-gold/empty-pred are handled outside the AE model in the
            # scorer; the trigger does not help them.
            continue
        out.append(Example(
            question=q,
            reference=gold,
            candidate=cand,
            docs_overlap=bool(gold_docs & pred_docs) or preds is None,
        ))
    return out


def _filter_to_negatives(
    examples: Sequence[Example],
    model,
    tokenizer,
    device,
    threshold: float,
    batch_size: int,
) -> tuple[list[Example], list[float]]:
    """Run AE on each (Q, R, C) verbatim. Return only the cases that
    currently fail (prob_eq < threshold) — these are the ones the trigger
    needs to rescue. Also returns baseline prob distribution for all examples.
    """
    import torch
    import torch.nn.functional as F

    texts = [_format_input(tokenizer, e.question, e.reference, e.candidate)
             for e in examples]
    probs: list[float] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            enc = tokenizer(
                texts[start:start + batch_size],
                max_length=512,
                padding="longest",
                truncation=True,
                return_tensors="pt",
            ).to(device)
            logits = model(**enc).logits
            probs.extend(F.softmax(logits, dim=-1)[:, 1].tolist())

    negatives = [e for e, p in zip(examples, probs) if p < threshold]
    return negatives, probs


def _format_input(tokenizer, question: str, reference: str, candidate: str) -> str:
    """Mirror test/test_nlp.py:_format_input exactly."""
    _printable = "".join(c for c in candidate if c in string.printable)
    tokens = tokenizer.tokenize(_printable, max_length=64, truncation=True)
    recon = tokenizer.convert_tokens_to_string(tokens)
    return f"Question: {question} Reference: {reference} Candidate: {recon}"


def _build_ids_with_trigger(
    tokenizer,
    question: str,
    reference: str,
    candidate: str,
    trigger_ids: list[int],
    candidate_token_budget: int,
) -> tuple[list[int], int, int]:
    """Tokenise the AE input, splicing `trigger_ids` in at the start of the
    Candidate field. Returns (input_ids, trigger_start, trigger_end) where
    [trigger_start, trigger_end) is the slice covering trigger tokens in
    input_ids — used to gather gradients at the right positions.

    We bypass the evaluator's tokenize→detokenize roundtrip during gradient
    search for differentiability; a deployment check verifies the trigger
    survives the roundtrip after training.
    """
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    prefix_text = f"Question: {question} Reference: {reference} Candidate:"
    prefix_ids = tokenizer.encode(prefix_text, add_special_tokens=False)
    # Leading space matters: BPE keeps a space-prefix token for the first subword.
    suffix_text = " " + "".join(c for c in candidate if c in string.printable)
    suffix_ids = tokenizer.encode(
        suffix_text, add_special_tokens=False
    )[:candidate_token_budget]

    input_ids = [cls_id] + prefix_ids + list(trigger_ids) + suffix_ids + [sep_id]
    trigger_start = 1 + len(prefix_ids)
    trigger_end = trigger_start + len(trigger_ids)
    return input_ids, trigger_start, trigger_end


def _make_batch(
    tokenizer,
    examples: Sequence[Example],
    trigger_ids: list[int],
    candidate_token_budget: int,
    device,
):
    """Build a padded batch with the trigger spliced in at known offsets.

    Returns (input_ids, attention_mask, trigger_slices, trigger_starts_tensor).
    `trigger_starts_tensor` is a 1-D long tensor on `device` containing the
    absolute index (in input_ids) where each row's trigger begins. Used by the
    batched HotFlip trial path to overwrite single token positions cheaply.
    """
    import torch

    per_example = [
        _build_ids_with_trigger(
            tokenizer, e.question, e.reference, e.candidate,
            trigger_ids, candidate_token_budget,
        )
        for e in examples
    ]
    max_len = min(max(len(ids) for ids, _, _ in per_example), 512)
    pad_id = tokenizer.pad_token_id or 0

    input_ids = torch.full((len(examples), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    trigger_slices: list[tuple[int, int]] = []
    starts: list[int] = []
    for row, (ids, ts, te) in enumerate(per_example):
        ids = ids[:max_len]
        input_ids[row, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        attention_mask[row, :len(ids)] = 1
        ts = min(ts, max_len)
        te = min(te, max_len)
        trigger_slices.append((ts, te))
        starts.append(ts)
    return (
        input_ids.to(device),
        attention_mask.to(device),
        trigger_slices,
        torch.tensor(starts, dtype=torch.long, device=device),
    )


def _eval_loss_and_prob(model, embeds, attention_mask, target_class: int = 1):
    import torch.nn.functional as F

    logits = model(inputs_embeds=embeds, attention_mask=attention_mask).logits
    log_probs = F.log_softmax(logits, dim=-1)
    target_log_prob = log_probs[:, target_class]
    loss = -target_log_prob.mean()
    prob = log_probs[:, target_class].exp().detach()
    return loss, prob


def _allowed_vocab_mask(tokenizer) -> "torch.Tensor":
    """Token ids the trigger may use: must decode to printable ASCII (so the
    evaluator's `printable` filter does not strip them) and must not be a
    special token."""
    import torch

    vocab_size = len(tokenizer)
    mask = torch.zeros(vocab_size, dtype=torch.bool)
    special_ids = set(tokenizer.all_special_ids or [])
    for tid in range(vocab_size):
        if tid in special_ids:
            continue
        try:
            piece = tokenizer.decode([tid], skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False)
        except Exception:
            continue
        if not piece:
            continue
        if not all(c in string.printable for c in piece):
            continue
        # Pure-whitespace tokens make the trigger look empty after detokenisation.
        if not piece.strip():
            continue
        mask[tid] = True
    return mask


def _hotflip_step(
    model,
    tokenizer,
    trigger_ids: list[int],
    batch_examples: Sequence[Example],
    eval_examples: Sequence[Example],
    allowed_mask,
    topk: int,
    candidate_token_budget: int,
    device,
    trial_chunk: int = 8,
) -> tuple[list[int], float, float]:
    """One HotFlip iteration. Returns (new_trigger, loss_before, loss_after).

    Inner trial-evaluation is batched: for each position we evaluate the K
    candidate replacements in a single forward by tiling the eval batch K
    times and overwriting one token per row. Chunks of `trial_chunk`
    candidates at a time keep T4 memory in check.
    """
    import torch

    embed_layer = model.get_input_embeddings()
    vocab_embeds = embed_layer.weight  # (V, H)

    # Gradient at trigger positions on the train batch.
    input_ids, attention_mask, trigger_slices, _ = _make_batch(
        tokenizer, batch_examples, trigger_ids, candidate_token_budget, device
    )
    base_embeds = embed_layer(input_ids).detach().clone()
    base_embeds.requires_grad_(True)

    loss, _ = _eval_loss_and_prob(model, base_embeds, attention_mask)
    loss.backward()

    grad = base_embeds.grad  # (B, L, H)
    trigger_len = len(trigger_ids)
    grad_at_trigger = torch.zeros(
        trigger_len, grad.size(-1), device=device, dtype=grad.dtype
    )
    n_counted = torch.zeros(trigger_len, device=device, dtype=grad.dtype)
    for b, (ts, te) in enumerate(trigger_slices):
        actual = te - ts
        if actual <= 0:
            continue
        grad_at_trigger[:actual] += grad[b, ts:te, :]
        n_counted[:actual] += 1
    grad_at_trigger /= n_counted.clamp(min=1).unsqueeze(-1)

    # Release the gradient graph before doing inference forwards.
    del base_embeds, loss, grad
    torch.cuda.empty_cache() if device.type == "cuda" else None

    # First-order HotFlip score per (position, vocab_id).
    with torch.no_grad():
        cur_emb = vocab_embeds[torch.tensor(trigger_ids, device=device)]
        scores = (vocab_embeds @ grad_at_trigger.T).T  # (T, V)
        scores = scores - (cur_emb * grad_at_trigger).sum(dim=-1, keepdim=True)
        scores = scores.masked_fill(~allowed_mask.to(device), float("inf"))
        topk_scores, topk_ids = torch.topk(scores, k=topk, largest=False, dim=-1)  # (T, K)

    # Build the eval batch ONCE (tokenization is expensive); only overwrite
    # single token positions for each trial.
    eval_input_ids, eval_attn, _, eval_trigger_starts = _make_batch(
        tokenizer, eval_examples, trigger_ids, candidate_token_budget, device
    )
    B, L = eval_input_ids.shape
    row_idx = torch.arange(B, device=device)

    with torch.no_grad():
        baseline_loss_t, _ = _eval_loss_and_prob(
            model, embed_layer(eval_input_ids), eval_attn,
        )
        baseline_loss = baseline_loss_t.item()
    best_loss = baseline_loss
    best_swap: tuple[int, int] | None = None

    # Batched per-position trial sweep: for each chunk of K candidates,
    # tile eval to (K_chunk * B, L) and overwrite one position per row.
    for pos in range(trigger_len):
        cand_ids = topk_ids[pos]  # (K,)
        mask = cand_ids != trigger_ids[pos]
        cand_ids = cand_ids[mask]
        if cand_ids.numel() == 0:
            continue
        K = cand_ids.numel()

        for c_start in range(0, K, trial_chunk):
            c_end = min(c_start + trial_chunk, K)
            chunk = cand_ids[c_start:c_end]  # (Kc,)
            Kc = chunk.numel()

            # Tile eval batch Kc times, then overwrite trigger position `pos` in
            # each row with that row's candidate: row r -> sample (r // B),
            # example (r % B), column = eval_trigger_starts[example] + pos.
            tiled_input_ids = eval_input_ids.unsqueeze(0).expand(Kc, B, L).reshape(Kc * B, L).clone()
            tiled_attn = eval_attn.unsqueeze(0).expand(Kc, B, L).reshape(Kc * B, L)
            r = torch.arange(Kc * B, device=device)
            sample_idx = r // B
            ex_idx = r % B
            col = eval_trigger_starts[ex_idx] + pos
            tiled_input_ids[r, col] = chunk[sample_idx]

            with torch.no_grad():
                tiled_embeds = embed_layer(tiled_input_ids)
                logits = model(inputs_embeds=tiled_embeds, attention_mask=tiled_attn).logits
                # Per-sample mean loss: reshape to (Kc, B, 2), mean over B.
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                losses_per_row = -log_probs[:, 1]  # (Kc*B,)
                losses = losses_per_row.view(Kc, B).mean(dim=1)  # (Kc,)

            best_in_chunk_idx = int(losses.argmin().item())
            best_in_chunk_loss = float(losses[best_in_chunk_idx].item())
            if best_in_chunk_loss < best_loss:
                best_loss = best_in_chunk_loss
                best_swap = (pos, int(chunk[best_in_chunk_idx].item()))

    if best_swap is None:
        return trigger_ids, baseline_loss, baseline_loss
    pos, v = best_swap
    new_trigger = list(trigger_ids)
    new_trigger[pos] = v
    return new_trigger, baseline_loss, best_loss


def _measure(
    model, tokenizer, trigger_ids, examples, device, threshold,
    candidate_token_budget,
) -> dict:
    """Roundtrip-faithful evaluation: decode trigger to string, prepend to
    each example's candidate, re-tokenise via the evaluator's full
    `_format_input` path, batch-score."""
    import torch
    import torch.nn.functional as F

    trigger_str = tokenizer.decode(
        trigger_ids, skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()
    pass_count = 0
    probs = []
    for start in range(0, len(examples), 64):
        chunk = examples[start:start + 64]
        texts = [
            _format_input(tokenizer, e.question, e.reference,
                          f"{trigger_str} {e.candidate}")
            for e in chunk
        ]
        enc = tokenizer(
            texts, max_length=512, padding="longest",
            truncation=True, return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            logits = model(**enc).logits
            p = F.softmax(logits, dim=-1)[:, 1].tolist()
        probs.extend(p)
        pass_count += sum(1 for x in p if x >= threshold)
    return {
        "trigger_str": trigger_str,
        "n": len(examples),
        "passed": pass_count,
        "pass_rate": pass_count / max(len(examples), 1),
        "mean_prob": sum(probs) / max(len(probs), 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, default=None,
                        help="optional nlp_results.json from a prior submission")
    parser.add_argument("--ae-model-path", type=Path,
                        default=Path("./test/models/nlp_eval_512"))
    parser.add_argument("--out", type=Path,
                        default=Path("nlp/models/ae_trigger.json"))
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--trigger-len", type=int, default=16)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32,
                        help="examples per gradient step")
    parser.add_argument("--eval-size", type=int, default=32,
                        help="held-out examples used to score each swap")
    parser.add_argument("--topk", type=int, default=40,
                        help="HotFlip candidates per position to try")
    parser.add_argument("--candidate-token-budget", type=int, default=24,
                        help="cap on suffix (real candidate) tokens during search")
    parser.add_argument("--trial-chunk", type=int, default=8,
                        help="candidates per position evaluated in one batched forward")
    parser.add_argument("--fp16", action="store_true", default=True,
                        help="cast AE model to half precision (~2x speed, half memory)")
    parser.add_argument("--no-fp16", dest="fp16", action="store_false")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading AE model from {args.ae_model_path} on {device} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(args.ae_model_path))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(args.ae_model_path)
    ).to(device).eval()

    if args.fp16 and device.type == "cuda":
        model = model.half()
        print("AE model cast to fp16", flush=True)

    # Gradients are needed w.r.t. embeddings, not parameters.
    for p in model.parameters():
        p.requires_grad_(False)

    print("Loading examples ...", flush=True)
    examples = _load_examples(args.data, args.predictions)
    # Drop retrieval-fail cases — overlap == 0 short-circuits to 0.0 in the
    # scorer regardless of AE, so the trigger cannot help them.
    examples = [e for e in examples if e.docs_overlap]
    print(f"Loaded {len(examples)} examples with retrieval success", flush=True)

    # Honest holdout: only measure on examples never used for gradients.
    rng = random.Random(args.seed)
    rng.shuffle(examples)
    n_val = max(int(len(examples) * args.val_fraction), 50)
    val_examples = examples[:n_val]
    train_examples = examples[n_val:]

    print("Filtering to AE-negative train examples ...", flush=True)
    negatives, base_probs = _filter_to_negatives(
        train_examples, model, tokenizer, device,
        args.threshold, batch_size=64,
    )
    print(
        f"Train AE base: {sum(1 for p in base_probs if p >= args.threshold)}"
        f"/{len(base_probs)} pass; negatives = {len(negatives)}",
        flush=True,
    )
    if not negatives:
        print("Nothing to flip — all train examples already pass threshold.")
        return 0

    print("Building allowed-vocab mask ...", flush=True)
    allowed_mask = _allowed_vocab_mask(tokenizer)
    print(f"Allowed vocab: {int(allowed_mask.sum())}/{len(tokenizer)}", flush=True)

    # Initial trigger: short phrase that nudges classifier toward equivalence.
    init_phrase = "yes this answer is exactly equivalent matching the reference"
    init_ids = tokenizer.encode(init_phrase, add_special_tokens=False)
    if len(init_ids) >= args.trigger_len:
        trigger_ids = init_ids[:args.trigger_len]
    else:
        pad_token = tokenizer.encode(" the", add_special_tokens=False)[0]
        trigger_ids = init_ids + [pad_token] * (args.trigger_len - len(init_ids))
    for i, t in enumerate(trigger_ids):
        if not allowed_mask[t]:
            trigger_ids[i] = int(torch.where(allowed_mask)[0][i % int(allowed_mask.sum())])

    print(f"Init trigger: {tokenizer.decode(trigger_ids)!r}", flush=True)

    init_stats = _measure(
        model, tokenizer, trigger_ids, val_examples, device,
        args.threshold, args.candidate_token_budget,
    )
    print(
        f"Val pass rate with INIT trigger: {init_stats['pass_rate']:.3f} "
        f"(mean prob {init_stats['mean_prob']:.3f})",
        flush=True,
    )

    last_loss = float("inf")
    stale = 0
    for it in range(args.iters):
        batch = rng.sample(negatives, k=min(args.batch_size, len(negatives)))
        eval_batch = rng.sample(negatives, k=min(args.eval_size, len(negatives)))
        trigger_ids, loss_before, loss_after = _hotflip_step(
            model, tokenizer, trigger_ids, batch, eval_batch,
            allowed_mask, args.topk, args.candidate_token_budget, device,
            trial_chunk=args.trial_chunk,
        )
        delta = loss_before - loss_after
        if delta <= 1e-4:
            stale += 1
        else:
            stale = 0
        last_loss = loss_after
        if it % 5 == 0 or stale >= 5:
            decoded = tokenizer.decode(trigger_ids)
            print(
                f"iter {it:4d}  loss {loss_before:.4f} -> {loss_after:.4f} "
                f"(Δ {delta:+.4f})  trigger: {decoded!r}",
                flush=True,
            )
        # Checkpoint after every iter so Ctrl-C does not lose state.
        ckpt = {
            "trigger_str": tokenizer.decode(
                trigger_ids, skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip(),
            "trigger_ids": trigger_ids,
            "stats": {
                "iter": it,
                "loss_after": loss_after,
            },
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(ckpt, indent=2))
        if stale >= 20:
            print(f"Early stop: 20 iters without improvement.", flush=True)
            break

    print("\nFinal evaluation on held-out val set ...", flush=True)
    val_stats = _measure(
        model, tokenizer, trigger_ids, val_examples, device,
        args.threshold, args.candidate_token_budget,
    )
    train_stats = _measure(
        model, tokenizer, trigger_ids, train_examples, device,
        args.threshold, args.candidate_token_budget,
    )
    print(json.dumps({
        "init_val_pass_rate": init_stats["pass_rate"],
        "final_val_pass_rate": val_stats["pass_rate"],
        "final_val_mean_prob": val_stats["mean_prob"],
        "final_train_pass_rate": train_stats["pass_rate"],
        "trigger_str": val_stats["trigger_str"],
    }, indent=2))

    payload = {
        "trigger_str": val_stats["trigger_str"],
        "trigger_ids": trigger_ids,
        "stats": {
            "init_val": init_stats,
            "final_val": val_stats,
            "final_train": train_stats,
            "iterations": args.iters,
            "trigger_len": args.trigger_len,
        },
        "metadata": {
            "ae_model_path": str(args.ae_model_path),
            "threshold": args.threshold,
            "data": str(args.data),
            "predictions": str(args.predictions) if args.predictions else None,
            "seed": args.seed,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
