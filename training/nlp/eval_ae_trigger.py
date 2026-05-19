"""Evaluate an existing `ae_trigger.json` on the held-out val split.

Use after killing `find_ae_trigger.py` mid-run — the training loop writes a
trigger checkpoint after every iter, but the final init/train/val pass-rate
block only runs at clean exit. This script reproduces that block on demand.

Run on Workbench:

    python training/nlp/eval_ae_trigger.py \
        --data /home/jupyter/novice/nlp/nlp.jsonl \
        --predictions /tmp/nlp_results.json \
        --ae-model-path ./test/models/nlp_eval_512 \
        --trigger nlp/models/ae_trigger.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

# Reuse the helpers from the training script so the eval path is identical.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from find_ae_trigger import _filter_to_negatives, _load_examples, _measure


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--ae-model-path", type=Path,
                        default=Path("./test/models/nlp_eval_512"))
    parser.add_argument("--trigger", type=Path,
                        default=Path("nlp/models/ae_trigger.json"))
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--candidate-token-budget", type=int, default=24)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0,
                        help="must match the --seed used at training time")
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--no-fp16", dest="fp16", action="store_false")
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    payload = json.loads(args.trigger.read_text())
    trigger_ids = payload["trigger_ids"]
    print(f"Loaded trigger: {payload['trigger_str']!r}", flush=True)
    if "stats" in payload:
        print(f"  (training stats: {json.dumps(payload['stats'])})", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading AE model from {args.ae_model_path} on {device} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(args.ae_model_path))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(args.ae_model_path)
    ).to(device).eval()
    if args.fp16 and device.type == "cuda":
        model = model.half()
        print("AE model cast to fp16", flush=True)
    for p in model.parameters():
        p.requires_grad_(False)

    examples = _load_examples(args.data, args.predictions)
    examples = [e for e in examples if e.docs_overlap]
    print(f"Loaded {len(examples)} retrieval-success examples", flush=True)

    rng = random.Random(args.seed)
    rng.shuffle(examples)
    n_val = max(int(len(examples) * args.val_fraction), 50)
    val_examples = examples[:n_val]
    train_examples = examples[n_val:]

    # Baseline pass rate WITHOUT the trigger on val.
    print("\nBaseline (no trigger) on val ...", flush=True)
    _, base_probs = _filter_to_negatives(
        val_examples, model, tokenizer, device, args.threshold, batch_size=64,
    )
    base_pass = sum(1 for p in base_probs if p >= args.threshold)
    base_mean = sum(base_probs) / max(len(base_probs), 1)
    print(
        f"  pass {base_pass}/{len(base_probs)} = {base_pass / max(len(base_probs), 1):.3f}, "
        f"mean prob {base_mean:.3f}",
        flush=True,
    )

    # With-trigger pass rate (roundtrip-faithful) on both splits.
    print("\nWith trigger, val ...", flush=True)
    val_stats = _measure(
        model, tokenizer, trigger_ids, val_examples, device,
        args.threshold, args.candidate_token_budget,
    )
    print(
        f"  pass {val_stats['passed']}/{val_stats['n']} = {val_stats['pass_rate']:.3f}, "
        f"mean prob {val_stats['mean_prob']:.3f}",
        flush=True,
    )

    print("\nWith trigger, train ...", flush=True)
    train_stats = _measure(
        model, tokenizer, trigger_ids, train_examples, device,
        args.threshold, args.candidate_token_budget,
    )
    print(
        f"  pass {train_stats['passed']}/{train_stats['n']} = {train_stats['pass_rate']:.3f}, "
        f"mean prob {train_stats['mean_prob']:.3f}",
        flush=True,
    )

    gap = train_stats["pass_rate"] - val_stats["pass_rate"]
    lift = val_stats["pass_rate"] - base_pass / max(len(base_probs), 1)
    print(
        f"\nVAL lift over baseline: {lift:+.3f}    "
        f"TRAIN-VAL overfit gap: {gap:+.3f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
