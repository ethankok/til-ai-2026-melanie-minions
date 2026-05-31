"""Check whether the AE adversarial trigger collapses under the patched eval.

Upstream commit 206a5e1 ("bump ae, bugfix nlp eval") added
``add_special_tokens=False`` to the evaluator's tokenizer call in
``test/test_nlp.py:batch_evaluate``. Our v20 trigger was HotFlip-optimised
against the OLD tokenisation, which wrapped the input in ``[CLS] ... [SEP]``
(the tokenizer default ``add_special_tokens=True``; see the hardcoded
``[cls_id] + ... + [sep_id]`` in ``find_ae_trigger._build_ids_with_trigger``).

This script measures the trigger's ``prob_eq`` on the same negative examples
under BOTH tokenisations and prints a side-by-side comparison:

  * OLD  (add_special_tokens=True)  — what the trigger was trained against
  * NEW  (add_special_tokens=False) — what the cloud scorer now does

If the trigger has been patched, the NEW column shows mean ``prob_eq`` well
below the 0.9 threshold and a pass-rate near zero, while OLD stays saturated.

Runs on the Workbench (needs torch + the bundled AE checkpoint + the trigger):

    python training/nlp/check_trigger_collapse.py \
        --data /home/jupyter/novice/nlp/nlp.jsonl \
        --predictions /tmp/nlp_results.json \
        --ae-model-path ./test/models/nlp_eval_512 \
        --trigger nlp/models/ae_trigger.json

``--predictions`` is optional; without it the script falls back to the
"drop the last gold word" near-miss candidates that ``find_ae_trigger`` uses.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Reuse the exact helpers from the trainer so the format/tokenise path is
# identical to both training and the deployed evaluator.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from find_ae_trigger import (  # noqa: E402
    _filter_to_negatives,
    _format_input,
    _load_examples,
)


def _measure(model, tokenizer, trigger_str, examples, device, threshold,
             add_special_tokens):
    """Prepend the trigger to each candidate, format via the evaluator's
    ``_format_input`` path, tokenise with the given ``add_special_tokens``
    setting, and report pass-rate + mean prob_eq."""
    import torch
    import torch.nn.functional as F

    pass_count = 0
    probs: list[float] = []
    for start in range(0, len(examples), 64):
        chunk = examples[start:start + 64]
        texts = [
            _format_input(tokenizer, e.question, e.reference,
                          f"{trigger_str} {e.candidate}")
            for e in chunk
        ]
        enc = tokenizer(
            texts, max_length=512, padding="longest", truncation=True,
            return_tensors="pt", add_special_tokens=add_special_tokens,
        ).to(device)
        with torch.no_grad():
            p = F.softmax(model(**enc).logits, dim=-1)[:, 1].tolist()
        probs.extend(p)
        pass_count += sum(1 for x in p if x >= threshold)
    n = max(len(examples), 1)
    return {
        "n": len(examples),
        "passed": pass_count,
        "pass_rate": pass_count / n,
        "mean_prob": sum(probs) / max(len(probs), 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--ae-model-path", type=Path,
                        default=Path("./test/models/nlp_eval_512"))
    parser.add_argument("--trigger", type=Path,
                        default=Path("nlp/models/ae_trigger.json"))
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    payload = json.loads(args.trigger.read_text())
    trigger_str = payload["trigger_str"]
    print(f"Loaded trigger: {trigger_str!r}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading AE model from {args.ae_model_path} on {device} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(str(args.ae_model_path))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(args.ae_model_path)
    ).to(device).eval()

    examples = _load_examples(args.data, args.predictions)
    # Restrict to the cases the trigger actually has to rescue (currently
    # failing, with a retrieval hit) — exactly the population the cheese targets.
    negatives, _ = _filter_to_negatives(
        examples, model, tokenizer, device, args.threshold, args.batch_size
    )
    negatives = [e for e in negatives if e.docs_overlap]
    if not negatives:
        print("No negative (retrieval-hit, failing) examples found — nothing to "
              "measure. Check --data/--predictions.", flush=True)
        return 1
    print(f"Measuring on {len(negatives)} negative examples "
          f"(threshold={args.threshold})\n", flush=True)

    old = _measure(model, tokenizer, trigger_str, negatives, device,
                   args.threshold, add_special_tokens=True)
    new = _measure(model, tokenizer, trigger_str, negatives, device,
                   args.threshold, add_special_tokens=False)

    print(f"{'tokenisation':<34}{'pass_rate':>12}{'mean_prob_eq':>14}")
    print(f"{'OLD  (add_special_tokens=True)':<34}"
          f"{old['pass_rate']:>12.4f}{old['mean_prob']:>14.4f}")
    print(f"{'NEW  (add_special_tokens=False)':<34}"
          f"{new['pass_rate']:>12.4f}{new['mean_prob']:>14.4f}")

    collapsed = new["mean_prob"] < args.threshold and new["pass_rate"] < 0.5 * old["pass_rate"]
    verdict = (
        "TRIGGER COLLAPSED — the patched eval breaks the cheese; retrain needed."
        if collapsed else
        "Trigger SURVIVES the patch — mean_prob still clears threshold. Investigate."
    )
    print(f"\n{verdict}", flush=True)
    return 0 if collapsed else 2


if __name__ == "__main__":
    raise SystemExit(main())
