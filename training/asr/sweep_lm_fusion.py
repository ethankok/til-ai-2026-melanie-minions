"""Sweep NGPU-LM fusion hyper-parameters on the held-out ASR val slice.

Workbench-only (needs NeMo + the acoustic .nemo + the n-gram LM). Loads the
acoustic model once, then for each (alpha, beam) reconfigures the decoder via the
same ``_build_lm_decoding_cfg`` helper the server uses, transcribes the held-out
slice, and prints WER + wall-clock. Use it to pick a config before ``til test``.

The ASR manifest has no clean train/test split (see ERROR_ANALYSIS.md "leaky
val"), so this is a proxy for ranking configs, not an absolute WER. The real
validator is ``til test`` / cloud. We take a deterministic last-10% slice so the
ranking is reproducible.

Usage::

    python training/asr/sweep_lm_fusion.py \
        --asr-jsonl /home/jupyter/novice/asr/asr.jsonl \
        --audio-dir /home/jupyter/novice/asr \
        --nemo-model asr/models/parakeet-tdt-0.6b-v2.nemo \
        --ngram-lm asr/models/ngram_lm.nemo \
        --alphas 0.1 0.2 0.3 0.4 0.5 --beams 2 4
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Reuse the exact decoding-config builder the server uses.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "asr" / "src"))
from asr_manager import _build_lm_decoding_cfg  # noqa: E402


def _load_val_slice(asr_jsonl: Path, frac: float) -> list[dict]:
    rows = []
    with open(asr_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    n_val = max(1, int(len(rows) * frac))
    return rows[-n_val:]  # deterministic last-N slice


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--asr-jsonl", type=Path, default=Path("/home/jupyter/novice/asr/asr.jsonl"))
    ap.add_argument("--audio-dir", type=Path, default=Path("/home/jupyter/novice/asr"))
    ap.add_argument("--nemo-model", type=Path, required=True)
    ap.add_argument("--ngram-lm", type=Path, required=True)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4, 0.5])
    ap.add_argument("--beams", type=int, nargs="+", default=[2, 4])
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--strategy", default="malsd_batch")
    ap.add_argument("--pruning", default="late")
    ap.add_argument("--blank-mode", default="lm_weighted_full")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    import jiwer
    from omegaconf import OmegaConf
    from nemo.collections.asr.models import ASRModel

    val = _load_val_slice(args.asr_jsonl, args.val_frac)
    audio_paths = [str(args.audio_dir / r["audio"]) for r in val]
    refs = [r["transcript"] for r in val]
    print(f"Held-out val slice: {len(val)} clips", flush=True)

    model = ASRModel.restore_from(restore_path=str(args.nemo_model))
    try:
        model = model.cuda().half()
    except Exception as exc:
        print(f"[sweep] GPU/half unavailable ({exc}); running on default device", flush=True)
    model.eval()

    base = OmegaConf.to_container(model.cfg.decoding, resolve=True)
    if not isinstance(base, dict):
        base = {}

    # Greedy baseline for reference (no LM).
    def _run() -> tuple[float, float]:
        t0 = time.time()
        out = model.transcribe(audio_paths, batch_size=args.batch_size, verbose=False)
        if isinstance(out, tuple):
            out = out[0]
        hyps = [(h.text if hasattr(h, "text") else h) or "" for h in out]
        elapsed = time.time() - t0
        wer = jiwer.wer(refs, hyps)
        return wer, elapsed

    print("\nconfig                         WER       sec", flush=True)
    print("-" * 50, flush=True)
    wer, sec = _run()
    print(f"greedy (baseline)              {wer:.4f}   {sec:6.1f}", flush=True)

    for beam in args.beams:
        for alpha in args.alphas:
            cfg = _build_lm_decoding_cfg(
                base,
                lm_path=str(args.ngram_lm),
                alpha=alpha,
                beam_size=beam,
                strategy=args.strategy,
                pruning_mode=args.pruning,
                blank_lm_score_mode=args.blank_mode,
            )
            try:
                model.change_decoding_strategy(OmegaConf.create(cfg))
            except Exception as exc:
                print(f"beam={beam} alpha={alpha}: change_decoding_strategy failed: {exc}", flush=True)
                continue
            wer, sec = _run()
            print(f"beam={beam} alpha={alpha:<4}              {wer:.4f}   {sec:6.1f}", flush=True)


if __name__ == "__main__":
    main()
