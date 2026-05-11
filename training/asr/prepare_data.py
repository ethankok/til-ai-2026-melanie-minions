"""Build a HuggingFace dataset for fine-tuning distil-whisper on novice ASR.

Reads ``asr.jsonl`` (audio relpath + transcript per line), 90/10 splits, and
optionally over-samples clips whose transcript contains in-world slang.

Usage on the GCP Workbench instance::

    python training/asr/prepare_data.py \
        --data-dir /home/jupyter/novice/asr \
        --slang-file asr/models/slang_prompt.txt \
        --out-dir training/asr/data \
        --slang-multiplier 2

The output is a ``DatasetDict`` saved to disk (``train`` and ``validation``
splits). Audio is left as file paths; resampling / feature extraction happens
in the training script so we don't materialize a giant cached tensor here.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from datasets import Audio, Dataset, DatasetDict


def _load_jsonl(path: Path) -> list[dict]:
    out: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _load_slang(slang_file: Path | None) -> set[str]:
    if slang_file is None or not slang_file.exists():
        return set()
    return {t for t in slang_file.read_text(encoding="utf-8").split() if t}


def _contains_slang(transcript: str, slang: set[str]) -> bool:
    if not slang:
        return False
    lc = transcript.lower()
    return any(s in lc for s in slang)


def _length_quartile(text: str, quartiles: list[int]) -> int:
    n = len(text)
    for i, q in enumerate(quartiles):
        if n <= q:
            return i
    return len(quartiles)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/home/jupyter/novice/asr"),
        help="Directory containing asr.jsonl and the audio files.",
    )
    ap.add_argument(
        "--manifest",
        type=str,
        default="asr.jsonl",
        help="Manifest file name relative to --data-dir.",
    )
    ap.add_argument(
        "--slang-file",
        type=Path,
        default=Path("asr/models/slang_prompt.txt"),
        help="slang_prompt.txt from extract_slang.py (optional).",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("training/asr/data"),
    )
    ap.add_argument(
        "--val-frac",
        type=float,
        default=0.10,
    )
    ap.add_argument(
        "--slang-multiplier",
        type=int,
        default=2,
        help="Duplicate slang-containing clips this many times in train.",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    manifest_path = args.data_dir / args.manifest
    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found: {manifest_path}")

    rows = _load_jsonl(manifest_path)
    # Filter to rows that have a transcript and an audio relpath.
    rows = [
        r
        for r in rows
        if r.get("transcript") and (r.get("audio") or r.get("path"))
    ]
    for r in rows:
        if "audio" not in r and "path" in r:
            r["audio"] = r["path"]
        r["audio_path"] = str(args.data_dir / r["audio"])

    rng = random.Random(args.seed)

    # Stratified split by transcript-length quartile to balance long/short clips.
    lengths = sorted(len(r["transcript"]) for r in rows)
    n = len(lengths)
    quartiles = [lengths[n // 4], lengths[n // 2], lengths[3 * n // 4]]
    buckets: dict[int, list[dict]] = {0: [], 1: [], 2: [], 3: []}
    for r in rows:
        buckets[_length_quartile(r["transcript"], quartiles)].append(r)

    train_rows: list[dict] = []
    val_rows: list[dict] = []
    for q, items in buckets.items():
        rng.shuffle(items)
        n_val = max(1, int(len(items) * args.val_frac))
        val_rows.extend(items[:n_val])
        train_rows.extend(items[n_val:])

    slang = _load_slang(args.slang_file)
    if slang and args.slang_multiplier > 1:
        boosted: list[dict] = []
        for r in train_rows:
            boosted.append(r)
            if _contains_slang(r["transcript"], slang):
                for _ in range(args.slang_multiplier - 1):
                    boosted.append(r)
        rng.shuffle(boosted)
        train_rows = boosted

    def _to_dict(rs: list[dict]) -> dict:
        return {
            "audio": [r["audio_path"] for r in rs],
            "text": [r["transcript"] for r in rs],
        }

    ds = DatasetDict(
        {
            "train": Dataset.from_dict(_to_dict(train_rows)).cast_column(
                "audio", Audio(sampling_rate=16000)
            ),
            "validation": Dataset.from_dict(_to_dict(val_rows)).cast_column(
                "audio", Audio(sampling_rate=16000)
            ),
        }
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(args.out_dir))

    print(
        f"Saved DatasetDict to {args.out_dir} "
        f"(train={len(ds['train'])}, validation={len(ds['validation'])})"
    )
    if slang:
        n_slang = sum(_contains_slang(r["transcript"], slang) for r in train_rows)
        print(f"Slang-containing rows in train (post-oversample): {n_slang}")


if __name__ == "__main__":
    main()
