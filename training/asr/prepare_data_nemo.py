"""Build NeMo manifests for Parakeet fine-tuning on novice ASR.

NeMo's `setup_training_data` / `setup_validation_data` expect manifests in
NeMo's standard JSONL format:

    {"audio_filepath": "/abs/path/to/sample_0.wav", "duration": 12.34, "text": "..."}

This is different from the HF `DatasetDict` shape produced by `prepare_data.py`
for the distil-whisper pipeline. Both pipelines coexist; pick the one that
matches the trainer.

Usage::

    python training/asr/prepare_data_nemo.py \\
        --data-dir /home/jupyter/novice/asr \\
        --slang-file asr/models/slang_prompt.txt \\
        --out-dir training/asr/data_nemo \\
        --slang-multiplier 2

Outputs:
    training/asr/data_nemo/train_manifest.jsonl
    training/asr/data_nemo/val_manifest.jsonl
    training/asr/data_nemo/stats.json     (small summary, useful for debugging)

Strategy: same Option B as before — train on ~90% of `asr.jsonl`, hold 10%
out only for early stopping / WER tracking. The full 4110 manifest is also
what `test_asr.py` evaluates against locally, so the "validation" WER is
strictly leaky vs local but still useful as an in-training signal.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import soundfile as sf


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _load_slang(slang_file: Path | None) -> set[str]:
    if slang_file is None or not slang_file.exists():
        return set()
    return {t.lower() for t in slang_file.read_text(encoding="utf-8").split() if t}


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


def _read_duration(path: Path) -> float:
    info = sf.info(str(path))
    return float(info.frames) / float(info.samplerate)


def _write_manifest(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


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
        help="Source manifest name relative to --data-dir.",
    )
    ap.add_argument(
        "--slang-file",
        type=Path,
        default=Path("asr/models/slang_prompt.txt"),
        help="slang_prompt.txt from extract_slang.py (optional, used for "
             "stratified oversampling).",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("training/asr/data_nemo"),
    )
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument(
        "--slang-multiplier",
        type=int,
        default=2,
        help="Duplicate slang-containing clips this many times in train.",
    )
    ap.add_argument(
        "--max-duration",
        type=float,
        default=40.0,
        help="Drop training clips longer than this (Parakeet-TDT-0.6B-v2's "
             "training cutoff is 40s).",
    )
    ap.add_argument(
        "--min-duration",
        type=float,
        default=0.5,
        help="Drop training clips shorter than this — they're almost always "
             "noise / breath bursts and were the source of the "
             "'Thank you.' / 'I' hallucinations on the original Whisper path.",
    )
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    manifest_path = args.data_dir / args.manifest
    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found: {manifest_path}")

    rows = _load_jsonl(manifest_path)
    rows = [
        r for r in rows
        if r.get("transcript") and (r.get("audio") or r.get("path"))
    ]
    for r in rows:
        if "audio" not in r and "path" in r:
            r["audio"] = r["path"]
        r["audio_path"] = str(args.data_dir / r["audio"])

    print(f"Loaded {len(rows)} rows from {manifest_path}")

    # Resolve duration once per clip. soundfile.info is cheap (header read,
    # no decode), so doing this for 4110 clips takes a couple of seconds.
    print("Reading audio durations ...", flush=True)
    kept: list[dict] = []
    skipped_short = 0
    skipped_long = 0
    skipped_missing = 0
    for r in rows:
        ap_path = Path(r["audio_path"])
        if not ap_path.exists():
            skipped_missing += 1
            continue
        try:
            dur = _read_duration(ap_path)
        except Exception as exc:
            print(f"WARN: failed to read {ap_path}: {exc}", flush=True)
            skipped_missing += 1
            continue
        if dur < args.min_duration:
            skipped_short += 1
            continue
        if dur > args.max_duration:
            skipped_long += 1
            continue
        r["duration"] = dur
        kept.append(r)

    print(
        f"Kept {len(kept)} clips "
        f"(skipped: missing={skipped_missing}, short={skipped_short}, long={skipped_long})"
    )

    rng = random.Random(args.seed)

    # Stratified split by transcript-length quartile to balance long/short clips.
    lengths = sorted(len(r["transcript"]) for r in kept)
    n = len(lengths)
    quartiles = [lengths[n // 4], lengths[n // 2], lengths[3 * n // 4]]
    buckets: dict[int, list[dict]] = {0: [], 1: [], 2: [], 3: []}
    for r in kept:
        buckets[_length_quartile(r["transcript"], quartiles)].append(r)

    train_rows: list[dict] = []
    val_rows: list[dict] = []
    for q, items in buckets.items():
        rng.shuffle(items)
        n_val = max(1, int(len(items) * args.val_frac))
        val_rows.extend(items[:n_val])
        train_rows.extend(items[n_val:])

    slang = _load_slang(args.slang_file)
    n_slang_pre = sum(_contains_slang(r["transcript"], slang) for r in train_rows)
    if slang and args.slang_multiplier > 1:
        boosted: list[dict] = []
        for r in train_rows:
            boosted.append(r)
            if _contains_slang(r["transcript"], slang):
                for _ in range(args.slang_multiplier - 1):
                    boosted.append(r)
        rng.shuffle(boosted)
        train_rows = boosted
    n_slang_post = sum(_contains_slang(r["transcript"], slang) for r in train_rows)

    def _to_nemo(rs: list[dict]) -> list[dict]:
        out: list[dict] = []
        for r in rs:
            out.append(
                {
                    "audio_filepath": r["audio_path"],
                    "duration": r["duration"],
                    "text": r["transcript"],
                }
            )
        return out

    train_manifest = args.out_dir / "train_manifest.jsonl"
    val_manifest = args.out_dir / "val_manifest.jsonl"
    _write_manifest(_to_nemo(train_rows), train_manifest)
    _write_manifest(_to_nemo(val_rows), val_manifest)

    stats = {
        "source_manifest": str(manifest_path),
        "kept_total": len(kept),
        "skipped_missing": skipped_missing,
        "skipped_short": skipped_short,
        "skipped_long": skipped_long,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "slang_terms_loaded": len(slang),
        "slang_rows_in_train_pre_oversample": n_slang_pre,
        "slang_rows_in_train_post_oversample": n_slang_post,
        "slang_multiplier": args.slang_multiplier,
        "val_frac": args.val_frac,
        "min_duration": args.min_duration,
        "max_duration": args.max_duration,
        "seed": args.seed,
    }
    (args.out_dir / "stats.json").write_text(json.dumps(stats, indent=2))

    print(
        f"\nWrote NeMo manifests:\n"
        f"  {train_manifest}  (rows={len(train_rows)}, "
        f"slang-rows post-boost={n_slang_post})\n"
        f"  {val_manifest}    (rows={len(val_rows)})\n"
        f"  {args.out_dir / 'stats.json'}"
    )


if __name__ == "__main__":
    main()
