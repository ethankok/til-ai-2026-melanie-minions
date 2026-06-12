#!/usr/bin/env python3
"""Probe what *kind* of distribution shift drives the local->cloud CV gap.

Background: tier1 (cv-yolo-v2-tier1-best) scores hard held-out mAP 0.8947 but
cloud mAP 0.556 — a 0.349 gap. Pass A on 16/05 ruled out class confusion and
low-conf FP cleanup. We have NOT empirically tested whether the gap is from:

  (a) JPEG quality shift -- hidden eval images may be lower-quality JPEGs than
      our training set (which is photo-composited at near-lossless quality).
  (b) Resolution / scale shift -- hidden images may be at a different native
      resolution, so the model's effective object pixel sizes differ.

This script applies in-memory image transforms BEFORE posting to the running
CV container and re-scores against the hard held-out split. If a transform
drops mAP by >= 0.05, that axis is a real component of the gap, and the
matching training-time augmentation has high EV. If a transform barely moves
mAP, that hypothesis is dead before we spend a single training hour on it.

Usage on Workbench:

    # 1. Run the shipped tier1 container with its production env
    docker run -d --rm --name cv-tier1 -p 5002:5002 \
      -e CV_MODEL_PATH=/workspace/models/cv/best.pt \
      -e CV_CONF=0.20 -e CV_IOU=0.60 -e CV_IMGSZ=896 \
      -e CV_AUGMENT=1 -e CV_HALF=1 \
      --gpus all melanie-minions-cv:cv-yolo-v2-tier1-best

    # 2. Wait for ready (~30s), then run the diagnostic
    python training/cv/gap_diagnose.py \
      --data-dir /home/jupyter/novice/cv \
      --annotations /home/jupyter/cv_yolo_dataset/coco/annotations_test.json \
      --out /home/jupyter/cv_eval_sweeps/gap-diagnose.json

    # 3. Cleanup
    docker rm -f cv-tier1

Expected runtime: ~6 minutes per transform x 6 transforms = ~35 minutes total.
Use --transforms to subset for a faster smoke run.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import sys
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Callable

import requests
from PIL import Image
from tqdm import tqdm

# Reuse the existing scoring path so numbers are directly comparable to
# eval_cv_http.py output.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_cv_http import (  # noqa: E402
    _clean_detection,
    _subset_annotations,
    score_predictions,
)


# -------- Transforms --------
# Each transform takes raw JPEG bytes and returns transformed JPEG bytes.
# `baseline` is None (no transform) and is the sanity check that the pipeline
# matches the prior 0.8947 number end-to-end.

ByteTransform = Callable[[bytes], bytes] | None


def _jpeg_recompress(quality: int) -> ByteTransform:
    """Re-encode the image at a lower JPEG quality without changing pixel dims."""

    def transform(raw: bytes) -> bytes:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return buf.getvalue()

    return transform


def _downsample_upsample(factor: int, quality: int = 92) -> ByteTransform:
    """Downsample by `factor` then upsample back. Simulates lower-res capture."""

    def transform(raw: bytes) -> bytes:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        w, h = img.size
        small = img.resize((max(1, w // factor), max(1, h // factor)), Image.BILINEAR)
        restored = small.resize((w, h), Image.BILINEAR)
        buf = io.BytesIO()
        restored.save(buf, format="JPEG", quality=quality)
        return buf.getvalue()

    return transform


def _jpeg_plus_downsample(quality: int, factor: int) -> ByteTransform:
    """Combined low-quality + low-effective-resolution shift."""

    def transform(raw: bytes) -> bytes:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        w, h = img.size
        small = img.resize((max(1, w // factor), max(1, h // factor)), Image.BILINEAR)
        restored = small.resize((w, h), Image.BILINEAR)
        buf = io.BytesIO()
        restored.save(buf, format="JPEG", quality=quality)
        return buf.getvalue()

    return transform


TRANSFORMS: dict[str, ByteTransform] = {
    "baseline": None,
    "jpeg-q70": _jpeg_recompress(70),
    "jpeg-q50": _jpeg_recompress(50),
    "jpeg-q30": _jpeg_recompress(30),
    "downsample-2x": _downsample_upsample(2),
    "downsample-3x": _downsample_upsample(3),
    "jpeg50-down2": _jpeg_plus_downsample(50, 2),
}


# -------- Eval loop --------


def _batched(items: Sequence[Mapping[str, Any]], batch_size: int) -> Iterator[list[Mapping[str, Any]]]:
    for idx in range(0, len(items), batch_size):
        yield list(items[idx : idx + batch_size])


def _collect_predictions_with_transform(
    annotations: Mapping[str, Any],
    images_dir: Path,
    endpoint: str,
    transform: ByteTransform,
    batch_size: int,
    timeout: float,
    desc: str,
) -> list[dict[str, Any]]:
    images = list(annotations["images"])
    results: list[dict[str, Any]] = []
    total = math.ceil(len(images) / batch_size)
    for batch in tqdm(list(_batched(images, batch_size)), total=total, desc=desc):
        instances: list[dict[str, Any]] = []
        for image in batch:
            with open(images_dir / str(image["file_name"]), "rb") as f:
                raw = f.read()
            payload = transform(raw) if transform is not None else raw
            instances.append(
                {"key": image["id"], "b64": base64.b64encode(payload).decode("ascii")}
            )
        response = requests.post(endpoint, json={"instances": instances}, timeout=timeout)
        response.raise_for_status()
        predictions = response.json()["predictions"]
        if len(predictions) != len(instances):
            raise RuntimeError(
                f"Service returned {len(predictions)} predictions for {len(instances)} instances"
            )
        for instance, detections in zip(instances, predictions):
            for det in detections:
                cleaned = _clean_detection(det, instance["key"])
                if cleaned is not None:
                    results.append(cleaned)
    return results


# -------- Reporting --------


def _print_table(rows: list[dict[str, Any]], baseline_map: float | None) -> None:
    header = f"{'transform':<18} {'mAP':>8} {'small':>8} {'medium':>8} {'large':>8} {'Δ map':>8}"
    print()
    print(header)
    print("-" * len(header))
    for row in rows:
        delta_str = "—"
        if baseline_map is not None and row["transform"] != "baseline":
            delta = row["map"] - baseline_map
            sign = "+" if delta >= 0 else "−"
            delta_str = f"{sign}{abs(delta):.4f}"
        per_area = row["per_area"]
        small = per_area.get("small")
        medium = per_area.get("medium")
        large = per_area.get("large")
        print(
            f"{row['transform']:<18} "
            f"{row['map']:>8.4f} "
            f"{small if small is not None else 0.0:>8.4f} "
            f"{medium if medium is not None else 0.0:>8.4f} "
            f"{large if large is not None else 0.0:>8.4f} "
            f"{delta_str:>8}"
        )


def _interpret(rows: list[dict[str, Any]], baseline_map: float | None) -> None:
    """Quick verdict per axis, to guide the Phase C scope."""
    if baseline_map is None:
        print("\nNo baseline row — skipping interpretation.")
        return

    by_name = {row["transform"]: row for row in rows}
    print()
    print("Verdict:")

    jpeg_drops = []
    for name in ("jpeg-q70", "jpeg-q50", "jpeg-q30"):
        if name in by_name:
            jpeg_drops.append((name, baseline_map - by_name[name]["map"]))
    if jpeg_drops:
        worst_jpeg = max(jpeg_drops, key=lambda x: x[1])
        verdict = (
            "JPEG augmentation worth training" if worst_jpeg[1] >= 0.05
            else "JPEG augmentation likely dead — gap is not quality"
        )
        print(f"  JPEG (worst: {worst_jpeg[0]}, drop {worst_jpeg[1]:+.4f}) -> {verdict}")

    res_drops = []
    for name in ("downsample-2x", "downsample-3x"):
        if name in by_name:
            res_drops.append((name, baseline_map - by_name[name]["map"]))
    if res_drops:
        worst_res = max(res_drops, key=lambda x: x[1])
        verdict = (
            "Native-res tile crops worth building" if worst_res[1] >= 0.05
            else "Resolution shift unlikely to be the gap"
        )
        print(f"  Resolution (worst: {worst_res[0]}, drop {worst_res[1]:+.4f}) -> {verdict}")

    if "jpeg50-down2" in by_name:
        combined_drop = baseline_map - by_name["jpeg50-down2"]["map"]
        verdict = (
            "Stacked aug very likely worth doing" if combined_drop >= 0.10
            else "Combined effect is modest; single-axis aug is the right scope"
        )
        print(f"  Combined (drop {combined_drop:+.4f}) -> {verdict}")


# -------- Main --------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("/home/jupyter/novice/cv"))
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("/home/jupyter/cv_yolo_dataset/coco/annotations_test.json"),
    )
    parser.add_argument("--endpoint", default="http://localhost:5002/cv")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--transforms",
        default=",".join(TRANSFORMS.keys()),
        help="Comma-separated transform names from: " + ", ".join(TRANSFORMS.keys()),
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit images for a smoke test")
    parser.add_argument("--out", type=Path, help="Optional JSON output path with full per-transform scores")
    args = parser.parse_args()

    annotations = json.loads(args.annotations.read_text())
    if args.limit:
        image_ids = {img["id"] for img in annotations["images"][: args.limit]}
        annotations = _subset_annotations(annotations, image_ids)

    requested = [name.strip() for name in args.transforms.split(",") if name.strip()]
    unknown = [name for name in requested if name not in TRANSFORMS]
    if unknown:
        raise SystemExit(f"Unknown transform(s): {unknown}. Choose from {list(TRANSFORMS.keys())}")

    images_dir = args.data_dir / "images"
    rows: list[dict[str, Any]] = []
    baseline_map: float | None = None

    for name in requested:
        transform = TRANSFORMS[name]
        t0 = time.time()
        predictions = _collect_predictions_with_transform(
            annotations=annotations,
            images_dir=images_dir,
            endpoint=args.endpoint,
            transform=transform,
            batch_size=args.batch_size,
            timeout=args.timeout,
            desc=f"{name:<14}",
        )
        elapsed = time.time() - t0
        summary = score_predictions(predictions, annotations)
        row = {
            "transform": name,
            "map": summary["map"],
            "map50": summary["map50"],
            "map75": summary["map75"],
            "per_area": summary["per_area"],
            "elapsed_s": round(elapsed, 1),
        }
        rows.append(row)
        if name == "baseline":
            baseline_map = summary["map"]
        print(
            f"  {name:<14}  mAP={summary['map']:.4f}  "
            f"small={summary['per_area'].get('small') or 0:.4f}  "
            f"({elapsed:.1f}s)"
        )

    _print_table(rows, baseline_map)
    _interpret(rows, baseline_map)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        payload = {"baseline_map": baseline_map, "rows": rows}
        args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nFull results: {args.out}")


if __name__ == "__main__":
    main()
