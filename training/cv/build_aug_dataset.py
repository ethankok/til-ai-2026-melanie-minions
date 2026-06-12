#!/usr/bin/env python3
"""Build the Phase C.1 augmented training dataset for the v8s@1024 retrain.

The 16/05 diagnostic (`training/cv/gap_diagnose.py`) showed that BOTH JPEG
quality and effective resolution shifts crater the model's small-AP on the
hard held-out (-0.13 to -0.19 small AP), even though their hit on total mAP
looks modest. Since the cloud failure mode is small-AP per Pass A, train-time
augmentation along these axes is the right move.

This script expands the train split offline (val/test untouched) with three
families of samples per original train image:

  (1) Original image at native quality (kept as-is).
  (2) JPEG-recompressed copy at random quality in [40, 85] — exposes the model
      to compression artifacts it would otherwise never see.
  (3) One random 1024x1024 crop from the native 1920x1080 — small objects
      retain their native pixel size during training instead of being
      down-sampled to ~14 px in the model's input. Boxes are clipped to the
      crop and dropped if visible area < 0.5 of the original box.

Resulting train set is ~3x the original size. Val and test splits are copied
or symlinked through untouched so the existing eval pipeline stays valid.

Run on Workbench after `prepare_yolo_dataset.py` has produced the canonical
split at `/home/jupyter/cv_yolo_dataset/`:

    python training/cv/build_aug_dataset.py \\
      --in-dir /home/jupyter/cv_yolo_dataset \\
      --out-dir /home/jupyter/cv_yolo_dataset_augc1 \\
      --crops-per-image 1 \\
      --jpeg-per-image 1

Idempotent: removes the output dir before writing. Deterministic with --seed.
"""

from __future__ import annotations

import argparse
import io
import json
import random
import shutil
import sys
from pathlib import Path

from PIL import Image
from tqdm import tqdm


def _read_yolo_labels(path: Path) -> list[tuple[int, float, float, float, float]]:
    """Read YOLO `cls xc yc w h` (normalized) labels. Returns empty list if missing."""
    if not path.exists():
        return []
    boxes: list[tuple[int, float, float, float, float]] = []
    for line in path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        cls = int(parts[0])
        xc, yc, bw, bh = (float(p) for p in parts[1:])
        boxes.append((cls, xc, yc, bw, bh))
    return boxes


def _write_yolo_labels(path: Path, boxes: list[tuple[int, float, float, float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"
        for cls, xc, yc, bw, bh in boxes
    ]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _random_jpeg_bytes(image: Image.Image, quality: int) -> bytes:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _random_crop(
    image: Image.Image,
    boxes: list[tuple[int, float, float, float, float]],
    crop_size: int,
    min_visibility: float,
    rng: random.Random,
    max_attempts: int = 10,
) -> tuple[Image.Image, list[tuple[int, float, float, float, float]]] | None:
    """Return a random native-res crop that contains at least one visible box.

    Returns None if the image is smaller than the crop or no good crop is
    found within max_attempts.
    """
    w, h = image.size
    if w < crop_size or h < crop_size:
        return None
    if not boxes:
        return None

    abs_boxes: list[tuple[int, float, float, float, float, float]] = []
    for cls, xc, yc, bw, bh in boxes:
        abs_xc = xc * w
        abs_yc = yc * h
        abs_bw = bw * w
        abs_bh = bh * h
        if abs_bw <= 0 or abs_bh <= 0:
            continue
        abs_boxes.append(
            (
                cls,
                abs_xc - abs_bw / 2.0,
                abs_yc - abs_bh / 2.0,
                abs_xc + abs_bw / 2.0,
                abs_yc + abs_bh / 2.0,
                abs_bw * abs_bh,
            )
        )

    if not abs_boxes:
        return None

    for _ in range(max_attempts):
        # Bias the crop toward this box's center.
        anchor = rng.choice(abs_boxes)
        cx = (anchor[1] + anchor[3]) / 2.0
        cy = (anchor[2] + anchor[4]) / 2.0
        x0 = int(max(0, min(w - crop_size, cx - crop_size / 2 + rng.uniform(-crop_size / 4, crop_size / 4))))
        y0 = int(max(0, min(h - crop_size, cy - crop_size / 2 + rng.uniform(-crop_size / 4, crop_size / 4))))
        x1 = x0 + crop_size
        y1 = y0 + crop_size

        new_boxes: list[tuple[int, float, float, float, float]] = []
        for cls, bx1, by1, bx2, by2, orig_area in abs_boxes:
            ix1 = max(bx1, x0)
            iy1 = max(by1, y0)
            ix2 = min(bx2, x1)
            iy2 = min(by2, y1)
            iw = ix2 - ix1
            ih = iy2 - iy1
            if iw <= 0 or ih <= 0:
                continue
            visible_area = iw * ih
            if orig_area <= 0 or visible_area / orig_area < min_visibility:
                continue
            nxc = ((ix1 + ix2) / 2.0 - x0) / crop_size
            nyc = ((iy1 + iy2) / 2.0 - y0) / crop_size
            nbw = iw / crop_size
            nbh = ih / crop_size
            if nbw <= 0 or nbh <= 0:
                continue
            new_boxes.append((cls, nxc, nyc, nbw, nbh))

        if new_boxes:
            cropped = image.crop((x0, y0, x1, y1))
            return cropped, new_boxes
    return None


def _copy_or_link(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        try:
            dst.symlink_to(src)
        except OSError:
            shutil.copy2(src, dst)


def build(
    in_dir: Path,
    out_dir: Path,
    crops_per_image: int,
    jpeg_per_image: int,
    crop_size: int,
    min_visibility: float,
    jpeg_quality_low: int,
    jpeg_quality_high: int,
    seed: int,
    copy_eval_images: bool,
) -> None:
    rng = random.Random(seed)

    train_img_dir = in_dir / "images" / "train"
    train_lbl_dir = in_dir / "labels" / "train"
    if not train_img_dir.exists():
        raise SystemExit(f"Missing train image dir: {train_img_dir}")
    if not train_lbl_dir.exists():
        raise SystemExit(f"Missing train label dir: {train_lbl_dir}")

    if out_dir.exists():
        shutil.rmtree(out_dir)

    out_train_img = out_dir / "images" / "train"
    out_train_lbl = out_dir / "labels" / "train"
    out_train_img.mkdir(parents=True, exist_ok=True)
    out_train_lbl.mkdir(parents=True, exist_ok=True)

    for split in ("val", "test"):
        src_img = in_dir / "images" / split
        src_lbl = in_dir / "labels" / split
        if src_img.exists():
            dst_img = out_dir / "images" / split
            dst_img.mkdir(parents=True, exist_ok=True)
            for entry in src_img.iterdir():
                _copy_or_link(entry, dst_img / entry.name, copy=copy_eval_images)
        if src_lbl.exists():
            dst_lbl = out_dir / "labels" / split
            dst_lbl.mkdir(parents=True, exist_ok=True)
            for entry in src_lbl.iterdir():
                _copy_or_link(entry, dst_lbl / entry.name, copy=True)

    coco_src = in_dir / "coco"
    if coco_src.exists():
        coco_dst = out_dir / "coco"
        coco_dst.mkdir(parents=True, exist_ok=True)
        for entry in coco_src.iterdir():
            _copy_or_link(entry, coco_dst / entry.name, copy=True)

    train_images = sorted(p for p in train_img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not train_images:
        raise SystemExit(f"No train images found in {train_img_dir}")

    stats = {
        "original": 0,
        "jpeg_recompress": 0,
        "crop": 0,
        "crop_skipped_too_small": 0,
        "crop_skipped_no_visible_boxes": 0,
        "images_with_no_boxes": 0,
    }

    for src_img in tqdm(train_images, desc="building augmented train"):
        stem = src_img.stem
        src_lbl = train_lbl_dir / f"{stem}.txt"
        boxes = _read_yolo_labels(src_lbl)
        if not boxes:
            stats["images_with_no_boxes"] += 1

        dst_img = out_train_img / src_img.name
        _copy_or_link(src_img, dst_img, copy=True)
        if src_lbl.exists():
            _copy_or_link(src_lbl, out_train_lbl / src_lbl.name, copy=True)
        else:
            _write_yolo_labels(out_train_lbl / f"{stem}.txt", [])
        stats["original"] += 1

        with Image.open(src_img) as image:
            image.load()
            image_rgb = image.convert("RGB")

            for j in range(jpeg_per_image):
                quality = rng.randint(jpeg_quality_low, jpeg_quality_high)
                derivative_name = f"{stem}__jpegq{quality}_{j}.jpg"
                derivative_img = out_train_img / derivative_name
                derivative_lbl = out_train_lbl / f"{derivative_name[:-4]}.txt"
                buf = io.BytesIO(_random_jpeg_bytes(image_rgb, quality))
                with open(derivative_img, "wb") as f:
                    f.write(buf.getvalue())
                _write_yolo_labels(derivative_lbl, boxes)
                stats["jpeg_recompress"] += 1

            for c in range(crops_per_image):
                if not boxes:
                    stats["crop_skipped_no_visible_boxes"] += 1
                    continue
                if image_rgb.width < crop_size or image_rgb.height < crop_size:
                    stats["crop_skipped_too_small"] += 1
                    continue
                result = _random_crop(
                    image_rgb,
                    boxes,
                    crop_size=crop_size,
                    min_visibility=min_visibility,
                    rng=rng,
                )
                if result is None:
                    stats["crop_skipped_no_visible_boxes"] += 1
                    continue
                cropped, new_boxes = result
                derivative_name = f"{stem}__crop{c}.jpg"
                cropped.save(out_train_img / derivative_name, format="JPEG", quality=92)
                _write_yolo_labels(out_train_lbl / f"{stem}__crop{c}.txt", new_boxes)
                stats["crop"] += 1

    data_yaml_src = in_dir / "data.yaml"
    if data_yaml_src.exists():
        text = data_yaml_src.read_text()
        new_lines: list[str] = []
        for line in text.splitlines():
            if line.startswith("path:"):
                new_lines.append(f"path: {out_dir}")
            else:
                new_lines.append(line)
        (out_dir / "data.yaml").write_text("\n".join(new_lines) + "\n")
    else:
        sys.stderr.write(f"WARN: {data_yaml_src} not found; emitting minimal data.yaml\n")
        (out_dir / "data.yaml").write_text(
            f"path: {out_dir}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        )

    print()
    print("Augmented train counts:")
    for k, v in stats.items():
        print(f"  {k:<32} {v}")
    total = stats["original"] + stats["jpeg_recompress"] + stats["crop"]
    print(f"  {'TOTAL train samples':<32} {total}")
    print(f"\nOutput: {out_dir}")
    print(f"Data yaml: {out_dir / 'data.yaml'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-dir", type=Path, default=Path("/home/jupyter/cv_yolo_dataset"))
    parser.add_argument("--out-dir", type=Path, default=Path("/home/jupyter/cv_yolo_dataset_augc1"))
    parser.add_argument("--crops-per-image", type=int, default=1)
    parser.add_argument("--jpeg-per-image", type=int, default=1)
    parser.add_argument("--crop-size", type=int, default=1024)
    parser.add_argument("--min-visibility", type=float, default=0.5)
    parser.add_argument("--jpeg-quality-low", type=int, default=40)
    parser.add_argument("--jpeg-quality-high", type=int, default=85)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--copy-eval-images",
        action="store_true",
        default=True,
        help="Copy val/test images into the new dataset (default: copy). "
        "Pass --no-copy-eval-images to symlink instead.",
    )
    parser.add_argument("--no-copy-eval-images", dest="copy_eval_images", action="store_false")
    args = parser.parse_args()

    build(
        in_dir=args.in_dir,
        out_dir=args.out_dir,
        crops_per_image=args.crops_per_image,
        jpeg_per_image=args.jpeg_per_image,
        crop_size=args.crop_size,
        min_visibility=args.min_visibility,
        jpeg_quality_low=args.jpeg_quality_low,
        jpeg_quality_high=args.jpeg_quality_high,
        seed=args.seed,
        copy_eval_images=args.copy_eval_images,
    )


if __name__ == "__main__":
    main()
