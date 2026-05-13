#!/usr/bin/env python3
"""Convert the TIL CV COCO-style annotations into an Ultralytics YOLO dataset.

Expected input layout on Workbench:

    /home/jupyter/novice/cv/
      annotations.json
      images/<file_name>.jpg

Output layout:

    /home/jupyter/cv_yolo_dataset/
      data.yaml
      images/train/*.jpg
      images/val/*.jpg
      labels/train/*.txt
      labels/val/*.txt

YOLO labels are: class_idx x_center y_center width height, normalized to [0, 1].
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any


def _safe_name(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_")


def _link_or_copy(src: Path, dst: Path, copy: bool) -> None:
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


def _clip_box(x: float, y: float, w: float, h: float, img_w: float, img_h: float):
    x1 = max(0.0, min(x, img_w))
    y1 = max(0.0, min(y, img_h))
    x2 = max(0.0, min(x + w, img_w))
    y2 = max(0.0, min(y + h, img_h))
    bw = x2 - x1
    bh = y2 - y1
    if bw <= 0.0 or bh <= 0.0 or img_w <= 0.0 or img_h <= 0.0:
        return None
    return ((x1 + x2) / 2.0 / img_w, (y1 + y2) / 2.0 / img_h, bw / img_w, bh / img_h)


def convert(data_dir: Path, out_dir: Path, val_frac: float, seed: int, copy_images: bool) -> None:
    ann_path = data_dir / "annotations.json"
    images_dir = data_dir / "images"
    annotations: dict[str, Any] = json.loads(ann_path.read_text())

    categories = sorted(annotations.get("categories", []), key=lambda c: int(c["id"]))
    if not categories:
        # TIL observed IDs/classes; fallback if annotations omit category metadata.
        names = [
            "cargo aircraft", "commercial aircraft", "drone", "fighter jet", "fighter plane",
            "helicopter", "light aircraft", "missile", "truck", "car", "tank", "bus", "van",
            "cargo ship", "yacht", "cruise ship", "warship", "sailboat",
        ]
        categories = [{"id": i, "name": name} for i, name in enumerate(names)]
    cat_id_to_idx = {int(cat["id"]): idx for idx, cat in enumerate(categories)}
    names = [_safe_name(str(cat.get("name", cat["id"]))) for cat in categories]

    images = list(annotations["images"])
    rng = random.Random(seed)
    rng.shuffle(images)
    val_count = max(1, int(round(len(images) * val_frac))) if len(images) > 1 else 0
    val_ids = {img["id"] for img in images[:val_count]}
    image_by_id = {img["id"]: img for img in annotations["images"]}

    anns_by_image: dict[Any, list[dict[str, Any]]] = {img["id"]: [] for img in annotations["images"]}
    skipped_cats = 0
    for ann in annotations.get("annotations", []):
        image_id = ann.get("image_id")
        if image_id not in anns_by_image:
            continue
        if int(ann.get("category_id", -1)) not in cat_id_to_idx:
            skipped_cats += 1
            continue
        if ann.get("iscrowd", 0):
            continue
        anns_by_image[image_id].append(ann)

    for split in ["train", "val"]:
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    split_counts = {"train": 0, "val": 0}
    box_counts = {"train": 0, "val": 0}
    for img in annotations["images"]:
        split = "val" if img["id"] in val_ids else "train"
        file_name = img["file_name"]
        src_img = images_dir / file_name
        if not src_img.exists():
            raise FileNotFoundError(f"Missing image: {src_img}")
        dst_img = out_dir / "images" / split / Path(file_name).name
        _link_or_copy(src_img, dst_img, copy_images)

        img_w = float(img["width"])
        img_h = float(img["height"])
        lines: list[str] = []
        for ann in anns_by_image.get(img["id"], []):
            box = _clip_box(*map(float, ann["bbox"]), img_w, img_h)
            if box is None:
                continue
            cls_idx = cat_id_to_idx[int(ann["category_id"])]
            xc, yc, bw, bh = box
            lines.append(f"{cls_idx} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
        (out_dir / "labels" / split / (Path(file_name).stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""))
        split_counts[split] += 1
        box_counts[split] += len(lines)

    yaml_lines = [
        f"path: {out_dir}",
        "train: images/train",
        "val: images/val",
        f"nc: {len(names)}",
        "names:",
    ]
    yaml_lines.extend(f"  {idx}: {name}" for idx, name in enumerate(names))
    (out_dir / "data.yaml").write_text("\n".join(yaml_lines) + "\n")

    print(f"Wrote {out_dir / 'data.yaml'}")
    print(f"Classes: {len(names)}")
    print(f"Images: train={split_counts['train']} val={split_counts['val']}")
    print(f"Boxes: train={box_counts['train']} val={box_counts['val']}")
    if skipped_cats:
        print(f"Skipped annotations with unknown category_id: {skipped_cats}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("/home/jupyter/novice/cv"))
    parser.add_argument("--out-dir", type=Path, default=Path("/home/jupyter/cv_yolo_dataset"))
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copy-images", action="store_true", help="Copy images instead of symlinking them")
    args = parser.parse_args()
    convert(args.data_dir, args.out_dir, args.val_frac, args.seed, args.copy_images)


if __name__ == "__main__":
    main()
