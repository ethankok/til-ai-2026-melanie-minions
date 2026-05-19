#!/usr/bin/env python3
"""Build a final-training YOLO dataset from the existing CV split manifest.

The normal hard split reserves many rare/dense images for val/test. That is
useful for diagnostics, but the final leaderboard model can train on more of
the labeled novice data. This script keeps the split choices explicit:

* default train set: old train + old val
* default validation set: old hard test

It writes a standard Ultralytics layout with symlinked images and YOLO labels.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from prepare_yolo_dataset import (
    _clip_box,
    _link_or_copy,
    _load_categories,
    _safe_image_name,
    _safe_name,
    _split_annotations,
)


def _parse_splits(raw: str) -> list[str]:
    splits = [part.strip() for part in raw.split(",") if part.strip()]
    if not splits:
        raise ValueError("At least one split is required")
    return splits


def _ids_for_splits(manifest: dict[str, Any], split_names: list[str]) -> set[Any]:
    manifest_splits = manifest.get("splits", {})
    missing = [split for split in split_names if split not in manifest_splits]
    if missing:
        available = ", ".join(sorted(manifest_splits))
        raise KeyError(f"Missing split(s) {missing}; available splits: {available}")
    ids: set[Any] = set()
    for split in split_names:
        ids.update(manifest_splits[split])
    return ids


def _write_split(
    split: str,
    image_ids: set[Any],
    annotations: dict[str, Any],
    anns_by_image: dict[Any, list[dict[str, Any]]],
    cat_id_to_idx: dict[int, int],
    source_images_dir: Path,
    out_dir: Path,
    copy_images: bool,
) -> tuple[int, int]:
    image_count = 0
    box_count = 0
    for image in annotations["images"]:
        if image["id"] not in image_ids:
            continue
        src_img = source_images_dir / image["file_name"]
        if not src_img.exists():
            raise FileNotFoundError(f"Missing image: {src_img}")

        dst_img = out_dir / "images" / split / _safe_image_name(image["file_name"])
        _link_or_copy(src_img, dst_img, copy_images)

        img_w = float(image["width"])
        img_h = float(image["height"])
        lines: list[str] = []
        for ann in anns_by_image.get(image["id"], []):
            box = _clip_box(*map(float, ann["bbox"]), img_w, img_h)
            if box is None:
                continue
            cls_idx = cat_id_to_idx[int(ann["category_id"])]
            xc, yc, bw, bh = box
            lines.append(f"{cls_idx} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        label_path = out_dir / "labels" / split / (dst_img.stem + ".txt")
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        image_count += 1
        box_count += len(lines)
    return image_count, box_count


def build_dataset(
    data_dir: Path,
    base_dataset: Path,
    out_dir: Path,
    train_splits: list[str],
    val_split: str,
    copy_images: bool,
) -> None:
    annotations_path = data_dir / "annotations.json"
    manifest_path = base_dataset / "split_manifest.json"
    annotations: dict[str, Any] = json.loads(annotations_path.read_text())
    manifest: dict[str, Any] = json.loads(manifest_path.read_text())

    train_ids = _ids_for_splits(manifest, train_splits)
    val_ids = _ids_for_splits(manifest, [val_split])
    overlap = train_ids & val_ids
    if overlap:
        print(f"Warning: train/val overlap contains {len(overlap)} images")

    categories = _load_categories(annotations)
    cat_id_to_idx = {int(cat["id"]): idx for idx, cat in enumerate(categories)}
    names = [_safe_name(str(cat.get("name", cat["id"]))) for cat in categories]

    anns_by_image: dict[Any, list[dict[str, Any]]] = {img["id"]: [] for img in annotations["images"]}
    for ann in annotations.get("annotations", []):
        image_id = ann.get("image_id")
        if image_id not in anns_by_image:
            continue
        if ann.get("iscrowd", 0):
            continue
        if int(ann.get("category_id", -1)) not in cat_id_to_idx:
            continue
        anns_by_image[image_id].append(ann)

    for split in ("train", "val"):
        shutil.rmtree(out_dir / "images" / split, ignore_errors=True)
        shutil.rmtree(out_dir / "labels" / split, ignore_errors=True)
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    source_images_dir = data_dir / "images"
    train_counts = _write_split(
        "train",
        train_ids,
        annotations,
        anns_by_image,
        cat_id_to_idx,
        source_images_dir,
        out_dir,
        copy_images,
    )
    val_counts = _write_split(
        "val",
        val_ids,
        annotations,
        anns_by_image,
        cat_id_to_idx,
        source_images_dir,
        out_dir,
        copy_images,
    )

    coco_dir = out_dir / "coco"
    coco_dir.mkdir(parents=True, exist_ok=True)
    (coco_dir / "annotations_train.json").write_text(
        json.dumps(_split_annotations(annotations, train_ids, categories)),
        encoding="utf-8",
    )
    (coco_dir / "annotations_val.json").write_text(
        json.dumps(_split_annotations(annotations, val_ids, categories)),
        encoding="utf-8",
    )

    final_manifest = {
        "source_annotations": str(annotations_path),
        "base_manifest": str(manifest_path),
        "train_splits": train_splits,
        "val_split": val_split,
        "splits": {
            "train": [img["id"] for img in annotations["images"] if img["id"] in train_ids],
            "val": [img["id"] for img in annotations["images"] if img["id"] in val_ids],
        },
    }
    (out_dir / "split_manifest.json").write_text(
        json.dumps(final_manifest, indent=2),
        encoding="utf-8",
    )

    yaml_lines = [
        f"path: {out_dir}",
        "train: images/train",
        "val: images/val",
        f"nc: {len(names)}",
        "names:",
    ]
    yaml_lines.extend(f"  {idx}: {name}" for idx, name in enumerate(names))
    (out_dir / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    print(f"Wrote {out_dir / 'data.yaml'}")
    print(f"Train splits: {','.join(train_splits)}")
    print(f"Val split: {val_split}")
    print(f"Images: train={train_counts[0]} val={val_counts[0]}")
    print(f"Boxes: train={train_counts[1]} val={val_counts[1]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("/home/jupyter/novice/cv"))
    parser.add_argument("--base-dataset", type=Path, default=Path("/home/jupyter/cv_yolo_dataset"))
    parser.add_argument("--out-dir", type=Path, default=Path("/home/jupyter/cv_yolo_dataset_plusval"))
    parser.add_argument("--train-splits", default="train,val")
    parser.add_argument("--val-split", default="test")
    parser.add_argument("--copy-images", action="store_true")
    args = parser.parse_args()

    build_dataset(
        data_dir=args.data_dir,
        base_dataset=args.base_dataset,
        out_dir=args.out_dir,
        train_splits=_parse_splits(args.train_splits),
        val_split=args.val_split,
        copy_images=args.copy_images,
    )


if __name__ == "__main__":
    main()
