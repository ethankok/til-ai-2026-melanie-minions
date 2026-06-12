#!/usr/bin/env python3
"""Build an RF-DETR (COCO-format) dataset from the existing CV split manifest.

RF-DETR (Roboflow) expects a different on-disk layout from the Ultralytics YOLO
trainer: three sibling directories ``train/ valid/ test/``, each holding the
split's image files **and** a single ``_annotations.coco.json`` next to them
(file_name fields are basenames, not paths into an images/ subdir).

This mirrors the all-data recipe of ``build_final_yolo_dataset.py``:

* ``train/`` = old train + old val images (more labels for the final model)
* ``valid/`` = old hard test split (the diagnostic stress split)
* ``test/``  = same hard test split (RF-DETR requires all three dirs to exist)

Category indexing (the one RF-DETR gotcha):
  The source TIL annotations use category_id 0..17. RF-DETR / the Roboflow COCO
  convention reserves class id 0 as a placeholder, so by default we shift every
  category id by ``--category-offset`` (default 1) -> ids 1..18, and emit a dummy
  id-0 "background" category with no annotations. The served model then predicts
  class_id in 1..18, which the manager maps back to TIL 0..17 via CV_CATEGORY_MAP.

  With the default offset=1, set on the rfdetr container:
    CV_CATEGORY_MAP='{"1":0,"2":1,"3":2,"4":3,"5":4,"6":5,"7":6,"8":7,"9":8,
                      "10":9,"11":10,"12":11,"13":12,"14":13,"15":14,"16":15,
                      "17":16,"18":17}'
  If a 3-epoch smoke shows RF-DETR uses 0-indexed ids instead, rebuild with
  ``--category-offset 0`` and use the identity CV_CATEGORY_MAP. VERIFY before the
  full run (see cv/NOTES.md).

Usage (Workbench):
  python training/cv/build_rfdetr_dataset.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from prepare_yolo_dataset import (
    _link_or_copy,
    _load_categories,
    _safe_image_name,
    _safe_name,
    _split_annotations,
)
from build_final_yolo_dataset import _ids_for_splits, _parse_splits


_RFDETR_SPLIT_DIRS = ("train", "valid", "test")


def _remap_coco(
    coco: dict[str, Any],
    categories: list[dict[str, Any]],
    offset: int,
) -> dict[str, Any]:
    """Apply the category-id offset and rewrite file_names to safe basenames.

    Returns a new COCO dict whose ``categories`` are offset (+ dummy id 0 when
    offset>0), whose annotation ``category_id``s are shifted, and whose image
    ``file_name``s match the files copied next to the json.
    """
    new_categories: list[dict[str, Any]] = []
    if offset > 0:
        new_categories.append({"id": 0, "name": "background", "supercategory": "none"})
    for cat in categories:
        new_categories.append(
            {
                "id": int(cat["id"]) + offset,
                "name": _safe_name(str(cat.get("name", cat["id"]))),
                "supercategory": cat.get("supercategory", "none"),
            }
        )

    new_images = []
    for img in coco["images"]:
        new_img = dict(img)
        new_img["file_name"] = _safe_image_name(img["file_name"])
        new_images.append(new_img)

    new_annotations = []
    for ann in coco["annotations"]:
        new_ann = dict(ann)
        new_ann["category_id"] = int(ann["category_id"]) + offset
        new_annotations.append(new_ann)

    return {
        "info": coco.get("info", {"description": "TIL CV RF-DETR split"}),
        "licenses": coco.get("licenses", []),
        "categories": new_categories,
        "images": new_images,
        "annotations": new_annotations,
    }


def _write_split(
    split_dir: str,
    image_ids: set[Any],
    annotations: dict[str, Any],
    categories: list[dict[str, Any]],
    source_images_dir: Path,
    out_dir: Path,
    offset: int,
    copy_images: bool,
) -> tuple[int, int]:
    target = out_dir / split_dir
    target.mkdir(parents=True, exist_ok=True)

    coco = _split_annotations(annotations, image_ids, categories)
    for image in coco["images"]:
        src_img = source_images_dir / image["file_name"]
        if not src_img.exists():
            raise FileNotFoundError(f"Missing image: {src_img}")
        dst_img = target / _safe_image_name(image["file_name"])
        _link_or_copy(src_img, dst_img, copy_images)

    remapped = _remap_coco(coco, categories, offset)
    (target / "_annotations.coco.json").write_text(
        json.dumps(remapped), encoding="utf-8"
    )
    return len(remapped["images"]), len(remapped["annotations"])


def build_dataset(
    data_dir: Path,
    base_dataset: Path,
    out_dir: Path,
    train_splits: list[str],
    val_split: str,
    offset: int,
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
    source_images_dir = data_dir / "images"

    split_ids = {"train": train_ids, "valid": val_ids, "test": val_ids}
    counts: dict[str, tuple[int, int]] = {}
    for split_dir in _RFDETR_SPLIT_DIRS:
        counts[split_dir] = _write_split(
            split_dir,
            split_ids[split_dir],
            annotations,
            categories,
            source_images_dir,
            out_dir,
            offset,
            copy_images,
        )

    final_manifest = {
        "source_annotations": str(annotations_path),
        "base_manifest": str(manifest_path),
        "train_splits": train_splits,
        "val_split": val_split,
        "category_offset": offset,
        "layout": "rfdetr-coco",
    }
    (out_dir / "split_manifest.json").write_text(
        json.dumps(final_manifest, indent=2), encoding="utf-8"
    )

    print(f"Wrote RF-DETR dataset to {out_dir}")
    print(f"Train splits: {','.join(train_splits)}  Val/test split: {val_split}")
    print(f"Category offset: {offset} (categories ids {offset}..{offset + len(categories) - 1})")
    for split_dir in _RFDETR_SPLIT_DIRS:
        img_n, box_n = counts[split_dir]
        print(f"  {split_dir}: images={img_n} boxes={box_n}")
    if offset > 0:
        pairs = ",".join(f'"{i + offset}":{i}' for i in range(len(categories)))
        print("Set on the rfdetr container:")
        print(f"  CV_CATEGORY_MAP='{{{pairs}}}'")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("/home/jupyter/novice/cv"))
    parser.add_argument("--base-dataset", type=Path, default=Path("/home/jupyter/cv_yolo_dataset"))
    parser.add_argument("--out-dir", type=Path, default=Path("/home/jupyter/cv_rfdetr_dataset"))
    parser.add_argument("--train-splits", default="train,val")
    parser.add_argument("--val-split", default="test")
    parser.add_argument(
        "--category-offset",
        type=int,
        default=1,
        help="Shift applied to source category ids (default 1 for RF-DETR's "
        "reserved class 0). Use 0 if a smoke run shows 0-indexed classes.",
    )
    parser.add_argument(
        "--symlink-images",
        action="store_true",
        help="Symlink instead of copy images (default copies, safer for RF-DETR's loader).",
    )
    args = parser.parse_args()

    build_dataset(
        data_dir=args.data_dir,
        base_dataset=args.base_dataset,
        out_dir=args.out_dir,
        train_splits=_parse_splits(args.train_splits),
        val_split=args.val_split,
        offset=args.category_offset,
        copy_images=not args.symlink_images,
    )


if __name__ == "__main__":
    main()
