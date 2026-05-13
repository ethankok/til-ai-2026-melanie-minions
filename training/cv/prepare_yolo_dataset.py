#!/usr/bin/env python3
"""Convert the TIL CV COCO-style annotations into an Ultralytics YOLO dataset.

Expected input layout on Workbench:

    /home/jupyter/novice/cv/
      annotations.json
      images/<file_name>.jpg

Output layout:

    /home/jupyter/cv_yolo_dataset/
      data.yaml
      split_manifest.json
      coco/annotations_train.json
      coco/annotations_val.json
      coco/annotations_test.json
      images/train/*.jpg
      images/val/*.jpg
      images/test/*.jpg
      labels/train/*.txt
      labels/val/*.txt
      labels/test/*.txt

YOLO labels are: class_idx x_center y_center width height, normalized to [0, 1].
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any


TIL_CATEGORY_NAMES = [
    "cargo aircraft",
    "commercial aircraft",
    "drone",
    "fighter jet",
    "fighter plane",
    "helicopter",
    "light aircraft",
    "missile",
    "truck",
    "car",
    "tank",
    "bus",
    "van",
    "cargo ship",
    "yacht",
    "cruise ship",
    "warship",
    "sailboat",
]


def _safe_name(name: str) -> str:
    return name.replace("/", "__").replace("\\", "__").replace(" ", "_")


def _safe_image_name(file_name: str) -> str:
    path = Path(file_name)
    return _safe_name(str(path.with_suffix(""))) + path.suffix


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


def _load_categories(annotations: dict[str, Any]) -> list[dict[str, Any]]:
    categories = sorted(annotations.get("categories", []), key=lambda c: int(c["id"]))
    if categories:
        return categories
    return [{"id": i, "name": name} for i, name in enumerate(TIL_CATEGORY_NAMES)]


def _split_annotations(
    annotations: dict[str, Any],
    image_ids: set[Any],
    categories: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "info": annotations.get("info", {"description": "TIL CV split"}),
        "licenses": annotations.get("licenses", []),
        "categories": categories,
        "images": [img for img in annotations["images"] if img["id"] in image_ids],
        "annotations": [
            ann for ann in annotations.get("annotations", []) if ann.get("image_id") in image_ids
        ],
    }


def _difficulty_scores(
    images: list[dict[str, Any]],
    anns_by_image: dict[Any, list[dict[str, Any]]],
    class_counts: dict[int, int],
) -> dict[Any, float]:
    """Score images that are useful held-out stress cases.

    Higher is harder: small boxes, rare classes, and crowded scenes. Empty
    images still receive a small score so the eval keeps some false-positive
    pressure, but hard splits mainly reserve challenging target images.
    """
    max_class_count = max(class_counts.values(), default=1)
    scores: dict[Any, float] = {}
    for img in images:
        img_w = max(float(img.get("width", 1)), 1.0)
        img_h = max(float(img.get("height", 1)), 1.0)
        anns = anns_by_image.get(img["id"], [])
        if not anns:
            scores[img["id"]] = 0.1
            continue

        small_score = 0.0
        rare_score = 0.0
        for ann in anns:
            _, _, bw, bh = map(float, ann["bbox"])
            area_ratio = max((bw * bh) / (img_w * img_h), 1e-9)
            small_score += min(1.0, 0.01 / area_ratio)
            rare_score += 1.0 - (class_counts.get(int(ann["category_id"]), 0) / max_class_count)
        scores[img["id"]] = (small_score / len(anns)) + rare_score + min(len(anns), 8) * 0.05
    return scores


def _assign_splits(
    images: list[dict[str, Any]],
    anns_by_image: dict[Any, list[dict[str, Any]]],
    val_frac: float,
    test_frac: float,
    seed: int,
    split_mode: str,
) -> dict[Any, str]:
    if val_frac < 0 or test_frac < 0 or val_frac + test_frac >= 1:
        raise ValueError("--val-frac and --test-frac must be >= 0 and sum to < 1")

    rng = random.Random(seed)
    shuffled = list(images)
    rng.shuffle(shuffled)
    n_images = len(shuffled)
    val_count = max(1, int(round(n_images * val_frac))) if val_frac > 0 and n_images > 1 else 0
    test_count = max(1, int(round(n_images * test_frac))) if test_frac > 0 and n_images > 2 else 0
    if val_count + test_count >= n_images:
        test_count = max(0, n_images - val_count - 1)

    if split_mode == "hard":
        class_counts: dict[int, int] = {}
        for anns in anns_by_image.values():
            for ann in anns:
                cls_id = int(ann["category_id"])
                class_counts[cls_id] = class_counts.get(cls_id, 0) + 1
        scores = _difficulty_scores(shuffled, anns_by_image, class_counts)
        shuffled.sort(key=lambda img: (scores.get(img["id"], 0.0), rng.random()), reverse=True)

    split_by_id: dict[Any, str] = {}
    for idx, img in enumerate(shuffled):
        if idx < test_count:
            split_by_id[img["id"]] = "test"
        elif idx < test_count + val_count:
            split_by_id[img["id"]] = "val"
        else:
            split_by_id[img["id"]] = "train"
    return split_by_id


def convert(
    data_dir: Path,
    out_dir: Path,
    val_frac: float,
    test_frac: float,
    seed: int,
    split_mode: str,
    copy_images: bool,
) -> None:
    ann_path = data_dir / "annotations.json"
    images_dir = data_dir / "images"
    annotations: dict[str, Any] = json.loads(ann_path.read_text())

    categories = _load_categories(annotations)
    cat_id_to_idx = {int(cat["id"]): idx for idx, cat in enumerate(categories)}
    names = [_safe_name(str(cat.get("name", cat["id"]))) for cat in categories]

    images = list(annotations["images"])
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

    split_by_id = _assign_splits(images, anns_by_image, val_frac, test_frac, seed, split_mode)
    split_names = ["train", "val"] + (["test"] if any(s == "test" for s in split_by_id.values()) else [])

    for split in ["train", "val", "test"]:
        shutil.rmtree(out_dir / "images" / split, ignore_errors=True)
        shutil.rmtree(out_dir / "labels" / split, ignore_errors=True)
    for split in split_names:
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    split_counts = {split: 0 for split in split_names}
    box_counts = {split: 0 for split in split_names}
    for img in annotations["images"]:
        split = split_by_id[img["id"]]
        file_name = img["file_name"]
        src_img = images_dir / file_name
        if not src_img.exists():
            raise FileNotFoundError(f"Missing image: {src_img}")
        dst_img = out_dir / "images" / split / _safe_image_name(file_name)
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
        label_path = out_dir / "labels" / split / (dst_img.stem + ".txt")
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        split_counts[split] += 1
        box_counts[split] += len(lines)

    coco_dir = out_dir / "coco"
    coco_dir.mkdir(parents=True, exist_ok=True)
    for split in split_names:
        split_ids = {image_id for image_id, assigned in split_by_id.items() if assigned == split}
        split_ann = _split_annotations(annotations, split_ids, categories)
        (coco_dir / f"annotations_{split}.json").write_text(json.dumps(split_ann), encoding="utf-8")

    manifest = {
        "source_annotations": str(ann_path),
        "seed": seed,
        "split_mode": split_mode,
        "val_frac": val_frac,
        "test_frac": test_frac,
        "splits": {
            split: [img["id"] for img in annotations["images"] if split_by_id[img["id"]] == split]
            for split in split_names
        },
    }
    (out_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    yaml_lines = [
        f"path: {out_dir}",
        "train: images/train",
        "val: images/val",
        f"nc: {len(names)}",
        "names:",
    ]
    if "test" in split_names:
        yaml_lines.insert(3, "test: images/test")
    yaml_lines.extend(f"  {idx}: {name}" for idx, name in enumerate(names))
    (out_dir / "data.yaml").write_text("\n".join(yaml_lines) + "\n")

    print(f"Wrote {out_dir / 'data.yaml'}")
    print(f"Wrote {out_dir / 'split_manifest.json'}")
    print(f"Classes: {len(names)}")
    print("Images: " + " ".join(f"{split}={split_counts[split]}" for split in split_names))
    print("Boxes: " + " ".join(f"{split}={box_counts[split]}" for split in split_names))
    if skipped_cats:
        print(f"Skipped annotations with unknown category_id: {skipped_cats}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("/home/jupyter/novice/cv"))
    parser.add_argument("--out-dir", type=Path, default=Path("/home/jupyter/cv_yolo_dataset"))
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split-mode",
        choices=["random", "hard"],
        default="hard",
        help="Use random splits or reserve hard small/rare/crowded images for held-out eval.",
    )
    parser.add_argument("--copy-images", action="store_true", help="Copy images instead of symlinking them")
    args = parser.parse_args()
    convert(
        args.data_dir,
        args.out_dir,
        args.val_frac,
        args.test_frac,
        args.seed,
        args.split_mode,
        args.copy_images,
    )


if __name__ == "__main__":
    main()
