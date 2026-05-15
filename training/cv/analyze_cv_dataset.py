#!/usr/bin/env python3
"""Summarize a local TIL CV COCO-style dataset.

This is intentionally lightweight: it checks annotation/image consistency,
class balance, object sizes, and dense scenes without needing Ultralytics or a
GPU. It is useful after downloading `/home/jupyter/novice/cv` locally.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    from PIL import Image
except Exception:  # pragma: no cover - optional local dependency
    Image = None


def _quantiles(values: list[float], qs: list[float]) -> list[tuple[int, float]]:
    if not values:
        return [(int(q * 100), 0.0) for q in qs]
    ordered = sorted(values)
    out = []
    for q in qs:
        idx = round(q * (len(ordered) - 1))
        out.append((int(q * 100), ordered[idx]))
    return out


def _sort_filename(name: str) -> tuple[int, str]:
    stem = Path(name).stem
    return (int(stem), name) if stem.isdigit() else (10**12, name)


def _verify_images(image_dir: Path) -> tuple[Counter[tuple[int, int]], list[tuple[str, str]]]:
    dims: Counter[tuple[int, int]] = Counter()
    bad: list[tuple[str, str]] = []
    if Image is None:
        return dims, [("<all>", "Pillow unavailable; skipped decode verification")]

    for path in sorted(image_dir.glob("*"), key=lambda p: _sort_filename(p.name)):
        if not path.is_file():
            continue
        try:
            with Image.open(path) as img:
                img.verify()
                dims[(img.width, img.height)] += 1
        except Exception as exc:
            bad.append((path.name, str(exc)))
    return dims, bad


def analyze(root: Path, verify_images: bool) -> dict[str, Any]:
    ann_path = root / "annotations.json"
    image_dir = root / "images"
    data = json.loads(ann_path.read_text())

    images = data["images"]
    annotations = data["annotations"]
    categories = {int(cat["id"]): cat["name"] for cat in data["categories"]}
    image_by_id = {img["id"]: img for img in images}

    ann_by_img: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    class_boxes: Counter[int] = Counter()
    class_images: dict[int, set[Any]] = defaultdict(set)
    area_groups: dict[int, Counter[str]] = defaultdict(Counter)
    rel_areas: list[float] = []
    invalid_bbox_ids: list[Any] = []

    for ann in annotations:
        image = image_by_id[ann["image_id"]]
        ann_by_img[ann["image_id"]].append(ann)
        cid = int(ann["category_id"])
        class_boxes[cid] += 1
        class_images[cid].add(ann["image_id"])

        x, y, w, h = (float(v) for v in ann["bbox"])
        if (
            w <= 0
            or h <= 0
            or x < 0
            or y < 0
            or x + w > float(image["width"]) + 1e-3
            or y + h > float(image["height"]) + 1e-3
        ):
            invalid_bbox_ids.append(ann["id"])

        area = w * h
        if area < 32 * 32:
            area_groups[cid]["small"] += 1
        elif area < 96 * 96:
            area_groups[cid]["medium"] += 1
        else:
            area_groups[cid]["large"] += 1
        rel_areas.append(area / (float(image["width"]) * float(image["height"])))

    present = {path.name for path in image_dir.glob("*") if path.is_file()}
    expected = {img["file_name"] for img in images}
    missing = sorted(expected - present, key=_sort_filename)
    extra = sorted(present - expected, key=_sort_filename)
    present_image_ids = {img["id"] for img in images if img["file_name"] in present}
    present_annotations = [ann for ann in annotations if ann["image_id"] in present_image_ids]

    decoded_dims: Counter[tuple[int, int]] = Counter()
    bad_images: list[tuple[str, str]] = []
    if verify_images:
        decoded_dims, bad_images = _verify_images(image_dir)

    boxes_per_image = [len(ann_by_img[img["id"]]) for img in images]
    dense_images = []
    for image_id, rows in sorted(
        ann_by_img.items(), key=lambda item: len(item[1]), reverse=True
    )[:10]:
        image = image_by_id[image_id]
        top_classes = Counter(categories[int(ann["category_id"])] for ann in rows)
        dense_images.append(
            {
                "file_name": image["file_name"],
                "boxes": len(rows),
                "present": image["file_name"] in present,
                "top_classes": top_classes.most_common(3),
            }
        )

    class_rows = []
    for cid in sorted(categories):
        groups = area_groups[cid]
        class_rows.append(
            {
                "id": cid,
                "name": categories[cid],
                "boxes": class_boxes[cid],
                "images": len(class_images[cid]),
                "small": groups["small"],
                "medium": groups["medium"],
                "large": groups["large"],
            }
        )

    return {
        "images": len(images),
        "annotations": len(annotations),
        "categories": len(categories),
        "present_images": len(present),
        "missing_images": len(missing),
        "extra_images": len(extra),
        "missing_samples": missing[:10],
        "bad_images": bad_images[:10],
        "present_annotations": len(present_annotations),
        "invalid_bboxes": len(invalid_bbox_ids),
        "annotation_dims": Counter((img["width"], img["height"]) for img in images),
        "decoded_dims": decoded_dims,
        "boxes_per_image": boxes_per_image,
        "zero_box_images": sum(1 for value in boxes_per_image if value == 0),
        "relative_areas": rel_areas,
        "class_rows": class_rows,
        "dense_images": dense_images,
    }


def print_report(summary: dict[str, Any]) -> None:
    print("Dataset sanity")
    print(f"  images in annotations: {summary['images']}")
    print(f"  annotations: {summary['annotations']}")
    print(f"  categories: {summary['categories']}")
    print(f"  extracted image files: {summary['present_images']} / {summary['images']}")
    print(f"  missing image files: {summary['missing_images']}")
    print(f"  extra image files: {summary['extra_images']}")
    print(f"  bad/corrupt extracted images: {len(summary['bad_images'])}")
    if summary["missing_samples"]:
        print(f"  first missing: {summary['missing_samples']}")
    if summary["bad_images"]:
        print(f"  bad samples: {summary['bad_images']}")
    print(
        f"  extracted annotations covered: "
        f"{summary['present_annotations']} / {summary['annotations']}"
    )
    print(f"  invalid bbox annotations: {summary['invalid_bboxes']}")

    print("\nImage dimensions from annotations")
    for (width, height), count in summary["annotation_dims"].most_common(10):
        print(f"  {width}x{height}: {count}")
    if summary["decoded_dims"]:
        print("\nDecoded extracted image dimensions")
        for (width, height), count in summary["decoded_dims"].most_common(10):
            print(f"  {width}x{height}: {count}")

    print("\nBoxes per image")
    for q, value in _quantiles(summary["boxes_per_image"], [0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1]):
        print(f"  p{q:02d}: {value:.0f}")
    print(f"  mean: {statistics.mean(summary['boxes_per_image']):.2f}")
    print(f"  zero-box images: {summary['zero_box_images']}")

    print("\nBbox relative area")
    for q, value in _quantiles(summary["relative_areas"], [0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1]):
        print(f"  p{q:02d}: {100 * value:.4f}%")

    print("\nClass counts")
    for row in summary["class_rows"]:
        print(
            f"  {row['id']:02d} {row['name']:<22} "
            f"boxes={row['boxes']:5d} images={row['images']:4d} "
            f"small={row['small']:4d} medium={row['medium']:4d} large={row['large']:4d}"
        )

    print("\nDensest images")
    for row in summary["dense_images"]:
        print(
            f"  {row['file_name']:>8} boxes={row['boxes']:2d} "
            f"present={row['present']} top={row['top_classes']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/novice/cv"))
    parser.add_argument("--skip-image-verify", action="store_true")
    args = parser.parse_args()

    print_report(analyze(args.root, verify_images=not args.skip_image_verify))


if __name__ == "__main__":
    main()
