#!/usr/bin/env python3
"""Evaluate the running CV service against a held-out COCO split.

This intentionally calls the HTTP endpoint instead of loading YOLO directly so
the score covers the same decode, inference, category-map, and LTWH adapter path
used by `til test` and official submissions.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
    COCO = None
    COCOeval = None
    _PYCOCOTOOLS_ERROR = exc
else:
    _PYCOCOTOOLS_ERROR = None


class COCOPatched(COCO if COCO is not None else object):
    """COCO wrapper that accepts an in-memory annotation dictionary."""

    def __init__(self, annotations: dict[str, Any]):
        if COCO is None:
            raise RuntimeError(
                "pycocotools is required for CV scoring. Run `pip install -r requirements-dev.txt`."
            ) from _PYCOCOTOOLS_ERROR
        self.dataset, self.anns, self.cats, self.imgs = {}, {}, {}, {}
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
        self.dataset = annotations
        self.createIndex()


def _batched(items: Sequence[Mapping[str, Any]], batch_size: int) -> Iterator[list[Mapping[str, Any]]]:
    for idx in range(0, len(items), batch_size):
        yield list(items[idx : idx + batch_size])


def _subset_annotations(annotations: dict[str, Any], image_ids: set[Any]) -> dict[str, Any]:
    return {
        "info": annotations.get("info", {"description": "TIL CV eval subset"}),
        "licenses": annotations.get("licenses", []),
        "categories": annotations.get("categories", []),
        "images": [img for img in annotations["images"] if img["id"] in image_ids],
        "annotations": [
            ann for ann in annotations.get("annotations", []) if ann.get("image_id") in image_ids
        ],
    }


def _image_instances(images: Iterable[Mapping[str, Any]], images_dir: Path) -> Iterator[dict[str, Any]]:
    for image in images:
        with open(images_dir / str(image["file_name"]), "rb") as img_file:
            yield {
                "key": image["id"],
                "b64": base64.b64encode(img_file.read()).decode("ascii"),
            }


def _clean_detection(detection: Mapping[str, Any], image_id: Any) -> dict[str, Any] | None:
    try:
        bbox = [float(v) for v in detection["bbox"]]
        category_id = int(detection["category_id"])
    except (KeyError, TypeError, ValueError):
        return None
    if len(bbox) != 4 or bbox[2] <= 0 or bbox[3] <= 0:
        return None
    return {
        "image_id": image_id,
        "category_id": category_id,
        "bbox": [round(v, 4) for v in bbox],
        "score": float(detection.get("score", 1.0)),
    }


def collect_predictions(
    annotations: dict[str, Any],
    images_dir: Path,
    endpoint: str,
    batch_size: int,
    timeout: float,
) -> list[dict[str, Any]]:
    images = annotations["images"]
    results: list[dict[str, Any]] = []
    total = math.ceil(len(images) / batch_size)
    for image_batch in tqdm(list(_batched(images, batch_size)), total=total, desc="CV eval"):
        instances = list(_image_instances(image_batch, images_dir))
        response = requests.post(endpoint, json={"instances": instances}, timeout=timeout)
        response.raise_for_status()
        predictions = response.json()["predictions"]
        if len(predictions) != len(instances):
            raise RuntimeError(
                f"Service returned {len(predictions)} predictions for {len(instances)} instances"
            )
        for instance, detections in zip(instances, predictions):
            for detection in detections:
                cleaned = _clean_detection(detection, instance["key"])
                if cleaned is not None:
                    results.append(cleaned)
    return results


def _mean_valid(values) -> float | None:
    valid = values[values > -1]
    if valid.size == 0:
        return None
    return float(valid.mean())


def score_predictions(
    predictions: Sequence[Mapping[str, Any]],
    annotations: dict[str, Any],
) -> dict[str, Any]:
    if not predictions:
        return {
            "map": 0.0,
            "map50": 0.0,
            "map75": 0.0,
            "per_class": {},
            "per_area": {"all": 0.0, "small": 0.0, "medium": 0.0, "large": 0.0},
        }

    ground_truth = COCOPatched(annotations)
    detections = ground_truth.loadRes(list(predictions))
    evaluator = COCOeval(ground_truth, detections, "bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()

    precision = evaluator.eval["precision"]  # [iou, recall, class, area, max_det]
    area_labels = evaluator.params.areaRngLbl
    max_det_index = len(evaluator.params.maxDets) - 1

    per_area: dict[str, float | None] = {}
    for area_idx, area_name in enumerate(area_labels):
        per_area[area_name] = _mean_valid(precision[:, :, :, area_idx, max_det_index])

    per_class: dict[str, dict[str, float | None]] = {}
    for class_idx, category_id in enumerate(evaluator.params.catIds):
        category = ground_truth.cats[category_id]
        per_class[str(category_id)] = {
            "name": category.get("name", str(category_id)),
            "ap": _mean_valid(precision[:, :, class_idx, 0, max_det_index]),
            "ap50": _mean_valid(precision[0, :, class_idx, 0, max_det_index]),
            "ap75": _mean_valid(precision[5, :, class_idx, 0, max_det_index]),
            "ap_small": _mean_valid(precision[:, :, class_idx, 1, max_det_index]),
            "ap_medium": _mean_valid(precision[:, :, class_idx, 2, max_det_index]),
            "ap_large": _mean_valid(precision[:, :, class_idx, 3, max_det_index]),
        }

    return {
        "map": float(evaluator.stats[0]),
        "map50": float(evaluator.stats[1]),
        "map75": float(evaluator.stats[2]),
        "per_area": per_area,
        "per_class": per_class,
    }


def print_summary(summary: Mapping[str, Any]) -> None:
    print("\nHeld-out CV summary")
    print(f"mAP50-95: {summary['map']:.4f}")
    print(f"mAP50:    {summary['map50']:.4f}")
    print(f"mAP75:    {summary['map75']:.4f}")
    print("\nArea AP")
    for area_name, score in summary["per_area"].items():
        label = "n/a" if score is None else f"{score:.4f}"
        print(f"  {area_name:>6}: {label}")
    print("\nPer-class AP")
    for category_id, row in summary["per_class"].items():
        ap = "n/a" if row["ap"] is None else f"{row['ap']:.4f}"
        ap50 = "n/a" if row["ap50"] is None else f"{row['ap50']:.4f}"
        print(f"  {int(category_id):02d} {row['name']:<22} AP={ap} AP50={ap50}")


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
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--limit", type=int, default=0, help="Limit images for a smoke test")
    parser.add_argument("--predictions-json", type=Path)
    parser.add_argument("--summary-json", type=Path)
    args = parser.parse_args()

    annotations = json.loads(args.annotations.read_text())
    if args.limit:
        image_ids = {img["id"] for img in annotations["images"][: args.limit]}
        annotations = _subset_annotations(annotations, image_ids)

    predictions = collect_predictions(
        annotations=annotations,
        images_dir=args.data_dir / "images",
        endpoint=args.endpoint,
        batch_size=args.batch_size,
        timeout=args.timeout,
    )
    summary = score_predictions(predictions, annotations)
    print_summary(summary)

    if args.predictions_json:
        args.predictions_json.parent.mkdir(parents=True, exist_ok=True)
        args.predictions_json.write_text(json.dumps(predictions), encoding="utf-8")
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
