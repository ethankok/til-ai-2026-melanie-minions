"""Unit tests for the noise-disruption metric core (training/noise/measure_disruption.py).

These exercise the pure COCO-mAP plumbing without loading any model: perfect
predictions score ~1.0, empty predictions score 0 (a noise that suppresses every
box must read as mAP 0, not crash), and dropped detections lower the mAP.
"""

import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "training" / "noise"))

pytest.importorskip("pycocotools")

from measure_disruption import build_coco_gt, dets_to_coco, eval_map  # noqa: E402


def _toy_annotations():
    # two images, one box each (category 3), 100x100 canvas
    return {
        "info": {},
        "licenses": [],
        "categories": [{"id": 3, "name": "thing"}],
        "images": [
            {"id": 1, "width": 100, "height": 100, "file_name": "1.jpg"},
            {"id": 2, "width": 100, "height": 100, "file_name": "2.jpg"},
        ],
        "annotations": [
            {"id": 0, "image_id": 1, "category_id": 3, "bbox": [10, 10, 20, 20], "area": 400, "iscrowd": 0},
            {"id": 1, "image_id": 2, "category_id": 3, "bbox": [40, 40, 30, 30], "area": 900, "iscrowd": 0},
        ],
    }


def test_perfect_predictions_score_high():
    ann = _toy_annotations()
    gt = build_coco_gt(ann, [1, 2])
    preds = (
        dets_to_coco(1, [{"bbox": [10, 10, 20, 20], "category_id": 3, "score": 0.9}])
        + dets_to_coco(2, [{"bbox": [40, 40, 30, 30], "category_id": 3, "score": 0.9}])
    )
    res = eval_map(gt, preds)
    assert res["map50"] == pytest.approx(1.0, abs=1e-6)


def test_empty_predictions_score_zero_not_crash():
    ann = _toy_annotations()
    gt = build_coco_gt(ann, [1, 2])
    res = eval_map(gt, [])
    assert res["map"] == 0.0
    assert res["map50"] == 0.0


def test_suppressed_detection_drops_map():
    """Dropping one image's box (what disruptive noise should do) lowers mAP."""
    ann = _toy_annotations()
    gt = build_coco_gt(ann, [1, 2])
    full = (
        dets_to_coco(1, [{"bbox": [10, 10, 20, 20], "category_id": 3, "score": 0.9}])
        + dets_to_coco(2, [{"bbox": [40, 40, 30, 30], "category_id": 3, "score": 0.9}])
    )
    suppressed = dets_to_coco(1, [{"bbox": [10, 10, 20, 20], "category_id": 3, "score": 0.9}])
    assert eval_map(gt, full)["map50"] > eval_map(gt, suppressed)["map50"]


def test_dets_to_coco_preserves_ltwh_and_score():
    rec = dets_to_coco(7, [{"bbox": [1.0, 2.0, 3.0, 4.0], "category_id": 5, "score": 0.42}])
    assert rec == [{"image_id": 7, "category_id": 5, "bbox": [1.0, 2.0, 3.0, 4.0], "score": 0.42}]
