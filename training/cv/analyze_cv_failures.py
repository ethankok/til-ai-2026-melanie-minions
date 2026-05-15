#!/usr/bin/env python3
"""Bucket the local mAP gap by class, area, and image.

Reads the predictions JSON emitted by ``eval_cv_http.py`` (one detection per
list element with ``image_id``, ``category_id``, ``bbox``, ``score``) plus the
matching held-out COCO annotations file, and prints:

  1. Per-class AP table (sorted by mAP-loss contribution).
  2. Per-class × per-area AP breakdown.
  3. Class confusion at IoU >= 0.5 (gold label -> predicted label).
  4. Worst-N images by per-image AP loss (for eyeballing).
  5. Score-bucket FP rate (does a higher CV_CONF help?).

Usage on the Workbench (after running eval_cv_http.py with --predictions-json):

    python training/cv/analyze_cv_failures.py \
        --predictions /home/jupyter/cv_eval_sweeps/tier1_predictions.json \
        --annotations /home/jupyter/cv_yolo_dataset/coco/annotations_test.json \
        --top-images 20 \
        --confusion-iou 0.50

Stdlib + pycocotools (already a dependency of eval_cv_http.py).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
    raise SystemExit(
        "pycocotools is required. Install it via requirements-dev.txt."
    ) from exc


class COCOPatched(COCO):
    """COCO loader that takes an in-memory annotation dict."""

    def __init__(self, annotations: dict[str, Any]):
        self.dataset, self.anns, self.cats, self.imgs = {}, {}, {}, {}
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
        self.dataset = annotations
        self.createIndex()


def _xywh_to_xyxy(box: Sequence[float]) -> tuple[float, float, float, float]:
    x, y, w, h = box
    return float(x), float(y), float(x) + float(w), float(y) + float(h)


def _iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = _xywh_to_xyxy(a)
    bx1, by1, bx2, by2 = _xywh_to_xyxy(b)
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    union = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1) + max(0.0, bx2 - bx1) * max(0.0, by2 - by1) - inter
    if union <= 0:
        return 0.0
    return inter / union


def _mean_valid(values) -> float | None:
    valid = values[values > -1]
    if valid.size == 0:
        return None
    return float(valid.mean())


def _global_summary(coco_eval: COCOeval) -> dict[str, Any]:
    precision = coco_eval.eval["precision"]  # [iou, recall, class, area, max_det]
    max_det_index = len(coco_eval.params.maxDets) - 1
    per_area = {
        name: _mean_valid(precision[:, :, :, idx, max_det_index])
        for idx, name in enumerate(coco_eval.params.areaRngLbl)
    }
    return {
        "map": float(coco_eval.stats[0]),
        "map50": float(coco_eval.stats[1]),
        "map75": float(coco_eval.stats[2]),
        "per_area": per_area,
    }


def _per_class_summary(
    coco_eval: COCOeval,
    ground_truth: COCO,
    box_counts: Mapping[int, int],
) -> list[dict[str, Any]]:
    precision = coco_eval.eval["precision"]
    max_det_index = len(coco_eval.params.maxDets) - 1
    rows: list[dict[str, Any]] = []
    for class_idx, category_id in enumerate(coco_eval.params.catIds):
        cat = ground_truth.cats[category_id]
        ap = _mean_valid(precision[:, :, class_idx, 0, max_det_index])
        ap50 = _mean_valid(precision[0, :, class_idx, 0, max_det_index])
        ap75 = _mean_valid(precision[5, :, class_idx, 0, max_det_index])
        ap_small = _mean_valid(precision[:, :, class_idx, 1, max_det_index])
        ap_medium = _mean_valid(precision[:, :, class_idx, 2, max_det_index])
        ap_large = _mean_valid(precision[:, :, class_idx, 3, max_det_index])
        boxes = box_counts.get(int(category_id), 0)
        ap_value = ap if ap is not None else 0.0
        loss = (1.0 - ap_value) * boxes
        rows.append(
            {
                "id": int(category_id),
                "name": cat.get("name", str(category_id)),
                "boxes": boxes,
                "ap": ap,
                "ap50": ap50,
                "ap75": ap75,
                "ap_small": ap_small,
                "ap_medium": ap_medium,
                "ap_large": ap_large,
                "loss_contribution": loss,
            }
        )
    rows.sort(key=lambda row: row["loss_contribution"], reverse=True)
    return rows


def _per_image_loss(
    coco_eval: COCOeval,
    ground_truth: COCO,
) -> list[dict[str, Any]]:
    precision = coco_eval.eval["precision"]
    eval_imgs = coco_eval.evalImgs
    params = coco_eval.params
    max_det_index = len(params.maxDets) - 1
    image_ids = list(params.imgIds)

    img_metric: dict[Any, list[float]] = defaultdict(list)
    n_iou = len(params.iouThrs)
    for iou_idx in range(n_iou):
        for class_idx, category_id in enumerate(params.catIds):
            for img_idx, image_id in enumerate(image_ids):
                # eval_imgs is indexed [class, area, image] flattened by COCOeval
                base = (
                    class_idx * len(params.areaRng) * len(image_ids)
                    + 0 * len(image_ids)
                    + img_idx
                )
                row = eval_imgs[base]
                if row is None:
                    continue
                gt_matches = row["gtMatches"][iou_idx]
                gt_ignore = row["gtIgnore"]
                if len(gt_matches) == 0:
                    continue
                hit = sum(1 for j, m in enumerate(gt_matches) if m and not gt_ignore[j])
                total = sum(1 for ig in gt_ignore if not ig)
                if total == 0:
                    continue
                img_metric[image_id].append(hit / total)

    rows: list[dict[str, Any]] = []
    for image_id in image_ids:
        ratios = img_metric.get(image_id, [])
        recall = sum(ratios) / len(ratios) if ratios else 0.0
        gt_count = sum(1 for ann in ground_truth.imgToAnns[image_id] if not ann.get("iscrowd", 0))
        if gt_count == 0:
            continue
        loss = (1.0 - recall) * gt_count
        img = ground_truth.imgs[image_id]
        rows.append(
            {
                "image_id": image_id,
                "file_name": img.get("file_name", str(image_id)),
                "boxes": gt_count,
                "recall": recall,
                "loss_contribution": loss,
            }
        )
    rows.sort(key=lambda row: row["loss_contribution"], reverse=True)
    return rows


def _confusion(
    annotations: dict[str, Any],
    predictions: Sequence[Mapping[str, Any]],
    iou_threshold: float,
    conf_threshold: float,
) -> tuple[Counter, Counter, Counter, int, int]:
    """Compute class confusion at the given IoU/conf thresholds.

    Returns (class_pair_counts, class_correct, class_unmatched_gt, fp, gt_total).
    """
    cat_id_to_name = {int(cat["id"]): cat["name"] for cat in annotations["categories"]}
    gt_by_image: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for ann in annotations.get("annotations", []):
        if ann.get("iscrowd", 0):
            continue
        gt_by_image[ann["image_id"]].append(
            {
                "bbox": ann["bbox"],
                "category_id": int(ann["category_id"]),
                "matched": False,
            }
        )

    pred_by_image: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for pred in predictions:
        if float(pred.get("score", 1.0)) < conf_threshold:
            continue
        pred_by_image[pred["image_id"]].append(
            {
                "bbox": pred["bbox"],
                "category_id": int(pred["category_id"]),
                "score": float(pred.get("score", 1.0)),
            }
        )

    pair_counts: Counter = Counter()
    correct: Counter = Counter()
    unmatched_gt: Counter = Counter()
    false_positives: Counter = Counter()
    gt_total = 0
    fp_total = 0

    for image_id, gts in gt_by_image.items():
        gt_total += len(gts)
        preds = sorted(
            pred_by_image.get(image_id, []), key=lambda p: p["score"], reverse=True
        )
        for pred in preds:
            best_idx = -1
            best_iou = iou_threshold
            for i, gt in enumerate(gts):
                if gt["matched"]:
                    continue
                iou = _iou(pred["bbox"], gt["bbox"])
                if iou >= best_iou:
                    best_iou = iou
                    best_idx = i
            if best_idx >= 0:
                gt = gts[best_idx]
                gt["matched"] = True
                if gt["category_id"] == pred["category_id"]:
                    correct[gt["category_id"]] += 1
                else:
                    pair_counts[(gt["category_id"], pred["category_id"])] += 1
            else:
                false_positives[pred["category_id"]] += 1
                fp_total += 1

    for image_id, gts in gt_by_image.items():
        for gt in gts:
            if not gt["matched"]:
                unmatched_gt[gt["category_id"]] += 1

    return pair_counts, correct, unmatched_gt, fp_total, gt_total


def _conf_buckets(
    annotations: dict[str, Any],
    predictions: Sequence[Mapping[str, Any]],
    iou_threshold: float,
    edges: Sequence[float],
) -> list[dict[str, Any]]:
    """For each conf bucket, count TP/FP at the given IoU threshold."""
    gt_by_image: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for ann in annotations.get("annotations", []):
        if ann.get("iscrowd", 0):
            continue
        gt_by_image[ann["image_id"]].append(
            {"bbox": ann["bbox"], "category_id": int(ann["category_id"]), "matched": False}
        )

    sorted_preds = sorted(
        ({**pred, "score": float(pred.get("score", 1.0))} for pred in predictions),
        key=lambda p: p["score"],
        reverse=True,
    )

    bucket_counts = [{"low": edges[i], "high": edges[i + 1] if i + 1 < len(edges) else 1.0001, "tp": 0, "fp": 0}
                     for i in range(len(edges))]

    def find_bucket(score: float) -> int:
        for idx, bucket in enumerate(bucket_counts):
            if score >= bucket["low"] and score < bucket["high"]:
                return idx
        return -1

    for pred in sorted_preds:
        score = pred["score"]
        bucket = find_bucket(score)
        if bucket < 0:
            continue
        gts = gt_by_image.get(pred["image_id"], [])
        best_idx = -1
        best_iou = iou_threshold
        for i, gt in enumerate(gts):
            if gt["matched"]:
                continue
            if gt["category_id"] != pred["category_id"]:
                continue
            iou = _iou(pred["bbox"], gt["bbox"])
            if iou >= best_iou:
                best_iou = iou
                best_idx = i
        if best_idx >= 0:
            gts[best_idx]["matched"] = True
            bucket_counts[bucket]["tp"] += 1
        else:
            bucket_counts[bucket]["fp"] += 1

    return bucket_counts


def print_class_table(rows: Iterable[Mapping[str, Any]]) -> None:
    print("\nPer-class AP (sorted by mAP-loss contribution = (1 - AP) * box_count)")
    header = f"  {'id':>2}  {'name':<22} {'boxes':>5} {'AP':>6} {'AP50':>6} {'AP75':>6} {'small':>6} {'medium':>6} {'large':>6} {'loss':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for row in rows:
        def fmt(value: float | None) -> str:
            return f"{value:.4f}" if isinstance(value, float) else "n/a"
        print(
            f"  {row['id']:02d}  {row['name']:<22} {row['boxes']:>5} "
            f"{fmt(row['ap']):>6} {fmt(row['ap50']):>6} {fmt(row['ap75']):>6} "
            f"{fmt(row['ap_small']):>6} {fmt(row['ap_medium']):>6} {fmt(row['ap_large']):>6} "
            f"{row['loss_contribution']:>8.2f}"
        )


def print_image_table(rows: Iterable[Mapping[str, Any]], top_n: int) -> None:
    print(f"\nWorst {top_n} images by per-image loss = (1 - mean_iou_threshold_recall) * box_count")
    print(f"  {'image_id':>10} {'file_name':<22} {'boxes':>5} {'recall':>7} {'loss':>8}")
    print("  " + "-" * 56)
    for row in list(rows)[:top_n]:
        print(
            f"  {str(row['image_id']):>10} {row['file_name']:<22} {row['boxes']:>5} "
            f"{row['recall']:>7.3f} {row['loss_contribution']:>8.2f}"
        )


def print_confusion(
    pair_counts: Counter,
    unmatched_gt: Counter,
    correct: Counter,
    fp_total: int,
    gt_total: int,
    cat_id_to_name: Mapping[int, str],
    top_n: int,
) -> None:
    print(f"\nConfusion at the chosen IoU and confidence thresholds")
    matched = sum(correct.values()) + sum(pair_counts.values())
    print(
        f"  ground-truth boxes: {gt_total}, matched: {matched} "
        f"(class-correct {sum(correct.values())}, class-wrong {sum(pair_counts.values())}), "
        f"unmatched-gt: {sum(unmatched_gt.values())}, false-positives: {fp_total}"
    )
    pairs = pair_counts.most_common(top_n)
    print(f"\nTop {top_n} class-confusion pairs (gold -> predicted)")
    for (gold, pred), count in pairs:
        gold_name = cat_id_to_name.get(gold, str(gold))
        pred_name = cat_id_to_name.get(pred, str(pred))
        print(f"  {gold_name:<22} -> {pred_name:<22} count={count}")
    print(f"\nTop unmatched gold classes (FN by class)")
    for cls, count in unmatched_gt.most_common(top_n):
        name = cat_id_to_name.get(cls, str(cls))
        print(f"  {name:<22} unmatched={count}")


def print_conf_buckets(buckets: Iterable[Mapping[str, Any]]) -> None:
    print("\nFP/TP by score bucket (matched at the chosen IoU threshold)")
    header = f"  {'score':>14} {'tp':>6} {'fp':>6} {'precision':>10}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for bucket in buckets:
        total = bucket["tp"] + bucket["fp"]
        prec = bucket["tp"] / total if total else 0.0
        label = f"{bucket['low']:.2f}-{bucket['high']:.2f}"
        print(f"  {label:>14} {bucket['tp']:>6} {bucket['fp']:>6} {prec:>10.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("/home/jupyter/cv_yolo_dataset/coco/annotations_test.json"),
    )
    parser.add_argument("--top-images", type=int, default=20)
    parser.add_argument("--top-pairs", type=int, default=15)
    parser.add_argument("--confusion-iou", type=float, default=0.5)
    parser.add_argument("--confusion-conf", type=float, default=0.20)
    parser.add_argument(
        "--score-buckets",
        default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9",
        help="Comma-separated lower edges; the final bucket extends to 1.0.",
    )
    parser.add_argument("--summary-json", type=Path)
    args = parser.parse_args()

    annotations = json.loads(args.annotations.read_text())
    predictions = json.loads(args.predictions.read_text())
    if not predictions:
        raise SystemExit("No predictions in JSON; nothing to analyze.")

    ground_truth = COCOPatched(annotations)
    detections = ground_truth.loadRes(list(predictions))
    coco_eval = COCOeval(ground_truth, detections, "bbox")
    silent = StringIO()
    with redirect_stdout(silent):
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

    box_counts = Counter()
    for ann in annotations.get("annotations", []):
        if ann.get("iscrowd", 0):
            continue
        box_counts[int(ann["category_id"])] += 1

    summary = _global_summary(coco_eval)
    print("Global")
    print(f"  mAP50-95: {summary['map']:.4f}")
    print(f"  mAP50:    {summary['map50']:.4f}")
    print(f"  mAP75:    {summary['map75']:.4f}")
    print("  area AP:")
    for area_name, value in summary["per_area"].items():
        label = "n/a" if value is None else f"{value:.4f}"
        print(f"    {area_name:>6}: {label}")

    class_rows = _per_class_summary(coco_eval, ground_truth, box_counts)
    print_class_table(class_rows)

    image_rows = _per_image_loss(coco_eval, ground_truth)
    print_image_table(image_rows, args.top_images)

    cat_id_to_name = {int(cat["id"]): cat["name"] for cat in annotations["categories"]}
    pair_counts, correct, unmatched_gt, fp_total, gt_total = _confusion(
        annotations,
        predictions,
        iou_threshold=args.confusion_iou,
        conf_threshold=args.confusion_conf,
    )
    print(
        f"\n(Confusion uses IoU >= {args.confusion_iou}, conf >= {args.confusion_conf}, "
        f"greedy class-agnostic matching by score.)"
    )
    print_confusion(
        pair_counts, unmatched_gt, correct, fp_total, gt_total, cat_id_to_name, args.top_pairs
    )

    edges = [float(x) for x in args.score_buckets.split(",") if x.strip()]
    bucket_rows = _conf_buckets(annotations, predictions, iou_threshold=args.confusion_iou, edges=edges)
    print_conf_buckets(bucket_rows)

    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(
            json.dumps(
                {
                    "global": summary,
                    "per_class": class_rows,
                    "worst_images": image_rows[: args.top_images],
                    "confusion_pairs": [
                        {
                            "gold": cat_id_to_name.get(gold, str(gold)),
                            "predicted": cat_id_to_name.get(pred, str(pred)),
                            "count": count,
                        }
                        for (gold, pred), count in pair_counts.most_common(args.top_pairs)
                    ],
                    "unmatched_gt": [
                        {"name": cat_id_to_name.get(cls, str(cls)), "count": count}
                        for cls, count in unmatched_gt.most_common()
                    ],
                    "score_buckets": bucket_rows,
                    "fp_total": fp_total,
                    "gt_total": gt_total,
                },
                indent=2,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
