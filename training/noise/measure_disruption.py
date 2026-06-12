"""Offline noise-disruption harness: does our Finals noise actually drop a
detector's mAP?

Background (TIL25 cross-pollination scan, 8 Jun 2026): every TIL25 finalist
shipped an object *detector*, but our AdvGAN noise generator is trained against
ResNet18 *classification* on Imagenette. We have only ever verified the noise
passes the fairness gate (L2/SSIM bounds) -- never that it lowers any detector's
accuracy. This script closes that hole using our own champion YOLO11l as a free
in-house surrogate victim: it runs a set of real CV images through the deployed
``CVManager`` twice -- clean, and after ``NoiseManager.noise()`` -- and reports
the COCO mAP drop (the disruption signal) plus detection-count / confidence drop.

Everything runs on CPU and needs no Workbench: the AdvGAN generator
(``noise/src/advgan_generator.pth``) and the COCO-format CV data ship/stage
locally. Point ``--weights`` at the YOLO11l checkpoint.

Example::

    .venv/bin/python training/noise/measure_disruption.py \\
        --weights /tmp/cvw/cv_next_iter_bundle/yolo11l-896-plusval-best.pt \\
        --images-dir data/novice/cv/images \\
        --annotations data/novice/cv/annotations.json \\
        --num 150 --strength 1.0 --imgsz 896 \\
        --out training/noise/data/disruption-strength1.json

Sweep ``--strength`` (NOISE_STRESS_STRENGTH) to find the mAP-drop vs fairness
trade-off; re-validate any strength bump against the fairness gate separately.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import io
import json
import os
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Metric core (pure, unit-testable) -- COCO mAP over an image subset.
# ---------------------------------------------------------------------------
def build_coco_gt(annotations: dict, image_ids: list[int]):
    """Return a pycocotools COCO object holding only ``image_ids``."""
    from pycocotools.coco import COCO

    keep = set(image_ids)
    # COCOeval uses GT annotation id 0 as its "unmatched" sentinel in
    # dtMatches, so an annotation with id == 0 (annotations.json numbers
    # from 0) is silently scored as a false positive -- reassign 1-based ids.
    anns = []
    for new_id, a in enumerate(
        (a for a in annotations["annotations"] if a["image_id"] in keep), start=1
    ):
        anns.append({**a, "id": new_id})
    subset = {
        "info": annotations.get("info", {}),
        "licenses": annotations.get("licenses", []),
        "categories": annotations["categories"],
        "images": [im for im in annotations["images"] if im["id"] in keep],
        "annotations": anns,
    }
    coco = COCO()
    coco.dataset = subset
    with contextlib.redirect_stdout(io.StringIO()):
        coco.createIndex()
    return coco


def dets_to_coco(image_id: int, dets: list[dict]) -> list[dict]:
    """Convert ``CVManager.cv()`` output to COCO detection records.

    cv() emits ``{"bbox": [l, t, w, h], "category_id": int, "score": float}`` --
    already the COCO ``bbox`` (LTWH) convention COCOeval expects.
    """
    out = []
    for d in dets:
        out.append(
            {
                "image_id": int(image_id),
                "category_id": int(d["category_id"]),
                "bbox": [float(v) for v in d["bbox"]],
                "score": float(d.get("score", 1.0)),
            }
        )
    return out


def eval_map(coco_gt, preds: list[dict]) -> dict:
    """Return COCO mAP stats for ``preds`` against ``coco_gt``.

    Empty predictions score 0 (COCOeval would otherwise error on an empty
    result), so a noise that suppresses every box reads as mAP 0, not a crash.
    """
    from pycocotools.cocoeval import COCOeval

    if not preds:
        return {"map": 0.0, "map50": 0.0, "map75": 0.0}
    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(preds)
        ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    s = ev.stats
    return {"map": float(s[0]), "map50": float(s[1]), "map75": float(s[2])}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def _select_image_ids(annotations: dict, images_dir: Path, num: int, seed: int) -> list[int]:
    """Annotated images that exist on disk, sampled deterministically."""
    have = {p.name for p in images_dir.iterdir()}
    annotated = {a["image_id"] for a in annotations["annotations"]}
    usable = [
        im["id"]
        for im in annotations["images"]
        if im["id"] in annotated and im["file_name"] in have
    ]
    usable.sort()
    rng = random.Random(seed)
    rng.shuffle(usable)
    return usable[:num]


def run(args: argparse.Namespace) -> dict:
    sys.path.insert(0, str(REPO_ROOT / "cv" / "src"))
    sys.path.insert(0, str(REPO_ROOT / "noise" / "src"))

    # Force a deterministic CPU path for both models.
    os.environ.setdefault("CV_DEVICE", "cpu")
    os.environ["CV_HALF"] = "0"  # no FP16 on CPU
    os.environ["CV_MODEL_PATH"] = str(Path(args.weights).resolve())
    os.environ["CV_IMGSZ"] = str(args.imgsz)
    os.environ["NOISE_STRESS_STRENGTH"] = str(args.strength)
    # Replicate the deployed CV env (cv/Dockerfile): our champion YOLO's class
    # indices ARE the 18 TIL categories in order, so override the manager's
    # generic-COCO default with the identity map -- without this every box
    # gets the wrong category_id and mAP collapses to 0. (TTA/CV_AUGMENT left
    # off for CPU speed; doesn't affect relative clean-vs-noised disruption.)
    os.environ.setdefault("CV_CATEGORY_MAP", json.dumps(list(range(18))))
    os.environ.setdefault("CV_CONF", "0.20")
    os.environ.setdefault("CV_IOU", "0.55")

    from cv_manager import CVManager
    from noise_manager import NoiseManager

    annotations = json.loads(Path(args.annotations).read_text())
    images_dir = Path(args.images_dir)
    image_ids = _select_image_ids(annotations, images_dir, args.num, args.seed)
    if not image_ids:
        raise SystemExit("no annotated images found on disk under --images-dir")

    id_to_file = {im["id"]: im["file_name"] for im in annotations["images"]}

    cv = CVManager()
    nz = NoiseManager()
    if not nz.generator_loaded:
        print("[warn] AdvGAN generator not loaded -- measuring stress-only noise", flush=True)

    clean_preds: list[dict] = []
    noised_preds: list[dict] = []
    n_clean = n_noised = 0
    conf_clean = conf_noised = 0.0

    for i, img_id in enumerate(image_ids):
        raw = (images_dir / id_to_file[img_id]).read_bytes()
        clean_dets = cv.cv(raw)
        noised_b64 = nz.noise(raw)
        noised_dets = cv.cv(base64.b64decode(noised_b64))

        clean_preds += dets_to_coco(img_id, clean_dets)
        noised_preds += dets_to_coco(img_id, noised_dets)
        n_clean += len(clean_dets)
        n_noised += len(noised_dets)
        conf_clean += sum(d["score"] for d in clean_dets)
        conf_noised += sum(d["score"] for d in noised_dets)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(image_ids)} images", flush=True)

    coco_gt = build_coco_gt(annotations, image_ids)
    map_clean = eval_map(coco_gt, clean_preds)
    map_noised = eval_map(coco_gt, noised_preds)

    drop = map_clean["map"] - map_noised["map"]
    drop50 = map_clean["map50"] - map_noised["map50"]
    report = {
        "weights": os.environ["CV_MODEL_PATH"],
        "n_images": len(image_ids),
        "strength": args.strength,
        "imgsz": args.imgsz,
        "map_clean": map_clean,
        "map_noised": map_noised,
        "map_drop": round(drop, 4),
        "map_drop_pct": round(100.0 * drop / map_clean["map"], 1) if map_clean["map"] else None,
        "map50_drop": round(drop50, 4),
        "dets_clean": n_clean,
        "dets_noised": n_noised,
        "det_count_drop_pct": round(100.0 * (n_clean - n_noised) / n_clean, 1) if n_clean else None,
        "mean_conf_clean": round(conf_clean / n_clean, 4) if n_clean else 0.0,
        "mean_conf_noised": round(conf_noised / n_noised, 4) if n_noised else 0.0,
    }
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", required=True, help="YOLO11l (or any CV) checkpoint path")
    p.add_argument("--images-dir", default=str(REPO_ROOT / "data/novice/cv/images"))
    p.add_argument("--annotations", default=str(REPO_ROOT / "data/novice/cv/annotations.json"))
    p.add_argument("--num", type=int, default=150)
    p.add_argument("--strength", type=float, default=1.0, help="NOISE_STRESS_STRENGTH")
    p.add_argument("--imgsz", type=int, default=896)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    report = run(args)
    print("\n=== NOISE DISRUPTION ===")
    print(json.dumps(report, indent=2))
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
        print(f"\nsummary -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
