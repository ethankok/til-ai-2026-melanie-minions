#!/usr/bin/env python3
"""Run CV confidence/NMS/image-size sweeps through the Docker HTTP service."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import requests

from eval_cv_http import collect_predictions, score_predictions


def _parse_floats(text: str) -> list[float]:
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def _parse_ints(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def _start_container(
    image: str,
    name: str,
    port: int,
    env: dict[str, str],
    use_gpus: bool,
) -> None:
    subprocess.run(["docker", "rm", "-f", name], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    cmd = ["docker", "run", "-d", "--rm", "--name", name, "-p", f"{port}:5002"]
    if use_gpus:
        cmd.extend(["--gpus", "all"])
    for key, value in env.items():
        cmd.extend(["-e", f"{key}={value}"])
    cmd.append(image)
    subprocess.run(cmd, check=True)


def _stop_container(name: str) -> None:
    subprocess.run(["docker", "rm", "-f", name], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _wait_for_health(port: int, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    health_url = f"http://localhost:{port}/health"
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            response = requests.get(health_url, timeout=2.0)
            if response.ok:
                return
        except requests.RequestException as exc:
            last_error = exc
        time.sleep(1.0)
    raise TimeoutError(f"CV container did not become healthy at {health_url}: {last_error}")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["conf", "iou", "imgsz", "max_det", "map", "map50", "map75", "predictions"]
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Docker image built by `til build cv <tag>`")
    parser.add_argument("--data-dir", type=Path, default=Path("/home/jupyter/novice/cv"))
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("/home/jupyter/cv_yolo_dataset/coco/annotations_test.json"),
    )
    parser.add_argument("--out-dir", type=Path, default=Path("/home/jupyter/cv_eval_sweeps"))
    parser.add_argument("--conf", default="0.05,0.10,0.15,0.20,0.25")
    parser.add_argument("--iou", default="0.50,0.60,0.70")
    parser.add_argument("--imgsz", default="640,768")
    parser.add_argument("--max-det", default="100")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--request-timeout", type=float, default=60.0)
    parser.add_argument("--startup-timeout", type=float, default=120.0)
    parser.add_argument("--port", type=int, default=5002)
    parser.add_argument("--limit", type=int, default=0, help="Limit images for a quick smoke sweep")
    parser.add_argument("--cpu", action="store_true", help="Run Docker without --gpus all")
    args = parser.parse_args()

    annotations = json.loads(args.annotations.read_text())
    if args.limit:
        keep_ids = {img["id"] for img in annotations["images"][: args.limit]}
        annotations = {
            **annotations,
            "images": [img for img in annotations["images"] if img["id"] in keep_ids],
            "annotations": [
                ann for ann in annotations.get("annotations", []) if ann.get("image_id") in keep_ids
            ],
        }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    container_name = f"til-cv-sweep-{os.getpid()}"
    combos = list(
        itertools.product(
            _parse_floats(args.conf),
            _parse_floats(args.iou),
            _parse_ints(args.imgsz),
            _parse_ints(args.max_det),
        )
    )

    for idx, (conf, iou, imgsz, max_det) in enumerate(combos, start=1):
        print(f"\n[{idx}/{len(combos)}] conf={conf} iou={iou} imgsz={imgsz} max_det={max_det}")
        env = {
            "CV_CONF": str(conf),
            "CV_IOU": str(iou),
            "CV_IMGSZ": str(imgsz),
            "CV_MAX_DET": str(max_det),
        }
        try:
            _start_container(args.image, container_name, args.port, env, use_gpus=not args.cpu)
            _wait_for_health(args.port, args.startup_timeout)
            predictions = collect_predictions(
                annotations=annotations,
                images_dir=args.data_dir / "images",
                endpoint=f"http://localhost:{args.port}/cv",
                batch_size=args.batch_size,
                timeout=args.request_timeout,
            )
            summary = score_predictions(predictions, annotations)
        finally:
            _stop_container(container_name)

        row = {
            "conf": conf,
            "iou": iou,
            "imgsz": imgsz,
            "max_det": max_det,
            "map": summary["map"],
            "map50": summary["map50"],
            "map75": summary["map75"],
            "predictions": len(predictions),
            "summary": summary,
        }
        rows.append(row)
        run_name = f"conf{conf:.2f}_iou{iou:.2f}_img{imgsz}_max{max_det}".replace(".", "p")
        (args.out_dir / f"{run_name}.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        best = max(rows, key=lambda item: item["map"])
        print(
            f"mAP={summary['map']:.4f} mAP50={summary['map50']:.4f} "
            f"best={best['map']:.4f} @ conf={best['conf']} iou={best['iou']} imgsz={best['imgsz']}"
        )

    rows.sort(key=lambda item: item["map"], reverse=True)
    (args.out_dir / "sweep_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    _write_csv(args.out_dir / "sweep_results.csv", rows)
    print("\nTop sweep results")
    for row in rows[:10]:
        print(
            f"mAP={row['map']:.4f} mAP50={row['map50']:.4f} "
            f"conf={row['conf']} iou={row['iou']} imgsz={row['imgsz']} max_det={row['max_det']}"
        )


if __name__ == "__main__":
    main()
