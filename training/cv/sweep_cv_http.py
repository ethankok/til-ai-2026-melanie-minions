#!/usr/bin/env python3
"""Run CV confidence/NMS/image-size sweeps through the Docker HTTP service."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import socket
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


def _parse_optional_ints(text: str) -> list[int | None]:
    values = _parse_ints(text)
    return values if values else [None]


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


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
    fields = [
        "conf",
        "iou",
        "imgsz",
        "max_det",
        "augment",
        "cross_class_nms_iou",
        "model_family",
        "rtdetr_eval_idx",
        "rtdetr_num_queries",
        "rfdetr_resolution",
        "map",
        "map50",
        "map75",
        "elapsed_s",
        "images_per_s",
        "est_full_s",
        "est_speed",
        "est_blended",
        "port",
        "predictions",
    ]
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
    parser.add_argument("--conf", default="0.20,0.25,0.30,0.40,0.50,0.60")
    parser.add_argument("--iou", default="0.45,0.50,0.55,0.60,0.70")
    parser.add_argument("--imgsz", default="768,896,1024")
    parser.add_argument("--max-det", default="100")
    parser.add_argument("--augment", default="0,1", help="Comma-separated 0/1 to toggle Ultralytics TTA")
    parser.add_argument(
        "--cross-class-nms-iou",
        default="0",
        help="Comma-separated high-IoU class-agnostic post-NMS thresholds; 0 disables.",
    )
    parser.add_argument(
        "--model-family",
        choices=["auto", "yolo", "rtdetr", "rfdetr"],
        default="auto",
        help="Set CV_MODEL_FAMILY for the container. Use rtdetr for RT-DETR best.pt "
        "files, rfdetr for RF-DETR .pth checkpoints.",
    )
    parser.add_argument(
        "--rfdetr-resolution",
        default="",
        help="Comma-separated RF-DETR resolutions to sweep (divisible by 56); "
        "empty uses the container default.",
    )
    parser.add_argument(
        "--rtdetr-eval-idx",
        default="",
        help="Comma-separated RT-DETR decoder eval_idx values to sweep; empty leaves unset.",
    )
    parser.add_argument(
        "--rtdetr-num-queries",
        default="",
        help="Comma-separated RT-DETR num_queries values to sweep; empty leaves unset.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--request-timeout", type=float, default=60.0)
    parser.add_argument("--startup-timeout", type=float, default=120.0)
    parser.add_argument(
        "--port",
        type=int,
        default=0,
        help="Host port for the temporary container. Use 0 to auto-pick a free port.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit images for a quick smoke sweep")
    parser.add_argument("--cpu", action="store_true", help="Run Docker without --gpus all")
    parser.add_argument(
        "--speed-images",
        type=int,
        default=500,
        help="Image count used to extrapolate challenge speed from subset sweeps",
    )
    parser.add_argument(
        "--speed-max-seconds",
        type=float,
        default=1800.0,
        help="Challenge speed denominator; qualifier uses 30 minutes",
    )
    parser.add_argument("--accuracy-weight", type=float, default=0.75)
    parser.add_argument("--speed-weight", type=float, default=0.25)
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

    # RF-DETR is NMS-free and resolution-fixed-at-load, so iou/imgsz/augment/
    # RT-DETR knobs are no-ops here; only conf and RF-DETR resolution matter.
    if args.model_family == "rfdetr":
        args.iou = "0"
        args.imgsz = "0"
        args.augment = "0"
        args.rtdetr_eval_idx = ""
        args.rtdetr_num_queries = ""

    combos = list(
        itertools.product(
            _parse_floats(args.conf),
            _parse_floats(args.iou),
            _parse_ints(args.imgsz),
            _parse_ints(args.max_det),
            _parse_ints(args.augment),
            _parse_floats(args.cross_class_nms_iou),
            _parse_optional_ints(args.rtdetr_eval_idx),
            _parse_optional_ints(args.rtdetr_num_queries),
            _parse_optional_ints(args.rfdetr_resolution),
        )
    )

    for idx, (
        conf,
        iou,
        imgsz,
        max_det,
        augment,
        cross_class_nms_iou,
        rtdetr_eval_idx,
        rtdetr_num_queries,
        rfdetr_resolution,
    ) in enumerate(combos, start=1):
        print(
            f"\n[{idx}/{len(combos)}] conf={conf} iou={iou} imgsz={imgsz} "
            f"max_det={max_det} augment={augment} family={args.model_family} "
            f"cross_nms={cross_class_nms_iou} eval_idx={rtdetr_eval_idx} "
            f"queries={rtdetr_num_queries} rfdetr_res={rfdetr_resolution}"
        )
        env = {
            "CV_MODEL_FAMILY": args.model_family,
            "CV_CONF": str(conf),
            "CV_IOU": str(iou),
            "CV_IMGSZ": str(imgsz),
            "CV_MAX_DET": str(max_det),
            "CV_AUGMENT": str(augment),
            "CV_HALF": "1",
            "CV_CROSS_CLASS_NMS_IOU": str(cross_class_nms_iou),
        }
        if rtdetr_eval_idx is not None:
            env["CV_RTDETR_EVAL_IDX"] = str(rtdetr_eval_idx)
        if rtdetr_num_queries is not None:
            env["CV_RTDETR_NUM_QUERIES"] = str(rtdetr_num_queries)
        if rfdetr_resolution is not None:
            env["CV_RFDETR_RESOLUTION"] = str(rfdetr_resolution)
        host_port = _find_free_port() if args.port == 0 else args.port
        try:
            _start_container(args.image, container_name, host_port, env, use_gpus=not args.cpu)
            _wait_for_health(host_port, args.startup_timeout)
            eval_start = time.time()
            predictions = collect_predictions(
                annotations=annotations,
                images_dir=args.data_dir / "images",
                endpoint=f"http://localhost:{host_port}/cv",
                batch_size=args.batch_size,
                timeout=args.request_timeout,
            )
            elapsed_s = time.time() - eval_start
            summary = score_predictions(predictions, annotations)
        finally:
            _stop_container(container_name)

        image_count = max(1, len(annotations["images"]))
        images_per_s = image_count / elapsed_s if elapsed_s > 0 else 0.0
        est_full_s = elapsed_s * (args.speed_images / image_count)
        est_speed = 1.0 - min(est_full_s, args.speed_max_seconds) / args.speed_max_seconds
        est_blended = args.accuracy_weight * summary["map"] + args.speed_weight * est_speed
        row = {
            "conf": conf,
            "iou": iou,
            "imgsz": imgsz,
            "max_det": max_det,
            "augment": augment,
            "cross_class_nms_iou": cross_class_nms_iou,
            "model_family": args.model_family,
            "rtdetr_eval_idx": rtdetr_eval_idx,
            "rtdetr_num_queries": rtdetr_num_queries,
            "rfdetr_resolution": rfdetr_resolution,
            "map": summary["map"],
            "map50": summary["map50"],
            "map75": summary["map75"],
            "elapsed_s": elapsed_s,
            "images_per_s": images_per_s,
            "est_full_s": est_full_s,
            "est_speed": est_speed,
            "est_blended": est_blended,
            "port": host_port,
            "predictions": len(predictions),
            "summary": summary,
        }
        rows.append(row)
        run_name = (
            f"{args.model_family}_conf{conf:.2f}_iou{iou:.2f}_img{imgsz}"
            f"_max{max_det}_aug{augment}_xnms{cross_class_nms_iou:.2f}"
        )
        if rtdetr_eval_idx is not None:
            run_name += f"_eval{rtdetr_eval_idx}"
        if rtdetr_num_queries is not None:
            run_name += f"_q{rtdetr_num_queries}"
        if rfdetr_resolution is not None:
            run_name += f"_res{rfdetr_resolution}"
        run_name = run_name.replace(".", "p")
        (args.out_dir / f"{run_name}.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        best = max(rows, key=lambda item: item["map"])
        print(
            f"mAP={summary['map']:.4f} mAP50={summary['map50']:.4f} "
            f"elapsed={elapsed_s:.1f}s est_speed={est_speed:.3f} "
            f"est_blended={est_blended:.4f} "
            f"best={best['map']:.4f} @ conf={best['conf']} iou={best['iou']} "
            f"imgsz={best['imgsz']} aug={best['augment']} "
            f"cross_nms={best['cross_class_nms_iou']} "
            f"eval_idx={best['rtdetr_eval_idx']} queries={best['rtdetr_num_queries']}"
        )

    rows.sort(key=lambda item: item["map"], reverse=True)
    (args.out_dir / "sweep_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    _write_csv(args.out_dir / "sweep_results.csv", rows)
    print("\nTop sweep results")
    for row in rows[:10]:
        print(
            f"mAP={row['map']:.4f} mAP50={row['map50']:.4f} "
            f"est_speed={row['est_speed']:.3f} est_blended={row['est_blended']:.4f} "
            f"conf={row['conf']} iou={row['iou']} imgsz={row['imgsz']} "
            f"max_det={row['max_det']} aug={row['augment']} "
            f"cross_nms={row['cross_class_nms_iou']} "
            f"family={row['model_family']} eval_idx={row['rtdetr_eval_idx']} "
            f"queries={row['rtdetr_num_queries']}"
        )
    print("\nTop estimated blended results")
    for row in sorted(rows, key=lambda item: item["est_blended"], reverse=True)[:10]:
        print(
            f"blend={row['est_blended']:.4f} mAP={row['map']:.4f} "
            f"est_speed={row['est_speed']:.3f} elapsed={row['elapsed_s']:.1f}s "
            f"conf={row['conf']} iou={row['iou']} imgsz={row['imgsz']} "
            f"max_det={row['max_det']} aug={row['augment']} "
            f"cross_nms={row['cross_class_nms_iou']} "
            f"family={row['model_family']} eval_idx={row['rtdetr_eval_idx']} "
            f"queries={row['rtdetr_num_queries']}"
        )


if __name__ == "__main__":
    main()
