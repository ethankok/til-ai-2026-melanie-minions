"""CV manager backed by a small Ultralytics YOLO detector.

The first job is still robustness: any image/model failure returns an empty
detection list instead of taking down the request. When the model is available,
we run a pretrained COCO detector and convert Ultralytics' xyxy boxes/class
indices into the competition's COCO-style xywh/category_id schema.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - keeps importable without optional deps
    YOLO = None


# Ultralytics COCO class index -> official COCO category_id.
# COCO category IDs skip several integers, so returning raw YOLO class indices
# would silently produce the wrong labels.
YOLO_TO_COCO_CATEGORY_ID = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    27,
    28,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
    57,
    58,
    59,
    60,
    61,
    62,
    63,
    64,
    65,
    67,
    70,
    72,
    73,
    74,
    75,
    76,
    77,
    78,
    79,
    80,
    81,
    82,
    84,
    85,
    86,
    87,
    88,
    89,
    90,
]


class CVManager:
    """YOLO object-detection baseline with schema-safe fallbacks."""

    def __init__(self):
        self.model_path = os.environ.get("CV_MODEL_PATH", "yolov8n.pt")
        self.conf = float(os.environ.get("CV_CONF", "0.25"))
        self.iou = float(os.environ.get("CV_IOU", "0.70"))
        self.imgsz = int(os.environ.get("CV_IMGSZ", "640"))
        self.max_det = int(os.environ.get("CV_MAX_DET", "100"))
        self.device = os.environ.get("CV_DEVICE")
        self.category_map = self._load_category_map()
        self.model = None

        if YOLO is None:
            print("[CVManager] ultralytics unavailable; returning empty detections", flush=True)
            return

        try:
            self.model = YOLO(self.model_path)
            print(
                f"[CVManager] loaded {self.model_path} "
                f"conf={self.conf} iou={self.iou} imgsz={self.imgsz}",
                flush=True,
            )
        except Exception as exc:
            print(f"[CVManager] model load failed: {exc}; returning []", flush=True)

    def _load_category_map(self) -> dict[int, int]:
        """Load class-index overrides from CV_CATEGORY_MAP when provided.

        Accepts either a JSON string/path shaped like {"0": 1, "1": 2} or a
        list where the index is the YOLO class id and the value is eval id.
        Defaults to standard COCO IDs.
        """
        mapping = {idx: cat_id for idx, cat_id in enumerate(YOLO_TO_COCO_CATEGORY_ID)}
        raw = os.environ.get("CV_CATEGORY_MAP")
        if not raw:
            return mapping

        try:
            text = Path(raw).read_text(encoding="utf-8") if Path(raw).exists() else raw
            loaded = json.loads(text)
            if isinstance(loaded, list):
                return {idx: int(cat_id) for idx, cat_id in enumerate(loaded)}
            if isinstance(loaded, dict):
                return {int(cls_id): int(cat_id) for cls_id, cat_id in loaded.items()}
        except Exception as exc:
            print(f"[CVManager] could not parse CV_CATEGORY_MAP: {exc}", flush=True)
        return mapping

    def cv(self, image: bytes, key: Any | None = None) -> list[dict[str, Any]]:
        """Detect objects in one JPEG image.

        Args:
            image: JPEG bytes.
            key: Optional evaluator key, used only for debug logging.

        Returns:
            List of detections, each with bbox [x, y, w, h] and category_id.
            Empty list means "no objects detected" and is explicitly valid.
        """
        try:
            img = Image.open(io.BytesIO(image))
            img = ImageOps.exif_transpose(img).convert("RGB")
        except Exception as exc:
            print(
                f"[CVManager] decode failed key={key!r} bytes={len(image)}: {exc}",
                flush=True,
            )
            return []

        if self.model is None:
            return []

        try:
            kwargs: dict[str, Any] = {
                "source": img,
                "conf": self.conf,
                "iou": self.iou,
                "imgsz": self.imgsz,
                "max_det": self.max_det,
                "verbose": False,
            }
            if self.device:
                kwargs["device"] = self.device
            result = self.model.predict(**kwargs)[0]
        except Exception as exc:
            print(f"[CVManager] inference failed key={key!r}: {exc}", flush=True)
            return []

        width, height = img.size
        detections: list[dict[str, Any]] = []
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return detections

        xyxy = boxes.xyxy.cpu().numpy()
        classes = boxes.cls.cpu().numpy()
        for box, cls_value in zip(xyxy, classes):
            cls_idx = int(cls_value)
            category_id = self.category_map.get(cls_idx)
            if category_id is None:
                continue

            x1, y1, x2, y2 = (float(v) for v in box)
            x1 = max(0.0, min(x1, float(width)))
            y1 = max(0.0, min(y1, float(height)))
            x2 = max(0.0, min(x2, float(width)))
            y2 = max(0.0, min(y2, float(height)))
            w = x2 - x1
            h = y2 - y1
            if w <= 0.0 or h <= 0.0:
                continue

            detections.append(
                {
                    "bbox": [round(x1, 2), round(y1, 2), round(w, 2), round(h, 2)],
                    "category_id": int(category_id),
                }
            )
        return detections
