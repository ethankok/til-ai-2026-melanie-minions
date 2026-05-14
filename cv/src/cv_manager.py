"""CV manager backed by an Ultralytics YOLO detector.

The first job is still robustness: any image/model failure returns an empty
detection list instead of taking down the request. When the model is available,
we run a YOLO detector and convert Ultralytics' xyxy boxes/class indices into
the competition's LTWH bbox/category_id schema.
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


# Ultralytics COCO class index -> TIL CV category_id.
# The Workbench annotations use a custom label space:
# 0 cargo aircraft, 1 commercial aircraft, 2 drone, 3 fighter jet,
# 4 fighter plane, 5 helicopter, 6 light aircraft, 7 missile, 8 truck,
# 9 car, 10 tank, 11 bus, 12 van, 13 cargo ship, 14 yacht,
# 15 cruise ship, 16 warship, 17 sailboat.
#
# Pretrained COCO YOLO only has broad overlapping classes. Keep this mapping
# sparse so unrelated COCO detections are ignored instead of mislabeled.
DEFAULT_TIL_CATEGORY_MAP = {
    2: 9,    # car -> car
    4: 1,    # airplane -> commercial aircraft (best single broad aircraft bucket)
    5: 11,   # bus -> bus
    7: 8,    # truck -> truck
    8: 13,   # boat -> cargo ship (best single broad ship bucket)
}


class CVManager:
    """YOLO object-detection baseline with schema-safe fallbacks."""

    def __init__(self):
        self.model_path = os.environ.get("CV_MODEL_PATH", "yolov8n.pt")
        self.conf = float(os.environ.get("CV_CONF", "0.25"))
        self.iou = float(os.environ.get("CV_IOU", "0.70"))
        self.imgsz = int(os.environ.get("CV_IMGSZ", "640"))
        self.max_det = int(os.environ.get("CV_MAX_DET", "100"))
        self.augment = os.environ.get("CV_AUGMENT", "0").lower() in ("1", "true", "yes")
        self.half = os.environ.get("CV_HALF", "1").lower() in ("1", "true", "yes")
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
                f"conf={self.conf} iou={self.iou} imgsz={self.imgsz} "
                f"max_det={self.max_det} augment={self.augment} half={self.half}",
                flush=True,
            )
        except Exception as exc:
            print(f"[CVManager] model load failed: {exc}; returning []", flush=True)

    def _load_category_map(self) -> dict[int, int]:
        """Load class-index overrides from CV_CATEGORY_MAP when provided.

        Accepts either a JSON string/path shaped like {"0": 1, "1": 2} or a
        list where the index is the YOLO class id and the value is eval id.
        Defaults to the sparse custom TIL label mapping above.
        """
        mapping = dict(DEFAULT_TIL_CATEGORY_MAP)
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
            List of detections, each with bbox [left, top, width, height] and category_id.
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
                "augment": self.augment,
                "half": self.half,
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
        confs = boxes.conf.cpu().numpy()
        for box, cls_value, conf_value in zip(xyxy, classes, confs):
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
                    "score": float(conf_value),
                }
            )
        return detections
