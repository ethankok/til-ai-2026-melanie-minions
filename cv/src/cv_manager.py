"""CV manager backed by an Ultralytics YOLO detector.

The first job is still robustness: any image/model failure returns an empty
detection list instead of taking down the request. When the model is available,
we run a YOLO detector and convert Ultralytics' xyxy boxes/class indices into
the competition's LTWH bbox/category_id schema.

Inference modes (gated by ``CV_TILE_MODE``):

* unset/``off``: single forward pass, identical to the prior shipped behavior.
* ``2x2`` / ``2x1`` / ``3x2``: tiled inference — split the image into a grid of
  overlapping tiles, run the detector on each, and merge results with a
  full-image pass. Designed to recover small-object recall lost when the
  detector down-samples 1920x1080 inputs to a 768/896 inference resolution.

All tiled-mode behavior is opt-in. Misconfiguration falls back to single-pass
inference rather than failing.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
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


# Tiling grids: rows x cols. Overlap is applied along both axes.
_TILE_GRIDS = {
    "off": None,
    "2x2": (2, 2),
    "2x1": (1, 2),
    "3x2": (2, 3),
}


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


class CVManager:
    """YOLO object-detection baseline with schema-safe fallbacks."""

    def __init__(self):
        self.model_path = os.environ.get("CV_MODEL_PATH", "yolov8n.pt")
        self.conf = float(os.environ.get("CV_CONF", "0.25"))
        self.iou = float(os.environ.get("CV_IOU", "0.70"))
        self.imgsz = int(os.environ.get("CV_IMGSZ", "640"))
        self.max_det = int(os.environ.get("CV_MAX_DET", "100"))
        self.augment = _env_bool("CV_AUGMENT", default=False)
        self.half = _env_bool("CV_HALF", default=True)
        self.device = os.environ.get("CV_DEVICE")

        # Tiled-inference settings (all opt-in; defaults reproduce single-pass).
        tile_raw = os.environ.get("CV_TILE_MODE", "off").strip().lower()
        if tile_raw not in _TILE_GRIDS:
            print(
                f"[CVManager] unknown CV_TILE_MODE={tile_raw!r}; falling back to off",
                flush=True,
            )
            tile_raw = "off"
        self.tile_mode = tile_raw
        self.tile_grid = _TILE_GRIDS[tile_raw]
        self.tile_imgsz = int(os.environ.get("CV_TILE_IMGSZ", "768"))
        self.tile_overlap = float(os.environ.get("CV_TILE_OVERLAP", "0.20"))
        self.tile_overlap = min(max(self.tile_overlap, 0.0), 0.6)
        self.tile_edge_margin = float(os.environ.get("CV_TILE_EDGE_MARGIN", "4"))
        self.tile_merge_iou = float(os.environ.get("CV_TILE_MERGE_IOU", "0.50"))
        self.tile_full_pass = _env_bool("CV_TILE_FULL_PASS", default=True)
        self.tile_augment = _env_bool("CV_TILE_AUGMENT", default=False)

        self.category_map = self._load_category_map()
        self.model = None
        self._inference_count = 0

        if YOLO is None:
            print("[CVManager] ultralytics unavailable; returning empty detections", flush=True)
            return

        try:
            self.model = YOLO(self.model_path)
            print(
                f"[CVManager] loaded {self.model_path} "
                f"conf={self.conf} iou={self.iou} imgsz={self.imgsz} "
                f"max_det={self.max_det} augment={self.augment} half={self.half} "
                f"tile_mode={self.tile_mode} tile_imgsz={self.tile_imgsz} "
                f"tile_overlap={self.tile_overlap} tile_full_pass={self.tile_full_pass}",
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

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def _yolo_predict(
        self,
        img: Image.Image,
        imgsz: int,
        augment: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run a single Ultralytics forward pass and return (xyxy, cls, conf).

        Returns three empty arrays on failure or zero detections.
        """
        kwargs: dict[str, Any] = {
            "source": img,
            "conf": self.conf,
            "iou": self.iou,
            "imgsz": imgsz,
            "max_det": self.max_det,
            "augment": augment,
            "half": self.half,
            "verbose": False,
        }
        if self.device:
            kwargs["device"] = self.device
        result = self.model.predict(**kwargs)[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            empty = np.zeros((0, 4), dtype=np.float32)
            return empty, np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float32)
        xyxy = boxes.xyxy.cpu().numpy().astype(np.float32, copy=False)
        cls = boxes.cls.cpu().numpy().astype(np.int64, copy=False)
        conf = boxes.conf.cpu().numpy().astype(np.float32, copy=False)
        return xyxy, cls, conf

    def _tile_offsets(self, width: int, height: int) -> list[tuple[int, int, int, int]]:
        """Compute (x1, y1, x2, y2) tile rectangles in image pixel coords."""
        if self.tile_grid is None:
            return []
        rows, cols = self.tile_grid
        # Strides chosen so that adjacent tiles overlap by ``tile_overlap``.
        denom_w = cols - (cols - 1) * self.tile_overlap
        denom_h = rows - (rows - 1) * self.tile_overlap
        tile_w = max(1, int(round(width / denom_w)))
        tile_h = max(1, int(round(height / denom_h)))
        stride_w = max(1, int(round(tile_w * (1.0 - self.tile_overlap))))
        stride_h = max(1, int(round(tile_h * (1.0 - self.tile_overlap))))

        rects: list[tuple[int, int, int, int]] = []
        for r in range(rows):
            for c in range(cols):
                x1 = c * stride_w
                y1 = r * stride_h
                # Last tile snaps to the right/bottom edge so we always cover.
                if c == cols - 1:
                    x1 = max(0, width - tile_w)
                if r == rows - 1:
                    y1 = max(0, height - tile_h)
                x2 = min(width, x1 + tile_w)
                y2 = min(height, y1 + tile_h)
                if x2 - x1 <= 0 or y2 - y1 <= 0:
                    continue
                rects.append((x1, y1, x2, y2))
        # Deduplicate identical rectangles (small images can collapse).
        seen: set[tuple[int, int, int, int]] = set()
        unique: list[tuple[int, int, int, int]] = []
        for rect in rects:
            if rect not in seen:
                seen.add(rect)
                unique.append(rect)
        return unique

    def _gather_detections(
        self, img: Image.Image
    ) -> tuple[list[list[float]], list[int], list[float]]:
        """Run the configured inference path and collect raw image-space boxes.

        Returns (xyxy_list, cls_list, conf_list). Boxes are in original image
        coordinates and may overlap; class-aware NMS is applied by the caller.
        """
        width, height = img.size
        all_xyxy: list[list[float]] = []
        all_cls: list[int] = []
        all_conf: list[float] = []

        # Tile passes: collect detections inside each tile and offset back to
        # the full image. Drop boxes that hug an internal tile edge — they are
        # almost certainly clipped objects and the full-image pass (or an
        # overlapping tile) should provide a clean detection.
        for rect in self._tile_offsets(width, height):
            x1, y1, x2, y2 = rect
            tile = img.crop(rect)
            try:
                xyxy, cls, conf = self._yolo_predict(
                    tile, imgsz=self.tile_imgsz, augment=self.tile_augment
                )
            except Exception as exc:
                print(f"[CVManager] tile {rect} failed: {exc}", flush=True)
                continue
            if xyxy.size == 0:
                continue

            tile_w = x2 - x1
            tile_h = y2 - y1
            margin = self.tile_edge_margin
            for box, cls_value, conf_value in zip(xyxy, cls, conf):
                bx1, by1, bx2, by2 = (float(v) for v in box)
                # Reject boxes touching an internal tile edge (likely truncated).
                touches_left = x1 > 0 and bx1 <= margin
                touches_right = x2 < width and bx2 >= tile_w - margin
                touches_top = y1 > 0 and by1 <= margin
                touches_bot = y2 < height and by2 >= tile_h - margin
                if touches_left or touches_right or touches_top or touches_bot:
                    continue
                gx1 = bx1 + x1
                gy1 = by1 + y1
                gx2 = bx2 + x1
                gy2 = by2 + y1
                all_xyxy.append([gx1, gy1, gx2, gy2])
                all_cls.append(int(cls_value))
                all_conf.append(float(conf_value))

        # Full-image pass: required when tiling is off; configurable when on.
        if self.tile_grid is None or self.tile_full_pass:
            try:
                xyxy, cls, conf = self._yolo_predict(
                    img, imgsz=self.imgsz, augment=self.augment
                )
            except Exception as exc:
                print(f"[CVManager] full-image inference failed: {exc}", flush=True)
                xyxy = np.zeros((0, 4), dtype=np.float32)
                cls = np.zeros((0,), dtype=np.int64)
                conf = np.zeros((0,), dtype=np.float32)
            for box, cls_value, conf_value in zip(xyxy, cls, conf):
                bx1, by1, bx2, by2 = (float(v) for v in box)
                all_xyxy.append([bx1, by1, bx2, by2])
                all_cls.append(int(cls_value))
                all_conf.append(float(conf_value))

        return all_xyxy, all_cls, all_conf

    @staticmethod
    def _class_aware_nms(
        xyxy: np.ndarray,
        cls: np.ndarray,
        conf: np.ndarray,
        iou_threshold: float,
    ) -> np.ndarray:
        """Greedy class-aware NMS. Returns sorted survivor indices."""
        if xyxy.size == 0:
            return np.zeros((0,), dtype=np.int64)
        order = conf.argsort()[::-1]
        keep: list[int] = []
        suppressed = np.zeros(len(order), dtype=bool)
        for i, idx in enumerate(order):
            if suppressed[i]:
                continue
            keep.append(int(idx))
            ax1, ay1, ax2, ay2 = xyxy[idx]
            a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
            for j in range(i + 1, len(order)):
                if suppressed[j]:
                    continue
                jdx = order[j]
                if cls[jdx] != cls[idx]:
                    continue
                bx1, by1, bx2, by2 = xyxy[jdx]
                ix1 = max(ax1, bx1)
                iy1 = max(ay1, by1)
                ix2 = min(ax2, bx2)
                iy2 = min(ay2, by2)
                iw = max(0.0, ix2 - ix1)
                ih = max(0.0, iy2 - iy1)
                inter = iw * ih
                b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
                union = a_area + b_area - inter
                if union <= 0.0:
                    continue
                if inter / union >= iou_threshold:
                    suppressed[j] = True
        return np.asarray(keep, dtype=np.int64)

    # ------------------------------------------------------------------
    # Public entrypoint
    # ------------------------------------------------------------------

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
            xyxy_list, cls_list, conf_list = self._gather_detections(img)
        except Exception as exc:
            print(f"[CVManager] inference failed key={key!r}: {exc}", flush=True)
            return []

        width, height = img.size
        if not xyxy_list:
            return []

        xyxy = np.asarray(xyxy_list, dtype=np.float32)
        cls = np.asarray(cls_list, dtype=np.int64)
        conf = np.asarray(conf_list, dtype=np.float32)

        # NMS only needed when tiling produced overlapping candidates from
        # multiple sources. Single-pass (off-mode) detections already came
        # from Ultralytics' built-in NMS, so skipping here keeps that path a
        # bit-for-bit no-op.
        if self.tile_grid is not None:
            survivors = self._class_aware_nms(xyxy, cls, conf, self.tile_merge_iou)
            xyxy = xyxy[survivors]
            cls = cls[survivors]
            conf = conf[survivors]

        # Optional one-time log so we can verify the configured path fired.
        if self._inference_count == 0:
            print(
                f"[CVManager] first inference: tile_mode={self.tile_mode} "
                f"raw={len(xyxy_list)} kept={len(xyxy)}",
                flush=True,
            )
        self._inference_count += 1

        detections: list[dict[str, Any]] = []
        for box, cls_value, conf_value in zip(xyxy, cls, conf):
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
