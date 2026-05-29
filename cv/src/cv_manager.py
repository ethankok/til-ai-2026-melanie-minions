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

try:
    from ultralytics import RTDETR
except Exception:  # pragma: no cover - older ultralytics / optional deps
    RTDETR = None

try:
    from rfdetr import RFDETRBase, RFDETRLarge
except Exception:  # pragma: no cover - rfdetr is an optional, container-only dep
    RFDETRBase = None
    RFDETRLarge = None

try:
    import torch
except Exception:  # pragma: no cover - optional for non-OWLv2 paths
    torch = None

try:
    from transformers import Owlv2ForObjectDetection, Owlv2Processor
except Exception:  # pragma: no cover - optional for YOLO/RT-DETR paths
    Owlv2ForObjectDetection = None
    Owlv2Processor = None


TIL_CATEGORY_NAMES = [
    "cargo aircraft",
    "commercial aircraft",
    "drone",
    "fighter jet",
    "fighter plane",
    "helicopter",
    "light aircraft",
    "missile",
    "truck",
    "car",
    "tank",
    "bus",
    "van",
    "cargo ship",
    "yacht",
    "cruise ship",
    "warship",
    "sailboat",
]


DEFAULT_OWLV2_PROMPTS: list[tuple[int, str]] = [
    (0, "a photo of a cargo aircraft"),
    (1, "a photo of a commercial aircraft"),
    (2, "a photo of a drone"),
    (3, "a photo of a fighter jet"),
    (4, "a photo of a fighter plane"),
    (5, "a photo of a helicopter"),
    (6, "a photo of a light aircraft"),
    (7, "a photo of a missile"),
    (8, "a photo of a truck"),
    (9, "a photo of a car"),
    (10, "a photo of a military tank"),
    (11, "a photo of a bus"),
    (12, "a photo of a van"),
    (13, "a photo of a cargo ship"),
    (14, "a photo of a yacht"),
    (15, "a photo of a cruise ship"),
    (16, "a photo of a warship"),
    (17, "a photo of a sailboat"),
]


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


def _env_optional_int(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    try:
        return int(raw)
    except ValueError:
        print(f"[CVManager] ignoring invalid {name}={raw!r}; expected int", flush=True)
        return None


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        print(f"[CVManager] ignoring invalid {name}={raw!r}; expected int", flush=True)
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        print(f"[CVManager] ignoring invalid {name}={raw!r}; expected float", flush=True)
        return default


class CVManager:
    """YOLO object-detection baseline with schema-safe fallbacks."""

    def __init__(self):
        self.model_path = os.environ.get("CV_MODEL_PATH", "yolov8n.pt")
        self.model_family = os.environ.get("CV_MODEL_FAMILY", "auto").strip().lower()
        if self.model_family not in ("auto", "yolo", "rtdetr", "owlv2", "rfdetr"):
            print(
                f"[CVManager] unknown CV_MODEL_FAMILY={self.model_family!r}; using auto",
                flush=True,
            )
            self.model_family = "auto"
        self.resolved_model_family = self._resolve_model_family(self.model_path)
        self.conf = float(os.environ.get("CV_CONF", "0.25"))
        self.iou = float(os.environ.get("CV_IOU", "0.70"))
        self.imgsz = int(os.environ.get("CV_IMGSZ", "640"))
        self.max_det = int(os.environ.get("CV_MAX_DET", "100"))
        self.augment = _env_bool("CV_AUGMENT", default=False)
        self.half = _env_bool("CV_HALF", default=True)
        self.device = os.environ.get("CV_DEVICE")
        self.rtdetr_eval_idx = _env_optional_int("CV_RTDETR_EVAL_IDX")
        self.rtdetr_num_queries = _env_optional_int("CV_RTDETR_NUM_QUERIES")

        # RF-DETR settings. CV_CONF is reused as the predict threshold; CV_IOU is
        # ignored because RF-DETR is NMS-free (DETR set prediction). Resolution
        # must be divisible by 56 and is snapped to the nearest valid multiple.
        rfdetr_variant = os.environ.get("CV_RFDETR_VARIANT", "base").strip().lower()
        if rfdetr_variant not in ("base", "large"):
            print(
                f"[CVManager] unknown CV_RFDETR_VARIANT={rfdetr_variant!r}; using base",
                flush=True,
            )
            rfdetr_variant = "base"
        self.rfdetr_variant = rfdetr_variant
        rfdetr_res = _env_int("CV_RFDETR_RESOLUTION", 728)
        snapped = max(56, int(round(rfdetr_res / 56)) * 56)
        if snapped != rfdetr_res:
            print(
                f"[CVManager] snapping CV_RFDETR_RESOLUTION {rfdetr_res} -> {snapped} "
                "(must be divisible by 56)",
                flush=True,
            )
        self.rfdetr_resolution = snapped
        self.cross_class_nms_iou = float(os.environ.get("CV_CROSS_CLASS_NMS_IOU", "0"))
        if self.cross_class_nms_iou < 0.0:
            self.cross_class_nms_iou = 0.0
        if self.cross_class_nms_iou > 1.0:
            self.cross_class_nms_iou = 1.0

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

        # Accuracy-first rescue pass. The second pass is deliberately
        # down-weighted so it can add high-IoU alternates without polluting the
        # top of the precision-recall curve.
        self.second_pass = _env_bool("CV_SECOND_PASS", default=False)
        self.second_conf = _env_float("CV_SECOND_CONF", self.conf)
        self.second_iou = _env_float("CV_SECOND_IOU", self.iou)
        self.second_imgsz = _env_int("CV_SECOND_IMGSZ", self.imgsz)
        self.second_augment = _env_bool("CV_SECOND_AUGMENT", default=True)
        self.second_score_scale = _env_float("CV_SECOND_SCORE_SCALE", 0.20)
        self.second_merge_iou = _env_float("CV_SECOND_MERGE_IOU", 0.95)
        self.second_min_detections = max(0, _env_int("CV_SECOND_MIN_DETECTIONS", 0))

        # OWLv2 zero-shot settings. CV_CONF is reused as the text-conditioned
        # detection threshold so sweeps can compare YOLO/OWLv2 with one knob.
        self.owlv2_model_id = os.environ.get("CV_OWLV2_MODEL_ID")
        if not self.owlv2_model_id:
            self.owlv2_model_id = "google/owlv2-base-patch16-ensemble"
            model_path_lower = self.model_path.lower()
            if (
                self.resolved_model_family == "owlv2"
                and (self.model_family == "auto" or not model_path_lower.endswith(".pt"))
            ):
                self.owlv2_model_id = self.model_path
        self.owlv2_nms_iou = min(max(_env_float("CV_OWLV2_NMS_IOU", 0.50), 0.0), 1.0)
        self.owlv2_local_files_only = _env_bool(
            "CV_OWLV2_LOCAL_FILES_ONLY",
            default=_env_bool("TRANSFORMERS_OFFLINE", default=False),
        )
        self.owlv2_prompt_specs = self._load_owlv2_prompts()
        self.owlv2_prompts = [prompt for _, prompt in self.owlv2_prompt_specs]
        self.owlv2_prompt_categories = [
            category_id for category_id, _ in self.owlv2_prompt_specs
        ]
        self.owlv2_prompt_to_category = {
            prompt: category_id for category_id, prompt in self.owlv2_prompt_specs
        }
        self.owlv2_loaded_model_ref = self.owlv2_model_id
        self.owlv2_device = None
        self.owlv2_dtype = None
        self.processor = None

        self.category_map = self._load_category_map()
        self.model = None
        self.loaded_model_family = "none"
        self._inference_count = 0
        self._last_second_pass_used = False

        if (
            self.resolved_model_family not in ("owlv2", "rfdetr")
            and YOLO is None
            and RTDETR is None
        ):
            print("[CVManager] ultralytics unavailable; returning empty detections", flush=True)
            return
        if self.resolved_model_family == "rfdetr" and (
            RFDETRBase is None or RFDETRLarge is None
        ):
            print(
                "[CVManager] RF-DETR requested but the rfdetr package is unavailable; "
                "returning empty detections",
                flush=True,
            )
            return
        if self.resolved_model_family == "owlv2" and (
            torch is None or Owlv2ForObjectDetection is None or Owlv2Processor is None
        ):
            print(
                "[CVManager] OWLv2 requested but torch/transformers are unavailable; "
                "returning empty detections",
                flush=True,
            )
            return

        try:
            self.model = self._load_model(self.model_path)
            model_ref = (
                self.owlv2_loaded_model_ref
                if self.loaded_model_family == "owlv2"
                else self.model_path
            )
            print(
                f"[CVManager] loaded {model_ref} family={self.loaded_model_family} "
                f"conf={self.conf} iou={self.iou} imgsz={self.imgsz} "
                f"max_det={self.max_det} augment={self.augment} half={self.half} "
                f"cross_class_nms_iou={self.cross_class_nms_iou} "
                f"tile_mode={self.tile_mode} tile_imgsz={self.tile_imgsz} "
                f"tile_overlap={self.tile_overlap} tile_full_pass={self.tile_full_pass} "
                f"second_pass={self.second_pass} second_conf={self.second_conf} "
                f"second_iou={self.second_iou} second_imgsz={self.second_imgsz} "
                f"second_augment={self.second_augment} "
                f"second_score_scale={self.second_score_scale} "
                f"second_merge_iou={self.second_merge_iou} "
                f"second_min_detections={self.second_min_detections} "
                f"owlv2_prompts={len(self.owlv2_prompts)} "
                f"owlv2_nms_iou={self.owlv2_nms_iou} "
                f"owlv2_local_files_only={self.owlv2_local_files_only}",
                flush=True,
            )
        except Exception as exc:
            print(f"[CVManager] model load failed: {exc}; returning []", flush=True)

    def _resolve_model_family(self, model_path: str) -> str:
        if self.model_family != "auto":
            return self.model_family
        name = Path(model_path).name.lower()
        lower_path = model_path.lower()
        if "owlv2" in lower_path or "owl-v2" in lower_path:
            return "owlv2"
        if "rfdetr" in name or "rf-detr" in name:
            return "rfdetr"
        if "rtdetr" in name or "rt-detr" in name:
            return "rtdetr"
        return "yolo"

    def _load_model(self, model_path: str):
        """Load an Ultralytics detector family while preserving one output adapter."""
        family = self.resolved_model_family
        if family == "owlv2":
            return self._load_owlv2_model()
        if family == "rfdetr":
            return self._load_rfdetr_model(model_path)
        if family == "rtdetr":
            if RTDETR is None:
                raise RuntimeError("RT-DETR requested but ultralytics.RTDETR is unavailable")
            model = RTDETR(model_path)
            self._apply_rtdetr_inference_knobs(model)
        else:
            if YOLO is None:
                raise RuntimeError("YOLO requested but ultralytics.YOLO is unavailable")
            try:
                model = YOLO(model_path)
            except Exception as yolo_exc:
                if self.model_family == "auto" and RTDETR is not None:
                    try:
                        model = RTDETR(model_path)
                    except Exception as rtdetr_exc:
                        raise RuntimeError(
                            "auto model load failed as both YOLO "
                            f"({yolo_exc}) and RT-DETR ({rtdetr_exc})"
                        ) from rtdetr_exc
                    self._apply_rtdetr_inference_knobs(model)
                    self.loaded_model_family = "rtdetr"
                    return model
                raise
        self.loaded_model_family = family
        return model

    def _resolve_owlv2_model_ref(self) -> str:
        """Prefer copied Docker model files, otherwise use the HF id/cache ref."""
        explicit_path = Path(self.owlv2_model_id)
        if explicit_path.exists():
            return self.owlv2_model_id

        model_dir_name = self.owlv2_model_id.rstrip("/\\").split("/")[-1].split("\\")[-1]
        for root in (Path("/workspace/models/cv"), Path("models"), Path("cv/models")):
            candidate = root / model_dir_name
            if candidate.exists():
                return str(candidate)
        return self.owlv2_model_id

    def _load_owlv2_model(self):
        """Load a Hugging Face OWLv2 zero-shot detector."""
        if torch is None or Owlv2ForObjectDetection is None or Owlv2Processor is None:
            raise RuntimeError("OWLv2 requires torch and transformers")

        self.owlv2_loaded_model_ref = self._resolve_owlv2_model_ref()
        self.processor = Owlv2Processor.from_pretrained(
            self.owlv2_loaded_model_ref,
            local_files_only=self.owlv2_local_files_only,
        )
        model = Owlv2ForObjectDetection.from_pretrained(
            self.owlv2_loaded_model_ref,
            local_files_only=self.owlv2_local_files_only,
        )

        device_name = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.owlv2_device = torch.device(device_name)
        self.owlv2_dtype = (
            torch.float16
            if self.half and self.owlv2_device.type == "cuda"
            else torch.float32
        )

        model.to(self.owlv2_device)
        if self.owlv2_dtype == torch.float16:
            model.half()
        model.eval()
        self.loaded_model_family = "owlv2"
        return model

    def _load_rfdetr_model(self, model_path: str):
        """Load a Roboflow RF-DETR detector (separate ``rfdetr`` package).

        Unlike the Ultralytics families, RF-DETR has its own model classes and a
        ``.pth`` checkpoint. Resolution is fixed at load time (positional
        embeddings depend on it), so it must match the training resolution.
        """
        cls = RFDETRLarge if self.rfdetr_variant == "large" else RFDETRBase
        if cls is None:
            raise RuntimeError("RF-DETR requested but the rfdetr package is unavailable")
        kwargs: dict[str, Any] = {"resolution": self.rfdetr_resolution}
        if model_path and Path(model_path).exists():
            kwargs["pretrain_weights"] = model_path
        model = cls(**kwargs)
        self.loaded_model_family = "rfdetr"
        return model

    def _apply_rtdetr_inference_knobs(self, model) -> None:
        """Optionally trade RT-DETR latency for accuracy via decoder/query limits."""
        if self.rtdetr_eval_idx is None and self.rtdetr_num_queries is None:
            return
        try:
            head = model.model.model[-1]
        except Exception as exc:
            print(f"[CVManager] could not access RT-DETR head: {exc}", flush=True)
            return
        if self.rtdetr_eval_idx is not None:
            try:
                head.decoder.eval_idx = self.rtdetr_eval_idx
            except Exception as exc:
                print(f"[CVManager] could not set CV_RTDETR_EVAL_IDX: {exc}", flush=True)
        if self.rtdetr_num_queries is not None:
            try:
                head.num_queries = self.rtdetr_num_queries
            except Exception as exc:
                print(f"[CVManager] could not set CV_RTDETR_NUM_QUERIES: {exc}", flush=True)

    def _load_category_map(self) -> dict[int, int]:
        """Load class-index overrides from CV_CATEGORY_MAP when provided.

        Accepts either a JSON string/path shaped like {"0": 1, "1": 2} or a
        list where the index is the YOLO class id and the value is eval id.
        Defaults to the sparse custom TIL label mapping above.
        """
        if self.resolved_model_family == "owlv2" and not os.environ.get("CV_CATEGORY_MAP"):
            return {idx: idx for idx in range(len(TIL_CATEGORY_NAMES))}

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

    def _load_owlv2_prompts(self) -> list[tuple[int, str]]:
        """Load text prompts for OWLv2.

        CV_OWLV2_PROMPTS may be a JSON string or path. Accepted shapes:
        - {"3": ["a fighter jet", "a military jet"], "9": "a car"}
        - ["cargo aircraft prompt", ..., "sailboat prompt"] in category order
        - [{"category_id": 3, "prompt": "a fighter jet"}, ...]
        """
        raw = os.environ.get("CV_OWLV2_PROMPTS")
        if not raw:
            return list(DEFAULT_OWLV2_PROMPTS)

        try:
            stripped = raw.strip()
            if stripped.startswith("{") or stripped.startswith("["):
                text = stripped
            else:
                text = Path(raw).read_text(encoding="utf-8")
            loaded = json.loads(text)
            specs: list[tuple[int, str]] = []

            if isinstance(loaded, dict):
                for category_raw, prompts_raw in loaded.items():
                    prompts = (
                        prompts_raw
                        if isinstance(prompts_raw, list)
                        else [prompts_raw]
                    )
                    for prompt in prompts:
                        specs.append((int(category_raw), str(prompt)))
            elif isinstance(loaded, list) and all(isinstance(item, str) for item in loaded):
                specs = [(idx, prompt) for idx, prompt in enumerate(loaded)]
            elif isinstance(loaded, list):
                for item in loaded:
                    if not isinstance(item, dict):
                        continue
                    category_id = int(item["category_id"])
                    prompts_raw = item.get("prompts", item.get("prompt", []))
                    prompts = (
                        prompts_raw
                        if isinstance(prompts_raw, list)
                        else [prompts_raw]
                    )
                    for prompt in prompts:
                        specs.append((category_id, str(prompt)))

            cleaned: list[tuple[int, str]] = []
            for category_id, prompt in specs:
                prompt = prompt.strip()
                if not prompt:
                    continue
                if 0 <= category_id < len(TIL_CATEGORY_NAMES):
                    cleaned.append((category_id, prompt))
                else:
                    print(
                        f"[CVManager] ignoring OWLv2 prompt with invalid "
                        f"category_id={category_id}",
                        flush=True,
                    )
            if cleaned:
                return cleaned
        except Exception as exc:
            print(f"[CVManager] could not parse CV_OWLV2_PROMPTS: {exc}", flush=True)
        return list(DEFAULT_OWLV2_PROMPTS)

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def _yolo_predict(
        self,
        img: Image.Image,
        imgsz: int,
        augment: bool,
        conf_threshold: float | None = None,
        iou_threshold: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run a single Ultralytics forward pass and return (xyxy, cls, conf).

        Returns three empty arrays on failure or zero detections.
        """
        kwargs: dict[str, Any] = {
            "source": img,
            "conf": self.conf if conf_threshold is None else conf_threshold,
            "iou": self.iou if iou_threshold is None else iou_threshold,
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

    def _post_process_owlv2(self, outputs, target_sizes):
        """Run the processor's OWLv2 postprocess across Transformers versions."""
        if self.processor is None:
            raise RuntimeError("OWLv2 processor is not loaded")

        if hasattr(self.processor, "post_process_grounded_object_detection"):
            try:
                return self.processor.post_process_grounded_object_detection(
                    outputs=outputs,
                    target_sizes=target_sizes,
                    threshold=self.conf,
                    text_labels=[self.owlv2_prompts],
                )[0]
            except TypeError:
                return self.processor.post_process_grounded_object_detection(
                    outputs=outputs,
                    target_sizes=target_sizes,
                    threshold=self.conf,
                )[0]

        if hasattr(self.processor, "post_process_object_detection"):
            return self.processor.post_process_object_detection(
                outputs=outputs,
                target_sizes=target_sizes,
                threshold=self.conf,
            )[0]

        raise RuntimeError("OWLv2 processor has no supported postprocess method")

    def _owlv2_predict(
        self,
        img: Image.Image,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run OWLv2 and return (xyxy, category_id, conf)."""
        if torch is None or self.processor is None or self.owlv2_device is None:
            empty = np.zeros((0, 4), dtype=np.float32)
            return empty, np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float32)

        inputs = self.processor(
            text=[self.owlv2_prompts],
            images=img,
            return_tensors="pt",
        )
        model_inputs = {}
        for key, value in inputs.items():
            if hasattr(value, "to"):
                value = value.to(self.owlv2_device)
                if key == "pixel_values" and self.owlv2_dtype is not None:
                    value = value.to(dtype=self.owlv2_dtype)
            model_inputs[key] = value

        with torch.inference_mode():
            outputs = self.model(**model_inputs)

        target_sizes = torch.tensor(
            [(img.height, img.width)],
            dtype=torch.float32,
            device=self.owlv2_device,
        )
        result = self._post_process_owlv2(outputs, target_sizes)
        boxes = result.get("boxes")
        scores = result.get("scores")
        labels = result.get("labels")
        text_labels = result.get("text_labels")

        if boxes is None or scores is None or len(boxes) == 0:
            empty = np.zeros((0, 4), dtype=np.float32)
            return empty, np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float32)

        if labels is not None and hasattr(labels, "detach"):
            label_values = labels.detach().cpu().numpy().astype(np.int64, copy=False)
            categories = []
            for label_value in label_values:
                idx = int(label_value)
                if 0 <= idx < len(self.owlv2_prompt_categories):
                    categories.append(self.owlv2_prompt_categories[idx])
                else:
                    categories.append(-1)
            cls = np.asarray(categories, dtype=np.int64)
        elif labels is not None:
            categories = []
            for label_value in labels:
                if isinstance(label_value, str):
                    categories.append(self.owlv2_prompt_to_category.get(label_value, -1))
                    continue
                idx = int(label_value)
                if 0 <= idx < len(self.owlv2_prompt_categories):
                    categories.append(self.owlv2_prompt_categories[idx])
                else:
                    categories.append(-1)
            cls = np.asarray(categories, dtype=np.int64)
        elif text_labels is not None:
            cls = np.asarray(
                [self.owlv2_prompt_to_category.get(str(label), -1) for label in text_labels],
                dtype=np.int64,
            )
        else:
            empty = np.zeros((0, 4), dtype=np.float32)
            return empty, np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float32)

        xyxy = boxes.detach().cpu().numpy().astype(np.float32, copy=False)
        conf = scores.detach().cpu().numpy().astype(np.float32, copy=False)
        valid = cls >= 0
        return xyxy[valid], cls[valid], conf[valid]

    def _rfdetr_predict(
        self,
        img: Image.Image,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run RF-DETR and return (xyxy, class_id, conf).

        RF-DETR returns a supervision ``Detections`` object with ``xyxy``,
        ``class_id`` and ``confidence`` arrays. CV_CONF is the predict threshold.
        Class ids are mapped to TIL category ids downstream via ``category_map``.
        """
        empty = (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            np.zeros((0,), dtype=np.float32),
        )
        if self.model is None:
            return empty

        detections = self.model.predict(img, threshold=self.conf)
        xyxy = getattr(detections, "xyxy", None)
        class_id = getattr(detections, "class_id", None)
        conf = getattr(detections, "confidence", None)
        if xyxy is None or class_id is None or conf is None or len(xyxy) == 0:
            return empty

        xyxy = np.asarray(xyxy, dtype=np.float32)
        cls = np.asarray(class_id, dtype=np.int64)
        conf = np.asarray(conf, dtype=np.float32)
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
        self._last_second_pass_used = False
        if self.loaded_model_family == "owlv2":
            xyxy, cls, conf = self._owlv2_predict(img)
            return xyxy.tolist(), cls.tolist(), conf.tolist()
        if self.loaded_model_family == "rfdetr":
            xyxy, cls, conf = self._rfdetr_predict(img)
            return xyxy.tolist(), cls.tolist(), conf.tolist()

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

        if self.second_pass and len(all_xyxy) >= self.second_min_detections:
            self._last_second_pass_used = True
            try:
                xyxy, cls, conf = self._yolo_predict(
                    img,
                    imgsz=self.second_imgsz,
                    augment=self.second_augment,
                    conf_threshold=self.second_conf,
                    iou_threshold=self.second_iou,
                )
            except Exception as exc:
                print(f"[CVManager] second-pass inference failed: {exc}", flush=True)
                xyxy = np.zeros((0, 4), dtype=np.float32)
                cls = np.zeros((0,), dtype=np.int64)
                conf = np.zeros((0,), dtype=np.float32)
            for box, cls_value, conf_value in zip(xyxy, cls, conf):
                bx1, by1, bx2, by2 = (float(v) for v in box)
                all_xyxy.append([bx1, by1, bx2, by2])
                all_cls.append(int(cls_value))
                all_conf.append(float(conf_value) * self.second_score_scale)

        return all_xyxy, all_cls, all_conf

    @staticmethod
    def _nms(
        xyxy: np.ndarray,
        cls: np.ndarray,
        conf: np.ndarray,
        iou_threshold: float,
        class_aware: bool,
    ) -> np.ndarray:
        """Greedy NMS. Returns survivor indices in descending confidence order."""
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
                if class_aware and cls[jdx] != cls[idx]:
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

    @staticmethod
    def _class_aware_nms(
        xyxy: np.ndarray,
        cls: np.ndarray,
        conf: np.ndarray,
        iou_threshold: float,
    ) -> np.ndarray:
        """Greedy class-aware NMS. Returns sorted survivor indices."""
        return CVManager._nms(xyxy, cls, conf, iou_threshold, class_aware=True)

    @staticmethod
    def _class_agnostic_nms(
        xyxy: np.ndarray,
        cls: np.ndarray,
        conf: np.ndarray,
        iou_threshold: float,
    ) -> np.ndarray:
        """Greedy class-agnostic NMS for near-identical subclass duplicates."""
        return CVManager._nms(xyxy, cls, conf, iou_threshold, class_aware=False)

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

        # NMS only needed when tiling or the rescue pass produced overlapping
        # candidates from multiple sources. Single-pass (off-mode) detections
        # already came from Ultralytics' built-in NMS, so skipping here keeps
        # that path a bit-for-bit no-op.
        if self.loaded_model_family == "owlv2":
            survivors = self._class_aware_nms(xyxy, cls, conf, self.owlv2_nms_iou)
            xyxy = xyxy[survivors]
            cls = cls[survivors]
            conf = conf[survivors]
        elif self.tile_grid is not None:
            survivors = self._class_aware_nms(xyxy, cls, conf, self.tile_merge_iou)
            xyxy = xyxy[survivors]
            cls = cls[survivors]
            conf = conf[survivors]
        elif self.second_pass:
            survivors = self._class_aware_nms(xyxy, cls, conf, self.second_merge_iou)
            xyxy = xyxy[survivors]
            cls = cls[survivors]
            conf = conf[survivors]

        # Some tuned checkpoints emit near-identical boxes with different fine
        # subclasses (e.g. fighter jet/fighter plane or cargo ship/warship).
        # A very high class-agnostic threshold removes only duplicates that are
        # effectively the same object while leaving normal overlaps intact.
        if self.cross_class_nms_iou > 0.0:
            survivors = self._class_agnostic_nms(
                xyxy, cls, conf, self.cross_class_nms_iou
            )
            xyxy = xyxy[survivors]
            cls = cls[survivors]
            conf = conf[survivors]

        # Optional one-time log so we can verify the configured path fired.
        if self._inference_count == 0:
            print(
                f"[CVManager] first inference: family={self.loaded_model_family} "
                f"tile_mode={self.tile_mode} "
                f"second_pass_used={self._last_second_pass_used} "
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
