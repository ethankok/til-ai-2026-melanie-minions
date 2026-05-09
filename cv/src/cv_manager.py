"""Baseline CV manager.

Returns no detections. This is schema-valid and useful for checking Docker/API
plumbing before adding a detector such as YOLO/RT-DETR.
"""

from typing import Any


class CVManager:
    """Valid-but-dumb object-detection baseline."""

    def __init__(self):
        # Later: load detector weights here once, not inside cv().
        pass

    def cv(self, image: bytes) -> list[dict[str, Any]]:
        """Detect objects in one JPEG image.

        Args:
            image: JPEG bytes.

        Returns:
            List of detections, each with bbox [x, y, w, h] and category_id.
            Empty list means "no objects detected" and is explicitly valid.
        """
        _ = image
        return []
