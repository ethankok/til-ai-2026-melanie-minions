"""Baseline adversarial-noise manager.

Passes images through unchanged. This is useful as a safety baseline: it should
pass image validity/fairness constraints while scoring low adversarial impact.
"""

import base64
import io

from PIL import Image


class NoiseManager:
    """Valid pass-through noising baseline."""

    def __init__(self):
        # Later: add bounded perturbations here while respecting fairness checks.
        pass

    def noise(self, image: bytes) -> str:
        """Return a valid base64-encoded JPEG image.

        Args:
            image: Input JPEG bytes.

        Returns:
            Base64-encoded JPEG bytes.
        """
        try:
            # Re-encode to ensure the output is a clean RGB JPEG even if the
            # input has EXIF/progressive/odd channel metadata.
            img = Image.open(io.BytesIO(image)).convert("RGB")
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG", quality=95)
            return base64.b64encode(buffered.getvalue()).decode("ascii")
        except Exception as exc:  # keep endpoint alive even on malformed input
            print(f"Noise baseline fallback after error: {exc}")
            return base64.b64encode(image).decode("ascii")
