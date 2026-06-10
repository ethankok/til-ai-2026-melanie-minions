import unittest
from unittest.mock import MagicMock, patch
import os
import io
import numpy as np
from PIL import Image
from cv_manager import CVManager


class TestCVPurify(unittest.TestCase):

    @patch("cv_manager.YOLO")
    @patch.dict(os.environ, {
        "CV_PURIFY": "True",
        "CV_PURIFY_JPEG_QUALITY": "75",
        "CV_PURIFY_BLUR_SIGMA": "0.5",
        "CV_PURIFY_MIN_SCALE": "0.5"
    })
    def test_purify_and_scaling(self, mock_yolo):
        """Test that purification runs and scales predicted boxes back to the original image dimensions."""
        manager = CVManager()
        self.assertTrue(manager.purify)
        self.assertEqual(manager.purify_jpeg_quality, 75)
        self.assertEqual(manager.purify_blur_sigma, 0.5)
        self.assertEqual(manager.purify_min_scale, 0.5)

        # Set a dummy model so CVManager doesn't return early
        manager.model = MagicMock()
        manager.loaded_model_family = "yolo"

        # Create a red 100x100 image
        img = Image.new("RGB", (100, 100), color="red")
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="JPEG")
        img_bytes = img_bytes.getvalue()

        # Mock _gather_detections to assert the image was resized to 50x50 (scale=0.5)
        # and return a box [10, 10, 20, 20] in the 50x50 space.
        def mock_gather(purified_img):
            self.assertEqual(purified_img.size, (50, 50))
            return [[10.0, 10.0, 20.0, 20.0]], [4], [0.9]

        manager._gather_detections = mock_gather

        # Force random scale to be exactly 0.5
        with patch("random.uniform", return_value=0.5):
            detections = manager.cv(img_bytes)

        # Resized space: box at [10, 10, 20, 20]
        # Scaling factor: 50 / 100 = 0.5
        # Original space: [10/0.5, 10/0.5, 20/0.5, 20/0.5] = [20, 20, 40, 40]
        # LTWH format: left=20, top=20, width=(40-20)=20, height=(40-20)=20
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]["bbox"], [20.0, 20.0, 20.0, 20.0])
        self.assertAlmostEqual(detections[0]["score"], 0.9, places=5)
        self.assertEqual(detections[0]["category_id"], 1)

    @patch("cv_manager.YOLO")
    @patch.dict(os.environ, {
        "CV_PURIFY": "False"
    })
    def test_purify_disabled(self, mock_yolo):
        """Test that purification is disabled by default and does not affect the image."""
        manager = CVManager()
        self.assertFalse(manager.purify)

        # Set a dummy model
        manager.model = MagicMock()
        manager.loaded_model_family = "yolo"

        # Create a red 100x100 image
        img = Image.new("RGB", (100, 100), color="red")
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="JPEG")
        img_bytes = img_bytes.getvalue()

        # Mock _gather_detections to assert the image was NOT resized
        def mock_gather(purified_img):
            self.assertEqual(purified_img.size, (100, 100))
            return [[10.0, 10.0, 20.0, 20.0]], [4], [0.9]

        manager._gather_detections = mock_gather

        detections = manager.cv(img_bytes)

        # Output box should be exactly [10.0, 10.0, 10.0, 10.0] (no scaling)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]["bbox"], [10.0, 10.0, 10.0, 10.0])
        self.assertEqual(detections[0]["category_id"], 1)


if __name__ == "__main__":
    unittest.main()
