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

        img = Image.new("RGB", (100, 100), color="red")
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="JPEG")
        img_bytes = img_bytes.getvalue()

        # Resize to 50x50 (scale=0.5); detection box [10,10,20,20] is in that space.
        def mock_gather(purified_img):
            self.assertEqual(purified_img.size, (50, 50))
            return [[10.0, 10.0, 20.0, 20.0]], [4], [0.9]

        manager._gather_detections = mock_gather

        # Force random scale to exactly 0.5
        with patch("random.uniform", return_value=0.5):
            detections = manager.cv(img_bytes)

        # Box scales back by 1/0.5: [10,10,20,20] -> [20,20,40,40] -> LTWH [20,20,20,20]
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

        manager.model = MagicMock()
        manager.loaded_model_family = "yolo"

        img = Image.new("RGB", (100, 100), color="red")
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="JPEG")
        img_bytes = img_bytes.getvalue()

        # Assert the image was NOT resized
        def mock_gather(purified_img):
            self.assertEqual(purified_img.size, (100, 100))
            return [[10.0, 10.0, 20.0, 20.0]], [4], [0.9]

        manager._gather_detections = mock_gather

        detections = manager.cv(img_bytes)

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]["bbox"], [10.0, 10.0, 10.0, 10.0])
        self.assertEqual(detections[0]["category_id"], 1)


if __name__ == "__main__":
    unittest.main()
