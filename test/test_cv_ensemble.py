import unittest
from unittest.mock import MagicMock, patch
import os
import io
from PIL import Image
from cv_manager import CVManager


class TestCVEnsemble(unittest.TestCase):

    @patch("cv_manager.YOLO")
    @patch.dict(os.environ, {
        "CV_MODEL_PATH": "model1.pt,model2.pt",
        "CV_PURIFY": "False"
    })
    def test_ensemble_inference(self, mock_yolo):
        """Test that ensembling loads both models and aggregates their predictions."""
        # Set up mock YOLO instances
        model1 = MagicMock()
        model2 = MagicMock()
        
        # When YOLO(path) is called, return model1 or model2
        def mock_init(path):
            if "model1" in path:
                return model1
            elif "model2" in path:
                return model2
            return MagicMock()
            
        mock_yolo.side_effect = mock_init

        manager = CVManager()
        self.assertEqual(len(manager.models), 2)
        self.assertEqual(manager.models[0], model1)
        self.assertEqual(manager.models[1], model2)
        
        # Override loaded_model_family to yolo
        manager.loaded_model_family = "yolo"

        # Create dummy image
        img = Image.new("RGB", (100, 100), color="blue")
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="JPEG")
        img_bytes = img_bytes.getvalue()

        # Mock predictions:
        # model1 predicts box [10.0, 10.0, 20.0, 20.0]
        # model2 predicts box [30.0, 30.0, 50.0, 50.0]
        def mock_predict_1(img, imgsz, augment, model):
            if model == model1:
                return [[10.0, 10.0, 20.0, 20.0]], [4], [0.9]
            elif model == model2:
                return [[30.0, 30.0, 50.0, 50.0]], [8], [0.8]
            return [], [], []

        manager._yolo_predict = mock_predict_1

        detections = manager.cv(img_bytes)

        # Output should contain both boxes
        # model1: class=4 -> mapped category_id=1? No, categories maps:
        # category_map default is identity 0..17 in test_cv_purify, let's see what category_map is loaded.
        # It loads from env or defaults. If identity map:
        # category_id = class_idx
        # Let's assert we have 2 detections.
        self.assertEqual(len(detections), 2)
        
        bboxes = [det["bbox"] for det in detections]
        # Both boxes should be present in LTWH format
        # [10, 10, 20, 20] -> left=10, top=10, width=10, height=10
        # [30, 30, 50, 50] -> left=30, top=30, width=20, height=20
        self.assertIn([10.0, 10.0, 10.0, 10.0], bboxes)
        self.assertIn([30.0, 30.0, 20.0, 20.0], bboxes)


if __name__ == "__main__":
    unittest.main()
