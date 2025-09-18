import unittest
import numpy as np
import cv2
import time
from unittest.mock import patch

from src.advanced_ocr.preprocessing.image_processor import ImageProcessor, PreprocessingResult
from src.advanced_ocr.preprocessing.text_detector import TextDetector
from src.advanced_ocr.core import OCRCore
from src.advanced_ocr.utils.model_utils import ModelLoader
from src.advanced_ocr.results import BoundingBox, BoundingBoxFormat, OCRResult
from src.advanced_ocr.preprocessing.quality_analyzer import QualityMetrics
from src.advanced_ocr.preprocessing.content_classifier import ContentClassification

class MockModelLoader(ModelLoader):
    def load_model(self, model_name: str, model_path: str = None):
        # Return a simple mock model object
        class MockModel:
            def __init__(self):
                pass
            def predict(self, *args, **kwargs):
                return None
        return MockModel()

class TestOCRPipeline(unittest.TestCase):
    def setUp(self):
        self.model_loader = MockModelLoader()
        self.test_image = self._create_test_image()
        self.config = None  # Use default config in components

    def _create_test_image(self):
        # Create a simple white image with black rectangles simulating text regions
        img = np.ones((600, 800, 3), dtype=np.uint8) * 255
        cv2.rectangle(img, (50, 50), (200, 100), (0, 0, 0), -1)
        cv2.rectangle(img, (300, 150), (500, 200), (0, 0, 0), -1)
        cv2.rectangle(img, (100, 300), (400, 350), (0, 0, 0), -1)
        return img

    @patch('src.advanced_ocr.preprocessing.text_detector.TextDetector.detect_text_regions')
    @patch('src.advanced_ocr.preprocessing.quality_analyzer.QualityAnalyzer.analyze_image_quality')
    def test_image_processor(self, mock_quality_analyze, mock_text_detect):
        # Mock quality analysis to return a fixed quality metric
        mock_quality_analyze.return_value = QualityMetrics(
            blur_score=0.9,
            blur_level=None,
            laplacian_variance=1000,
            noise_score=0.9,
            noise_level=None,
            noise_variance=5,
            contrast_score=0.9,
            contrast_level=None,
            contrast_rms=70,
            histogram_spread=90,
            resolution_score=0.9,
            resolution_level=None,
            effective_resolution=(800, 600),
            dpi_estimate=300,
            brightness_score=0.9,
            brightness_level=None,
            mean_brightness=130,
            brightness_uniformity=0.9,
            overall_score=0.9,
            overall_level=None,
            image_dimensions=(600, 800),
            color_channels=3,
            analysis_time=0.01
        )
        # Mock text detection to return 30 bounding boxes
        mock_text_detect.return_value = [
            BoundingBox((50, 50, 150, 50), BoundingBoxFormat.XYWH, 0.8) for _ in range(30)
        ]

        processor = ImageProcessor(self.model_loader, self.config)
        result = processor.process_image(self.test_image)

        self.assertIsInstance(result, PreprocessingResult)
        self.assertEqual(len(result.text_regions), 30)
        self.assertAlmostEqual(result.quality_metrics.overall_score, 0.9, places=2)

    def test_core_pipeline(self):
        core = OCRCore()
        result = core.extract_text(self.test_image)
        self.assertIsInstance(result, OCRResult)
        self.assertTrue(len(result.text) >= 0)  # Text may be empty in mock but should be string

    def test_batch_extract(self):
        core = OCRCore()
        images = [self.test_image for _ in range(3)]
        batch_result = core.batch_extract(images)
        self.assertEqual(batch_result.total_images, 3)
        self.assertEqual(batch_result.successful_images, 3)
        self.assertEqual(len(batch_result.results), 3)

if __name__ == "__main__":
    unittest.main()
