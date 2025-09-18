import pytest
import sys
import os
import numpy as np
from PIL import Image

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


@pytest.fixture
def sample_image(tmp_path):
    """Create a test image using pytest tmp_path fixture"""
    img_path = tmp_path / "test_image.jpg"
    img = Image.new('RGB', (800, 600), color='white')
    img_array = np.array(img)
    img_array[100:150, 200:600] = [0, 0, 0]  # Simulated text line 1
    img_array[200:250, 200:500] = [0, 0, 0]  # Simulated text line 2
    img = Image.fromarray(img_array)
    img.save(img_path)
    return img_path


class TestPreprocessingPipeline:
    """Test the complete preprocessing pipeline"""

    def test_image_processor_exists(self):
        try:
            from advanced_ocr.preprocessing.image_processor import ImageProcessor
        except ImportError as e:
            pytest.fail(f"ImageProcessor import failed: {e}")

    def test_quality_analyzer_exists(self):
        try:
            from advanced_ocr.preprocessing.quality_analyzer import QualityAnalyzer
        except ImportError as e:
            pytest.fail(f"QualityAnalyzer import failed: {e}")

    def test_text_detector_exists(self):
        try:
            from advanced_ocr.preprocessing.text_detector import TextDetector
        except ImportError as e:
            pytest.fail(f"TextDetector import failed: {e}")

    def test_image_processor_initialization(self):
        try:
            from advanced_ocr.preprocessing.image_processor import ImageProcessor
            from advanced_ocr.config import OCRConfig
            from advanced_ocr.utils.model_utils import ModelLoader

            config = OCRConfig("image_processor")
            model_loader = ModelLoader(config)
            processor = ImageProcessor(model_loader, config)
            assert processor is not None
        except Exception as e:
            pytest.fail(f"ImageProcessor initialization failed: {e}")

    def test_quality_analyzer_initialization(self):
        try:
            from advanced_ocr.preprocessing.quality_analyzer import QualityAnalyzer
            from advanced_ocr.config import OCRConfig

            config = OCRConfig("quality_analyzer")
            analyzer = QualityAnalyzer(config)
            assert analyzer is not None
        except Exception as e:
            pytest.fail(f"QualityAnalyzer initialization failed: {e}")

    def test_text_detector_initialization(self):
        try:
            from advanced_ocr.preprocessing.text_detector import TextDetector
            from advanced_ocr.config import OCRConfig
            from advanced_ocr.utils.model_utils import ModelLoader

            config = OCRConfig("text_detector")
            model_loader = ModelLoader(config)
            detector = TextDetector(model_loader, config)
            assert detector is not None
        except Exception as e:
            pytest.fail(f"TextDetector initialization failed: {e}")

    def test_quality_analyzer_analyze_method(self, sample_image):
        try:
            from advanced_ocr.preprocessing.quality_analyzer import QualityAnalyzer
            from advanced_ocr.config import OCRConfig
            from PIL import Image
            import numpy as np

            config = OCRConfig("quality_analyzer")
            analyzer = QualityAnalyzer(config)
            with Image.open(sample_image) as image:
                img_array = np.array(image)
                assert hasattr(analyzer, 'analyze_image_quality'), "QualityAnalyzer missing analyze_image_quality() method"
                quality_metrics = analyzer.analyze_image_quality(img_array)
                assert quality_metrics is not None
        except Exception as e:
            pytest.fail(f"QualityAnalyzer.analyze_image_quality() test failed: {e}")

    def test_text_detector_detect_method(self, sample_image):
        try:
            from advanced_ocr.preprocessing.text_detector import TextDetector
            from advanced_ocr.config import OCRConfig
            from advanced_ocr.utils.model_utils import ModelLoader
            from PIL import Image
            import numpy as np

            config = OCRConfig("text_detector")
            model_loader = ModelLoader(config)
            detector = TextDetector(model_loader, config)
            with Image.open(sample_image) as image:
                img_array = np.array(image)
                assert hasattr(detector, 'detect_text_regions'), "TextDetector missing detect_text_regions() method"
                text_regions = detector.detect_text_regions(img_array)
                assert text_regions is not None
                assert isinstance(text_regions, list)
                assert len(text_regions) < 500, f"Too many text regions detected: {len(text_regions)}"
        except Exception as e:
            pytest.fail(f"TextDetector.detect_text_regions() test failed: {e}")

    def test_image_processor_process_method(self, sample_image):
        try:
            from advanced_ocr.preprocessing.image_processor import ImageProcessor
            from advanced_ocr.config import OCRConfig
            from advanced_ocr.utils.model_utils import ModelLoader
            from PIL import Image
            import numpy as np

            config = OCRConfig("image_processor")
            model_loader = ModelLoader(config)
            processor = ImageProcessor(model_loader, config)
            with Image.open(sample_image) as image:
                img_array = np.array(image)
                assert hasattr(processor, 'process_image'), "ImageProcessor missing process_image() method"
                result = processor.process_image(img_array)
                assert result is not None
                expected_attrs = ['enhanced_image', 'text_regions', 'quality_metrics']
                for attr in expected_attrs:
                    assert hasattr(result, attr), f"PreprocessingResult missing attribute {attr}"
                assert isinstance(result.enhanced_image, np.ndarray), "enhanced_image should be numpy.ndarray"
                assert isinstance(result.text_regions, list), "text_regions should be list"
                assert len(result.text_regions) < 500, f"Too many text regions: {len(result.text_regions)}"
        except Exception as e:
            pytest.fail(f"ImageProcessor.process_image() test failed: {e}")

    def test_preprocessing_dependencies(self):
        try:
            from advanced_ocr.preprocessing import image_processor
            import inspect

            source = inspect.getsource(image_processor)
            assert 'quality_analyzer' in source.lower() or 'QualityAnalyzer' in source, "image_processor.py should import quality_analyzer"
            assert 'text_detector' in source.lower() or 'TextDetector' in source, "image_processor.py should import text_detector"
        except Exception as e:
            pytest.fail(f"Preprocessing dependencies test failed: {e}")

    def test_preprocessing_pipeline_integration(self, sample_image):
        try:
            from advanced_ocr.preprocessing.image_processor import ImageProcessor
            from advanced_ocr.config import OCRConfig
            from advanced_ocr.utils.model_utils import ModelLoader
            from PIL import Image
            import numpy as np

            config = OCRConfig("preprocessing_pipeline")
            model_loader = ModelLoader(config)
            processor = ImageProcessor(model_loader, config)
            with Image.open(sample_image) as image:
                img_array = np.array(image)
                result = processor.process_image(img_array)
                assert result.enhanced_image is not None
                assert result.text_regions is not None
                assert result.quality_metrics is not None
                assert 0 < len(result.text_regions) < 500, f"Text regions count unreasonable: {len(result.text_regions)}"
                print(f"✅ Preprocessing pipeline test passed:")
                print(f"   - Enhanced image: {type(result.enhanced_image)}")
                print(f"   - Text regions: {len(result.text_regions)} regions")
                print(f"   - Quality metrics: {type(result.quality_metrics)}")
        except Exception as e:
            pytest.fail(f"Preprocessing pipeline integration test failed: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
