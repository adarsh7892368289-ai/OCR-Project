import pytest
import numpy as np
from PIL import Image
from pathlib import Path

from advanced_ocr.config import OCRConfig
from advanced_ocr.results import BoundingBox
from advanced_ocr.engines.tesseract_engine import TesseractEngine
from advanced_ocr.engines.paddleocr_engine import PaddleOCREngine
from advanced_ocr.engines.easyocr_engine import EasyOCREngine
from advanced_ocr.engines.trocr_engine import TrOCREngine


@pytest.fixture(scope="module")
def load_image():
    img_path = Path("data/sample_images/img3.jpg")
    image = Image.open(img_path).convert("RGB")
    return np.array(image)


@pytest.fixture
def sample_text_regions():
    # Dummy text regions - these should correspond to actual detected regions ideally
    return [
        BoundingBox(coordinates=(50, 50, 200, 100), format='xywh'),
        BoundingBox(coordinates=(60, 160, 250, 80), format='xywh')
    ]


@pytest.mark.parametrize("engine_class", [
    TesseractEngine,
    PaddleOCREngine,
    EasyOCREngine,
    TrOCREngine
])
def test_ocr_engines(engine_class, load_image, sample_text_regions):
    config = OCRConfig()
    engine = engine_class(config)

    try:
        engine.initialize()
    except Exception as e:
        pytest.skip(f"Skipping {engine_class.__name__} initialization: {e}")

    try:
        result = engine.extract(load_image, sample_text_regions)
    except Exception as e:
        pytest.fail(f"{engine_class.__name__}.extract() failed: {e}")

    assert result is not None, f"{engine_class.__name__} returned None"
    assert hasattr(result, "text"), f"{engine_class.__name__} result missing text"
    assert isinstance(result.text, str), f"{engine_class.__name__} result text is not string"

    confidence = result.confidence
    # Support both float and structured confidence objects
    if not isinstance(confidence, (float, int)):
        confidence = getattr(confidence, 'overall', 0.0)
    assert 0.0 <= confidence <= 1.0, f"{engine_class.__name__} confidence out of range: {confidence}"

    # Warn if extracted text is empty (may be normal for test images)
    if len(result.text.strip()) == 0:
        pytest.warns(UserWarning, f"{engine_class.__name__} extracted empty text")

    # Check for bounding boxes attribute if applicable
    if hasattr(result, "bounding_boxes"):
        bboxes = result.bounding_boxes
        assert isinstance(bboxes, (list, tuple)), f"{engine_class.__name__} bounding_boxes not a list/tuple"
        for box in bboxes:
            assert hasattr(box, "x") and hasattr(box, "y"), f"{engine_class.__name__} bounding box missing coordinates"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
