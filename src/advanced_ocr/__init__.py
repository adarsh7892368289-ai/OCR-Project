"""Advanced OCR library with multi-engine support and intelligent preprocessing.

Provides a unified interface for multiple OCR engines (PaddleOCR, EasyOCR, 
Tesseract, TrOCR) with automatic quality analysis and image enhancement.

Examples
--------
    from advanced_ocr import extract_text, OCRLibrary, ProcessingOptions
    
    # Quick text extraction
    text = extract_text("document.jpg")
    
    # Advanced usage with options
    ocr = OCRLibrary()
    options = ProcessingOptions(enhance_image=True, engines=["paddleocr"])
    result = ocr.process_image("document.jpg", options)
    
    # Batch processing
    from advanced_ocr import process_images
    results = process_images(["doc1.jpg", "doc2.png", "doc3.jpg"])
"""

from .pipeline import OCRLibrary
from .types import (
    ProcessingOptions, 
    OCRResult, 
    QualityMetrics, 
    ProcessingStrategy,
    BatchResult
)
from .exceptions import (
    OCRLibraryError, 
    EngineNotAvailableError,
    ProcessingTimeoutError,
    EngineInitializationError
)

__all__ = [
    'OCRLibrary',
    'ProcessingOptions', 
    'OCRResult', 
    'QualityMetrics', 
    'ProcessingStrategy',
    'BatchResult',
    'OCRLibraryError', 
    'EngineNotAvailableError',
    'ProcessingTimeoutError',
    'EngineInitializationError',
    'extract_text',
    'process_images',
    'configure_logging',
]


def extract_text(image_path: str, **kwargs) -> str:
    """Extract text from an image using default settings.
    
    Convenience function for simple text extraction. For more control,
    use OCRLibrary class directly.
    """
    ocr = OCRLibrary()
    options = ProcessingOptions(**kwargs) if kwargs else None
    result = ocr.process_image(image_path, options)
    return result.text


def process_images(image_paths: list, **kwargs) -> list:
    """Process multiple images in batch and return results."""
    ocr = OCRLibrary()
    options = ProcessingOptions(**kwargs) if kwargs else None
    batch_result = ocr.process_batch(image_paths, options)
    return batch_result.results


def configure_logging(level: str = "INFO", log_file: str = None) -> None:
    """Configure logging level and output destination for the library."""
    from .utils.logging import setup_logging
    setup_logging(level=level, log_file=log_file)


# Initialize default logging
try:
    from .utils.logging import setup_logging
    setup_logging(level="WARNING")
except ImportError:
    pass