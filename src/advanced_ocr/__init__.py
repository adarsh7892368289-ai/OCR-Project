"""
OCR Library - Multi-engine text extraction with intelligent preprocessing
"""

from .pipeline import OCRLibrary, ProcessingOptions
from .types import OCRResult, QualityMetrics, ProcessingStrategy
from .exceptions import OCRLibraryError, EngineNotAvailableError

__version__ = "1.0.0"
__all__ = [
    'OCRLibrary', 'ProcessingOptions', 'BatchProcessor',
    'OCRResult', 'QualityMetrics', 'ProcessingStrategy',
    'OCRLibraryError', 'EngineNotAvailableError'
]

# Convenience function for quick usage
def extract_text(image_path: str, **kwargs) -> str:
    """Quick text extraction with default settings"""
    ocr = OCRLibrary()
    return ocr.extract_text(image_path, **kwargs).text