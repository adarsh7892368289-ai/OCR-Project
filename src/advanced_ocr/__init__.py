# Main package initialization and public API for advanced_ocr

"""
Advanced OCR Library - Multi-engine text extraction with intelligent preprocessing

This package provides a modern, modular OCR solution that:
- Supports multiple OCR engines (PaddleOCR, EasyOCR, Tesseract, TrOCR)
- Includes intelligent image preprocessing and quality analysis
- Offers flexible processing strategies and batch processing
- Provides consistent API across different OCR technologies

Quick Start:
    >>> from advanced_ocr import OCRLibrary, extract_text
    >>>
    >>> # Simple usage
    >>> text = extract_text("document.jpg")
    >>>
    >>> # Advanced usage
    >>> ocr = OCRLibrary()
    >>> result = ocr.process_image("document.jpg")
    >>> print(f"Text: {result.text}")
    >>> print(f"Confidence: {result.confidence}")
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
    # Main library class
    'OCRLibrary',
    
    # Core data types
    'ProcessingOptions', 
    'OCRResult', 
    'QualityMetrics', 
    'ProcessingStrategy',
    'BatchResult',
    
    # Exceptions
    'OCRLibraryError', 
    'EngineNotAvailableError',
    'ProcessingTimeoutError',
    'EngineInitializationError',
    
    # Convenience functions
    'extract_text',
    'process_images',
]

def extract_text(image_path: str, **kwargs) -> str:
    """
    Quick text extraction with default settings.
    
    Args:
        image_path: Path to image file
        **kwargs: Additional options passed to ProcessingOptions
        
    Returns:
        Extracted text as string
        
    Examples:
        >>> text = extract_text("document.jpg")
        >>> text = extract_text("receipt.png", enhance_image=True)
        >>> text = extract_text("form.pdf", engines=["tesseract"])
    """
    ocr = OCRLibrary()
    options = ProcessingOptions(**kwargs) if kwargs else None
    result = ocr.process_image(image_path, options)
    return result.text


def process_images(image_paths: list, **kwargs) -> list:
    """
    Process multiple images and return results.
    
    Args:
        image_paths: List of image file paths
        **kwargs: Additional options passed to ProcessingOptions
        
    Returns:
        List of OCRResult objects
        
    Examples:
        >>> results = process_images(["doc1.jpg", "doc2.png"])
        >>> for result in results:
        ...     print(f"File: {result.metadata.get('image_path')}")
        ...     print(f"Text: {result.text}")
    """
    ocr = OCRLibrary()
    options = ProcessingOptions(**kwargs) if kwargs else None
    batch_result = ocr.process_batch(image_paths, options)
    return batch_result.results


def configure_logging(level="INFO", log_file=None):
    """
    Configure logging for the entire library.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional path to log file
        
    Examples:
        >>> import advanced_ocr
        >>> advanced_ocr.configure_logging("DEBUG", "ocr.log")
    """
    from .utils.logging import setup_logging
    setup_logging(level=level, log_file=log_file)

try:
    from .utils.logging import setup_logging
    setup_logging(level="WARNING")  # Only warnings and errors by default
except ImportError:
    pass  # Logging setup not critical for package functionali