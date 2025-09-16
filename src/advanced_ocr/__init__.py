# src/advanced_ocr/__init__.py
"""
Advanced OCR Package Initialization

This module initializes the advanced OCR package and provides the main API
for OCR processing with multiple engines and optimization features.

The package provides:
- Multi-engine OCR processing with intelligent selection
- Advanced image preprocessing and enhancement
- Hierarchical text result structures
- Performance monitoring and optimization
- Batch processing capabilities

Classes:
    OCRResult: Primary OCR result container
    EngineManager: Multi-engine coordination and management
    BatchResult: Container for batch processing results

Functions:
    process_image: Main OCR processing function
    batch_process: Batch processing function

Example:
    >>> from advanced_ocr import process_image, EngineManager
    >>> result = process_image("document.jpg")
    >>> print(f"Extracted text: {result.text}")

"""

from .results import OCRResult, BatchResult
from .core import EngineManager

__version__ = "1.0.0"

__all__ = [
    'OCRResult',
    'BatchResult',
    'EngineManager',
    '__version__'
]
