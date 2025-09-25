# src/advanced_ocr/engines/__init__.py
"""
OCR Engines Module - Expose all engine implementations.

This module provides access to all OCR engine implementations.
Each engine handles its own initialization and text extraction.
"""

from .paddleocr import PaddleOCR
from .easyocr import EasyOCREngine

# Add these imports when you have the other engine files
# from .tesseract import TesseractEngine  
# from .trocr import TrOCREngine

# Export the engines for easy import
__all__ = [
    'PaddleOCR',
    'EasyOCREngine',
    'TesseractEngine',
    'TrOCREngine'
]
