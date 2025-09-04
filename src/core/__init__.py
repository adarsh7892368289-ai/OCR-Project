
"""
OCR Core Module
Provides base classes and engine management for OCR operations
"""

from .base_engine import (
    BaseOCREngine, 
    OCRResult, 
    DocumentResult, 
    TextRegion, 
    BoundingBox,
    DocumentStructure,
    TextType,
    DetectionMethod
)

from .engine_manager import OCREngineManager

# Create aliases for backward compatibility
OCREngine = BaseOCREngine  # This fixes the "cannot import OCREngine" error
EngineManager = OCREngineManager  # This fixes the "cannot import EngineManager" error

__all__ = [
    'BaseOCREngine',
    'OCREngine',  # Alias
    'OCREngineManager', 
    'EngineManager',  # Alias
    'OCRResult',
    'DocumentResult',
    'TextRegion',
    'BoundingBox', 
    'DocumentStructure',
    'TextType',
    'DetectionMethod'
]