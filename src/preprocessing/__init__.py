"""
Preprocessing module for OCR pipeline
"""

# Import main classes with expected names for compatibility
from .image_enhancer import AIImageEnhancer as ImageEnhancer
from .quality_analyzer import IntelligentQualityAnalyzer as QualityAnalyzer
from .text_detector import AdvancedTextDetector as TextDetector

# Also import the actual classes for direct access
from .image_enhancer import AIImageEnhancer
from .quality_analyzer import IntelligentQualityAnalyzer
from .text_detector import AdvancedTextDetector

# Import other preprocessing modules
from . import skew_corrector
from . import adaptive_processor

__all__ = [
    'ImageEnhancer',
    'QualityAnalyzer',
    'TextDetector',
    'AIImageEnhancer',
    'IntelligentQualityAnalyzer',
    'AdvancedTextDetector',
    'skew_corrector',
    'adaptive_processor'
]
