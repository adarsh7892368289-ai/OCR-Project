# =============================================================================
# src/advanced_ocr/utils/__init__.py - UTILITIES API
# =============================================================================
"""
Utilities Package - Core Support Functions and Classes
=====================================================

This package contains essential utilities used throughout the OCR pipeline:

Image Operations:
- ImageLoader: Memory-efficient image loading with format detection
- ImageProcessor: Core image transformation operations
- ImageValidator: Format and quality validation
- CoordinateTransformer: Bounding box coordinate conversions

Text Operations:
- TextCleaner: Basic text cleaning operations for OCR output
- UnicodeNormalizer: Unicode normalization and encoding fixes
- TextValidator: Text validation and quality scoring
- TextMerger: Text merging from multiple sources

System Utilities:
- ModelCache: Model loading and caching with LRU eviction
- OCRLogger: Structured logging and performance tracking
"""

# Image utilities
from .image_utils import (
    ImageLoader, ImageProcessor, ImageValidator, 
    CoordinateTransformer, ImageMemoryManager
)

# Text utilities  
from .text_utils import (
    TextCleaner, UnicodeNormalizer, TextValidator, TextMerger
)

# Model utilities
from .model_utils import (
    ModelCache, ModelDownloader, ModelVersionManager, 
    ModelLoader, cached_model_load
)

# Logging utilities
from .logger import (
    OCRLogger, ProcessingStageTimer, MetricsCollector,
    OCRDebugLogger, LogConfig
)

__all__ = [
    # Image utilities
    'ImageLoader',
    'ImageProcessor', 
    'ImageValidator',
    'CoordinateTransformer',
    'ImageMemoryManager',
    
    # Text utilities
    'TextCleaner',
    'UnicodeNormalizer', 
    'TextValidator',
    'TextMerger',
    
    # Model utilities
    'ModelCache',
    'ModelDownloader',
    'ModelVersionManager',
    'ModelLoader',
    'cached_model_load',
    
    # Logging utilities
    'OCRLogger',
    'ProcessingStageTimer',
    'MetricsCollector', 
    'OCRDebugLogger',
    'LogConfig'
]
