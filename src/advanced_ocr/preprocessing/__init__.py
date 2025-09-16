
# =============================================================================
# src/advanced_ocr/preprocessing/__init__.py - PREPROCESSING API
# =============================================================================
"""
Preprocessing Package - AI-Powered Image Enhancement Pipeline
============================================================

This package contains all image preprocessing and analysis components:

Main Orchestrator:
- ImageProcessor: Unified preprocessing pipeline coordinator

Specialized Analyzers:
- QualityAnalyzer: Image quality assessment and metrics
- ContentClassifier: Content type detection (handwritten/printed/mixed)
- TextDetector: Advanced text region detection with CRAFT

All preprocessing is AI-powered and optimized for OCR accuracy improvement.
"""

from .image_processor import ImageProcessor, PreprocessingResult, EnhancementStrategy
from .quality_analyzer import QualityAnalyzer, QualityLevel, QualityMetrics
from .content_classifier import ContentClassifier, ContentClassification
from .text_detector import TextDetector, CRAFTDetector, FastTextDetector

__all__ = [
    # Main coordinator
    'ImageProcessor',
    'PreprocessingResult',
    'EnhancementStrategy',
    
    # Quality analysis
    'QualityAnalyzer',
    'QualityLevel',
    'QualityMetrics',
    
    # Content classification
    'ContentClassifier',
    'ContentClassification',
    
    # Text detection
    'TextDetector',
    'CRAFTDetector', 
    'FastTextDetector'
]




