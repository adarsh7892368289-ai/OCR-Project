# =============================================================================
# src/advanced_ocr/engines/__init__.py - ENGINE LAYER API
# =============================================================================
"""
OCR Engines Package - Multi-Engine Text Extraction Layer
========================================================

This package contains all OCR engine implementations and coordination logic.

Available Engines:
- TesseractEngine: Traditional OCR with excellent printed text support
- PaddleOCREngine: Modern deep learning OCR with layout awareness
- EasyOCREngine: GPU-optimized OCR with multi-language support
- TrOCREngine: Transformer-based OCR optimized for handwriting

Engine Coordination:
- EngineCoordinator: Intelligent engine selection and result coordination
- BaseOCREngine: Abstract base class for all engines
"""

from .base_engine import BaseOCREngine, EngineStatus, EngineMetrics
from .engine_coordinator import EngineCoordinator, EngineStrategy, EngineSelection

# Import engines with error handling (they might have heavy dependencies)
try:
    from .tesseract_engine import TesseractEngine
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    from .paddleocr_engine import PaddleOCREngine
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False

try:
    from .easyocr_engine import EasyOCREngine
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

try:
    from .trocr_engine import TrOCREngine
    TROCR_AVAILABLE = True
except ImportError:
    TROCR_AVAILABLE = False

# Engine availability info
ENGINE_AVAILABILITY = {
    'tesseract': TESSERACT_AVAILABLE,
    'paddleocr': PADDLEOCR_AVAILABLE,
    'easyocr': EASYOCR_AVAILABLE,
    'trocr': TROCR_AVAILABLE
}

def get_available_engines():
    """Get list of available engine names."""
    return [name for name, available in ENGINE_AVAILABILITY.items() if available]

def get_engine_class(engine_name):
    """Get engine class by name."""
    engine_map = {}
    if TESSERACT_AVAILABLE:
        engine_map['tesseract'] = TesseractEngine
    if PADDLEOCR_AVAILABLE:
        engine_map['paddleocr'] = PaddleOCREngine
    if EASYOCR_AVAILABLE:
        engine_map['easyocr'] = EasyOCREngine
    if TROCR_AVAILABLE:
        engine_map['trocr'] = TrOCREngine
    
    return engine_map.get(engine_name.lower())

__all__ = [
    # Base classes
    'BaseOCREngine',
    'EngineStatus',
    'EngineMetrics',
    
    # Coordination
    'EngineCoordinator',
    'EngineStrategy',
    'EngineSelection',
    
    # Utility functions
    'get_available_engines',
    'get_engine_class',
    'ENGINE_AVAILABILITY'
]

# Add available engines to exports
if TESSERACT_AVAILABLE:
    __all__.append('TesseractEngine')
if PADDLEOCR_AVAILABLE:
    __all__.append('PaddleOCREngine')
if EASYOCR_AVAILABLE:
    __all__.append('EasyOCREngine')
if TROCR_AVAILABLE:
    __all__.append('TrOCREngine')
