# src/advanced_ocr/core/__init__.py
"""
Core OCR components.

Exports:
- BaseOCREngine: Abstract base class for all OCR engines
- EngineManager: Manages OCR engines and coordinates their use
"""

from .base_engine import BaseEngine
from .engine_manager import EngineManager

__all__ = [
    'BaseEngine',
    'EngineManager'
]