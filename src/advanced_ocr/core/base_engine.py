# src/advanced_ocr/core/base_engine.py
"""
Abstract base class for OCR engines.
Defines the interface that all OCR engines must implement.

Responsibility: Define standardized interface for OCR engines ONLY
- Standardize OCR engine interface
- Provide common initialization pattern  
- Define required methods all engines must implement
- Basic validation and logging support

"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np
import logging

# Import from our centralized types
from ..types import OCRResult

class BaseOCREngine(ABC):
    """
    Abstract base class for all OCR engines.
    
    All engines must inherit from this and implement the required methods.
    This ensures consistency across PaddleOCR, EasyOCR, Tesseract, and TrOCR.
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize OCR engine with configuration.
        
        Args:
            name: Engine name identifier (e.g., "PaddleOCR", "EasyOCR")
            config: Engine-specific configuration options
        """
        self.name = name
        self.config = config or {}
        self.is_initialized = False
        self.logger = logging.getLogger(f"advanced_ocr.engines.{name.lower()}")
        
        # Performance tracking
        self.processing_stats = {
            'total_processed': 0,
            'successful_extractions': 0,
            'total_time': 0.0,
            'errors': 0
        }
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the OCR engine and load models.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the OCR engine is available for use.
        
        Returns:
            bool: True if engine is available and ready, False otherwise
        """
        pass
    
    @abstractmethod
    def extract_text(self, image: np.ndarray) -> OCRResult:
        """
        Extract text from an image using this OCR engine.
        
        This is the CORE responsibility of every OCR engine.
        
        Args:
            image: Preprocessed image as numpy array (RGB/BGR/Grayscale)
            
        Returns:
            OCRResult: Raw OCR result with text regions and bounding boxes
            
        Important:
            - Image is already preprocessed by ImageEnhancer
            - Should ONLY do OCR text extraction
            - Should NOT do layout reconstruction (pipeline handles this)
            - Should NOT do image enhancement (ImageEnhancer handles this)
            - Should return raw text regions with coordinates
        """
        pass
    
    @abstractmethod
    def get_supported_languages(self) -> List[str]:
        """
        Get list of languages supported by this engine.
        
        Returns:
            List[str]: List of language codes (e.g., ['en', 'es', 'fr'])
        """
        pass
    
    # === COMMON UTILITY METHODS ===
    
    def validate_image(self, image: np.ndarray) -> bool:
        """
        Basic image validation - common for all engines.
        
        Args:
            image: Input image array
            
        Returns:
            bool: True if image is valid for OCR processing
        """
        if image is None or image.size == 0:
            self.logger.error("Image is None or empty")
            return False
        
        if len(image.shape) not in [2, 3]:  # Must be 2D or 3D
            self.logger.error(f"Invalid image shape: {image.shape}")
            return False
            
        if len(image.shape) == 3 and image.shape[2] not in [1, 3, 4]:  # Valid channels
            self.logger.error(f"Invalid number of channels: {image.shape[2]}")
            return False
        
        # Check minimum dimensions
        if image.shape[0] < 10 or image.shape[1] < 10:
            self.logger.error(f"Image too small: {image.shape}")
            return False
            
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get engine performance statistics"""
        stats = self.processing_stats.copy()
        
        if stats['total_processed'] > 0:
            stats['average_time'] = stats['total_time'] / stats['total_processed']
            stats['success_rate'] = stats['successful_extractions'] / stats['total_processed']
            stats['error_rate'] = stats['errors'] / stats['total_processed']
        else:
            stats['average_time'] = 0.0
            stats['success_rate'] = 0.0
            stats['error_rate'] = 0.0
            
        return stats
    
    def reset_stats(self):
        """Reset performance statistics"""
        self.processing_stats = {
            'total_processed': 0,
            'successful_extractions': 0,
            'total_time': 0.0,
            'errors': 0
        }
    
    def cleanup(self):
        """
        Cleanup engine resources.
        Default implementation - engines can override if needed.
        """
        self.is_initialized = False
        self.logger.debug(f"Cleaned up {self.name} engine")
    
    def __str__(self) -> str:
        return f"{self.name}(initialized={self.is_initialized})"
    
    def __repr__(self) -> str:
        return self.__str__()
    
BaseEngine=BaseOCREngine