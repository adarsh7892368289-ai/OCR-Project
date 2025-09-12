# src/advanced_ocr/engines/base_engine.py - Production Base OCR Engine

"""
Production-Grade Base OCR Engine
Provides abstract interface and common functionality for all OCR engines
Integrates with advanced_ocr project structure and result classes
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional, Union
import cv2
import numpy as np
import time
import logging
from pathlib import Path
import json
from enum import Enum

# Import from project structure
from ..results import OCRResult, TextRegion, BoundingBox
from ..utils.logger import get_logger
from ..utils.image_utils import validate_image, normalize_image

class OCREngineType(Enum):
    """Supported OCR engine types"""
    TESSERACT = "tesseract"
    EASYOCR = "easyocr" 
    PADDLEOCR = "paddleocr"
    TROCR = "trocr"
    CRAFT = "craft"
    PPOCR = "ppocr"

class TextType(Enum):
    """Text content classification"""
    PRINTED = "printed"
    HANDWRITTEN = "handwritten"
    MIXED = "mixed"
    UNKNOWN = "unknown"
    
    # Structural types for compatibility
    PARAGRAPH = "paragraph"
    LINE = "line"
    WORD = "word"
    BLOCK = "block"
    TITLE = "title"
    HEADER = "header"

class QualityLevel(Enum):
    """Image quality assessment levels"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    UNUSABLE = "unusable"

class BaseOCREngine(ABC):
    """
    Abstract base class for all OCR engines in the advanced_ocr system
    
    This class provides:
    - Common interface for all OCR engines
    - Error handling and logging
    - Performance monitoring
    - Image validation and preprocessing
    - Result standardization
    """
    
    def __init__(self, name: str = "", config: Optional[Dict[str, Any]] = None):
        """
        Initialize base OCR engine
        
        Args:
            name: Engine identifier name
            config: Engine configuration dictionary
        """
        self.name = name or self.__class__.__name__.replace('Engine', '').lower()
        self.config = config or {}
        self.is_initialized = False
        self.model_loaded = False
        
        # Setup logging
        self.logger = get_logger(f"engines.{self.name}")
        
        # Engine capabilities - to be overridden by subclasses
        self.supports_handwriting = False
        self.supports_multiple_languages = False
        self.supports_orientation_detection = False
        self.supports_structure_analysis = False
        self.supports_table_detection = False
        self.supported_languages = ['en']
        
        # Image processing limits
        self.max_image_size = self.config.get('max_image_size', (4096, 4096))
        self.min_image_size = self.config.get('min_image_size', (32, 32))
        
        # Performance settings
        self.batch_size = self.config.get('batch_size', 1)
        self.use_gpu = self.config.get('use_gpu', False)
        self.num_threads = self.config.get('num_threads', 1)
        self.timeout = self.config.get('timeout', 30.0)  # seconds
        
        # Processing statistics
        self.stats = {
            'total_processed': 0,
            'total_time': 0.0,
            'avg_confidence': 0.0,
            'errors': 0,
            'successful_extractions': 0,
            'avg_processing_time': 0.0,
            'success_rate': 0.0
        }
        
        self.logger.info(f"Initialized {self.name} engine")
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the OCR engine (load models, setup dependencies)
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        pass
    
    @abstractmethod
    def process_image(self, image: np.ndarray, **kwargs) -> OCRResult:
        """
        Process an image and extract text
        
        Args:
            image: Input image as numpy array
            **kwargs: Additional processing parameters
            
        Returns:
            OCRResult: Structured OCR results
        """
        pass
    
    @abstractmethod
    def get_supported_languages(self) -> List[str]:
        """
        Get list of languages supported by this engine
        
        Returns:
            List[str]: Language codes (e.g., ['en', 'es', 'fr'])
        """
        pass
    
    def is_available(self) -> bool:
        """
        Check if engine is available and ready for use
        
        Returns:
            bool: True if engine is available, False otherwise
        """
        try:
            return self.initialize() if not self.is_initialized else True
        except Exception as e:
            self.logger.error(f"Engine availability check failed: {e}")
            return False
    
    def extract_text(self, image: np.ndarray, **kwargs) -> OCRResult:
        """
        Main public interface for text extraction
        
        Args:
            image: Input image as numpy array
            **kwargs: Additional processing parameters
            
        Returns:
            OCRResult: Complete OCR results with metadata
        """
        start_time = time.time()
        
        try:
            # Validate inputs
            if not self._validate_inputs(image):
                raise ValueError("Invalid image input")
            
            # Ensure engine is initialized
            if not self.is_initialized and not self.initialize():
                raise RuntimeError(f"Failed to initialize {self.name} engine")
            
            # Validate image format and dimensions
            if not self._validate_image(image):
                raise ValueError("Invalid image format or dimensions")
            
            # Preprocess image if needed
            processed_image = self._preprocess_image(image, **kwargs)
            
            # Extract text using engine-specific implementation
            result = self.process_image(processed_image, **kwargs)
            
            # Post-process results
            result = self._postprocess_result(result, start_time)
            
            # Update statistics
            self._update_statistics(result, time.time() - start_time, success=True)
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Text extraction failed in {self.name}: {e}")
            
            # Update error statistics
            self._update_statistics(None, processing_time, success=False)
            
            # Return empty result with error information
            return self._create_error_result(str(e), processing_time)
    
    def extract_text_batch(self, images: List[np.ndarray], **kwargs) -> List[OCRResult]:
        """
        Process multiple images in batch
        
        Args:
            images: List of input images
            **kwargs: Processing parameters
            
        Returns:
            List[OCRResult]: Results for each image
        """
        results = []
        
        for i, image in enumerate(images):
            try:
                result = self.extract_text(image, **kwargs)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Failed to process image {i}: {e}")
                results.append(self._create_error_result(str(e), 0.0))
        
        return results
    
    def _validate_inputs(self, image: np.ndarray) -> bool:
        """Validate input parameters"""
        if image is None:
            self.logger.error("Input image is None")
            return False
        
        if not isinstance(image, np.ndarray):
            self.logger.error("Input must be numpy array")
            return False
            
        return True
    
    def _validate_image(self, image: np.ndarray) -> bool:
        """
        Validate image format and dimensions
        
        Args:
            image: Input image array
            
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            return validate_image(image, self.min_image_size, self.max_image_size)
        except Exception as e:
            self.logger.error(f"Image validation failed: {e}")
            return False
    
    def _preprocess_image(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Apply common preprocessing steps
        
        Args:
            image: Input image
            **kwargs: Processing parameters
            
        Returns:
            np.ndarray: Preprocessed image
        """
        try:
            # Normalize image format and color space
            processed_image = normalize_image(image)
            
            # Apply engine-specific preprocessing
            processed_image = self.preprocess_image(processed_image, **kwargs)
            
            return processed_image
            
        except Exception as e:
            self.logger.warning(f"Preprocessing failed, using original image: {e}")
            return image
    
    def preprocess_image(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """
        Engine-specific preprocessing - override in subclasses
        
        Args:
            image: Input image
            **kwargs: Processing parameters
            
        Returns:
            np.ndarray: Preprocessed image
        """
        return image
    
    def _postprocess_result(self, result: OCRResult, start_time: float) -> OCRResult:
        """
        Apply common post-processing to results
        
        Args:
            result: Raw OCR result
            start_time: Processing start time
            
        Returns:
            OCRResult: Enhanced result with metadata
        """
        # Set engine metadata
        result.engine_name = self.name
        result.processing_time = time.time() - start_time
        
        # Add engine capabilities to metadata
        if not result.metadata:
            result.metadata = {}
            
        result.metadata.update({
            'engine_type': self.name,
            'supports_handwriting': self.supports_handwriting,
            'supports_multiple_languages': self.supports_multiple_languages,
            'processing_timestamp': time.time()
        })
        
        return result
    
    def _create_error_result(self, error_message: str, processing_time: float) -> OCRResult:
        """
        Create OCR result for error cases
        
        Args:
            error_message: Error description
            processing_time: Time spent processing
            
        Returns:
            OCRResult: Error result
        """
        return OCRResult(
            text="",
            confidence=0.0,
            processing_time=processing_time,
            engine_name=self.name,
            metadata={
                'error': error_message,
                'success': False,
                'error_timestamp': time.time()
            }
        )
    
    def _update_statistics(self, result: Optional[OCRResult], processing_time: float, success: bool):
        """
        Update engine performance statistics
        
        Args:
            result: OCR result (None if failed)
            processing_time: Time taken for processing
            success: Whether processing succeeded
        """
        self.stats['total_processed'] += 1
        self.stats['total_time'] += processing_time
        
        if success and result:
            self.stats['successful_extractions'] += 1
            # Update running average confidence
            total_confidence = (self.stats['avg_confidence'] * 
                              (self.stats['successful_extractions'] - 1) + 
                              result.confidence)
            self.stats['avg_confidence'] = total_confidence / self.stats['successful_extractions']
        else:
            self.stats['errors'] += 1
        
        # Update derived statistics
        if self.stats['total_processed'] > 0:
            self.stats['avg_processing_time'] = (
                self.stats['total_time'] / self.stats['total_processed']
            )
            self.stats['success_rate'] = (
                self.stats['successful_extractions'] / self.stats['total_processed']
            )
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive engine performance statistics
        
        Returns:
            Dict[str, Any]: Performance statistics
        """
        stats = self.stats.copy()
        stats.update({
            'engine_name': self.name,
            'is_initialized': self.is_initialized,
            'supported_languages': self.supported_languages,
            'capabilities': {
                'handwriting': self.supports_handwriting,
                'multilingual': self.supports_multiple_languages,
                'orientation_detection': self.supports_orientation_detection,
                'structure_analysis': self.supports_structure_analysis,
                'table_detection': self.supports_table_detection
            }
        })
        return stats
    
    def reset_statistics(self):
        """Reset performance statistics"""
        self.stats = {
            'total_processed': 0,
            'total_time': 0.0,
            'avg_confidence': 0.0,
            'errors': 0,
            'successful_extractions': 0,
            'avg_processing_time': 0.0,
            'success_rate': 0.0
        }
        self.logger.info(f"Reset statistics for {self.name} engine")
    
    def cleanup(self):
        """
        Cleanup engine resources
        Override in subclasses for engine-specific cleanup
        """
        try:
            self.logger.info(f"Cleaning up {self.name} engine")
            self.is_initialized = False
            self.model_loaded = False
        except Exception as e:
            self.logger.error(f"Cleanup failed for {self.name}: {e}")
    
    def get_engine_info(self) -> Dict[str, Any]:
        """
        Get comprehensive engine information
        
        Returns:
            Dict[str, Any]: Engine information and capabilities
        """
        return {
            'name': self.name,
            'type': self.__class__.__name__,
            'is_initialized': self.is_initialized,
            'model_loaded': self.model_loaded,
            'supported_languages': self.supported_languages,
            'capabilities': {
                'handwriting': self.supports_handwriting,
                'multilingual': self.supports_multiple_languages,
                'orientation_detection': self.supports_orientation_detection,
                'structure_analysis': self.supports_structure_analysis,
                'table_detection': self.supports_table_detection
            },
            'configuration': {
                'max_image_size': self.max_image_size,
                'min_image_size': self.min_image_size,
                'batch_size': self.batch_size,
                'use_gpu': self.use_gpu,
                'timeout': self.timeout
            },
            'statistics': self.get_statistics()
        }
    
    def __enter__(self):
        """Context manager entry"""
        if not self.initialize():
            raise RuntimeError(f"Failed to initialize {self.name} engine")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()
    
    def __str__(self) -> str:
        return f"{self.name}Engine(initialized={self.is_initialized})"
    
    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(name='{self.name}', "
                f"initialized={self.is_initialized}, "
                f"languages={self.supported_languages})")

# Backward compatibility alias
OCREngine = BaseOCREngine

# Export classes
__all__ = [
    'BaseOCREngine',
    'OCREngine',
    'OCREngineType', 
    'TextType',
    'QualityLevel'
]