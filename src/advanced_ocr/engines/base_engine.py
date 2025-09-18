"""
Base engine interface for OCR implementations.
Provides standard interface, error handling, and performance tracking.
"""

import time
import logging
import traceback
from abc import ABC, abstractmethod
from typing import Optional, Any, Dict, List
from enum import Enum
import threading
from contextlib import contextmanager

from ..results import OCRResult, ProcessingMetrics, ConfidenceMetrics
from ..config import EngineConfig


class EngineStatus(Enum):
    """Engine operational status."""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"
    DISABLED = "disabled"


class BaseOCREngine(ABC):
    """
    Abstract base class for all OCR engines.
    Provides standard interface, error handling, and performance tracking.
    """
    
    def __init__(self, config: EngineConfig):
        """
        Initialize base engine.
        
        Args:
            config: Engine-specific configuration
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.config.name}")
        
        # Engine state
        self._status = EngineStatus.UNINITIALIZED
        self._status_lock = threading.RLock()
        self._last_error: Optional[str] = None
        self._initialization_time: Optional[float] = None
        
        # Performance metrics
        self._total_extractions = 0
        self._total_processing_time = 0.0
        self._successful_extractions = 0
        self._failed_extractions = 0
        
        # Model reference (to be set by subclasses)
        self._model: Optional[Any] = None
        
        self.logger.info(f"Initialized {self.config.name} engine with config: {self.config.name}")
    
    @property
    def name(self) -> str:
        """Get engine name."""
        return self.config.name
    
    @property
    def status(self) -> EngineStatus:
        """Get current engine status."""
        with self._status_lock:
            return self._status
    
    @property
    def is_ready(self) -> bool:
        """Check if engine is ready for processing."""
        return self.status == EngineStatus.READY
    
    @property
    def is_busy(self) -> bool:
        """Check if engine is currently busy."""
        return self.status == EngineStatus.BUSY
    
    @property
    def has_error(self) -> bool:
        """Check if engine has error status."""
        return self.status == EngineStatus.ERROR
    
    @property
    def last_error(self) -> Optional[str]:
        """Get last error message."""
        return self._last_error
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get engine performance statistics."""
        avg_processing_time = (
            self._total_processing_time / self._total_extractions 
            if self._total_extractions > 0 else 0.0
        )
        
        success_rate = (
            self._successful_extractions / self._total_extractions 
            if self._total_extractions > 0 else 0.0
        )
        
        return {
            'engine_name': self.name,
            'status': self.status.value,
            'total_extractions': self._total_extractions,
            'successful_extractions': self._successful_extractions,
            'failed_extractions': self._failed_extractions,
            'success_rate': success_rate,
            'total_processing_time': self._total_processing_time,
            'average_processing_time': avg_processing_time,
            'initialization_time': self._initialization_time,
            'last_error': self._last_error
        }
    
    def initialize(self) -> bool:
        """
        Initialize the engine and load models.
        
        Returns:
            True if initialization successful, False otherwise
        """
        with self._status_lock:
            if self._status == EngineStatus.READY:
                self.logger.debug(f"Engine {self.name} already initialized")
                return True
            
            if self._status == EngineStatus.INITIALIZING:
                self.logger.warning(f"Engine {self.name} already initializing")
                return False
            
            self._status = EngineStatus.INITIALIZING
        
        try:
            start_time = time.time()
            self.logger.info(f"Initializing {self.name} engine...")
            
            # Call subclass-specific initialization
            success = self._initialize_engine()
            
            initialization_time = time.time() - start_time
            self._initialization_time = initialization_time
            
            with self._status_lock:
                if success:
                    self._status = EngineStatus.READY
                    self._last_error = None
                    self.logger.info(
                        f"Successfully initialized {self.name} engine "
                        f"in {initialization_time:.2f}s"
                    )
                else:
                    self._status = EngineStatus.ERROR
                    self._last_error = "Initialization failed"
                    self.logger.error(f"Failed to initialize {self.name} engine")
            
            return success
            
        except Exception as e:
            error_msg = f"Engine initialization error: {str(e)}"
            self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
            
            with self._status_lock:
                self._status = EngineStatus.ERROR
                self._last_error = error_msg
            
            return False
    
    @abstractmethod
    def _initialize_engine(self) -> bool:
        """
        Subclass-specific initialization logic.
        
        Returns:
            True if successful, False otherwise
        """
        pass
    
    def extract(self, image: Any) -> Optional[OCRResult]:
        """
        Extract text from image with error handling and metrics.
        
        Args:
            image: Input image (PIL Image, numpy array, or file path)
            
        Returns:
            OCRResult or None if extraction failed
        """
        # Check engine status
        if not self.is_ready:
            if self._status == EngineStatus.UNINITIALIZED:
                self.logger.info(f"Auto-initializing {self.name} engine")
                if not self.initialize():
                    return None
            else:
                self.logger.error(f"Engine {self.name} not ready: {self.status.value}")
                return None
        
        # Set busy status
        with self._status_lock:
            if self._status == EngineStatus.BUSY:
                self.logger.warning(f"Engine {self.name} is busy")
                return None
            self._status = EngineStatus.BUSY
        
        try:
            with self._processing_context():
                start_time = time.time()
                self.logger.debug(f"Starting text extraction with {self.name}")
                
                # Call subclass-specific extraction
                result = self._extract_text(image)
                
                processing_time = time.time() - start_time
                
                if result:
                    # Add engine information to result
                    result.processing_time = processing_time
                    result.engine_info = {
                        'engine_name': self.name,
                        'engine_version': getattr(self, 'version', 'unknown'),
                        'processing_time': processing_time,
                        'configuration': {
                            'gpu_enabled': self.config.gpu_enabled,
                            'timeout': self.config.timeout,
                            'min_confidence': self.config.min_confidence
                        }
                    }
                    
                    # Add processing metrics
                    metrics = ProcessingMetrics(
                        stage_name=f"{self.name}_extraction",
                        duration=processing_time,
                        metadata={
                            'engine': self.name,
                            'image_processed': True,
                            'confidence_threshold': self.config.min_confidence
                        }
                    )
                    metrics.finish()
                    result.add_processing_metric(metrics)
                    
                    # Update success stats
                    self._successful_extractions += 1
                    self.logger.info(
                        f"Successfully extracted text with {self.name} "
                        f"in {processing_time:.2f}s"
                    )
                else:
                    # Update failure stats
                    self._failed_extractions += 1
                    self.logger.warning(f"Text extraction failed with {self.name}")
                
                # Update performance stats
                self._total_extractions += 1
                self._total_processing_time += processing_time
                
                return result
                
        except Exception as e:
            error_msg = f"Extraction error: {str(e)}"
            self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
            
            # Update failure stats
            self._failed_extractions += 1
            self._total_extractions += 1
            
            with self._status_lock:
                self._last_error = error_msg
            
            return None
        
        finally:
            # Reset to ready status
            with self._status_lock:
                if self._status == EngineStatus.BUSY:
                    self._status = EngineStatus.READY
    
    @abstractmethod
    def _extract_text(self, image: Any) -> Optional[OCRResult]:
        """
        Subclass-specific text extraction logic.
        
        Args:
            image: Input image
            
        Returns:
            OCRResult or None if extraction failed
        """
        pass
    
    @contextmanager
    def _processing_context(self):
        """Context manager for processing operations with timeout."""
        if self.config.timeout > 0:
            # Set up timeout handling here if needed
            pass
        
        try:
            yield
        except Exception as e:
            self.logger.error(f"Processing error in {self.name}: {e}")
            raise
    
    def _validate_image(self, image: Any) -> bool:
        """
        Validate input image.
        
        Args:
            image: Input image to validate
            
        Returns:
            True if valid, False otherwise
        """
        if image is None:
            self.logger.error("Input image is None")
            return False
        
        try:
            # Basic validation - subclasses can override for specific checks
            from PIL import Image
            import numpy as np
            
            if isinstance(image, str):
                # File path
                from pathlib import Path
                if not Path(image).exists():
                    self.logger.error(f"Image file not found: {image}")
                    return False
            elif isinstance(image, Image.Image):
                # PIL Image
                if image.size[0] == 0 or image.size[1] == 0:
                    self.logger.error("Image has zero dimensions")
                    return False
            elif isinstance(image, np.ndarray):
                # Numpy array
                if image.size == 0:
                    self.logger.error("Image array is empty")
                    return False
            else:
                self.logger.error(f"Unsupported image type: {type(image)}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Image validation error: {e}")
            return False
    
    def _create_default_confidence(self, raw_confidence: Optional[float] = None) -> ConfidenceMetrics:
        """
        Create default confidence metrics.
        
        Args:
            raw_confidence: Raw confidence score from engine
            
        Returns:
            ConfidenceMetrics object
        """
        # Use raw confidence or default to minimum threshold
        base_confidence = raw_confidence if raw_confidence is not None else self.config.min_confidence
        
        return ConfidenceMetrics(
            character_level=base_confidence,
            word_level=base_confidence,
            line_level=base_confidence,
            layout_level=base_confidence * 0.9,  # Slightly lower for layout
            text_quality=base_confidence * 0.8,   # Lower for text quality
            spatial_quality=base_confidence * 0.85, # Moderate for spatial
            engine_name=self.name,
            raw_confidence=raw_confidence
        )
    
    def shutdown(self) -> None:
        """Clean shutdown of the engine."""
        with self._status_lock:
            if self._status == EngineStatus.DISABLED:
                return
            
            self.logger.info(f"Shutting down {self.name} engine")
            
            try:
                # Call subclass-specific cleanup
                self._cleanup_engine()
            except Exception as e:
                self.logger.error(f"Cleanup error: {e}")
            finally:
                self._status = EngineStatus.DISABLED
                self._model = None
    
    def _cleanup_engine(self) -> None:
        """Subclass-specific cleanup logic."""
        pass
    
    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.shutdown()
        except:
            pass  # Ignore errors during destruction
    
    def __str__(self) -> str:
        """String representation of engine."""
        return f"{self.name}Engine(status={self.status.value})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (
            f"{self.__class__.__name__}("
            f"name='{self.name}', "
            f"status='{self.status.value}', "
            f"extractions={self._total_extractions}, "
            f"success_rate={self._successful_extractions/max(1,self._total_extractions):.2f}"
            f")"
        )