# src/advanced_ocr/engines/base_engine.py
"""
Advanced OCR Base Engine Module - FIXED VERSION

This module provides the abstract base class and common functionality for all OCR
engines in the advanced OCR system. It defines the standard interface that all
engines must implement while providing basic error handling, logging, and metrics
tracking.

The module focuses on:
- Defining the standard OCR engine interface
- Providing basic error handling and logging infrastructure
- Tracking essential performance metrics
- Managing engine lifecycle (initialization, cleanup)
- Ensuring consistent result formatting

Classes:
    EngineStatus: Enum representing engine operational states
    EngineMetrics: Dataclass capturing essential engine performance metrics
    BaseOCREngine: Abstract base class defining the OCR engine interface

Example:
    >>> class MyEngine(BaseOCREngine):
    ...     def _extract_implementation(self, image, text_regions):
    ...         # Engine-specific OCR logic here
    ...         return OCRResult(text="extracted text", confidence=0.95)
    >>>
    >>> engine = MyEngine(config)
    >>> engine.initialize()
    >>> result = engine.extract(image, text_regions)

"""

import time
from abc import ABC, abstractmethod
from typing import List, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np

from ..results import OCRResult, TextRegion
from ..config import OCRConfig
from ..utils.logger import OCRLogger


class EngineStatus(Enum):
    """Engine operational status"""
    UNINITIALIZED = "uninitialized"
    READY = "ready"
    PROCESSING = "processing"
    ERROR = "error"


@dataclass
class EngineMetrics:
    """Essential engine performance metrics"""
    total_extractions: int = 0
    successful_extractions: int = 0
    failed_extractions: int = 0
    avg_processing_time: float = 0.0
    avg_confidence: float = 0.0
    total_processing_time: float = 0.0
    last_error: Optional[str] = None
    
    @property
    def success_rate(self) -> float:
        """Calculate extraction success rate"""
        if self.total_extractions == 0:
            return 0.0
        return self.successful_extractions / self.total_extractions


class BaseOCREngine(ABC):
    """
    Abstract base class for all OCR engines
    
    CORRECT RESPONSIBILITIES:
    - Define standard engine interface 
    - Provide basic error handling and logging
    - Track simple performance metrics
    - Process ALREADY PREPROCESSED images and text regions
    - Return raw OCRResult (no postprocessing)
    
    """
    
    def __init__(self, config: OCRConfig):
        """Initialize base OCR engine"""
        self.config = config
        self.logger = OCRLogger(self.__class__.__name__)
        
        # Engine identification
        self.engine_name = self.__class__.__name__.lower().replace('engine', '').replace('ocr', '')
        
        # Simple state management
        self._status = EngineStatus.UNINITIALIZED
        self._metrics = EngineMetrics()
        
        # Basic configuration
        engine_config = getattr(config.engines, self.engine_name, {})
        self.timeout = engine_config.get('timeout', 30.0)
        self.min_confidence = engine_config.get('min_confidence', 0.0)
        
        self.logger.info(f"Initialized {self.engine_name} engine")
    
    def extract(self, image: np.ndarray, text_regions: List[TextRegion]) -> OCRResult:
        """
        Main extraction method - SIMPLIFIED
        
        Pipeline integration:
        1. Receives PREPROCESSED image from engine_coordinator.py
        2. Receives detected text regions from engine_coordinator.py
        3. Calls engine-specific implementation
        4. Returns RAW result (no postprocessing)
        
        Args:
            image: Preprocessed image (from image_processor.py)
            text_regions: Detected text regions (from text_detector.py)
            
        Returns:
            OCRResult with raw extraction results
        """
        start_time = time.time()
        self._status = EngineStatus.PROCESSING
        
        try:
            # Basic input validation only
            if image is None or image.size == 0:
                raise ValueError(f"{self.engine_name}: Invalid image")
            
            if text_regions is None:
                text_regions = []
            
            # Engine-specific extraction
            result = self._extract_implementation(image, text_regions)
            
            # Basic result validation
            if result is None:
                raise RuntimeError(f"{self.engine_name}: No result returned")
            
            # Add basic metadata
            processing_time = time.time() - start_time
            result.processing_time = processing_time
            result.engine_name = self.engine_name
            
            # Update success metrics
            self._update_success_metrics(result, processing_time)
            
            self.logger.debug(
                f"{self.engine_name}: extracted {len(result.text)} chars, "
                f"confidence {result.confidence:.3f}, time {processing_time:.3f}s"
            )
            
            return result
            
        except Exception as e:
            # Simple error handling
            processing_time = time.time() - start_time
            error_result = self._create_error_result(str(e), processing_time)
            self._update_error_metrics(str(e))
            
            self.logger.error(f"{self.engine_name} extraction failed: {e}")
            return error_result
            
        finally:
            self._status = EngineStatus.READY
    
    @abstractmethod
    def _extract_implementation(self, image: np.ndarray, text_regions: List[TextRegion]) -> OCRResult:
        """
        Engine-specific extraction implementation
        
        Must be implemented by concrete engines.
        Should contain only the core OCR logic for that engine.
        
        Args:
            image: Preprocessed image ready for OCR
            text_regions: Text regions to process (can be empty for full-page OCR)
            
        Returns:
            OCRResult with raw extraction results
        """
        pass
    
    def _create_error_result(self, error_msg: str, processing_time: float) -> OCRResult:
        """Create error result for failed extractions - FIXED VERSION"""
        return OCRResult(
            text="",
            confidence=0.0,
            processing_time=processing_time,
            engine_name=self.engine_name,
            success=False,
            error_message=error_msg,
            metadata={
                'error': error_msg,
                'extraction_failed': True
            }
        )
    
    def _update_success_metrics(self, result: OCRResult, processing_time: float):
        """Update metrics for successful extraction"""
        self._metrics.total_extractions += 1
        self._metrics.successful_extractions += 1
        self._metrics.total_processing_time += processing_time
        
        # Simple moving average
        n = self._metrics.total_extractions
        self._metrics.avg_processing_time = (
            (self._metrics.avg_processing_time * (n - 1) + processing_time) / n
        )
        self._metrics.avg_confidence = (
            (self._metrics.avg_confidence * (n - 1) + result.confidence) / n
        )
    
    def _update_error_metrics(self, error_msg: str):
        """Update metrics for failed extraction"""
        self._metrics.total_extractions += 1
        self._metrics.failed_extractions += 1
        self._metrics.last_error = error_msg
    
    def initialize(self):
        """Initialize engine resources if needed"""
        if self._status == EngineStatus.UNINITIALIZED:
            try:
                self._initialize_implementation()
                self._status = EngineStatus.READY
                self.logger.info(f"{self.engine_name} initialized successfully")
            except Exception as e:
                self._status = EngineStatus.ERROR
                self.logger.error(f"{self.engine_name} initialization failed: {e}")
                raise
    
    def _initialize_implementation(self):
        """Override for engine-specific initialization"""
        pass
    
    def cleanup(self):
        """Basic cleanup"""
        try:
            self._cleanup_implementation()
            self.logger.info(f"{self.engine_name} cleaned up")
        except Exception as e:
            self.logger.error(f"{self.engine_name} cleanup failed: {e}")
    
    def _cleanup_implementation(self):
        """Override for engine-specific cleanup"""
        pass
    
    # Simple public API
    def get_metrics(self) -> EngineMetrics:
        """Get current metrics"""
        return self._metrics
    
    def get_status(self) -> EngineStatus:
        """Get current status"""
        return self._status
    
    def is_ready(self) -> bool:
        """Check if engine is ready for processing"""
        return self._status == EngineStatus.READY
    
    def __str__(self) -> str:
        """String representation"""
        return f"{self.engine_name}Engine(status={self._status.value})"