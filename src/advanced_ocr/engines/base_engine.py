"""
Abstract Base OCR Engine for Advanced OCR System

Defines the standard interface and common functionality for all OCR engines.
Provides consistent API, error handling, performance tracking, and validation
for all engine implementations.

Architecture:
- Abstract base class defining engine contract
- Standard performance monitoring and metrics
- Consistent error handling and logging patterns
- Resource management and cleanup
- Validation and quality assurance framework

Modern Design Patterns:
- Template Method Pattern: Standard extraction workflow
- Strategy Pattern: Different engines implement same interface
- Observer Pattern: Performance and error monitoring
- Resource Management: Automatic cleanup and optimization

Author: Advanced OCR System
"""

import time
import traceback
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np

from ..results import OCRResult, TextRegion, BoundingBox
from ..config import OCRConfig
from ..utils.logger import Logger


class EngineStatus(Enum):
    """Engine operational status"""
    UNINITIALIZED = "uninitialized"
    INITIALIZING = "initializing"
    READY = "ready"
    PROCESSING = "processing"
    ERROR = "error"
    DISABLED = "disabled"


@dataclass
class EngineMetrics:
    """Comprehensive engine performance metrics"""
    total_extractions: int = 0
    successful_extractions: int = 0
    failed_extractions: int = 0
    avg_processing_time: float = 0.0
    avg_confidence: float = 0.0
    avg_text_length: float = 0.0
    total_processing_time: float = 0.0
    peak_memory_usage: float = 0.0
    last_error: Optional[str] = None
    initialization_time: float = 0.0
    
    @property
    def success_rate(self) -> float:
        """Calculate extraction success rate"""
        if self.total_extractions == 0:
            return 0.0
        return self.successful_extractions / self.total_extractions
    
    @property  
    def error_rate(self) -> float:
        """Calculate extraction error rate"""
        if self.total_extractions == 0:
            return 0.0
        return self.failed_extractions / self.total_extractions


class BaseOCREngine(ABC):
    """
    Abstract base class for all OCR engines
    
    Responsibilities:
    - Define standard OCR engine interface
    - Provide common functionality (logging, metrics, validation)
    - Handle resource management and cleanup
    - Implement template method pattern for extraction workflow
    - Ensure consistent error handling across all engines
    
    Template Method Pattern:
    extract() -> _validate_inputs() -> _extract_implementation() -> _post_process_result()
    """
    
    def __init__(self, config: OCRConfig):
        """
        Initialize base OCR engine with configuration
        
        Args:
            config: OCR system configuration object
        """
        self.config = config
        self.logger = Logger(self.__class__.__name__)
        
        # Engine identification
        self.engine_name = self.__class__.__name__.lower().replace('engine', '')
        self.version = getattr(config.engines, self.engine_name, {}).get('version', '1.0.0')
        
        # Engine state management
        self._status = EngineStatus.UNINITIALIZED
        self._initialization_start_time = None
        self._last_processing_time = 0.0
        
        # Performance metrics
        self._metrics = EngineMetrics()
        self._performance_history: List[Dict[str, Any]] = []
        
        # Configuration
        self.engine_config = getattr(config.engines, self.engine_name, {})
        self.timeout = self.engine_config.get('timeout', 30.0)
        self.min_confidence = self.engine_config.get('min_confidence', 0.1)
        self.max_retries = self.engine_config.get('max_retries', 2)
        
        # Resource management
        self._resource_cleanup_callbacks = []
        
        self.logger.info(f"Initialized {self.engine_name} engine v{self.version}")
    
    def extract(self, image: np.ndarray, text_regions: List[TextRegion]) -> OCRResult:
        """
        Main extraction method using Template Method Pattern
        
        Standard workflow:
        1. Validate inputs
        2. Update status and start timing
        3. Call engine-specific implementation
        4. Post-process and validate results
        5. Update metrics and log performance
        
        Args:
            image: Preprocessed image from image_processor.py
            text_regions: Detected text regions from text_detector.py
            
        Returns:
            OCRResult with extracted text and metadata
        """
        extraction_start = time.time()
        self._update_status(EngineStatus.PROCESSING)
        
        try:
            # Step 1: Input validation
            self._validate_extraction_inputs(image, text_regions)
            
            # Step 2: Pre-processing hooks
            image, text_regions = self._pre_process_inputs(image, text_regions)
            
            # Step 3: Engine-specific extraction (implemented by subclasses)
            result = self._extract_implementation(image, text_regions)
            
            # Step 4: Post-processing and validation
            result = self._post_process_result(result, extraction_start)
            
            # Step 5: Update success metrics
            self._update_success_metrics(result, extraction_start)
            
            self.logger.debug(
                f"{self.engine_name} extraction completed: "
                f"{len(result.text)} chars, {result.confidence:.3f} confidence, "
                f"{result.processing_time:.3f}s"
            )
            
            return result
            
        except Exception as e:
            # Handle extraction failure
            error_result = self._handle_extraction_error(e, extraction_start)
            self._update_error_metrics(e)
            return error_result
            
        finally:
            self._update_status(EngineStatus.READY)
    
    @abstractmethod
    def _extract_implementation(self, image: np.ndarray, text_regions: List[TextRegion]) -> OCRResult:
        """
        Engine-specific extraction implementation
        
        This method must be implemented by each concrete engine.
        It should contain the core OCR logic specific to that engine.
        
        Args:
            image: Validated and preprocessed image
            text_regions: Validated text regions
            
        Returns:
            OCRResult with raw extraction results
        """
        pass
    
    def _validate_extraction_inputs(self, image: np.ndarray, text_regions: List[TextRegion]) -> None:
        """Validate extraction inputs"""
        
        # Image validation
        if image is None or image.size == 0:
            raise ValueError(f"{self.engine_name}: Invalid or empty image provided")
        
        if len(image.shape) not in [2, 3]:
            raise ValueError(f"{self.engine_name}: Image must be 2D (grayscale) or 3D (color)")
        
        # Minimum image size check
        min_size = self.engine_config.get('min_image_size', (32, 32))
        if image.shape[0] < min_size[0] or image.shape[1] < min_size[1]:
            raise ValueError(f"{self.engine_name}: Image too small: {image.shape} < {min_size}")
        
        # Text regions validation
        if text_regions is None:
            text_regions = []
        
        # Validate individual regions
        for i, region in enumerate(text_regions):
            if not isinstance(region, TextRegion):
                raise TypeError(f"{self.engine_name}: Region {i} is not a TextRegion object")
            
            if region.width <= 0 or region.height <= 0:
                raise ValueError(f"{self.engine_name}: Region {i} has invalid dimensions")
            
            # Check region bounds
            if (region.x < 0 or region.y < 0 or 
                region.x + region.width > image.shape[1] or 
                region.y + region.height > image.shape[0]):
                raise ValueError(f"{self.engine_name}: Region {i} is outside image bounds")
    
    def _pre_process_inputs(self, image: np.ndarray, text_regions: List[TextRegion]) -> Tuple[np.ndarray, List[TextRegion]]:
        """
        Pre-process inputs before extraction
        
        Base implementation performs common preprocessing.
        Can be overridden by engines for specific needs.
        """
        
        # Filter low-confidence regions if specified
        if hasattr(self.engine_config, 'filter_low_confidence_regions') and self.engine_config.filter_low_confidence_regions:
            min_region_confidence = self.engine_config.get('min_region_confidence', 0.5)
            text_regions = [r for r in text_regions if r.confidence >= min_region_confidence]
            
            self.logger.debug(f"Filtered to {len(text_regions)} high-confidence regions")
        
        # Sort regions for consistent processing order
        text_regions.sort(key=lambda r: (r.y, r.x))  # Top-to-bottom, left-to-right
        
        return image, text_regions
    
    def _post_process_result(self, result: OCRResult, start_time: float) -> OCRResult:
        """
        Post-process extraction result
        
        Applies common post-processing like timing, validation, and cleanup.
        Can be overridden by engines for specific post-processing needs.
        """
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Update result with timing and engine info
        result.processing_time = processing_time
        result.engine_name = self.engine_name
        
        # Validate result quality
        if not self._validate_result_quality(result):
            self.logger.warning(f"{self.engine_name} produced low-quality result")
            result.metadata['quality_warning'] = True
        
        # Add engine-specific metadata
        result.metadata.update({
            'engine_version': self.version,
            'processing_timestamp': time.time(),
            'engine_config': {
                'timeout': self.timeout,
                'min_confidence': self.min_confidence
            }
        })
        
        return result
    
    def _validate_result_quality(self, result: OCRResult) -> bool:
        """
        Validate extraction result quality
        
        Basic quality checks that apply to all engines.
        Can be extended by specific engines.
        """
        
        # Check for empty results
        if not result.text or len(result.text.strip()) == 0:
            return False
        
        # Check confidence threshold
        if result.confidence < self.min_confidence:
            return False
        
        # Check for reasonable text characteristics
        text = result.text.strip()
        
        # Very short text might be noise
        if len(text) == 1 and not text.isalnum():
            return False
        
        # Check for excessive special characters (possible noise)
        special_char_ratio = sum(not c.isalnum() and not c.isspace() for c in text) / len(text)
        if special_char_ratio > 0.8:  # More than 80% special chars
            return False
        
        return True
    
    def _handle_extraction_error(self, error: Exception, start_time: float) -> OCRResult:
        """Handle extraction errors gracefully"""
        
        processing_time = time.time() - start_time
        error_msg = str(error)
        
        self.logger.error(f"{self.engine_name} extraction failed: {error_msg}")
        self.logger.debug(f"Error traceback: {traceback.format_exc()}")
        
        # Create error result
        return OCRResult(
            text="",
            confidence=0.0,
            bounding_boxes=[],
            engine_name=self.engine_name,
            processing_time=processing_time,
            metadata={
                'error': error_msg,
                'error_type': type(error).__name__,
                'extraction_failed': True
            }
        )
    
    def _update_success_metrics(self, result: OCRResult, start_time: float):
        """Update metrics for successful extraction"""
        
        processing_time = time.time() - start_time
        text_length = len(result.text)
        
        # Update counters
        self._metrics.total_extractions += 1
        self._metrics.successful_extractions += 1
        self._metrics.total_processing_time += processing_time
        
        # Update averages using exponential moving average
        alpha = 0.1  # Learning rate
        if self._metrics.total_extractions == 1:
            self._metrics.avg_processing_time = processing_time
            self._metrics.avg_confidence = result.confidence
            self._metrics.avg_text_length = text_length
        else:
            self._metrics.avg_processing_time = (
                alpha * processing_time + (1 - alpha) * self._metrics.avg_processing_time
            )
            self._metrics.avg_confidence = (
                alpha * result.confidence + (1 - alpha) * self._metrics.avg_confidence
            )
            self._metrics.avg_text_length = (
                alpha * text_length + (1 - alpha) * self._metrics.avg_text_length
            )
        
        # Store performance history (keep last 100 entries)
        self._performance_history.append({
            'timestamp': time.time(),
            'processing_time': processing_time,
            'confidence': result.confidence,
            'text_length': text_length,
            'success': True
        })
        
        if len(self._performance_history) > 100:
            self._performance_history.pop(0)
    
    def _update_error_metrics(self, error: Exception):
        """Update metrics for failed extraction"""
        
        self._metrics.total_extractions += 1
        self._metrics.failed_extractions += 1
        self._metrics.last_error = str(error)
        
        # Store error in performance history
        self._performance_history.append({
            'timestamp': time.time(),
            'error': str(error),
            'error_type': type(error).__name__,
            'success': False
        })
        
        if len(self._performance_history) > 100:
            self._performance_history.pop(0)
    
    def _update_status(self, status: EngineStatus):
        """Update engine operational status"""
        prev_status = self._status
        self._status = status
        
        if status != prev_status:
            self.logger.debug(f"{self.engine_name} status: {prev_status.value} → {status.value}")
    
    def initialize_engine(self):
        """
        Initialize engine resources
        
        Should be called before first extraction.
        Can be overridden by engines for specific initialization.
        """
        
        if self._status != EngineStatus.UNINITIALIZED:
            self.logger.warning(f"{self.engine_name} already initialized")
            return
        
        self._initialization_start_time = time.time()
        self._update_status(EngineStatus.INITIALIZING)
        
        try:
            # Engine-specific initialization
            self._initialize_implementation()
            
            # Record initialization time
            init_time = time.time() - self._initialization_start_time
            self._metrics.initialization_time = init_time
            
            self._update_status(EngineStatus.READY)
            self.logger.info(f"{self.engine_name} initialized successfully ({init_time:.3f}s)")
            
        except Exception as e:
            self._update_status(EngineStatus.ERROR)
            self.logger.error(f"{self.engine_name} initialization failed: {e}")
            raise
    
    def _initialize_implementation(self):
        """
        Engine-specific initialization implementation
        
        Override this method in concrete engines for specific initialization logic.
        Base implementation does nothing.
        """
        pass
    
    def cleanup(self):
        """Cleanup engine resources"""
        
        try:
            # Call engine-specific cleanup
            self._cleanup_implementation()
            
            # Execute registered cleanup callbacks
            for callback in self._resource_cleanup_callbacks:
                try:
                    callback()
                except Exception as e:
                    self.logger.warning(f"Cleanup callback failed: {e}")
            
            self._resource_cleanup_callbacks.clear()
            self._update_status(EngineStatus.DISABLED)
            
            self.logger.info(f"{self.engine_name} cleaned up successfully")
            
        except Exception as e:
            self.logger.error(f"{self.engine_name} cleanup failed: {e}")
    
    def _cleanup_implementation(self):
        """
        Engine-specific cleanup implementation
        
        Override this method in concrete engines for specific cleanup logic.
        Base implementation does nothing.
        """
        pass
    
    def register_cleanup_callback(self, callback):
        """Register a cleanup callback for resource management"""
        self._resource_cleanup_callbacks.append(callback)
    
    # Public API for monitoring and management
    
    def get_metrics(self) -> EngineMetrics:
        """Get current engine performance metrics"""
        return self._metrics
    
    def get_status(self) -> EngineStatus:
        """Get current engine status"""
        return self._status
    
    def get_performance_history(self) -> List[Dict[str, Any]]:
        """Get recent performance history"""
        return self._performance_history.copy()
    
    def reset_metrics(self):
        """Reset performance metrics (useful for testing)"""
        self._metrics = EngineMetrics()
        self._performance_history.clear()
        self.logger.info(f"{self.engine_name} metrics reset")
    
    def is_healthy(self) -> bool:
        """Check if engine is healthy and ready for processing"""
        
        if self._status not in [EngineStatus.READY, EngineStatus.PROCESSING]:
            return False
        
        # Check error rate
        if self._metrics.total_extractions > 10 and self._metrics.error_rate > 0.5:
            return False
        
        # Check recent performance
        recent_failures = sum(
            1 for entry in self._performance_history[-10:]
            if not entry.get('success', True)
        )
        
        if len(self._performance_history) >= 10 and recent_failures >= 8:
            return False
        
        return True
    
    def __str__(self) -> str:
        """String representation of engine"""
        return f"{self.engine_name.title()}Engine(status={self._status.value}, extractions={self._metrics.total_extractions})"
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        return (
            f"{self.__class__.__name__}("
            f"name='{self.engine_name}', "
            f"version='{self.version}', "
            f"status={self._status.value}, "
            f"success_rate={self._metrics.success_rate:.2f}"
            f")"
        )