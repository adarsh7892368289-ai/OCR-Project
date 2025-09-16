"""
Advanced OCR System - Core Orchestrator
=======================================

Main pipeline controller that coordinates the entire OCR processing flow.

PIPELINE FLOW:
1. Receive raw image from __init__.py
2. Call image_processor.py → get (enhanced_image, text_regions, quality_metrics)
3. Call engine_coordinator.py → get raw OCRResult(s)
4. Call text_processor.py → get final OCRResult
5. Return final result to __init__.py

DEPENDENCIES: image_processor.py, engine_coordinator.py, text_processor.py, config.py, logger.py
USED BY: __init__.py ONLY
"""

import time
from pathlib import Path
from typing import Union, Optional, List, Dict, Any
import numpy as np
from PIL import Image
import cv2

from .config import OCRConfig
from .results import OCRResult, ProcessingMetrics, BoundingBox
from .utils.logger import OCRLogger, ProcessingStageTimer
from .utils.image_utils import ImageLoader, ImageValidator
from .utils.model_utils import ModelLoader
from .preprocessing.image_processor import ImageProcessor, PreprocessingResult
from .engines.engine_coordinator import EngineCoordinator, CoordinationResult
from .postprocessing.text_processor import TextProcessor


class OCRCore:
    """
    Main orchestrator for the Advanced OCR pipeline.
    
    Coordinates the flow between preprocessing, engine coordination, and postprocessing
    while maintaining clean separation of concerns and comprehensive error handling.
    """
    
    def __init__(self, config: Optional[OCRConfig] = None):
        """
        Initialize the OCR core with configuration and component instances.
        
        Args:
            config: OCR configuration object. If None, uses default configuration.
        """
        self.config = config or OCRConfig()
        self.logger = OCRLogger(self.config.logging)
        
        # Initialize pipeline components
        self._initialize_components()
        
        # Performance tracking
        self.processing_metrics = []
        
        self.logger.info("OCR Core initialized successfully", extra={
            "config_hash": hash(str(self.config)),
            "components_loaded": len(self._get_component_status())
        })
    
    def _initialize_components(self) -> None:
        """Initialize all pipeline components with error handling."""
        try:
            # Initialize preprocessing pipeline
            self.image_processor = ImageProcessor(self.config.preprocessing)
            self.logger.debug("Image processor initialized")
            
            # Initialize engine coordination layer
            self.engine_coordinator = EngineCoordinator(self.config.engines)
            self.logger.debug("Engine coordinator initialized")
            
            # Initialize postprocessing pipeline
            self.text_processor = TextProcessor(self.config.postprocessing)
            self.logger.debug("Text processor initialized")
            
            # Initialize utilities
            self.image_loader = ImageLoader()
            self.image_validator = ImageValidator()
            
        except Exception as e:
            self.logger.error("Failed to initialize components", extra={
                "error": str(e),
                "error_type": type(e).__name__
            })
            raise RuntimeError(f"Core initialization failed: {str(e)}") from e
    
    def extract_text(
        self, 
        image: Union[str, Path, np.ndarray, Image.Image],
        region_filter: Optional[List[BoundingBox]] = None,
        processing_options: Optional[Dict[str, Any]] = None
    ) -> OCRResult:
        """
        Main text extraction method - orchestrates the entire pipeline.
        
        Args:
            image: Input image (file path, PIL Image, or numpy array)
            region_filter: Optional list of regions to focus on
            processing_options: Optional processing parameters override
            
        Returns:
            OCRResult: Complete extraction results with text, confidence, and metadata
            
        Raises:
            ValueError: If image is invalid or unsupported format
            RuntimeError: If processing pipeline fails
        """
        # Start overall timer
        with ProcessingStageTimer("total_processing", self.logger) as total_timer:
            
            # Stage 1: Load and validate input image
            with ProcessingStageTimer("image_loading", self.logger):
                validated_image = self._load_and_validate_image(image)
            
            # Stage 2: Preprocessing - get enhanced image, text regions, quality metrics
            with ProcessingStageTimer("preprocessing", self.logger):
                preprocessing_result = self._run_preprocessing(validated_image, region_filter)
            
            # Stage 3: Engine coordination - select engines and extract text
            with ProcessingStageTimer("engine_coordination", self.logger):
                coordination_result = self._run_engine_coordination(
                    preprocessing_result, processing_options
                )
            
            # Stage 4: Postprocessing - fusion, layout reconstruction, final confidence
            with ProcessingStageTimer("postprocessing", self.logger):
                final_result = self._run_postprocessing(
                    coordination_result, preprocessing_result
                )
            
            # Stage 5: Finalize result with metrics
            self._finalize_result(final_result, total_timer.elapsed_time)
            
            return final_result
    
    def _load_and_validate_image(
        self, 
        image: Union[str, Path, np.ndarray, Image.Image]
    ) -> np.ndarray:
        """
        Load and validate input image with comprehensive error handling.
        
        Args:
            image: Input image in various formats
            
        Returns:
            np.ndarray: Validated image as numpy array in BGR format
            
        Raises:
            ValueError: If image is invalid or unsupported
        """
        try:
            # Load image using ImageLoader utility
            if isinstance(image, (str, Path)):
                loaded_image = self.image_loader.load_from_path(image)
                self.logger.debug(f"Loaded image from path: {image}")
            elif isinstance(image, Image.Image):
                loaded_image = self.image_loader.load_from_pil(image)
                self.logger.debug("Loaded image from PIL Image")
            elif isinstance(image, np.ndarray):
                loaded_image = self.image_loader.load_from_numpy(image)
                self.logger.debug("Loaded image from numpy array")
            else:
                raise ValueError(f"Unsupported image type: {type(image)}")
            
            # Validate image quality and format
            validation_result = self.image_validator.validate(loaded_image)
            if not validation_result.is_valid:
                raise ValueError(f"Image validation failed: {validation_result.error_message}")
            
            self.logger.info("Image loaded and validated successfully", extra={
                "image_shape": loaded_image.shape,
                "image_size_mb": loaded_image.nbytes / (1024 * 1024),
                "validation_score": validation_result.quality_score
            })
            
            return loaded_image
            
        except Exception as e:
            self.logger.error("Image loading failed", extra={
                "error": str(e),
                "image_type": type(image).__name__
            })
            raise ValueError(f"Failed to load image: {str(e)}") from e
    
    def _run_preprocessing(
        self, 
        image: np.ndarray, 
        region_filter: Optional[List[BoundingBox]]
    ) -> PreprocessingResult:
        """
        Run the preprocessing pipeline through image_processor.py.
        
        Args:
            image: Validated input image
            region_filter: Optional regions to focus processing on
            
        Returns:
            PreprocessingResult: Enhanced image, text regions, and quality metrics
        """
        try:
            # Call image_processor.py for unified preprocessing
            preprocessing_result = self.image_processor.process(
                image=image,
                region_filter=region_filter
            )
            
            self.logger.info("Preprocessing completed successfully", extra={
                "text_regions_detected": len(preprocessing_result.text_regions),
                "enhancement_applied": preprocessing_result.enhancement_applied,
                "quality_score": preprocessing_result.quality_metrics.overall_score,
                "processing_time_ms": preprocessing_result.processing_time_ms
            })
            
            return preprocessing_result
            
        except Exception as e:
            self.logger.error("Preprocessing failed", extra={
                "error": str(e),
                "stage": "preprocessing"
            })
            raise RuntimeError(f"Preprocessing pipeline failed: {str(e)}") from e
    
    def _run_engine_coordination(
        self, 
        preprocessing_result: PreprocessingResult,
        processing_options: Optional[Dict[str, Any]]
    ) -> CoordinationResult:
        """
        Run engine coordination through engine_coordinator.py.
        
        Args:
            preprocessing_result: Results from preprocessing stage
            processing_options: Optional processing parameter overrides
            
        Returns:
            CoordinationResult: Raw OCR results from selected engines
        """
        try:
            # Call engine_coordinator.py for intelligent engine selection and execution
            coordination_result = self.engine_coordinator.coordinate_extraction(
                enhanced_image=preprocessing_result.enhanced_image,
                text_regions=preprocessing_result.text_regions,
                quality_metrics=preprocessing_result.quality_metrics,
                processing_options=processing_options or {}
            )
            
            self.logger.info("Engine coordination completed successfully", extra={
                "engines_used": [engine.name for engine in coordination_result.engines_used],
                "content_type_detected": coordination_result.content_classification.primary_type,
                "results_count": len(coordination_result.ocr_results),
                "total_confidence": sum(r.confidence.overall for r in coordination_result.ocr_results) / len(coordination_result.ocr_results)
            })
            
            return coordination_result
            
        except Exception as e:
            self.logger.error("Engine coordination failed", extra={
                "error": str(e),
                "stage": "engine_coordination"
            })
            raise RuntimeError(f"Engine coordination failed: {str(e)}") from e
    
    def _run_postprocessing(
        self, 
        coordination_result: CoordinationResult,
        preprocessing_result: PreprocessingResult
    ) -> OCRResult:
        """
        Run postprocessing through text_processor.py.
        
        Args:
            coordination_result: Results from engine coordination
            preprocessing_result: Original preprocessing results for context
            
        Returns:
            OCRResult: Final processed and enhanced OCR result
        """
        try:
            # Call text_processor.py for result fusion, layout reconstruction, and final processing
            final_result = self.text_processor.process_results(
                ocr_results=coordination_result.ocr_results,
                content_classification=coordination_result.content_classification,
                quality_metrics=preprocessing_result.quality_metrics,
                original_image_shape=preprocessing_result.original_image_shape
            )
            
            self.logger.info("Postprocessing completed successfully", extra={
                "final_text_length": len(final_result.text),
                "confidence_score": final_result.confidence.overall,
                "layout_elements": len(final_result.layout_elements) if final_result.layout_elements else 0,
                "processing_applied": final_result.processing_metadata.get("postprocessing_steps", [])
            })
            
            return final_result
            
        except Exception as e:
            self.logger.error("Postprocessing failed", extra={
                "error": str(e),
                "stage": "postprocessing"
            })
            raise RuntimeError(f"Postprocessing pipeline failed: {str(e)}") from e
    
    def _finalize_result(self, result: OCRResult, total_processing_time: float) -> None:
        """
        Finalize the OCR result with comprehensive metrics and metadata.
        
        Args:
            result: The final OCR result to enhance
            total_processing_time: Total processing time in seconds
        """
        # Add processing metrics
        if not hasattr(result, 'processing_metrics') or result.processing_metrics is None:
            result.processing_metrics = ProcessingMetrics()
        
        result.processing_metrics.total_processing_time = total_processing_time
        result.processing_metrics.timestamp = time.time()
        
        # Add system metadata
        if not hasattr(result, 'processing_metadata') or result.processing_metadata is None:
            result.processing_metadata = {}
        
        result.processing_metadata.update({
            "ocr_core_version": "1.0.0",
            "pipeline_components": self._get_component_status(),
            "configuration_hash": hash(str(self.config)),
            "total_processing_stages": 4
        })
        
        # Store metrics for analysis
        self.processing_metrics.append({
            "timestamp": result.processing_metrics.timestamp,
            "processing_time": total_processing_time,
            "text_length": len(result.text),
            "confidence": result.confidence.overall
        })
        
        self.logger.info("OCR processing completed", extra={
            "total_time_s": total_processing_time,
            "text_extracted_chars": len(result.text),
            "final_confidence": result.confidence.overall,
            "success": True
        })
    
    def _get_component_status(self) -> Dict[str, bool]:
        """Get status of all pipeline components."""
        return {
            "image_processor": hasattr(self, 'image_processor') and self.image_processor is not None,
            "engine_coordinator": hasattr(self, 'engine_coordinator') and self.engine_coordinator is not None,
            "text_processor": hasattr(self, 'text_processor') and self.text_processor is not None,
            "image_loader": hasattr(self, 'image_loader') and self.image_loader is not None,
            "image_validator": hasattr(self, 'image_validator') and self.image_validator is not None
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive performance metrics from recent processing.
        
        Returns:
            Dict containing processing performance statistics
        """
        if not self.processing_metrics:
            return {"status": "no_metrics", "message": "No processing completed yet"}
        
        recent_metrics = self.processing_metrics[-10:]  # Last 10 processes
        
        return {
            "total_processes": len(self.processing_metrics),
            "recent_processes": len(recent_metrics),
            "average_processing_time": sum(m["processing_time"] for m in recent_metrics) / len(recent_metrics),
            "average_confidence": sum(m["confidence"] for m in recent_metrics) / len(recent_metrics),
            "average_text_length": sum(m["text_length"] for m in recent_metrics) / len(recent_metrics),
            "component_status": self._get_component_status()
        }
    
    def cleanup(self) -> None:
        """Clean up resources and close connections."""
        try:
            # Cleanup components if they have cleanup methods
            for component_name in ["image_processor", "engine_coordinator", "text_processor"]:
                component = getattr(self, component_name, None)
                if component and hasattr(component, 'cleanup'):
                    component.cleanup()
            
            self.logger.info("OCR Core cleanup completed successfully")
            
        except Exception as e:
            self.logger.error("Error during cleanup", extra={"error": str(e)})
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()
        
        if exc_type is not None:
            self.logger.error("OCR Core exited with exception", extra={
                "exception_type": exc_type.__name__,
                "exception_message": str(exc_val)
            })
        
        return False  # Don't suppress exceptions