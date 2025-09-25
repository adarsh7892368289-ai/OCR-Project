"""
Main OCR Pipeline - Orchestrates the entire OCR process.

This module coordinates between components without doing their actual work.
CLEAN IMPLEMENTATION - Only orchestration and coordination logic.
"""

import time
from pathlib import Path
from typing import Union, Optional, List, Dict
import numpy as np

from .types import (
    OCRResult, ProcessingOptions, QualityMetrics, 
    ProcessingStrategy, BatchResult
)
from .exceptions import OCRLibraryError, EngineNotAvailableError
from .core.engine_manager import EngineManager  
from .preprocessing.quality_analyzer import QualityAnalyzer
from .preprocessing.image_enhancer import ImageEnhancer
from .utils.config import load_config
from .utils.logging import setup_logger
from .utils.images import load_image, validate_image, detect_rotation, correct_rotation


class OCRLibrary:
    """
    Main OCR Library class - coordinates the entire OCR pipeline.
    
    CLEAN IMPLEMENTATION - Pure orchestration:
    - Coordinates workflow between components
    - Loads and validates images using utils
    - Makes simple strategy decisions
    - Packages final results
    - Provides clean public API
    
    Does NOT:
    - Perform image analysis (QualityAnalyzer's job)
    - Enhance images (ImageEnhancer's job)
    - Register engines (EngineManager's job)
    - Run OCR engines (EngineManager's job)
    - Implement algorithms (components' jobs)
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize OCR Library with configuration"""
        self.logger = setup_logger(self.__class__.__name__)
        
        # Load configuration
        self.config = load_config(config_path)
        
        # Initialize components (they handle their own setup)
        self.quality_analyzer = QualityAnalyzer(
            self.config.get("quality_analyzer", {})
        )
        self.image_enhancer = ImageEnhancer(
            self.config.get("image_enhancer", {})
        )
        self.engine_manager = EngineManager(self.config)
        
        # Let EngineManager handle all engine setup
        self._initialize_engines()
        
        self.logger.info("OCR Library initialized successfully")
    
    def process_image(self, 
                     image_input: Union[str, Path, np.ndarray], 
                     options: Optional[ProcessingOptions] = None) -> OCRResult:
        """
        Process a single image through the OCR pipeline.
        
        CLEAN ORCHESTRATION: Each step delegates to appropriate component.
        
        Args:
            image_input: Image file path or numpy array
            options: Processing options
            
        Returns:
            OCR result with extracted text and metadata
        """
        start_time = time.time()
        
        try:
            # Use default options if none provided
            if options is None:
                options = ProcessingOptions()
            
            # Step 1: Load and validate image (delegate to utils.images)
            image = self._load_and_validate_image(image_input)
            
            # Step 2: Analyze image quality (delegate to QualityAnalyzer)
            quality_metrics = self.quality_analyzer.analyze_image(image)
            
            # Step 3: Determine processing strategy (simple pipeline logic)
            strategy = self._determine_strategy(quality_metrics, options)
            
            # Step 4: Preprocess image (delegate to components)
            processed_image = self._preprocess_image(image, quality_metrics, strategy, options)
            
            # Step 5: Extract text (delegate to EngineManager) 
            ocr_result = self._extract_text(processed_image, options)
            
            # Step 6: Package final result (coordination only)
            final_result = self._package_result(
                ocr_result, quality_metrics, strategy, time.time() - start_time
            )
            
            return final_result
            
        except Exception as e:
            self.logger.error(f"OCR processing failed: {e}")
            raise OCRLibraryError(f"Failed to process image: {str(e)}") from e
    
    def process_batch(self, 
                     image_paths: List[Union[str, Path]], 
                     options: Optional[ProcessingOptions] = None) -> BatchResult:
        """
        Process multiple images in batch.
        
        CLEAN IMPLEMENTATION: Simple iteration with result aggregation.
        
        Args:
            image_paths: List of image file paths
            options: Processing options
            
        Returns:
            Batch processing results
        """
        start_time = time.time()
        results = []
        successful_count = 0
        total_confidence = 0.0
        
        for image_path in image_paths:
            try:
                result = self.process_image(image_path, options)
                results.append(result)
                
                if result.success:
                    successful_count += 1
                    total_confidence += result.confidence
                    
            except Exception as e:
                # Create error result for failed image
                error_result = self._create_error_result(str(image_path), str(e))
                results.append(error_result)
                self.logger.error(f"Failed to process {image_path}: {e}")
        
        # Calculate batch statistics (simple aggregation)
        total_time = time.time() - start_time
        failed_count = len(results) - successful_count
        avg_confidence = total_confidence / successful_count if successful_count > 0 else 0.0
        
        return BatchResult(
            results=results,
            total_processing_time=total_time,
            successful_count=successful_count,
            failed_count=failed_count,
            average_confidence=avg_confidence,
            metadata={
                'total_images': len(image_paths),
                'options_used': options.__dict__ if options else {}
            }
        )
    
    def get_available_engines(self) -> List[str]:
        """Get list of available OCR engines (delegate to EngineManager)"""
        return self.engine_manager.get_available_engines()
    
    def get_engine_info(self) -> Dict[str, Dict]:
        """Get detailed information about engines (delegate to EngineManager)"""
        return self.engine_manager.get_engine_info()
    
    # Private helper methods (CLEAN - only coordination logic)
    
    def _initialize_engines(self) -> None:
        """
        Initialize engines through EngineManager.
        
        CLEAN: No engine instantiation, just tell EngineManager to handle it.
        """
        try:
            # Let EngineManager handle everything
            init_results = self.engine_manager.initialize_available_engines()
            
            successful = [name for name, success in init_results.items() if success]
            if successful:
                self.logger.info(f"Initialized engines: {successful}")
            else:
                self.logger.warning("No engines initialized successfully")
                
        except Exception as e:
            self.logger.error(f"Engine initialization failed: {e}")
    
    def _load_and_validate_image(self, image_input: Union[str, Path, np.ndarray]) -> np.ndarray:
        """
        Load and validate image input.
        
        CLEAN: Only uses utils.images, no processing logic.
        """
        if isinstance(image_input, np.ndarray):
            if not validate_image(image_input):
                raise OCRLibraryError("Invalid image array provided")
            return image_input
            
        elif isinstance(image_input, (str, Path)):
            image = load_image(image_input)
            if image is None:
                raise OCRLibraryError(f"Could not load image: {image_input}")
            return image
            
        else:
            raise OCRLibraryError(f"Unsupported image input type: {type(image_input)}")
    
    def _determine_strategy(self, quality_metrics: QualityMetrics, 
                          options: ProcessingOptions) -> ProcessingStrategy:
        """
        Determine processing strategy based on quality and options.
        
        CLEAN: Simple decision logic, no complex algorithms.
        """
        # Use explicit strategy if provided
        if options.strategy:
            return options.strategy
        
        # Auto-determine based on quality metrics (simple thresholds)
        if quality_metrics.overall_score >= 0.8:
            return ProcessingStrategy.MINIMAL
        elif quality_metrics.overall_score >= 0.5:
            return ProcessingStrategy.BALANCED
        else:
            return ProcessingStrategy.ENHANCED
    
    def _preprocess_image(self, image: np.ndarray, 
                         quality_metrics: QualityMetrics,
                         strategy: ProcessingStrategy, 
                         options: ProcessingOptions) -> np.ndarray:
        """
        Apply preprocessing by delegating to appropriate components.
        
        CLEAN: Pure delegation, no processing algorithms.
        """
        processed_image = image.copy()
        
        # Step 1: Rotation correction (delegate to utils.images)
        if options.detect_orientation and options.correct_rotation:
            processed_image = self._handle_rotation_correction(processed_image)
        
        # Step 2: Image enhancement (delegate to ImageEnhancer)
        if options.enhance_image:
            processed_image = self._handle_image_enhancement(
                processed_image, quality_metrics, strategy
            )
        
        return processed_image
    
    def _handle_rotation_correction(self, image: np.ndarray) -> np.ndarray:
        """
        Handle rotation correction using utils.images.
        
        CLEAN: Only coordination, actual work done by utils.
        """
        try:
            rotation_angle = detect_rotation(image)
            if abs(rotation_angle) > 1.0:  # Only correct significant rotations
                corrected_image = correct_rotation(image, rotation_angle)
                self.logger.debug(f"Corrected rotation by {rotation_angle} degrees")
                return corrected_image
            return image
        except Exception as e:
            self.logger.warning(f"Rotation correction failed: {e}")
            return image
    
    def _handle_image_enhancement(self, image: np.ndarray, 
                                 quality_metrics: QualityMetrics,
                                 strategy: ProcessingStrategy) -> np.ndarray:
        """
        Handle image enhancement using ImageEnhancer.
        
        CLEAN: Only coordination, actual work done by ImageEnhancer.
        """
        try:
            # Let ImageEnhancer decide if enhancement is needed and apply it
            enhancement_result = self.image_enhancer.enhance_image(
                image, strategy, quality_metrics
            )
            
            if enhancement_result.was_enhanced:
                self.logger.debug(f"Applied {enhancement_result.enhancement_applied} enhancement")
                return enhancement_result.enhanced_image
            
            return image
        except Exception as e:
            self.logger.warning(f"Image enhancement failed: {e}")
            return image
    
    def _extract_text(self, image: np.ndarray, options: ProcessingOptions) -> OCRResult:
        """
        Extract text by delegating everything to EngineManager.
        
        CLEAN: All engine logic handled by EngineManager.
        """
        available_engines = self.engine_manager.get_available_engines()
        
        if not available_engines:
            raise EngineNotAvailableError("No engines are available")
        
        # Determine engines to use (simple logic)
        engines_to_use = options.engines or available_engines[:1]  # Default to first available
        engines_to_use = [eng for eng in engines_to_use if eng in available_engines]
        
        if not engines_to_use:
            raise EngineNotAvailableError("No requested engines are available")
        
        # Single engine case
        if len(engines_to_use) == 1:
            result = self.engine_manager.execute_engine(engines_to_use[0], image, options)
            
            if result.confidence >= options.min_confidence:
                return result
            else:
                raise OCRLibraryError(
                    f"Result confidence ({result.confidence:.3f}) below threshold ({options.min_confidence})"
                )
        
        # Multi-engine case - delegate to EngineManager
        else:
            results = self.engine_manager.execute_multiple_engines(
                engines_to_use, image, options, use_parallel=options.use_parallel_processing
            )
            
            # Simple result selection (just pick highest confidence)
            if not results:
                raise OCRLibraryError("No engines produced results")
                
            best_result = max(results.values(), key=lambda r: r.confidence)
            
            if best_result.confidence < options.min_confidence:
                raise OCRLibraryError("No engine produced acceptable results")
            
            return best_result
    
    def _package_result(self, ocr_result: OCRResult, quality_metrics: QualityMetrics,
                       strategy: ProcessingStrategy, total_time: float) -> OCRResult:
        """
        Package final result with metadata.
        
        CLEAN: Simple result packaging, no processing logic.
        """
        # Update the result with pipeline metadata
        ocr_result.processing_time = total_time
        ocr_result.quality_metrics = quality_metrics
        ocr_result.strategy_used = strategy
        
        # Add pipeline metadata
        ocr_result.metadata.update({
            'pipeline_version': '1.0',
            'strategy_used': strategy.value,
            'total_processing_time': total_time,
            'quality_score': quality_metrics.overall_score,
        })
        
        return ocr_result
    
    def _create_error_result(self, image_path: str, error_message: str) -> OCRResult:
        """
        Create error result for failed processing.
        
        CLEAN: Simple error result creation.
        """
        error_metrics = QualityMetrics(
            overall_score=0.0,
            sharpness_score=0.0, 
            noise_level=1.0,
            contrast_score=0.0,
            brightness_score=0.0,
            needs_enhancement=False
        )
        
        return OCRResult(
            text="",
            confidence=0.0,
            processing_time=0.0,
            engine_used="none",
            quality_metrics=error_metrics,
            strategy_used=ProcessingStrategy.MINIMAL,
            metadata={
                'error': error_message,
                'image_path': image_path,
                'failed': True
            }
        )