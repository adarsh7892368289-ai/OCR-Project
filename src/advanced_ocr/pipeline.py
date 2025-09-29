# Main OCR Pipeline - Orchestrates the entire OCR process

"""
Main OCR Pipeline - Orchestrates the entire OCR process.

This module coordinates between components without doing their actual work.
It uses the EngineManager with smart selection and result combination.
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

    - Coordinates workflow between components
    - Loads and validates images using utils
    - Uses EngineManager's intelligent selection and combination
    - Packages final results
    - Provides clean public API

    Does NOT:
    - Perform image analysis (QualityAnalyzer's job)
    - Enhance images (ImageEnhancer's job)
    - Select engines manually (EngineManager's job now)
    - Combine results manually (EngineManager's job now)
    - Implement algorithms (components' jobs)
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize OCR Library with enhanced configuration"""
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
        # Pass full config to EngineManager for engine configuration
        self.engine_manager = EngineManager(self.config)
        
        # Let EngineManager handle all engine setup with enhanced capabilities
        self._initialize_engines()
        
        self.logger.info("OCR Library initialized successfully with enhanced engine management")
    
    def process_image(self,
                     image_input: Union[str, Path, np.ndarray],
                     options: Optional[ProcessingOptions] = None) -> OCRResult:
        """
        Process a single image through the OCR pipeline.

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
            
            # Step 3: Determine processing strategy (enhanced logic)
            strategy = self._determine_strategy(quality_metrics, options)
            
            # Step 4: Preprocess image (delegate to components)
            processed_image = self._preprocess_image(image, quality_metrics, strategy, options)
            
            # Step 5: Extract text (UPDATED - delegate to EngineManager's intelligence) 
            ocr_result = self._extract_text(processed_image, options, quality_metrics, strategy)
            
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
                'options_used': options.__dict__ if options else {},
                'enhanced_engine_management': True
            }
        )
    
    def get_available_engines(self) -> List[str]:
        """Get list of available OCR engines (delegate to EngineManager)"""
        return self.engine_manager.get_available_engines()
    
    def get_engine_info(self) -> Dict[str, Dict]:
        """Get detailed information about engines (delegate to EngineManager)"""
        return self.engine_manager.get_engine_info()
    
    # Private helper methods

    def _initialize_engines(self) -> None:
        """
        Initialize engines through enhanced EngineManager.
        """
        try:
            # Let EngineManager handle everything with enhanced capabilities
            init_results = self.engine_manager.initialize_available_engines()
            
            successful = [name for name, success in init_results.items() if success]
            if successful:
                self.logger.info(f"Initialized engines with smart selection: {successful}")
            else:
                self.logger.warning("No engines initialized successfully")
                
        except Exception as e:
            self.logger.error(f"Enhanced engine initialization failed: {e}")
    
    def _load_and_validate_image(self, image_input: Union[str, Path, np.ndarray]) -> np.ndarray:
        """
        Load and validate image input.
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
        Determine processing strategy with MULTI_ENGINE support.
        """
        # Use explicit strategy if provided
        if options.strategy:
            return options.strategy
        
        # Auto-determine based on quality metrics (enhanced thresholds)
        if quality_metrics.overall_score >= 0.8:
            return ProcessingStrategy.MINIMAL
        elif quality_metrics.overall_score >= 0.5:
            return ProcessingStrategy.BALANCED
        elif quality_metrics.overall_score >= 0.2:
            return ProcessingStrategy.ENHANCED
        else:
            # Very poor quality - use multiple engines for best chance
            self.logger.info(f"Very poor image quality ({quality_metrics.overall_score:.3f}), using MULTI_ENGINE strategy")
            return ProcessingStrategy.MULTI_ENGINE
    
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
    
    def _extract_text(self, image: np.ndarray, options: ProcessingOptions,
                     quality_metrics: Optional[QualityMetrics] = None,
                     strategy: Optional[ProcessingStrategy] = None) -> OCRResult:
        """
        Extract text using EngineManager's smart selection and combination.

        UPDATED: Now uses EngineManager's intelligence instead of manual selection.
        """
        available_engines = self.engine_manager.get_available_engines()
        
        if not available_engines:
            raise EngineNotAvailableError("No engines are available")
        
        # Use determined strategy or default
        processing_strategy = strategy or options.strategy or ProcessingStrategy.BALANCED
        
        try:
            # Use EngineManager's smart selection based on strategy
            if processing_strategy == ProcessingStrategy.MULTI_ENGINE:
                # Multi-engine processing with intelligent result combination
                self.logger.info("Using MULTI_ENGINE strategy for consensus-based OCR")
                
                selected_engines = self.engine_manager.select_engines_for_multi_engine(
                    preferred_engines=options.engines,
                    languages=options.languages,
                    max_engines=1  # Limit to 3 engines for performance
                )
                
                self.logger.debug(f"Selected engines for multi-engine: {selected_engines}")
                
                results = self.engine_manager.execute_multiple_engines(
                    selected_engines, 
                    image, 
                    options, 
                    use_parallel=options.use_parallel_processing
                )
                
                # Let EngineManager intelligently combine results
                combined_result = self.engine_manager.combine_results(results)
                
                if combined_result.confidence < options.min_confidence:
                    self.logger.warning(
                        f"Multi-engine result confidence ({combined_result.confidence:.3f}) "
                        f"below threshold ({options.min_confidence}), but proceeding"
                    )
                
                self.logger.info(f"Multi-engine processing complete: {combined_result.engine_used}, "
                               f"confidence: {combined_result.confidence:.3f}")
                return combined_result
                
            else:
                # Single engine selection (MINIMAL/BALANCED/ENHANCED)
                self.logger.debug(f"Using single engine strategy: {processing_strategy.value}")
                
                best_engine = self.engine_manager.select_best_engine(
                    strategy=processing_strategy,
                    preferred_engines=options.engines,
                    languages=options.languages,
                    quality_metrics=quality_metrics
                )
                
                self.logger.debug(f"Selected best engine: {best_engine}")
                
                result = self.engine_manager.execute_engine(best_engine, image, options)
                
                if result.confidence < options.min_confidence:
                    # For single engine, we can try fallback to multi-engine if confidence is too low
                    if processing_strategy != ProcessingStrategy.ENHANCED:
                        self.logger.info(
                            f"Single engine confidence ({result.confidence:.3f}) below threshold "
                            f"({options.min_confidence}), attempting multi-engine fallback"
                        )
                        
                        # Fallback to multi-engine
                        fallback_engines = self.engine_manager.select_engines_for_multi_engine(
                            preferred_engines=options.engines,
                            languages=options.languages,
                            max_engines=2
                        )
                        
                        fallback_results = self.engine_manager.execute_multiple_engines(
                            fallback_engines, image, options, use_parallel=True
                        )
                        
                        fallback_result = self.engine_manager.combine_results(fallback_results)
                        
                        if fallback_result.confidence > result.confidence:
                            self.logger.info(f"Fallback improved confidence: {fallback_result.confidence:.3f}")
                            fallback_result.metadata["fallback_used"] = True
                            return fallback_result
                
                self.logger.info(f"Single engine processing complete: {result.engine_used}, "
                               f"confidence: {result.confidence:.3f}")
                return result
                
        except EngineNotAvailableError:
            raise
        except Exception as e:
            self.logger.error(f"Enhanced text extraction failed: {e}")
            raise OCRLibraryError(f"Text extraction failed: {str(e)}") from e
    
    def _package_result(self, ocr_result: OCRResult, quality_metrics: QualityMetrics,
                       strategy: ProcessingStrategy, total_time: float) -> OCRResult:
        """
        Package final result with enhanced metadata.
        
        UPDATED: Includes enhanced engine management metadata.
        """
        # Update the result with pipeline metadata
        ocr_result.processing_time = total_time
        ocr_result.quality_metrics = quality_metrics
        ocr_result.strategy_used = strategy
        
        # Add enhanced pipeline metadata
        ocr_result.metadata.update({
            'pipeline_version': '2.0',  # Updated version with enhanced engine management
            'strategy_used': strategy.value,
            'total_processing_time': total_time,
            'quality_score': quality_metrics.overall_score,
            'enhanced_engine_management': True,
            'smart_selection_used': True
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
                'failed': True,
                'enhanced_engine_management': True
            }
        )