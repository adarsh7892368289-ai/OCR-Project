"""Main OCR pipeline for coordinating the complete text extraction workflow.

Orchestrates image loading, quality analysis, preprocessing, and text extraction
using multiple OCR engines.

Examples
--------
    from advanced_ocr import OCRLibrary, ProcessingOptions
    
    # Basic usage
    ocr = OCRLibrary()
    result = ocr.process_image("document.jpg")
    print(result.text)
    
    # With custom options
    options = ProcessingOptions(enhance_image=True, engines=["paddleocr"])
    result = ocr.process_image("document.jpg", options)
    
    # Batch processing
    images = ["doc1.jpg", "doc2.jpg", "doc3.jpg"]
    batch_result = ocr.process_batch(images)
    print(f"Processed {batch_result.successful_count} images")
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
    """Main OCR library for text extraction from images.
    
    Coordinates the complete OCR pipeline including image loading, quality
    analysis, preprocessing, and text extraction using multiple OCR engines.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize OCR library with optional custom config."""
        self.logger = setup_logger(self.__class__.__name__)
        self.config = load_config(config_path)
        
        # Initialize components
        self.quality_analyzer = QualityAnalyzer(self.config.get("quality_analyzer", {}))
        self.image_enhancer = ImageEnhancer(self.config.get("image_enhancer", {}))
        self.engine_manager = EngineManager(self.config)
        
        self._initialize_engines()
        self.logger.info("OCR Library initialized successfully")
    
    def process_image(self,
                     image_input: Union[str, Path, np.ndarray],
                     options: Optional[ProcessingOptions] = None) -> OCRResult:
        """Process a single image and extract text.
        
        Executes the complete OCR pipeline: loads image, analyzes quality,
        applies preprocessing, and extracts text.
        """
        start_time = time.time()
        
        try:
            if options is None:
                options = ProcessingOptions()
            
            # Load and validate
            image = self._load_and_validate_image(image_input)
            
            # Analyze quality
            quality_metrics = self.quality_analyzer.analyze_image(image)
            strategy = options.strategy or ProcessingStrategy.BALANCED
            
            # Preprocess
            processed_image = self._preprocess_image(image, quality_metrics, strategy, options)
            
            # Extract text
            ocr_result = self._extract_text(processed_image, options, strategy)
            
            # Package result
            return self._package_result(ocr_result, quality_metrics, strategy, 
                                       time.time() - start_time)
            
        except Exception as e:
            self.logger.error(f"OCR processing failed: {e}")
            raise OCRLibraryError(f"Failed to process image: {str(e)}") from e
    
    def process_batch(self,
                     image_paths: List[Union[str, Path]],
                     options: Optional[ProcessingOptions] = None) -> BatchResult:
        """Process multiple images in batch.
        
        Processes a list of images sequentially with the same options.
        Failed images are recorded with error information.
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
                error_result = self._create_error_result(str(image_path), str(e))
                results.append(error_result)
                self.logger.error(f"Failed to process {image_path}: {e}")
        
        # Calculate statistics
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
                'simplified_engine_management': True
            }
        )
    
    def get_available_engines(self) -> List[str]:
        """Get list of available OCR engines."""
        return self.engine_manager.get_available_engines()
    
    def get_engine_info(self) -> Dict[str, Dict]:
        """Get detailed information about all engines."""
        return self.engine_manager.get_engine_info()
    
    # Private methods
    
    def _initialize_engines(self) -> None:
        """Initialize all available OCR engines."""
        try:
            init_results = self.engine_manager.initialize_available_engines()
            successful = [name for name, success in init_results.items() if success]
            
            if successful:
                self.logger.info(f"Initialized engines: {successful}")
            else:
                self.logger.warning("No engines initialized successfully")
                
        except Exception as e:
            self.logger.error(f"Engine initialization failed: {e}")
    
    def _load_and_validate_image(self, image_input: Union[str, Path, np.ndarray]) -> np.ndarray:
        """Load and validate image from path or array."""
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
    
    def _preprocess_image(self, image: np.ndarray,
                         quality_metrics: QualityMetrics,
                         strategy: ProcessingStrategy,
                         options: ProcessingOptions) -> np.ndarray:
        """Apply preprocessing based on quality and options."""
        processed_image = image.copy()
        
        # Rotation correction
        if options.detect_orientation and options.correct_rotation:
            processed_image = self._handle_rotation_correction(processed_image)
        
        # Image enhancement
        if options.enhance_image:
            processed_image = self._handle_image_enhancement(
                processed_image, quality_metrics, strategy
            )
        
        return processed_image
    
    def _handle_rotation_correction(self, image: np.ndarray) -> np.ndarray:
        """Detect and correct image rotation if needed."""
        try:
            rotation_angle = detect_rotation(image)
            if abs(rotation_angle) > 1.0:
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
        """Apply image enhancement to improve OCR accuracy."""
        try:
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
                     strategy: ProcessingStrategy) -> OCRResult:
        """Extract text using appropriate engine(s)."""
        available_engines = self.engine_manager.get_available_engines()
        
        if not available_engines:
            raise EngineNotAvailableError("No engines are available")
        
        try:
            if strategy == ProcessingStrategy.MULTI_ENGINE:
                # Use all available engines in parallel
                self.logger.info("Using MULTI_ENGINE strategy")
                results = self.engine_manager.execute_multiple_engines(
                    available_engines, image, options, use_parallel=True
                )
                combined_result = self.engine_manager.combine_results(results)
                self.logger.info(f"Multi-engine complete: {combined_result.engine_used}, "
                               f"confidence: {combined_result.confidence:.3f}")
                return combined_result
                
            else:
                # Use PaddleOCR as single engine
                if "paddleocr" not in available_engines:
                    raise EngineNotAvailableError("PaddleOCR not available")
                
                self.logger.info("Using PaddleOCR")
                result = self.engine_manager.execute_engine("paddleocr", image, options)
                self.logger.info(f"PaddleOCR complete: confidence {result.confidence:.3f}")
                return result
                
        except EngineNotAvailableError:
            raise
        except Exception as e:
            self.logger.error(f"Text extraction failed: {e}")
            raise OCRLibraryError(f"Text extraction failed: {str(e)}") from e
    
    def _package_result(self, ocr_result: OCRResult, quality_metrics: QualityMetrics,
                       strategy: ProcessingStrategy, total_time: float) -> OCRResult:
        """Package final result with metadata."""
        ocr_result.processing_time = total_time
        ocr_result.quality_metrics = quality_metrics
        ocr_result.strategy_used = strategy
        
        ocr_result.metadata.update({
            'pipeline_version': '2.0_simplified',
            'strategy_used': strategy.value,
            'total_processing_time': total_time,
            'quality_score': quality_metrics.overall_score,
            'simplified_engine_management': True
        })
        
        return ocr_result
    
    def _create_error_result(self, image_path: str, error_message: str) -> OCRResult:
        """Create error result for failed processing."""
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