# src/core/enhanced_engine_manager.py - Updated with Step 4 Integration

import cv2
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import existing components
from .base_engine import BaseOCREngine, OCRResult, ProcessingLevel
from .engine_manager import EngineManager
from ..preprocessing.adaptive_processor import (
    AdaptivePreprocessor, ProcessingOptions, ProcessingLevel as PrepLevel,
    PipelineStrategy, ProcessingResult
)
from ..utils.config import ConfigManager

logger = logging.getLogger(__name__)

@dataclass
class EnhancedProcessingOptions:
    """Enhanced processing options that include preprocessing"""
    # OCR Engine options
    engines: List[str] = field(default_factory=lambda: ["paddleocr", "easyocr"])
    parallel_processing: bool = True
    confidence_threshold: float = 0.7
    
    # Preprocessing options
    enable_preprocessing: bool = True
    preprocessing_level: PrepLevel = PrepLevel.BALANCED
    preprocessing_strategy: PipelineStrategy = PipelineStrategy.CONTENT_AWARE
    preprocess_per_engine: bool = True  # Different preprocessing for each engine
    
    # Integration options
    validate_preprocessing: bool = True
    fallback_to_original: bool = True
    max_processing_time: float = 300.0  # 5 minutes max

class EnhancedEngineManager(EngineManager):
    """
    Enhanced Engine Manager with integrated adaptive preprocessing
    """
    
    def __init__(self, config_path: Optional[str] = None):
        super().__init__(config_path)
        
        # Initialize adaptive preprocessor
        preprocessing_config = self.config.get("preprocessing", {})
        self.preprocessor = AdaptivePreprocessor(preprocessing_config)
        
        # Engine-specific preprocessing configurations
        self.engine_preprocessing_configs = {
            "tesseract": {
                "processing_level": PrepLevel.BALANCED,
                "strategy": PipelineStrategy.CONTENT_AWARE,
                "additional_steps": ["binarization", "morphological_cleaning"]
            },
            "easyocr": {
                "processing_level": PrepLevel.LIGHT,
                "strategy": PipelineStrategy.SPEED_OPTIMIZED,
                "additional_steps": ["contrast_enhancement"]
            },
            "paddleocr": {
                "processing_level": PrepLevel.BALANCED,
                "strategy": PipelineStrategy.CONTENT_AWARE,
                "additional_steps": ["structure_preservation"]
            },
            "trocr": {
                "processing_level": PrepLevel.INTENSIVE,
                "strategy": PipelineStrategy.QUALITY_OPTIMIZED,
                "additional_steps": ["handwriting_optimization"]
            }
        }
        
        logger.info("Enhanced Engine Manager with preprocessing initialized")
    
    def process_image(self, image: np.ndarray, 
                     options: Optional[EnhancedProcessingOptions] = None) -> Dict[str, Any]:
        """
        Enhanced image processing with integrated preprocessing
        
        Args:
            image: Input image as numpy array
            options: Enhanced processing options
            
        Returns:
            Dictionary with comprehensive results
        """
        options = options or EnhancedProcessingOptions()
        start_time = time.time()
        
        results = {
            "success": False,
            "engines_used": [],
            "preprocessing_results": {},
            "ocr_results": {},
            "combined_result": None,
            "processing_time": 0.0,
            "warnings": [],
            "metadata": {}
        }
        
        try:
            # Step 1: Preprocessing (if enabled)
            preprocessed_images = {}
            
            if options.enable_preprocessing:
                logger.info("Starting adaptive preprocessing...")
                
                if options.preprocess_per_engine:
                    # Different preprocessing for each engine
                    for engine_name in options.engines:
                        preprocessed_images[engine_name] = self._preprocess_for_engine(
                            image, engine_name, options
                        )
                else:
                    # Single preprocessing for all engines
                    single_preprocessing = self._preprocess_image(image, options)
                    for engine_name in options.engines:
                        preprocessed_images[engine_name] = single_preprocessing
                
                results["preprocessing_results"] = {
                    engine: {
                        "success": result.success,
                        "processing_time": result.processing_time,
                        "quality_improvement": result.metadata.get("quality_improvement", 0),
                        "pipeline_used": result.metadata.get("pipeline_used"),
                        "warnings": result.warnings
                    }
                    for engine, result in preprocessed_images.items()
                }
            else:
                # Use original image for all engines
                for engine_name in options.engines:
                    preprocessed_images[engine_name] = type('MockResult', (), {
                        'processed_image': image, 'success': True, 'warnings': []
                    })()
            
            # Step 2: OCR Processing
            logger.info(f"Processing with engines: {', '.join(options.engines)}")
            
            if options.parallel_processing and len(options.engines) > 1:
                ocr_results = self._parallel_ocr_processing(preprocessed_images, options)
            else:
                ocr_results = self._sequential_ocr_processing(preprocessed_images, options)
            
            results["ocr_results"] = ocr_results
            results["engines_used"] = list(ocr_results.keys())
            
            # Step 3: Combine results
            combined_result = self._combine_ocr_results(ocr_results, options)
            results["combined_result"] = combined_result
            results["success"] = combined_result is not None
            
            # Step 4: Validation and quality checks
            if options.validate_preprocessing and results["success"]:
                validation_results = self._validate_results(
                    image, preprocessed_images, ocr_results, combined_result
                )
                results["validation"] = validation_results
                
                # Apply fallback if needed
                if (validation_results.get("quality_degraded", False) and 
                    options.fallback_to_original):
                    logger.warning("Quality degradation detected, falling back to original image")
                    fallback_results = self._fallback_processing(image, options)
                    if fallback_results and fallback_results.confidence > combined_result.confidence:
                        results["combined_result"] = fallback_results
                        results["warnings"].append("Used fallback to original image")
            
            # Calculate total processing time
            results["processing_time"] = time.time() - start_time
            
            # Add metadata
            results["metadata"] = {
                "original_image_size": image.shape,
                "preprocessing_enabled": options.enable_preprocessing,
                "engines_count": len(options.engines),
                "parallel_processing": options.parallel_processing,
                "total_processing_time": results["processing_time"]
            }
            
            logger.info(f"Enhanced processing completed in {results['processing_time']:.2f}s")
            
        except Exception as e:
            logger.error(f"Enhanced processing failed: {e}")
            results["success"] = False
            results["warnings"].append(f"Processing failed: {str(e)}")
            results["processing_time"] = time.time() - start_time
        
        return results
    
    def _preprocess_for_engine(self, image: np.ndarray, engine_name: str,
                              options: EnhancedProcessingOptions) -> ProcessingResult:
        """Preprocess image specifically for a given OCR engine"""
        
        engine_config = self.engine_preprocessing_configs.get(
            engine_name, self.engine_preprocessing_configs["paddleocr"]
        )
        
        # Create engine-specific processing options
        processing_options = ProcessingOptions(
            processing_level=engine_config["processing_level"],
            strategy=engine_config["strategy"],
            enable_quality_validation=options.validate_preprocessing,
            max_processing_iterations=2 if engine_name == "trocr" else 1,
            processing_timeout=options.max_processing_time / len(options.engines)
        )
        
        # Process image
        result = self.preprocessor.process_image(image, processing_options, f"{engine_name}_prep")
        
        logger.debug(f"Preprocessing for {engine_name}: "
                    f"quality improvement {result.metadata.get('quality_improvement', 0):.3f}")
        
        return result
    
    def _preprocess_image(self, image: np.ndarray, 
                         options: EnhancedProcessingOptions) -> ProcessingResult:
        """Single preprocessing for all engines"""
        
        processing_options = ProcessingOptions(
            processing_level=options.preprocessing_level,
            strategy=options.preprocessing_strategy,
            enable_quality_validation=options.validate_preprocessing,
            max_processing_iterations=2,
            processing_timeout=options.max_processing_time * 0.3  # 30% of total time for preprocessing
        )
        
        return self.preprocessor.process_image(image, processing_options, "general_prep")
    
    def _parallel_ocr_processing(self, preprocessed_images: Dict[str, ProcessingResult],
                               options: EnhancedProcessingOptions) -> Dict[str, OCRResult]:
        """Process images with multiple OCR engines in parallel"""
        
        ocr_results = {}
        
        with ThreadPoolExecutor(max_workers=len(options.engines)) as executor:
            # Submit OCR tasks
            future_to_engine = {}
            
            for engine_name in options.engines:
                if engine_name in self.engines:
                    preprocessed_result = preprocessed_images[engine_name]
                    if preprocessed_result.success:
                        future = executor.submit(
                            self._run_single_engine,
                            engine_name,
                            preprocessed_result.processed_image,
                            options
                        )
                        future_to_engine[future] = engine_name
            
            # Collect results
            for future in as_completed(future_to_engine):
                engine_name = future_to_engine[future]
                try:
                    result = future.result(timeout=options.max_processing_time)
                    if result:
                        ocr_results[engine_name] = result
                        logger.info(f"{engine_name} completed: confidence {result.confidence:.3f}")
                except Exception as e:
                    logger.error(f"{engine_name} failed: {e}")
                    ocr_results[engine_name] = None
        
        return ocr_results
    
    def _sequential_ocr_processing(self, preprocessed_images: Dict[str, ProcessingResult],
                                 options: EnhancedProcessingOptions) -> Dict[str, OCRResult]:
        """Process images with OCR engines sequentially"""
        
        ocr_results = {}
        
        for engine_name in options.engines:
            if engine_name in self.engines:
                preprocessed_result = preprocessed_images[engine_name]
                if preprocessed_result.success:
                    try:
                        result = self._run_single_engine(
                            engine_name, preprocessed_result.processed_image, options
                        )
                        if result:
                            ocr_results[engine_name] = result
                            logger.info(f"{engine_name} completed: confidence {result.confidence:.3f}")
                    except Exception as e:
                        logger.error(f"{engine_name} failed: {e}")
                        ocr_results[engine_name] = None
        
        return ocr_results
    
    def _run_single_engine(self, engine_name: str, image: np.ndarray,
                          options: EnhancedProcessingOptions) -> Optional[OCRResult]:
        """Run a single OCR engine on preprocessed image"""
        
        engine = self.engines[engine_name]
        
        try:
            # Set processing level based on preprocessing results
            processing_level = ProcessingLevel.BALANCED
            
            # Run OCR
            result = engine.extract_text(image, processing_level)
            
            # Filter by confidence threshold
            if result and result.confidence >= options.confidence_threshold:
                return result
            else:
                logger.warning(f"{engine_name} result below confidence threshold: "
                             f"{result.confidence if result else 0:.3f}")
                return None
                
        except Exception as e:
            logger.error(f"Error running {engine_name}: {e}")
            return None
    
    def _combine_ocr_results(self, ocr_results: Dict[str, OCRResult],
                           options: EnhancedProcessingOptions) -> Optional[OCRResult]:
        """Combine results from multiple OCR engines"""
        
        valid_results = {k: v for k, v in ocr_results.items() if v is not None}
        
        if not valid_results:
            return None
        
        if len(valid_results) == 1:
            return list(valid_results.values())[0]
        
        # Use the existing combination logic from parent class
        return super().combine_results(list(valid_results.values()))
    
    def _validate_results(self, original_image: np.ndarray,
                         preprocessed_images: Dict[str, ProcessingResult],
                         ocr_results: Dict[str, OCRResult],
                         combined_result: OCRResult) -> Dict[str, Any]:
        """Validate preprocessing and OCR results"""
        
        validation = {
            "preprocessing_success_rate": 0.0,
            "ocr_success_rate": 0.0,
            "quality_degraded": False,
            "confidence_improvement": 0.0,
            "processing_efficiency": 0.0
        }
        
        try:
            # Preprocessing validation
            successful_preprocessing = sum(
                1 for result in preprocessed_images.values() if result.success
            )
            validation["preprocessing_success_rate"] = (
                successful_preprocessing / len(preprocessed_images) if preprocessed_images else 0
            )
            
            # OCR validation
            successful_ocr = sum(1 for result in ocr_results.values() if result is not None)
            validation["ocr_success_rate"] = (
                successful_ocr / len(ocr_results) if ocr_results else 0
            )
            
            # Quality degradation check (simplified)
            if combined_result:
                # Check if combined confidence is reasonable
                individual_confidences = [r.confidence for r in ocr_results.values() if r]
                if individual_confidences:
                    avg_individual = sum(individual_confidences) / len(individual_confidences)
                    validation["confidence_improvement"] = combined_result.confidence - avg_individual
                    validation["quality_degraded"] = combined_result.confidence < avg_individual * 0.8
            
            # Processing efficiency (simple metric)
            total_text_length = len(combined_result.text) if combined_result else 0
            if total_text_length > 0:
                preprocessing_time = sum(
                    r.processing_time for r in preprocessed_images.values()
                )
                # Characters per second as efficiency metric
                validation["processing_efficiency"] = total_text_length / (preprocessing_time + 1e-6)
            
        except Exception as e:
            logger.warning(f"Result validation failed: {e}")
            validation["validation_error"] = str(e)
        
        return validation
    
    def _fallback_processing(self, original_image: np.ndarray,
                           options: EnhancedProcessingOptions) -> Optional[OCRResult]:
        """Fallback processing using original image without preprocessing"""
        
        logger.info("Running fallback processing with original image")
        
        try:
            # Use the best available engine
            best_engine_name = self._get_best_engine(options.engines)
            if best_engine_name and best_engine_name in self.engines:
                engine = self.engines[best_engine_name]
                result = engine.extract_text(original_image, ProcessingLevel.BALANCED)
                
                if result:
                    logger.info(f"Fallback processing successful: confidence {result.confidence:.3f}")
                return result
                
        except Exception as e:
            logger.error(f"Fallback processing failed: {e}")
        
        return None
    
    def _get_best_engine(self, available_engines: List[str]) -> Optional[str]:
        """Get the best available engine based on historical performance"""
        
        # Priority order based on general performance
        priority_order = ["paddleocr", "easyocr", "tesseract", "trocr"]
        
        for engine in priority_order:
            if engine in available_engines and engine in self.engines:
                return engine
        
        # Return first available if no priority match
        for engine in available_engines:
            if engine in self.engines:
                return engine
        
        return None
    
    def get_preprocessing_statistics(self) -> Dict[str, Any]:
        """Get preprocessing statistics"""
        return self.preprocessor.get_processing_statistics()
    
    def configure_preprocessing(self, config: Dict[str, Any]):
        """Configure preprocessing settings"""
        self.preprocessor.configure_component("all", config)
    
    def shutdown(self):
        """Shutdown the enhanced engine manager"""
        super().shutdown()
        if self.preprocessor:
            self.preprocessor.shutdown()
        logger.info("Enhanced Engine Manager shutdown complete")

# Convenience functions for easy integration

def create_enhanced_manager(config_path: Optional[str] = None) -> EnhancedEngineManager:
    """Create an enhanced engine manager with default settings"""
    return EnhancedEngineManager(config_path)

def quick_ocr_with_preprocessing(image: np.ndarray, 
                                engines: List[str] = None,
                                preprocessing_level: PrepLevel = PrepLevel.BALANCED) -> str:
    """Quick OCR processing with preprocessing"""
    engines = engines or ["paddleocr", "easyocr"]
    
    manager = EnhancedEngineManager()
    options = EnhancedProcessingOptions(
        engines=engines,
        preprocessing_level=preprocessing_level,
        enable_preprocessing=True
    )
    
    try:
        results = manager.process_image(image, options)
        if results["success"] and results["combined_result"]:
            return results["combined_result"].text
        else:
            return ""
    finally:
        manager.shutdown()

def batch_ocr_with_preprocessing(images: List[np.ndarray],
                               engines: List[str] = None,
                               preprocessing_level: PrepLevel = PrepLevel.BALANCED,
                               progress_callback: Optional[callable] = None) -> List[str]:
    """Batch OCR processing with preprocessing"""
    engines = engines or ["paddleocr", "easyocr"]
    
    manager = EnhancedEngineManager()
    options = EnhancedProcessingOptions(
        engines=engines,
        preprocessing_level=preprocessing_level,
        enable_preprocessing=True,
        parallel_processing=True
    )
    
    texts = []
    
    try:
        for i, image in enumerate(images):
            results = manager.process_image(image, options)
            if results["success"] and results["combined_result"]:
                texts.append(results["combined_result"].text)
            else:
                texts.append("")
            
            if progress_callback:
                progress_callback(i + 1, len(images))
        
        return texts
        
    finally:
        manager.shutdown()