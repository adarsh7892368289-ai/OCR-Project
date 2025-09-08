# src/preprocessing/adaptive_processor.py - Step 4: Intelligent Preprocessing Pipeline

import cv2
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging
import time
import threading
from pathlib import Path
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

# Import your existing components
from .quality_analyzer import IntelligentQualityAnalyzer, QualityMetrics, ImageType, ImageQuality
from .image_enhancer import AIImageEnhancer, EnhancementResult, EnhancementStrategy
from .skew_corrector import EnhancedSkewCorrector, SkewDetectionResult, SkewCorrectionResult
from ..utils.config import ConfigManager
from ..utils.logger import setup_logger

logger = logging.getLogger(__name__)

class ProcessingLevel(Enum):
    """Processing intensity levels"""
    MINIMAL = "minimal"         # Basic corrections only
    LIGHT = "light"            # Light enhancement
    BALANCED = "balanced"      # Standard processing
    INTENSIVE = "intensive"    # Heavy processing
    MAXIMUM = "maximum"        # All available enhancements

class PipelineStrategy(Enum):
    """Processing pipeline strategies"""
    SPEED_OPTIMIZED = "speed_optimized"
    QUALITY_OPTIMIZED = "quality_optimized" 
    CONTENT_AWARE = "content_aware"
    CUSTOM = "custom"

@dataclass
class ProcessingOptions:
    """Options for preprocessing pipeline"""
    processing_level: ProcessingLevel = ProcessingLevel.BALANCED
    strategy: PipelineStrategy = PipelineStrategy.CONTENT_AWARE
    target_engines: List[str] = field(default_factory=lambda: ["paddleocr", "easyocr"])
    enable_quality_validation: bool = True
    max_processing_iterations: int = 3
    quality_improvement_threshold: float = 0.05
    processing_timeout: float = 60.0
    preserve_original: bool = True
    enable_statistics: bool = True

@dataclass 
class ProcessingStep:
    """Individual processing step definition"""
    name: str
    function: Callable
    parameters: Dict[str, Any] = field(default_factory=dict)
    conditions: Dict[str, Any] = field(default_factory=dict)
    priority: int = 10
    enabled: bool = True
    timeout: float = 30.0

@dataclass
class ProcessingResult:
    """Result of preprocessing pipeline"""
    processed_image: np.ndarray
    original_image: np.ndarray
    processing_steps: List[str]
    quality_metrics: Dict[str, Any]
    processing_time: float
    success: bool
    warnings: List[str]
    metadata: Dict[str, Any]
    performance_stats: Dict[str, Any]

class AdaptivePreprocessor:
    """
    Intelligent preprocessing pipeline that adapts to image content and quality
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the adaptive preprocessor
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration - FIXED VERSION
        self.config_manager = ConfigManager()
        
        if config_path:
            try:
                self.config_manager.load_config(config_path)
                self.config = self.config_manager.get_config()  # ✅ GET the loaded config
            except Exception as e:
                logger.warning(f"Failed to load config from {config_path}: {e}")
                self.config = self._get_default_config()
        else:
            self.config = self._get_default_config()
        
        # Ensure config is not None
        if self.config is None:
            logger.warning("Config is None, using default config")
            self.config = self._get_default_config()
        
        # Initialize components - now self.config is guaranteed to be a dict
        self.quality_analyzer = IntelligentQualityAnalyzer(
            self.config.get("quality_analyzer", {})
        )
        self.image_enhancer = AIImageEnhancer(
            self.config.get("image_enhancer", {})
        )
        self.skew_corrector = EnhancedSkewCorrector(
            self.config.get("skew_corrector", {})
        )
        
        # Rest of initialization...
        self.pipelines = self._initialize_pipelines()
        self.custom_pipelines = {}
        
        # Statistics and monitoring
        self.processing_stats = {
            "total_processed": 0,
            "successful_processing": 0,
            "average_processing_time": 0.0,
            "quality_improvements": 0,
            "pipeline_usage": {},
            "performance_metrics": {}
        }
        
        # Threading for parallel processing
        self.max_workers = self.config.get("system", {}).get("max_workers", 4)
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        
        logger.info("Adaptive Preprocessor initialized")
    
    def process_image(self, image: np.ndarray, 
                     options: Optional[ProcessingOptions] = None,
                     cache_key: Optional[str] = None) -> ProcessingResult:
        """
        Main processing pipeline with intelligent adaptation
        
        Args:
            image: Input image as numpy array
            options: Processing options
            cache_key: Optional cache key
            
        Returns:
            ProcessingResult with processed image and metadata
        """
        start_time = time.time()
        options = options or ProcessingOptions()
        warnings = []
        
        try:
            # Step 1: Quality analysis
            logger.info("Starting adaptive preprocessing pipeline")
            quality_metrics = self.quality_analyzer.analyze_image(image, cache_key)
            
            # Step 2: Select optimal pipeline
            pipeline = self._select_pipeline(quality_metrics, options)
            logger.info(f"Selected pipeline: {pipeline['name']} for {quality_metrics.image_type.value}")
            
            # Step 3: Execute preprocessing pipeline
            processed_image, processing_steps, step_warnings = self._execute_pipeline(
                image, pipeline, quality_metrics, options
            )
            warnings.extend(step_warnings)
            
            # Step 4: Quality validation (if enabled)
            if options.enable_quality_validation:
                final_quality = self.quality_analyzer.analyze_image(processed_image)
                
                # Check for improvement
                improvement = final_quality.overall_score - quality_metrics.overall_score
                if improvement < options.quality_improvement_threshold and options.max_processing_iterations > 1:
                    # Try iterative improvement
                    processed_image, additional_steps, iter_warnings = self._iterative_improvement(
                        processed_image, quality_metrics, final_quality, options
                    )
                    processing_steps.extend(additional_steps)
                    warnings.extend(iter_warnings)
            else:
                final_quality = quality_metrics
            
            # Step 5: Finalize results
            processing_time = time.time() - start_time
            success = len(warnings) == 0 or all("warning" in w.lower() for w in warnings)
            
            # Update statistics
            self._update_statistics(pipeline["name"], processing_time, 
                                  final_quality.overall_score - quality_metrics.overall_score)
            
            # Prepare metadata
            metadata = {
                "original_quality": quality_metrics,
                "final_quality": final_quality,
                "pipeline_used": pipeline["name"],
                "processing_level": options.processing_level.value,
                "strategy": options.strategy.value,
                "quality_improvement": final_quality.overall_score - quality_metrics.overall_score
            }
            
            # Performance stats
            performance_stats = {
                "processing_time": processing_time,
                "steps_executed": len(processing_steps),
                "quality_change": final_quality.overall_score - quality_metrics.overall_score,
                "memory_usage": self._estimate_memory_usage(image, processed_image)
            }
            
            result = ProcessingResult(
                processed_image=processed_image,
                original_image=image.copy() if options.preserve_original else None,
                processing_steps=processing_steps,
                quality_metrics={
                    "original": quality_metrics.__dict__,
                    "final": final_quality.__dict__
                },
                processing_time=processing_time,
                success=success,
                warnings=warnings,
                metadata=metadata,
                performance_stats=performance_stats
            )
            
            logger.info(f"Preprocessing completed in {processing_time:.2f}s "
                       f"(quality: {quality_metrics.overall_score:.3f} → {final_quality.overall_score:.3f})")
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Preprocessing pipeline failed: {e}")
            
            return ProcessingResult(
                processed_image=image.copy(),
                original_image=image.copy() if options.preserve_original else None,
                processing_steps=["error_fallback"],
                quality_metrics={},
                processing_time=processing_time,
                success=False,
                warnings=[f"Pipeline failed: {str(e)}"],
                metadata={"error": True},
                performance_stats={"processing_time": processing_time}
            )
    
    def process_batch(self, images: List[np.ndarray], 
                     options: Optional[ProcessingOptions] = None,
                     progress_callback: Optional[Callable] = None) -> List[ProcessingResult]:
        """
        Process multiple images in parallel
        
        Args:
            images: List of images to process
            options: Processing options
            progress_callback: Optional progress callback function
            
        Returns:
            List of ProcessingResult objects
        """
        options = options or ProcessingOptions()
        results = []
        
        logger.info(f"Starting batch processing of {len(images)} images")
        
        # Submit all tasks
        future_to_index = {}
        for i, image in enumerate(images):
            future = self.executor.submit(self.process_image, image, options, f"batch_{i}")
            future_to_index[future] = i
        
        # Collect results
        completed = 0
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            try:
                result = future.result(timeout=options.processing_timeout)
                results.append((index, result))
            except Exception as e:
                logger.error(f"Batch processing failed for image {index}: {e}")
                # Create error result
                error_result = ProcessingResult(
                    processed_image=images[index].copy(),
                    original_image=images[index].copy(),
                    processing_steps=["batch_error"],
                    quality_metrics={},
                    processing_time=0.0,
                    success=False,
                    warnings=[f"Batch processing failed: {str(e)}"],
                    metadata={"batch_error": True},
                    performance_stats={}
                )
                results.append((index, error_result))
            
            completed += 1
            if progress_callback:
                progress_callback(completed, len(images))
        
        # Sort by original index
        results.sort(key=lambda x: x[0])
        return [result for _, result in results]
    
    def _select_pipeline(self, quality_metrics: QualityMetrics, 
                        options: ProcessingOptions) -> Dict[str, Any]:
        """Select optimal pipeline based on analysis"""
        
        if options.strategy == PipelineStrategy.SPEED_OPTIMIZED:
            return self.pipelines["speed_optimized"]
        elif options.strategy == PipelineStrategy.QUALITY_OPTIMIZED:
            return self.pipelines["quality_optimized"]
        elif options.strategy == PipelineStrategy.CUSTOM:
            # Use custom pipeline if available
            custom_name = f"custom_{quality_metrics.image_type.value}"
            return self.custom_pipelines.get(custom_name, self.pipelines["balanced"])
        else:  # CONTENT_AWARE
            return self._select_content_aware_pipeline(quality_metrics, options)
    
    def _select_content_aware_pipeline(self, quality_metrics: QualityMetrics,
                                     options: ProcessingOptions) -> Dict[str, Any]:
        """Select pipeline based on content analysis"""
        
        # Image type specific pipelines
        if quality_metrics.image_type == ImageType.HANDWRITTEN_TEXT:
            return self.pipelines["handwriting_optimized"]
        elif quality_metrics.image_type == ImageType.TABLE_DOCUMENT:
            return self.pipelines["table_optimized"]
        elif quality_metrics.image_type == ImageType.FORM_DOCUMENT:
            return self.pipelines["form_optimized"]
        elif quality_metrics.image_type == ImageType.LOW_QUALITY:
            return self.pipelines["restoration_focused"]
        elif quality_metrics.quality_level == ImageQuality.VERY_POOR:
            return self.pipelines["aggressive_enhancement"]
        elif quality_metrics.quality_level in [ImageQuality.GOOD, ImageQuality.EXCELLENT]:
            return self.pipelines["conservative_enhancement"]
        else:
            return self.pipelines["balanced"]
    
    def _execute_pipeline(self, image: np.ndarray, pipeline: Dict[str, Any],
                         quality_metrics: QualityMetrics, 
                         options: ProcessingOptions) -> Tuple[np.ndarray, List[str], List[str]]:
        """Execute the selected pipeline"""
        
        current_image = image.copy()
        processing_steps = []
        warnings = []
        
        # Get pipeline steps
        steps = pipeline["steps"]
        
        for step_config in steps:
            try:
                step_name = step_config["name"]
                step_params = step_config.get("parameters", {})
                
                # Check if step should be executed
                if not self._should_execute_step(step_config, quality_metrics, current_image):
                    continue
                
                # Execute step
                step_start = time.time()
                
                if step_name == "skew_correction":
                    current_image, step_warnings = self._execute_skew_correction(
                        current_image, step_params
                    )
                elif step_name == "enhancement":
                    current_image, step_warnings = self._execute_enhancement(
                        current_image, quality_metrics, step_params
                    )
                elif step_name == "noise_reduction":
                    current_image, step_warnings = self._execute_noise_reduction(
                        current_image, step_params
                    )
                elif step_name == "contrast_enhancement":
                    current_image, step_warnings = self._execute_contrast_enhancement(
                        current_image, step_params
                    )
                elif step_name == "sharpening":
                    current_image, step_warnings = self._execute_sharpening(
                        current_image, step_params
                    )
                elif step_name == "morphological_cleaning":
                    current_image, step_warnings = self._execute_morphological_cleaning(
                        current_image, step_params
                    )
                elif step_name == "custom":
                    current_image, step_warnings = self._execute_custom_step(
                        current_image, step_params
                    )
                else:
                    step_warnings = [f"Unknown step: {step_name}"]
                
                step_time = time.time() - step_start
                processing_steps.append(f"{step_name} ({step_time:.2f}s)")
                warnings.extend(step_warnings)
                
                logger.debug(f"Executed step '{step_name}' in {step_time:.2f}s")
                
            except Exception as e:
                warning_msg = f"Step '{step_name}' failed: {str(e)}"
                warnings.append(warning_msg)
                logger.warning(warning_msg)
        
        return current_image, processing_steps, warnings
    
    def _should_execute_step(self, step_config: Dict[str, Any], 
                           quality_metrics: QualityMetrics,
                           current_image: np.ndarray) -> bool:
        """Determine if a step should be executed based on conditions"""
        
        conditions = step_config.get("conditions", {})
        
        # Check image type conditions
        if "image_types" in conditions:
            if quality_metrics.image_type.value not in conditions["image_types"]:
                return False
        
        # Check quality conditions
        if "min_quality" in conditions:
            if quality_metrics.overall_score < conditions["min_quality"]:
                return False
        
        if "max_quality" in conditions:
            if quality_metrics.overall_score > conditions["max_quality"]:
                return False
        
        # Check specific metric conditions
        if "requires_skew_correction" in conditions:
            if abs(quality_metrics.skew_angle) < conditions["requires_skew_correction"]:
                return False
        
        if "requires_noise_reduction" in conditions:
            if quality_metrics.noise_level < conditions["requires_noise_reduction"]:
                return False
        
        return True
    
    def _execute_skew_correction(self, image: np.ndarray, 
                               params: Dict[str, Any]) -> Tuple[np.ndarray, List[str]]:
        """Execute skew correction step"""
        try:
            result = self.skew_corrector.correct_skew(image, **params)
            return result.corrected_image, result.warnings
        except Exception as e:
            return image, [f"Skew correction failed: {str(e)}"]
    
    def _execute_enhancement(self, image: np.ndarray, quality_metrics: QualityMetrics,
                           params: Dict[str, Any]) -> Tuple[np.ndarray, List[str]]:
        """Execute image enhancement step"""
        try:
            strategy = params.get("strategy", None)
            if strategy:
                strategy = EnhancementStrategy(strategy)
            result = self.image_enhancer.enhance_image(image, strategy)
            return result.enhanced_image, result.warnings
        except Exception as e:
            return image, [f"Enhancement failed: {str(e)}"]
    
    def _execute_noise_reduction(self, image: np.ndarray, 
                               params: Dict[str, Any]) -> Tuple[np.ndarray, List[str]]:
        """Execute noise reduction step"""
        try:
            method = params.get("method", "bilateral")
            if method == "bilateral":
                result = cv2.bilateralFilter(image, 9, 75, 75)
            elif method == "non_local_means":
                if len(image.shape) == 3:
                    result = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
                else:
                    result = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
            else:
                result = cv2.GaussianBlur(image, (5, 5), 0)
            
            return result, []
        except Exception as e:
            return image, [f"Noise reduction failed: {str(e)}"]
    
    def _execute_contrast_enhancement(self, image: np.ndarray,
                                    params: Dict[str, Any]) -> Tuple[np.ndarray, List[str]]:
        """Execute contrast enhancement step"""
        try:
            method = params.get("method", "clahe")
            if method == "clahe":
                if len(image.shape) == 3:
                    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                    l, a, b = cv2.split(lab)
                    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                    l = clahe.apply(l)
                    result = cv2.merge([l, a, b])
                    result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
                else:
                    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                    result = clahe.apply(image)
            else:
                result = cv2.equalizeHist(image)
            
            return result, []
        except Exception as e:
            return image, [f"Contrast enhancement failed: {str(e)}"]
    
    def _execute_sharpening(self, image: np.ndarray,
                          params: Dict[str, Any]) -> Tuple[np.ndarray, List[str]]:
        """Execute sharpening step"""
        try:
            strength = params.get("strength", 0.5)
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]) * strength
            
            sharpened = cv2.filter2D(image, -1, kernel)
            result = cv2.addWeighted(image, 1-strength, sharpened, strength, 0)
            
            return result, []
        except Exception as e:
            return image, [f"Sharpening failed: {str(e)}"]
    
    def _execute_morphological_cleaning(self, image: np.ndarray,
                                      params: Dict[str, Any]) -> Tuple[np.ndarray, List[str]]:
        """Execute morphological cleaning step"""
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Opening to remove noise
            kernel_size = params.get("kernel_size", 2)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            cleaned = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
            
            # Closing to fill gaps
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
            
            return cleaned, []
        except Exception as e:
            return image, [f"Morphological cleaning failed: {str(e)}"]
    
    def _execute_custom_step(self, image: np.ndarray,
                           params: Dict[str, Any]) -> Tuple[np.ndarray, List[str]]:
        """Execute custom processing step"""
        try:
            # Placeholder for custom processing steps
            # Users can extend this method for their specific needs
            function_name = params.get("function")
            if function_name and hasattr(self, function_name):
                function = getattr(self, function_name)
                return function(image, params)
            else:
                return image, [f"Custom step function '{function_name}' not found"]
        except Exception as e:
            return image, [f"Custom step failed: {str(e)}"]
    
    def _iterative_improvement(self, image: np.ndarray, original_quality: QualityMetrics,
                              current_quality: QualityMetrics, 
                              options: ProcessingOptions) -> Tuple[np.ndarray, List[str], List[str]]:
        """Attempt iterative improvement if quality gains are minimal"""
        
        warnings = []
        additional_steps = []
        current_image = image.copy()
        
        for iteration in range(1, options.max_processing_iterations):
            logger.info(f"Attempting iterative improvement {iteration}")
            
            # Analyze what needs improvement
            improvement_pipeline = self._create_improvement_pipeline(
                original_quality, current_quality
            )
            
            if not improvement_pipeline["steps"]:
                break
            
            # Execute improvement pipeline
            improved_image, iter_steps, iter_warnings = self._execute_pipeline(
                current_image, improvement_pipeline, current_quality, options
            )
            
            # Check if we got improvement
            new_quality = self.quality_analyzer.analyze_image(improved_image)
            improvement = new_quality.overall_score - current_quality.overall_score
            
            if improvement > options.quality_improvement_threshold:
                current_image = improved_image
                current_quality = new_quality
                additional_steps.extend([f"iteration_{iteration}_{step}" for step in iter_steps])
                warnings.extend(iter_warnings)
                logger.info(f"Iteration {iteration} improved quality by {improvement:.3f}")
            else:
                logger.info(f"Iteration {iteration} did not improve quality significantly")
                break
        
        return current_image, additional_steps, warnings
    
    def _create_improvement_pipeline(self, original_quality: QualityMetrics,
                                   current_quality: QualityMetrics) -> Dict[str, Any]:
        """Create pipeline for iterative improvement"""
        
        steps = []
        
        # Check what still needs improvement
        if current_quality.sharpness_score < 0.6:
            steps.append({
                "name": "sharpening",
                "parameters": {"strength": 0.3},
                "conditions": {}
            })
        
        if current_quality.noise_level > 0.4:
            steps.append({
                "name": "noise_reduction",
                "parameters": {"method": "non_local_means"},
                "conditions": {}
            })
        
        if current_quality.contrast_score < 0.5:
            steps.append({
                "name": "contrast_enhancement", 
                "parameters": {"method": "clahe"},
                "conditions": {}
            })
        
        return {
            "name": "iterative_improvement",
            "description": "Iterative quality improvement",
            "steps": steps
        }
    
    def _initialize_pipelines(self) -> Dict[str, Any]:
        """Initialize built-in processing pipelines"""
        
        pipelines = {
            "speed_optimized": {
                "name": "speed_optimized",
                "description": "Fast processing with basic corrections",
                "steps": [
                    {
                        "name": "skew_correction",
                        "parameters": {"quality": "fast"},
                        "conditions": {"requires_skew_correction": 1.0}
                    },
                    {
                        "name": "enhancement",
                        "parameters": {"strategy": "conservative"},
                        "conditions": {}
                    }
                ]
            },
            
            "quality_optimized": {
                "name": "quality_optimized", 
                "description": "Maximum quality processing",
                "steps": [
                    {
                        "name": "skew_correction",
                        "parameters": {"quality": "preserve_quality"},
                        "conditions": {"requires_skew_correction": 0.5}
                    },
                    {
                        "name": "enhancement",
                        "parameters": {"strategy": "aggressive"},
                        "conditions": {}
                    },
                    {
                        "name": "noise_reduction",
                        "parameters": {"method": "non_local_means"},
                        "conditions": {"requires_noise_reduction": 0.2}
                    },
                    {
                        "name": "morphological_cleaning",
                        "parameters": {"kernel_size": 2},
                        "conditions": {}
                    }
                ]
            },
            
            "balanced": {
                "name": "balanced",
                "description": "Balanced processing pipeline",
                "steps": [
                    {
                        "name": "skew_correction",
                        "parameters": {"quality": "balanced"},
                        "conditions": {"requires_skew_correction": 1.0}
                    },
                    {
                        "name": "enhancement", 
                        "parameters": {"strategy": "balanced"},
                        "conditions": {}
                    },
                    {
                        "name": "noise_reduction",
                        "parameters": {"method": "bilateral"},
                        "conditions": {"requires_noise_reduction": 0.3}
                    }
                ]
            },
            
            "handwriting_optimized": {
                "name": "handwriting_optimized",
                "description": "Optimized for handwritten text",
                "steps": [
                    {
                        "name": "skew_correction",
                        "parameters": {"quality": "high_quality"},
                        "conditions": {"requires_skew_correction": 0.5}
                    },
                    {
                        "name": "enhancement",
                        "parameters": {"strategy": "conservative"},
                        "conditions": {}
                    },
                    {
                        "name": "noise_reduction",
                        "parameters": {"method": "bilateral"},
                        "conditions": {"requires_noise_reduction": 0.2}
                    }
                ]
            },
            
            "table_optimized": {
                "name": "table_optimized",
                "description": "Optimized for table documents",
                "steps": [
                    {
                        "name": "skew_correction",
                        "parameters": {"quality": "high_quality"},
                        "conditions": {"requires_skew_correction": 0.5}
                    },
                    {
                        "name": "enhancement",
                        "parameters": {"strategy": "balanced"},
                        "conditions": {}
                    },
                    {
                        "name": "contrast_enhancement",
                        "parameters": {"method": "clahe"},
                        "conditions": {}
                    },
                    {
                        "name": "morphological_cleaning",
                        "parameters": {"kernel_size": 1},
                        "conditions": {}
                    }
                ]
            },
            
            "form_optimized": {
                "name": "form_optimized", 
                "description": "Optimized for form documents",
                "steps": [
                    {
                        "name": "skew_correction",
                        "parameters": {"quality": "balanced"},
                        "conditions": {"requires_skew_correction": 1.0}
                    },
                    {
                        "name": "enhancement",
                        "parameters": {"strategy": "balanced"},
                        "conditions": {}
                    },
                    {
                        "name": "contrast_enhancement",
                        "parameters": {"method": "clahe"}, 
                        "conditions": {}
                    }
                ]
            },
            
            "restoration_focused": {
                "name": "restoration_focused",
                "description": "Heavy restoration for low quality images", 
                "steps": [
                    {
                        "name": "skew_correction",
                        "parameters": {"quality": "balanced"},
                        "conditions": {"requires_skew_correction": 0.5}
                    },
                    {
                        "name": "enhancement",
                        "parameters": {"strategy": "aggressive"},
                        "conditions": {}
                    },
                    {
                        "name": "noise_reduction",
                        "parameters": {"method": "non_local_means"},
                        "conditions": {}
                    },
                    {
                        "name": "contrast_enhancement",
                        "parameters": {"method": "clahe"},
                        "conditions": {}
                    },
                    {
                        "name": "sharpening",
                        "parameters": {"strength": 0.6},
                        "conditions": {}
                    },
                    {
                        "name": "morphological_cleaning",
                        "parameters": {"kernel_size": 3},
                        "conditions": {}
                    }
                ]
            },
            
            "aggressive_enhancement": {
                "name": "aggressive_enhancement",
                "description": "Aggressive enhancement for very poor quality images",
                "steps": [
                    {
                        "name": "skew_correction",
                        "parameters": {"quality": "high_quality"},
                        "conditions": {"requires_skew_correction": 0.5}
                    },
                    {
                        "name": "enhancement",
                        "parameters": {"strategy": "aggressive"},
                        "conditions": {}
                    },
                    {
                        "name": "noise_reduction", 
                        "parameters": {"method": "non_local_means"},
                        "conditions": {}
                    },
                    {
                        "name": "contrast_enhancement",
                        "parameters": {"method": "clahe"},
                        "conditions": {}
                    },
                    {
                        "name": "sharpening",
                        "parameters": {"strength": 0.8},
                        "conditions": {}
                    }
                ]
            },
            
            "conservative_enhancement": {
                "name": "conservative_enhancement", 
                "description": "Light enhancement for already good quality images",
                "steps": [
                    {
                        "name": "skew_correction",
                        "parameters": {"quality": "balanced"},
                        "conditions": {"requires_skew_correction": 2.0}
                    },
                    {
                        "name": "enhancement",
                        "parameters": {"strategy": "conservative"},
                        "conditions": {}
                    }
                ]
            }
        }
        
        return pipelines
    
    def _estimate_memory_usage(self, original: np.ndarray, processed: np.ndarray) -> Dict[str, float]:
        """Estimate memory usage for processing"""
        original_size = original.nbytes / (1024 * 1024)  # MB
        processed_size = processed.nbytes / (1024 * 1024)  # MB
        
        return {
            "original_image_mb": original_size,
            "processed_image_mb": processed_size,
            "total_mb": original_size + processed_size,
            "memory_overhead": processed_size - original_size
        }
    
    def _update_statistics(self, pipeline_name: str, processing_time: float, quality_improvement: float):
        """Update processing statistics"""
        self.processing_stats["total_processed"] += 1
        
        if quality_improvement > 0:
            self.processing_stats["successful_processing"] += 1
            self.processing_stats["quality_improvements"] += 1
        
        # Update average processing time
        total = self.processing_stats["total_processed"]
        current_avg = self.processing_stats["average_processing_time"]
        self.processing_stats["average_processing_time"] = (
            (current_avg * (total - 1) + processing_time) / total
        )
        
        # Update pipeline usage
        if pipeline_name not in self.processing_stats["pipeline_usage"]:
            self.processing_stats["pipeline_usage"][pipeline_name] = 0
        self.processing_stats["pipeline_usage"][pipeline_name] += 1
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "quality_analyzer": {
                "enable_deep_analysis": True,
                "analysis_cache": True
            },
            "image_enhancer": {
                "enhancement_level": "medium",
                "enable_ai_guidance": True,
                "cache_enhanced_images": True
            },
            "skew_corrector": {
                "correction_quality": "balanced",
                "enable_validation": True
            },
            "system": {
                "max_workers": 4,
                "enable_parallel": True,
                "memory_limit": 2048
            }
        }
    
    # Public API methods for customization
    
    def add_custom_pipeline(self, name: str, pipeline: Dict[str, Any]):
        """Add a custom processing pipeline"""
        self.custom_pipelines[name] = pipeline
        logger.info(f"Added custom pipeline: {name}")
    
    def remove_custom_pipeline(self, name: str):
        """Remove a custom processing pipeline"""
        if name in self.custom_pipelines:
            del self.custom_pipelines[name]
            logger.info(f"Removed custom pipeline: {name}")
    
    def get_available_pipelines(self) -> List[str]:
        """Get list of available pipelines"""
        built_in = list(self.pipelines.keys())
        custom = list(self.custom_pipelines.keys())
        return built_in + custom
    
    def get_pipeline_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific pipeline"""
        if name in self.pipelines:
            return self.pipelines[name]
        elif name in self.custom_pipelines:
            return self.custom_pipelines[name]
        else:
            return None
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get processing statistics"""
        stats = self.processing_stats.copy()
        
        # Calculate success rate
        if stats["total_processed"] > 0:
            stats["success_rate"] = stats["successful_processing"] / stats["total_processed"]
            stats["quality_improvement_rate"] = stats["quality_improvements"] / stats["total_processed"]
        else:
            stats["success_rate"] = 0.0
            stats["quality_improvement_rate"] = 0.0
        
        return stats
    
    def reset_statistics(self):
        """Reset all processing statistics"""
        self.processing_stats = {
            "total_processed": 0,
            "successful_processing": 0,
            "average_processing_time": 0.0,
            "quality_improvements": 0,
            "pipeline_usage": {},
            "performance_metrics": {}
        }
        logger.info("Processing statistics reset")
    
    def configure_component(self, component: str, config: Dict[str, Any]):
        """Reconfigure a component at runtime"""
        if component == "quality_analyzer":
            self.quality_analyzer = IntelligentQualityAnalyzer(config)
        elif component == "image_enhancer":
            self.image_enhancer = AIImageEnhancer(config)
        elif component == "skew_corrector":
            self.skew_corrector = EnhancedSkewCorrector(config)
        else:
            raise ValueError(f"Unknown component: {component}")
        
        logger.info(f"Reconfigured component: {component}")
    
    def validate_pipeline(self, pipeline: Dict[str, Any]) -> List[str]:
        """Validate a pipeline configuration"""
        errors = []
        
        if "name" not in pipeline:
            errors.append("Pipeline must have a 'name' field")
        
        if "steps" not in pipeline:
            errors.append("Pipeline must have a 'steps' field")
        elif not isinstance(pipeline["steps"], list):
            errors.append("Pipeline 'steps' must be a list")
        else:
            # Validate each step
            for i, step in enumerate(pipeline["steps"]):
                if not isinstance(step, dict):
                    errors.append(f"Step {i} must be a dictionary")
                    continue
                
                if "name" not in step:
                    errors.append(f"Step {i} must have a 'name' field")
                
                # Validate step names
                valid_steps = [
                    "skew_correction", "enhancement", "noise_reduction",
                    "contrast_enhancement", "sharpening", "morphological_cleaning",
                    "custom"
                ]
                
                if step.get("name") not in valid_steps:
                    errors.append(f"Step {i} has invalid name: {step.get('name')}")
        
        return errors
    
    def export_config(self, filepath: str):
        """Export current configuration to file"""
        config_data = {
            "quality_analyzer": self.quality_analyzer.config,
            "image_enhancer": self.image_enhancer.config,
            "skew_corrector": self.skew_corrector.config,
            "custom_pipelines": self.custom_pipelines,
            "system": self.config.get("system", {})
        }
        
        with open(filepath, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False)
        
        logger.info(f"Configuration exported to: {filepath}")
    
    def load_config_from_file(self, filepath: str):
        """Load configuration from file"""
        with open(filepath, 'r') as f:
            config = yaml.safe_load(f)
        
        self.config = config
        
        # Reinitialize components
        if "quality_analyzer" in config:
            self.quality_analyzer = IntelligentQualityAnalyzer(config["quality_analyzer"])
        if "image_enhancer" in config:
            self.image_enhancer = AIImageEnhancer(config["image_enhancer"])  
        if "skew_corrector" in config:
            self.skew_corrector = EnhancedSkewCorrector(config["skew_corrector"])
        if "custom_pipelines" in config:
            self.custom_pipelines = config["custom_pipelines"]
        
        logger.info(f"Configuration loaded from: {filepath}")
    
    def shutdown(self):
        """Shutdown the preprocessor and cleanup resources"""
        self.executor.shutdown(wait=True)
        logger.info("Adaptive Preprocessor shutdown complete")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()

# Utility functions for easy usage

def create_default_preprocessor(config_path: Optional[str] = None) -> AdaptivePreprocessor:
    """Create a preprocessor with default settings"""
    return AdaptivePreprocessor(config_path)

def quick_process_image(image: np.ndarray, 
                       processing_level: ProcessingLevel = ProcessingLevel.BALANCED) -> np.ndarray:
    """Quick image processing with minimal setup"""
    preprocessor = AdaptivePreprocessor()
    options = ProcessingOptions(processing_level=processing_level)
    result = preprocessor.process_image(image, options)
    preprocessor.shutdown()
    return result.processed_image

def batch_process_images(images: List[np.ndarray],
                        processing_level: ProcessingLevel = ProcessingLevel.BALANCED,
                        progress_callback: Optional[Callable] = None) -> List[np.ndarray]:
    """Batch process multiple images"""
    preprocessor = AdaptivePreprocessor()
    options = ProcessingOptions(processing_level=processing_level)
    results = preprocessor.process_batch(images, options, progress_callback)
    preprocessor.shutdown()
    return [result.processed_image for result in results]