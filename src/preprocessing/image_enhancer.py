# src/preprocessing/image_enhancer.py - Fixed Version

import cv2
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
import logging
from typing import Dict, List, Any, Optional, Tuple
import time

try:
    from .quality_analyzer import QualityMetrics, ImageType
except ImportError:
    from src.preprocessing.quality_analyzer import QualityMetrics, ImageType

logger = logging.getLogger(__name__)

class EnhancementStrategy(Enum):
    """Enhancement strategies for different scenarios"""
    CONSERVATIVE = "conservative"      # Minimal enhancement
    BALANCED = "balanced"             # Standard enhancement  
    AGGRESSIVE = "aggressive"         # Maximum enhancement

@dataclass
class EnhancementResult:
    """Result of image enhancement operation"""
    enhanced_image: np.ndarray
    enhancement_applied: str = "balanced"
    quality_improvement: float = 0.0
    processing_time: float = 0.0
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Enhancement details
    operations_performed: List[str] = field(default_factory=list)
    parameters_used: Dict[str, Any] = field(default_factory=dict)

class AIImageEnhancer:
    """
    AI-guided image enhancement for OCR preprocessing
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize image enhancer with configuration"""
        self.config = config or {}
        
        # Enhancement settings
        self.enhancement_level = self.config.get("enhancement_level", "medium")
        self.preserve_aspect_ratio = self.config.get("preserve_aspect_ratio", True)
        self.enable_ai_guidance = self.config.get("enable_ai_guidance", True)
        self.cache_enhanced_images = self.config.get("cache_enhanced_images", False)
        self.measure_improvement = self.config.get("measure_improvement", True)
        
        # Enhancement parameters
        self.enhancement_params = self._load_enhancement_parameters()
        
        # Cache for enhanced images
        self.image_cache = {} if self.cache_enhanced_images else None
        
        logger.info(f"Image enhancer initialized with {self.enhancement_level} level")
    
    def enhance_image(self, image: np.ndarray, 
                     strategy: Optional[EnhancementStrategy] = None,
                     quality_metrics: Optional[QualityMetrics] = None) -> EnhancementResult:
        """
        Enhance image for better OCR results
        
        Args:
            image: Input image as numpy array
            strategy: Enhancement strategy to use
            quality_metrics: Optional quality metrics to guide enhancement
            
        Returns:
            EnhancementResult with enhanced image and metadata
        """
        start_time = time.time()
        
        if image is None:
            return EnhancementResult(
                enhanced_image=np.zeros((100, 100, 3), dtype=np.uint8),
                enhancement_applied="none",
                processing_time=0.0,
                warnings=["Input image is None"]
            )
        
        if len(image.shape) < 2:
            return EnhancementResult(
                enhanced_image=image.copy() if image.size > 0 else np.zeros((100, 100, 3), dtype=np.uint8),
                enhancement_applied="none",
                processing_time=0.0,
                warnings=["Invalid image dimensions"]
            )
        
        try:
            # Determine strategy
            if strategy is None:
                strategy = self._select_enhancement_strategy(image, quality_metrics)
            
            # Apply enhancement based on strategy
            enhanced_image, operations, parameters = self._apply_enhancement(
                image, strategy, quality_metrics
            )
            
            # Measure improvement if enabled
            quality_improvement = 0.0
            if self.measure_improvement and quality_metrics:
                quality_improvement = self._measure_improvement(image, enhanced_image, quality_metrics)
            
            processing_time = time.time() - start_time
            
            result = EnhancementResult(
                enhanced_image=enhanced_image,
                enhancement_applied=strategy.value,
                quality_improvement=quality_improvement,
                processing_time=processing_time,
                operations_performed=operations,
                parameters_used=parameters,
                metadata={
                    "original_shape": image.shape,
                    "enhanced_shape": enhanced_image.shape,
                    "strategy_used": strategy.value
                }
            )
            
            logger.debug(f"Enhancement completed in {processing_time:.3f}s with {strategy.value} strategy")
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Image enhancement failed: {e}")
            
            return EnhancementResult(
                enhanced_image=image.copy(),
                enhancement_applied="failed",
                processing_time=processing_time,
                warnings=[f"Enhancement failed: {str(e)}"]
            )
    
    def _select_enhancement_strategy(self, image: np.ndarray, 
                                   quality_metrics: Optional[QualityMetrics]) -> EnhancementStrategy:
        """Select appropriate enhancement strategy"""
        
        if not self.enable_ai_guidance or quality_metrics is None:
            return EnhancementStrategy.BALANCED
        
        # Strategy selection based on quality metrics
        overall_score = quality_metrics.overall_score
        
        if overall_score >= 0.8:
            # High quality image - use conservative enhancement
            return EnhancementStrategy.CONSERVATIVE
        elif overall_score <= 0.4:
            # Poor quality image - use aggressive enhancement
            return EnhancementStrategy.AGGRESSIVE
        else:
            # Medium quality - use balanced enhancement
            return EnhancementStrategy.BALANCED
    
    def _apply_enhancement(self, image: np.ndarray, strategy: EnhancementStrategy,
                         quality_metrics: Optional[QualityMetrics]) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
        """Apply enhancement operations based on strategy"""
        
        enhanced = image.copy()
        operations = []
        parameters = {}
        
        if strategy == EnhancementStrategy.CONSERVATIVE:
            enhanced, ops, params = self._conservative_enhancement(enhanced, quality_metrics)
        elif strategy == EnhancementStrategy.AGGRESSIVE:
            enhanced, ops, params = self._aggressive_enhancement(enhanced, quality_metrics)
        else:  # BALANCED
            enhanced, ops, params = self._balanced_enhancement(enhanced, quality_metrics)
        
        operations.extend(ops)
        parameters.update(params)
        
        return enhanced, operations, parameters
    
    def _conservative_enhancement(self, image: np.ndarray, 
                                quality_metrics: Optional[QualityMetrics]) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
        """Conservative enhancement - minimal processing"""
        enhanced = image.copy()
        operations = []
        parameters = {}
        
        try:
            # Convert to grayscale if needed for processing
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                is_color = True
            else:
                gray = image.copy()
                is_color = False
            
            # Light noise reduction only if needed
            if quality_metrics and quality_metrics.noise_level > 0.3:
                if is_color:
                    enhanced = cv2.bilateralFilter(enhanced, 5, 25, 25)
                else:
                    enhanced = cv2.bilateralFilter(gray, 5, 25, 25)
                operations.append("light_noise_reduction")
                parameters["bilateral_filter"] = {"d": 5, "sigma_color": 25, "sigma_space": 25}
            
            # Gentle contrast enhancement if needed
            if quality_metrics and quality_metrics.contrast_score < 0.4:
                if is_color:
                    lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
                    l, a, b = cv2.split(lab)
                    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))
                    l = clahe.apply(l)
                    enhanced = cv2.merge([l, a, b])
                    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
                else:
                    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(4, 4))
                    enhanced = clahe.apply(enhanced)
                
                operations.append("gentle_contrast_enhancement")
                parameters["clahe"] = {"clip_limit": 1.5, "tile_grid_size": (4, 4)}
        
        except Exception as e:
            logger.warning(f"Conservative enhancement failed: {e}")
        
        return enhanced, operations, parameters
    
    def _balanced_enhancement(self, image: np.ndarray,
                            quality_metrics: Optional[QualityMetrics]) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
        """Balanced enhancement - standard processing"""
        enhanced = image.copy()
        operations = []
        parameters = {}
        
        try:
            # Convert to grayscale if needed for processing
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                is_color = True
            else:
                gray = image.copy()
                is_color = False
            
            # Noise reduction
            if quality_metrics and quality_metrics.noise_level > 0.2:
                if is_color:
                    enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
                else:
                    enhanced = cv2.bilateralFilter(gray, 9, 75, 75)
                operations.append("noise_reduction")
                parameters["bilateral_filter"] = {"d": 9, "sigma_color": 75, "sigma_space": 75}
            
            # Contrast enhancement
            if quality_metrics and quality_metrics.contrast_score < 0.6:
                if is_color:
                    lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
                    l, a, b = cv2.split(lab)
                    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                    l = clahe.apply(l)
                    enhanced = cv2.merge([l, a, b])
                    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
                else:
                    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                    enhanced = clahe.apply(enhanced)
                
                operations.append("contrast_enhancement")
                parameters["clahe"] = {"clip_limit": 3.0, "tile_grid_size": (8, 8)}
            
            # Sharpening if needed
            if quality_metrics and quality_metrics.sharpness_score < 0.6:
                kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]], dtype=np.float32)
                kernel = kernel / np.sum(kernel) if np.sum(kernel) != 0 else kernel
                
                if is_color:
                    enhanced = cv2.filter2D(enhanced, -1, kernel)
                else:
                    enhanced = cv2.filter2D(enhanced, -1, kernel)
                
                operations.append("sharpening")
                parameters["sharpening_kernel"] = "3x3_laplacian"
        
        except Exception as e:
            logger.warning(f"Balanced enhancement failed: {e}")
        
        return enhanced, operations, parameters
    
    def _aggressive_enhancement(self, image: np.ndarray,
                              quality_metrics: Optional[QualityMetrics]) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
        """Aggressive enhancement - maximum processing"""
        enhanced = image.copy()
        operations = []
        parameters = {}
        
        try:
            # Convert to grayscale if needed for processing
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                is_color = True
            else:
                gray = image.copy()
                is_color = False
            
            # Strong noise reduction
            if is_color:
                enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
            else:
                enhanced = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
            operations.append("strong_noise_reduction")
            parameters["non_local_means"] = {"h": 10, "template_window_size": 7, "search_window_size": 21}
            
            # Strong contrast enhancement
            if is_color:
                lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
                l = clahe.apply(l)
                enhanced = cv2.merge([l, a, b])
                enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            else:
                clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(enhanced)
            
            operations.append("strong_contrast_enhancement")
            parameters["clahe"] = {"clip_limit": 4.0, "tile_grid_size": (8, 8)}
            
            # Adaptive sharpening
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]], dtype=np.float32) * 0.8
            if is_color:
                sharpened = cv2.filter2D(enhanced, -1, kernel)
                enhanced = cv2.addWeighted(enhanced, 0.7, sharpened, 0.3, 0)
            else:
                sharpened = cv2.filter2D(enhanced, -1, kernel)
                enhanced = cv2.addWeighted(enhanced, 0.7, sharpened, 0.3, 0)
            
            operations.append("adaptive_sharpening")
            parameters["sharpening"] = {"kernel": "enhanced_laplacian", "blend_ratio": 0.3}
            
            # Morphological operations for text cleanup
            if not is_color:
                # Opening to remove small noise
                kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
                enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_OPEN, kernel_open)
                
                # Closing to connect broken text
                kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
                enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel_close)
                
                operations.append("morphological_cleanup")
                parameters["morphology"] = {"opening": (2, 2), "closing": (3, 1)}
        
        except Exception as e:
            logger.warning(f"Aggressive enhancement failed: {e}")
        
        return enhanced, operations, parameters
    
    def _measure_improvement(self, original: np.ndarray, enhanced: np.ndarray,
                           quality_metrics: QualityMetrics) -> float:
        """Measure quality improvement after enhancement"""
        try:
            # Simple improvement measurement based on contrast and sharpness
            if len(original.shape) == 3:
                orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
                enh_gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
            else:
                orig_gray = original
                enh_gray = enhanced
            
            # Measure contrast improvement
            orig_contrast = np.std(orig_gray) / 128.0
            enh_contrast = np.std(enh_gray) / 128.0
            contrast_improvement = enh_contrast - orig_contrast
            
            # Measure sharpness improvement
            orig_sharp = cv2.Laplacian(orig_gray, cv2.CV_64F).var() / 500.0
            enh_sharp = cv2.Laplacian(enh_gray, cv2.CV_64F).var() / 500.0
            sharpness_improvement = enh_sharp - orig_sharp
            
            # Combined improvement score
            improvement = (contrast_improvement + sharpness_improvement) / 2
            return max(0.0, min(1.0, improvement))
            
        except Exception as e:
            logger.warning(f"Could not measure improvement: {e}")
            return 0.0
    
    def _load_enhancement_parameters(self) -> Dict[str, Any]:
        """Load enhancement parameters from config"""
        default_params = {
            "bilateral_filter": {
                "d": 9,
                "sigma_color": 75,
                "sigma_space": 75
            },
            "non_local_means": {
                "h": 10,
                "template_window_size": 7,
                "search_window_size": 21
            },
            "clahe": {
                "clip_limit": 3.0,
                "tile_grid_size": (8, 8)
            },
            "morphological": {
                "opening_kernel": (2, 2),
                "closing_kernel": (3, 1),
                "iterations": 1
            }
        }
        
        # Override with config values if provided
        config_params = self.config.get("enhancement_parameters", {})
        for key, value in config_params.items():
            if key in default_params:
                default_params[key].update(value)
        
        return default_params
    
    def enhance_for_engine(self, image: np.ndarray, engine_name: str,
                          quality_metrics: Optional[QualityMetrics] = None) -> EnhancementResult:
        """Enhance image specifically for a target OCR engine"""
        
        # Engine-specific enhancement strategies
        engine_strategies = {
            "tesseract": EnhancementStrategy.BALANCED,
            "easyocr": EnhancementStrategy.CONSERVATIVE,
            "paddleocr": EnhancementStrategy.BALANCED,
            "trocr": EnhancementStrategy.AGGRESSIVE
        }
        
        strategy = engine_strategies.get(engine_name, EnhancementStrategy.BALANCED)
        
        result = self.enhance_image(image, strategy, quality_metrics)
        result.metadata["target_engine"] = engine_name
        result.metadata["engine_optimized"] = True
        
        return result
    
    def batch_enhance(self, images: List[np.ndarray],
                     strategy: Optional[EnhancementStrategy] = None,
                     progress_callback: Optional[callable] = None) -> List[EnhancementResult]:
        """Enhance multiple images in batch"""
        results = []
        
        for i, image in enumerate(images):
            result = self.enhance_image(image, strategy)
            results.append(result)
            
            if progress_callback:
                progress_callback(i + 1, len(images))
        
        return results
    
    def get_enhancement_statistics(self) -> Dict[str, Any]:
        """Get enhancement usage statistics"""
        # This would track enhancement usage over time
        # For now, return basic info
        return {
            "enhancement_level": self.enhancement_level,
            "ai_guidance_enabled": self.enable_ai_guidance,
            "cache_enabled": self.cache_enhanced_images,
            "cache_size": len(self.image_cache) if self.image_cache else 0
        }
    
    def clear_cache(self):
        """Clear enhancement cache"""
        if self.image_cache:
            self.image_cache.clear()
            logger.info("Enhancement cache cleared")
            
    # Add these methods to your existing AIImageEnhancer class in image_enhancer.py

    def should_enhance_image(self, image: np.ndarray, 
                           quality_metrics: Optional[QualityMetrics] = None) -> Tuple[bool, str]:
        """
        CONDITIONAL ENHANCEMENT - Decide if enhancement is beneficial
        
        Returns:
            (should_enhance: bool, reason: str)
        """
        if quality_metrics is None:
            return True, "no_quality_metrics_available"
        
        # High quality images (>0.75) - skip enhancement
        if quality_metrics.overall_score > 0.75:
            return False, f"high_quality_image_score_{quality_metrics.overall_score:.3f}"
        
        # Very good quality (0.65-0.75) - conditional enhancement
        if quality_metrics.overall_score > 0.65:
            # Only enhance if specific issues exist
            needs_contrast = quality_metrics.contrast_score < 0.4
            needs_sharpening = quality_metrics.sharpness_score < 0.5
            high_noise = quality_metrics.noise_level > 0.3
            
            if needs_contrast or needs_sharpening or high_noise:
                issues = []
                if needs_contrast: issues.append("low_contrast")
                if needs_sharpening: issues.append("low_sharpness") 
                if high_noise: issues.append("high_noise")
                return True, f"targeted_enhancement_needed_{'+'.join(issues)}"
            else:
                return False, f"good_quality_no_issues_score_{quality_metrics.overall_score:.3f}"
        
        # Medium quality (0.4-0.65) - standard enhancement
        if quality_metrics.overall_score > 0.4:
            return True, f"medium_quality_standard_enhancement_score_{quality_metrics.overall_score:.3f}"
        
        # Low quality (<0.4) - always enhance
        return True, f"low_quality_enhancement_required_score_{quality_metrics.overall_score:.3f}"
    
    def smart_enhance_image(self, image: np.ndarray, 
                          quality_metrics: Optional[QualityMetrics] = None,
                          force_enhance: bool = False) -> EnhancementResult:
        """
        SMART ENHANCEMENT with conditional processing
        
        Args:
            image: Input image
            quality_metrics: Quality analysis results
            force_enhance: Force enhancement even if not recommended
            
        Returns:
            EnhancementResult with processing decision info
        """
        start_time = time.time()
        
        if image is None:
            return EnhancementResult(
                enhanced_image=np.zeros((100, 100, 3), dtype=np.uint8),
                enhancement_applied="failed",
                processing_time=0.0,
                warnings=["Input image is None"]
            )
        
        # Decide if enhancement is needed
        should_enhance, reason = self.should_enhance_image(image, quality_metrics)
        
        if not should_enhance and not force_enhance:
            # Skip enhancement - return original with metadata
            processing_time = time.time() - start_time
            
            logger.info(f"Enhancement SKIPPED: {reason}")
            
            return EnhancementResult(
                enhanced_image=image.copy(),
                enhancement_applied="skipped",
                quality_improvement=0.0,
                processing_time=processing_time,
                operations_performed=["conditional_skip"],
                parameters_used={"skip_reason": reason},
                metadata={
                    "original_shape": image.shape,
                    "enhanced_shape": image.shape,
                    "strategy_used": "conditional_skip",
                    "skip_reason": reason,
                    "quality_score": quality_metrics.overall_score if quality_metrics else None
                },
                warnings=[f"Enhancement skipped: {reason}"]
            )
        
        # Proceed with regular enhancement
        logger.info(f"Enhancement PROCEEDING: {reason}")
        return self.enhance_image(image, None, quality_metrics)
    
    def targeted_enhance_image(self, image: np.ndarray,
                             quality_metrics: QualityMetrics) -> EnhancementResult:
        """
        TARGETED ENHANCEMENT - Only fix specific issues
        
        For medium-quality images, apply only necessary enhancements
        """
        start_time = time.time()
        enhanced = image.copy()
        operations = []
        parameters = {}
        
        try:
            # Convert to grayscale if needed for processing
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                is_color = True
            else:
                gray = image.copy()
                is_color = False
            
            # Only fix contrast if really needed
            if quality_metrics.contrast_score < 0.4:
                if is_color:
                    lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
                    l, a, b = cv2.split(lab)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    l = clahe.apply(l)
                    enhanced = cv2.merge([l, a, b])
                    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
                else:
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    enhanced = clahe.apply(enhanced)
                
                operations.append("targeted_contrast_enhancement")
                parameters["clahe"] = {"clip_limit": 2.0, "tile_grid_size": (8, 8)}
            
            # Only fix sharpness if really needed
            if quality_metrics.sharpness_score < 0.5:
                kernel = np.array([[-0.5,-0.5,-0.5], [-0.5,5,-0.5], [-0.5,-0.5,-0.5]], dtype=np.float32)
                enhanced = cv2.filter2D(enhanced, -1, kernel)
                operations.append("targeted_sharpening")
                parameters["sharpening"] = "mild_laplacian"
            
            # Only reduce noise if really needed
            if quality_metrics.noise_level > 0.3:
                if is_color:
                    enhanced = cv2.bilateralFilter(enhanced, 5, 50, 50)
                else:
                    enhanced = cv2.bilateralFilter(enhanced, 5, 50, 50)
                operations.append("targeted_noise_reduction")
                parameters["bilateral_filter"] = {"d": 5, "sigma_color": 50, "sigma_space": 50}
        
        except Exception as e:
            logger.warning(f"Targeted enhancement failed: {e}")
            enhanced = image.copy()
            operations = ["failed"]
        
        processing_time = time.time() - start_time
        
        return EnhancementResult(
            enhanced_image=enhanced,
            enhancement_applied="targeted",
            quality_improvement=self._measure_improvement(image, enhanced, quality_metrics),
            processing_time=processing_time,
            operations_performed=operations,
            parameters_used=parameters,
            metadata={
                "original_shape": image.shape,
                "enhanced_shape": enhanced.shape,
                "strategy_used": "targeted_enhancement"
            }
        )
ImageEnhancer = AIImageEnhancer
