"""Image enhancement for improving OCR accuracy.

Applies various enhancement techniques based on image quality metrics and
processing strategy. Includes noise reduction, contrast enhancement, sharpening,
and morphological operations.

Examples
--------
    from advanced_ocr.preprocessing import ImageEnhancer
    from advanced_ocr.types import ProcessingStrategy
    
    # Create enhancer with config
    enhancer = ImageEnhancer(config)
    
    # Apply enhancement
    result = enhancer.enhance_image(
        image,
        strategy=ProcessingStrategy.BALANCED,
        quality_metrics=metrics
    )
    
    # Check results
    if result.was_enhanced:
        print(f"Applied: {result.enhancement_applied}")
        print(f"Improvement: {result.quality_improvement:.2f}")
        enhanced_image = result.enhanced_image
"""

import cv2
import numpy as np
import logging
from typing import Dict, Any, List, Tuple, Optional
import time

from ..types import EnhancementResult, QualityMetrics, ProcessingStrategy
from ..utils.config import get_config_value

logger = logging.getLogger(__name__)


class ImageEnhancer:
    """Apply image enhancement techniques to improve OCR accuracy."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize image enhancer with configuration."""
        self.config = config
        
        self.denoise_strength = get_config_value(
            config, 'image_enhancer.denoise_strength', 3
        )
        self.sharpen_strength = get_config_value(
            config, 'image_enhancer.sharpen_strength', 0.5
        )
        self.contrast_factor = get_config_value(
            config, 'image_enhancer.contrast_factor', 1.2
        )
        self.brightness_adjustment = get_config_value(
            config, 'image_enhancer.brightness_adjustment', 0
        )
        
        logger.debug("Image enhancer initialized with config parameters")
    
    def enhance_image(
        self,
        image: np.ndarray, 
        strategy: ProcessingStrategy = ProcessingStrategy.BALANCED,
        quality_metrics: Optional[QualityMetrics] = None
    ) -> EnhancementResult:
        """Apply enhancement based on strategy and quality metrics."""
        start_time = time.time()
        
        if image is None or len(image.shape) < 2:
            logger.error("Invalid image provided for enhancement")
            return self._create_error_result("Invalid image input", start_time)
        
        try:
            # Apply appropriate enhancement level
            if strategy == ProcessingStrategy.MINIMAL:
                enhanced, operations = self._apply_minimal_enhancement(
                    image, quality_metrics
                )
            elif strategy == ProcessingStrategy.ENHANCED:
                enhanced, operations = self._apply_enhanced_processing(
                    image, quality_metrics
                )
            else:
                enhanced, operations = self._apply_balanced_enhancement(
                    image, quality_metrics
                )
            
            processing_time = time.time() - start_time
            
            quality_improvement = (
                self._measure_improvement(image, enhanced) if quality_metrics else 0.0
            )
            
            result = EnhancementResult(
                enhanced_image=enhanced,
                original_image=image,
                enhancement_applied=strategy.value,
                processing_time=processing_time,
                quality_improvement=quality_improvement,
                metadata={
                    'strategy_used': strategy.value,
                    'operations_performed': operations,
                    'original_shape': image.shape,
                    'enhanced_shape': enhanced.shape
                }
            )
            
            logger.debug(
                f"Enhancement completed in {processing_time:.3f}s "
                f"using {strategy.value} strategy"
            )
            return result
            
        except Exception as e:
            logger.error(f"Image enhancement failed: {e}")
            return self._create_error_result(f"Enhancement error: {str(e)}", start_time)
    
    def _apply_minimal_enhancement(
        self,
        image: np.ndarray, 
        quality_metrics: Optional[QualityMetrics]
    ) -> Tuple[np.ndarray, List[str]]:
        """Apply minimal enhancement for high-quality images."""
        enhanced = image.copy()
        operations = []
        
        try:
            # Light denoising if needed
            if quality_metrics and quality_metrics.noise_level > 0.3:
                enhanced = cv2.bilateralFilter(enhanced, 5, 20, 20)
                operations.append("light_denoising")
            
            # Gentle contrast boost if needed
            if quality_metrics and quality_metrics.contrast_score < 0.3:
                enhanced = cv2.convertScaleAbs(enhanced, alpha=1.1, beta=0)
                operations.append("gentle_contrast")
            
            if not operations:
                operations.append("no_enhancement_needed")
                
        except Exception as e:
            logger.warning(f"Minimal enhancement failed: {e}")
            enhanced = image.copy()
            operations = ["minimal_enhancement_failed"]
        
        return enhanced, operations
    
    def _apply_balanced_enhancement(
        self,
        image: np.ndarray,
        quality_metrics: Optional[QualityMetrics]
    ) -> Tuple[np.ndarray, List[str]]:
        """Apply balanced enhancement for medium-quality images."""
        enhanced = image.copy()
        operations = []
        
        try:
            # Bilateral filtering for noise reduction
            enhanced = cv2.bilateralFilter(
                enhanced, self.denoise_strength * 3, 50, 50
            )
            operations.append("noise_reduction")
            
            # CLAHE for adaptive contrast enhancement
            if len(image.shape) == 3:
                # Color image: apply CLAHE to L channel in LAB space
                lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(
                    clipLimit=2.0 * self.contrast_factor,
                    tileGridSize=(8, 8)
                )
                l = clahe.apply(l)
                enhanced = cv2.merge([l, a, b])
                enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            else:
                # Grayscale: apply CLAHE directly
                clahe = cv2.createCLAHE(
                    clipLimit=2.0 * self.contrast_factor,
                    tileGridSize=(8, 8)
                )
                enhanced = clahe.apply(enhanced)
            operations.append("contrast_enhancement")
            
            # Sharpening if image is blurry
            if quality_metrics and quality_metrics.sharpness_score < 0.6:
                kernel = np.array([
                    [-1, -1, -1],
                    [-1,  9, -1],
                    [-1, -1, -1]
                ]) * self.sharpen_strength
                enhanced = cv2.filter2D(enhanced, -1, kernel)
                operations.append("sharpening")
            
            # Brightness adjustment if configured
            if self.brightness_adjustment != 0:
                enhanced = cv2.convertScaleAbs(
                    enhanced,
                    alpha=1.0,
                    beta=self.brightness_adjustment
                )
                operations.append("brightness_adjustment")
                
        except Exception as e:
            logger.warning(f"Balanced enhancement failed: {e}")
            enhanced = image.copy()
            operations = ["balanced_enhancement_failed"]
        
        return enhanced, operations
    
    def _apply_enhanced_processing(
        self,
        image: np.ndarray,
        quality_metrics: Optional[QualityMetrics]
    ) -> Tuple[np.ndarray, List[str]]:
        """Apply aggressive enhancement for poor-quality images."""
        enhanced = image.copy()
        operations = []
        
        try:
            # Strong noise reduction using Non-local Means
            if len(image.shape) == 3:
                enhanced = cv2.fastNlMeansDenoisingColored(
                    enhanced,
                    None,
                    h=self.denoise_strength * 3,
                    hColor=self.denoise_strength * 3,
                    templateWindowSize=7,
                    searchWindowSize=21
                )
            else:
                enhanced = cv2.fastNlMeansDenoising(
                    enhanced,
                    None,
                    h=self.denoise_strength * 3,
                    templateWindowSize=7,
                    searchWindowSize=21
                )
            operations.append("strong_noise_reduction")
            
            # Aggressive CLAHE for contrast
            if len(image.shape) == 3:
                lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(
                    clipLimit=3.0 * self.contrast_factor,
                    tileGridSize=(8, 8)
                )
                l = clahe.apply(l)
                enhanced = cv2.merge([l, a, b])
                enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            else:
                clahe = cv2.createCLAHE(
                    clipLimit=3.0 * self.contrast_factor,
                    tileGridSize=(8, 8)
                )
                enhanced = clahe.apply(enhanced)
            operations.append("strong_contrast_enhancement")
            
            # Aggressive sharpening with blending
            kernel = np.array([
                [-1, -1, -1],
                [-1,  9, -1],
                [-1, -1, -1]
            ]) * (self.sharpen_strength * 1.5)
            sharpened = cv2.filter2D(enhanced, -1, kernel)
            enhanced = cv2.addWeighted(enhanced, 0.7, sharpened, 0.3, 0)
            operations.append("aggressive_sharpening")
            
            # Morphological operations for text cleanup (grayscale only)
            if len(image.shape) == 2:
                kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
                enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_OPEN, kernel_open)
                
                kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
                enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel_close)
                operations.append("morphological_cleanup")
            
            if self.brightness_adjustment != 0:
                enhanced = cv2.convertScaleAbs(
                    enhanced,
                    alpha=1.0,
                    beta=self.brightness_adjustment
                )
                operations.append("brightness_adjustment")
                
        except Exception as e:
            logger.warning(f"Enhanced processing failed: {e}")
            enhanced = image.copy()
            operations = ["enhanced_processing_failed"]
        
        return enhanced, operations
    
    def _measure_improvement(
        self,
        original: np.ndarray,
        enhanced: np.ndarray
    ) -> float:
        """Measure quality improvement using contrast and sharpness metrics."""
        try:
            # Convert to grayscale for comparison
            if len(original.shape) == 3:
                orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
                enh_gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
            else:
                orig_gray = original
                enh_gray = enhanced
            
            # Measure contrast improvement
            orig_contrast = np.std(orig_gray)
            enh_contrast = np.std(enh_gray)
            contrast_improvement = (
                (enh_contrast - orig_contrast) / orig_contrast
                if orig_contrast > 0 else 0
            )
            
            # Measure sharpness improvement
            orig_sharp = cv2.Laplacian(orig_gray, cv2.CV_64F).var()
            enh_sharp = cv2.Laplacian(enh_gray, cv2.CV_64F).var()
            sharpness_improvement = (
                (enh_sharp - orig_sharp) / orig_sharp
                if orig_sharp > 0 else 0
            )
            
            # Average improvement, clamped to [0, 1]
            improvement = (contrast_improvement + sharpness_improvement) / 2
            return max(0.0, min(1.0, improvement))
            
        except Exception as e:
            logger.warning(f"Could not measure improvement: {e}")
            return 0.0
    
    def _create_error_result(self, error_message: str, start_time: float) -> EnhancementResult:
        """Create error state result."""
        processing_time = time.time() - start_time
        error_image = np.zeros((100, 100, 3), dtype=np.uint8)
        
        return EnhancementResult(
            enhanced_image=error_image,
            original_image=error_image,
            enhancement_applied="failed",
            processing_time=processing_time,
            quality_improvement=0.0,
            metadata={
                'error': error_message,
                'strategy_used': 'error_fallback'
            }
        )