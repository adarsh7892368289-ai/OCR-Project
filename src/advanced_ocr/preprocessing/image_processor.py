"""
Advanced OCR Image Preprocessing Module

This module provides comprehensive image preprocessing orchestration for the advanced OCR system.
It coordinates quality analysis, intelligent image enhancement, and text region detection to
optimize images for OCR engine processing.

The module focuses on:
- Unified preprocessing pipeline orchestration from raw images to OCR-ready data
- Intelligent enhancement strategies based on comprehensive quality analysis
- Text region detection and filtering to reduce false positives
- Memory-efficient processing with configurable image size limits
- Integration with quality analyzer and text detector components

Classes:
    EnhancementStrategy: Enumeration of image enhancement strategies
    PreprocessingResult: Data container for complete preprocessing results
    ImageEnhancer: Intelligent image enhancement based on quality metrics
    ImageProcessor: Main orchestrator coordinating all preprocessing steps

Functions:
    create_image_processor: Factory function for creating image processor instances
    validate_preprocessing_result: Validation function for preprocessing results
    extract_enhancement_summary: Utility for extracting enhancement operation summaries

Example:
    >>> from advanced_ocr.preprocessing.image_processor import ImageProcessor
    >>> from advanced_ocr.utils.model_utils import ModelLoader
    >>> processor = ImageProcessor(ModelLoader(), config)
    >>> result = processor.process_image(raw_image)
    >>> print(f"Enhanced image shape: {result.enhanced_image.shape}")
    >>> print(f"Detected regions: {len(result.text_regions)}")

    >>> # Access quality metrics
    >>> quality = result.quality_metrics
    >>> print(f"Overall quality: {quality.overall_level.value} ({quality.overall_score:.3f})")
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Union
import time
from dataclasses import dataclass
from enum import Enum

# Import from parent modules (correct relative imports)
from ...config import OCRConfig
from ...utils.logger import OCRLogger
from ...utils.image_utils import ImageProcessor as ImageUtils
from ...utils.model_utils import ModelLoader
from ...results import TextRegion, BoundingBox
from .quality_analyzer import QualityAnalyzer, QualityMetrics, QualityLevel
from .text_detector import TextDetector


class EnhancementStrategy(Enum):
    """Image enhancement strategies based on quality analysis."""
    MINIMAL = "minimal"           # Excellent quality - minimal processing
    STANDARD = "standard"         # Good quality - standard enhancements
    AGGRESSIVE = "aggressive"     # Poor quality - aggressive processing
    DENOISING = "denoising"       # High noise - focus on noise reduction
    SHARPENING = "sharpening"     # Blurry - focus on sharpening
    CONTRAST = "contrast"         # Low contrast - focus on contrast enhancement
    BRIGHTNESS = "brightness"     # Poor lighting - focus on brightness correction


@dataclass
class PreprocessingResult:
    """
    Complete preprocessing result containing enhanced image, regions, and metrics.
    """
    # Enhanced image data
    enhanced_image: np.ndarray          # Final preprocessed image
    original_image: np.ndarray          # Original input image (for reference)
    
    # Text detection results  
    text_regions: List[TextRegion]      # Detected text regions (20-80 regions)
    
    # Quality analysis
    quality_metrics: QualityMetrics     # Comprehensive quality analysis
    
    # Processing metadata
    enhancement_strategy: EnhancementStrategy  # Strategy used for enhancement
    processing_time: float              # Total processing time in seconds
    enhancements_applied: List[str]     # List of enhancement operations applied
    
    # Image metadata
    original_dimensions: Tuple[int, int]   # Original image dimensions (W, H)
    final_dimensions: Tuple[int, int]      # Final image dimensions (W, H)
    scale_factor: float                    # Scale factor applied (if any)


class ImageEnhancer:
    """
    Intelligent image enhancement based on quality analysis.
    """
    
    def __init__(self, config: OCRConfig):
        """
        Initialize image enhancer.
        
        Args:
            config (OCRConfig): OCR configuration
        """
        self.config = config
        self.logger = OCRLogger()
        
        # Enhancement parameters from config
        self.enable_denoising = config.get("preprocessing.enhancement.enable_denoising", True)
        self.enable_sharpening = config.get("preprocessing.enhancement.enable_sharpening", True)
        self.enable_contrast = config.get("preprocessing.enhancement.enable_contrast", True)
        self.enable_brightness = config.get("preprocessing.enhancement.enable_brightness", True)
        
        # Enhancement thresholds
        self.blur_threshold_for_sharpening = config.get("preprocessing.thresholds.blur_for_sharpening", 0.6)
        self.noise_threshold_for_denoising = config.get("preprocessing.thresholds.noise_for_denoising", 0.6)
        self.contrast_threshold_for_enhancement = config.get("preprocessing.thresholds.contrast_for_enhancement", 0.6)
        self.brightness_threshold_for_correction = config.get("preprocessing.thresholds.brightness_for_correction", 0.6)
    
    def determine_enhancement_strategy(self, quality_metrics: QualityMetrics) -> EnhancementStrategy:
        """
        Determine optimal enhancement strategy based on quality analysis.
        
        Args:
            quality_metrics (QualityMetrics): Quality analysis results
            
        Returns:
            EnhancementStrategy: Recommended enhancement strategy
        """
        # Check overall quality first
        if quality_metrics.overall_level == QualityLevel.EXCELLENT:
            return EnhancementStrategy.MINIMAL
        
        # Identify primary quality issues and select specialized strategy
        quality_issues = []
        
        if quality_metrics.noise_score < self.noise_threshold_for_denoising:
            quality_issues.append(('noise', quality_metrics.noise_score))
        
        if quality_metrics.blur_score < self.blur_threshold_for_sharpening:
            quality_issues.append(('blur', quality_metrics.blur_score))
        
        if quality_metrics.contrast_score < self.contrast_threshold_for_enhancement:
            quality_issues.append(('contrast', quality_metrics.contrast_score))
        
        if quality_metrics.brightness_score < self.brightness_threshold_for_correction:
            quality_issues.append(('brightness', quality_metrics.brightness_score))
        
        # Select strategy based on most severe issue
        if not quality_issues:
            return EnhancementStrategy.STANDARD
        
        # Sort by severity (lowest score = most severe)
        quality_issues.sort(key=lambda x: x[1])
        primary_issue = quality_issues[0][0]
        
        strategy_map = {
            'noise': EnhancementStrategy.DENOISING,
            'blur': EnhancementStrategy.SHARPENING,
            'contrast': EnhancementStrategy.CONTRAST,
            'brightness': EnhancementStrategy.BRIGHTNESS
        }
        
        # If multiple severe issues, use aggressive strategy
        severe_issues = [issue for issue, score in quality_issues if score < 0.4]
        if len(severe_issues) >= 2:
            return EnhancementStrategy.AGGRESSIVE
        
        return strategy_map.get(primary_issue, EnhancementStrategy.STANDARD)
    
    def enhance_image(self, image: np.ndarray, strategy: EnhancementStrategy, 
                     quality_metrics: QualityMetrics) -> Tuple[np.ndarray, List[str]]:
        """
        Apply image enhancements based on strategy and quality metrics.
        
        Args:
            image (np.ndarray): Input image
            strategy (EnhancementStrategy): Enhancement strategy
            quality_metrics (QualityMetrics): Quality analysis results
            
        Returns:
            Tuple[np.ndarray, List[str]]: (enhanced_image, applied_enhancements)
        """
        enhanced_image = image.copy()
        applied_enhancements = []
        
        self.logger.debug(f"Applying enhancement strategy: {strategy.value}")
        
        if strategy == EnhancementStrategy.MINIMAL:
            # Minimal processing for excellent quality images
            enhanced_image = self._apply_minimal_enhancements(enhanced_image)
            applied_enhancements.extend(['gamma_correction', 'slight_sharpening'])
        
        elif strategy == EnhancementStrategy.DENOISING:
            # Focus on noise reduction
            enhanced_image = self._apply_denoising(enhanced_image, quality_metrics)
            applied_enhancements.extend(['bilateral_filter', 'median_filter'])
            
            # Also apply other enhancements if needed
            enhanced_image = self._apply_contrast_enhancement(enhanced_image, quality_metrics)
            applied_enhancements.append('contrast_enhancement')
        
        elif strategy == EnhancementStrategy.SHARPENING:
            # Focus on sharpening
            enhanced_image = self._apply_sharpening(enhanced_image, quality_metrics)
            applied_enhancements.extend(['unsharp_mask', 'edge_enhancement'])
            
            # Also apply contrast if needed
            enhanced_image = self._apply_contrast_enhancement(enhanced_image, quality_metrics)
            applied_enhancements.append('contrast_enhancement')
        
        elif strategy == EnhancementStrategy.CONTRAST:
            # Focus on contrast enhancement
            enhanced_image = self._apply_contrast_enhancement(enhanced_image, quality_metrics)
            enhanced_image = self._apply_histogram_equalization(enhanced_image)
            applied_enhancements.extend(['contrast_enhancement', 'histogram_equalization'])
        
        elif strategy == EnhancementStrategy.BRIGHTNESS:
            # Focus on brightness correction
            enhanced_image = self._apply_brightness_correction(enhanced_image, quality_metrics)
            enhanced_image = self._apply_contrast_enhancement(enhanced_image, quality_metrics)
            applied_enhancements.extend(['brightness_correction', 'contrast_enhancement'])
        
        elif strategy == EnhancementStrategy.AGGRESSIVE:
            # Apply comprehensive enhancements
            enhanced_image = self._apply_comprehensive_enhancement(enhanced_image, quality_metrics)
            applied_enhancements.extend(['denoising', 'sharpening', 'contrast', 'brightness', 'morphology'])
        
        else:  # STANDARD
            # Standard enhancement pipeline
            enhanced_image = self._apply_standard_enhancements(enhanced_image, quality_metrics)
            applied_enhancements.extend(['denoising', 'contrast', 'sharpening'])
        
        self.logger.debug(f"Applied enhancements: {', '.join(applied_enhancements)}")
        
        return enhanced_image, applied_enhancements
    
    def _apply_minimal_enhancements(self, image: np.ndarray) -> np.ndarray:
        """Apply minimal enhancements for high-quality images."""
        # Slight gamma correction
        gamma = 1.1
        enhanced = np.power(image / 255.0, 1.0 / gamma)
        enhanced = (enhanced * 255.0).astype(np.uint8)
        
        # Very light sharpening
        kernel = np.array([[-0.1, -0.1, -0.1],
                          [-0.1,  1.8, -0.1],
                          [-0.1, -0.1, -0.1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        return enhanced
    
    def _apply_denoising(self, image: np.ndarray, quality_metrics: QualityMetrics) -> np.ndarray:
        """Apply noise reduction based on noise level."""
        if quality_metrics.noise_level == QualityLevel.VERY_POOR:
            # Aggressive denoising
            enhanced = cv2.bilateralFilter(image, 9, 75, 75)
            enhanced = cv2.medianBlur(enhanced, 5)
        elif quality_metrics.noise_level == QualityLevel.POOR:
            # Moderate denoising
            enhanced = cv2.bilateralFilter(image, 7, 50, 50)
            enhanced = cv2.medianBlur(enhanced, 3)
        else:
            # Light denoising
            enhanced = cv2.bilateralFilter(image, 5, 30, 30)
        
        return enhanced
    
    def _apply_sharpening(self, image: np.ndarray, quality_metrics: QualityMetrics) -> np.ndarray:
        """Apply sharpening based on blur level."""
        if quality_metrics.blur_level == QualityLevel.VERY_POOR:
            # Strong sharpening
            kernel = np.array([[-1, -1, -1],
                              [-1,  9, -1],
                              [-1, -1, -1]])
            enhanced = cv2.filter2D(image, -1, kernel)
            
            # Additional unsharp masking
            gaussian = cv2.GaussianBlur(enhanced, (5, 5), 1.0)
            enhanced = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)
        
        elif quality_metrics.blur_level == QualityLevel.POOR:
            # Moderate sharpening  
            kernel = np.array([[-0.5, -1, -0.5],
                              [-1,    7,   -1],
                              [-0.5, -1, -0.5]])
            enhanced = cv2.filter2D(image, -1, kernel)
        
        else:
            # Light sharpening
            kernel = np.array([[0, -0.5, 0],
                              [-0.5, 3, -0.5],
                              [0, -0.5, 0]])
            enhanced = cv2.filter2D(image, -1, kernel)
        
        return enhanced
    
    def _apply_contrast_enhancement(self, image: np.ndarray, quality_metrics: QualityMetrics) -> np.ndarray:
        """Apply contrast enhancement based on contrast level."""
        if quality_metrics.contrast_level in [QualityLevel.VERY_POOR, QualityLevel.POOR]:
            # CLAHE (Contrast Limited Adaptive Histogram Equalization)
            if len(image.shape) == 3:
                # Convert to LAB color space for better contrast enhancement
                lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                l_channel, a_channel, b_channel = cv2.split(lab)
                
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                l_channel = clahe.apply(l_channel)
                
                enhanced = cv2.merge([l_channel, a_channel, b_channel])
                enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            else:
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(image)
        else:
            # Simple contrast stretching
            enhanced = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        
        return enhanced
    
    def _apply_brightness_correction(self, image: np.ndarray, quality_metrics: QualityMetrics) -> np.ndarray:
        """Apply brightness correction based on brightness analysis."""
        mean_brightness = quality_metrics.mean_brightness
        target_brightness = 140  # Optimal brightness for OCR
        
        # Calculate brightness adjustment
        brightness_adjustment = target_brightness - mean_brightness
        
        # Apply gamma correction for brightness adjustment
        if brightness_adjustment != 0:
            if brightness_adjustment > 0:  # Image is too dark
                gamma = 0.7 + (brightness_adjustment / 100.0)  # Brighten
            else:  # Image is too bright
                gamma = 1.3 + (abs(brightness_adjustment) / 100.0)  # Darken
            
            enhanced = np.power(image / 255.0, 1.0 / gamma)
            enhanced = (enhanced * 255.0).astype(np.uint8)
        else:
            enhanced = image.copy()
        
        return enhanced
    
    def _apply_histogram_equalization(self, image: np.ndarray) -> np.ndarray:
        """Apply histogram equalization."""
        if len(image.shape) == 3:
            # Convert to YUV and equalize Y channel
            yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
            enhanced = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        else:
            enhanced = cv2.equalizeHist(image)
        
        return enhanced
    
    def _apply_standard_enhancements(self, image: np.ndarray, quality_metrics: QualityMetrics) -> np.ndarray:
        """Apply standard enhancement pipeline."""
        enhanced = image.copy()
        
        # Light denoising
        if self.enable_denoising and quality_metrics.noise_score < 0.8:
            enhanced = cv2.bilateralFilter(enhanced, 5, 30, 30)
        
        # Contrast enhancement
        if self.enable_contrast and quality_metrics.contrast_score < 0.8:
            enhanced = self._apply_contrast_enhancement(enhanced, quality_metrics)
        
        # Light sharpening
        if self.enable_sharpening and quality_metrics.blur_score < 0.8:
            kernel = np.array([[0, -0.25, 0],
                              [-0.25, 2, -0.25],
                              [0, -0.25, 0]])
            enhanced = cv2.filter2D(enhanced, -1, kernel)
        
        return enhanced
    
    def _apply_comprehensive_enhancement(self, image: np.ndarray, quality_metrics: QualityMetrics) -> np.ndarray:
        """Apply comprehensive enhancements for very poor quality images."""
        enhanced = image.copy()
        
        # Step 1: Noise reduction
        enhanced = cv2.bilateralFilter(enhanced, 9, 75, 75)
        enhanced = cv2.medianBlur(enhanced, 3)
        
        # Step 2: Brightness and contrast
        enhanced = self._apply_brightness_correction(enhanced, quality_metrics)
        enhanced = self._apply_contrast_enhancement(enhanced, quality_metrics)
        
        # Step 3: Sharpening
        enhanced = self._apply_sharpening(enhanced, quality_metrics)
        
        # Step 4: Morphological operations to clean up text
        if len(enhanced.shape) == 3:
            gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
        else:
            gray = enhanced.copy()
        
        # Apply morphological closing to connect broken text
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        
        # Convert back to color if needed
        if len(image.shape) == 3 and len(enhanced.shape) == 1:
            enhanced = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        else:
            enhanced = gray
        
        return enhanced


class ImageProcessor:
    """
    Main image preprocessing orchestrator that coordinates quality analysis,
    text detection, and intelligent image enhancement.
    """
    
    def __init__(self, model_loader: ModelLoader, config: OCRConfig):
        """
        Initialize image processor with model loader and configuration.
        
        Args:
            model_loader (ModelLoader): Model loader instance
            config (OCRConfig): OCR configuration
        """
        self.model_loader = model_loader
        self.config = config
        self.logger = OCRLogger()
        
        # Initialize components
        self.quality_analyzer = QualityAnalyzer(config)
        self.text_detector = TextDetector(model_loader, config)
        self.image_enhancer = ImageEnhancer(config)
        self.image_utils = ImageUtils()
        
        # Processing configuration
        self.enable_quality_analysis = config.get("preprocessing.enable_quality_analysis", True)
        self.enable_enhancement = config.get("preprocessing.enable_enhancement", True)
        self.enable_text_detection = config.get("preprocessing.enable_text_detection", True)
        
        # Image resizing configuration
        self.max_image_size = config.get("preprocessing.max_image_size", 2048)
        self.maintain_aspect_ratio = config.get("preprocessing.maintain_aspect_ratio", True)
    
    def process_image(self, image: np.ndarray) -> PreprocessingResult:
        """
        Complete image preprocessing pipeline.
        
        CRITICAL: This is the ONLY method called by core.py
        Orchestrates quality analysis, enhancement, and text detection.
        
        Args:
            image (np.ndarray): Raw input image from core.py
            
        Returns:
            PreprocessingResult: Complete preprocessing results
        """
        start_time = time.time()
        
        self.logger.info(f"Starting image preprocessing for image shape: {image.shape}")
        
        # Store original image and dimensions
        original_image = image.copy()
        original_height, original_width = image.shape[:2]
        original_dimensions = (original_width, original_height)
        
        # Step 1: Resize image if too large (for performance)
        processed_image, scale_factor = self._resize_image_if_needed(image)
        
        # Step 2: Quality analysis
        quality_metrics = None
        if self.enable_quality_analysis:
            self.logger.debug("Performing quality analysis")
            quality_metrics = self.quality_analyzer.analyze_image_quality(processed_image)
        else:
            # Create default quality metrics
            quality_metrics = self._create_default_quality_metrics(processed_image)
        
        # Step 3: Intelligent image enhancement
        enhanced_image = processed_image.copy()
        enhancement_strategy = EnhancementStrategy.MINIMAL
        applied_enhancements = []
        
        if self.enable_enhancement:
            self.logger.debug("Applying intelligent image enhancement")
            enhancement_strategy = self.image_enhancer.determine_enhancement_strategy(quality_metrics)
            enhanced_image, applied_enhancements = self.image_enhancer.enhance_image(
                processed_image, enhancement_strategy, quality_metrics
            )
        
        # Step 4: Text region detection
        text_regions = []
        if self.enable_text_detection:
            self.logger.debug("Detecting text regions")
            text_regions = self.text_detector.detect_text_regions(enhanced_image)
        
        # Calculate final dimensions
        final_height, final_width = enhanced_image.shape[:2]
        final_dimensions = (final_width, final_height)
        
        # Calculate total processing time
        processing_time = time.time() - start_time
        
        # Create comprehensive result
        result = PreprocessingResult(
            enhanced_image=enhanced_image,
            original_image=original_image,
            text_regions=text_regions,
            quality_metrics=quality_metrics,
            enhancement_strategy=enhancement_strategy,
            processing_time=processing_time,
            enhancements_applied=applied_enhancements,
            original_dimensions=original_dimensions,
            final_dimensions=final_dimensions,
            scale_factor=scale_factor
        )
        
        self.logger.info(
            f"Image preprocessing completed: "
            f"quality={quality_metrics.overall_level.value} ({quality_metrics.overall_score:.3f}), "
            f"strategy={enhancement_strategy.value}, "
            f"regions={len(text_regions)}, "
            f"time={processing_time:.3f}s"
        )
        
        return result
    
    def _resize_image_if_needed(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Resize image if it exceeds maximum dimensions.
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            Tuple[np.ndarray, float]: (resized_image, scale_factor)
        """
        height, width = image.shape[:2]
        max_dimension = max(width, height)
        
        if max_dimension <= self.max_image_size:
            return image.copy(), 1.0
        
        # Calculate scale factor
        scale_factor = self.max_image_size / max_dimension
        
        if self.maintain_aspect_ratio:
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
        else:
            new_width = self.max_image_size
            new_height = self.max_image_size
        
        # Resize image
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        
        self.logger.debug(f"Resized image from {width}x{height} to {new_width}x{new_height} (scale: {scale_factor:.3f})")
        
        return resized_image, scale_factor
    
    def _create_default_quality_metrics(self, image: np.ndarray) -> QualityMetrics:
        """
        Create default quality metrics when quality analysis is disabled.
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            QualityMetrics: Default quality metrics
        """
        from .quality_analyzer import QualityMetrics, QualityLevel
        
        height, width = image.shape[:2]
        color_channels = image.shape[2] if len(image.shape) == 3 else 1
        
        return QualityMetrics(
            blur_score=0.8,
            blur_level=QualityLevel.GOOD,
            laplacian_variance=500.0,
            noise_score=0.8,
            noise_level=QualityLevel.GOOD,
            noise_variance=10.0,
            contrast_score=0.8,
            contrast_level=QualityLevel.GOOD,
            contrast_rms=60.0,
            histogram_spread=80.0,
            resolution_score=0.8,
            resolution_level=QualityLevel.GOOD,
            effective_resolution=(width, height),
            dpi_estimate=None,
            brightness_score=0.8,
            brightness_level=QualityLevel.GOOD,
            mean_brightness=140.0,
            brightness_uniformity=0.8,
            overall_score=0.8,
            overall_level=QualityLevel.GOOD,
            image_dimensions=(width, height),
            color_channels=color_channels,
            analysis_time=0.0
        )
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Get preprocessing statistics and configuration.
        
        Returns:
            Dict[str, Any]: Processing statistics
        """
        return {
            'quality_analysis_enabled': self.enable_quality_analysis,
            'enhancement_enabled': self.enable_enhancement,
            'text_detection_enabled': self.enable_text_detection,
            'max_image_size': self.max_image_size,
            'maintain_aspect_ratio': self.maintain_aspect_ratio,
            'quality_analyzer_config': self.quality_analyzer.get_analysis_config(),
            'text_detector_stats': self.text_detector.get_detection_stats()
        }


# Utility functions for external use
def create_image_processor(model_loader: ModelLoader, 
                          config: Optional[OCRConfig] = None) -> ImageProcessor:
    """
    Create an image processor instance.
    
    Args:
        model_loader (ModelLoader): Model loader instance
        config (Optional[OCRConfig]): OCR configuration
        
    Returns:
        ImageProcessor: Configured image processor
    """
    if config is None:
        from ...config import OCRConfig
        config = OCRConfig()
    
    return ImageProcessor(model_loader, config)


def validate_preprocessing_result(result: PreprocessingResult) -> bool:
    """
    Validate preprocessing result completeness.
    
    Args:
        result (PreprocessingResult): Preprocessing result to validate
        
    Returns:
        bool: True if result is valid and complete
    """
    if result is None:
        return False
    
    # Check essential components
    if result.enhanced_image is None or result.enhanced_image.size == 0:
        return False
    
    if result.quality_metrics is None:
        return False
    
    if result.processing_time <= 0:
        return False
    
    return True


def extract_enhancement_summary(result: PreprocessingResult) -> Dict[str, Any]:
    """
    Extract summary of enhancement operations applied.
    
    Args:
        result (PreprocessingResult): Preprocessing result
        
    Returns:
        Dict[str, Any]: Enhancement summary
    """
    return {
        'strategy': result.enhancement_strategy.value,
        'enhancements_applied': result.enhancements_applied,
        'quality_improvement': result.quality_metrics.overall_score,
        'processing_time': result.processing_time,
        'regions_detected': len(result.text_regions),
        'scale_factor': result.scale_factor
    }



__all__ = [
    'EnhancementStrategy', 'PreprocessingResult', 'ImageEnhancer', 'ImageProcessor',
    'create_image_processor', 'validate_preprocessing_result', 'extract_enhancement_summary'
]