# src/advanced_ocr/preprocessing/quality_analyzer.py
"""
Advanced OCR Image Quality Analysis Module

This module provides comprehensive image quality analysis for the advanced OCR system.
It analyzes multiple quality dimensions including blur, noise, contrast, resolution,
brightness, and overall quality to enable intelligent preprocessing decisions.

The module focuses on:
- Multi-dimensional quality assessment using computer vision techniques
- Quality level classification with configurable thresholds
- Statistical analysis of image characteristics for OCR optimization
- Performance-optimized analysis with minimal computational overhead
- Integration with preprocessing pipeline for adaptive enhancement strategies

Classes:
    QualityLevel: Enumeration of quality assessment levels
    QualityMetrics: Data container for comprehensive quality analysis results
    BlurAnalyzer: Specialized analyzer for image blur detection
    NoiseAnalyzer: Specialized analyzer for image noise assessment
    ContrastAnalyzer: Specialized analyzer for contrast and histogram analysis
    ResolutionAnalyzer: Specialized analyzer for resolution and DPI estimation
    BrightnessAnalyzer: Specialized analyzer for lighting and brightness analysis
    QualityAnalyzer: Main orchestrator coordinating all quality analysis components

Functions:
    create_quality_analyzer: Factory function for creating quality analyzer instances
    get_quality_level_description: Utility for human-readable quality level descriptions
    analyze_quality_trends: Utility for analyzing quality trends across multiple images

Example:
    >>> from advanced_ocr.preprocessing.quality_analyzer import QualityAnalyzer
    >>> analyzer = QualityAnalyzer(config)
    >>> metrics = analyzer.analyze_image_quality(image)
    >>> print(f"Overall quality: {metrics.overall_level.value} ({metrics.overall_score:.3f})")
    >>> print(f"Blur level: {metrics.blur_level.value}, Noise level: {metrics.noise_level.value}")

    >>> # Access detailed metrics
    >>> print(f"Blur score: {metrics.blur_score:.3f}, Contrast score: {metrics.contrast_score:.3f}")
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional, Any, List
import math
from dataclasses import dataclass
from enum import Enum

# Import from parent modules (correct relative imports)
from ..config import OCRConfig
from ..utils.logger import OCRLogger


class QualityLevel(Enum):
    """Enumeration for image quality levels."""
    EXCELLENT = "excellent"
    GOOD = "good" 
    FAIR = "fair"
    POOR = "poor"
    VERY_POOR = "very_poor"


@dataclass
class QualityMetrics:
    """
    Data class containing comprehensive image quality metrics.
    """
    # Blur analysis
    blur_score: float           # 0.0 = very blurry, 1.0 = sharp
    blur_level: QualityLevel
    laplacian_variance: float   # Raw Laplacian variance
    
    # Noise analysis  
    noise_score: float          # 0.0 = very noisy, 1.0 = clean
    noise_level: QualityLevel
    noise_variance: float       # Estimated noise variance
    
    # Contrast analysis
    contrast_score: float       # 0.0 = no contrast, 1.0 = excellent contrast
    contrast_level: QualityLevel
    contrast_rms: float         # RMS contrast value
    histogram_spread: float     # Histogram spread measure
    
    # Resolution analysis
    resolution_score: float     # 0.0 = very low, 1.0 = high resolution
    resolution_level: QualityLevel
    effective_resolution: Tuple[int, int]  # Effective resolution (width, height)
    dpi_estimate: Optional[float]          # Estimated DPI if calculable
    
    # Lighting analysis
    brightness_score: float     # 0.0 = too dark/bright, 1.0 = optimal
    brightness_level: QualityLevel
    mean_brightness: float      # Mean brightness value (0-255)
    brightness_uniformity: float # Uniformity of brightness distribution
    
    # Overall quality
    overall_score: float        # Weighted combination of all metrics
    overall_level: QualityLevel
    
    # Additional metadata
    image_dimensions: Tuple[int, int]  # Original image dimensions
    color_channels: int         # Number of color channels
    analysis_time: float        # Time taken for analysis in seconds


class BlurAnalyzer:
    """
    Analyzes image blur using multiple detection methods.
    """
    
    def __init__(self, config: OCRConfig):
        """
        Initialize blur analyzer.
        
        Args:
            config (OCRConfig): OCR configuration
        """
        self.config = config
        self.logger = OCRLogger()
        
        # Blur detection thresholds from config
        self.laplacian_threshold_excellent = config.get("quality.blur.laplacian_excellent", 1000)
        self.laplacian_threshold_good = config.get("quality.blur.laplacian_good", 500)
        self.laplacian_threshold_fair = config.get("quality.blur.laplacian_fair", 100)
        self.laplacian_threshold_poor = config.get("quality.blur.laplacian_poor", 50)
    
    def analyze_blur(self, image: np.ndarray) -> Tuple[float, QualityLevel, float]:
        """
        Analyze image blur using Laplacian variance method.
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            Tuple[float, QualityLevel, float]: (blur_score, blur_level, laplacian_variance)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Calculate Laplacian variance (focus measure)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_variance = laplacian.var()
        
        # Determine blur level based on thresholds
        if laplacian_variance >= self.laplacian_threshold_excellent:
            blur_level = QualityLevel.EXCELLENT
            blur_score = min(1.0, laplacian_variance / self.laplacian_threshold_excellent)
        elif laplacian_variance >= self.laplacian_threshold_good:
            blur_level = QualityLevel.GOOD
            blur_score = 0.8 * (laplacian_variance / self.laplacian_threshold_good)
        elif laplacian_variance >= self.laplacian_threshold_fair:
            blur_level = QualityLevel.FAIR
            blur_score = 0.6 * (laplacian_variance / self.laplacian_threshold_fair)
        elif laplacian_variance >= self.laplacian_threshold_poor:
            blur_level = QualityLevel.POOR
            blur_score = 0.4 * (laplacian_variance / self.laplacian_threshold_poor)
        else:
            blur_level = QualityLevel.VERY_POOR
            blur_score = min(0.3, laplacian_variance / self.laplacian_threshold_poor)
        
        self.logger.debug(f"Blur analysis: variance={laplacian_variance:.2f}, score={blur_score:.3f}, level={blur_level.value}")
        
        return blur_score, blur_level, laplacian_variance
    
    def analyze_blur_gradient(self, image: np.ndarray) -> float:
        """
        Alternative blur analysis using gradient magnitude.
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            float: Gradient-based blur score (0.0-1.0)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Calculate gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate gradient magnitude
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Use mean gradient magnitude as sharpness measure
        mean_gradient = np.mean(gradient_magnitude)
        
        # Normalize to 0-1 scale (adjust based on empirical observations)
        normalized_score = min(1.0, mean_gradient / 100.0)
        
        return normalized_score


class NoiseAnalyzer:
    """
    Analyzes image noise using statistical methods.
    """
    
    def __init__(self, config: OCRConfig):
        """
        Initialize noise analyzer.
        
        Args:
            config (OCRConfig): OCR configuration
        """
        self.config = config
        self.logger = OCRLogger()
        
        # Noise detection parameters
        self.noise_kernel_size = config.get("quality.noise.kernel_size", 5)
        self.noise_threshold_excellent = config.get("quality.noise.threshold_excellent", 5)
        self.noise_threshold_good = config.get("quality.noise.threshold_good", 15)
        self.noise_threshold_fair = config.get("quality.noise.threshold_fair", 30)
        self.noise_threshold_poor = config.get("quality.noise.threshold_poor", 50)
    
    def analyze_noise(self, image: np.ndarray) -> Tuple[float, QualityLevel, float]:
        """
        Analyze image noise using local variance method.
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            Tuple[float, QualityLevel, float]: (noise_score, noise_level, noise_variance)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply median filter to get noise-free version
        median_filtered = cv2.medianBlur(gray, self.noise_kernel_size)
        
        # Calculate noise as difference between original and filtered
        noise = cv2.absdiff(gray.astype(np.float32), median_filtered.astype(np.float32))
        
        # Calculate noise variance
        noise_variance = np.var(noise)
        
        # Determine noise level and score
        if noise_variance <= self.noise_threshold_excellent:
            noise_level = QualityLevel.EXCELLENT
            noise_score = 1.0 - (noise_variance / self.noise_threshold_excellent) * 0.2
        elif noise_variance <= self.noise_threshold_good:
            noise_level = QualityLevel.GOOD
            noise_score = 0.8 - (noise_variance / self.noise_threshold_good) * 0.2
        elif noise_variance <= self.noise_threshold_fair:
            noise_level = QualityLevel.FAIR
            noise_score = 0.6 - (noise_variance / self.noise_threshold_fair) * 0.2
        elif noise_variance <= self.noise_threshold_poor:
            noise_level = QualityLevel.POOR
            noise_score = 0.4 - (noise_variance / self.noise_threshold_poor) * 0.2
        else:
            noise_level = QualityLevel.VERY_POOR
            noise_score = max(0.0, 0.2 - (noise_variance / self.noise_threshold_poor) * 0.2)
        
        self.logger.debug(f"Noise analysis: variance={noise_variance:.2f}, score={noise_score:.3f}, level={noise_level.value}")
        
        return noise_score, noise_level, noise_variance


class ContrastAnalyzer:
    """
    Analyzes image contrast using multiple metrics.
    """
    
    def __init__(self, config: OCRConfig):
        """
        Initialize contrast analyzer.
        
        Args:
            config (OCRConfig): OCR configuration
        """
        self.config = config
        self.logger = OCRLogger()
        
        # Contrast analysis parameters
        self.rms_excellent = config.get("quality.contrast.rms_excellent", 80)
        self.rms_good = config.get("quality.contrast.rms_good", 60)
        self.rms_fair = config.get("quality.contrast.rms_fair", 40)
        self.rms_poor = config.get("quality.contrast.rms_poor", 20)
    
    def analyze_contrast(self, image: np.ndarray) -> Tuple[float, QualityLevel, float, float]:
        """
        Analyze image contrast using RMS and histogram methods.
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            Tuple[float, QualityLevel, float, float]: (contrast_score, contrast_level, rms_contrast, histogram_spread)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Calculate RMS contrast
        mean_intensity = np.mean(gray)
        rms_contrast = np.sqrt(np.mean((gray - mean_intensity) ** 2))
        
        # Calculate histogram spread
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_normalized = hist.flatten() / hist.sum()
        
        # Calculate histogram spread (standard deviation of histogram)
        bin_centers = np.arange(256)
        histogram_mean = np.sum(bin_centers * hist_normalized)
        histogram_spread = np.sqrt(np.sum(hist_normalized * (bin_centers - histogram_mean) ** 2))
        
        # Determine contrast level based on RMS contrast
        if rms_contrast >= self.rms_excellent:
            contrast_level = QualityLevel.EXCELLENT
            contrast_score = min(1.0, rms_contrast / self.rms_excellent)
        elif rms_contrast >= self.rms_good:
            contrast_level = QualityLevel.GOOD
            contrast_score = 0.8 * (rms_contrast / self.rms_good)
        elif rms_contrast >= self.rms_fair:
            contrast_level = QualityLevel.FAIR
            contrast_score = 0.6 * (rms_contrast / self.rms_fair)
        elif rms_contrast >= self.rms_poor:
            contrast_level = QualityLevel.POOR
            contrast_score = 0.4 * (rms_contrast / self.rms_poor)
        else:
            contrast_level = QualityLevel.VERY_POOR
            contrast_score = min(0.3, rms_contrast / self.rms_poor)
        
        self.logger.debug(f"Contrast analysis: RMS={rms_contrast:.2f}, spread={histogram_spread:.2f}, score={contrast_score:.3f}, level={contrast_level.value}")
        
        return contrast_score, contrast_level, rms_contrast, histogram_spread


class ResolutionAnalyzer:
    """
    Analyzes image resolution and estimates effective DPI.
    """
    
    def __init__(self, config: OCRConfig):
        """
        Initialize resolution analyzer.
        
        Args:
            config (OCRConfig): OCR configuration
        """
        self.config = config
        self.logger = OCRLogger()
        
        # Resolution thresholds (pixels)
        self.min_width_excellent = config.get("quality.resolution.min_width_excellent", 1200)
        self.min_height_excellent = config.get("quality.resolution.min_height_excellent", 1600)
        self.min_width_good = config.get("quality.resolution.min_width_good", 800)
        self.min_height_good = config.get("quality.resolution.min_height_good", 1000)
        self.min_width_fair = config.get("quality.resolution.min_width_fair", 600)
        self.min_height_fair = config.get("quality.resolution.min_height_fair", 800)
    
    def analyze_resolution(self, image: np.ndarray) -> Tuple[float, QualityLevel, Tuple[int, int], Optional[float]]:
        """
        Analyze image resolution and estimate DPI.
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            Tuple[float, QualityLevel, Tuple[int, int], Optional[float]]: 
                (resolution_score, resolution_level, effective_resolution, dpi_estimate)
        """
        height, width = image.shape[:2]
        effective_resolution = (width, height)
        
        # Calculate resolution score based on minimum dimension
        min_dimension = min(width, height)
        
        if width >= self.min_width_excellent and height >= self.min_height_excellent:
            resolution_level = QualityLevel.EXCELLENT
            resolution_score = 1.0
        elif width >= self.min_width_good and height >= self.min_height_good:
            resolution_level = QualityLevel.GOOD
            resolution_score = 0.8
        elif width >= self.min_width_fair and height >= self.min_height_fair:
            resolution_level = QualityLevel.FAIR
            resolution_score = 0.6
        elif min_dimension >= 300:  # Minimum usable resolution
            resolution_level = QualityLevel.POOR
            resolution_score = 0.4
        else:
            resolution_level = QualityLevel.VERY_POOR
            resolution_score = 0.2
        
        # Estimate DPI (very rough estimation)
        # Assume typical document is 8.5" x 11" for estimation
        dpi_estimate = None
        if width > 0 and height > 0:
            # Estimate based on typical document proportions
            aspect_ratio = width / height
            if 0.7 <= aspect_ratio <= 0.8:  # Portrait document-like
                dpi_estimate = height / 11.0  # Assume 11 inch height
            elif 1.2 <= aspect_ratio <= 1.4:  # Landscape document-like
                dpi_estimate = width / 11.0   # Assume 11 inch width
            else:
                dpi_estimate = min_dimension / 8.5  # Use smaller dimension
        
        self.logger.debug(f"Resolution analysis: {width}x{height}, score={resolution_score:.3f}, level={resolution_level.value}, DPI≈{dpi_estimate:.0f}" if dpi_estimate else f"Resolution analysis: {width}x{height}, score={resolution_score:.3f}, level={resolution_level.value}")
        
        return resolution_score, resolution_level, effective_resolution, dpi_estimate


class BrightnessAnalyzer:
    """
    Analyzes image brightness and lighting conditions.
    """
    
    def __init__(self, config: OCRConfig):
        """
        Initialize brightness analyzer.
        
        Args:
            config (OCRConfig): OCR configuration
        """
        self.config = config
        self.logger = OCRLogger()
        
        # Optimal brightness range
        self.optimal_brightness_min = config.get("quality.brightness.optimal_min", 100)
        self.optimal_brightness_max = config.get("quality.brightness.optimal_max", 180)
        self.acceptable_brightness_min = config.get("quality.brightness.acceptable_min", 80)
        self.acceptable_brightness_max = config.get("quality.brightness.acceptable_max", 200)
    
    def analyze_brightness(self, image: np.ndarray) -> Tuple[float, QualityLevel, float, float]:
        """
        Analyze image brightness and uniformity.
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            Tuple[float, QualityLevel, float, float]: 
                (brightness_score, brightness_level, mean_brightness, uniformity)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Calculate mean brightness
        mean_brightness = np.mean(gray)
        
        # Calculate brightness uniformity (inverse of standard deviation)
        brightness_std = np.std(gray)
        brightness_uniformity = max(0.0, 1.0 - (brightness_std / 128.0))  # Normalize by max possible std
        
        # Determine brightness score and level
        if self.optimal_brightness_min <= mean_brightness <= self.optimal_brightness_max:
            brightness_level = QualityLevel.EXCELLENT
            brightness_score = 1.0 - abs(mean_brightness - 140) / 40  # Peak at 140
        elif self.acceptable_brightness_min <= mean_brightness <= self.acceptable_brightness_max:
            brightness_level = QualityLevel.GOOD
            if mean_brightness < self.optimal_brightness_min:
                brightness_score = 0.8 * (mean_brightness / self.optimal_brightness_min)
            else:
                brightness_score = 0.8 * (self.acceptable_brightness_max / mean_brightness)
        elif 50 <= mean_brightness <= 220:  # Usable range
            brightness_level = QualityLevel.FAIR
            brightness_score = 0.6
        elif 30 <= mean_brightness <= 240:  # Marginal range
            brightness_level = QualityLevel.POOR
            brightness_score = 0.4
        else:  # Too dark or too bright
            brightness_level = QualityLevel.VERY_POOR
            brightness_score = 0.2
        
        # Adjust score based on uniformity
        brightness_score *= (0.7 + 0.3 * brightness_uniformity)
        
        self.logger.debug(f"Brightness analysis: mean={mean_brightness:.1f}, uniformity={brightness_uniformity:.3f}, score={brightness_score:.3f}, level={brightness_level.value}")
        
        return brightness_score, brightness_level, mean_brightness, brightness_uniformity


class QualityAnalyzer:
    """
    Main image quality analyzer that orchestrates all quality analysis components.
    Provides comprehensive quality metrics without making processing decisions.
    """
    
    def __init__(self, config: OCRConfig):
        """
        Initialize quality analyzer with configuration.
        
        Args:
            config (OCRConfig): OCR configuration
        """
        self.config = config
        self.logger = OCRLogger()
        
        # Initialize component analyzers
        self.blur_analyzer = BlurAnalyzer(config)
        self.noise_analyzer = NoiseAnalyzer(config)
        self.contrast_analyzer = ContrastAnalyzer(config)
        self.resolution_analyzer = ResolutionAnalyzer(config)
        self.brightness_analyzer = BrightnessAnalyzer(config)
        
        # Quality weighting factors from config
        self.weight_blur = config.get("quality.weights.blur", 0.25)
        self.weight_noise = config.get("quality.weights.noise", 0.20)
        self.weight_contrast = config.get("quality.weights.contrast", 0.20)
        self.weight_resolution = config.get("quality.weights.resolution", 0.15)
        self.weight_brightness = config.get("quality.weights.brightness", 0.20)
        
        # Normalize weights
        total_weight = (self.weight_blur + self.weight_noise + self.weight_contrast + 
                       self.weight_resolution + self.weight_brightness)
        if total_weight > 0:
            self.weight_blur /= total_weight
            self.weight_noise /= total_weight
            self.weight_contrast /= total_weight
            self.weight_resolution /= total_weight
            self.weight_brightness /= total_weight
    
    def analyze_image_quality(self, image: np.ndarray) -> QualityMetrics:
        """
        Comprehensive image quality analysis.
        
        CRITICAL: This is the ONLY method called by image_processor.py
        Returns quality metrics for intelligent preprocessing decisions.
        
        Args:
            image (np.ndarray): Input image (original, not preprocessed)
            
        Returns:
            QualityMetrics: Comprehensive quality analysis results
        """
        start_time = cv2.getTickCount()
        
        self.logger.debug(f"Starting quality analysis for image shape: {image.shape}")
        
        # Analyze blur
        blur_score, blur_level, laplacian_variance = self.blur_analyzer.analyze_blur(image)
        
        # Analyze noise
        noise_score, noise_level, noise_variance = self.noise_analyzer.analyze_noise(image)
        
        # Analyze contrast
        contrast_score, contrast_level, contrast_rms, histogram_spread = self.contrast_analyzer.analyze_contrast(image)
        
        # Analyze resolution
        resolution_score, resolution_level, effective_resolution, dpi_estimate = self.resolution_analyzer.analyze_resolution(image)
        
        # Analyze brightness
        brightness_score, brightness_level, mean_brightness, brightness_uniformity = self.brightness_analyzer.analyze_brightness(image)
        
        # Calculate overall quality score
        overall_score = (
            self.weight_blur * blur_score +
            self.weight_noise * noise_score +
            self.weight_contrast * contrast_score +
            self.weight_resolution * resolution_score +
            self.weight_brightness * brightness_score
        )
        
        # Determine overall quality level
        if overall_score >= 0.9:
            overall_level = QualityLevel.EXCELLENT
        elif overall_score >= 0.75:
            overall_level = QualityLevel.GOOD
        elif overall_score >= 0.6:
            overall_level = QualityLevel.FAIR
        elif overall_score >= 0.4:
            overall_level = QualityLevel.POOR
        else:
            overall_level = QualityLevel.VERY_POOR
        
        # Calculate analysis time
        end_time = cv2.getTickCount()
        analysis_time = (end_time - start_time) / cv2.getTickFrequency()
        
        # Get image metadata
        height, width = image.shape[:2]
        color_channels = image.shape[2] if len(image.shape) == 3 else 1
        
        # Create comprehensive quality metrics
        quality_metrics = QualityMetrics(
            blur_score=blur_score,
            blur_level=blur_level,
            laplacian_variance=laplacian_variance,
            noise_score=noise_score,
            noise_level=noise_level,
            noise_variance=noise_variance,
            contrast_score=contrast_score,
            contrast_level=contrast_level,
            contrast_rms=contrast_rms,
            histogram_spread=histogram_spread,
            resolution_score=resolution_score,
            resolution_level=resolution_level,
            effective_resolution=effective_resolution,
            dpi_estimate=dpi_estimate,
            brightness_score=brightness_score,
            brightness_level=brightness_level,
            mean_brightness=mean_brightness,
            brightness_uniformity=brightness_uniformity,
            overall_score=overall_score,
            overall_level=overall_level,
            image_dimensions=(width, height),
            color_channels=color_channels,
            analysis_time=analysis_time
        )
        
        self.logger.info(
            f"Quality analysis completed: overall={overall_level.value} ({overall_score:.3f}), "
            f"blur={blur_level.value} ({blur_score:.3f}), "
            f"noise={noise_level.value} ({noise_score:.3f}), "
            f"contrast={contrast_level.value} ({contrast_score:.3f}), "
            f"resolution={resolution_level.value} ({resolution_score:.3f}), "
            f"brightness={brightness_level.value} ({brightness_score:.3f}), "
            f"time={analysis_time:.3f}s"
        )
        
        return quality_metrics
    
    def get_analysis_config(self) -> Dict[str, Any]:
        """
        Get current analysis configuration.
        
        Returns:
            Dict[str, Any]: Analysis configuration
        """
        return {
            'weights': {
                'blur': self.weight_blur,
                'noise': self.weight_noise,
                'contrast': self.weight_contrast,
                'resolution': self.weight_resolution,
                'brightness': self.weight_brightness
            },
            'thresholds': {
                'blur_excellent': self.blur_analyzer.laplacian_threshold_excellent,
                'noise_excellent': self.noise_analyzer.noise_threshold_excellent,
                'contrast_excellent': self.contrast_analyzer.rms_excellent,
                'resolution_min_width': self.resolution_analyzer.min_width_excellent,
                'brightness_optimal_min': self.brightness_analyzer.optimal_brightness_min
            }
        }


# Utility functions for external use
def create_quality_analyzer(config: Optional[OCRConfig] = None) -> QualityAnalyzer:
    """
    Create a quality analyzer instance.
    
    Args:
        config (Optional[OCRConfig]): OCR configuration
        
    Returns:
        QualityAnalyzer: Configured quality analyzer
    """
    if config is None:
        from advanced_ocr.config import OCRConfig
        config = OCRConfig()
    
    return QualityAnalyzer(config)


def get_quality_level_description(level: QualityLevel) -> str:
    """
    Get human-readable description of quality level.
    
    Args:
        level (QualityLevel): Quality level enum
        
    Returns:
        str: Description of the quality level
    """
    descriptions = {
        QualityLevel.EXCELLENT: "Excellent quality - optimal for OCR",
        QualityLevel.GOOD: "Good quality - suitable for OCR with minimal preprocessing",
        QualityLevel.FAIR: "Fair quality - may benefit from preprocessing",
        QualityLevel.POOR: "Poor quality - requires significant preprocessing",
        QualityLevel.VERY_POOR: "Very poor quality - may not be suitable for OCR"
    }
    
    return descriptions.get(level, "Unknown quality level")


def analyze_quality_trends(quality_history: List[QualityMetrics]) -> Dict[str, Any]:
    """
    Analyze trends in quality metrics over multiple images.
    
    Args:
        quality_history (List[QualityMetrics]): List of quality metrics from multiple images
        
    Returns:
        Dict[str, Any]: Quality trend analysis
    """
    if not quality_history:
        return {}
    
    # Extract scores for trend analysis
    overall_scores = [q.overall_score for q in quality_history]
    blur_scores = [q.blur_score for q in quality_history]
    noise_scores = [q.noise_score for q in quality_history]
    
    return {
        'count': len(quality_history),
        'overall_mean': np.mean(overall_scores),
        'overall_std': np.std(overall_scores),
        'blur_mean': np.mean(blur_scores),
        'noise_mean': np.mean(noise_scores),
        'trend_improving': overall_scores[-1] > overall_scores[0] if len(overall_scores) > 1 else None
    }

__all__ = [
    'QualityLevel', 'QualityMetrics', 'BlurAnalyzer', 'NoiseAnalyzer', 
    'ContrastAnalyzer', 'ResolutionAnalyzer', 'BrightnessAnalyzer', 'QualityAnalyzer',
    'create_quality_analyzer', 'get_quality_level_description', 'analyze_quality_trends'
]