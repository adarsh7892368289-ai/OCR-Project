"""
Advanced OCR System - Image Quality Analysis
Analyze image quality to guide enhancement decisions for optimal OCR results.
"""

import cv2
import numpy as np
import logging
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum
import math

from ..config import OCRConfig


class QualityIssue(Enum):
    """Types of image quality issues that can affect OCR."""
    
    BLUR = "blur"
    NOISE = "noise"
    LOW_CONTRAST = "low_contrast"
    POOR_RESOLUTION = "poor_resolution"
    SKEW = "skew"
    UNEVEN_LIGHTING = "uneven_lighting"
    ARTIFACTS = "artifacts"


@dataclass
class QualityMetrics:
    """Comprehensive image quality metrics for OCR optimization."""
    
    # Core quality scores (0.0 = poor, 1.0 = excellent)
    blur_score: float = 0.0
    noise_score: float = 0.0
    contrast_score: float = 0.0
    resolution_score: float = 0.0
    lighting_score: float = 0.0
    
    # Geometric properties
    skew_angle: float = 0.0
    text_orientation: float = 0.0
    
    # Overall assessment
    overall_score: float = 0.0
    ocr_readiness: float = 0.0
    
    # Detected issues and recommendations
    issues: List[QualityIssue] = None
    recommendations: List[str] = None
    
    # Enhancement parameters (for adaptive processing)
    enhancement_params: Dict[str, float] = None
    
    def __post_init__(self):
        if self.issues is None:
            self.issues = []
        if self.recommendations is None:
            self.recommendations = []
        if self.enhancement_params is None:
            self.enhancement_params = {}


class QualityAnalyzer:
    """
    Advanced image quality analyzer for OCR preprocessing.
    Provides comprehensive quality assessment and enhancement recommendations.
    """
    
    def __init__(self, config: Optional[OCRConfig] = None):
        """
        Initialize quality analyzer.
        
        Args:
            config: OCR configuration with quality thresholds
        """
        self.config = config or OCRConfig()
        self.logger = logging.getLogger(__name__)
        
        # CONSERVATIVE Quality thresholds - much less sensitive
        self.thresholds = {
            'blur_threshold': 50.0,       # Lower threshold (was 100.0)
            'noise_threshold': 0.25,      # Higher tolerance (was 0.15) 
            'contrast_threshold': 0.2,    # Lower requirement (was 0.3)
            'resolution_threshold': 100,  # Lower requirement (was 150)
            'skew_threshold': 3.0,        # Higher tolerance (was 2.0)
            'lighting_variance': 0.1      # Much lower sensitivity (was 0.2)
        }
        
        # Update thresholds from config if available
        if hasattr(self.config, 'quality_thresholds'):
            self.thresholds.update(self.config.quality_thresholds)
    
    def analyze_quality(self, image: np.ndarray) -> QualityMetrics:
        """
        Perform comprehensive quality analysis of the image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            QualityMetrics with detailed quality assessment
        """
        if image is None or image.size == 0:
            self.logger.warning("Empty or invalid image provided")
            return QualityMetrics()
        
        try:
            # Ensure image is in correct format
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            metrics = QualityMetrics()
            
            # Core quality assessments
            metrics.blur_score = self._assess_blur(gray)
            metrics.noise_score = self._assess_noise(gray)
            metrics.contrast_score = self._assess_contrast(gray)
            metrics.resolution_score = self._assess_resolution(gray)
            metrics.lighting_score = self._assess_lighting(gray)
            
            # Geometric assessments
            metrics.skew_angle = self._detect_skew(gray)
            metrics.text_orientation = self._detect_text_orientation(gray)
            
            # Overall scores
            metrics.overall_score = self._calculate_overall_score(metrics)
            metrics.ocr_readiness = self._calculate_ocr_readiness(metrics)
            
            # Issue detection and recommendations (more conservative)
            metrics.issues = self._detect_issues_conservative(metrics)
            metrics.recommendations = self._generate_recommendations_conservative(metrics)
            metrics.enhancement_params = self._calculate_enhancement_params_conservative(metrics)
            
            self.logger.debug(
                f"Quality analysis: Overall={metrics.overall_score:.3f}, "
                f"OCR Readiness={metrics.ocr_readiness:.3f}, "
                f"Issues={len(metrics.issues)}"
            )
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Quality analysis failed: {e}")
            return QualityMetrics(overall_score=0.5, ocr_readiness=0.5)
    
    def _assess_blur(self, gray: np.ndarray) -> float:
        """
        Assess image blur using Laplacian variance method.
        
        Args:
            gray: Grayscale image
            
        Returns:
            Blur score (0.0 = very blurry, 1.0 = sharp)
        """
        try:
            # Compute Laplacian variance
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            variance = laplacian.var()
            
            # Also compute gradient magnitude for additional validation
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2).mean()
            
            # Combine both metrics (more lenient thresholds)
            blur_score = min(variance / self.thresholds['blur_threshold'], 1.0)
            gradient_score = min(gradient_magnitude / 30.0, 1.0)  # Lower threshold
            
            # Weighted combination
            final_score = (blur_score * 0.7 + gradient_score * 0.3)
            
            return min(final_score, 1.0)
            
        except Exception as e:
            self.logger.warning(f"Blur assessment failed: {e}")
            return 0.5
    
    def _assess_noise(self, gray: np.ndarray) -> float:
        """
        Assess image noise level.
        
        Args:
            gray: Grayscale image
            
        Returns:
            Noise score (0.0 = very noisy, 1.0 = clean)
        """
        try:
            # Method 1: Standard deviation of Laplacian
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            noise_level_1 = laplacian.std()
            
            # Method 2: Difference from median filtered image
            median_filtered = cv2.medianBlur(gray, 5)
            noise_level_2 = np.std(gray.astype(np.float32) - median_filtered.astype(np.float32))
            
            # Method 3: High frequency content analysis
            kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
            high_freq = cv2.filter2D(gray, cv2.CV_64F, kernel)
            noise_level_3 = np.std(high_freq)
            
            # Combine methods
            combined_noise = (noise_level_1 * 0.4 + noise_level_2 * 0.4 + noise_level_3 * 0.2)
            
            # More lenient normalization
            max_expected_noise = gray.std() * 0.8  # Higher tolerance
            noise_ratio = min(combined_noise / max_expected_noise, 2.0)
            
            # Convert to quality score (more lenient)
            noise_score = max(0.0, 1.0 - (noise_ratio * 0.3))  # Reduced penalty
            
            return noise_score
            
        except Exception as e:
            self.logger.warning(f"Noise assessment failed: {e}")
            return 0.5
    
    def _assess_contrast(self, gray: np.ndarray) -> float:
        """
        Assess image contrast quality.
        
        Args:
            gray: Grayscale image
            
        Returns:
            Contrast score (0.0 = poor contrast, 1.0 = excellent contrast)
        """
        try:
            # Method 1: RMS contrast
            mean_val = gray.mean()
            if mean_val > 0:
                rms_contrast = gray.std() / mean_val
            else:
                rms_contrast = 0.0
            
            # Method 2: Michelson contrast
            max_val, min_val = gray.max(), gray.min()
            
            # Convert to safe float types BEFORE any operations to prevent overflow
            max_val_safe = np.float64(max_val)
            min_val_safe = np.float64(min_val)
            
            # Now check conditions using safe values
            if max_val_safe > 0 and min_val_safe >= 0:
                denominator = max_val_safe + min_val_safe
                if denominator > 0:
                    michelson_contrast = (max_val_safe - min_val_safe) / denominator
                else:
                    michelson_contrast = 0.0
            else:
                michelson_contrast = 0.0
            
            # Method 3: Histogram analysis
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            hist_spread = np.sum(hist > np.max(hist) * 0.01)  # Count significant bins
            spread_score = min(hist_spread / 128.0, 1.0)  # Normalize
            
            # Combine methods (more lenient thresholds)
            contrast_score = (
                min(rms_contrast / 0.2, 1.0) * 0.4 +  # Lower threshold
                michelson_contrast * 0.4 +
                spread_score * 0.2
            )
            
            return min(contrast_score, 1.0)
            
        except Exception as e:
            self.logger.warning(f"Contrast assessment failed: {e}")
            return 0.5
    
    def _assess_resolution(self, gray: np.ndarray) -> float:
        """
        Assess image resolution adequacy for OCR.
        
        Args:
            gray: Grayscale image
            
        Returns:
            Resolution score (0.0 = too low, 1.0 = adequate/high)
        """
        try:
            height, width = gray.shape
            total_pixels = height * width
            
            # More lenient resolution assessment
            # Use edge detection to find potential text regions
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Analyze contour heights as proxy for text size
                heights = []
                for contour in contours:
                    _, _, _, h = cv2.boundingRect(contour)
                    if 3 < h < height // 4:  # More lenient filtering
                        heights.append(h)
                
                if heights:
                    median_text_height = np.median(heights)
                    # More lenient - 15 pixels is acceptable
                    resolution_score = min(median_text_height / 15.0, 1.0)
                else:
                    # Fallback: assume reasonable text size
                    estimated_text_height = height / 20
                    resolution_score = min(estimated_text_height / 12.0, 1.0)  # More lenient
            else:
                # No edges found - more lenient pixel count requirement
                min_pixels = 200 * 150  # Lower requirement
                resolution_score = min(total_pixels / min_pixels, 1.0)
            
            # Bonus for higher resolutions
            if total_pixels > 800000:  # > 0.8MP (lower threshold)
                resolution_score = min(resolution_score * 1.1, 1.0)
            
            return resolution_score
            
        except Exception as e:
            self.logger.warning(f"Resolution assessment failed: {e}")
            return 0.5
    
    def _assess_lighting(self, gray: np.ndarray) -> float:
        """
        Assess lighting uniformity and quality.
        
        Args:
            gray: Grayscale image
            
        Returns:
            Lighting score (0.0 = poor lighting, 1.0 = uniform lighting)
        """
        try:
            # Method 1: Local variance analysis
            h, w = gray.shape
            block_size = min(h, w) // 8
            
            if block_size < 10:
                return 0.8  # More lenient for small images
            
            variances = []
            means = []
            
            for y in range(0, h - block_size, block_size):
                for x in range(0, w - block_size, block_size):
                    block = gray[y:y+block_size, x:x+block_size]
                    variances.append(np.var(block))
                    means.append(np.mean(block))
            
            # More lenient assessment
            mean_variance = np.mean(variances)
            variance_std = np.std(variances)
            mean_std = np.std(means)
            
            # Lighting uniformity score (more tolerant)
            if mean_variance > 0:
                uniformity_score = 1.0 - min(variance_std / mean_variance, 1.0) * 0.7  # Reduced penalty
            else:
                uniformity_score = 1.0
            
            # Mean intensity distribution score (more tolerant)
            mean_distribution_score = 1.0 - min(mean_std / 150.0, 1.0)  # Higher threshold
            
            # Method 2: Gradient-based assessment (more lenient)
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            # Look for gradual lighting changes
            kernel = np.ones((10, 10), np.float32) / 100
            smoothed_grad = cv2.filter2D(gradient_magnitude, -1, kernel)
            lighting_gradient_score = 1.0 - min(np.mean(smoothed_grad) / 70.0, 1.0)  # Higher threshold
            
            # Combine scores (more weight on being lenient)
            lighting_score = (
                uniformity_score * 0.3 +      # Reduced weight
                mean_distribution_score * 0.3 +
                lighting_gradient_score * 0.4  # Increased weight on gradients
            )
            
            return min(lighting_score, 1.0)
            
        except Exception as e:
            self.logger.warning(f"Lighting assessment failed: {e}")
            return 0.6  # More lenient default
    
    def _detect_skew(self, gray: np.ndarray) -> float:
        """
        Detect text skew angle using Hough line detection.
        
        Args:
            gray: Grayscale image
            
        Returns:
            Skew angle in degrees (-45 to +45)
        """
        try:
            # Edge detection
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # Hough line detection
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            if lines is None or len(lines) == 0:
                return 0.0
            
            # Analyze line angles
            angles = []
            for line in lines:
                rho, theta = line[0]
                angle = theta * 180.0 / np.pi - 90.0
                
                # Focus on near-horizontal lines (likely text lines)
                if -45 < angle < 45:
                    angles.append(angle)
            
            if not angles:
                return 0.0
            
            # Use median angle to reduce outlier impact
            skew_angle = np.median(angles)
            
            # Validate reasonable range
            return max(-45.0, min(45.0, skew_angle))
            
        except Exception as e:
            self.logger.warning(f"Skew detection failed: {e}")
            return 0.0
    
    def _detect_text_orientation(self, gray: np.ndarray) -> float:
        """
        Detect overall text orientation.
        
        Args:
            gray: Grayscale image
            
        Returns:
            Text orientation in degrees (0, 90, 180, 270)
        """
        try:
            # Use morphological operations to find text regions
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 2))
            horizontal = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 20))
            vertical = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
            
            # Count horizontal vs vertical features
            h_score = np.sum(horizontal > 0)
            v_score = np.sum(vertical > 0)
            
            # Determine orientation
            if h_score > v_score * 1.5:
                return 0.0  # Horizontal text
            elif v_score > h_score * 1.5:
                return 90.0  # Vertical text
            else:
                return 0.0  # Default to horizontal
                
        except Exception as e:
            self.logger.warning(f"Text orientation detection failed: {e}")
            return 0.0
    
    def _calculate_overall_score(self, metrics: QualityMetrics) -> float:
        """Calculate weighted overall quality score."""
        weights = {
            'blur': 0.25,
            'noise': 0.15,      # Reduced weight
            'contrast': 0.25,
            'resolution': 0.25, # Increased weight
            'lighting': 0.10
        }
        
        overall = (
            metrics.blur_score * weights['blur'] +
            metrics.noise_score * weights['noise'] +
            metrics.contrast_score * weights['contrast'] +
            metrics.resolution_score * weights['resolution'] +
            metrics.lighting_score * weights['lighting']
        )
        
        return min(overall, 1.0)
    
    def _calculate_ocr_readiness(self, metrics: QualityMetrics) -> float:
        """Calculate OCR-specific readiness score."""
        # OCR readiness prioritizes sharpness, contrast, and resolution
        ocr_weights = {
            'blur': 0.30,      # Slightly reduced
            'contrast': 0.30,  
            'resolution': 0.25, # Increased
            'noise': 0.10,     
            'lighting': 0.05   
        }
        
        readiness = (
            metrics.blur_score * ocr_weights['blur'] +
            metrics.contrast_score * ocr_weights['contrast'] +
            metrics.resolution_score * ocr_weights['resolution'] +
            metrics.noise_score * ocr_weights['noise'] +
            metrics.lighting_score * ocr_weights['lighting']
        )
        
        # Reduced penalty for skew
        skew_penalty = min(abs(metrics.skew_angle) / self.thresholds['skew_threshold'], 1.0)
        readiness *= (1.0 - skew_penalty * 0.1)  # Reduced penalty
        
        return min(readiness, 1.0)
    
    def _detect_issues_conservative(self, metrics: QualityMetrics) -> List[QualityIssue]:
        """Detect specific quality issues based on metrics (more conservative)."""
        issues = []
        
        # Much more conservative thresholds
        if metrics.blur_score < 0.4:  # Was 0.6
            issues.append(QualityIssue.BLUR)
        
        if metrics.noise_score < 0.3:  # Was 0.6
            issues.append(QualityIssue.NOISE)
        
        if metrics.contrast_score < 0.3:  # Was 0.5
            issues.append(QualityIssue.LOW_CONTRAST)
        
        if metrics.resolution_score < 0.3:  # Was 0.5
            issues.append(QualityIssue.POOR_RESOLUTION)
        
        if abs(metrics.skew_angle) > self.thresholds['skew_threshold']:  # Now 3.0
            issues.append(QualityIssue.SKEW)
        
        if metrics.lighting_score < 0.3:  # Was 0.5
            issues.append(QualityIssue.UNEVEN_LIGHTING)
        
        return issues
    
    def _generate_recommendations_conservative(self, metrics: QualityMetrics) -> List[str]:
        """Generate enhancement recommendations (more conservative)."""
        recommendations = []
        
        for issue in metrics.issues:
            if issue == QualityIssue.BLUR:
                recommendations.append("Apply sharpening filter")
            elif issue == QualityIssue.NOISE:
                recommendations.append("Apply noise reduction (median filter)")
            elif issue == QualityIssue.LOW_CONTRAST:
                recommendations.append("Enhance contrast (CLAHE or histogram equalization)")
            elif issue == QualityIssue.POOR_RESOLUTION:
                recommendations.append("Consider image upscaling")
            elif issue == QualityIssue.SKEW:
                recommendations.append(f"Correct skew (rotate by {-metrics.skew_angle:.1f}°)")
            elif issue == QualityIssue.UNEVEN_LIGHTING:
                recommendations.append("Apply illumination correction")
        
        if not recommendations:
            recommendations.append("Image quality is good for OCR")
        
        return recommendations
    
    def _calculate_enhancement_params_conservative(self, metrics: QualityMetrics) -> Dict[str, float]:
        """Calculate specific parameters for adaptive enhancement (more conservative)."""
        params = {}
        
        # Much more conservative sharpening parameters
        if metrics.blur_score < 0.5:  # Higher threshold
            sharpening_strength = (0.5 - metrics.blur_score) * 1.0  # Reduced multiplier
            params['sharpen_strength'] = min(sharpening_strength, 0.5)  # Lower cap
        
        # More conservative noise reduction parameters  
        if metrics.noise_score < 0.4:  # Higher threshold
            noise_reduction_strength = (0.4 - metrics.noise_score) * 1.0
            params['noise_reduction_strength'] = min(noise_reduction_strength, 0.5)
        
        # More conservative contrast enhancement parameters
        if metrics.contrast_score < 0.4:  # Higher threshold
            contrast_enhancement = (0.4 - metrics.contrast_score) * 1.0
            params['contrast_enhancement'] = min(contrast_enhancement, 0.5)
        
        # Less aggressive skew correction
        if abs(metrics.skew_angle) > 2.0:  # Higher threshold
            params['rotation_angle'] = -metrics.skew_angle
        
        # Much more conservative illumination correction
        if metrics.lighting_score < 0.3:  # Much lower threshold
            params['illumination_correction'] = True
        
        return params