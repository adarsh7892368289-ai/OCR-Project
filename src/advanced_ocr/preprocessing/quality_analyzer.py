# src/advanced_ocr/preprocessing/quality_analyzer.py
"""
Image quality analyzer for OCR preprocessing.
Analyzes image characteristics and determines if enhancement is needed.
"""

import cv2
import numpy as np
import logging
from typing import Dict, Any, Optional, List
import time

# Import from our centralized types and config
from ..types import QualityMetrics, ImageType, ImageQuality, ProcessingStrategy
from ..utils.config import get_config_value

logger = logging.getLogger(__name__)


class QualityAnalyzer:
    """
    Analyzes image quality characteristics for OCR processing.
    Determines if image needs enhancement and recommends processing strategy.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize quality analyzer with configuration.
        
        Args:
            config: Configuration dictionary from config system
        """
        self.config = config
        
        # Extract quality thresholds from config
        self.thresholds = {
            'sharpness_min': get_config_value(config, 'quality_analyzer.sharpness_threshold', 100.0),
            'noise_max': get_config_value(config, 'quality_analyzer.noise_threshold', 0.1),
            'contrast_min': get_config_value(config, 'quality_analyzer.contrast_threshold', 0.3),
            'brightness_min': get_config_value(config, 'quality_analyzer.brightness_min', 50),
            'brightness_max': get_config_value(config, 'quality_analyzer.brightness_max', 200),
        }
        
        logger.debug("Quality analyzer initialized with thresholds: %s", self.thresholds)
    
    def analyze_image(self, image: np.ndarray) -> QualityMetrics:
        """
        Analyze image quality and characteristics.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            QualityMetrics object with analysis results
        """
        start_time = time.time()
        
        # Validate input
        if image is None or len(image.shape) < 2:
            logger.warning("Invalid image provided for analysis")
            return self._create_error_metrics("Invalid image input")
        
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                color_channels = image.shape[2]
            else:
                gray = image.copy()
                color_channels = 1
            
            # Calculate core quality metrics
            sharpness_score = self._calculate_sharpness(gray)
            noise_level = self._calculate_noise_level(gray)
            contrast_score = self._calculate_contrast(gray)
            brightness_score = self._calculate_brightness(gray)
            
            # Calculate additional metrics
            blur_variance = self._calculate_blur_variance(gray)
            edge_density = self._calculate_edge_density(gray)
            text_region_count = self._estimate_text_regions(gray)
            estimated_dpi = self._estimate_dpi(gray.shape)
            
            # Determine image type
            image_type = self._classify_image_type(gray)
            
            # Calculate overall score
            overall_score = self._calculate_overall_score(
                sharpness_score, noise_level, contrast_score, brightness_score
            )
            
            # Determine quality level
            quality_level = self._determine_quality_level(overall_score)
            
            # Determine if enhancement is needed
            needs_enhancement = self._needs_enhancement(
                overall_score, sharpness_score, noise_level, contrast_score
            )
            
            # Recommend processing strategy
            recommended_strategy = self._recommend_strategy(overall_score, needs_enhancement)
            
            # Generate enhancement suggestions
            enhancement_suggestions = self._generate_suggestions(
                sharpness_score, noise_level, contrast_score, brightness_score
            )
            
            # Create metrics object
            metrics = QualityMetrics(
                overall_score=overall_score,
                sharpness_score=sharpness_score,
                noise_level=noise_level,
                contrast_score=contrast_score,
                brightness_score=brightness_score,
                needs_enhancement=needs_enhancement,
                image_type=image_type,
                quality_level=quality_level,
                blur_variance=blur_variance,
                edge_density=edge_density,
                text_region_count=text_region_count,
                estimated_dpi=estimated_dpi,
                color_channels=color_channels,
                recommended_strategy=recommended_strategy,
                enhancement_suggestions=enhancement_suggestions
            )
            
            processing_time = time.time() - start_time
            logger.debug(f"Quality analysis completed in {processing_time:.3f}s - Score: {overall_score:.2f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Quality analysis failed: {e}")
            return self._create_error_metrics(f"Analysis error: {str(e)}")
    
    def _calculate_sharpness(self, gray: np.ndarray) -> float:
        """Calculate image sharpness using Laplacian variance."""
        try:
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            variance = laplacian.var()
            
            # Normalize to 0-1 scale based on threshold
            normalized = min(variance / self.thresholds['sharpness_min'], 1.0)
            return max(0.0, normalized)
        except Exception:
            return 0.5
    
    def _calculate_noise_level(self, gray: np.ndarray) -> float:
        """Estimate noise level using median filtering difference."""
        try:
            # Use median filter to estimate noise
            median_filtered = cv2.medianBlur(gray, 5)
            noise = cv2.absdiff(gray.astype(np.float32), median_filtered.astype(np.float32))
            noise_level = np.mean(noise) / 255.0
            
            return min(max(noise_level, 0.0), 1.0)
        except Exception:
            return 0.1
    
    def _calculate_contrast(self, gray: np.ndarray) -> float:
        """Calculate image contrast using standard deviation."""
        try:
            contrast = np.std(gray) / 128.0
            return min(max(contrast, 0.0), 1.0)
        except Exception:
            return 0.5
    
    def _calculate_brightness(self, gray: np.ndarray) -> float:
        """Calculate brightness quality score."""
        try:
            mean_brightness = np.mean(gray)
            
            # Score based on how close to optimal range
            min_thresh = self.thresholds['brightness_min']
            max_thresh = self.thresholds['brightness_max']
            optimal_center = (min_thresh + max_thresh) / 2
            
            if min_thresh <= mean_brightness <= max_thresh:
                # Within good range - score based on distance from center
                distance_from_optimal = abs(mean_brightness - optimal_center)
                max_distance = (max_thresh - min_thresh) / 2
                return 1.0 - (distance_from_optimal / max_distance) * 0.3
            else:
                # Outside good range - lower score
                if mean_brightness < min_thresh:
                    return 0.3 + (mean_brightness / min_thresh) * 0.4
                else:  # mean_brightness > max_thresh
                    excess = mean_brightness - max_thresh
                    return max(0.1, 0.7 - (excess / (255 - max_thresh)) * 0.6)
        except Exception:
            return 0.5
    
    def _calculate_blur_variance(self, gray: np.ndarray) -> float:
        """Calculate blur variance (same as sharpness calculation)."""
        try:
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            return float(laplacian.var())
        except Exception:
            return 0.0
    
    def _calculate_edge_density(self, gray: np.ndarray) -> float:
        """Calculate edge pixel density."""
        try:
            edges = cv2.Canny(gray, 100, 200)
            edge_pixels = np.sum(edges > 0)
            total_pixels = edges.size
            
            density = edge_pixels / total_pixels
            return min(density * 10, 1.0)  # Scale for typical documents
        except Exception:
            return 0.5
    
    def _estimate_text_regions(self, gray: np.ndarray) -> int:
        """Estimate number of text regions using connected components."""
        try:
            # Threshold image
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Invert if text is white on black
            if np.mean(binary) > 127:
                binary = cv2.bitwise_not(binary)
            
            # Find connected components
            num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
            
            # Count text-like components
            text_regions = 0
            total_area = gray.size
            
            for i in range(1, num_labels):  # Skip background
                area = stats[i, cv2.CC_STAT_AREA]
                width = stats[i, cv2.CC_STAT_WIDTH] 
                height = stats[i, cv2.CC_STAT_HEIGHT]
                
                # Heuristic for text-like regions
                if (10 < area < total_area * 0.01 and 
                    2 < width < total_area * 0.1 and 
                    5 < height < 100):
                    text_regions += 1
            
            return text_regions
        except Exception:
            return 0
    
    def _estimate_dpi(self, shape: tuple) -> int:
        """Estimate DPI based on image dimensions."""
        height, width = shape
        pixel_count = height * width
        
        # Rough DPI estimation based on pixel count
        if pixel_count >= 4000000:    # 4MP+
            return 300
        elif pixel_count >= 2000000:  # 2MP+
            return 250
        elif pixel_count >= 1000000:  # 1MP+
            return 200
        elif pixel_count >= 500000:   # 500K+
            return 150
        else:
            return 100
    
    def _classify_image_type(self, gray: np.ndarray) -> ImageType:
        """Classify image type based on structural analysis."""
        try:
            # Simple classification based on edge patterns
            edges = cv2.Canny(gray, 100, 200)
            
            # Check for table/form structure
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
            
            horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
            vertical_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel)
            
            h_line_count = np.sum(horizontal_lines > 0)
            v_line_count = np.sum(vertical_lines > 0)
            
            # Classification logic
            if h_line_count > 1000 and v_line_count > 1000:
                return ImageType.TABLE
            elif h_line_count > 500 or v_line_count > 500:
                return ImageType.FORM
            else:
                return ImageType.DOCUMENT
        except Exception:
            return ImageType.DOCUMENT
    
    def _calculate_overall_score(self, sharpness: float, noise: float, 
                               contrast: float, brightness: float) -> float:
        """Calculate weighted overall quality score."""
        # Weights for different factors
        weights = {
            'sharpness': 0.30,
            'noise': 0.25,      # Lower noise is better
            'contrast': 0.25,
            'brightness': 0.20
        }
        
        # Invert noise (lower is better)
        noise_score = 1.0 - noise
        
        # Calculate weighted score
        overall = (
            sharpness * weights['sharpness'] +
            noise_score * weights['noise'] +
            contrast * weights['contrast'] +
            brightness * weights['brightness']
        )
        
        return min(max(overall, 0.0), 1.0)
    
    def _determine_quality_level(self, overall_score: float) -> ImageQuality:
        """Determine quality level from overall score."""
        if overall_score >= 0.8:
            return ImageQuality.EXCELLENT
        elif overall_score >= 0.6:
            return ImageQuality.GOOD
        elif overall_score >= 0.4:
            return ImageQuality.FAIR
        elif overall_score >= 0.2:
            return ImageQuality.POOR
        else:
            return ImageQuality.UNUSABLE
    
    def _needs_enhancement(self, overall_score: float, sharpness: float,
                          noise: float, contrast: float) -> bool:
        """Determine if image needs enhancement."""
        return (
            overall_score < 0.6 or          # Low overall quality
            sharpness < 0.4 or              # Blurry image
            noise > self.thresholds['noise_max'] or  # High noise
            contrast < self.thresholds['contrast_min']  # Low contrast
        )
    
    def _recommend_strategy(self, overall_score: float, needs_enhancement: bool) -> ProcessingStrategy:
        """Recommend processing strategy based on quality."""
        if overall_score >= 0.8 and not needs_enhancement:
            return ProcessingStrategy.MINIMAL
        elif overall_score >= 0.4:
            return ProcessingStrategy.BALANCED
        else:
            return ProcessingStrategy.ENHANCED
    
    def _generate_suggestions(self, sharpness: float, noise: float,
                            contrast: float, brightness: float) -> List[str]:
        """Generate enhancement suggestions."""
        suggestions = []
        
        if sharpness < 0.4:
            suggestions.append("Apply sharpening filter")
        
        if noise > self.thresholds['noise_max']:
            suggestions.append("Apply noise reduction")
        
        if contrast < self.thresholds['contrast_min']:
            suggestions.append("Enhance contrast")
        
        if brightness < 0.3:
            suggestions.append("Increase brightness")
        elif brightness > 0.8:
            suggestions.append("Decrease brightness")
        
        return suggestions
    
    def _create_error_metrics(self, error_message: str) -> QualityMetrics:
        """Create metrics object for error cases."""
        return QualityMetrics(
            overall_score=0.1,
            sharpness_score=0.0,
            noise_level=1.0,
            contrast_score=0.0,
            brightness_score=0.0,
            needs_enhancement=True,
            image_type=ImageType.UNKNOWN,
            quality_level=ImageQuality.UNUSABLE,
            recommended_strategy=ProcessingStrategy.ENHANCED,
            enhancement_suggestions=[error_message]
        )