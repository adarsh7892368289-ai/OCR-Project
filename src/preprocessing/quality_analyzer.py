# src/preprocessing/quality_analyzer.py - Fixed Version

import cv2
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
import logging
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)

class ImageType(Enum):
    """Types of images for OCR processing"""
    PRINTED_TEXT = "printed_text"
    HANDWRITTEN_TEXT = "handwritten_text"
    TABLE_DOCUMENT = "table_document"
    FORM_DOCUMENT = "form_document"
    LOW_QUALITY = "low_quality"
    NATURAL_SCENE = "natural_scene"
    MIXED_CONTENT = "mixed_content"

class ImageQuality(Enum):
    """Image quality levels"""
    VERY_POOR = "very_poor"
    POOR = "poor"
    FAIR = "fair"
    GOOD = "good"
    EXCELLENT = "excellent"

@dataclass
class QualityMetrics:
    """Comprehensive image quality metrics"""
    overall_score: float = 0.7
    sharpness_score: float = 0.8
    noise_level: float = 0.2
    contrast_score: float = 0.6
    brightness_score: float = 0.7
    skew_angle: float = 0.0
    resolution_score: float = 0.8
    text_density: float = 0.5
    image_type: ImageType = ImageType.PRINTED_TEXT
    quality_level: ImageQuality = ImageQuality.GOOD
    
    # Additional metrics
    blur_score: float = 0.8
    uniformity_score: float = 0.7
    edge_density: float = 0.6
    
    # Metadata
    processing_time: float = 0.0
    confidence: float = 0.8
    warnings: List[str] = field(default_factory=list)

class IntelligentQualityAnalyzer:
    """
    Intelligent image quality analyzer for OCR preprocessing
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize quality analyzer with configuration"""
        self.config = config or {}
        
        # Quality thresholds
        self.thresholds = self.config.get("quality_thresholds", {
            "sharpness_min": 100.0,
            "noise_max": 0.3,
            "contrast_min": 0.3,
            "brightness_range": [0.2, 0.8],
            "resolution_min": 150,
            "skew_tolerance": 2.0
        })
        
        # Analysis settings
        self.enable_deep_analysis = self.config.get("enable_deep_analysis", True)
        self.analysis_cache = self.config.get("analysis_cache", {})
        
        logger.info("Quality analyzer initialized")
    
    def analyze_image(self, image: np.ndarray, cache_key: Optional[str] = None) -> QualityMetrics:
        """
        Analyze image quality and characteristics
        
        Args:
            image: Input image as numpy array
            cache_key: Optional cache key for storing results
            
        Returns:
            QualityMetrics object with comprehensive analysis
        """
        import time
        start_time = time.time()
        
        # Check cache first
        if cache_key and cache_key in self.analysis_cache:
            logger.debug(f"Using cached analysis for {cache_key}")
            return self.analysis_cache[cache_key]
        
        # Handle invalid inputs
        if image is None:
            return QualityMetrics(
                overall_score=0.0,
                quality_level=ImageQuality.VERY_POOR,
                warnings=["Image is None"]
            )
        
        if len(image.shape) < 2:
            return QualityMetrics(
                overall_score=0.1,
                quality_level=ImageQuality.VERY_POOR,
                warnings=["Invalid image dimensions"]
            )
        
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Basic metrics
            height, width = gray.shape
            
            # Calculate individual metrics
            sharpness_score = self._calculate_sharpness(gray)
            noise_level = self._calculate_noise_level(gray)
            contrast_score = self._calculate_contrast(gray)
            brightness_score = self._calculate_brightness(gray)
            skew_angle = self._detect_skew_angle(gray)
            resolution_score = self._calculate_resolution_score(width, height)
            
            # Advanced metrics if enabled
            if self.enable_deep_analysis:
                blur_score = self._calculate_blur_score(gray)
                uniformity_score = self._calculate_uniformity(gray)
                edge_density = self._calculate_edge_density(gray)
                text_density = self._estimate_text_density(gray)
                image_type = self._classify_image_type(gray, image)
            else:
                blur_score = sharpness_score
                uniformity_score = 0.7
                edge_density = 0.6
                text_density = 0.5
                image_type = ImageType.PRINTED_TEXT
            
            # Calculate overall score
            overall_score = self._calculate_overall_score(
                sharpness_score, noise_level, contrast_score, 
                brightness_score, resolution_score
            )
            
            # Determine quality level
            quality_level = self._determine_quality_level(overall_score)
            
            # Generate warnings
            warnings = self._generate_warnings(
                sharpness_score, noise_level, contrast_score, 
                brightness_score, abs(skew_angle)
            )
            
            # Create metrics object
            metrics = QualityMetrics(
                overall_score=overall_score,
                sharpness_score=sharpness_score,
                noise_level=noise_level,
                contrast_score=contrast_score,
                brightness_score=brightness_score,
                skew_angle=skew_angle,
                resolution_score=resolution_score,
                text_density=text_density,
                image_type=image_type,
                quality_level=quality_level,
                blur_score=blur_score,
                uniformity_score=uniformity_score,
                edge_density=edge_density,
                processing_time=time.time() - start_time,
                confidence=0.8,
                warnings=warnings
            )
            
            # Cache results if enabled
            if cache_key:
                self.analysis_cache[cache_key] = metrics
            
            logger.debug(f"Quality analysis completed in {metrics.processing_time:.3f}s")
            return metrics
            
        except Exception as e:
            logger.error(f"Quality analysis failed: {e}")
            return QualityMetrics(
                overall_score=0.3,
                quality_level=ImageQuality.POOR,
                warnings=[f"Analysis error: {str(e)}"],
                processing_time=time.time() - start_time
            )
    
    def _calculate_sharpness(self, gray: np.ndarray) -> float:
        """Calculate image sharpness using Laplacian variance"""
        try:
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            variance = laplacian.var()
            
            # Normalize to 0-1 scale
            normalized = min(variance / 500.0, 1.0)
            return normalized
        except:
            return 0.5
    
    def _calculate_noise_level(self, gray: np.ndarray) -> float:
        """Estimate noise level in the image"""
        try:
            # Use difference between original and median filtered
            median_filtered = cv2.medianBlur(gray, 5)
            noise = cv2.absdiff(gray.astype(np.float32), median_filtered.astype(np.float32))
            noise_level = np.mean(noise) / 255.0
            
            return min(noise_level, 1.0)
        except:
            return 0.3
    
    def _calculate_contrast(self, gray: np.ndarray) -> float:
        """Calculate image contrast using standard deviation"""
        try:
            contrast = np.std(gray) / 128.0
            return min(contrast, 1.0)
        except:
            return 0.5
    
    def _calculate_brightness(self, gray: np.ndarray) -> float:
        """Calculate average brightness"""
        try:
            brightness = np.mean(gray) / 255.0
            
            # Optimal brightness is around 0.4-0.6
            if 0.4 <= brightness <= 0.6:
                return 1.0
            elif 0.2 <= brightness <= 0.8:
                return 0.8
            else:
                return 0.5
        except:
            return 0.5
    
    def _detect_skew_angle(self, gray: np.ndarray) -> float:
        """Detect skew angle using Hough line detection"""
        try:
            # Edge detection
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # Hough line detection
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            if lines is not None and len(lines) > 0:
                angles = []
                for rho, theta in lines[:10]:  # Use first 10 lines
                    angle = (theta - np.pi/2) * 180 / np.pi
                    if -45 <= angle <= 45:
                        angles.append(angle)
                
                if angles:
                    return np.median(angles)
            
            return 0.0
        except:
            return 0.0
    
    def _calculate_resolution_score(self, width: int, height: int) -> float:
        """Calculate resolution adequacy score"""
        pixel_count = width * height
        
        if pixel_count >= 1000000:  # 1MP+
            return 1.0
        elif pixel_count >= 500000:  # 500K+
            return 0.8
        elif pixel_count >= 200000:  # 200K+
            return 0.6
        elif pixel_count >= 100000:  # 100K+
            return 0.4
        else:
            return 0.2
    
    def _calculate_blur_score(self, gray: np.ndarray) -> float:
        """Calculate blur score (inverse of blur)"""
        try:
            # Use variance of Laplacian (same as sharpness)
            return self._calculate_sharpness(gray)
        except:
            return 0.5
    
    def _calculate_uniformity(self, gray: np.ndarray) -> float:
        """Calculate lighting uniformity"""
        try:
            # Divide image into blocks and check variation
            h, w = gray.shape
            block_size = min(h, w) // 4
            
            if block_size < 10:
                return 0.7
            
            block_means = []
            for i in range(0, h - block_size, block_size):
                for j in range(0, w - block_size, block_size):
                    block = gray[i:i+block_size, j:j+block_size]
                    block_means.append(np.mean(block))
            
            if len(block_means) > 1:
                uniformity = 1.0 - (np.std(block_means) / np.mean(block_means))
                return max(0.0, min(1.0, uniformity))
            else:
                return 0.7
        except:
            return 0.7
    
    def _calculate_edge_density(self, gray: np.ndarray) -> float:
        """Calculate edge density"""
        try:
            edges = cv2.Canny(gray, 100, 200)
            edge_pixels = np.sum(edges > 0)
            total_pixels = edges.shape[0] * edges.shape[1]
            
            density = edge_pixels / total_pixels
            return min(density * 10, 1.0)  # Scale for typical document images
        except:
            return 0.6
    
    def _estimate_text_density(self, gray: np.ndarray) -> float:
        """Estimate text density in the image"""
        try:
            # Use connected components to estimate text regions
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Invert if needed (text should be black)
            if np.mean(binary) > 127:
                binary = cv2.bitwise_not(binary)
            
            # Find connected components
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
            
            # Filter components by size to identify text-like regions
            text_area = 0
            total_area = gray.shape[0] * gray.shape[1]
            
            for i in range(1, num_labels):  # Skip background
                area = stats[i, cv2.CC_STAT_AREA]
                width = stats[i, cv2.CC_STAT_WIDTH]
                height = stats[i, cv2.CC_STAT_HEIGHT]
                
                # Heuristic for text-like components
                if 10 < area < total_area * 0.01 and 2 < width < total_area * 0.1 and 5 < height < 100:
                    text_area += area
            
            density = text_area / total_area
            return min(density * 5, 1.0)  # Scale for typical documents
        except:
            return 0.5
    
    def _classify_image_type(self, gray: np.ndarray, original: np.ndarray) -> ImageType:
        """Classify the type of image"""
        try:
            # Simple classification based on characteristics
            height, width = gray.shape
            
            # Check aspect ratio for forms/tables
            aspect_ratio = width / height
            
            # Analyze structure
            edges = cv2.Canny(gray, 100, 200)
            
            # Count horizontal and vertical lines
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
            
            horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
            vertical_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel)
            
            h_line_count = np.sum(horizontal_lines > 0)
            v_line_count = np.sum(vertical_lines > 0)
            
            # Classification logic
            if h_line_count > 1000 and v_line_count > 1000:
                return ImageType.TABLE_DOCUMENT
            elif h_line_count > 500 or v_line_count > 500:
                return ImageType.FORM_DOCUMENT
            elif self._has_handwriting_characteristics(gray):
                return ImageType.HANDWRITTEN_TEXT
            else:
                return ImageType.PRINTED_TEXT
                
        except:
            return ImageType.PRINTED_TEXT
    
    def _has_handwriting_characteristics(self, gray: np.ndarray) -> bool:
        """Check for handwriting characteristics"""
        try:
            # Look for irregular stroke patterns
            # This is a simplified heuristic
            edges = cv2.Canny(gray, 50, 150)
            
            # Analyze stroke thickness variation
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            dilated = cv2.dilate(edges, kernel, iterations=1)
            
            # Count thickness variations (simplified)
            variation = np.std(dilated)
            
            return variation > 50  # Threshold for handwriting
        except:
            return False
    
    def _calculate_overall_score(self, sharpness: float, noise: float, 
                               contrast: float, brightness: float, resolution: float) -> float:
        """Calculate weighted overall quality score"""
        
        # Weights for different factors
        weights = {
            'sharpness': 0.25,
            'noise': 0.20,      # Lower noise is better
            'contrast': 0.25,
            'brightness': 0.15,
            'resolution': 0.15
        }
        
        # Invert noise (lower is better)
        noise_score = 1.0 - noise
        
        # Calculate weighted score
        overall = (
            sharpness * weights['sharpness'] +
            noise_score * weights['noise'] +
            contrast * weights['contrast'] +
            brightness * weights['brightness'] +
            resolution * weights['resolution']
        )
        
        return min(max(overall, 0.0), 1.0)
    
    def _determine_quality_level(self, overall_score: float) -> ImageQuality:
        """Determine quality level from overall score"""
        if overall_score >= 0.8:
            return ImageQuality.EXCELLENT
        elif overall_score >= 0.65:
            return ImageQuality.GOOD
        elif overall_score >= 0.45:
            return ImageQuality.FAIR
        elif overall_score >= 0.25:
            return ImageQuality.POOR
        else:
            return ImageQuality.VERY_POOR
    
    def _generate_warnings(self, sharpness: float, noise: float, 
                         contrast: float, brightness: float, skew: float) -> List[str]:
        """Generate warnings based on quality metrics"""
        warnings = []
        
        if sharpness < 0.3:
            warnings.append("Image appears blurry or out of focus")
        
        if noise > 0.4:
            warnings.append("High noise levels detected")
        
        if contrast < 0.3:
            warnings.append("Low contrast may affect text recognition")
        
        if brightness < 0.2 or brightness > 0.8:
            warnings.append("Suboptimal brightness levels")
        
        if abs(skew) > 5.0:
            warnings.append(f"Significant skew detected: {skew:.1f} degrees")
        
        return warnings