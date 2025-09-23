# src/preprocessing/skew_corrector.py - Fixed Version

import cv2
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
import logging
from typing import Dict, List, Any, Optional, Tuple
import time

logger = logging.getLogger(__name__)

@dataclass
class SkewDetectionResult:
    """Result of skew detection operation"""
    angle: float = 0.0
    confidence: float = 0.0
    detection_method: str = "hough"
    processing_time: float = 0.0
    warnings: List[str] = field(default_factory=list)

@dataclass
class SkewCorrectionResult:
    """Result of skew correction operation"""
    corrected_image: np.ndarray
    correction_applied: bool = False
    original_angle: float = 0.0
    corrected_angle: float = 0.0
    processing_time: float = 0.0
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

class EnhancedSkewCorrector:
    """
    Enhanced skew detection and correction for OCR preprocessing
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize skew corrector with configuration"""
        self.config = config or {}
        
        # Detection parameters
        self.angle_range = self.config.get("angle_range", 45.0)
        self.angle_step = self.config.get("angle_step", 0.1)
        self.confidence_threshold = self.config.get("confidence_threshold", 0.7)
        self.enable_validation = self.config.get("enable_validation", True)
        
        # Line detection parameters
        self.min_line_length = self.config.get("min_line_length", 100)
        self.max_line_gap = self.config.get("max_line_gap", 10)
        self.hough_threshold = self.config.get("hough_threshold", 100)
        
        # Correction parameters
        self.correction_quality = self.config.get("correction_quality", "balanced")
        self.preserve_size = self.config.get("preserve_size", False)
        self.border_mode = self.config.get("border_mode", "replicate")
        
        logger.info("Enhanced skew corrector initialized")
    
    def detect_skew(self, image: np.ndarray) -> SkewDetectionResult:
        """
        Detect skew angle in the image
        
        Args:
            image: Input image as numpy array
            
        Returns:
            SkewDetectionResult with detected angle and confidence
        """
        start_time = time.time()
        
        if image is None:
            return SkewDetectionResult(
                angle=0.0,
                confidence=0.0,
                warnings=["Input image is None"]
            )
        
        if len(image.shape) < 2:
            return SkewDetectionResult(
                angle=0.0,
                confidence=0.0,
                warnings=["Invalid image dimensions"]
            )
        
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Try multiple detection methods
            angles_confidences = []
            
            # Method 1: Hough line detection
            angle1, conf1 = self._detect_skew_hough(gray)
            if conf1 > 0.3:
                angles_confidences.append((angle1, conf1, "hough"))
            
            # Method 2: Projection profile
            angle2, conf2 = self._detect_skew_projection(gray)
            if conf2 > 0.3:
                angles_confidences.append((angle2, conf2, "projection"))
            
            # Method 3: Fourier transform (for high accuracy)
            if self.correction_quality in ["high_quality", "preserve_quality"]:
                angle3, conf3 = self._detect_skew_fourier(gray)
                if conf3 > 0.3:
                    angles_confidences.append((angle3, conf3, "fourier"))
            
            # Select best result
            if angles_confidences:
                # Weight by confidence and select best
                best_angle, best_confidence, best_method = max(
                    angles_confidences, key=lambda x: x[1]
                )
            else:
                best_angle, best_confidence, best_method = 0.0, 0.0, "none"
            
            processing_time = time.time() - start_time
            
            return SkewDetectionResult(
                angle=best_angle,
                confidence=best_confidence,
                detection_method=best_method,
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Skew detection failed: {e}")
            
            return SkewDetectionResult(
                angle=0.0,
                confidence=0.0,
                processing_time=processing_time,
                warnings=[f"Detection failed: {str(e)}"]
            )
    
    def correct_skew(self, image: np.ndarray, **params) -> SkewCorrectionResult:
        """
        Detect and correct skew in the image
        
        Args:
            image: Input image as numpy array
            **params: Additional correction parameters
            
        Returns:
            SkewCorrectionResult with corrected image
        """
        start_time = time.time()
        
        if image is None:
            return SkewCorrectionResult(
                corrected_image=np.zeros((100, 100, 3), dtype=np.uint8),
                correction_applied=False,
                processing_time=0.0,
                warnings=["Input image is None"]
            )
        
        if len(image.shape) < 2:
            return SkewCorrectionResult(
                corrected_image=image.copy() if image.size > 0 else np.zeros((100, 100, 3), dtype=np.uint8),
                correction_applied=False,
                processing_time=0.0,
                warnings=["Invalid image dimensions"]
            )
        
        try:
            # Override config with params
            quality = params.get("quality", self.correction_quality)
            angle_threshold = params.get("angle_threshold", 0.5)
            
            # Detect skew
            detection_result = self.detect_skew(image)
            detected_angle = detection_result.angle
            confidence = detection_result.confidence
            
            warnings = []
            
            # Check if correction is needed
            if abs(detected_angle) < angle_threshold:
                processing_time = time.time() - start_time
                return SkewCorrectionResult(
                    corrected_image=image.copy(),
                    correction_applied=False,
                    original_angle=detected_angle,
                    corrected_angle=detected_angle,
                    processing_time=processing_time,
                    warnings=["Skew angle below threshold, no correction applied"]
                )
            
            # Check confidence
            if confidence < self.confidence_threshold:
                warnings.append(f"Low confidence skew detection: {confidence:.2f}")
            
            # Apply correction
            corrected_image = self._apply_rotation(image, -detected_angle, quality)
            
            # Validate correction if enabled
            final_angle = detected_angle
            if self.enable_validation:
                validation_result = self.detect_skew(corrected_image)
                final_angle = validation_result.angle
                
                if abs(final_angle) > abs(detected_angle) * 0.8:
                    warnings.append("Correction validation suggests limited improvement")
            
            processing_time = time.time() - start_time
            
            return SkewCorrectionResult(
                corrected_image=corrected_image,
                correction_applied=True,
                original_angle=detected_angle,
                corrected_angle=final_angle,
                processing_time=processing_time,
                warnings=warnings,
                metadata={
                    "detection_confidence": confidence,
                    "detection_method": detection_result.detection_method,
                    "correction_quality": quality
                }
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"Skew correction failed: {e}")
            
            return SkewCorrectionResult(
                corrected_image=image.copy(),
                correction_applied=False,
                processing_time=processing_time,
                warnings=[f"Correction failed: {str(e)}"]
            )
    
    def _detect_skew_hough(self, gray: np.ndarray) -> Tuple[float, float]:
        """Detect skew using Hough line detection"""
        try:
            # Edge detection
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # Hough line detection
            lines = cv2.HoughLinesP(
                edges, 
                1, 
                np.pi/180, 
                threshold=self.hough_threshold,
                minLineLength=self.min_line_length,
                maxLineGap=self.max_line_gap
            )
            
            if lines is None or len(lines) == 0:
                return 0.0, 0.0
            
            # Calculate angles
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                
                # Normalize to [-45, 45] range
                if angle > 45:
                    angle -= 90
                elif angle < -45:
                    angle += 90
                
                if abs(angle) <= self.angle_range:
                    angles.append(angle)
            
            if not angles:
                return 0.0, 0.0
            
            # Use median angle
            median_angle = np.median(angles)
            
            # Calculate confidence based on consistency
            deviations = [abs(angle - median_angle) for angle in angles]
            consistency = 1.0 - (np.mean(deviations) / 45.0)
            
            # Confidence also depends on number of lines found
            line_confidence = min(len(lines) / 50.0, 1.0)
            
            confidence = (consistency + line_confidence) / 2.0
            
            return median_angle, confidence
            
        except Exception as e:
            logger.warning(f"Hough skew detection failed: {e}")
            return 0.0, 0.0
    
    def _detect_skew_projection(self, gray: np.ndarray) -> Tuple[float, float]:
        """Detect skew using projection profile method"""
        try:
            height, width = gray.shape
            
            # Test different angles
            angles_to_test = np.arange(-self.angle_range, self.angle_range + self.angle_step, self.angle_step)
            
            best_angle = 0.0
            best_score = 0.0
            
            for angle in angles_to_test:
                # Rotate image
                center = (width // 2, height // 2)
                rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(gray, rotation_matrix, (width, height))
                
                # Calculate horizontal projection
                projection = np.sum(rotated, axis=1)
                
                # Score is variance of projection (higher is better)
                score = np.var(projection)
                
                if score > best_score:
                    best_score = score
                    best_angle = angle
            
            # Calculate confidence based on score improvement
            # Compare with 0-degree rotation score
            center = (width // 2, height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, 0.0, 1.0)
            original_proj = np.sum(gray, axis=1)
            original_score = np.var(original_proj)
            
            if original_score > 0:
                improvement = (best_score - original_score) / original_score
                confidence = min(improvement / 0.5, 1.0)  # Normalize
            else:
                confidence = 0.0
            
            return best_angle, max(0.0, confidence)
            
        except Exception as e:
            logger.warning(f"Projection skew detection failed: {e}")
            return 0.0, 0.0
    
    def _detect_skew_fourier(self, gray: np.ndarray) -> Tuple[float, float]:
        """Detect skew using Fourier transform method"""
        try:
            # This is a simplified version - full implementation would be more complex
            height, width = gray.shape
            
            # Apply FFT
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            
            # Calculate power spectrum
            magnitude_spectrum = np.log(np.abs(f_shift) + 1)
            
            # Find dominant orientation in frequency domain
            # This is a simplified approach
            center_y, center_x = height // 2, width // 2
            
            # Sample radial lines to find dominant direction
            angles_to_test = np.linspace(-45, 45, 91)
            scores = []
            
            for angle in angles_to_test:
                # Create a line at this angle through the center
                radian = np.radians(angle)
                
                # Sample points along the line
                max_dist = min(center_x, center_y)
                distances = np.arange(1, max_dist, 2)
                
                line_sum = 0
                for dist in distances:
                    x = int(center_x + dist * np.cos(radian))
                    y = int(center_y + dist * np.sin(radian))
                    
                    if 0 <= x < width and 0 <= y < height:
                        line_sum += magnitude_spectrum[y, x]
                
                scores.append(line_sum)
            
            if scores:
                best_idx = np.argmax(scores)
                best_angle = angles_to_test[best_idx]
                
                # Calculate confidence
                max_score = max(scores)
                mean_score = np.mean(scores)
                if mean_score > 0:
                    confidence = min((max_score - mean_score) / mean_score, 1.0)
                else:
                    confidence = 0.0
                
                return best_angle, confidence
            else:
                return 0.0, 0.0
                
        except Exception as e:
            logger.warning(f"Fourier skew detection failed: {e}")
            return 0.0, 0.0
    
    def _apply_rotation(self, image: np.ndarray, angle: float, quality: str) -> np.ndarray:
        """Apply rotation to correct skew"""
        try:
            height, width = image.shape[:2]
            center = (width // 2, height // 2)
            
            # Create rotation matrix
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # Determine output size
            if self.preserve_size:
                new_width, new_height = width, height
            else:
                # Calculate new size to fit entire rotated image
                cos_angle = abs(rotation_matrix[0, 0])
                sin_angle = abs(rotation_matrix[0, 1])
                
                new_width = int(height * sin_angle + width * cos_angle)
                new_height = int(height * cos_angle + width * sin_angle)
                
                # Adjust translation
                rotation_matrix[0, 2] += (new_width - width) / 2
                rotation_matrix[1, 2] += (new_height - height) / 2
            
            # Determine interpolation method based on quality
            if quality == "fast":
                interpolation = cv2.INTER_NEAREST
            elif quality in ["high_quality", "preserve_quality"]:
                interpolation = cv2.INTER_CUBIC
            else:  # balanced
                interpolation = cv2.INTER_LINEAR
            
            # Determine border mode
            border_modes = {
                "constant": cv2.BORDER_CONSTANT,
                "replicate": cv2.BORDER_REPLICATE,
                "reflect": cv2.BORDER_REFLECT,
                "wrap": cv2.BORDER_WRAP
            }
            border_mode = border_modes.get(self.border_mode, cv2.BORDER_REPLICATE)
            
            # Apply rotation
            corrected = cv2.warpAffine(
                image, 
                rotation_matrix, 
                (new_width, new_height),
                flags=interpolation,
                borderMode=border_mode,
                borderValue=(255, 255, 255)
            )
            
            return corrected
            
        except Exception as e:
            logger.error(f"Rotation application failed: {e}")
            return image.copy()
    
    def batch_correct(self, images: List[np.ndarray], 
                     progress_callback: Optional[callable] = None) -> List[SkewCorrectionResult]:
        """Correct skew for multiple images"""
        results = []
        
        for i, image in enumerate(images):
            result = self.correct_skew(image)
            results.append(result)
            
            if progress_callback:
                progress_callback(i + 1, len(images))
        
        return results
    
    def get_correction_statistics(self) -> Dict[str, Any]:
        """Get skew correction statistics"""
        return {
            "angle_range": self.angle_range,
            "confidence_threshold": self.confidence_threshold,
            "correction_quality": self.correction_quality,
            "validation_enabled": self.enable_validation
        }