# src/preprocessing/skew_corrector.py
import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any
import skimage.restoration
from skimage import filters, morphology, measure
from scipy import ndimage

class SkewCorrector:
    """Detect and correct document skew"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.angle_range = config.get("angle_range", 45)
        self.angle_step = config.get("angle_step", 0.5)
        
    def detect_skew(self, image: np.ndarray) -> float:
        """Detect skew angle using Hough transform"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Hough line detection
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        if lines is None:
            return 0.0
            
        # Calculate angles
        angles = []
        for rho, theta in lines[:, 0]:
            angle = np.rad2deg(theta) - 90
            angles.append(angle)
            
        if not angles:
            return 0.0
            
        # Find most common angle
        hist, bins = np.histogram(angles, bins=90, range=(-45, 45))
        peak_angle_idx = np.argmax(hist)
        peak_angle = (bins[peak_angle_idx] + bins[peak_angle_idx + 1]) / 2
        
        return peak_angle
        
    def correct_skew(self, image: np.ndarray, angle: Optional[float] = None) -> np.ndarray:
        """Correct image skew"""
        if angle is None:
            angle = self.detect_skew(image)
            
        if abs(angle) < 0.5:  # Skip correction for very small angles
            return image
            
        # Get image dimensions
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        
        # Create rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate new dimensions after rotation
        cos_val = abs(rotation_matrix[0, 0])
        sin_val = abs(rotation_matrix[0, 1])
        new_width = int((height * sin_val) + (width * cos_val))
        new_height = int((height * cos_val) + (width * sin_val))
        
        # Adjust translation
        rotation_matrix[0, 2] += (new_width / 2) - center[0]
        rotation_matrix[1, 2] += (new_height / 2) - center[1]
        
        # Apply rotation
        corrected = cv2.warpAffine(
            image, rotation_matrix, (new_width, new_height),
            flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )
        
        return corrected