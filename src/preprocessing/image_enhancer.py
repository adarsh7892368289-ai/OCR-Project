# src/preprocessing/image_enhancer.py

import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any
import skimage.restoration
from skimage import filters, morphology, measure
from scipy import ndimage

class ImageEnhancer:
    """Advanced image enhancement for better OCR results"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.enhancement_level = config.get("enhancement_level", "medium")  # low, medium, high
        self.preserve_aspect_ratio = config.get("preserve_aspect_ratio", True)
        
    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        """Main enhancement pipeline"""
        enhanced = image.copy()
        
        # Step 1: Noise reduction
        enhanced = self.reduce_noise(enhanced)
        
        # Step 2: Contrast enhancement
        enhanced = self.enhance_contrast(enhanced)
        
        # Step 3: Sharpening (if needed)
        if self.enhancement_level in ["medium", "high"]:
            enhanced = self.sharpen_image(enhanced)
            
        # Step 4: Morphological cleaning
        if self.enhancement_level == "high":
            enhanced = self.morphological_cleaning(enhanced)
            
        return enhanced
        
    def reduce_noise(self, image: np.ndarray) -> np.ndarray:
        """Advanced noise reduction"""
        if len(image.shape) == 3:
            # Color image
            return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        else:
            # Grayscale image
            return cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
            
    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Multi-level contrast enhancement"""
        if len(image.shape) == 3:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge back
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        else:
            # Grayscale CLAHE
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(image)
            
        return enhanced
        
    def sharpen_image(self, image: np.ndarray) -> np.ndarray:
        """Intelligent sharpening"""
        # Check if image needs sharpening
        if self._is_blurry(image):
            # Create sharpening kernel
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            
            if len(image.shape) == 3:
                sharpened = cv2.filter2D(image, -1, kernel)
                # Blend original and sharpened
                enhanced = cv2.addWeighted(image, 0.6, sharpened, 0.4, 0)
            else:
                sharpened = cv2.filter2D(image, -1, kernel)
                enhanced = cv2.addWeighted(image, 0.6, sharpened, 0.4, 0)
                
            return enhanced
        return image
        
    def _is_blurry(self, image: np.ndarray) -> bool:
        """Detect if image is blurry using Laplacian variance"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var < 100  # Threshold for blur detection
        
    def morphological_cleaning(self, image: np.ndarray) -> np.ndarray:
        """Clean up text using morphological operations"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Remove small noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        cleaned = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        
        # Close gaps in text
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
        
    def correct_illumination(self, image: np.ndarray) -> np.ndarray:
        """Correct uneven illumination"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Create background model
        background = cv2.medianBlur(gray, 19)
        
        # Subtract background
        corrected = cv2.absdiff(gray, background)
        corrected = 255 - corrected
        
        return corrected
        
    def adaptive_threshold(self, image: np.ndarray) -> np.ndarray:
        """Apply adaptive thresholding for better binarization"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Try multiple thresholding methods and choose the best
        methods = [
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        ]
        
        results = []
        for method in methods:
            thresh = cv2.adaptiveThreshold(
                gray, 255, method, cv2.THRESH_BINARY, 15, 10
            )
            # Calculate quality score
            score = self._calculate_threshold_quality(thresh)
            results.append((thresh, score))
            
        # Return best result
        best_thresh, _ = max(results, key=lambda x: x[1])
        return best_thresh
        
    def _calculate_threshold_quality(self, binary_image: np.ndarray) -> float:
        """Calculate quality score for thresholded image"""
        # Count connected components
        num_labels, labels = cv2.connectedComponents(binary_image)
        
        # Prefer moderate number of components
        ideal_components = 50
        component_score = 1.0 / (1.0 + abs(num_labels - ideal_components) / ideal_components)
        
        return component_score