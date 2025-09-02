"""
Enhanced Image Preprocessing for OCR
Fixes common issues: contrast, noise, skew, blur
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import logging
from typing import Tuple, Optional
import math

logger = logging.getLogger(__name__)

class ImagePreprocessor:
    """Advanced image preprocessing for better OCR results"""
    
    def __init__(self):
        self.debug_mode = True
        
    def enhance_image_for_ocr(self, image_path: str, save_debug: bool = False) -> np.ndarray:
        """
        Comprehensive image enhancement pipeline for OCR
        
        Args:
            image_path: Path to input image
            save_debug: Save intermediate processing steps
            
        Returns:
            Enhanced image as numpy array
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
                
            logger.info(f"Original image shape: {image.shape}")
            original = image.copy()
            
            # Step 1: Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            if save_debug:
                cv2.imwrite("debug_01_grayscale.jpg", gray)
            
            # Step 2: Noise reduction
            gray = self.reduce_noise(gray)
            if save_debug:
                cv2.imwrite("debug_02_denoised.jpg", gray)
            
            # Step 3: Fix skew/rotation
            gray = self.correct_skew(gray)
            if save_debug:
                cv2.imwrite("debug_03_deskewed.jpg", gray)
            
            # Step 4: Enhance contrast - CRITICAL for OCR
            gray = self.enhance_contrast_adaptive(gray)
            if save_debug:
                cv2.imwrite("debug_04_contrast.jpg", gray)
            
            # Step 5: Sharpen text
            gray = self.sharpen_text(gray)
            if save_debug:
                cv2.imwrite("debug_05_sharpened.jpg", gray)
            
            # Step 6: Binarization (make text pure black on white)
            binary = self.adaptive_threshold(gray)
            if save_debug:
                cv2.imwrite("debug_06_binary.jpg", binary)
            
            # Step 7: Final cleanup
            final = self.morphological_cleanup(binary)
            if save_debug:
                cv2.imwrite("debug_07_final.jpg", final)
            
            # Resize if too small (OCR works better on larger images)
            final = self.resize_for_ocr(final)
            if save_debug:
                cv2.imwrite("debug_08_resized.jpg", final)
            
            logger.info(f"Enhanced image shape: {final.shape}")
            return final
            
        except Exception as e:
            logger.error(f"Image preprocessing failed: {e}")
            # Return original grayscale as fallback
            return cv2.cvtColor(original, cv2.COLOR_BGR2GRAY) if len(original.shape) == 3 else original
    
    def reduce_noise(self, image: np.ndarray) -> np.ndarray:
        """Remove noise while preserving text edges"""
        # Use bilateral filter - removes noise but keeps edges sharp
        denoised = cv2.bilateralFilter(image, 9, 75, 75)
        return denoised
    
    def correct_skew(self, image: np.ndarray) -> np.ndarray:
        """Detect and correct document skew"""
        try:
            # Find text regions
            coords = np.column_stack(np.where(image > 0))
            if len(coords) < 100:  # Not enough points
                return image
                
            # Find minimum area rectangle
            rect = cv2.minAreaRect(coords)
            angle = rect[2]
            
            # Adjust angle
            if angle < -45:
                angle = 90 + angle
            elif angle > 45:
                angle = angle - 90
                
            # Only correct if angle is significant
            if abs(angle) > 0.5:
                logger.info(f"Correcting skew: {angle:.2f} degrees")
                height, width = image.shape[:2]
                center = (width // 2, height // 2)
                matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                corrected = cv2.warpAffine(image, matrix, (width, height), 
                                         flags=cv2.INTER_CUBIC, 
                                         borderMode=cv2.BORDER_REPLICATE)
                return corrected
            
            return image
            
        except Exception as e:
            logger.warning(f"Skew correction failed: {e}")
            return image
    
    def enhance_contrast_adaptive(self, image: np.ndarray) -> np.ndarray:
        """Enhanced contrast using multiple techniques"""
        # Method 1: CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)
        
        # Method 2: Gamma correction for dark images
        gamma = self.calculate_optimal_gamma(image)
        if gamma != 1.0:
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255
                            for i in np.arange(0, 256)]).astype("uint8")
            enhanced = cv2.LUT(enhanced, table)
        
        # Method 3: Normalize contrast
        enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
        
        return enhanced
    
    def calculate_optimal_gamma(self, image: np.ndarray) -> float:
        """Calculate optimal gamma for image brightness"""
        mean_brightness = np.mean(image)
        
        if mean_brightness < 60:  # Very dark
            return 0.5
        elif mean_brightness < 100:  # Dark
            return 0.7
        elif mean_brightness > 180:  # Very bright
            return 1.5
        else:  # Normal
            return 1.0
    
    def sharpen_text(self, image: np.ndarray) -> np.ndarray:
        """Sharpen text edges for better OCR"""
        # Unsharp masking
        gaussian = cv2.GaussianBlur(image, (0, 0), 2.0)
        sharpened = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
        
        # Alternative: Use a sharpening kernel
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        sharpened = cv2.filter2D(sharpened, -1, kernel)
        
        return sharpened
    
    def adaptive_threshold(self, image: np.ndarray) -> np.ndarray:
        """Convert to binary image with adaptive thresholding"""
        # Try multiple threshold methods and pick the best
        
        # Method 1: Otsu's thresholding
        _, thresh1 = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Method 2: Adaptive threshold
        thresh2 = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
        
        # Method 3: Adaptive threshold with different parameters
        thresh3 = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                       cv2.THRESH_BINARY, 15, 3)
        
        # Choose the method that produces the most text-like regions
        best_thresh = self.select_best_threshold(image, [thresh1, thresh2, thresh3])
        
        return best_thresh
    
    def select_best_threshold(self, original: np.ndarray, thresholds: list) -> np.ndarray:
        """Select the threshold method that produces the best text regions"""
        best_score = -1
        best_thresh = thresholds[0]
        
        for thresh in thresholds:
            # Count connected components (text regions)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)
            
            # Score based on number and size of components
            # Good text images have moderate number of reasonably sized components
            valid_components = 0
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                width = stats[i, cv2.CC_STAT_WIDTH]
                height = stats[i, cv2.CC_STAT_HEIGHT]
                
                # Filter for text-like components
                if 10 < area < 10000 and 2 < width < 500 and 5 < height < 100:
                    valid_components += 1
            
            score = valid_components
            if score > best_score:
                best_score = score
                best_thresh = thresh
        
        logger.info(f"Selected threshold with {best_score} text regions")
        return best_thresh
    
    def morphological_cleanup(self, image: np.ndarray) -> np.ndarray:
        """Clean up binary image using morphological operations"""
        # Remove small noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        cleaned = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        
        # Connect broken characters
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    def resize_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """Resize image to optimal size for OCR"""
        height, width = image.shape[:2]
        
        # Target height around 64-100 pixels for typical text
        # But ensure minimum resolution for small text
        min_height = 300
        max_height = 2000
        
        if height < min_height:
            # Scale up small images
            scale_factor = min_height / height
            new_width = int(width * scale_factor)
            new_height = min_height
            resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            logger.info(f"Upscaled image from {width}x{height} to {new_width}x{new_height}")
            return resized
        elif height > max_height:
            # Scale down very large images
            scale_factor = max_height / height
            new_width = int(width * scale_factor)
            new_height = max_height
            resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            logger.info(f"Downscaled image from {width}x{height} to {new_width}x{new_height}")
            return resized
        
        return image
    
    def analyze_image_quality(self, image: np.ndarray) -> dict:
        """Analyze image quality metrics"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Calculate blur score
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Calculate brightness
        brightness = np.mean(gray)
        
        # Calculate contrast
        contrast = np.std(gray)
        
        # Quality assessment
        quality = {
            'blur_score': float(blur_score),
            'brightness': float(brightness),
            'contrast': float(contrast),
            'is_blurry': blur_score < 100,
            'is_dark': brightness < 80,
            'is_bright': brightness > 200,
            'low_contrast': contrast < 50
        }
        
        return quality


def preprocess_for_paddle(image_path: str) -> np.ndarray:
    """Preprocessing optimized for PaddleOCR"""
    preprocessor = ImagePreprocessor()
    enhanced = preprocessor.enhance_image_for_ocr(image_path, save_debug=True)
    
    # PaddleOCR works better with 3-channel images
    if len(enhanced.shape) == 2:
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    
    return enhanced


def preprocess_for_trocr(image_path: str) -> Image.Image:
    """Preprocessing optimized for TrOCR (handwritten text)"""
    preprocessor = ImagePreprocessor()
    enhanced = preprocessor.enhance_image_for_ocr(image_path)
    
    # TrOCR expects PIL Image
    pil_image = Image.fromarray(enhanced)
    
    # Additional enhancement for handwritten text
    enhancer = ImageEnhance.Contrast(pil_image)
    pil_image = enhancer.enhance(1.5)
    
    enhancer = ImageEnhance.Sharpness(pil_image)
    pil_image = enhancer.enhance(1.2)
    
    return pil_image


def preprocess_for_easyocr(image_path: str) -> np.ndarray:
    """Preprocessing optimized for EasyOCR"""
    preprocessor = ImagePreprocessor()
    enhanced = preprocessor.enhance_image_for_ocr(image_path)
    return enhanced


def preprocess_for_tesseract(image_path: str) -> np.ndarray:
    """Preprocessing optimized for Tesseract"""
    preprocessor = ImagePreprocessor()
    enhanced = preprocessor.enhance_image_for_ocr(image_path)
    
    # Tesseract works better with high DPI
    # Ensure minimum size
    height, width = enhanced.shape[:2]
    if height < 100:
        scale = 100 / height
        new_width = int(width * scale)
        enhanced = cv2.resize(enhanced, (new_width, 100), interpolation=cv2.INTER_CUBIC)
    
    return enhanced