# Enhanced src/utils/image_utils.py - Additional functions for TrOCR

import cv2
import numpy as np
from typing import Tuple, Optional, List, Union
from PIL import Image
import io

class ImageUtils:
    """Enhanced utility functions for image processing"""
    
    @staticmethod
    def load_image(image_path: str) -> np.ndarray:
        """Load image from file path"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        return image
    
    @staticmethod
    def load_image_from_bytes(image_bytes: bytes) -> np.ndarray:
        """Load image from byte data"""
        image = Image.open(io.BytesIO(image_bytes))
        return np.array(image)
    
    @staticmethod
    def resize_image(image: np.ndarray, size: Union[Tuple[int, int], int] = None, 
                     max_width: int = 2048, max_height: int = 2048) -> np.ndarray:
        """Resize image while maintaining aspect ratio
        
        Args:
            image: Input image
            size: Either (width, height) tuple or single dimension, or None for max constraints
            max_width: Maximum width when size is None
            max_height: Maximum height when size is None
        """
        height, width = image.shape[:2]
        
        if size is not None:
            if isinstance(size, tuple) and len(size) == 2:
                # Direct resize to specific dimensions
                new_width, new_height = size
                return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
            elif isinstance(size, int):
                # Resize to square with given dimension
                new_width = new_height = size
                return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        
        # Use max_width and max_height constraints
        if width <= max_width and height <= max_height:
            return image
        
        # Calculate scaling factor
        scale_w = max_width / width
        scale_h = max_height / height
        scale = min(scale_w, scale_h)
        
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    
    @staticmethod
    def convert_color_space(image: np.ndarray, conversion: str) -> np.ndarray:
        """Convert image color space"""
        conversions = {
            "BGR2RGB": cv2.COLOR_BGR2RGB,
            "RGB2BGR": cv2.COLOR_RGB2BGR,
            "BGR2GRAY": cv2.COLOR_BGR2GRAY,
            "RGB2GRAY": cv2.COLOR_RGB2GRAY,
            "GRAY2BGR": cv2.COLOR_GRAY2BGR,
            "GRAY2RGB": cv2.COLOR_GRAY2RGB
        }
        
        if conversion not in conversions:
            raise ValueError(f"Unsupported conversion: {conversion}")
        
        return cv2.cvtColor(image, conversions[conversion])
    
    @staticmethod
    def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
        """Convert image to grayscale (added for test compatibility)"""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
    
    @staticmethod
    def calculate_image_stats(image: np.ndarray) -> dict:
        """Calculate image statistics"""
        if image is None:
            return {}
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        return {
            "width": image.shape[1],
            "height": image.shape[0],
            "channels": len(image.shape),
            "mean_brightness": float(np.mean(gray)),
            "std_brightness": float(np.std(gray)),
            "min_brightness": int(np.min(gray)),
            "max_brightness": int(np.max(gray)),
            "total_pixels": int(image.size)
        }
    
    @staticmethod
    def save_image(image: np.ndarray, output_path: str, quality: int = 95):
        """Save image to file"""
        success = cv2.imwrite(output_path, image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not success:
            raise ValueError(f"Failed to save image to {output_path}")
    
    # NEW FUNCTIONS FOR TROCR INTEGRATION
    
    @staticmethod
    def preprocess_image_for_recognition(image: Union[np.ndarray, Image.Image]) -> Image.Image:
        """Preprocess image specifically for TrOCR recognition"""
        if isinstance(image, np.ndarray):
            # Convert BGR to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        
        # Ensure RGB format
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply preprocessing steps
        image = ImageUtils._enhance_contrast_for_ocr(image)
        image = ImageUtils._denoise_for_text_recognition(image)
        image = ImageUtils._optimize_resolution_for_ocr(image)
        
        return image
    
    @staticmethod
    def _enhance_contrast_for_ocr(image: Image.Image) -> Image.Image:
        """Enhance contrast specifically for OCR"""
        img_array = np.array(image)
        
        # Convert to LAB color space for better contrast enhancement
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        
        # Apply CLAHE to the L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        
        # Convert back to RGB
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return Image.fromarray(enhanced)
    
    @staticmethod
    def _denoise_for_text_recognition(image: Image.Image) -> Image.Image:
        """Apply gentle denoising that preserves text quality"""
        img_array = np.array(image)
        
        # Use bilateral filter to reduce noise while preserving edges
        # This is gentler than Gaussian blur and preserves text edges better
        denoised = cv2.bilateralFilter(img_array, 9, 75, 75)
        
        return Image.fromarray(denoised)
    
    @staticmethod
    def _optimize_resolution_for_ocr(image: Image.Image) -> Image.Image:
        """Optimize image resolution for OCR engines"""
        width, height = image.size
        
        # OCR works best with certain resolutions
        # Ensure minimum resolution while maintaining aspect ratio
        min_dimension = 384  # Good for TrOCR
        max_dimension = 2048  # Prevent memory issues
        
        # Calculate scaling
        scale = 1.0
        if min(width, height) < min_dimension:
            scale = min_dimension / min(width, height)
        elif max(width, height) > max_dimension:
            scale = max_dimension / max(width, height)
        
        if scale != 1.0:
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = image.resize((new_width, new_height), Image.LANCZOS)
        
        return image
    
    @staticmethod
    def crop_text_region(image: Union[np.ndarray, Image.Image], 
                        bbox: Tuple[int, int, int, int],
                        padding: int = 5) -> Image.Image:
        """Crop a text region from image with optional padding"""
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        
        x, y, w, h = bbox
        
        # Add padding
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(image.width - x, w + 2 * padding)
        h = min(image.height - y, h + 2 * padding)
        
        # Crop the region
        crop_box = (x, y, x + w, y + h)
        cropped = image.crop(crop_box)
        
        return cropped
    
    @staticmethod
    def detect_image_quality(image: Union[np.ndarray, Image.Image]) -> dict:
        """Detect image quality metrics relevant for OCR"""
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Calculate various quality metrics
        
        # 1. Blur detection using Laplacian variance
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # 2. Noise estimation
        noise_score = np.std(gray)
        
        # 3. Contrast measurement
        contrast_score = gray.std()
        
        # 4. Brightness analysis
        brightness = np.mean(gray)
        
        # 5. Resolution check
        height, width = gray.shape
        resolution_score = width * height
        
        return {
            'blur_score': float(blur_score),
            'is_blurry': blur_score < 100,  # Threshold for blur detection
            'noise_level': float(noise_score),
            'contrast_level': float(contrast_score),
            'brightness_level': float(brightness),
            'is_too_dark': brightness < 50,
            'is_too_bright': brightness > 200,
            'resolution': resolution_score,
            'width': width,
            'height': height,
            'recommended_for_ocr': (
                blur_score > 100 and  # Not too blurry
                50 < brightness < 200 and  # Good brightness
                contrast_score > 20  # Sufficient contrast
            )
        }

    @staticmethod
    def detect_text_orientation(image: np.ndarray) -> float:
        """Detect the orientation angle of text in the image using Hough transform for lines."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Use Hough transform to detect lines
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
        
        if lines is None:
            return 0.0  # No rotation detected
        
        angles = []
        for line in lines:
            rho, theta = line[0]
            angle = theta * 180 / np.pi
            angles.append(angle)
        
        # Calculate the most common angle (likely text orientation)
        if angles:
            angle = np.median(angles)
            # Round to nearest 90 degrees for common orientations
            return round(angle / 90) * 90
        return 0.0

    @staticmethod
    def correct_image_rotation(image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate the image by the given angle."""
        if angle == 0:
            return image
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h))
        return rotated
