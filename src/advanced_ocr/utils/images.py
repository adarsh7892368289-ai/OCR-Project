"""
Basic image utilities for the Advanced OCR Library.

This module handles ONLY basic image I/O, format conversion, and simple operations.
It does NOT perform enhancement, quality analysis, or complex processing.
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def load_image(image_path: Union[str, Path]) -> Optional[np.ndarray]:
    """
    Load image from file path.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Image as numpy array or None if loading failed
    """
    try:
        image_path = str(image_path)
        image = cv2.imread(image_path)
        
        if image is None:
            logger.error(f"Could not load image from {image_path}")
            return None
            
        return image
        
    except Exception as e:
        logger.error(f"Error loading image {image_path}: {e}")
        return None


def save_image(image: np.ndarray, output_path: Union[str, Path], 
               quality: int = 95) -> bool:
    """
    Save image to file.
    
    Args:
        image: Image as numpy array
        output_path: Path to save image
        quality: JPEG quality (0-100)
        
    Returns:
        True if successful, False otherwise
    """
    try:
        output_path = str(output_path)
        
        # Create directory if it doesn't exist
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save with appropriate parameters based on file extension
        if output_path.lower().endswith(('.jpg', '.jpeg')):
            success = cv2.imwrite(output_path, image, 
                                [cv2.IMWRITE_JPEG_QUALITY, quality])
        else:
            success = cv2.imwrite(output_path, image)
            
        if not success:
            logger.error(f"Failed to save image to {output_path}")
            
        return success
        
    except Exception as e:
        logger.error(f"Error saving image to {output_path}: {e}")
        return False


def validate_image(image: Union[np.ndarray, str, Path]) -> bool:
    """
    Validate that image is usable.
    
    Args:
        image: Image array or path to image
        
    Returns:
        True if image is valid, False otherwise
    """
    try:
        if isinstance(image, (str, Path)):
            # Load and validate file
            img_array = load_image(image)
            if img_array is None:
                return False
            image = img_array
        
        if not isinstance(image, np.ndarray):
            return False
            
        # Check basic properties
        if len(image.shape) < 2:
            return False
            
        if image.shape[0] < 1 or image.shape[1] < 1:
            return False
            
        # Check for valid data type
        if image.dtype not in [np.uint8, np.uint16, np.float32, np.float64]:
            return False
            
        return True
        
    except Exception as e:
        logger.debug(f"Image validation failed: {e}")
        return False


def convert_color_space(image: np.ndarray, conversion: str) -> Optional[np.ndarray]:
    """
    Convert image between color spaces.
    
    Args:
        image: Input image
        conversion: Conversion type (e.g., 'BGR2RGB', 'RGB2GRAY')
        
    Returns:
        Converted image or None if conversion failed
    """
    try:
        conversions = {
            "BGR2RGB": cv2.COLOR_BGR2RGB,
            "RGB2BGR": cv2.COLOR_RGB2BGR,
            "BGR2GRAY": cv2.COLOR_BGR2GRAY,
            "RGB2GRAY": cv2.COLOR_RGB2GRAY,
            "GRAY2BGR": cv2.COLOR_GRAY2BGR,
            "GRAY2RGB": cv2.COLOR_GRAY2RGB,
        }
        
        if conversion not in conversions:
            logger.error(f"Unsupported conversion: {conversion}")
            return None
            
        converted = cv2.cvtColor(image, conversions[conversion])
        return converted
        
    except Exception as e:
        logger.error(f"Color conversion failed: {e}")
        return None


def resize_image(image: np.ndarray, 
                target_size: Tuple[int, int], 
                interpolation: int = cv2.INTER_LANCZOS4) -> Optional[np.ndarray]:
    """
    Resize image to target dimensions.
    
    Args:
        image: Input image
        target_size: (width, height) tuple
        interpolation: OpenCV interpolation method
        
    Returns:
        Resized image or None if failed
    """
    try:
        width, height = target_size
        resized = cv2.resize(image, (width, height), interpolation=interpolation)
        return resized
        
    except Exception as e:
        logger.error(f"Image resize failed: {e}")
        return None


def detect_rotation(image: np.ndarray) -> float:
    """
    Detect rotation angle of text in image using simple Hough line detection.
    
    Args:
        image: Input image
        
    Returns:
        Detected rotation angle in degrees (0, 90, 180, 270)
    """
    try:
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Simple edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Hough line detection
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
        
        if lines is None or len(lines) == 0:
            return 0.0
        
        # Calculate angles
        angles = []
        for line in lines:
            rho, theta = line[0]
            angle = theta * 180 / np.pi
            angles.append(angle)
        
        if not angles:
            return 0.0
        
        # Find most common angle (rounded to nearest 90 degrees)
        median_angle = np.median(angles)
        
        # Round to nearest 90-degree increment
        normalized_angle = round(median_angle / 90) * 90
        return normalized_angle % 360
        
    except Exception as e:
        logger.error(f"Rotation detection failed: {e}")
        return 0.0


def correct_rotation(image: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotate image by specified angle.
    
    Args:
        image: Input image
        angle: Rotation angle in degrees
        
    Returns:
        Rotated image (returns original if rotation fails)
    """
    try:
        if abs(angle) < 0.1:  # No rotation needed
            return image
        
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        
        # Calculate rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Apply rotation
        rotated = cv2.warpAffine(image, rotation_matrix, (width, height))
        return rotated
        
    except Exception as e:
        logger.error(f"Image rotation failed: {e}")
        return image


def get_image_info(image: np.ndarray) -> dict:
    """
    Get basic information about an image.
    
    Args:
        image: Input image
        
    Returns:
        Dictionary with image information
    """
    try:
        info = {
            'width': image.shape[1],
            'height': image.shape[0],
            'channels': len(image.shape),
            'dtype': str(image.dtype),
            'size': image.size,
        }
        
        if len(image.shape) == 3:
            info['channels'] = image.shape[2]
        else:
            info['channels'] = 1
            
        return info
        
    except Exception as e:
        logger.error(f"Could not get image info: {e}")
        return {}


class ImageUtils:
    """
    Utility class providing static methods for basic image operations.
    
    This maintains compatibility with your existing pipeline code.
    """
    
    @staticmethod
    def load_image(image_path: Union[str, Path]) -> Optional[np.ndarray]:
        """Load image from file path"""
        return load_image(image_path)
    
    @staticmethod
    def detect_text_orientation(image: np.ndarray) -> float:
        """Detect text orientation (compatibility method)"""
        return detect_rotation(image)
    
    @staticmethod
    def correct_image_rotation(image: np.ndarray, angle: float) -> np.ndarray:
        """Correct image rotation (compatibility method)"""
        return correct_rotation(image, angle)


# Export public functions
__all__ = [
    'load_image',
    'save_image', 
    'validate_image',
    'convert_color_space',
    'resize_image',
    'detect_rotation',
    'correct_rotation',
    'get_image_info',
]