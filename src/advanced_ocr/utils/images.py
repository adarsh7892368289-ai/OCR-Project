"""Basic image utilities for loading, validation, and simple operations.

Provides fundamental image I/O operations, format conversions, and basic
transformations. Does NOT perform enhancement, quality analysis, or complex
image processing - those belong in preprocessing modules.

Examples
--------
    from advanced_ocr.utils.images import load_image, validate_image
    from advanced_ocr.utils.images import detect_rotation, correct_rotation
    
    image = load_image("document.jpg")
    if validate_image(image):
        angle = detect_rotation(image)
        if abs(angle) > 1:
            corrected = correct_rotation(image, angle)
"""

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
from typing import Union, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def load_image(image_path: Union[str, Path]) -> Optional[np.ndarray]:
    """Load an image from file."""
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
    """Save an image to file."""
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
    """Validate that an image is usable."""
    try:
        if isinstance(image, (str, Path)):
            img_array = load_image(image)
            if img_array is None:
                return False
            image = img_array
        
        if not isinstance(image, np.ndarray):
            return False
            
        if len(image.shape) < 2:
            return False
            
        if image.shape[0] < 1 or image.shape[1] < 1:
            return False
            
        if image.dtype not in [np.uint8, np.uint16, np.float32, np.float64]:
            return False
            
        return True
        
    except Exception as e:
        logger.debug(f"Image validation failed: {e}")
        return False


def convert_color_space(image: np.ndarray, conversion: str) -> Optional[np.ndarray]:
    """Convert image between color spaces (e.g., 'BGR2RGB', 'BGR2GRAY')."""
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
    """Resize image to target dimensions (width, height)."""
    try:
        width, height = target_size
        resized = cv2.resize(image, (width, height), interpolation=interpolation)
        return resized
        
    except Exception as e:
        logger.error(f"Image resize failed: {e}")
        return None


def detect_rotation(image: np.ndarray) -> float:
    """Detect rotation angle of text using Hough line detection."""
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
        
        # Round to nearest 90-degree increment
        median_angle = np.median(angles)
        normalized_angle = round(median_angle / 90) * 90
        return normalized_angle % 360
        
    except Exception as e:
        logger.error(f"Rotation detection failed: {e}")
        return 0.0


def correct_rotation(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate image by specified angle in degrees."""
    try:
        if abs(angle) < 0.1:
            return image
        
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, rotation_matrix, (width, height))
        return rotated
        
    except Exception as e:
        logger.error(f"Image rotation failed: {e}")
        return image


def get_image_info(image: np.ndarray) -> dict:
    """Get basic information about an image (width, height, channels, dtype, size)."""
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
    """Utility class for basic image operations (backwards compatibility wrapper)."""
    
    @staticmethod
    def load_image(image_path: Union[str, Path]) -> Optional[np.ndarray]:
        """Load image from file path."""
        return load_image(image_path)
    
    @staticmethod
    def detect_text_orientation(image: np.ndarray) -> float:
        """Detect text orientation."""
        return detect_rotation(image)
    
    @staticmethod
    def correct_image_rotation(image: np.ndarray, angle: float) -> np.ndarray:
        """Correct image rotation."""
        return correct_rotation(image, angle)


__all__ = [
    'load_image',
    'save_image', 
    'validate_image',
    'convert_color_space',
    'resize_image',
    'detect_rotation',
    'correct_rotation',
    'get_image_info',
    'ImageUtils',
]