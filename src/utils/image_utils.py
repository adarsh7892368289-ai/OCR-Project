# src/utils/image_utils.py

import cv2
import numpy as np
from typing import Tuple, Optional, List
from PIL import Image
import io

class ImageUtils:
    """Utility functions for image processing"""
    
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
    def resize_image(image: np.ndarray, max_width: int = 2048, max_height: int = 2048) -> np.ndarray:
        """Resize image while maintaining aspect ratio"""
        height, width = image.shape[:2]
        
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
    def calculate_image_stats(image: np.ndarray) -> dict:
        """Calculate image statistics"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        return {
            "width": image.shape[1],
            "height": image.shape[0],
            "channels": len(image.shape),
            "mean_brightness": np.mean(gray),
            "std_brightness": np.std(gray),
            "min_brightness": np.min(gray),
            "max_brightness": np.max(gray),
            "total_pixels": image.size
        }
    
    @staticmethod
    def save_image(image: np.ndarray, output_path: str, quality: int = 95):
        """Save image to file"""
        success = cv2.imwrite(output_path, image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not success:
            raise ValueError(f"Failed to save image to {output_path}")
