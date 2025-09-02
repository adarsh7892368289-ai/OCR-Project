"""
Utility functions for OCR processing
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance
import os

def validate_image(image_path: str) -> bool:
    """Validate if the file is a valid image"""
    if not os.path.exists(image_path):
        return False
    
    try:
        with Image.open(image_path) as img:
            img.verify()
        return True
    except:
        return False

def resize_image(image: np.ndarray, max_width: int = 2048, max_height: int = 2048) -> np.ndarray:
    """Resize image if it's too large"""
    height, width = image.shape[:2]
    
    if width <= max_width and height <= max_height:
        return image
    
    # Calculate scaling factor
    scale = min(max_width / width, max_height / height)
    
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

def enhance_image_quality(image: np.ndarray) -> np.ndarray:
    """Enhance image quality for better OCR"""
    # Convert to PIL for enhancement
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(pil_image)
    pil_image = enhancer.enhance(1.2)
    
    # Enhance sharpness
    enhancer = ImageEnhance.Sharpness(pil_image)
    pil_image = enhancer.enhance(1.1)
    
    # Convert back to OpenCV format
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def save_processed_image(image: np.ndarray, output_path: str):
    """Save processed image for debugging"""
    cv2.imwrite(output_path, image)
    print(f"Processed image saved to: {output_path}")
