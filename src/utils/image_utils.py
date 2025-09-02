import cv2
import numpy as np
from PIL import Image
from typing import Dict, Any, List, Optional
from pathlib import Path

def get_image_quality(image_path: str) -> Dict[str, Any]:
    """Analyze image quality and characteristics."""
    try:
        image = Image.open(image_path).convert('RGB')
        np_image = np.array(image)
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)
        
        # Calculate image metrics
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        is_color = np_image.shape[2] == 3 and not (np_image[:,:,0] == np_image[:,:,1]).all() and not (np_image[:,:,0] == np_image[:,:,2]).all()
        
        # Simple heuristic for handwritten vs printed
        # This is a basic check and can be improved with ML models
        is_handwritten = sharpness < 100 # A very low sharpness might indicate handwriting
        
        return {
            'image_loaded': True,
            'sharpness': sharpness,
            'is_color': is_color,
            'is_handwritten': is_handwritten,
            'file_size_kb': Path(image_path).stat().st_size / 1024
        }
    except Exception as e:
        return {'image_loaded': False, 'error': str(e)}

def enhance_image_quality(image_path: str) -> np.ndarray:
    """
    Applies a series of image enhancement steps to a given image.
    
    Args:
        image_path (str): The path to the image file.
        
    Returns:
        np.ndarray: The enhanced image as a NumPy array.
    """
    try:
        image = Image.open(image_path).convert('RGB')
        np_image = np.array(image)
        
        # Convert to grayscale
        gray = cv2.cvtColor(np_image, cv2.COLOR_RGB2GRAY)
        
        # Denoise the image
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        
        # Apply adaptive thresholding to get a clean black and white image
        enhanced = cv2.adaptiveThreshold(denoised, 255, 
                                         cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, 11, 2)
        
        return enhanced
    except Exception as e:
        print(f"Error enhancing image: {e}")
        # Return the original image if enhancement fails
        return np.array(Image.open(image_path).convert('RGB'))

def validate_image_file(image_path: str) -> bool:
    """
    Validates if the given file path is a valid image file.
    
    Args:
        image_path (str): The path to the image file.
        
    Returns:
        bool: True if the file is a valid image, False otherwise.
    """
    valid_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}
    
    if not Path(image_path).exists():
        print(f"Error: Image file '{image_path}' not found.")
        return False
        
    file_ext = Path(image_path).suffix.lower()
    if file_ext not in valid_extensions:
        print(f"Error: Unsupported file format '{file_ext}'. Supported formats: {', '.join(valid_extensions)}")
        return False
        
    return True

def get_image_info(image_path: str) -> Optional[Dict[str, Any]]:
    """
    Provides a comprehensive analysis of an image file.

    This function first validates the file and then returns quality metrics.
    
    Args:
        image_path (str): The path to the image file.
    
    Returns:
        Optional[Dict[str, Any]]: A dictionary with image info, or None if validation fails.
    """
    if not validate_image_file(image_path):
        return None
    
    info = get_image_quality(image_path)
    return info
