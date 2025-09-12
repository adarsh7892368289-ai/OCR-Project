"""
Advanced OCR System - Image Processing Utilities
================================================

Production-grade image utilities for OCR preprocessing with OpenCV, Pillow,
and NumPy integration. Handles loading, conversion, enhancement, and analysis.

Author: Production OCR Team
Version: 2.0.0
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ExifTags
import io
import base64
from pathlib import Path
from typing import Union, Tuple, Optional, List, Dict, Any
import logging
from dataclasses import dataclass
import math

# Import logger from utils
try:
    from .logger import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


@dataclass
class ImageMetadata:
    """Container for image metadata and properties."""
    
    # Basic properties
    width: int
    height: int
    channels: int
    dtype: str
    
    # File properties
    format: Optional[str] = None
    file_size: Optional[int] = None
    
    # EXIF data
    dpi: Optional[Tuple[int, int]] = None
    orientation: int = 1
    color_space: Optional[str] = None
    
    # Computed properties
    aspect_ratio: float = 0.0
    total_pixels: int = 0
    
    def __post_init__(self):
        """Calculate derived properties."""
        self.aspect_ratio = self.width / self.height if self.height > 0 else 0.0
        self.total_pixels = self.width * self.height
    
    @property
    def is_grayscale(self) -> bool:
        """Check if image is grayscale."""
        return self.channels == 1
    
    @property
    def is_large_image(self) -> bool:
        """Check if image is considered large (>2MP)."""
        return self.total_pixels > 2_000_000
    
    @property
    def estimated_memory_mb(self) -> float:
        """Estimate memory usage in MB."""
        bytes_per_pixel = 4 if self.channels == 4 else self.channels
        return (self.total_pixels * bytes_per_pixel) / (1024 * 1024)


class ImageLoadError(Exception):
    """Exception raised when image loading fails."""
    pass


class ImageProcessingError(Exception):
    """Exception raised during image processing operations."""
    pass


def load_image(image_source: Union[str, Path, bytes, np.ndarray, Image.Image], 
               target_format: str = 'rgb') -> Tuple[np.ndarray, ImageMetadata]:
    """
    Universal image loader supporting multiple input formats.
    
    Args:
        image_source: Image from file path, bytes, numpy array, or PIL Image
        target_format: Output format ('rgb', 'bgr', 'gray')
    
    Returns:
        Tuple of (image_array, metadata)
    
    Raises:
        ImageLoadError: If image loading fails
    """
    try:
        image = None
        metadata = None
        
        # Handle different input types
        if isinstance(image_source, (str, Path)):
            # Load from file path
            image, metadata = _load_from_path(image_source)
            
        elif isinstance(image_source, bytes):
            # Load from bytes
            image, metadata = _load_from_bytes(image_source)
            
        elif isinstance(image_source, np.ndarray):
            # Already a numpy array
            image = image_source.copy()
            metadata = _extract_metadata_from_array(image)
            
        elif isinstance(image_source, Image.Image):
            # PIL Image
            image = np.array(image_source)
            metadata = _extract_metadata_from_pil(image_source)
            
        else:
            raise ImageLoadError(f"Unsupported image source type: {type(image_source)}")
        
        # Convert to target format
        if target_format.lower() != 'original':
            image = convert_color_format(image, target_format)
        
        logger.debug(f"Image loaded successfully: {metadata.width}x{metadata.height}, "
                    f"format: {target_format}, size: {metadata.estimated_memory_mb:.1f}MB")
        
        return image, metadata
    
    except Exception as e:
        raise ImageLoadError(f"Failed to load image: {str(e)}") from e


def _load_from_path(file_path: Union[str, Path]) -> Tuple[np.ndarray, ImageMetadata]:
    """Load image from file path."""
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise ImageLoadError(f"Image file not found: {file_path}")
    
    try:
        # Try OpenCV first (handles most formats efficiently)
        image = cv2.imread(str(file_path), cv2.IMREAD_COLOR)
        
        if image is not None:
            # OpenCV loads as BGR, convert to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            metadata = _extract_metadata_from_file(file_path, image)
            return image, metadata
        
        # Fallback to PIL for other formats
        with Image.open(file_path) as pil_image:
            # Handle EXIF orientation
            pil_image = _apply_exif_orientation(pil_image)
            image = np.array(pil_image.convert('RGB'))
            metadata = _extract_metadata_from_pil(pil_image, file_path)
            return image, metadata
    
    except Exception as e:
        raise ImageLoadError(f"Failed to load image from {file_path}: {str(e)}")


def _load_from_bytes(image_bytes: bytes) -> Tuple[np.ndarray, ImageMetadata]:
    """Load image from bytes."""
    try:
        # Try OpenCV first
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            metadata = _extract_metadata_from_array(image)
            metadata.file_size = len(image_bytes)
            return image, metadata
        
        # Fallback to PIL
        with Image.open(io.BytesIO(image_bytes)) as pil_image:
            pil_image = _apply_exif_orientation(pil_image)
            image = np.array(pil_image.convert('RGB'))
            metadata = _extract_metadata_from_pil(pil_image)
            metadata.file_size = len(image_bytes)
            return image, metadata
    
    except Exception as e:
        raise ImageLoadError(f"Failed to load image from bytes: {str(e)}")


def _extract_metadata_from_file(file_path: Path, image: np.ndarray) -> ImageMetadata:
    """Extract metadata from file and image array."""
    height, width = image.shape[:2]
    channels = 1 if len(image.shape) == 2 else image.shape[2]
    
    metadata = ImageMetadata(
        width=width,
        height=height,
        channels=channels,
        dtype=str(image.dtype),
        format=file_path.suffix.upper().lstrip('.'),
        file_size=file_path.stat().st_size if file_path.exists() else None
    )
    
    # Try to get DPI from EXIF
    try:
        with Image.open(file_path) as pil_image:
            dpi = pil_image.info.get('dpi')
            if dpi:
                metadata.dpi = dpi
    except:
        pass
    
    return metadata


def _extract_metadata_from_pil(pil_image: Image.Image, file_path: Optional[Path] = None) -> ImageMetadata:
    """Extract metadata from PIL image."""
    width, height = pil_image.size
    channels = len(pil_image.getbands())
    
    metadata = ImageMetadata(
        width=width,
        height=height,
        channels=channels,
        dtype='uint8',  # PIL default
        format=pil_image.format,
        dpi=pil_image.info.get('dpi'),
        color_space=pil_image.mode
    )
    
    if file_path and file_path.exists():
        metadata.file_size = file_path.stat().st_size
    
    return metadata


def _extract_metadata_from_array(image: np.ndarray) -> ImageMetadata:
    """Extract metadata from numpy array."""
    height, width = image.shape[:2]
    channels = 1 if len(image.shape) == 2 else image.shape[2]
    
    return ImageMetadata(
        width=width,
        height=height,
        channels=channels,
        dtype=str(image.dtype)
    )


def _apply_exif_orientation(pil_image: Image.Image) -> Image.Image:
    """Apply EXIF orientation correction to PIL image."""
    try:
        exif = pil_image._getexif()
        if exif is not None:
            orientation_key = next((k for k, v in ExifTags.TAGS.items() if v == 'Orientation'), None)
            if orientation_key and orientation_key in exif:
                orientation = exif[orientation_key]
                
                # Apply rotation based on EXIF orientation
                if orientation == 2:
                    pil_image = pil_image.transpose(Image.FLIP_LEFT_RIGHT)
                elif orientation == 3:
                    pil_image = pil_image.rotate(180, expand=True)
                elif orientation == 4:
                    pil_image = pil_image.rotate(180, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
                elif orientation == 5:
                    pil_image = pil_image.rotate(-90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
                elif orientation == 6:
                    pil_image = pil_image.rotate(-90, expand=True)
                elif orientation == 7:
                    pil_image = pil_image.rotate(90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
                elif orientation == 8:
                    pil_image = pil_image.rotate(90, expand=True)
    except:
        # If EXIF processing fails, continue without rotation
        pass
    
    return pil_image


def convert_color_format(image: np.ndarray, target_format: str) -> np.ndarray:
    """
    Convert image between color formats.
    
    Args:
        image: Input image array
        target_format: Target format ('rgb', 'bgr', 'gray', 'hsv', 'lab')
    
    Returns:
        Converted image array
    
    Raises:
        ImageProcessingError: If conversion fails
    """
    try:
        current_channels = 1 if len(image.shape) == 2 else image.shape[2]
        target_format = target_format.lower()
        
        # Handle grayscale input
        if current_channels == 1:
            if target_format == 'gray':
                return image
            elif target_format in ['rgb', 'bgr']:
                return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB if target_format == 'rgb' else cv2.COLOR_GRAY2BGR)
            elif target_format == 'hsv':
                rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
            elif target_format == 'lab':
                rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2LAB)
        
        # Handle color input (assume RGB)
        elif current_channels == 3:
            if target_format == 'rgb':
                return image
            elif target_format == 'bgr':
                return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            elif target_format == 'gray':
                return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            elif target_format == 'hsv':
                return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif target_format == 'lab':
                return cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Handle RGBA input
        elif current_channels == 4:
            if target_format == 'rgb':
                return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            elif target_format == 'bgr':
                return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
            elif target_format == 'gray':
                rgb_image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
                return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        
        raise ImageProcessingError(f"Unsupported color format conversion: {current_channels} channels to {target_format}")
    
    except cv2.error as e:
        raise ImageProcessingError(f"OpenCV color conversion failed: {str(e)}")
    except Exception as e:
        raise ImageProcessingError(f"Color format conversion failed: {str(e)}")


def resize_image(image: np.ndarray, 
                target_size: Optional[Tuple[int, int]] = None,
                max_dimension: Optional[int] = None,
                min_dimension: Optional[int] = None,
                maintain_aspect_ratio: bool = True,
                interpolation: str = 'lanczos') -> Tuple[np.ndarray, float]:
    """
    Resize image with various sizing strategies.
    
    Args:
        image: Input image array
        target_size: Exact target size (width, height)
        max_dimension: Maximum dimension (longest side)
        min_dimension: Minimum dimension (shortest side)
        maintain_aspect_ratio: Whether to preserve aspect ratio
        interpolation: Interpolation method ('lanczos', 'cubic', 'linear', 'nearest')
    
    Returns:
        Tuple of (resized_image, scale_factor)
    
    Raises:
        ImageProcessingError: If resizing fails
    """
    try:
        height, width = image.shape[:2]
        original_size = (width, height)
        
        # Determine target dimensions
        if target_size:
            target_width, target_height = target_size
            if not maintain_aspect_ratio:
                new_size = target_size
            else:
                # Maintain aspect ratio, fit within target size
                scale = min(target_width / width, target_height / height)
                new_size = (int(width * scale), int(height * scale))
        
        elif max_dimension:
            current_max = max(width, height)
            if current_max <= max_dimension:
                return image.copy(), 1.0
            
            scale = max_dimension / current_max
            new_size = (int(width * scale), int(height * scale))
        
        elif min_dimension:
            current_min = min(width, height)
            if current_min >= min_dimension:
                return image.copy(), 1.0
            
            scale = min_dimension / current_min
            new_size = (int(width * scale), int(height * scale))
        
        else:
            raise ImageProcessingError("Must specify target_size, max_dimension, or min_dimension")
        
        # Ensure minimum size
        new_width, new_height = new_size
        new_width = max(1, new_width)
        new_height = max(1, new_height)
        
        # Select interpolation method
        interpolation_map = {
            'lanczos': cv2.INTER_LANCZOS4,
            'cubic': cv2.INTER_CUBIC,
            'linear': cv2.INTER_LINEAR,
            'nearest': cv2.INTER_NEAREST,
            'area': cv2.INTER_AREA
        }
        
        interp_method = interpolation_map.get(interpolation.lower(), cv2.INTER_LANCZOS4)
        
        # Perform resize
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=interp_method)
        
        # Calculate scale factor
        scale_factor = new_width / width
        
        logger.debug(f"Image resized from {original_size} to ({new_width}, {new_height}), "
                    f"scale: {scale_factor:.3f}, method: {interpolation}")
        
        return resized_image, scale_factor
    
    except Exception as e:
        raise ImageProcessingError(f"Image resizing failed: {str(e)}")


def rotate_image(image: np.ndarray, 
                angle: float, 
                background_color: Tuple[int, int, int] = (255, 255, 255),
                crop_to_original: bool = False) -> np.ndarray:
    """
    Rotate image by specified angle.
    
    Args:
        image: Input image array
        angle: Rotation angle in degrees (positive = clockwise)
        background_color: Fill color for areas outside original image
        crop_to_original: Whether to crop result to original size
    
    Returns:
        Rotated image array
    
    Raises:
        ImageProcessingError: If rotation fails
    """
    try:
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        
        # Calculate rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, -angle, 1.0)
        
        if crop_to_original:
            # Rotate and crop to original size
            rotated = cv2.warpAffine(image, rotation_matrix, (width, height), 
                                   borderValue=background_color)
        else:
            # Calculate new image dimensions to fit entire rotated image
            cos_angle = abs(rotation_matrix[0, 0])
            sin_angle = abs(rotation_matrix[0, 1])
            
            new_width = int((height * sin_angle) + (width * cos_angle))
            new_height = int((height * cos_angle) + (width * sin_angle))
            
            # Adjust rotation matrix for new center
            rotation_matrix[0, 2] += (new_width / 2) - center[0]
            rotation_matrix[1, 2] += (new_height / 2) - center[1]
            
            rotated = cv2.warpAffine(image, rotation_matrix, (new_width, new_height),
                                   borderValue=background_color)
        
        logger.debug(f"Image rotated by {angle} degrees")
        return rotated
    
    except Exception as e:
        raise ImageProcessingError(f"Image rotation failed: {str(e)}")


def detect_skew_angle(image: np.ndarray, accuracy: str = 'medium') -> float:
    """
    Detect document skew angle using Hough line detection.
    
    Args:
        image: Input image (preferably grayscale)
        accuracy: Detection accuracy ('low', 'medium', 'high')
    
    Returns:
        Skew angle in degrees (-45 to 45)
    
    Raises:
        ImageProcessingError: If skew detection fails
    """
    try:
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Set parameters based on accuracy
        if accuracy == 'low':
            rho, theta, threshold = 1, np.pi/180, max(100, min(gray.shape)//10)
        elif accuracy == 'medium':
            rho, theta, threshold = 1, np.pi/360, max(150, min(gray.shape)//8)
        else:  # high
            rho, theta, threshold = 1, np.pi/720, max(200, min(gray.shape)//6)
        
        # Detect lines using Hough transform
        lines = cv2.HoughLines(edges, rho, theta, threshold)
        
        if lines is None or len(lines) == 0:
            logger.debug("No lines detected for skew analysis")
            return 0.0
        
        # Calculate angles of detected lines
        angles = []
        for line in lines:
            rho, theta = line[0]
            angle = theta * 180 / np.pi
            
            # Convert to angle range -45 to 45
            if angle > 90:
                angle = angle - 180
            elif angle > 45:
                angle = angle - 90
            elif angle < -45:
                angle = angle + 90
            
            # Filter out near-vertical and near-horizontal lines
            if abs(angle) > 2 and abs(angle) < 88:
                angles.append(angle)
        
        if not angles:
            logger.debug("No valid angles found for skew detection")
            return 0.0
        
        # Calculate median angle (more robust than mean)
        skew_angle = float(np.median(angles))
        
        logger.debug(f"Detected skew angle: {skew_angle:.2f} degrees")
        return skew_angle
    
    except Exception as e:
        logger.warning(f"Skew detection failed: {str(e)}")
        return 0.0


def deskew_image(image: np.ndarray, 
                angle: Optional[float] = None,
                background_color: Tuple[int, int, int] = (255, 255, 255)) -> Tuple[np.ndarray, float]:
    """
    Automatically detect and correct document skew.
    
    Args:
        image: Input image array
        angle: Specific angle to correct (if None, auto-detect)
        background_color: Fill color for rotation
    
    Returns:
        Tuple of (deskewed_image, correction_angle)
    
    Raises:
        ImageProcessingError: If deskewing fails
    """
    try:
        if angle is None:
            angle = detect_skew_angle(image, accuracy='medium')
        
        # Only correct if angle is significant
        if abs(angle) < 0.5:
            logger.debug("Skew angle too small, no correction needed")
            return image.copy(), 0.0
        
        # Rotate to correct skew
        deskewed = rotate_image(image, angle, background_color, crop_to_original=True)
        
        logger.debug(f"Image deskewed by {angle:.2f} degrees")
        return deskewed, angle
    
    except Exception as e:
        raise ImageProcessingError(f"Deskewing failed: {str(e)}")


def crop_image(image: np.ndarray, 
              bbox: Tuple[int, int, int, int],
              padding: int = 0) -> np.ndarray:
    """
    Crop image to specified bounding box with optional padding.
    
    Args:
        image: Input image array
        bbox: Bounding box (x, y, width, height)
        padding: Padding around crop area
    
    Returns:
        Cropped image array
    
    Raises:
        ImageProcessingError: If cropping fails
    """
    try:
        height, width = image.shape[:2]
        x, y, w, h = bbox
        
        # Add padding and ensure bounds
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(width, x + w + padding)
        y2 = min(height, y + h + padding)
        
        if x1 >= x2 or y1 >= y2:
            raise ImageProcessingError("Invalid crop coordinates")
        
        cropped = image[y1:y2, x1:x2]
        
        logger.debug(f"Image cropped to {cropped.shape[:2]} from bbox {bbox}")
        return cropped
    
    except Exception as e:
        raise ImageProcessingError(f"Image cropping failed: {str(e)}")


def enhance_contrast(image: np.ndarray, 
                    method: str = 'clahe',
                    strength: float = 1.0) -> np.ndarray:
    """
    Enhance image contrast using various methods.
    
    Args:
        image: Input image array
        method: Enhancement method ('clahe', 'histogram_eq', 'gamma', 'adaptive')
        strength: Enhancement strength (0.5 to 2.0)
    
    Returns:
        Enhanced image array
    
    Raises:
        ImageProcessingError: If enhancement fails
    """
    try:
        strength = max(0.5, min(2.0, strength))
        
        # Convert to grayscale for processing if needed
        is_color = len(image.shape) == 3 and image.shape[2] > 1
        
        if method == 'clahe':
            # Contrast Limited Adaptive Histogram Equalization
            clip_limit = 2.0 * strength
            tile_size = (8, 8)
            
            if is_color:
                # Apply CLAHE to L channel in LAB color space
                lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
                lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            else:
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
                enhanced = clahe.apply(image)
        
        elif method == 'histogram_eq':
            # Global histogram equalization
            if is_color:
                lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
                lab[:, :, 0] = cv2.equalizeHist(lab[:, :, 0])
                enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            else:
                enhanced = cv2.equalizeHist(image)
        
        elif method == 'gamma':
            # Gamma correction
            gamma = 1.0 / strength
            enhanced = np.power(image / 255.0, gamma) * 255.0
            enhanced = enhanced.astype(np.uint8)
        
        elif method == 'adaptive':
            # Adaptive contrast enhancement
            if is_color:
                enhanced = image.copy()
                for i in range(3):
                    channel = enhanced[:, :, i]
                    # Apply adaptive histogram equalization per channel
                    clahe = cv2.createCLAHE(clipLimit=2.0 * strength, tileGridSize=(8, 8))
                    enhanced[:, :, i] = clahe.apply(channel)
            else:
                clahe = cv2.createCLAHE(clipLimit=2.0 * strength, tileGridSize=(8, 8))
                enhanced = clahe.apply(image)
        
        else:
            raise ImageProcessingError(f"Unknown contrast enhancement method: {method}")
        
        logger.debug(f"Contrast enhanced using {method} method, strength: {strength}")
        return enhanced
    
    except Exception as e:
        raise ImageProcessingError(f"Contrast enhancement failed: {str(e)}")


def save_image(image: np.ndarray, 
              output_path: Union[str, Path],
              quality: int = 95,
              format: Optional[str] = None) -> Path:
    """
    Save image to file with specified quality and format.
    
    Args:
        image: Image array to save
        output_path: Output file path
        quality: JPEG quality (1-100)
        format: Output format (auto-detected from extension if None)
    
    Returns:
        Path object of saved file
    
    Raises:
        ImageProcessingError: If saving fails
    """
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Determine format
        if format is None:
            format = output_path.suffix.lower().lstrip('.')
        
        format = format.lower()
        
        # Ensure image is in correct format for saving
        if len(image.shape) == 3:
            # Convert RGB to BGR for OpenCV
            save_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            save_image = image
        
        # Set compression parameters
        if format in ['jpg', 'jpeg']:
            params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        elif format == 'png':
            compression = max(0, min(9, int((100 - quality) / 10)))
            params = [cv2.IMWRITE_PNG_COMPRESSION, compression]
        elif format in ['tif', 'tiff']:
            params = [cv2.IMWRITE_TIFF_COMPRESSION, 1]  # LZW compression
        else:
            params = []
        
        # Save image
        success = cv2.imwrite(str(output_path), save_image, params)
        
        if not success:
            raise ImageProcessingError(f"Failed to save image to {output_path}")
        
        logger.debug(f"Image saved to {output_path}, format: {format}, quality: {quality}")
        return output_path
    
    except Exception as e:
        raise ImageProcessingError(f"Image saving failed: {str(e)}")


def image_to_base64(image: np.ndarray, format: str = 'png') -> str:
    """
    Convert image array to base64 string.
    
    Args:
        image: Image array
        format: Output format ('png', 'jpg')
    
    Returns:
        Base64 encoded string
    
    Raises:
        ImageProcessingError: If conversion fails
    """
    try:
        # Convert to PIL Image
        if len(image.shape) == 3:
            pil_image = Image.fromarray(image, 'RGB')
        else:
            pil_image = Image.fromarray(image, 'L')
        
        # Convert to bytes
        buffer = io.BytesIO()
        pil_image.save(buffer, format=format.upper())
        image_bytes = buffer.getvalue()
        
        # Encode to base64
        base64_string = base64.b64encode(image_bytes).decode('utf-8')
        
        return base64_string
    
    except Exception as e:
        raise ImageProcessingError(f"Base64 conversion failed: {str(e)}")


def base64_to_image(base64_string: str) -> np.ndarray:
    """
    Convert base64 string to image array.
    
    Args:
        base64_string: Base64 encoded image string
    
    Returns:
        Image array
    
    Raises:
        ImageLoadError: If conversion fails
    """
    try:
        # Decode base64 to bytes
        image_bytes = base64.b64decode(base64_string)
        
        # Load image from bytes
        image, _ = _load_from_bytes(image_bytes)
        
        return image
    
    except Exception as e:
        raise ImageLoadError(f"Base64 to image conversion failed: {str(e)}")


def calculate_image_quality_metrics(image: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive image quality metrics for OCR preprocessing decisions.
    
    Args:
        image: Input image array
    
    Returns:
        Dictionary of quality metrics (0.0 to 1.0 scale)
    
    Raises:
        ImageProcessingError: If analysis fails
    """
    try:
        # Convert to grayscale for analysis
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        metrics = {}
        
        # 1. Sharpness (Laplacian variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        metrics['sharpness'] = min(1.0, laplacian_var / 1000.0)  # Normalize
        
        # 2. Contrast (RMS contrast)
        contrast = gray.std()
        metrics['contrast'] = min(1.0, contrast / 128.0)  # Normalize to 0-1
        
        # 3. Brightness (mean intensity)
        brightness = gray.mean() / 255.0
        # Optimal brightness is around 0.4-0.6, penalize extremes
        if brightness < 0.2 or brightness > 0.8:
            metrics['brightness'] = 0.3
        elif 0.4 <= brightness <= 0.6:
            metrics['brightness'] = 1.0
        else:
            metrics['brightness'] = 0.7
        
        # 4. Noise level (using standard deviation of smoothed vs original)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        noise_level = np.std(gray.astype(float) - blurred.astype(float))
        metrics['noise'] = max(0.0, 1.0 - (noise_level / 50.0))  # Lower noise = higher score
        
        # 5. Dynamic range
        dynamic_range = (gray.max() - gray.min()) / 255.0
        metrics['dynamic_range'] = dynamic_range
        
        # 6. Edge density (indicator of text content)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        metrics['edge_density'] = min(1.0, edge_density * 10.0)  # Normalize
        
        # 7. Overall quality score (weighted combination)
        weights = {
            'sharpness': 0.25,
            'contrast': 0.25, 
            'brightness': 0.15,
            'noise': 0.15,
            'dynamic_range': 0.10,
            'edge_density': 0.10
        }
        
        overall_score = sum(metrics[key] * weights[key] for key in weights)
        metrics['overall'] = overall_score
        
        logger.debug(f"Image quality metrics calculated: overall={overall_score:.3f}")
        return metrics
    
    except Exception as e:
        raise ImageProcessingError(f"Quality metrics calculation failed: {str(e)}")


def detect_text_regions_simple(image: np.ndarray, 
                              min_area: int = 100,
                              aspect_ratio_range: Tuple[float, float] = (0.1, 10.0)) -> List[Tuple[int, int, int, int]]:
    """
    Simple text region detection using morphological operations.
    
    Args:
        image: Input image array
        min_area: Minimum area for text regions
        aspect_ratio_range: Valid aspect ratio range (min, max)
    
    Returns:
        List of bounding boxes (x, y, width, height)
    
    Raises:
        ImageProcessingError: If detection fails
    """
    try:
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Apply morphological operations to connect text components
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        
        # Binary threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Morphological operations
        morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours based on area and aspect ratio
        text_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            aspect_ratio = w / h if h > 0 else 0
            
            if (area >= min_area and 
                aspect_ratio_range[0] <= aspect_ratio <= aspect_ratio_range[1]):
                text_regions.append((x, y, w, h))
        
        logger.debug(f"Detected {len(text_regions)} potential text regions")
        return text_regions
    
    except Exception as e:
        raise ImageProcessingError(f"Text region detection failed: {str(e)}")


def normalize_image_for_ocr(image: np.ndarray) -> np.ndarray:
    """
    Normalize image for optimal OCR performance.
    
    Args:
        image: Input image array
    
    Returns:
        Normalized image optimized for OCR
    
    Raises:
        ImageProcessingError: If normalization fails
    """
    try:
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Apply Gaussian blur to reduce noise
        denoised = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Apply adaptive thresholding for better text separation
        normalized = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Apply morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        normalized = cv2.morphologyEx(normalized, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        logger.debug("Image normalized for OCR processing")
        return normalized
    
    except Exception as e:
        raise ImageProcessingError(f"OCR normalization failed: {str(e)}")


def create_image_thumbnail(image: np.ndarray, 
                         max_size: int = 256,
                         quality: int = 85) -> bytes:
    """
    Create compressed thumbnail for image preview.
    
    Args:
        image: Input image array
        max_size: Maximum dimension for thumbnail
        quality: JPEG compression quality
    
    Returns:
        Thumbnail as JPEG bytes
    
    Raises:
        ImageProcessingError: If thumbnail creation fails
    """
    try:
        # Resize image maintaining aspect ratio
        thumbnail, _ = resize_image(image, max_dimension=max_size)
        
        # Convert to PIL for JPEG compression
        if len(thumbnail.shape) == 3:
            pil_image = Image.fromarray(thumbnail, 'RGB')
        else:
            pil_image = Image.fromarray(thumbnail, 'L')
        
        # Save to bytes with JPEG compression
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG', quality=quality, optimize=True)
        thumbnail_bytes = buffer.getvalue()
        
        logger.debug(f"Thumbnail created: {len(thumbnail_bytes)} bytes")
        return thumbnail_bytes
    
    except Exception as e:
        raise ImageProcessingError(f"Thumbnail creation failed: {str(e)}")


def batch_process_images(image_paths: List[Union[str, Path]], 
                        processing_function,
                        output_dir: Optional[Union[str, Path]] = None,
                        max_workers: int = 4,
                        **kwargs) -> List[Dict[str, Any]]:
    """
    Process multiple images in parallel with progress tracking.
    
    Args:
        image_paths: List of image file paths
        processing_function: Function to apply to each image
        output_dir: Directory for processed images (optional)
        max_workers: Number of parallel workers
        **kwargs: Arguments to pass to processing function
    
    Returns:
        List of processing results with metadata
    
    Raises:
        ImageProcessingError: If batch processing fails
    """
    try:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import time
        
        results = []
        
        def process_single_image(image_path):
            start_time = time.time()
            try:
                # Load image
                image, metadata = load_image(image_path)
                
                # Apply processing function
                processed_result = processing_function(image, **kwargs)
                
                # Save if output directory specified
                if output_dir:
                    output_path = Path(output_dir) / Path(image_path).name
                    if isinstance(processed_result, np.ndarray):
                        save_image(processed_result, output_path)
                
                processing_time = time.time() - start_time
                
                return {
                    'path': str(image_path),
                    'status': 'success',
                    'processing_time': processing_time,
                    'metadata': metadata,
                    'result': processed_result
                }
                
            except Exception as e:
                processing_time = time.time() - start_time
                return {
                    'path': str(image_path),
                    'status': 'error',
                    'processing_time': processing_time,
                    'error': str(e)
                }
        
        # Process images in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_path = {
                executor.submit(process_single_image, path): path 
                for path in image_paths
            }
            
            for future in as_completed(future_to_path):
                result = future.result()
                results.append(result)
                
                # Log progress
                completed = len(results)
                total = len(image_paths)
                if result['status'] == 'success':
                    logger.info(f"Processed {completed}/{total}: {result['path']}")
                else:
                    logger.error(f"Failed {completed}/{total}: {result['path']} - {result['error']}")
        
        successful = len([r for r in results if r['status'] == 'success'])
        logger.info(f"Batch processing completed: {successful}/{len(image_paths)} successful")
        
        return results
    
    except Exception as e:
        raise ImageProcessingError(f"Batch processing failed: {str(e)}")


def validate_image_for_ocr(image: np.ndarray, metadata: ImageMetadata) -> Tuple[bool, List[str]]:
    """
    Validate if image is suitable for OCR processing.
    
    Args:
        image: Image array to validate
        metadata: Image metadata
    
    Returns:
        Tuple of (is_valid, list_of_issues)
    """
    issues = []
    
    # Check minimum dimensions
    if metadata.width < 64 or metadata.height < 64:
        issues.append(f"Image too small: {metadata.width}x{metadata.height} (minimum 64x64)")
    
    # Check maximum dimensions
    if metadata.total_pixels > 50_000_000:  # ~50MP
        issues.append(f"Image too large: {metadata.total_pixels} pixels (maximum 50MP)")
    
    # Check aspect ratio
    if metadata.aspect_ratio > 20 or metadata.aspect_ratio < 0.05:
        issues.append(f"Extreme aspect ratio: {metadata.aspect_ratio:.2f}")
    
    # Check if image is completely black or white
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    mean_intensity = gray.mean()
    if mean_intensity < 10:
        issues.append("Image appears to be mostly black")
    elif mean_intensity > 245:
        issues.append("Image appears to be mostly white")
    
    # Check dynamic range
    intensity_range = gray.max() - gray.min()
    if intensity_range < 50:
        issues.append(f"Low dynamic range: {intensity_range} (may need contrast enhancement)")
    
    is_valid = len(issues) == 0
    
    if not is_valid:
        logger.warning(f"Image validation issues: {', '.join(issues)}")
    
    return is_valid, issues


# Convenience functions and constants
SUPPORTED_IMAGE_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']

def is_supported_format(file_path: Union[str, Path]) -> bool:
    """Check if file format is supported for image processing."""
    return Path(file_path).suffix.lower() in SUPPORTED_IMAGE_FORMATS


def get_optimal_resize_dimensions(current_width: int, 
                                current_height: int,
                                target_dpi: int = 300,
                                max_dimension: int = 2048) -> Tuple[int, int]:
    """
    Calculate optimal resize dimensions for OCR processing.
    
    Args:
        current_width: Current image width
        current_height: Current image height
        target_dpi: Target DPI for OCR
        max_dimension: Maximum allowed dimension
    
    Returns:
        Tuple of (optimal_width, optimal_height)
    """
    # Calculate scale factor based on target DPI
    # Assume current image is 150 DPI if unknown
    current_dpi = 150
    scale_factor = target_dpi / current_dpi
    
    new_width = int(current_width * scale_factor)
    new_height = int(current_height * scale_factor)
    
    # Constrain to maximum dimension
    max_current = max(new_width, new_height)
    if max_current > max_dimension:
        constraint_scale = max_dimension / max_current
        new_width = int(new_width * constraint_scale)
        new_height = int(new_height * constraint_scale)
    
    return new_width, new_height


# Export all public functions and classes
__all__ = [
    # Core classes
    'ImageMetadata',
    'ImageLoadError', 
    'ImageProcessingError',
    
    # Loading and conversion
    'load_image',
    'convert_color_format',
    'image_to_base64',
    'base64_to_image',
    
    # Geometric operations
    'resize_image',
    'rotate_image',
    'crop_image',
    'deskew_image',
    'detect_skew_angle',
    
    # Enhancement operations
    'enhance_contrast',
    'normalize_image_for_ocr',
    
    # Analysis and detection
    'calculate_image_quality_metrics',
    'detect_text_regions_simple',
    'validate_image_for_ocr',
    
    # I/O operations
    'save_image',
    'create_image_thumbnail',
    
    # Batch processing
    'batch_process_images',
    
    # Utility functions
    'is_supported_format',
    'get_optimal_resize_dimensions',
    'SUPPORTED_IMAGE_FORMATS'
]