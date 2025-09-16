# src/advanced_ocr/utils/image_utils.py
"""
Advanced OCR Image Utilities

This module provides fundamental image processing operations and utilities for the
advanced OCR system. It handles low-level image manipulations, format conversions,
coordinate transformations, and validation operations that serve as building blocks
for higher-level image processing components.

The module focuses on:
- Efficient image loading and format handling
- Memory-optimized image operations
- Coordinate system transformations for bounding boxes
- Image format validation and conversion
- Basic geometric transformations (resize, rotate, crop)

Classes:
    ImageLoader: Memory-efficient image loading with format detection
    ImageProcessor: Core image transformation operations  
    ImageValidator: Format and quality validation
    CoordinateTransformer: Bounding box coordinate conversions
    ImageMemoryManager: Memory optimization for large images

Functions:
    load_image: Load image from file or array with format detection
    save_image: Save image with format optimization
    resize_image: Intelligent resizing with aspect ratio preservation
    validate_image: Comprehensive image validation
    normalize_coordinates: Convert coordinates between different formats

Example:
    >>> from advanced_ocr.utils.image_utils import load_image, resize_image
    >>> image = load_image("document.jpg")
    >>> resized = resize_image(image, max_size=1024, preserve_aspect=True)
    >>> print(f"Original: {image.shape}, Resized: {resized.shape}")
    
    >>> coords = normalize_coordinates((10, 20, 100, 80), "xywh", "xyxy")
    >>> print(f"Converted coordinates: {coords}")
"""

import cv2
import numpy as np
from PIL import Image, ImageFile
from pathlib import Path
import io
from typing import Union, Tuple, List, Optional, Any, Dict
from enum import Enum
import warnings
from dataclasses import dataclass

# Enable loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageFormat(Enum):
    """Supported image formats with their characteristics."""
    JPEG = "jpeg"
    PNG = "png"
    TIFF = "tiff"
    BMP = "bmp"
    WEBP = "webp"
    PDF = "pdf"


class ColorSpace(Enum):
    """Color space representations for image processing."""
    RGB = "rgb"
    BGR = "bgr"
    GRAY = "gray"
    HSV = "hsv"
    LAB = "lab"


class ResizeMethod(Enum):
    """Image resizing interpolation methods."""
    NEAREST = cv2.INTER_NEAREST
    LINEAR = cv2.INTER_LINEAR
    CUBIC = cv2.INTER_CUBIC
    LANCZOS = cv2.INTER_LANCZOS4
    AREA = cv2.INTER_AREA


@dataclass
class ImageProperties:
    """
    Container for image properties and metadata.
    
    Attributes:
        width (int): Image width in pixels
        height (int): Image height in pixels
        channels (int): Number of color channels
        dtype (str): Image data type (uint8, float32, etc.)
        format (ImageFormat): Detected image format
        color_space (ColorSpace): Current color space
        file_size_mb (float): File size in megabytes
        dpi (Tuple[int, int]): DPI information if available
        has_transparency (bool): Whether image has alpha channel
    """
    width: int
    height: int
    channels: int
    dtype: str
    format: ImageFormat
    color_space: ColorSpace
    file_size_mb: float = 0.0
    dpi: Optional[Tuple[int, int]] = None
    has_transparency: bool = False
    
    @property
    def size(self) -> Tuple[int, int]:
        """Get image dimensions as (width, height) tuple."""
        return (self.width, self.height)
    
    @property
    def shape(self) -> Tuple[int, int, int]:
        """Get image shape as (height, width, channels) tuple."""
        return (self.height, self.width, self.channels)
    
    @property
    def aspect_ratio(self) -> float:
        """Calculate image aspect ratio (width/height)."""
        return self.width / self.height if self.height > 0 else 0.0
    
    @property
    def total_pixels(self) -> int:
        """Calculate total number of pixels."""
        return self.width * self.height


class ImageValidator:
    """
    Image format and quality validation utilities.
    
    Provides comprehensive validation for image inputs including format checking,
    dimension validation, file size limits, and quality assessments.
    """
    
    # Supported image extensions
    SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.webp', '.pdf'}
    
    # Default validation limits
    DEFAULT_MAX_SIZE_MB = 50
    DEFAULT_MAX_DIMENSION = 8192
    DEFAULT_MIN_DIMENSION = 32
    
    @staticmethod
    def validate_file_path(file_path: Union[str, Path]) -> bool:
        """
        Validate image file path and extension.
        
        Args:
            file_path: Path to image file
            
        Returns:
            bool: True if path is valid and supported
            
        Raises:
            ValueError: If file path is invalid or unsupported
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise ValueError(f"Image file does not exist: {file_path}")
        
        if not file_path.is_file():
            raise ValueError(f"Path is not a file: {file_path}")
        
        if file_path.suffix.lower() not in ImageValidator.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported image format: {file_path.suffix}")
        
        return True
    
    @staticmethod
    def validate_image_array(image: np.ndarray) -> bool:
        """
        Validate numpy image array format and properties.
        
        Args:
            image: Numpy array representing image
            
        Returns:
            bool: True if array is valid image
            
        Raises:
            ValueError: If array format is invalid
        """
        if not isinstance(image, np.ndarray):
            raise ValueError("Image must be numpy array")
        
        if image.ndim not in [2, 3]:
            raise ValueError(f"Image must be 2D or 3D array, got {image.ndim}D")
        
        if image.ndim == 3 and image.shape[2] not in [1, 3, 4]:
            raise ValueError(f"3D image must have 1, 3, or 4 channels, got {image.shape[2]}")
        
        if image.size == 0:
            raise ValueError("Image array is empty")
        
        return True
    
    @staticmethod
    def validate_dimensions(
        width: int, 
        height: int, 
        max_dimension: int = DEFAULT_MAX_DIMENSION,
        min_dimension: int = DEFAULT_MIN_DIMENSION
    ) -> bool:
        """
        Validate image dimensions against size limits.
        
        Args:
            width: Image width in pixels
            height: Image height in pixels
            max_dimension: Maximum allowed dimension
            min_dimension: Minimum allowed dimension
            
        Returns:
            bool: True if dimensions are valid
            
        Raises:
            ValueError: If dimensions are outside valid range
        """
        if width < min_dimension or height < min_dimension:
            raise ValueError(f"Image dimensions too small: {width}x{height} < {min_dimension}")
        
        if width > max_dimension or height > max_dimension:
            raise ValueError(f"Image dimensions too large: {width}x{height} > {max_dimension}")
        
        return True
    
    @staticmethod
    def validate_file_size(file_path: Union[str, Path], max_size_mb: float = DEFAULT_MAX_SIZE_MB) -> bool:
        """
        Validate image file size against limit.
        
        Args:
            file_path: Path to image file
            max_size_mb: Maximum file size in megabytes
            
        Returns:
            bool: True if file size is within limit
            
        Raises:
            ValueError: If file size exceeds limit
        """
        file_path = Path(file_path)
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        
        if file_size_mb > max_size_mb:
            raise ValueError(f"Image file too large: {file_size_mb:.1f}MB > {max_size_mb}MB")
        
        return True


class ImageLoader:
    """
    Memory-efficient image loading with comprehensive format support.
    
    Provides optimized image loading from various sources (files, bytes, arrays)
    with automatic format detection, error handling, and memory management.
    """
    
    def __init__(self, max_size_mb: float = 50, auto_convert_grayscale: bool = False):
        """
        Initialize image loader.
        
        Args:
            max_size_mb: Maximum file size to load (MB)
            auto_convert_grayscale: Automatically convert single-channel images to RGB
        """
        self.max_size_mb = max_size_mb
        self.auto_convert_grayscale = auto_convert_grayscale
        self.validator = ImageValidator()
    
    def load_from_file(self, file_path: Union[str, Path], color_mode: str = "rgb") -> np.ndarray:
        """
        Load image from file with format auto-detection.
        
        Args:
            file_path: Path to image file
            color_mode: Color mode ('rgb', 'bgr', 'gray')
            
        Returns:
            np.ndarray: Loaded image array
            
        Raises:
            ValueError: If file is invalid or unsupported
            IOError: If file cannot be loaded
        """
        file_path = Path(file_path)
        
        # Validate file
        self.validator.validate_file_path(file_path)
        self.validator.validate_file_size(file_path, self.max_size_mb)
        
        try:
            # Handle PDF files separately
            if file_path.suffix.lower() == '.pdf':
                return self._load_pdf(file_path, color_mode)
            
            # Load with PIL for better format support
            with Image.open(file_path) as pil_image:
                # Convert to RGB if needed
                if pil_image.mode not in ['RGB', 'RGBA', 'L']:
                    pil_image = pil_image.convert('RGB')
                
                # Convert PIL to numpy
                image_array = np.array(pil_image)
                
                # Handle color mode conversion
                return self._convert_color_mode(image_array, pil_image.mode, color_mode)
                
        except Exception as e:
            # Fallback to OpenCV
            return self._load_with_opencv(file_path, color_mode)
    
    def load_from_bytes(self, image_bytes: bytes, color_mode: str = "rgb") -> np.ndarray:
        """
        Load image from bytes data.
        
        Args:
            image_bytes: Image data as bytes
            color_mode: Color mode ('rgb', 'bgr', 'gray')
            
        Returns:
            np.ndarray: Loaded image array
            
        Raises:
            ValueError: If bytes data is invalid
        """
        if not image_bytes:
            raise ValueError("Empty image bytes data")
        
        try:
            # Try PIL first
            with Image.open(io.BytesIO(image_bytes)) as pil_image:
                if pil_image.mode not in ['RGB', 'RGBA', 'L']:
                    pil_image = pil_image.convert('RGB')
                
                image_array = np.array(pil_image)
                return self._convert_color_mode(image_array, pil_image.mode, color_mode)
                
        except Exception:
            # Fallback to OpenCV
            try:
                image_array = cv2.imdecode(
                    np.frombuffer(image_bytes, np.uint8), 
                    cv2.IMREAD_COLOR if color_mode != 'gray' else cv2.IMREAD_GRAYSCALE
                )
                
                if image_array is None:
                    raise ValueError("Failed to decode image bytes")
                
                if color_mode == 'rgb' and len(image_array.shape) == 3:
                    image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
                
                return image_array
                
            except Exception as e:
                raise ValueError(f"Failed to load image from bytes: {e}")
    
    def load_from_array(self, image_array: np.ndarray, validate: bool = True) -> np.ndarray:
        """
        Load and validate image from numpy array.
        
        Args:
            image_array: Numpy array representing image
            validate: Whether to validate array format
            
        Returns:
            np.ndarray: Validated image array
            
        Raises:
            ValueError: If array format is invalid
        """
        if validate:
            self.validator.validate_image_array(image_array)
        
        # Ensure proper data type
        if image_array.dtype != np.uint8:
            if image_array.max() <= 1.0:
                image_array = (image_array * 255).astype(np.uint8)
            else:
                image_array = np.clip(image_array, 0, 255).astype(np.uint8)
        
        return image_array.copy()
    
    def _load_pdf(self, file_path: Path, color_mode: str) -> np.ndarray:
        """Load first page of PDF as image."""
        try:
            import fitz  # PyMuPDF
            
            with fitz.open(file_path) as pdf:
                if len(pdf) == 0:
                    raise ValueError("PDF file has no pages")
                
                # Get first page
                page = pdf[0]
                
                # Render at high DPI
                mat = fitz.Matrix(2.0, 2.0)  # 2x zoom = ~144 DPI
                pix = page.get_pixmap(matrix=mat)
                
                # Convert to numpy array
                image_data = pix.tobytes("ppm")
                image_array = np.frombuffer(image_data, dtype=np.uint8)
                
                # Reshape to image dimensions
                image_array = image_array.reshape(pix.height, pix.width, 3)
                
                return self._convert_color_mode(image_array, 'RGB', color_mode)
                
        except ImportError:
            raise ValueError("PyMuPDF required for PDF loading: pip install PyMuPDF")
        except Exception as e:
            raise IOError(f"Failed to load PDF: {e}")
    
    def _load_with_opencv(self, file_path: Path, color_mode: str) -> np.ndarray:
        """Fallback image loading with OpenCV."""
        try:
            if color_mode == 'gray':
                image_array = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
            else:
                image_array = cv2.imread(str(file_path), cv2.IMREAD_COLOR)
                
                if color_mode == 'rgb':
                    image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            
            if image_array is None:
                raise IOError(f"OpenCV failed to load image: {file_path}")
            
            return image_array
            
        except Exception as e:
            raise IOError(f"Failed to load image with OpenCV: {e}")
    
    def _convert_color_mode(self, image_array: np.ndarray, source_mode: str, target_mode: str) -> np.ndarray:
        """Convert image between color modes."""
        if target_mode == 'gray':
            if len(image_array.shape) == 3:
                if source_mode == 'RGBA':
                    # Convert RGBA to RGB first
                    rgb_array = image_array[:, :, :3]
                    return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)
                else:
                    return cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            else:
                return image_array
        
        elif target_mode == 'rgb':
            if source_mode == 'RGBA':
                # Remove alpha channel
                return image_array[:, :, :3]
            elif len(image_array.shape) == 2:
                # Convert grayscale to RGB
                return cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
            else:
                return image_array
        
        elif target_mode == 'bgr':
            if source_mode in ['RGB', 'RGBA']:
                rgb_array = image_array[:, :, :3] if source_mode == 'RGBA' else image_array
                return cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
            elif len(image_array.shape) == 2:
                return cv2.cvtColor(image_array, cv2.COLOR_GRAY2BGR)
            else:
                return image_array
        
        return image_array


class ImageProcessor:
    """
    Core image transformation operations for OCR preprocessing.
    
    Provides essential image processing operations including resizing, rotation,
    cropping, and format conversions optimized for OCR workflows.
    """
    
    def __init__(self):
        """Initialize image processor."""
        self.validator = ImageValidator()
    
    def resize_image(
        self,
        image: np.ndarray,
        target_size: Optional[Tuple[int, int]] = None,
        max_size: Optional[int] = None,
        scale_factor: Optional[float] = None,
        preserve_aspect: bool = True,
        method: ResizeMethod = ResizeMethod.LINEAR
    ) -> np.ndarray:
        """
        Resize image with multiple sizing options.
        
        Args:
            image: Input image array
            target_size: Exact target size as (width, height)
            max_size: Maximum dimension size (preserves aspect ratio)
            scale_factor: Scaling factor (e.g., 0.5 for half size)
            preserve_aspect: Whether to preserve aspect ratio
            method: Interpolation method for resizing
            
        Returns:
            np.ndarray: Resized image
            
        Raises:
            ValueError: If sizing parameters are invalid
        """
        self.validator.validate_image_array(image)
        
        height, width = image.shape[:2]
        
        # Determine target dimensions
        if target_size is not None:
            target_width, target_height = target_size
        elif max_size is not None:
            # Resize to fit within max_size while preserving aspect ratio
            if width > height:
                target_width = max_size
                target_height = int(height * max_size / width)
            else:
                target_height = max_size
                target_width = int(width * max_size / height)
        elif scale_factor is not None:
            target_width = int(width * scale_factor)
            target_height = int(height * scale_factor)
        else:
            raise ValueError("Must specify target_size, max_size, or scale_factor")
        
        # Ensure minimum dimensions
        target_width = max(1, target_width)
        target_height = max(1, target_height)
        
        # Skip resizing if dimensions are the same
        if target_width == width and target_height == height:
            return image.copy()
        
        # Perform resize
        try:
            resized = cv2.resize(image, (target_width, target_height), interpolation=method.value)
            return resized
        except Exception as e:
            raise ValueError(f"Failed to resize image: {e}")
    
    def rotate_image(
        self,
        image: np.ndarray,
        angle: float,
        center: Optional[Tuple[int, int]] = None,
        scale: float = 1.0,
        fill_color: Union[int, Tuple[int, int, int]] = 0
    ) -> np.ndarray:
        """
        Rotate image by specified angle.
        
        Args:
            image: Input image array
            angle: Rotation angle in degrees (positive = counterclockwise)
            center: Rotation center point (default: image center)
            scale: Scaling factor to apply during rotation
            fill_color: Fill color for empty areas
            
        Returns:
            np.ndarray: Rotated image
        """
        self.validator.validate_image_array(image)
        
        height, width = image.shape[:2]
        
        # Use image center if not specified
        if center is None:
            center = (width // 2, height // 2)
        
        # Get rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
        
        # Perform rotation
        try:
            rotated = cv2.warpAffine(
                image, 
                rotation_matrix, 
                (width, height),
                borderValue=fill_color
            )
            return rotated
        except Exception as e:
            raise ValueError(f"Failed to rotate image: {e}")
    
    def crop_image(
        self,
        image: np.ndarray,
        x: int,
        y: int,
        width: int,
        height: int,
        validate_bounds: bool = True
    ) -> np.ndarray:
        """
        Crop image to specified rectangular region.
        
        Args:
            image: Input image array
            x: Top-left x coordinate
            y: Top-left y coordinate
            width: Crop width
            height: Crop height
            validate_bounds: Whether to validate crop bounds
            
        Returns:
            np.ndarray: Cropped image
            
        Raises:
            ValueError: If crop bounds are invalid
        """
        self.validator.validate_image_array(image)
        
        img_height, img_width = image.shape[:2]
        
        if validate_bounds:
            # Validate crop bounds
            if x < 0 or y < 0:
                raise ValueError(f"Crop coordinates must be non-negative: ({x}, {y})")
            
            if x + width > img_width or y + height > img_height:
                raise ValueError(f"Crop region exceeds image bounds: ({x}, {y}, {width}, {height})")
            
            if width <= 0 or height <= 0:
                raise ValueError(f"Crop dimensions must be positive: ({width}, {height})")
        else:
            # Clamp to image bounds
            x = max(0, min(x, img_width - 1))
            y = max(0, min(y, img_height - 1))
            width = min(width, img_width - x)
            height = min(height, img_height - y)
        
        # Perform crop
        try:
            cropped = image[y:y+height, x:x+width].copy()
            return cropped
        except Exception as e:
            raise ValueError(f"Failed to crop image: {e}")
    
    def flip_image(self, image: np.ndarray, horizontal: bool = False, vertical: bool = False) -> np.ndarray:
        """
        Flip image horizontally and/or vertically.
        
        Args:
            image: Input image array
            horizontal: Whether to flip horizontally
            vertical: Whether to flip vertically
            
        Returns:
            np.ndarray: Flipped image
        """
        self.validator.validate_image_array(image)
        
        result = image.copy()
        
        if horizontal:
            result = cv2.flip(result, 1)
        
        if vertical:
            result = cv2.flip(result, 0)
        
        return result
    
    def pad_image(
        self,
        image: np.ndarray,
        padding: Union[int, Tuple[int, int], Tuple[int, int, int, int]],
        fill_color: Union[int, Tuple[int, int, int]] = 0
    ) -> np.ndarray:
        """
        Add padding around image.
        
        Args:
            image: Input image array
            padding: Padding specification:
                - int: same padding on all sides
                - (horizontal, vertical): symmetric padding
                - (top, right, bottom, left): specific padding per side
            fill_color: Fill color for padding areas
            
        Returns:
            np.ndarray: Padded image
        """
        self.validator.validate_image_array(image)
        
        # Parse padding specification
        if isinstance(padding, int):
            top = right = bottom = left = padding
        elif len(padding) == 2:
            top = bottom = padding[1]
            left = right = padding[0]
        elif len(padding) == 4:
            top, right, bottom, left = padding
        else:
            raise ValueError("Padding must be int, (h,v), or (top,right,bottom,left)")
        
        # Apply padding
        try:
            padded = cv2.copyMakeBorder(
                image,
                top, bottom, left, right,
                cv2.BORDER_CONSTANT,
                value=fill_color
            )
            return padded
        except Exception as e:
            raise ValueError(f"Failed to pad image: {e}")


class CoordinateTransformer:
    """
    Coordinate system transformations for bounding box operations.
    
    Provides utilities for converting coordinates between different formats
    and transforming coordinates through image operations.
    """
    
    @staticmethod
    def normalize_coordinates(
        coords: Tuple[float, float, float, float],
        source_format: str,
        target_format: str
    ) -> Tuple[float, float, float, float]:
        """
        Convert coordinates between different formats.
        
        Args:
            coords: Coordinate tuple
            source_format: Source format ('xyxy', 'xywh', 'center_wh')
            target_format: Target format ('xyxy', 'xywh', 'center_wh')
            
        Returns:
            Tuple[float, float, float, float]: Converted coordinates
            
        Raises:
            ValueError: If format is unsupported
        """
        # First convert to xyxy format
        if source_format == 'xyxy':
            x1, y1, x2, y2 = coords
        elif source_format == 'xywh':
            x, y, w, h = coords
            x1, y1, x2, y2 = x, y, x + w, y + h
        elif source_format == 'center_wh':
            cx, cy, w, h = coords
            x1, y1, x2, y2 = cx - w/2, cy - h/2, cx + w/2, cy + h/2
        else:
            raise ValueError(f"Unsupported source format: {source_format}")
        
        # Then convert from xyxy to target format
        if target_format == 'xyxy':
            return (x1, y1, x2, y2)
        elif target_format == 'xywh':
            return (x1, y1, x2 - x1, y2 - y1)
        elif target_format == 'center_wh':
            w, h = x2 - x1, y2 - y1
            return (x1 + w/2, y1 + h/2, w, h)
        else:
            raise ValueError(f"Unsupported target format: {target_format}")
    
    @staticmethod
    def transform_coordinates(
        coords: Tuple[float, float, float, float],
        transform_matrix: np.ndarray
    ) -> Tuple[float, float, float, float]:
        """
        Apply transformation matrix to coordinates.
        
        Args:
            coords: Coordinate tuple in xyxy format
            transform_matrix: 2x3 transformation matrix
            
        Returns:
            Tuple[float, float, float, float]: Transformed coordinates
        """
        x1, y1, x2, y2 = coords
        
        # Convert corners to homogeneous coordinates
        corners = np.array([
            [x1, y1, 1],
            [x2, y2, 1]
        ]).T
        
        # Apply transformation
        transformed = transform_matrix @ corners
        
        # Extract transformed coordinates
        tx1, ty1 = transformed[:, 0]
        tx2, ty2 = transformed[:, 1]
        
        # Ensure proper ordering
        min_x, max_x = min(tx1, tx2), max(tx1, tx2)
        min_y, max_y = min(ty1, ty2), max(ty1, ty2)
        
        return (min_x, min_y, max_x, max_y)
    
    @staticmethod
    def scale_coordinates(
        coords: Tuple[float, float, float, float],
        scale_x: float,
        scale_y: float
    ) -> Tuple[float, float, float, float]:
        """
        Scale coordinates by given factors.
        
        Args:
            coords: Coordinate tuple in xyxy format
            scale_x: Horizontal scaling factor
            scale_y: Vertical scaling factor
            
        Returns:
            Tuple[float, float, float, float]: Scaled coordinates
        """
        x1, y1, x2, y2 = coords
        return (x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y)


class ImageMemoryManager:
    """
    Memory optimization utilities for large image processing.
    
    Provides memory-efficient operations for handling large images and
    batch processing scenarios.
    """
    
    @staticmethod
    def estimate_memory_usage(width: int, height: int, channels: int = 3, dtype: str = "uint8") -> float:
        """
        Estimate memory usage for image array.
        
        Args:
            width: Image width
            height: Image height
            channels: Number of channels
            dtype: Data type
            
        Returns:
            float: Estimated memory usage in MB
        """
        dtype_sizes = {
            'uint8': 1,
            'uint16': 2,
            'float32': 4,
            'float64': 8
        }
        
        bytes_per_pixel = dtype_sizes.get(dtype, 1) * channels
        total_bytes = width * height * bytes_per_pixel
        return total_bytes / (1024 * 1024)  # Convert to MB
    
    @staticmethod
    def optimize_for_memory(image: np.ndarray, max_dimension: int = 2048) -> np.ndarray:
        """
        Optimize image for memory usage by resizing if necessary.
        
        Args:
            image: Input image array
            max_dimension: Maximum allowed dimension
            
        Returns:
            np.ndarray: Memory-optimized image
        """
        height, width = image.shape[:2]
        
        if max(width, height) <= max_dimension:
            return image
        
        # Calculate scaling factor
        scale = max_dimension / max(width, height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Resize image
        processor = ImageProcessor()
        return processor.resize_image(image, target_size=(new_width, new_height))


# Convenience functions for common operations
def load_image(
    source: Union[str, Path, bytes, np.ndarray],
    color_mode: str = "rgb",
    max_size_mb: float = 50
) -> np.ndarray:
    """
    Load image from various sources with automatic format detection.
    
    Args:
        source: Image source (file path, bytes, or array)
        color_mode: Target color mode ('rgb', 'bgr', 'gray')
        max_size_mb: Maximum file size limit in MB
        
    Returns:
        np.ndarray: Loaded image array
        
    Example:
        >>> image = load_image("document.jpg", color_mode="rgb")
        >>> image = load_image(image_bytes, color_mode="gray")
    """
    loader = ImageLoader(max_size_mb=max_size_mb)
    
    if isinstance(source, (str, Path)):
        return loader.load_from_file(source, color_mode)
    elif isinstance(source, bytes):
        return loader.load_from_bytes(source, color_mode)
    elif isinstance(source, np.ndarray):
        return loader.load_from_array(source)
    else:
        raise ValueError(f"Unsupported image source type: {type(source)}")


def save_image(
    image: np.ndarray,
    file_path: Union[str, Path],
    quality: int = 95,
    optimize: bool = True
) -> bool:
    """
    Save image array to file with format optimization.
    
    Args:
        image: Image array to save
        file_path: Output file path
        quality: JPEG quality (1-100)
        optimize: Enable format-specific optimizations
        
    Returns:
        bool: True if save successful
        
    Raises:
        ValueError: If image or path is invalid
        IOError: If file cannot be written
    """
    ImageValidator.validate_image_array(image)
    
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Determine format from extension
        ext = file_path.suffix.lower()
        
        if ext in ['.jpg', '.jpeg']:
            # JPEG format
            if len(image.shape) == 3:
                # Convert RGB to BGR for OpenCV
                save_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                save_image = image
            
            cv2.imwrite(str(file_path), save_image, [cv2.IMWRITE_JPEG_QUALITY, quality])
            
        elif ext == '.png':
            # PNG format
            if len(image.shape) == 3:
                save_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                save_image = image
            
            compression_level = 6 if optimize else 1
            cv2.imwrite(str(file_path), save_image, [cv2.IMWRITE_PNG_COMPRESSION, compression_level])
            
        elif ext in ['.tiff', '.tif']:
            # TIFF format - use PIL for better support
            if len(image.shape) == 2:
                mode = 'L'
            else:
                mode = 'RGB'
            
            pil_image = Image.fromarray(image, mode=mode)
            pil_image.save(file_path, optimize=optimize)
            
        else:
            # Generic format - use OpenCV
            if len(image.shape) == 3:
                save_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                save_image = image
            
            cv2.imwrite(str(file_path), save_image)
        
        return True
        
    except Exception as e:
        raise IOError(f"Failed to save image: {e}")


def resize_image(
    image: np.ndarray,
    target_size: Optional[Tuple[int, int]] = None,
    max_size: Optional[int] = None,
    scale_factor: Optional[float] = None,
    preserve_aspect: bool = True,
    method: str = "linear"
) -> np.ndarray:
    """
    Resize image with intelligent dimension handling.
    
    Args:
        image: Input image array
        target_size: Exact target size as (width, height)
        max_size: Maximum dimension (preserves aspect ratio)
        scale_factor: Scaling factor (e.g., 0.5 for half size)
        preserve_aspect: Whether to preserve aspect ratio
        method: Interpolation method ('nearest', 'linear', 'cubic', 'lanczos', 'area')
        
    Returns:
        np.ndarray: Resized image
        
    Example:
        >>> resized = resize_image(image, max_size=1024)
        >>> scaled = resize_image(image, scale_factor=0.5)
    """
    # Convert method string to enum
    method_map = {
        'nearest': ResizeMethod.NEAREST,
        'linear': ResizeMethod.LINEAR,
        'cubic': ResizeMethod.CUBIC,
        'lanczos': ResizeMethod.LANCZOS,
        'area': ResizeMethod.AREA
    }
    
    resize_method = method_map.get(method.lower(), ResizeMethod.LINEAR)
    
    processor = ImageProcessor()
    return processor.resize_image(
        image=image,
        target_size=target_size,
        max_size=max_size,
        scale_factor=scale_factor,
        preserve_aspect=preserve_aspect,
        method=resize_method
    )


def crop_image(
    image: np.ndarray,
    bbox: Tuple[int, int, int, int],
    format: str = "xyxy"
) -> np.ndarray:
    """
    Crop image using bounding box coordinates.
    
    Args:
        image: Input image array
        bbox: Bounding box coordinates
        format: Coordinate format ('xyxy', 'xywh')
        
    Returns:
        np.ndarray: Cropped image
        
    Example:
        >>> cropped = crop_image(image, (10, 20, 100, 80), format="xyxy")
    """
    processor = ImageProcessor()
    
    if format == "xyxy":
        x1, y1, x2, y2 = bbox
        x, y, width, height = x1, y1, x2 - x1, y2 - y1
    elif format == "xywh":
        x, y, width, height = bbox
    else:
        raise ValueError(f"Unsupported coordinate format: {format}")
    
    return processor.crop_image(image, x, y, width, height)


def rotate_image(
    image: np.ndarray,
    angle: float,
    center: Optional[Tuple[int, int]] = None,
    fill_color: Union[int, Tuple[int, int, int]] = 0
) -> np.ndarray:
    """
    Rotate image by specified angle.
    
    Args:
        image: Input image array
        angle: Rotation angle in degrees (positive = counterclockwise)
        center: Rotation center (default: image center)
        fill_color: Fill color for empty areas
        
    Returns:
        np.ndarray: Rotated image
        
    Example:
        >>> rotated = rotate_image(image, 90)  # 90 degrees counterclockwise
    """
    processor = ImageProcessor()
    return processor.rotate_image(image, angle, center, fill_color=fill_color)


def validate_image(
    image_source: Union[str, Path, np.ndarray],
    max_size_mb: float = 50,
    max_dimension: int = 8192,
    min_dimension: int = 32
) -> bool:
    """
    Validate image source and properties.
    
    Args:
        image_source: Image file path or array
        max_size_mb: Maximum file size in MB
        max_dimension: Maximum allowed dimension
        min_dimension: Minimum allowed dimension
        
    Returns:
        bool: True if image is valid
        
    Raises:
        ValueError: If image is invalid
        
    Example:
        >>> is_valid = validate_image("document.jpg", max_size_mb=10)
    """
    validator = ImageValidator()
    
    if isinstance(image_source, (str, Path)):
        # Validate file
        validator.validate_file_path(image_source)
        validator.validate_file_size(image_source, max_size_mb)
        
        # Load and validate dimensions
        with Image.open(image_source) as img:
            validator.validate_dimensions(img.width, img.height, max_dimension, min_dimension)
    
    elif isinstance(image_source, np.ndarray):
        # Validate array
        validator.validate_image_array(image_source)
        
        height, width = image_source.shape[:2]
        validator.validate_dimensions(width, height, max_dimension, min_dimension)
    
    else:
        raise ValueError(f"Unsupported image source type: {type(image_source)}")
    
    return True


def get_image_properties(image_source: Union[str, Path, np.ndarray]) -> ImageProperties:
    """
    Extract comprehensive image properties and metadata.
    
    Args:
        image_source: Image file path or array
        
    Returns:
        ImageProperties: Complete image property information
        
    Example:
        >>> props = get_image_properties("document.jpg")
        >>> print(f"Size: {props.width}x{props.height}, Format: {props.format}")
    """
    if isinstance(image_source, (str, Path)):
        file_path = Path(image_source)
        
        # Get file size
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        
        # Open with PIL to get metadata
        with Image.open(file_path) as img:
            width, height = img.size
            
            # Determine format
            format_map = {
                'JPEG': ImageFormat.JPEG,
                'PNG': ImageFormat.PNG,
                'TIFF': ImageFormat.TIFF,
                'BMP': ImageFormat.BMP,
                'WEBP': ImageFormat.WEBP
            }
            img_format = format_map.get(img.format, ImageFormat.JPEG)
            
            # Determine channels and color space
            if img.mode == 'L':
                channels = 1
                color_space = ColorSpace.GRAY
            elif img.mode == 'RGB':
                channels = 3
                color_space = ColorSpace.RGB
            elif img.mode == 'RGBA':
                channels = 4
                color_space = ColorSpace.RGB
            else:
                channels = 3
                color_space = ColorSpace.RGB
            
            # Get DPI if available
            dpi = img.info.get('dpi', None)
            
            return ImageProperties(
                width=width,
                height=height,
                channels=channels,
                dtype='uint8',
                format=img_format,
                color_space=color_space,
                file_size_mb=file_size_mb,
                dpi=dpi,
                has_transparency=(img.mode == 'RGBA')
            )
    
    elif isinstance(image_source, np.ndarray):
        height, width = image_source.shape[:2]
        channels = image_source.shape[2] if len(image_source.shape) == 3 else 1
        
        # Determine color space
        if channels == 1:
            color_space = ColorSpace.GRAY
        else:
            color_space = ColorSpace.RGB  # Assume RGB for multi-channel
        
        return ImageProperties(
            width=width,
            height=height,
            channels=channels,
            dtype=str(image_source.dtype),
            format=ImageFormat.JPEG,  # Unknown format for arrays
            color_space=color_space,
            has_transparency=(channels == 4)
        )
    
    else:
        raise ValueError(f"Unsupported image source type: {type(image_source)}")


def normalize_coordinates(
    coords: Tuple[float, float, float, float],
    source_format: str,
    target_format: str
) -> Tuple[float, float, float, float]:
    """
    Convert coordinates between different formats.
    
    Args:
        coords: Coordinate tuple
        source_format: Source format ('xyxy', 'xywh', 'center_wh')
        target_format: Target format ('xyxy', 'xywh', 'center_wh')
        
    Returns:
        Tuple[float, float, float, float]: Converted coordinates
        
    Example:
        >>> xyxy_coords = normalize_coordinates((10, 20, 50, 30), "xywh", "xyxy")
        >>> # Converts (x, y, width, height) to (x1, y1, x2, y2)
    """
    return CoordinateTransformer.normalize_coordinates(coords, source_format, target_format)


def estimate_processing_memory(width: int, height: int, channels: int = 3) -> float:
    """
    Estimate memory usage for image processing operations.
    
    Args:
        width: Image width
        height: Image height
        channels: Number of channels
        
    Returns:
        float: Estimated memory usage in MB
        
    Example:
        >>> memory_mb = estimate_processing_memory(1920, 1080, 3)
        >>> print(f"Estimated memory usage: {memory_mb:.1f} MB")
    """
    return ImageMemoryManager.estimate_memory_usage(width, height, channels)


__all__ = [
    # Classes
    'ImageLoader',
    'ImageProcessor',
    'ImageValidator',
    'CoordinateTransformer',
    'ImageMemoryManager',
    'ImageProperties',
    
    # Enums
    'ImageFormat',
    'ColorSpace',
    'ResizeMethod',
    
    # Convenience functions
    'load_image',
    'save_image',
    'resize_image',
    'crop_image',
    'rotate_image',
    'validate_image',
    'get_image_properties',
    'normalize_coordinates',
    'estimate_processing_memory'
]