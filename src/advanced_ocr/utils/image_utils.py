"""
Efficient image processing utilities with memory management.
Provides modern image operations for OCR preprocessing and coordinate transformations.
"""

import numpy as np
import cv2
from PIL import Image, ImageOps, ImageEnhance
import base64
import io
from typing import Union, Tuple, Optional, List, Any
from pathlib import Path
from dataclasses import dataclass
import warnings

from ..results import BoundingBox, CoordinateFormat


@dataclass
class ImageMetadata:
    """Metadata container for image information."""
    
    width: int
    height: int
    channels: int
    dtype: str
    format: str = "unknown"
    dpi: Tuple[int, int] = (72, 72)
    color_space: str = "RGB"
    file_size: Optional[int] = None
    
    @property
    def aspect_ratio(self) -> float:
        """Calculate aspect ratio (width/height)."""
        return self.width / self.height if self.height > 0 else 1.0
    
    @property
    def total_pixels(self) -> int:
        """Calculate total number of pixels."""
        return self.width * self.height
    
    @property
    def memory_size_mb(self) -> float:
        """Estimate memory size in MB."""
        bytes_per_pixel = 4 if self.channels == 4 else 3
        return (self.total_pixels * bytes_per_pixel) / (1024 * 1024)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'width': self.width,
            'height': self.height,
            'channels': self.channels,
            'dtype': self.dtype,
            'format': self.format,
            'dpi': self.dpi,
            'color_space': self.color_space,
            'aspect_ratio': self.aspect_ratio,
            'total_pixels': self.total_pixels,
            'memory_size_mb': self.memory_size_mb,
            'file_size': self.file_size
        }


class ImageLoader:
    """Efficient image loading with support for multiple input formats."""
    
    @staticmethod
    def load(source: Union[str, Path, bytes, np.ndarray, Image.Image]) -> Tuple[np.ndarray, ImageMetadata]:
        """
        Load image from various sources and return as numpy array with metadata.
        
        Args:
            source: Image source (file path, bytes, base64, PIL Image, or numpy array)
            
        Returns:
            Tuple of (image_array, metadata)
        """
        if isinstance(source, (str, Path)):
            return ImageLoader._load_from_file(source)
        elif isinstance(source, bytes):
            return ImageLoader._load_from_bytes(source)
        elif isinstance(source, str) and source.startswith('data:image'):
            return ImageLoader._load_from_base64(source)
        elif isinstance(source, Image.Image):
            return ImageLoader._load_from_pil(source)
        elif isinstance(source, np.ndarray):
            return ImageLoader._load_from_numpy(source)
        else:
            raise ValueError(f"Unsupported image source type: {type(source)}")
    
    @staticmethod
    def _load_from_file(file_path: Union[str, Path]) -> Tuple[np.ndarray, ImageMetadata]:
        """Load image from file path."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Image file not found: {file_path}")
        
        try:
            # Use PIL for better format support and metadata extraction
            with Image.open(file_path) as pil_image:
                # Get metadata
                width, height = pil_image.size
                file_size = file_path.stat().st_size
                format_name = pil_image.format or "unknown"
                
                # Get DPI if available
                dpi = pil_image.info.get('dpi', (72, 72))
                if isinstance(dpi, (int, float)):
                    dpi = (int(dpi), int(dpi))
                
                # Convert to RGB if necessary
                if pil_image.mode not in ['RGB', 'RGBA', 'L']:
                    pil_image = pil_image.convert('RGB')
                
                # Convert to numpy array
                image_array = np.array(pil_image)
                
                # Handle grayscale
                if len(image_array.shape) == 2:
                    image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
                elif image_array.shape[2] == 4:  # RGBA
                    image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
                
                metadata = ImageMetadata(
                    width=width,
                    height=height,
                    channels=image_array.shape[2] if len(image_array.shape) == 3 else 1,
                    dtype=str(image_array.dtype),
                    format=format_name,
                    dpi=dpi,
                    color_space="RGB",
                    file_size=file_size
                )
                
                return image_array, metadata
                
        except Exception as e:
            # Fallback to OpenCV
            try:
                image_array = cv2.imread(str(file_path))
                if image_array is None:
                    raise ValueError(f"Could not load image: {file_path}")
                
                # OpenCV loads as BGR, convert to RGB
                image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
                
                height, width, channels = image_array.shape
                file_size = file_path.stat().st_size
                
                metadata = ImageMetadata(
                    width=width,
                    height=height,
                    channels=channels,
                    dtype=str(image_array.dtype),
                    format=file_path.suffix[1:].upper(),
                    file_size=file_size
                )
                
                return image_array, metadata
                
            except Exception:
                raise ValueError(f"Failed to load image from {file_path}: {e}")
    
    @staticmethod
    def _load_from_bytes(image_bytes: bytes) -> Tuple[np.ndarray, ImageMetadata]:
        """Load image from bytes."""
        try:
            # Try PIL first
            with Image.open(io.BytesIO(image_bytes)) as pil_image:
                width, height = pil_image.size
                format_name = pil_image.format or "unknown"
                
                if pil_image.mode not in ['RGB', 'RGBA', 'L']:
                    pil_image = pil_image.convert('RGB')
                
                image_array = np.array(pil_image)
                
                if len(image_array.shape) == 2:
                    image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
                elif image_array.shape[2] == 4:
                    image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
                
                metadata = ImageMetadata(
                    width=width,
                    height=height,
                    channels=image_array.shape[2] if len(image_array.shape) == 3 else 1,
                    dtype=str(image_array.dtype),
                    format=format_name,
                    file_size=len(image_bytes)
                )
                
                return image_array, metadata
                
        except Exception:
            # Fallback to OpenCV
            try:
                nparr = np.frombuffer(image_bytes, np.uint8)
                image_array = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if image_array is None:
                    raise ValueError("Could not decode image from bytes")
                
                image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
                height, width, channels = image_array.shape
                
                metadata = ImageMetadata(
                    width=width,
                    height=height,
                    channels=channels,
                    dtype=str(image_array.dtype),
                    file_size=len(image_bytes)
                )
                
                return image_array, metadata
                
            except Exception as e:
                raise ValueError(f"Failed to load image from bytes: {e}")
    
    @staticmethod
    def _load_from_base64(base64_string: str) -> Tuple[np.ndarray, ImageMetadata]:
        """Load image from base64 string."""
        try:
            # Extract base64 data from data URL if present
            if base64_string.startswith('data:image'):
                base64_data = base64_string.split(',')[1]
            else:
                base64_data = base64_string
            
            image_bytes = base64.b64decode(base64_data)
            return ImageLoader._load_from_bytes(image_bytes)
            
        except Exception as e:
            raise ValueError(f"Failed to load image from base64: {e}")
    
    @staticmethod
    def _load_from_pil(pil_image: Image.Image) -> Tuple[np.ndarray, ImageMetadata]:
        """Load image from PIL Image."""
        try:
            width, height = pil_image.size
            format_name = pil_image.format or "PIL"
            
            # Get DPI if available
            dpi = pil_image.info.get('dpi', (72, 72))
            if isinstance(dpi, (int, float)):
                dpi = (int(dpi), int(dpi))
            
            # Ensure RGB format
            if pil_image.mode not in ['RGB', 'RGBA', 'L']:
                pil_image = pil_image.convert('RGB')
            
            image_array = np.array(pil_image)
            
            if len(image_array.shape) == 2:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
            elif image_array.shape[2] == 4:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
            
            metadata = ImageMetadata(
                width=width,
                height=height,
                channels=image_array.shape[2] if len(image_array.shape) == 3 else 1,
                dtype=str(image_array.dtype),
                format=format_name,
                dpi=dpi,
                color_space="RGB"
            )
            
            return image_array, metadata
            
        except Exception as e:
            raise ValueError(f"Failed to load PIL image: {e}")
    
    @staticmethod
    def _load_from_numpy(image_array: np.ndarray) -> Tuple[np.ndarray, ImageMetadata]:
        """Load image from numpy array."""
        try:
            # Validate array
            if len(image_array.shape) not in [2, 3]:
                raise ValueError("Image array must be 2D or 3D")
            
            # Handle different array shapes
            if len(image_array.shape) == 2:
                height, width = image_array.shape
                channels = 1
                # Convert grayscale to RGB
                image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
            else:
                height, width, channels = image_array.shape
                if channels == 4:  # RGBA
                    image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)
                    channels = 3
                elif channels == 1:  # Grayscale with channel dimension
                    image_array = cv2.cvtColor(image_array.squeeze(), cv2.COLOR_GRAY2RGB)
                    channels = 3
            
            # Ensure correct data type
            if image_array.dtype != np.uint8:
                if image_array.max() <= 1.0:
                    image_array = (image_array * 255).astype(np.uint8)
                else:
                    image_array = image_array.astype(np.uint8)
            
            metadata = ImageMetadata(
                width=width,
                height=height,
                channels=channels,
                dtype=str(image_array.dtype),
                format="numpy",
                color_space="RGB"
            )
            
            return image_array, metadata
            
        except Exception as e:
            raise ValueError(f"Failed to load numpy array: {e}")


class ImageProcessor:
    """Core image transformations with quality preservation."""
    
    @staticmethod
    def resize(image: np.ndarray, target_size: Tuple[int, int], 
               preserve_aspect_ratio: bool = True, 
               interpolation: int = cv2.INTER_LANCZOS4) -> Tuple[np.ndarray, float]:
        """
        Resize image with quality preservation.
        
        Args:
            image: Input image array
            target_size: Target (width, height)
            preserve_aspect_ratio: Whether to preserve aspect ratio
            interpolation: OpenCV interpolation method
            
        Returns:
            Tuple of (resized_image, scale_factor)
        """
        height, width = image.shape[:2]
        target_width, target_height = target_size
        
        if preserve_aspect_ratio:
            # Calculate scale factor to fit within target size
            scale_x = target_width / width
            scale_y = target_height / height
            scale = min(scale_x, scale_y)
            
            new_width = int(width * scale)
            new_height = int(height * scale)
        else:
            new_width, new_height = target_width, target_height
            scale = (target_width / width + target_height / height) / 2
        
        # Choose interpolation method based on scaling direction
        if scale < 1.0:
            # Downscaling - use area interpolation for better quality
            interpolation = cv2.INTER_AREA
        else:
            # Upscaling - use cubic or lanczos
            interpolation = interpolation or cv2.INTER_CUBIC
        
        resized = cv2.resize(image, (new_width, new_height), interpolation=interpolation)
        
        return resized, scale
    
    @staticmethod
    def rotate(image: np.ndarray, angle: float, 
               background_color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
        """
        Rotate image by given angle with automatic size adjustment.
        
        Args:
            image: Input image array
            angle: Rotation angle in degrees (positive = counterclockwise)
            background_color: Background color for padded areas
            
        Returns:
            Rotated image
        """
        height, width = image.shape[:2]
        
        # Calculate rotation matrix
        center = (width // 2, height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate new dimensions to fit rotated image
        cos_angle = abs(rotation_matrix[0, 0])
        sin_angle = abs(rotation_matrix[0, 1])
        
        new_width = int((height * sin_angle) + (width * cos_angle))
        new_height = int((height * cos_angle) + (width * sin_angle))
        
        # Adjust translation to center the rotated image
        rotation_matrix[0, 2] += (new_width / 2) - center[0]
        rotation_matrix[1, 2] += (new_height / 2) - center[1]
        
        # Perform rotation
        rotated = cv2.warpAffine(
            image, rotation_matrix, (new_width, new_height),
            borderValue=background_color,
            flags=cv2.INTER_CUBIC
        )
        
        return rotated
    
    @staticmethod
    def crop(image: np.ndarray, bbox: BoundingBox) -> np.ndarray:
        """
        Crop image using bounding box.
        
        Args:
            image: Input image array
            bbox: Bounding box for cropping
            
        Returns:
            Cropped image
        """
        x1, y1, x2, y2 = bbox.xyxy
        
        # Ensure coordinates are within image bounds
        height, width = image.shape[:2]
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(width, int(x2))
        y2 = min(height, int(y2))
        
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Invalid bounding box for cropping")
        
        return image[y1:y2, x1:x2]
    
    @staticmethod
    def enhance_contrast(image: np.ndarray, factor: float = 1.2) -> np.ndarray:
        """
        Enhance image contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization).
        
        Args:
            image: Input image array
            factor: Contrast enhancement factor
            
        Returns:
            Contrast-enhanced image
        """
        if len(image.shape) == 3:
            # Convert to LAB color space for better contrast enhancement
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=factor, tileGridSize=(8, 8))
            enhanced_l = clahe.apply(l_channel)
            
            # Merge channels and convert back to RGB
            enhanced_lab = cv2.merge([enhanced_l, a_channel, b_channel])
            enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)
        else:
            # Grayscale image
            clahe = cv2.createCLAHE(clipLimit=factor, tileGridSize=(8, 8))
            enhanced = clahe.apply(image)
        
        return enhanced
    
    @staticmethod
    def reduce_noise(image: np.ndarray, strength: float = 0.3) -> np.ndarray:
        """
        Reduce image noise using Non-local Means Denoising.
        
        Args:
            image: Input image array
            strength: Denoising strength (0.0 to 1.0)
            
        Returns:
            Denoised image
        """
        # Scale strength to appropriate range
        h = strength * 30  # Filter strength
        template_window_size = 7
        search_window_size = 21
        
        if len(image.shape) == 3:
            # Color image
            denoised = cv2.fastNlMeansDenoisingColored(
                image, None, h, h, template_window_size, search_window_size
            )
        else:
            # Grayscale image
            denoised = cv2.fastNlMeansDenoising(
                image, None, h, template_window_size, search_window_size
            )
        
        return denoised
    
    @staticmethod
    def sharpen(image: np.ndarray, strength: float = 0.5) -> np.ndarray:
        """
        Sharpen image using unsharp masking.
        
        Args:
            image: Input image array
            strength: Sharpening strength (0.0 to 2.0)
            
        Returns:
            Sharpened image
        """
        # Create Gaussian blur for unsharp mask
        blurred = cv2.GaussianBlur(image, (0, 0), 1.0)
        
        # Create sharpened image
        sharpened = cv2.addWeighted(image, 1 + strength, blurred, -strength, 0)
        
        return np.clip(sharpened, 0, 255).astype(np.uint8)
    
    @staticmethod
    def adjust_brightness(image: np.ndarray, adjustment: float = 0.0) -> np.ndarray:
        """
        Adjust image brightness.
        
        Args:
            image: Input image array
            adjustment: Brightness adjustment (-1.0 to 1.0)
            
        Returns:
            Brightness-adjusted image
        """
        # Convert adjustment to 0-255 range
        brightness_offset = adjustment * 255
        
        adjusted = image.astype(np.float32) + brightness_offset
        adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
        
        return adjusted
    
    @staticmethod
    def normalize_orientation(image: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Detect and correct image orientation based on text lines.
        
        Args:
            image: Input image array
            
        Returns:
            Tuple of (corrected_image, rotation_angle)
        """
        try:
            # Convert to grayscale for line detection
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()
            
            # Apply edge detection
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # Detect lines using Hough transform
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            if lines is None or len(lines) < 3:
                return image, 0.0
            
            # Calculate dominant angle
            angles = []
            for line in lines:
                rho, theta = line[0]
                angle = (theta - np.pi/2) * 180 / np.pi
                
                # Normalize angle to [-45, 45] range
                while angle > 45:
                    angle -= 90
                while angle < -45:
                    angle += 90
                
                angles.append(angle)
            
            # Find dominant angle using histogram
            angle_hist, angle_bins = np.histogram(angles, bins=90, range=(-45, 45))
            dominant_angle = angle_bins[np.argmax(angle_hist)]
            
            # Only correct if angle is significant
            if abs(dominant_angle) > 1.0:
                corrected = ImageProcessor.rotate(image, -dominant_angle)
                return corrected, -dominant_angle
            else:
                return image, 0.0
                
        except Exception:
            # Return original image if orientation detection fails
            return image, 0.0


class CoordinateTransformer:
    """Bounding box operations and coordinate system conversions."""
    
    @staticmethod
    def scale_coordinates(coords: List[BoundingBox], scale_factor: float) -> List[BoundingBox]:
        """
        Scale bounding box coordinates by given factor.
        
        Args:
            coords: List of bounding boxes
            scale_factor: Scaling factor
            
        Returns:
            List of scaled bounding boxes
        """
        return [coord.scale(scale_factor) for coord in coords]
    
    @staticmethod
    def translate_coordinates(coords: List[BoundingBox], 
                            offset: Tuple[float, float]) -> List[BoundingBox]:
        """
        Translate bounding box coordinates by offset.
        
        Args:
            coords: List of bounding boxes
            offset: (x_offset, y_offset)
            
        Returns:
            List of translated bounding boxes
        """
        dx, dy = offset
        translated = []
        
        for coord in coords:
            if coord.format == CoordinateFormat.XYXY:
                x1, y1, x2, y2 = coord.coordinates
                new_coords = (x1 + dx, y1 + dy, x2 + dx, y2 + dy)
            elif coord.format == CoordinateFormat.XYWH:
                x, y, w, h = coord.coordinates
                new_coords = (x + dx, y + dy, w, h)
            else:  # CENTER format
                cx, cy, w, h = coord.coordinates
                new_coords = (cx + dx, cy + dy, w, h)
            
            translated.append(BoundingBox(new_coords, coord.format))
        
        return translated
    
    @staticmethod
    def rotate_coordinates(coords: List[BoundingBox], angle: float, 
                          image_center: Tuple[float, float],
                          new_image_size: Tuple[int, int]) -> List[BoundingBox]:
        """
        Rotate bounding box coordinates around image center.
        
        Args:
            coords: List of bounding boxes
            angle: Rotation angle in degrees
            image_center: Original image center (x, y)
            new_image_size: New image size after rotation (width, height)
            
        Returns:
            List of rotated bounding boxes
        """
        angle_rad = np.radians(angle)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        cx, cy = image_center
        new_cx, new_cy = new_image_size[0] / 2, new_image_size[1] / 2
        
        rotated = []
        
        for coord in coords:
            # Get all four corners of bounding box
            x1, y1, x2, y2 = coord.xyxy
            corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            
            # Rotate each corner
            rotated_corners = []
            for x, y in corners:
                # Translate to origin
                x_rel, y_rel = x - cx, y - cy
                
                # Rotate
                x_rot = x_rel * cos_a - y_rel * sin_a
                y_rot = x_rel * sin_a + y_rel * cos_a
                
                # Translate to new center
                x_new = x_rot + new_cx
                y_new = y_rot + new_cy
                
                rotated_corners.append((x_new, y_new))
            
            # Find new bounding box from rotated corners
            xs, ys = zip(*rotated_corners)
            new_x1, new_y1 = min(xs), min(ys)
            new_x2, new_y2 = max(xs), max(ys)
            
            rotated.append(BoundingBox(
                (new_x1, new_y1, new_x2, new_y2),
                CoordinateFormat.XYXY
            ))
        
        return rotated
    
    @staticmethod
    def crop_coordinates(coords: List[BoundingBox], 
                        crop_bbox: BoundingBox) -> List[BoundingBox]:
        """
        Adjust coordinates for cropped image.
        
        Args:
            coords: List of bounding boxes
            crop_bbox: Cropping bounding box
            
        Returns:
            List of adjusted bounding boxes (only those within crop area)
        """
        crop_x1, crop_y1, _, _ = crop_bbox.xyxy
        adjusted = []
        
        for coord in coords:
            x1, y1, x2, y2 = coord.xyxy
            
            # Check if bounding box intersects with crop area
            if coord.intersects(crop_bbox):
                # Adjust coordinates relative to crop area
                new_x1 = max(0, x1 - crop_x1)
                new_y1 = max(0, y1 - crop_y1)
                new_x2 = max(0, x2 - crop_x1)
                new_y2 = max(0, y2 - crop_y1)
                
                # Only add if resulting box is valid
                if new_x2 > new_x1 and new_y2 > new_y1:
                    adjusted.append(BoundingBox(
                        (new_x1, new_y1, new_x2, new_y2),
                        CoordinateFormat.XYXY
                    ))
        
        return adjusted
    
    @staticmethod
    def convert_format(coord: BoundingBox, target_format: CoordinateFormat) -> BoundingBox:
        """
        Convert bounding box to different coordinate format.
        
        Args:
            coord: Input bounding box
            target_format: Target coordinate format
            
        Returns:
            Bounding box in target format
        """
        if target_format == CoordinateFormat.XYXY:
            return BoundingBox(coord.xyxy, CoordinateFormat.XYXY)
        elif target_format == CoordinateFormat.XYWH:
            return BoundingBox(coord.xywh, CoordinateFormat.XYWH)
        elif target_format == CoordinateFormat.CENTER:
            return BoundingBox(coord.center, CoordinateFormat.CENTER)
        else:
            raise ValueError(f"Unsupported coordinate format: {target_format}")
    
    @staticmethod
    def filter_by_size(coords: List[BoundingBox], 
                      min_area: float = 10.0, 
                      max_area: Optional[float] = None,
                      min_aspect_ratio: float = 0.1,
                      max_aspect_ratio: float = 10.0) -> List[BoundingBox]:
        """
        Filter bounding boxes by size and aspect ratio criteria.
        
        Args:
            coords: List of bounding boxes
            min_area: Minimum area threshold
            max_area: Maximum area threshold (None for no limit)
            min_aspect_ratio: Minimum aspect ratio
            max_aspect_ratio: Maximum aspect ratio
            
        Returns:
            Filtered list of bounding boxes
        """
        filtered = []
        
        for coord in coords:
            area = coord.area
            _, _, w, h = coord.xywh
            aspect_ratio = w / h if h > 0 else 0
            
            # Check area constraints
            if area < min_area:
                continue
            if max_area is not None and area > max_area:
                continue
            
            # Check aspect ratio constraints
            if aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
                continue
            
            filtered.append(coord)
        
        return filtered
    
    @staticmethod
    def merge_overlapping(coords: List[BoundingBox], 
                         overlap_threshold: float = 0.5) -> List[BoundingBox]:
        """
        Merge overlapping bounding boxes.
        
        Args:
            coords: List of bounding boxes
            overlap_threshold: Minimum overlap ratio to merge
            
        Returns:
            List with merged bounding boxes
        """
        if not coords:
            return []
        
        # Sort by area (largest first)
        sorted_coords = sorted(coords, key=lambda x: x.area, reverse=True)
        merged = []
        
        for coord in sorted_coords:
            should_merge = False
            
            for i, existing in enumerate(merged):
                # Calculate intersection over union (IoU)
                x1_1, y1_1, x2_1, y2_1 = coord.xyxy
                x1_2, y1_2, x2_2, y2_2 = existing.xyxy
                
                # Calculate intersection
                int_x1 = max(x1_1, x1_2)
                int_y1 = max(y1_1, y1_2)
                int_x2 = min(x2_1, x2_2)
                int_y2 = min(y2_1, y2_2)
                
                if int_x2 > int_x1 and int_y2 > int_y1:
                    intersection = (int_x2 - int_x1) * (int_y2 - int_y1)
                    union = coord.area + existing.area - intersection
                    
                    if intersection / union >= overlap_threshold:
                        # Merge boxes by taking union
                        new_x1 = min(x1_1, x1_2)
                        new_y1 = min(y1_1, y1_2)
                        new_x2 = max(x2_1, x2_2)
                        new_y2 = max(y2_1, y2_2)
                        
                        merged[i] = BoundingBox(
                            (new_x1, new_y1, new_x2, new_y2),
                            CoordinateFormat.XYXY
                        )
                        should_merge = True
                        break
            
            if not should_merge:
                merged.append(coord)
        
        return merged


def smart_resize(image: np.ndarray, max_dimension: int = 3000, 
                 min_dimension: int = 100, target_dpi: int = 300) -> Tuple[np.ndarray, float]:
    """
    Smart image resizing based on content and OCR requirements.
    
    Args:
        image: Input image array
        max_dimension: Maximum allowed dimension
        min_dimension: Minimum required dimension
        target_dpi: Target DPI for OCR
        
    Returns:
        Tuple of (resized_image, scale_factor)
    """
    height, width = image.shape[:2]
    max_current_dim = max(height, width)
    min_current_dim = min(height, width)
    
    scale_factor = 1.0
    
    # Check if resizing is needed
    if max_current_dim > max_dimension:
        # Downscale to fit maximum dimension
        scale_factor = max_dimension / max_current_dim
    elif min_current_dim < min_dimension:
        # Upscale to meet minimum dimension
        scale_factor = min_dimension / min_current_dim
    else:
        # Check DPI-based scaling (assuming current image is 72 DPI)
        current_dpi = 72
        if target_dpi != current_dpi:
            dpi_scale = target_dpi / current_dpi
            # Only apply DPI scaling if it doesn't violate size constraints
            new_max_dim = max_current_dim * dpi_scale
            if min_dimension <= new_max_dim <= max_dimension:
                scale_factor = dpi_scale
    
    if scale_factor != 1.0:
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        resized, _ = ImageProcessor.resize(image, (new_width, new_height))
        return resized, scale_factor
    
    return image, 1.0


def create_thumbnail(image: np.ndarray, size: Tuple[int, int] = (150, 150)) -> np.ndarray:
    """
    Create thumbnail of image for preview purposes.
    
    Args:
        image: Input image array
        size: Thumbnail size (width, height)
        
    Returns:
        Thumbnail image
    """
    thumbnail, _ = ImageProcessor.resize(image, size, preserve_aspect_ratio=True)
    
    # Pad to exact size if needed
    thumb_height, thumb_width = thumbnail.shape[:2]
    target_width, target_height = size
    
    if thumb_width < target_width or thumb_height < target_height:
        # Calculate padding
        pad_x = max(0, target_width - thumb_width) // 2
        pad_y = max(0, target_height - thumb_height) // 2
        
        # Create padded image
        if len(thumbnail.shape) == 3:
            padded = np.full((target_height, target_width, 3), 255, dtype=np.uint8)
            padded[pad_y:pad_y+thumb_height, pad_x:pad_x+thumb_width] = thumbnail
        else:
            padded = np.full((target_height, target_width), 255, dtype=np.uint8)
            padded[pad_y:pad_y+thumb_height, pad_x:pad_x+thumb_width] = thumbnail
        
        return padded
    
    return thumbnail


def save_image(image: np.ndarray, file_path: Union[str, Path], 
               quality: int = 95, format: str = None) -> bool:
    """
    Save image to file with quality preservation.
    
    Args:
        image: Image array to save
        file_path: Output file path
        quality: JPEG quality (1-100)
        format: Force specific format (optional)
        
    Returns:
        Success status
    """
    try:
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert RGB to BGR for OpenCV
        if len(image.shape) == 3:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image
        
        # Determine format from extension or parameter
        if format is None:
            format = file_path.suffix.lower()
        
        # Set appropriate encoding parameters
        if format in ['.jpg', '.jpeg']:
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        elif format == '.png':
            encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 9 - (quality // 11)]
        elif format == '.tiff':
            encode_params = [cv2.IMWRITE_TIFF_COMPRESSION, 1]
        else:
            encode_params = []
        
        success = cv2.imwrite(str(file_path), image_bgr, encode_params)
        return success
        
    except Exception:
        return False