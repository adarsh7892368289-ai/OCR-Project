"""
Advanced OCR Text Detection Module

This module provides advanced text region detection capabilities for the advanced OCR system.
It uses state-of-the-art deep learning models like CRAFT (Character Region Awareness for Text Detection)
to accurately identify text regions in images, providing precise bounding boxes and confidence scores.

The module focuses on:
- High-precision text region detection using CRAFT and fallback methods
- Confidence scoring for detected text regions to filter false positives
- Multi-scale detection for handling various text sizes and orientations
- Memory-efficient processing with configurable region limits
- Integration with preprocessing pipeline for targeted image enhancement

Classes:
    TextRegion: Data container for detected text regions with bounding boxes and scores
    DetectionResult: Container for complete text detection results
    CRAFTDetector: CRAFT-based text detector implementation
    FastTextDetector: Fast morphological text detector for fallback
    TextDetector: Main orchestrator coordinating text detection with fallback mechanisms

Functions:
    create_text_detector: Factory function for creating text detector instances
    validate_text_regions: Utility for validating text regions against image boundaries

Example:
    >>> from advanced_ocr.preprocessing.text_detector import TextDetector
    >>> from advanced_ocr.utils.model_utils import ModelLoader
    >>> detector = TextDetector(ModelLoader(), config)
    >>> result = detector.detect_text_regions(image)
    >>> print(f"Detected {len(result)} text regions")
    >>> for region in result[:3]:
    >>>     print(f"Region: confidence={region.confidence:.3f}, bbox={region.bbox}")

    >>> # Filter high-confidence regions
    >>> high_conf_regions = [r for r in result if r.confidence > 0.8]
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Union
import time
from pathlib import Path
import math

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None

# Import from parent modules (correct relative imports)
from ...config import OCRConfig
from ...utils.logger import OCRLogger
from ...utils.model_utils import ModelLoader
from ...results import BoundingBox, TextRegion


class CRAFTDetector:
    """
    CRAFT (Character Region Awareness for Text detection) implementation.
    Detects text regions with high precision and proper region filtering.
    """
    
    def __init__(self, model_loader: ModelLoader, config: OCRConfig):
        """
        Initialize CRAFT detector.
        
        Args:
            model_loader (ModelLoader): Model loader instance
            config (OCRConfig): OCR configuration
        """
        self.model_loader = model_loader
        self.config = config
        self.logger = OCRLogger()
        self.model = None
        self.device = 'cuda' if TORCH_AVAILABLE and torch.cuda.is_available() else 'cpu'
        
        # Detection parameters from config
        self.text_threshold = config.get("text_detection.craft.text_threshold", 0.7)  # FIXED: was 0.1
        self.link_threshold = config.get("text_detection.craft.link_threshold", 0.4)
        self.low_text = config.get("text_detection.craft.low_text", 0.4)
        self.canvas_size = config.get("text_detection.craft.canvas_size", 1280)
        self.mag_ratio = config.get("text_detection.craft.mag_ratio", 1.5)
        
        # Region filtering parameters - CRITICAL for performance
        self.min_region_area = config.get("text_detection.min_region_area", 100)
        self.max_region_area = config.get("text_detection.max_region_area", 50000)
        self.min_aspect_ratio = config.get("text_detection.min_aspect_ratio", 0.1)
        self.max_aspect_ratio = config.get("text_detection.max_aspect_ratio", 20.0)
        self.nms_threshold = config.get("text_detection.nms_threshold", 0.3)
        self.max_regions = config.get("text_detection.max_regions", 80)  # Enforce limit
    
    def load_model(self) -> None:
        """Load CRAFT model with proper error handling."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for CRAFT text detection")
        
        try:
            self.logger.info("Loading CRAFT text detection model")
            self.model = self.model_loader.load_model("craft_mlt_25k", "pytorch", device=self.device)
            self.model.eval()
            self.model.to(self.device)
            self.logger.info(f"CRAFT model loaded successfully on {self.device}")
        
        except Exception as e:
            self.logger.error(f"Failed to load CRAFT model: {str(e)}")
            raise e
    
    def detect_text_regions(self, image: np.ndarray) -> List[TextRegion]:
        """
        Detect text regions in image using CRAFT.
        
        Args:
            image (np.ndarray): Input image (already preprocessed)
            
        Returns:
            List[TextRegion]: List of detected text regions (20-80 regions)
        """
        if self.model is None:
            self.load_model()
        
        start_time = time.time()
        
        # Prepare image for CRAFT
        processed_image, ratio_h, ratio_w = self._prepare_image_for_craft(image)
        
        # Run CRAFT inference
        with torch.no_grad():
            text_map, link_map = self._run_craft_inference(processed_image)
        
        # Extract regions from heatmaps
        raw_regions = self._extract_regions_from_heatmaps(
            text_map, link_map, ratio_h, ratio_w
        )
        
        # CRITICAL: Apply aggressive filtering to reduce regions
        filtered_regions = self._filter_and_limit_regions(raw_regions, image.shape)
        
        detection_time = time.time() - start_time
        
        self.logger.info(
            f"CRAFT detection: {len(raw_regions)}→{len(filtered_regions)} regions "
            f"in {detection_time:.3f}s"
        )
        
        return filtered_regions
    
    def _prepare_image_for_craft(self, image: np.ndarray) -> Tuple[torch.Tensor, float, float]:
        """
        Prepare image for CRAFT model input.
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            Tuple[torch.Tensor, float, float]: (processed_tensor, ratio_h, ratio_w)
        """
        # Get image dimensions
        img_height, img_width = image.shape[:2]
        
        # Calculate target size maintaining aspect ratio
        target_size = self.canvas_size
        ratio = min(target_size / img_width, target_size / img_height)
        
        # Apply magnitude ratio for better detection
        ratio = ratio * self.mag_ratio
        
        new_width = int(img_width * ratio)
        new_height = int(img_height * ratio)
        
        # Resize image
        resized_img = cv2.resize(image, (new_width, new_height))
        
        # Pad image to target size
        target_h = target_size
        target_w = target_size
        
        # Create padded image
        padded_img = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        padded_img[:new_height, :new_width] = resized_img
        
        # Convert to tensor
        img_tensor = torch.from_numpy(padded_img).permute(2, 0, 1).float()
        img_tensor = img_tensor.unsqueeze(0) / 255.0
        img_tensor = img_tensor.to(self.device)
        
        # Calculate ratios for coordinate mapping
        ratio_h = new_height / img_height
        ratio_w = new_width / img_width
        
        return img_tensor, ratio_h, ratio_w
    
    def _run_craft_inference(self, image_tensor: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run CRAFT model inference.
        
        Args:
            image_tensor (torch.Tensor): Preprocessed image tensor
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (text_map, link_map)
        """
        try:
            # CRAFT forward pass
            y, feature = self.model(image_tensor)
            
            # Extract text and link score maps
            score_text = y[0, :, :, 0].cpu().data.numpy()
            score_link = y[0, :, :, 1].cpu().data.numpy()
            
            return score_text, score_link
        
        except Exception as e:
            self.logger.error(f"CRAFT inference failed: {str(e)}")
            # Return zero maps as fallback
            h, w = image_tensor.shape[2:]
            return np.zeros((h, w)), np.zeros((h, w))
    
    def _extract_regions_from_heatmaps(self, text_map: np.ndarray, link_map: np.ndarray,
                                     ratio_h: float, ratio_w: float) -> List[TextRegion]:
        """
        Extract text regions from CRAFT heatmaps.
        
        Args:
            text_map (np.ndarray): Text confidence heatmap
            link_map (np.ndarray): Link confidence heatmap
            ratio_h (float): Height ratio for coordinate mapping
            ratio_w (float): Width ratio for coordinate mapping
            
        Returns:
            List[TextRegion]: Raw detected regions (before filtering)
        """
        # Threshold the text map
        text_mask = text_map > self.text_threshold
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            text_mask.astype(np.uint8), connectivity=8
        )
        
        regions = []
        
        for i in range(1, num_labels):  # Skip background label 0
            # Get component statistics
            x, y, w, h, area = stats[i]
            
            # Basic area filter
            if area < self.min_region_area:
                continue
            
            # Calculate confidence score from text map
            mask = (labels == i)
            confidence = float(np.mean(text_map[mask]))
            
            # Skip low confidence regions
            if confidence < self.low_text:
                continue
            
            # Convert coordinates back to original image scale
            orig_x = int(x / ratio_w)
            orig_y = int(y / ratio_h)
            orig_w = int(w / ratio_w)
            orig_h = int(h / ratio_h)
            
            # Create bounding box
            bbox = BoundingBox(
                x=orig_x,
                y=orig_y,
                width=orig_w,
                height=orig_h
            )
            
            # Create text region
            region = TextRegion(
                text="",  # Text will be extracted by OCR engines
                confidence=confidence,
                bbox=bbox,
                metadata={
                    'detection_method': 'craft',
                    'area': area,
                    'centroid': (centroids[i][0] / ratio_w, centroids[i][1] / ratio_h)
                }
            )
            
            regions.append(region)
        
        return regions
    
    def _filter_and_limit_regions(self, regions: List[TextRegion], 
                                image_shape: Tuple[int, int]) -> List[TextRegion]:
        """
        CRITICAL: Apply aggressive filtering to reduce regions to 20-80.
        
        Args:
            regions (List[TextRegion]): Raw detected regions
            image_shape (Tuple[int, int]): Original image shape (H, W)
            
        Returns:
            List[TextRegion]: Filtered and limited regions
        """
        if not regions:
            return []
        
        img_height, img_width = image_shape[:2]
        filtered_regions = []
        
        for region in regions:
            bbox = region.bbox
            
            # Size filters
            area = bbox.width * bbox.height
            if not (self.min_region_area <= area <= self.max_region_area):
                continue
            
            # Aspect ratio filter
            aspect_ratio = bbox.width / max(bbox.height, 1)
            if not (self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio):
                continue
            
            # Boundary check - ensure region is within image
            if (bbox.x < 0 or bbox.y < 0 or 
                bbox.x + bbox.width > img_width or 
                bbox.y + bbox.height > img_height):
                continue
            
            # Minimum dimension check
            if bbox.width < 10 or bbox.height < 5:
                continue
            
            filtered_regions.append(region)
        
        # Apply Non-Maximum Suppression to remove overlapping regions
        nms_regions = self._apply_nms(filtered_regions)
        
        # Sort by confidence and limit to max_regions
        nms_regions.sort(key=lambda r: r.confidence, reverse=True)
        final_regions = nms_regions[:self.max_regions]
        
        self.logger.debug(
            f"Region filtering: {len(regions)}→{len(filtered_regions)}→"
            f"{len(nms_regions)}→{len(final_regions)} regions"
        )
        
        return final_regions
    
    def _apply_nms(self, regions: List[TextRegion]) -> List[TextRegion]:
        """
        Apply Non-Maximum Suppression to remove overlapping regions.
        
        Args:
            regions (List[TextRegion]): Input regions
            
        Returns:
            List[TextRegion]: Regions after NMS
        """
        if len(regions) <= 1:
            return regions
        
        # Convert to format suitable for NMS
        boxes = []
        scores = []
        
        for region in regions:
            bbox = region.bbox
            boxes.append([bbox.x, bbox.y, bbox.x + bbox.width, bbox.y + bbox.height])
            scores.append(region.confidence)
        
        boxes = np.array(boxes, dtype=np.float32)
        scores = np.array(scores, dtype=np.float32)
        
        # Apply NMS
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(), scores.tolist(), 
            score_threshold=self.low_text,
            nms_threshold=self.nms_threshold
        )
        
        # Extract kept regions
        if len(indices) > 0:
            indices = indices.flatten()
            return [regions[i] for i in indices]
        
        return []


class FastTextDetector:
    """
    Fast text detector using OpenCV's morphological operations.
    Fallback when CRAFT is not available or for speed-critical applications.
    """
    
    def __init__(self, config: OCRConfig):
        """
        Initialize fast text detector.
        
        Args:
            config (OCRConfig): OCR configuration
        """
        self.config = config
        self.logger = OCRLogger()
        
        # Detection parameters
        self.min_area = config.get("text_detection.fast.min_area", 100)
        self.max_area = config.get("text_detection.fast.max_area", 50000)
        self.max_regions = config.get("text_detection.fast.max_regions", 50)
    
    def detect_text_regions(self, image: np.ndarray) -> List[TextRegion]:
        """
        Fast text region detection using morphological operations.
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            List[TextRegion]: Detected text regions
        """
        start_time = time.time()
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply morphological operations to find text regions
        regions = self._detect_using_morphology(gray)
        
        detection_time = time.time() - start_time
        
        self.logger.info(
            f"Fast detection: {len(regions)} regions in {detection_time:.3f}s"
        )
        
        return regions
    
    def _detect_using_morphology(self, gray_image: np.ndarray) -> List[TextRegion]:
        """
        Detect text regions using morphological operations.
        
        Args:
            gray_image (np.ndarray): Grayscale image
            
        Returns:
            List[TextRegion]: Detected regions
        """
        # Apply gradient to highlight text
        sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        gradient = np.sqrt(sobelx**2 + sobely**2)
        
        # Normalize and threshold
        gradient = cv2.normalize(gradient, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        _, binary = cv2.threshold(gradient, 50, 255, cv2.THRESH_BINARY)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
        dilated = cv2.dilate(binary, kernel, iterations=1)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        dilated = cv2.dilate(dilated, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        regions = []
        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            
            # Filter by area
            if not (self.min_area <= area <= self.max_area):
                continue
            
            # Filter by aspect ratio
            aspect_ratio = w / max(h, 1)
            if not (0.2 <= aspect_ratio <= 15.0):
                continue
            
            # Create region
            bbox = BoundingBox(x=x, y=y, width=w, height=h)
            region = TextRegion(
                text="",
                confidence=0.8,  # Default confidence for morphological detection
                bbox=bbox,
                metadata={
                    'detection_method': 'morphological',
                    'area': area,
                    'aspect_ratio': aspect_ratio
                }
            )
            
            regions.append(region)
        
        # Sort by area (larger regions first) and limit
        regions.sort(key=lambda r: r.bbox.width * r.bbox.height, reverse=True)
        return regions[:self.max_regions]


class TextDetector:
    """
    Main text detector that orchestrates different detection methods.
    Automatically selects best available detector and applies consistent filtering.
    """
    
    def __init__(self, model_loader: ModelLoader, config: OCRConfig):
        """
        Initialize text detector with model loader and configuration.
        
        Args:
            model_loader (ModelLoader): Model loader instance  
            config (OCRConfig): OCR configuration
        """
        self.model_loader = model_loader
        self.config = config
        self.logger = OCRLogger()
        
        # Initialize detectors
        self.craft_detector = None
        self.fast_detector = FastTextDetector(config)
        
        # Detection method preference
        self.preferred_method = config.get("text_detection.method", "craft")
        self.fallback_enabled = config.get("text_detection.fallback_enabled", True)
        
        # Global region limits
        self.min_regions = config.get("text_detection.min_regions", 5)
        self.max_regions = config.get("text_detection.max_regions", 80)
    
    def detect_text_regions(self, image: np.ndarray) -> List[TextRegion]:
        """
        Detect text regions using the best available method.
        
        CRITICAL: This is the ONLY method called by image_processor.py
        Ensures 20-80 high-quality text regions are returned.
        
        Args:
            image (np.ndarray): Input image (already preprocessed by image_processor.py)
            
        Returns:
            List[TextRegion]: List of 20-80 detected text regions
        """
        start_time = time.time()
        
        # Try preferred method first
        regions = []
        method_used = None
        
        if self.preferred_method == "craft":
            regions, method_used = self._try_craft_detection(image)
        
        # Fallback to fast detection if needed
        if not regions and self.fallback_enabled:
            self.logger.info("Falling back to fast text detection")
            regions = self.fast_detector.detect_text_regions(image)
            method_used = "fast"
        
        # Final validation and limiting
        final_regions = self._validate_and_limit_regions(regions)
        
        detection_time = time.time() - start_time
        
        self.logger.info(
            f"Text detection ({method_used}): {len(final_regions)} regions "
            f"in {detection_time:.3f}s"
        )
        
        return final_regions
    
    def _try_craft_detection(self, image: np.ndarray) -> Tuple[List[TextRegion], str]:
        """
        Try CRAFT detection with error handling.
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            Tuple[List[TextRegion], str]: (regions, method_name)
        """
        try:
            if self.craft_detector is None:
                self.craft_detector = CRAFTDetector(self.model_loader, self.config)
            
            regions = self.craft_detector.detect_text_regions(image)
            return regions, "craft"
        
        except Exception as e:
            self.logger.warning(f"CRAFT detection failed: {str(e)}")
            return [], "failed"
    
    def _validate_and_limit_regions(self, regions: List[TextRegion]) -> List[TextRegion]:
        """
        Final validation and limiting of detected regions.
        
        Args:
            regions (List[TextRegion]): Input regions
            
        Returns:
            List[TextRegion]: Validated and limited regions
        """
        if not regions:
            self.logger.warning("No text regions detected")
            return []
        
        # Remove invalid regions
        valid_regions = []
        for region in regions:
            if (region.bbox.width > 0 and region.bbox.height > 0 and 
                region.confidence > 0):
                valid_regions.append(region)
        
        # Sort by confidence and apply final limit
        valid_regions.sort(key=lambda r: r.confidence, reverse=True)
        limited_regions = valid_regions[:self.max_regions]
        
        # Check if we have minimum regions
        if len(limited_regions) < self.min_regions:
            self.logger.warning(
                f"Only {len(limited_regions)} regions detected "
                f"(minimum: {self.min_regions})"
            )
        
        return limited_regions
    
    def get_detection_stats(self) -> Dict[str, Any]:
        """
        Get detection statistics and configuration.
        
        Returns:
            Dict[str, Any]: Detection statistics
        """
        return {
            'preferred_method': self.preferred_method,
            'fallback_enabled': self.fallback_enabled,
            'min_regions': self.min_regions,
            'max_regions': self.max_regions,
            'craft_available': TORCH_AVAILABLE and self.craft_detector is not None,
            'fast_available': True
        }


# Utility functions for external use
def create_text_detector(model_loader: ModelLoader, 
                        config: Optional[OCRConfig] = None) -> TextDetector:
    """
    Create a text detector instance.
    
    Args:
        model_loader (ModelLoader): Model loader instance
        config (Optional[OCRConfig]): OCR configuration
        
    Returns:
        TextDetector: Configured text detector
    """
    if config is None:
        from ...config import OCRConfig
        config = OCRConfig()
    
    return TextDetector(model_loader, config)


def validate_text_regions(regions: List[TextRegion], 
                         image_shape: Tuple[int, int]) -> List[TextRegion]:
    """
    Validate text regions against image boundaries.
    
    Args:
        regions (List[TextRegion]): Input regions
        image_shape (Tuple[int, int]): Image shape (H, W)
        
    Returns:
        List[TextRegion]: Valid regions
    """
    if not regions:
        return []
    
    img_height, img_width = image_shape[:2]
    valid_regions = []
    
    for region in regions:
        bbox = region.bbox
        
        # Check boundaries
        if (bbox.x >= 0 and bbox.y >= 0 and 
            bbox.x + bbox.width <= img_width and 
            bbox.y + bbox.height <= img_height and
            bbox.width > 0 and bbox.height > 0):
            valid_regions.append(region)
    
    return valid_regions


__all__ = [
    'CRAFTDetector', 'FastTextDetector', 'TextDetector',
    'create_text_detector', 'validate_text_regions'
]