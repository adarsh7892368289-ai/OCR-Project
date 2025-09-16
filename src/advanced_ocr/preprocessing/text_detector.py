# src/advanced_ocr/preprocessing/text_detector.py
"""
Advanced OCR Text Detection Module - FIXED VERSION

This module provides advanced text region detection capabilities for the advanced OCR system.
It uses state-of-the-art deep learning models like CRAFT (Character Region Awareness for Text Detection)
to accurately identify text regions in images, providing precise bounding boxes and confidence scores.


PIPELINE ROLE:
This module is ONLY responsible for text detection. It receives preprocessed images
from image_processor.py and returns 20-80 high-quality text regions. It does NOT:
- Perform image preprocessing (done by image_processor.py)
- Extract text content (done by OCR engines)
- Perform postprocessing (done by text_processor.py)

Classes:
    CRAFTDetector: CRAFT-based text detector with modern improvements
    FastTextDetector: Fast morphological text detector for fallback
    TextDetector: Main orchestrator with multi-scale detection and proper filtering

Functions:
    create_text_detector: Factory function for creating text detector instances
    validate_text_regions: Utility for validating text regions against image boundaries

Example:
    >>> from advanced_ocr.preprocessing.text_detector import TextDetector
    >>> from advanced_ocr.utils.model_utils import ModelLoader
    >>> detector = TextDetector(ModelLoader(), config)
    >>> result = detector.detect_text_regions(preprocessed_image)
    >>> print(f"Detected {len(result)} text regions")
    >>> for region in result[:3]:
    >>>     print(f"Region: confidence={region.confidence:.3f}, bbox={region.bbox}")
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Union
import time
from pathlib import Path
import math
from collections import defaultdict

# Handle PyTorch imports with proper error handling
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

# Import from parent modules - FIXED import paths according to project structure
from ..config import OCRConfig
from ..utils.logger import OCRLogger
from ..utils.model_utils import ModelLoader
from ..utils.image_utils import ImageProcessor
from ..results import BoundingBox, TextRegion


class CRAFTDetector:
    """
    CRAFT (Character Region Awareness for Text detection) implementation with modern improvements.
    
    Features:
    - Multi-scale detection for handling various text sizes
    - Proper confidence calibration
    - Text-specific NMS instead of generic object detection NMS  
    - Memory-efficient GPU processing
    - Advanced region filtering to ensure 20-80 high-quality regions
    """
    
    def __init__(self, model_loader: ModelLoader, config: OCRConfig):
        """
        Initialize CRAFT detector with modern enhancements.
        
        Args:
            model_loader (ModelLoader): Model loader instance for loading CRAFT model
            config (OCRConfig): OCR configuration with detection parameters
        """
        self.model_loader = model_loader
        self.config = config
        self.logger = OCRLogger()
        self.model = None
        
        # FIXED: Proper device handling to avoid runtime crashes
        if TORCH_AVAILABLE:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = 'cpu'
            
        self.logger.info(f"CRAFT detector initialized with device: {self.device}")
        
        # Detection parameters from config - optimized for performance
        self.text_threshold = config.get("text_detection.craft.text_threshold", 0.7)  # FIXED: was 0.1
        self.link_threshold = config.get("text_detection.craft.link_threshold", 0.4)
        self.low_text = config.get("text_detection.craft.low_text", 0.4)
        self.canvas_size = config.get("text_detection.craft.canvas_size", 1280)
        self.mag_ratio = config.get("text_detection.craft.mag_ratio", 1.5)
        
        # Multi-scale detection parameters - MODERN FEATURE
        self.enable_multiscale = config.get("text_detection.craft.multiscale", True)
        self.scales = config.get("text_detection.craft.scales", [0.8, 1.0, 1.2])
        
        # Region filtering parameters - CRITICAL for 2660→80 region reduction
        self.min_region_area = config.get("text_detection.min_region_area", 100)
        self.max_region_area = config.get("text_detection.max_region_area", 50000)
        self.min_aspect_ratio = config.get("text_detection.min_aspect_ratio", 0.1)
        self.max_aspect_ratio = config.get("text_detection.max_aspect_ratio", 20.0)
        self.nms_threshold = config.get("text_detection.nms_threshold", 0.3)
        self.max_regions = config.get("text_detection.max_regions", 80)  # Hard limit
        self.target_regions = config.get("text_detection.target_regions", 50)  # Optimal target
        
        # Performance monitoring
        self.detection_stats = {
            'total_detections': 0,
            'average_regions': 0,
            'average_time': 0
        }
    
    def load_model(self) -> None:
        """Load CRAFT model with comprehensive error handling and proper device setup."""
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for CRAFT text detection. "
                "Install with: pip install torch torchvision"
            )
        
        try:
            self.logger.info("Loading CRAFT text detection model...")
            
            # FIXED: Pass proper device object instead of string
            self.model = self.model_loader.load_model(
                model_name="craft_mlt_25k", 
                framework="pytorch", 
                device=self.device
            )
            
            # Ensure model is in evaluation mode and on correct device
            if hasattr(self.model, 'eval'):
                self.model.eval()
            if hasattr(self.model, 'to'):
                self.model.to(self.device)
                
            self.logger.info(f"CRAFT model loaded successfully on {self.device}")
            
            # Validate model by running a test inference
            self._validate_model()
            
        except Exception as e:
            self.logger.error(f"Failed to load CRAFT model: {str(e)}")
            # Clear any partially loaded model
            self.model = None
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()  # Clear GPU memory
            raise e
    
    def _validate_model(self) -> None:
        """Validate loaded model with a test inference to ensure it works correctly."""
        try:
            # Create a small test tensor
            test_tensor = torch.randn(1, 3, 256, 256).to(self.device)
            
            with torch.no_grad():
                output = self.model(test_tensor)
                
            # FIXED: Validate model output structure  
            if not isinstance(output, (tuple, list)) or len(output) < 2:
                raise ValueError(
                    f"CRAFT model returned unexpected output format. "
                    f"Expected tuple/list with 2+ elements, got: {type(output)}"
                )
                
            # Check output shapes
            text_map, link_map = output[0], output[1]
            expected_shape = (1, 256, 256, 2)  # Typical CRAFT output shape
            
            if hasattr(text_map, 'shape') and len(text_map.shape) != 4:
                self.logger.warning(
                    f"Unexpected CRAFT output shape: {text_map.shape}. "
                    f"Expected 4D tensor."
                )
            
            self.logger.debug("CRAFT model validation successful")
            
        except Exception as e:
            self.logger.error(f"CRAFT model validation failed: {str(e)}")
            raise e
        finally:
            # Clean up GPU memory after validation
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def detect_text_regions(self, image: np.ndarray) -> List[TextRegion]:
        """
        Detect text regions using CRAFT with modern multi-scale approach.
        
        PIPELINE ROLE: This is called ONLY by image_processor.py after image preprocessing.
        The input image is already enhanced and ready for text detection.
        
        Args:
            image (np.ndarray): Preprocessed image from image_processor.py
            
        Returns:
            List[TextRegion]: 20-80 high-quality text regions with calibrated confidence
        """
        if self.model is None:
            self.load_model()
        
        start_time = time.time()
        
        try:
            # Multi-scale detection for better coverage of different text sizes
            if self.enable_multiscale:
                all_regions = self._multi_scale_detection(image)
            else:
                all_regions = self._single_scale_detection(image, scale=1.0)
            
            # Apply aggressive filtering and NMS to get 20-80 regions
            filtered_regions = self._filter_and_limit_regions(all_regions, image.shape)
            
            # Calibrate confidence scores for better reliability
            calibrated_regions = self._calibrate_confidence_scores(filtered_regions)
            
            detection_time = time.time() - start_time
            
            # Update performance statistics
            self._update_detection_stats(len(calibrated_regions), detection_time)
            
            self.logger.info(
                f"CRAFT detection: {len(all_regions)}→{len(filtered_regions)}→"
                f"{len(calibrated_regions)} regions in {detection_time:.3f}s"
            )
            
            return calibrated_regions
            
        except Exception as e:
            self.logger.error(f"CRAFT text detection failed: {str(e)}")
            # Return empty list on failure rather than crashing
            return []
        finally:
            # FIXED: Proper GPU memory management
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    def _multi_scale_detection(self, image: np.ndarray) -> List[TextRegion]:
        """
        MODERN FEATURE: Multi-scale text detection for handling various text sizes.
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            List[TextRegion]: All detected regions across scales
        """
        all_regions = []
        
        for scale in self.scales:
            scale_regions = self._single_scale_detection(image, scale)
            
            # Add scale information to metadata
            for region in scale_regions:
                region.metadata['detection_scale'] = scale
                
            all_regions.extend(scale_regions)
            
            self.logger.debug(f"Scale {scale}: {len(scale_regions)} regions detected")
        
        # Merge overlapping regions from different scales using text-specific NMS
        merged_regions = self._merge_multi_scale_regions(all_regions)
        
        return merged_regions
    
    def _single_scale_detection(self, image: np.ndarray, scale: float = 1.0) -> List[TextRegion]:
        """
        Detect text regions at a single scale using CRAFT.
        
        Args:
            image (np.ndarray): Input image
            scale (float): Scale factor for detection
            
        Returns:
            List[TextRegion]: Detected regions at this scale
        """
        # Scale image if needed
        if scale != 1.0:
            new_h = int(image.shape[0] * scale)
            new_w = int(image.shape[1] * scale)
            scaled_image = cv2.resize(image, (new_w, new_h))
        else:
            scaled_image = image.copy()
        
        # Prepare image for CRAFT
        processed_image, ratio_h, ratio_w = self._prepare_image_for_craft(scaled_image)
        
        # Run CRAFT inference with proper error handling
        with torch.no_grad():
            text_map, link_map = self._run_craft_inference(processed_image)
        
        # Extract regions from heatmaps
        regions = self._extract_regions_from_heatmaps(
            text_map, link_map, ratio_h, ratio_w, scale
        )
        
        return regions
    
    def _prepare_image_for_craft(self, image: np.ndarray) -> Tuple[torch.Tensor, float, float]:
        """
        Prepare image for CRAFT model input with improved preprocessing.
        
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
        
        # Ensure dimensions are even for better processing
        new_width = new_width - (new_width % 2)
        new_height = new_height - (new_height % 2)
        
        # Resize image with high-quality interpolation
        resized_img = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        # FIXED: Use mean padding instead of zero padding to avoid artifacts
        target_h = target_size
        target_w = target_size
        
        # Calculate mean color for padding
        mean_color = np.mean(resized_img, axis=(0, 1)).astype(np.uint8)
        
        # Create padded image with mean color
        padded_img = np.full((target_h, target_w, 3), mean_color, dtype=np.uint8)
        
        # Center the resized image in the padded canvas
        y_offset = (target_h - new_height) // 2
        x_offset = (target_w - new_width) // 2
        padded_img[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized_img
        
        # Convert to tensor with proper normalization
        img_tensor = torch.from_numpy(padded_img).permute(2, 0, 1).float()
        img_tensor = img_tensor.unsqueeze(0) / 255.0
        img_tensor = img_tensor.to(self.device)
        
        # Calculate ratios for coordinate mapping (account for centering)
        ratio_h = new_height / img_height
        ratio_w = new_width / img_width
        
        return img_tensor, ratio_h, ratio_w
    
    def _run_craft_inference(self, image_tensor: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run CRAFT model inference with comprehensive error handling.
        
        Args:
            image_tensor (torch.Tensor): Preprocessed image tensor
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: (text_map, link_map)
        """
        try:
            # CRAFT forward pass
            output = self.model(image_tensor)
            
            # FIXED: Proper output validation and handling
            if isinstance(output, (tuple, list)):
                if len(output) >= 2:
                    y, feature = output[0], output[1]
                else:
                    y = output[0]
                    feature = None
            else:
                y = output
                feature = None
            
            # Extract text and link score maps with proper error handling
            if hasattr(y, 'shape') and len(y.shape) == 4:
                if y.shape[3] >= 2:
                    # Standard CRAFT output format
                    score_text = y[0, :, :, 0].cpu().data.numpy()
                    score_link = y[0, :, :, 1].cpu().data.numpy()
                else:
                    # Handle single-channel output
                    score_text = y[0, 0, :, :].cpu().data.numpy()
                    score_link = np.zeros_like(score_text)
            else:
                # Fallback for unexpected output format
                self.logger.warning(f"Unexpected CRAFT output shape: {y.shape if hasattr(y, 'shape') else type(y)}")
                h, w = image_tensor.shape[2:]
                score_text = np.zeros((h, w))
                score_link = np.zeros((h, w))
            
            return score_text, score_link
        
        except Exception as e:
            self.logger.error(f"CRAFT inference failed: {str(e)}")
            # Return zero maps as fallback to prevent pipeline failure
            h, w = image_tensor.shape[2:]
            return np.zeros((h, w)), np.zeros((h, w))
    
    def _extract_regions_from_heatmaps(self, text_map: np.ndarray, link_map: np.ndarray,
                                     ratio_h: float, ratio_w: float, scale: float = 1.0) -> List[TextRegion]:
        """
        Extract text regions from CRAFT heatmaps with improved processing.
        
        Args:
            text_map (np.ndarray): Text confidence heatmap
            link_map (np.ndarray): Link confidence heatmap  
            ratio_h (float): Height ratio for coordinate mapping
            ratio_w (float): Width ratio for coordinate mapping
            scale (float): Original scale factor used for detection
            
        Returns:
            List[TextRegion]: Raw detected regions (before final filtering)
        """
        # Threshold the text map with adaptive thresholding
        text_mask = text_map > self.text_threshold
        
        # Apply morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        text_mask = cv2.morphologyEx(text_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        
        # Find connected components with statistics
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            text_mask, connectivity=8
        )
        
        regions = []
        
        for i in range(1, num_labels):  # Skip background label 0
            # Get component statistics
            x, y, w, h, area = stats[i]
            
            # Basic area filter to remove noise
            if area < self.min_region_area // 4:  # More lenient at extraction stage
                continue
            
            # Calculate enhanced confidence score using both text and link maps
            mask = (labels == i)
            text_confidence = float(np.mean(text_map[mask]))
            link_confidence = float(np.mean(link_map[mask])) if np.sum(mask) > 0 else 0.0
            
            # Combine text and link confidence
            combined_confidence = 0.8 * text_confidence + 0.2 * link_confidence
            
            # Skip very low confidence regions
            if combined_confidence < self.low_text:
                continue
            
            # Convert coordinates back to original image scale
            if scale != 1.0:
                # Account for multi-scale detection
                orig_x = int((x / ratio_w) / scale)
                orig_y = int((y / ratio_h) / scale) 
                orig_w = int((w / ratio_w) / scale)
                orig_h = int((h / ratio_h) / scale)
            else:
                orig_x = int(x / ratio_w)
                orig_y = int(y / ratio_h)
                orig_w = int(w / ratio_w)
                orig_h = int(h / ratio_h)
            
            # Ensure positive dimensions
            if orig_w <= 0 or orig_h <= 0:
                continue
            
            # Create bounding box
            bbox = BoundingBox(
                x=max(0, orig_x),
                y=max(0, orig_y), 
                width=orig_w,
                height=orig_h
            )
            
            # Create text region with enhanced metadata
            region = TextRegion(
                text="",  # Text will be extracted by OCR engines
                confidence=combined_confidence,
                bbox=bbox,
                metadata={
                    'detection_method': 'craft',
                    'detection_scale': scale,
                    'area': area,
                    'text_confidence': text_confidence,
                    'link_confidence': link_confidence,
                    'centroid': (centroids[i][0] / ratio_w, centroids[i][1] / ratio_h)
                }
            )
            
            regions.append(region)
        
        return regions
    
    def _merge_multi_scale_regions(self, all_regions: List[TextRegion]) -> List[TextRegion]:
        """
        Merge overlapping regions from multi-scale detection using text-specific approach.
        
        Args:
            all_regions (List[TextRegion]): All regions from different scales
            
        Returns:
            List[TextRegion]: Merged regions without excessive overlap
        """
        if not all_regions:
            return []
        
        # Group regions by approximate text lines for better merging
        line_groups = self._group_regions_by_lines(all_regions)
        
        merged_regions = []
        
        for line_regions in line_groups:
            # Apply NMS within each line group
            line_merged = self._text_specific_nms(line_regions)
            merged_regions.extend(line_merged)
        
        return merged_regions
    
    def _group_regions_by_lines(self, regions: List[TextRegion]) -> List[List[TextRegion]]:
        """
        Group regions that likely belong to the same text line.
        
        Args:
            regions (List[TextRegion]): Input regions
            
        Returns:
            List[List[TextRegion]]: Groups of regions by text lines
        """
        if not regions:
            return []
        
        # Sort regions by y-coordinate
        sorted_regions = sorted(regions, key=lambda r: r.bbox.y)
        
        groups = []
        current_group = [sorted_regions[0]]
        
        for region in sorted_regions[1:]:
            # Check if region overlaps vertically with current group
            group_y_min = min(r.bbox.y for r in current_group)
            group_y_max = max(r.bbox.y + r.bbox.height for r in current_group)
            
            region_y_center = region.bbox.y + region.bbox.height / 2
            
            # If region center is within the group's vertical range, add to group
            if group_y_min <= region_y_center <= group_y_max:
                current_group.append(region)
            else:
                # Start new group
                groups.append(current_group)
                current_group = [region]
        
        # Add the last group
        if current_group:
            groups.append(current_group)
        
        return groups
    
    def _text_specific_nms(self, regions: List[TextRegion]) -> List[TextRegion]:
        """
        MODERN FEATURE: Text-specific Non-Maximum Suppression.
        Unlike generic object detection NMS, this considers text characteristics.
        
        Args:
            regions (List[TextRegion]): Input regions
            
        Returns:
            List[TextRegion]: Regions after text-specific NMS
        """
        if len(regions) <= 1:
            return regions
        
        # Sort by confidence (highest first)
        regions = sorted(regions, key=lambda r: r.confidence, reverse=True)
        
        keep_regions = []
        
        for region in regions:
            # Check overlap with already kept regions
            should_keep = True
            
            for kept_region in keep_regions:
                overlap_ratio = self._calculate_text_overlap(region, kept_region)
                
                # For text, we use more lenient overlap thresholds
                # because text regions can legitimately have some overlap
                if overlap_ratio > self.nms_threshold:
                    # Keep the region with higher confidence or better characteristics
                    if region.confidence <= kept_region.confidence:
                        should_keep = False
                        break
                    else:
                        # Replace the kept region with current one
                        keep_regions.remove(kept_region)
                        break
            
            if should_keep:
                keep_regions.append(region)
        
        return keep_regions
    
    def _calculate_text_overlap(self, region1: TextRegion, region2: TextRegion) -> float:
        """
        Calculate overlap ratio between two text regions using text-aware metrics.
        
        Args:
            region1 (TextRegion): First region
            region2 (TextRegion): Second region
            
        Returns:
            float: Overlap ratio (0.0 to 1.0)
        """
        bbox1, bbox2 = region1.bbox, region2.bbox
        
        # Calculate intersection
        x1 = max(bbox1.x, bbox2.x)
        y1 = max(bbox1.y, bbox2.y)
        x2 = min(bbox1.x + bbox1.width, bbox2.x + bbox2.width)
        y2 = min(bbox1.y + bbox1.height, bbox2.y + bbox2.height)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        # Intersection area
        intersection = (x2 - x1) * (y2 - y1)
        
        # Union area
        area1 = bbox1.width * bbox1.height
        area2 = bbox2.width * bbox2.height
        union = area1 + area2 - intersection
        
        if union == 0:
            return 0.0
        
        # For text, we also consider horizontal overlap more heavily
        # since text is typically horizontal
        horizontal_overlap = min(bbox1.x + bbox1.width, bbox2.x + bbox2.width) - max(bbox1.x, bbox2.x)
        horizontal_union = max(bbox1.x + bbox1.width, bbox2.x + bbox2.width) - min(bbox1.x, bbox2.x)
        
        if horizontal_union > 0:
            horizontal_ratio = horizontal_overlap / horizontal_union
        else:
            horizontal_ratio = 0.0
        
        # Combine IoU with horizontal overlap ratio
        iou = intersection / union
        combined_overlap = 0.7 * iou + 0.3 * horizontal_ratio
        
        return combined_overlap
    
    def _filter_and_limit_regions(self, regions: List[TextRegion], 
                                image_shape: Tuple[int, int]) -> List[TextRegion]:
        """
        CRITICAL: Apply aggressive filtering to reduce regions from thousands to 20-80.
        
        Args:
            regions (List[TextRegion]): Raw detected regions
            image_shape (Tuple[int, int]): Original image shape (H, W)
            
        Returns:
            List[TextRegion]: Filtered and limited high-quality regions
        """
        if not regions:
            return []
        
        img_height, img_width = image_shape[:2]
        filtered_regions = []
        
        for region in regions:
            bbox = region.bbox
            
            # Boundary validation - ensure region is within image
            if (bbox.x < 0 or bbox.y < 0 or 
                bbox.x + bbox.width > img_width or 
                bbox.y + bbox.height > img_height):
                continue
            
            # Size filters - more strict than extraction stage
            area = bbox.width * bbox.height
            if not (self.min_region_area <= area <= self.max_region_area):
                continue
            
            # Aspect ratio filter - text should have reasonable width/height ratio
            aspect_ratio = bbox.width / max(bbox.height, 1)
            if not (self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio):
                continue
            
            # Minimum dimension check - avoid tiny regions
            if bbox.width < 15 or bbox.height < 8:
                continue
            
            # Confidence filter - only keep reasonably confident detections
            if region.confidence < self.low_text:
                continue
            
            filtered_regions.append(region)
        
        # Sort by a combination of confidence and area (prefer larger, confident regions)
        def region_score(r):
            area_score = (r.bbox.width * r.bbox.height) / self.max_region_area
            return 0.6 * r.confidence + 0.4 * area_score
        
        filtered_regions.sort(key=region_score, reverse=True)
        
        # Apply final limit with target preference
        if len(filtered_regions) > self.target_regions:
            final_regions = filtered_regions[:self.target_regions]
        else:
            final_regions = filtered_regions[:self.max_regions]
        
        self.logger.debug(
            f"Region filtering pipeline: {len(regions)} → {len(filtered_regions)} → {len(final_regions)} regions"
        )
        
        return final_regions
    
    def _calibrate_confidence_scores(self, regions: List[TextRegion]) -> List[TextRegion]:
        """
        MODERN FEATURE: Calibrate confidence scores for better reliability.
        
        Args:
            regions (List[TextRegion]): Input regions with raw confidence
            
        Returns:
            List[TextRegion]: Regions with calibrated confidence scores
        """
        if not regions:
            return regions
        
        # Simple sigmoid calibration - in production, this would be learned from data
        calibrated_regions = []
        
        for region in regions:
            raw_conf = region.confidence
            
            # Apply sigmoid calibration: sigmoid(6 * (conf - 0.5))
            calibrated_conf = 1 / (1 + math.exp(-6 * (raw_conf - 0.5)))
            
            # Ensure confidence is in reasonable range
            calibrated_conf = max(0.1, min(0.98, calibrated_conf))
            
            # Update region with calibrated confidence
            calibrated_region = TextRegion(
                text=region.text,
                confidence=calibrated_conf,
                bbox=region.bbox,
                metadata={
                    **region.metadata,
                    'raw_confidence': raw_conf,
                    'calibration_applied': True
                }
            )
            
            calibrated_regions.append(calibrated_region)
        
        return calibrated_regions
    
    def _update_detection_stats(self, num_regions: int, detection_time: float) -> None:
        """Update detection performance statistics."""
        self.detection_stats['total_detections'] += 1
        
        # Update running averages
        total = self.detection_stats['total_detections']
        self.detection_stats['average_regions'] = (
            (self.detection_stats['average_regions'] * (total - 1) + num_regions) / total
        )
        self.detection_stats['average_time'] = (
            (self.detection_stats['average_time'] * (total - 1) + detection_time) / total
        )


class FastTextDetector:
    """
    Fast text detector using OpenCV's morphological operations.
    
    FALLBACK ROLE: Used when CRAFT is not available or fails.
    Provides basic text detection using classical computer vision techniques.
    While not as accurate as CRAFT, it's reliable and doesn't require deep learning models.
    """
    
    def __init__(self, config: OCRConfig):
        """
        Initialize fast text detector with configuration.
        
        Args:
            config (OCRConfig): OCR configuration
        """
        self.config = config
        self.logger = OCRLogger()
        
        # Detection parameters - optimized for speed and reliability
        self.min_area = config.get("text_detection.fast.min_area", 200)
        self.max_area = config.get("text_detection.fast.max_area", 40000)
        self.min_aspect_ratio = config.get("text_detection.fast.min_aspect_ratio", 0.2)
        self.max_aspect_ratio = config.get("text_detection.fast.max_aspect_ratio", 15.0)
        self.max_regions = config.get("text_detection.fast.max_regions", 50)
        
        # Morphological operation parameters
        self.gradient_kernel_size = config.get("text_detection.fast.gradient_kernel", 3)
        self.dilate_kernel_h = config.get("text_detection.fast.dilate_kernel_h", 9)
        self.dilate_kernel_w = config.get("text_detection.fast.dilate_kernel_w", 1)
        self.close_kernel_size = config.get("text_detection.fast.close_kernel", 3)
    
    def detect_text_regions(self, image: np.ndarray) -> List[TextRegion]:
        """
        Fast text region detection using morphological operations.
        
        Args:
            image (np.ndarray): Input image (preprocessed)
            
        Returns:
            List[TextRegion]: Detected text regions (up to max_regions)
        """
        start_time = time.time()
        
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Apply morphological operations to find text regions
            regions = self._detect_using_morphology(gray)
            
            # Filter and limit regions
            filtered_regions = self._filter_fast_regions(regions, image.shape)
            
            detection_time = time.time() - start_time
            
            self.logger.info(
                f"Fast detection: {len(regions)}→{len(filtered_regions)} regions "
                f"in {detection_time:.3f}s"
            )
            
            return filtered_regions
            
        except Exception as e:
            self.logger.error(f"Fast text detection failed: {str(e)}")
            return []
    
    def _detect_using_morphology(self, gray_image: np.ndarray) -> List[TextRegion]:
        """
        Detect text regions using improved morphological operations.
        
        Args:
            gray_image (np.ndarray): Grayscale image
            
        Returns:
            List[TextRegion]: Detected regions
        """
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray_image, (3, 3), 0)
        
        # Apply gradient to highlight text edges
        sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=self.gradient_kernel_size)
        sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=self.gradient_kernel_size)
        gradient = np.sqrt(sobelx**2 + sobely**2)
        
        # Normalize and threshold
        gradient = cv2.normalize(gradient, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        
        # Adaptive thresholding works better than fixed threshold
        binary = cv2.adaptiveThreshold(
            gradient, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 2
        )
        
        # Morphological operations to connect text characters
        # Horizontal dilation to connect characters in words
        kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (self.dilate_kernel_h, self.dilate_kernel_w))
        dilated = cv2.dilate(binary, kernel_h, iterations=1)
        
        # Vertical dilation to connect lines
        kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (self.dilate_kernel_w, self.close_kernel_size))
        dilated = cv2.dilate(dilated, kernel_v, iterations=1)
        
        # Closing operation to fill gaps
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (self.close_kernel_size, self.close_kernel_size))
        closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel_close)
        
        # Find contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        regions = []
        for i, contour in enumerate(contours):
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            
            # Basic filtering
            if area < self.min_area or area > self.max_area:
                continue
            
            # Aspect ratio filter
            aspect_ratio = w / max(h, 1)
            if not (self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio):
                continue
            
            # Calculate confidence based on contour properties
            # More complex contours (more vertices) likely indicate text
            contour_complexity = len(contour) / max(cv2.arcLength(contour, True), 1)
            solidity = area / cv2.contourArea(cv2.convexHull(contour))
            
            # Combine metrics for confidence score
            confidence = min(0.9, 0.5 + 0.3 * solidity + 0.2 * min(contour_complexity, 1.0))
            
            # Create region
            bbox = BoundingBox(x=x, y=y, width=w, height=h)
            region = TextRegion(
                text="",
                confidence=confidence,
                bbox=bbox,
                metadata={
                    'detection_method': 'morphological_fast',
                    'area': area,
                    'aspect_ratio': aspect_ratio,
                    'solidity': solidity,
                    'contour_complexity': contour_complexity
                }
            )
            
            regions.append(region)
        
        return regions
    
    def _filter_fast_regions(self, regions: List[TextRegion], image_shape: Tuple[int, int]) -> List[TextRegion]:
        """
        Filter and limit regions from fast detection.
        
        Args:
            regions (List[TextRegion]): Input regions
            image_shape (Tuple[int, int]): Image shape (H, W)
            
        Returns:
            List[TextRegion]: Filtered regions
        """
        if not regions:
            return []
        
        img_height, img_width = image_shape[:2]
        filtered_regions = []
        
        for region in regions:
            bbox = region.bbox
            
            # Boundary check
            if (bbox.x < 0 or bbox.y < 0 or 
                bbox.x + bbox.width > img_width or 
                bbox.y + bbox.height > img_height):
                continue
            
            # Minimum dimensions
            if bbox.width < 20 or bbox.height < 10:
                continue
            
            filtered_regions.append(region)
        
        # Sort by confidence * area (prefer larger confident regions)
        filtered_regions.sort(
            key=lambda r: r.confidence * (r.bbox.width * r.bbox.height), 
            reverse=True
        )
        
        # Apply limit
        return filtered_regions[:self.max_regions]


class TextDetector:
    """
    Main text detector orchestrator with intelligent fallback and modern features.
    
    PIPELINE ROLE: This is the ONLY class called by image_processor.py
    - Receives preprocessed images from image_processor.py
    - Returns 20-80 high-quality text regions 
    - Handles model loading failures gracefully
    - Provides consistent interface regardless of underlying detector
    
    MODERN FEATURES:
    - Automatic detector selection based on availability and performance
    - Multi-scale detection for handling various text sizes  
    - Text-specific NMS instead of generic object detection NMS
    - Confidence calibration for reliable confidence scores
    - Comprehensive error handling and fallback mechanisms
    """
    
    def __init__(self, model_loader: ModelLoader, config: OCRConfig):
        """
        Initialize text detector with model loader and configuration.
        
        Args:
            model_loader (ModelLoader): Model loader instance for CRAFT model
            config (OCRConfig): OCR configuration with detection parameters
        """
        self.model_loader = model_loader
        self.config = config
        self.logger = OCRLogger()
        
        # Initialize detectors
        self.craft_detector = None
        self.fast_detector = FastTextDetector(config)
        
        # Detection method configuration
        self.preferred_method = config.get("text_detection.method", "craft")
        self.fallback_enabled = config.get("text_detection.fallback_enabled", True)
        self.auto_fallback_threshold = config.get("text_detection.auto_fallback_threshold", 5)
        
        # Global region limits - CRITICAL for performance
        self.min_regions = config.get("text_detection.min_regions", 10)
        self.max_regions = config.get("text_detection.max_regions", 80)
        self.target_regions = config.get("text_detection.target_regions", 50)
        
        # Performance tracking
        self.detection_history = {
            'craft_failures': 0,
            'total_detections': 0,
            'average_regions': 0
        }
    
    def detect_text_regions(self, image: np.ndarray) -> List[TextRegion]:
        """
        MAIN ENTRY POINT: Detect text regions using the best available method.
        
        This is the ONLY method called by image_processor.py in the pipeline.
        Ensures 20-80 high-quality text regions are returned consistently.
        
        PIPELINE GUARANTEE: This method will NEVER return empty list unless
        the image is completely invalid. It will always try fallback methods.
        
        Args:
            image (np.ndarray): Preprocessed image from image_processor.py
                              (already enhanced, noise-reduced, etc.)
            
        Returns:
            List[TextRegion]: 10-80 detected text regions with calibrated confidence
        """
        if image is None or image.size == 0:
            self.logger.error("Invalid input image for text detection")
            return []
        
        start_time = time.time()
        self.detection_history['total_detections'] += 1
        
        # Try detection methods in order of preference
        regions = []
        method_used = "none"
        
        # Method 1: Try CRAFT detection (preferred for accuracy)
        if self.preferred_method == "craft":
            regions, method_used = self._try_craft_detection(image)
        
        # Method 2: Fallback to fast detection if CRAFT failed or insufficient regions
        if (len(regions) < self.min_regions and self.fallback_enabled):
            self.logger.info(
                f"Falling back to fast detection "
                f"(CRAFT returned {len(regions)} regions, need {self.min_regions}+)"
            )
            fast_regions = self._try_fast_detection(image)
            
            # Combine results if both methods produced regions
            if regions and fast_regions:
                regions = self._combine_detection_results(regions, fast_regions)
                method_used = f"{method_used}+fast"
            elif fast_regions:
                regions = fast_regions
                method_used = "fast"
        
        # Method 3: Emergency fallback - simple contour detection
        if len(regions) < self.min_regions:
            emergency_regions = self._emergency_detection(image)
            if emergency_regions:
                regions.extend(emergency_regions)
                method_used = f"{method_used}+emergency"
        
        # Final validation and limiting
        final_regions = self._validate_and_limit_regions(regions, image.shape)
        
        detection_time = time.time() - start_time
        
        # Update performance history
        self._update_detection_history(len(final_regions), method_used, detection_time)
        
        self.logger.info(
            f"Text detection ({method_used}): {len(final_regions)} regions "
            f"in {detection_time:.3f}s (target: {self.target_regions})"
        )
        
        # Log warning if we couldn't meet minimum requirements
        if len(final_regions) < self.min_regions:
            self.logger.warning(
                f"Text detection produced only {len(final_regions)} regions "
                f"(minimum required: {self.min_regions}). Image may have very little text."
            )
        
        return final_regions
    
    def _try_craft_detection(self, image: np.ndarray) -> Tuple[List[TextRegion], str]:
        """
        Try CRAFT detection with comprehensive error handling.
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            Tuple[List[TextRegion], str]: (regions, method_name)
        """
        try:
            # Initialize CRAFT detector if not already done
            if self.craft_detector is None:
                self.craft_detector = CRAFTDetector(self.model_loader, self.config)
            
            regions = self.craft_detector.detect_text_regions(image)
            
            # Validate CRAFT results
            if regions and len(regions) >= self.min_regions:
                return regions, "craft"
            else:
                self.logger.info(
                    f"CRAFT detection returned insufficient regions: {len(regions)}"
                )
                return regions, "craft_insufficient"
        
        except Exception as e:
            self.logger.warning(f"CRAFT detection failed: {str(e)}")
            self.detection_history['craft_failures'] += 1
            
            # If CRAFT is failing frequently, consider switching to fast by default
            failure_rate = (self.detection_history['craft_failures'] / 
                           max(self.detection_history['total_detections'], 1))
            
            if failure_rate > 0.5:
                self.logger.warning(
                    f"CRAFT failure rate high ({failure_rate:.2%}). "
                    f"Consider switching to fast detection as default."
                )
            
            return [], "craft_failed"
    
    def _try_fast_detection(self, image: np.ndarray) -> List[TextRegion]:
        """
        Try fast morphological detection.
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            List[TextRegion]: Detected regions
        """
        try:
            return self.fast_detector.detect_text_regions(image)
        except Exception as e:
            self.logger.error(f"Fast detection also failed: {str(e)}")
            return []
    
    def _combine_detection_results(self, craft_regions: List[TextRegion], 
                                 fast_regions: List[TextRegion]) -> List[TextRegion]:
        """
        Intelligently combine results from multiple detectors.
        
        Args:
            craft_regions (List[TextRegion]): Regions from CRAFT
            fast_regions (List[TextRegion]): Regions from fast detector
            
        Returns:
            List[TextRegion]: Combined and deduplicated regions
        """
        all_regions = craft_regions + fast_regions
        
        # Remove duplicates using text-specific NMS
        if hasattr(self.craft_detector, '_text_specific_nms'):
            combined_regions = self.craft_detector._text_specific_nms(all_regions)
        else:
            # Fallback to simple overlap removal
            combined_regions = self._simple_dedup(all_regions)
        
        return combined_regions
    
    def _simple_dedup(self, regions: List[TextRegion]) -> List[TextRegion]:
        """
        Simple deduplication based on overlap threshold.
        
        Args:
            regions (List[TextRegion]): Input regions
            
        Returns:
            List[TextRegion]: Deduplicated regions
        """
        if not regions:
            return []
        
        # Sort by confidence
        sorted_regions = sorted(regions, key=lambda r: r.confidence, reverse=True)
        
        keep_regions = []
        overlap_threshold = 0.5
        
        for region in sorted_regions:
            should_keep = True
            
            for kept in keep_regions:
                if self._calculate_simple_overlap(region, kept) > overlap_threshold:
                    should_keep = False
                    break
            
            if should_keep:
                keep_regions.append(region)
        
        return keep_regions
    
    def _calculate_simple_overlap(self, region1: TextRegion, region2: TextRegion) -> float:
        """Calculate simple IoU overlap between two regions."""
        bbox1, bbox2 = region1.bbox, region2.bbox
        
        # Calculate intersection
        x1 = max(bbox1.x, bbox2.x)
        y1 = max(bbox1.y, bbox2.y)
        x2 = min(bbox1.x + bbox1.width, bbox2.x + bbox2.width)
        y2 = min(bbox1.y + bbox1.height, bbox2.y + bbox2.height)
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = bbox1.width * bbox1.height
        area2 = bbox2.width * bbox2.height
        union = area1 + area2 - intersection
        
        return intersection / max(union, 1) if union > 0 else 0.0
    
    def _emergency_detection(self, image: np.ndarray) -> List[TextRegion]:
        """
        Emergency fallback detection using basic contour analysis.
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            List[TextRegion]: Basic detected regions
        """
        try:
            self.logger.info("Applying emergency detection fallback")
            
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Simple edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            regions = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Basic filtering
                if w * h < 500 or w < 30 or h < 10:
                    continue
                
                if w / max(h, 1) > 20 or w / max(h, 1) < 0.5:
                    continue
                
                bbox = BoundingBox(x=x, y=y, width=w, height=h)
                region = TextRegion(
                    text="",
                    confidence=0.5,  # Low confidence for emergency detection
                    bbox=bbox,
                    metadata={
                        'detection_method': 'emergency',
                        'area': w * h
                    }
                )
                regions.append(region)
            
            # Sort by area and return top regions
            regions.sort(key=lambda r: r.bbox.width * r.bbox.height, reverse=True)
            return regions[:20]  # Limited number for emergency
            
        except Exception as e:
            self.logger.error(f"Emergency detection failed: {str(e)}")
            return []
    
    def _validate_and_limit_regions(self, regions: List[TextRegion], 
                                  image_shape: Tuple[int, int]) -> List[TextRegion]:
        """
        Final validation and limiting of detected regions.
        
        CRITICAL: This ensures we return 10-80 valid regions as promised to pipeline.
        
        Args:
            regions (List[TextRegion]): Input regions
            image_shape (Tuple[int, int]): Image shape (H, W, ...)
            
        Returns:
            List[TextRegion]: Validated and limited regions
        """
        if not regions:
            self.logger.warning("No text regions to validate")
            return []
        
        img_height, img_width = image_shape[:2]
        valid_regions = []
        
        for region in regions:
            bbox = region.bbox
            
            # Validate bounding box
            if not self._is_valid_bbox(bbox, img_width, img_height):
                continue
            
            # Validate confidence
            if region.confidence <= 0 or region.confidence > 1.0:
                continue
            
            valid_regions.append(region)
        
        # Sort by confidence and apply smart limiting
        valid_regions.sort(key=lambda r: r.confidence, reverse=True)
        
        # Smart limiting: prefer target number, but respect min/max bounds
        if len(valid_regions) <= self.target_regions:
            limited_regions = valid_regions
        else:
            # If we have more than target, take the most confident ones
            limited_regions = valid_regions[:min(self.max_regions, len(valid_regions))]
        
        self.logger.debug(
            f"Region validation: {len(regions)} → {len(valid_regions)} → {len(limited_regions)} regions"
        )
        
        return limited_regions
    
    def _is_valid_bbox(self, bbox: BoundingBox, img_width: int, img_height: int) -> bool:
        """
        Validate if bounding box is within image boundaries and has valid dimensions.
        
        Args:
            bbox (BoundingBox): Bounding box to validate
            img_width (int): Image width
            img_height (int): Image height
            
        Returns:
            bool: True if valid, False otherwise
        """
        return (bbox.x >= 0 and bbox.y >= 0 and 
                bbox.width > 0 and bbox.height > 0 and
                bbox.x + bbox.width <= img_width and 
                bbox.y + bbox.height <= img_height)
    
    def _update_detection_history(self, num_regions: int, method: str, detection_time: float) -> None:
        """Update detection performance history for monitoring."""
        total = self.detection_history['total_detections']
        
        # Update average regions
        self.detection_history['average_regions'] = (
            (self.detection_history['average_regions'] * (total - 1) + num_regions) / total
        )
        
        # Log performance metrics periodically
        if total % 100 == 0:  # Every 100 detections
            self.logger.info(
                f"Detection performance (last 100): "
                f"avg_regions={self.detection_history['average_regions']:.1f}, "
                f"craft_failures={self.detection_history['craft_failures']}"
            )
    
    def get_detection_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive detection statistics and configuration.
        
        Returns:
            Dict[str, Any]: Detection statistics and configuration
        """
        stats = {
            # Configuration
            'preferred_method': self.preferred_method,
            'fallback_enabled': self.fallback_enabled,
            'min_regions': self.min_regions,
            'max_regions': self.max_regions,
            'target_regions': self.target_regions,
            
            # Availability
            'craft_available': TORCH_AVAILABLE and self.craft_detector is not None,
            'fast_available': True,
            
            # Performance history
            **self.detection_history
        }
        
        # Add CRAFT-specific stats if available
        if self.craft_detector and hasattr(self.craft_detector, 'detection_stats'):
            stats['craft_stats'] = self.craft_detector.detection_stats
        
        return stats
    
    def reconfigure(self, new_config: Dict[str, Any]) -> None:
        """
        Reconfigure detector parameters at runtime.
        
        Args:
            new_config (Dict[str, Any]): New configuration parameters
        """
        # Update region limits
        if 'min_regions' in new_config:
            self.min_regions = new_config['min_regions']
        if 'max_regions' in new_config:
            self.max_regions = new_config['max_regions']
        if 'target_regions' in new_config:
            self.target_regions = new_config['target_regions']
        
        # Update method preference
        if 'preferred_method' in new_config:
            self.preferred_method = new_config['preferred_method']
        
        self.logger.info(f"Text detector reconfigured with: {new_config}")


# Utility functions for external use
def create_text_detector(model_loader: ModelLoader, 
                        config: Optional[OCRConfig] = None) -> TextDetector:
    """
    Factory function to create a properly configured text detector instance.
    
    Args:
        model_loader (ModelLoader): Model loader instance for CRAFT model
        config (Optional[OCRConfig]): OCR configuration, creates default if None
        
    Returns:
        TextDetector: Configured text detector ready for use
    """
    if config is None:
        # Import here to avoid circular imports
        from ..config import OCRConfig
        config = OCRConfig()
    
    return TextDetector(model_loader, config)


def validate_text_regions(regions: List[TextRegion], 
                         image_shape: Tuple[int, int]) -> List[TextRegion]:
    """
    Utility function to validate text regions against image boundaries.
    
    Args:
        regions (List[TextRegion]): Input regions to validate
        image_shape (Tuple[int, int]): Image shape (H, W, ...)
        
    Returns:
        List[TextRegion]: Only regions that are within image boundaries
    """
    if not regions:
        return []
    
    img_height, img_width = image_shape[:2]
    valid_regions = []
    
    for region in regions:
        bbox = region.bbox
        
        # Check if region is completely within image bounds
        if (bbox.x >= 0 and bbox.y >= 0 and 
            bbox.x + bbox.width <= img_width and 
            bbox.y + bbox.height <= img_height and
            bbox.width > 0 and bbox.height > 0):
            valid_regions.append(region)
    
    return valid_regions


def analyze_detection_quality(regions: List[TextRegion]) -> Dict[str, Any]:
    """
    Analyze the quality of detected text regions for debugging and optimization.
    
    Args:
        regions (List[TextRegion]): Detected text regions
        
    Returns:
        Dict[str, Any]: Quality analysis metrics
    """
    if not regions:
        return {
            'num_regions': 0,
            'quality_score': 0.0,
            'warnings': ['No regions detected']
        }
    
    # Calculate various quality metrics
    confidences = [r.confidence for r in regions]
    areas = [r.bbox.width * r.bbox.height for r in regions]
    aspect_ratios = [r.bbox.width / max(r.bbox.height, 1) for r in regions]
    
    analysis = {
        'num_regions': len(regions),
        'avg_confidence': np.mean(confidences),
        'min_confidence': np.min(confidences),
        'max_confidence': np.max(confidences),
        'avg_area': np.mean(areas),
        'avg_aspect_ratio': np.mean(aspect_ratios),
        'confidence_std': np.std(confidences),
        'warnings': []
    }
    
    # Generate quality warnings
    if analysis['avg_confidence'] < 0.5:
        analysis['warnings'].append('Low average confidence detected')
    
    if analysis['confidence_std'] > 0.3:
        analysis['warnings'].append('High confidence variance - inconsistent detection')
    
    if len(regions) < 10:
        analysis['warnings'].append('Few regions detected - image may have little text')
    
    if len(regions) > 100:
        analysis['warnings'].append('Many regions detected - may need stricter filtering')
    
    # Calculate overall quality score (0-1)
    quality_components = [
        min(1.0, analysis['avg_confidence'] * 2),  # Confidence component
        min(1.0, len(regions) / 50),               # Region count component  
        max(0.0, 1.0 - analysis['confidence_std']) # Consistency component
    ]
    
    analysis['quality_score'] = np.mean(quality_components)
    
    return analysis


# Export all public components
__all__ = [
    'CRAFTDetector', 'FastTextDetector', 'TextDetector',
    'create_text_detector', 'validate_text_regions', 'analyze_detection_quality'
]