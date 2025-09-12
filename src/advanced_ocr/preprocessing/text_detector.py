# src/preprocessing/text_detector.py - FIXED VERSION
import cv2
import numpy as np
import time
import logging
import multiprocessing
from typing import List, Dict, Any, Tuple, Optional
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import your base classes
from ..core.base_engine import BoundingBox, TextRegion, TextType

class DetectionMethod(Enum):
    """Text detection methods"""
    AUTO = "auto"
    TRADITIONAL = "traditional"
    MSER = "mser"
    CRAFT = "craft"
    EAST = "east"
    HYBRID = "hybrid"

# Check if deep learning libraries are available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

class TraditionalDetector:
    """Traditional text detection methods"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("TraditionalDetector")
    
    def detect_mser(self, image: np.ndarray) -> List[TextRegion]:
        """MSER-based text detection - FIXED to return proper TextRegion objects"""
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Create MSER detector with correct parameter names
            try:
                # Try new parameter names first (OpenCV 4.x)
                mser = cv2.MSER_create(
                    min_area=30,
                    max_area=14400,
                    max_variation=0.25,
                    min_diversity=0.2,
                    max_evolution=200,
                    area_threshold=1.01,
                    min_margin=0.003,
                    edge_blur_size=5
                )
            except TypeError:
                # Fallback to old parameter names (OpenCV 3.x)
                try:
                    mser = cv2.MSER_create(
                        _min_area=30,
                        _max_area=14400,
                        _max_variation=0.25,
                        _min_diversity=0.2,
                        _max_evolution=200,
                        _area_threshold=1.01,
                        _min_margin=0.003,
                        _edge_blur_size=5
                    )
                except TypeError:
                    # Use default parameters if both fail
                    mser = cv2.MSER_create()
            
            # Detect regions
            regions, bboxes = mser.detectRegions(gray)
            
            # Convert to TextRegion objects with proper BoundingBox
            text_regions = []
            for bbox in bboxes:
                x, y, w, h = bbox
                
                # Create proper BoundingBox object - FIXED!
                bbox_obj = BoundingBox(
                    x=int(x), 
                    y=int(y), 
                    width=int(w), 
                    height=int(h),
                    confidence=0.7,  # Default confidence for MSER
                    text_type=TextType.PRINTED
                )
                
                # Create TextRegion with BoundingBox object - FIXED!
                region = TextRegion(
                    bbox=bbox_obj,  # Now a BoundingBox object, not tuple!
                    confidence=0.7,
                    text_type=TextType.PRINTED
                )
                
                text_regions.append(region)
            
            return text_regions
            
        except Exception as e:
            self.logger.error(f"MSER detection failed: {e}")
            return []
    
    def detect_morphological(self, image: np.ndarray) -> List[TextRegion]:
        """Morphological operations for text detection - FIXED"""
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
            
            # Threshold
            _, binary = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Convert to TextRegion objects - FIXED
            text_regions = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter small regions
                if w * h < 100:
                    continue
                
                # Create proper BoundingBox object - FIXED!
                bbox_obj = BoundingBox(
                    x=int(x), 
                    y=int(y), 
                    width=int(w), 
                    height=int(h),
                    confidence=0.6,
                    text_type=TextType.PRINTED
                )
                
                # Create TextRegion with BoundingBox object - FIXED!
                region = TextRegion(
                    bbox=bbox_obj,
                    confidence=0.6,
                    text_type=TextType.PRINTED
                )
                
                text_regions.append(region)
            
            return text_regions
            
        except Exception as e:
            self.logger.error(f"Morphological detection failed: {e}")
            return []
    
    def detect_gradient_based(self, image: np.ndarray) -> List[TextRegion]:
        """Gradient-based text detection - FIXED"""
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Sobel gradients
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            
            # Gradient magnitude
            gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
            gradient_mag = np.uint8(gradient_mag / gradient_mag.max() * 255)
            
            # Threshold
            _, binary = cv2.threshold(gradient_mag, 50, 255, cv2.THRESH_BINARY)
            
            # Morphological closing
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
            closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Convert to TextRegion objects - FIXED
            text_regions = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter regions
                if w < 10 or h < 8 or w * h < 80:
                    continue
                
                # Create proper BoundingBox object - FIXED!
                bbox_obj = BoundingBox(
                    x=int(x), 
                    y=int(y), 
                    width=int(w), 
                    height=int(h),
                    confidence=0.5,
                    text_type=TextType.PRINTED
                )
                
                # Create TextRegion with BoundingBox object - FIXED!
                region = TextRegion(
                    bbox=bbox_obj,
                    confidence=0.5,
                    text_type=TextType.PRINTED
                )
                
                text_regions.append(region)
            
            return text_regions
            
        except Exception as e:
            self.logger.error(f"Gradient-based detection failed: {e}")
            return []

class CRAFTDetector:
    """CRAFT text detector - placeholder implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("CRAFTDetector")
    
    def detect(self, image: np.ndarray) -> List[TextRegion]:
        """CRAFT detection - returns empty list if not implemented"""
        self.logger.warning("CRAFT detector not fully implemented")
        return []

class EASTDetector:
    """EAST text detector - placeholder implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("EASTDetector")
        self.model_loaded = False
    
    def load_model(self, model_path: str):
        """Load EAST model"""
        self.logger.warning("EAST model loading not implemented")
        self.model_loaded = False
    
    def detect(self, image: np.ndarray) -> List[TextRegion]:
        """EAST detection - returns empty list if not implemented"""
        self.logger.warning("EAST detector not fully implemented")
        return []

class AdvancedTextDetector:
    """Advanced text detection system - FIXED VERSION"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger("AdvancedTextDetector")
        
        # Detection parameters
        self.detection_method = DetectionMethod(self.config.get("method", "auto"))
        self.confidence_threshold = config.get("confidence_threshold", 0.6)
        self.nms_threshold = config.get("nms_threshold", 0.3)
        self.min_region_area = config.get("min_region_area", 150)
        self.max_region_area = config.get("max_region_area", 50000)
        
        # Deduplication parameters
        self.iou_threshold = config.get("iou_threshold", 0.4)
        self.merge_threshold = config.get("merge_threshold", 0.3)
        self.enable_smart_filtering = config.get("enable_smart_filtering", True)
        
        # Initialize detectors
        self.craft_detector = None
        self.east_detector = None
        self.traditional_detector = TraditionalDetector(config)
        
        # Performance tracking
        self.detection_stats = {
            'total_detections': 0,
            'method_usage': {},
            'avg_confidence': 0.0,
            'processing_times': []
        }
        
        self._initialize_deep_learning_detectors()
        self.enable_parallel = config.get("enable_parallel", True)
        self.max_workers = config.get("max_workers", min(4, multiprocessing.cpu_count()))
        self.parallel_threshold = config.get("parallel_threshold", 2000000)
        self.tile_overlap = config.get("tile_overlap", 0.1)
        
        # Reading order detection
        self.enable_reading_order = config.get("enable_reading_order", True)
        self.column_detection = config.get("column_detection", True)
        
        self.logger.info(f"Text detector initialized with {len(self._get_available_methods())} methods")
    
    def _get_available_methods(self) -> List[str]:
        """Get list of available detection methods"""
        methods = ["traditional", "mser"]
        if self.craft_detector:
            methods.append("craft")
        if self.east_detector:
            methods.append("east")
        return methods
    
    def _initialize_deep_learning_detectors(self):
        """Initialize deep learning detectors if available"""
        if TORCH_AVAILABLE and self.config.get("enable_craft", False):
            try:
                self.craft_detector = CRAFTDetector(self.config)
                self.logger.info("CRAFT detector initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize CRAFT detector: {e}")
        
        if self.config.get("enable_east", False):
            try:
                self.east_detector = EASTDetector(self.config)
                east_model_path = self.config.get("east_model_path")
                if east_model_path:
                    self.east_detector.load_model(east_model_path)
                    self.logger.info("EAST detector initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize EAST detector: {e}")
    
    def detect_text_regions(self, image: np.ndarray) -> List[TextRegion]:
        """Main text detection method - FIXED VERSION"""
        start_time = time.time()
        
        if image is None or image.size == 0:
            return []
        
        try:
            # Auto-select detection method if needed
            if self.detection_method == DetectionMethod.AUTO:
                selected_method = self._select_optimal_method(image)
            else:
                selected_method = self.detection_method
            
            # Perform detection
            regions = self._detect_with_method(image, selected_method)
            
            # Post-processing pipeline
            regions = self._modern_post_process_regions(regions, image.shape[:2])
            
            # Update statistics
            processing_time = time.time() - start_time
            self._update_stats(selected_method.value, regions, processing_time)
            
            self.logger.info(f"Detected {len(regions)} text regions using {selected_method.value} "
                           f"in {processing_time:.3f}s")
            
            return regions
            
        except Exception as e:
            self.logger.error(f"Text detection failed: {e}")
            return []
    
    def _select_optimal_method(self, image: np.ndarray) -> DetectionMethod:
        """Select optimal detection method based on image characteristics"""
        
        h, w = image.shape[:2]
        image_size = h * w
        
        # Convert to grayscale for analysis
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Calculate image characteristics
        edge_density = self._calculate_edge_density(gray)
        noise_level = self._estimate_noise_level(gray)
        contrast_level = np.std(gray)
        
        # Decision logic
        if image_size > 2000000:  # Large image
            if self.craft_detector and edge_density > 0.1:
                return DetectionMethod.CRAFT
            elif noise_level < 0.1 and contrast_level > 30:
                return DetectionMethod.MSER
            else:
                return DetectionMethod.TRADITIONAL
        
        elif noise_level > 0.2 or contrast_level < 20:  # Noisy or low contrast
            if self.craft_detector:
                return DetectionMethod.CRAFT
            else:
                return DetectionMethod.MSER
        
        else:  # Standard case
            return DetectionMethod.MSER
    
    def _calculate_edge_density(self, gray: np.ndarray) -> float:
        """Calculate edge density in the image"""
        edges = cv2.Canny(gray, 50, 150)
        return np.sum(edges > 0) / edges.size
    
    def _estimate_noise_level(self, gray: np.ndarray) -> float:
        """Estimate noise level using Laplacian variance"""
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var / (gray.shape[0] * gray.shape[1])
    
    def _detect_with_method(self, image: np.ndarray, method: DetectionMethod) -> List[TextRegion]:
        """Perform detection with specified method"""
        
        if method == DetectionMethod.CRAFT and self.craft_detector:
            return self.craft_detector.detect(image)
        
        elif method == DetectionMethod.EAST and self.east_detector:
            return self.east_detector.detect(image)
        
        elif method == DetectionMethod.MSER:
            return self.traditional_detector.detect_mser(image)
        
        elif method == DetectionMethod.TRADITIONAL:
            # Combine multiple traditional methods
            mser_regions = self.traditional_detector.detect_mser(image)
            morph_regions = self.traditional_detector.detect_morphological(image)
            grad_regions = self.traditional_detector.detect_gradient_based(image)
            
            all_regions = mser_regions + morph_regions + grad_regions
            return self._merge_similar_regions(all_regions)
        
        else:
            # Fallback to MSER
            return self.traditional_detector.detect_mser(image)
    
    def _modern_post_process_regions(self, regions: List[TextRegion], 
                                   image_shape: Tuple[int, int]) -> List[TextRegion]:
        """Modern post-processing pipeline"""
        
        if not regions:
            return regions
        
        # Stage 1: Validate bounding boxes
        validated_regions = self._validate_bounding_boxes(regions, image_shape)
        
        # Stage 2: Smart filtering
        if self.enable_smart_filtering:
            filtered_regions = self._smart_filter_regions(validated_regions)
        else:
            filtered_regions = validated_regions
        
        # Stage 3: Merge similar regions
        merged_regions = self._merge_similar_regions(filtered_regions)
        
        # Stage 4: Final quality filter
        final_regions = self._final_quality_filter(merged_regions)
        
        # Stage 5: Sort by reading order
        final_regions.sort(key=lambda r: (r.bbox.y, r.bbox.x))
        
        return final_regions
    
    def _validate_bounding_boxes(self, regions: List[TextRegion], 
                               image_shape: Tuple[int, int]) -> List[TextRegion]:
        """Validate and normalize bounding boxes - FIXED"""
        h, w = image_shape
        validated_regions = []
        
        for region in regions:
            # Access BoundingBox properties correctly - FIXED!
            x, y, rw, rh = region.bbox.x, region.bbox.y, region.bbox.width, region.bbox.height
            
            # Clamp coordinates to image bounds
            x = max(0, min(x, w - 1))
            y = max(0, min(y, h - 1))
            rw = max(1, min(rw, w - x))
            rh = max(1, min(rh, h - y))
            
            # Only keep regions with positive dimensions
            if rw > 0 and rh > 0:
                # Create new BoundingBox object - FIXED!
                bbox_obj = BoundingBox(
                    x=x, y=y, width=rw, height=rh,
                    confidence=region.bbox.confidence,
                    text_type=region.bbox.text_type
                )
                
                validated_region = TextRegion(
                    bbox=bbox_obj,
                    confidence=region.confidence,
                    text_type=region.text_type
                )
                validated_regions.append(validated_region)
        
        return validated_regions
    
    def _smart_filter_regions(self, regions: List[TextRegion]) -> List[TextRegion]:
        """Smart filtering to remove noise"""
        filtered = []
        
        for region in regions:
            # Use BoundingBox properties correctly - FIXED!
            area = region.bbox.area
            aspect_ratio = region.bbox.aspect_ratio
            w, h = region.bbox.width, region.bbox.height
            
            # Basic size filtering
            if area < self.min_region_area or area > self.max_region_area:
                continue
            
            # Confidence filtering
            if region.confidence < self.confidence_threshold:
                continue
            
            # Aspect ratio filtering
            if aspect_ratio < 0.05 or aspect_ratio > 50:
                continue
            
            # Dimension filtering
            if w < 8 or h < 6:
                continue
            
            filtered.append(region)
        
        return filtered
    
    def _merge_similar_regions(self, regions: List[TextRegion]) -> List[TextRegion]:
        """Merge overlapping and similar regions"""
        if len(regions) <= 1:
            return regions
        
        merged = []
        used_indices = set()
        
        for i, region in enumerate(regions):
            if i in used_indices:
                continue
            
            # Find overlapping regions
            merge_candidates = [region]
            used_indices.add(i)
            
            for j, other_region in enumerate(regions[i+1:], i+1):
                if j in used_indices:
                    continue
                
                # Calculate IoU using BoundingBox methods - FIXED!
                iou = region.bbox.iou(other_region.bbox)
                
                if iou > self.iou_threshold:
                    merge_candidates.append(other_region)
                    used_indices.add(j)
            
            # Create merged region
            if len(merge_candidates) == 1:
                merged.append(region)
            else:
                merged_region = self._merge_region_group(merge_candidates)
                merged.append(merged_region)
        
        return merged
    
    def _merge_region_group(self, regions: List[TextRegion]) -> TextRegion:
        """Merge a group of regions into one - FIXED"""
        # Calculate combined bounding box
        min_x = min(r.bbox.x for r in regions)
        min_y = min(r.bbox.y for r in regions)
        max_x = max(r.bbox.x + r.bbox.width for r in regions)
        max_y = max(r.bbox.y + r.bbox.height for r in regions)
        
        # Calculate average confidence
        avg_confidence = sum(r.confidence for r in regions) / len(regions)
        
        # Create merged BoundingBox - FIXED!
        merged_bbox = BoundingBox(
            x=min_x,
            y=min_y,
            width=max_x - min_x,
            height=max_y - min_y,
            confidence=avg_confidence,
            text_type=regions[0].bbox.text_type
        )
        
        # Create merged TextRegion - FIXED!
        return TextRegion(
            bbox=merged_bbox,
            confidence=avg_confidence,
            text_type=regions[0].text_type
        )
    
    def _final_quality_filter(self, regions: List[TextRegion]) -> List[TextRegion]:
        """Final quality filtering"""
        # For now, just return all regions that passed previous filters
        return regions
    
    def _update_stats(self, method: str, regions: List[TextRegion], processing_time: float):
        """Update detection statistics"""
        self.detection_stats['total_detections'] += 1
        self.detection_stats['processing_times'].append(processing_time)
        
        if method not in self.detection_stats['method_usage']:
            self.detection_stats['method_usage'][method] = 0
        self.detection_stats['method_usage'][method] += 1
        
        if regions:
            avg_conf = sum(r.confidence for r in regions) / len(regions)
            current_avg = self.detection_stats['avg_confidence']
            total = self.detection_stats['total_detections']
            self.detection_stats['avg_confidence'] = ((current_avg * (total - 1)) + avg_conf) / total
    
    def get_stats(self) -> Dict[str, Any]:
        """Get detection statistics"""
        stats = self.detection_stats.copy()
        if stats['processing_times']:
            stats['avg_processing_time'] = sum(stats['processing_times']) / len(stats['processing_times'])
        return stats

# Export the main class
__all__ = ['AdvancedTextDetector', 'DetectionMethod', 'TextRegion', 'BoundingBox']