# src/preprocessing/text_detector.py

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Union
import logging
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import time
import warnings
import json
warnings.filterwarnings("ignore")

# Deep learning imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Deep learning text detection disabled.")

try:
    import tensorflow as tf  # type: ignore
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# Traditional CV imports
from skimage import filters, morphology, measure, restoration
from scipy import ndimage

class DetectionMethod(Enum):
    """Available text detection methods"""
    TRADITIONAL = "traditional"
    CRAFT = "craft"
    EAST = "east"
    MSER = "mser"
    HYBRID = "hybrid"
    AUTO = "auto"

@dataclass
class TextRegion:
    """Detected text region with confidence and metadata"""
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    confidence: float
    angle: float = 0.0
    text_type: str = "unknown"  # printed, handwritten, mixed
    method: str = "unknown"
    
    @property
    def center(self) -> Tuple[int, int]:
        x, y, w, h = self.bbox
        return (x + w // 2, y + h // 2)
    
    @property
    def area(self) -> int:
        return self.bbox[2] * self.bbox[3]
    
    @property
    def aspect_ratio(self) -> float:
        w, h = self.bbox[2], self.bbox[3]
        return w / h if h > 0 else 0

class CRAFTDetector:
    """CRAFT (Character Region Awareness for Text) detector implementation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() and config.get("cuda", True) else "cpu"
        
        # CRAFT parameters
        self.text_threshold = config.get("text_threshold", 0.7)
        self.link_threshold = config.get("link_threshold", 0.4) 
        self.low_text = config.get("low_text", 0.4)
        self.canvas_size = config.get("canvas_size", 1280)
        self.mag_ratio = config.get("mag_ratio", 1.5)
        
        self.logger = logging.getLogger("TextDetector.CRAFT")
        
    def load_model(self, model_path: Optional[str] = None) -> bool:
        """Load CRAFT model"""
        if not TORCH_AVAILABLE:
            self.logger.error("PyTorch not available for CRAFT detection")
            return False
        
        try:
            # Load pre-trained CRAFT model
            if model_path and Path(model_path).exists():
                self.model = torch.load(model_path, map_location=self.device)
            else:
                # Use a simplified CRAFT-like model for demonstration
                self.model = self._create_simple_craft_model()
            
            self.model.to(self.device)
            self.model.eval()
            
            self.logger.info("CRAFT model loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load CRAFT model: {e}")
            return False
    
    def _create_simple_craft_model(self):
        """Create a simplified CRAFT-like model"""
        class SimpleCRAFT(nn.Module):
            def __init__(self):
                super(SimpleCRAFT, self).__init__()
                # Simplified architecture - in practice, use pre-trained CRAFT
                self.backbone = nn.Sequential(
                    nn.Conv2d(3, 64, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(128, 256, 3, padding=1),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((32, 32))
                )
                self.text_head = nn.Conv2d(256, 1, 1)
                self.link_head = nn.Conv2d(256, 1, 1)
            
            def forward(self, x):
                features = self.backbone(x)
                text_map = torch.sigmoid(self.text_head(features))
                link_map = torch.sigmoid(self.link_head(features))
                return text_map, link_map
        
        return SimpleCRAFT()
    
    def detect(self, image: np.ndarray) -> List[TextRegion]:
        """Detect text regions using CRAFT"""
        if self.model is None:
            if not self.load_model():
                return []
        
        try:
            # Preprocess image
            processed_img = self._preprocess_image(image)
            
            # Inference
            with torch.no_grad():
                text_map, link_map = self.model(processed_img)
            
            # Post-process predictions
            regions = self._postprocess_predictions(
                text_map.cpu().numpy()[0, 0],
                link_map.cpu().numpy()[0, 0],
                image.shape[:2]
            )
            
            return regions
            
        except Exception as e:
            self.logger.error(f"CRAFT detection failed: {e}")
            return []
    
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for CRAFT model"""
        # Resize image
        h, w = image.shape[:2]
        target_size = self.canvas_size
        
        # Calculate resize ratio
        ratio = min(target_size / w, target_size / h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        
        # Resize
        resized = cv2.resize(image, (new_w, new_h))
        
        # Pad to target size
        padded = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        padded[:new_h, :new_w] = resized
        
        # Convert to tensor
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        tensor = transform(padded).unsqueeze(0).to(self.device)
        return tensor
    
    def _postprocess_predictions(self, text_map: np.ndarray, link_map: np.ndarray,
                               original_shape: Tuple[int, int]) -> List[TextRegion]:
        """Post-process CRAFT predictions to get bounding boxes"""
        
        # Threshold text regions
        text_mask = text_map > self.text_threshold
        
        # Find connected components
        labeled = measure.label(text_mask)
        regions = []
        
        for region_props in measure.regionprops(labeled):
            if region_props.area < 100:  # Filter small regions
                continue
            
            # Get bounding box
            min_row, min_col, max_row, max_col = region_props.bbox
            
            # Scale back to original image size
            h_ratio = original_shape[0] / text_map.shape[0]
            w_ratio = original_shape[1] / text_map.shape[1]
            
            x = int(min_col * w_ratio)
            y = int(min_row * h_ratio)
            w = int((max_col - min_col) * w_ratio)
            h = int((max_row - min_row) * h_ratio)
            
            # Calculate confidence based on average text score
            mask = labeled == region_props.label
            confidence = float(np.mean(text_map[mask]))
            
            regions.append(TextRegion(
                bbox=(x, y, w, h),
                confidence=confidence,
                method="craft"
            ))
        
        return regions

class EASTDetector:
    """EAST (Efficient and Accurate Scene Text) detector"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.net = None
        
        # EAST parameters
        self.confidence_threshold = config.get("confidence_threshold", 0.5)
        self.nms_threshold = config.get("nms_threshold", 0.4)
        self.input_width = config.get("input_width", 320)
        self.input_height = config.get("input_height", 320)
        
        self.logger = logging.getLogger("TextDetector.EAST")
    
    def load_model(self, model_path: str) -> bool:
        """Load EAST model"""
        try:
            if Path(model_path).exists():
                self.net = cv2.dnn.readNet(model_path)
                self.logger.info("EAST model loaded successfully")
                return True
            else:
                self.logger.error(f"EAST model not found: {model_path}")
                return False
        except Exception as e:
            self.logger.error(f"Failed to load EAST model: {e}")
            return False
    
    def detect(self, image: np.ndarray) -> List[TextRegion]:
        """Detect text regions using EAST"""
        if self.net is None:
            self.logger.error("EAST model not loaded")
            return []
        
        try:
            # Create blob from image
            blob = cv2.dnn.blobFromImage(
                image, 1.0, 
                (self.input_width, self.input_height),
                (123.68, 116.78, 103.94), 
                swapRB=True, crop=False
            )
            
            # Set input to the network
            self.net.setInput(blob)
            
            # Run inference
            scores, geometry = self.net.forward(["feature_fusion/Conv_7/Sigmoid", 
                                               "feature_fusion/concat_3"])
            
            # Decode predictions
            regions = self._decode_predictions(scores, geometry, image.shape[:2])
            
            return regions
            
        except Exception as e:
            self.logger.error(f"EAST detection failed: {e}")
            return []
    
    def _decode_predictions(self, scores: np.ndarray, geometry: np.ndarray,
                          original_shape: Tuple[int, int]) -> List[TextRegion]:
        """Decode EAST predictions"""
        
        # Extract dimensions
        num_rows, num_cols = scores.shape[2:4]
        
        # Calculate ratios
        ratio_x = original_shape[1] / float(num_cols)
        ratio_y = original_shape[0] / float(num_rows)
        
        rectangles = []
        confidences = []
        
        # Loop over the number of rows and columns
        for y in range(num_rows):
            scores_data = scores[0, 0, y]
            x_data0 = geometry[0, 0, y]
            x_data1 = geometry[0, 1, y]
            x_data2 = geometry[0, 2, y]
            x_data3 = geometry[0, 3, y]
            angles_data = geometry[0, 4, y]
            
            for x in range(num_cols):
                if scores_data[x] < self.confidence_threshold:
                    continue
                
                # Calculate offset
                offset_x = x * 4.0
                offset_y = y * 4.0
                
                # Extract rotation angle and calculate cos/sin
                angle = angles_data[x]
                cos = np.cos(angle)
                sin = np.sin(angle)
                
                # Calculate dimensions
                h = x_data0[x] + x_data2[x]
                w = x_data1[x] + x_data3[x]
                
                # Calculate rotated rectangle
                end_x = int(offset_x + (cos * x_data1[x]) + (sin * x_data2[x]))
                end_y = int(offset_y - (sin * x_data1[x]) + (cos * x_data2[x]))
                start_x = int(end_x - w)
                start_y = int(end_y - h)
                
                # Scale coordinates
                start_x = int(start_x * ratio_x)
                start_y = int(start_y * ratio_y)
                end_x = int(end_x * ratio_x)
                end_y = int(end_y * ratio_y)
                
                rectangles.append((start_x, start_y, end_x - start_x, end_y - start_y))
                confidences.append(float(scores_data[x]))
        
        # Apply non-maximum suppression
        indices = cv2.dnn.NMSBoxes(rectangles, confidences, 
                                 self.confidence_threshold, self.nms_threshold)
        
        regions = []
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w, h = rectangles[i]
                regions.append(TextRegion(
                    bbox=(x, y, w, h),
                    confidence=confidences[i],
                    method="east"
                ))
        
        return regions

class TraditionalDetector:
    """Traditional text detection methods (MSER, contours, morphological)"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.min_text_size = config.get("min_text_size", 10)
        self.max_text_size = config.get("max_text_size", 300)
        self.logger = logging.getLogger("TextDetector.Traditional")
    
    def _create_mser_detector(self):
        """Helper to create a configured MSER detector instance"""
        try:
            # Use the consistent, modern MSER API with error handling for OpenCV versions
            if hasattr(cv2, 'MSER_create'):
                mser = cv2.MSER_create()
                # Set parameters with error handling for different OpenCV versions
                try:
                    mser.setMinArea(self.min_text_size)
                except AttributeError:
                    pass
                try:
                    mser.setMaxArea(self.max_text_size * 100)
                except AttributeError:
                    pass
                try:
                    mser.setMaxVariation(0.25)
                except AttributeError:
                    pass
                try:
                    mser.setMinDiversity(0.2)
                except AttributeError:
                    pass
                try:
                    mser.setMaxEvolution(200)
                except AttributeError:
                    pass
                try:
                    mser.setAreaThreshold(1.01)
                except AttributeError:
                    pass
                try:
                    mser.setMinMargin(0.003)
                except AttributeError:
                    pass
                try:
                    mser.setEdgeBlurSize(5)
                except AttributeError:
                    pass
                return mser
            else:
                self.logger.warning("MSER_create not available in cv2, using fallback detection")
                return None
        except Exception as e:
            self.logger.warning(f"MSER_create initialization failed: {e}")
            return None

    def detect_mser(self, image: np.ndarray) -> List[TextRegion]:
        """Detect text using MSER (Maximally Stable Extremal Regions) - FIXED VERSION"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        try:
            mser = self._create_mser_detector()
            regions, bboxes = mser.detectRegions(gray)

            detected_regions = []
            for i, bbox in enumerate(bboxes):
                x, y, w, h = bbox

                # Filter based on aspect ratio and size
                aspect_ratio = w / h if h > 0 else 0
                if (0.1 < aspect_ratio < 15 and
                    w > self.min_text_size and h > self.min_text_size and
                    w < self.max_text_size and h < self.max_text_size):

                    # Ensure bbox is within image bounds
                    x = max(0, min(x, gray.shape[1] - 1))
                    y = max(0, min(y, gray.shape[0] - 1))
                    w = min(w, gray.shape[1] - x)
                    h = min(h, gray.shape[0] - y)

                    if w > 0 and h > 0:
                        confidence = self._estimate_mser_confidence(regions[i] if i < len(regions) else None, gray[y:y+h, x:x+w])

                        detected_regions.append(TextRegion(
                            bbox=(x, y, w, h),
                            confidence=confidence,
                            method="mser"
                        ))

            self.logger.info(f"MSER detected {len(detected_regions)} regions")
            return detected_regions

        except Exception as e:
            self.logger.error(f"MSER detection failed: {e}")
            return []

    def _fallback_detection(self, gray: np.ndarray) -> List[TextRegion]:
        """Fallback detection when MSER fails"""
        detected_regions = []
        
        try:
            # Method 1: Adaptive thresholding + contours
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY, 11, 2)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filter reasonable text regions
                if (w > 15 and h > 8 and w < 500 and h < 100):
                    aspect_ratio = w / h
                    if 0.2 < aspect_ratio < 20:
                        detected_regions.append(TextRegion(
                            bbox=(x, y, w, h),
                            confidence=0.6,
                            method="fallback_contour"
                        ))
            
            # Method 2: If still no regions, create grid regions
            if not detected_regions:
                h, w = gray.shape[:2]
                # Create reasonable text line regions
                line_height = 30
                num_lines = h // line_height
                
                for i in range(num_lines):
                    y = i * line_height
                    detected_regions.append(TextRegion(
                        bbox=(10, y, w-20, line_height),
                        confidence=0.4,
                        method="fallback_grid"
                    ))
            
            self.logger.info(f"Fallback detection found {len(detected_regions)} regions")
            return detected_regions
            
        except Exception as e:
            self.logger.error(f"Fallback detection also failed: {e}")
            return []
    
    def detect_morphological(self, image: np.ndarray) -> List[TextRegion]:
        """Detect text using morphological operations"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
        morphed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter regions
            if (w > self.min_text_size and h > self.min_text_size and
                w < self.max_text_size and h < self.max_text_size):
                
                # Calculate confidence based on contour properties
                confidence = self._estimate_contour_confidence(contour, gray[y:y+h, x:x+w])
                
                detected_regions.append(TextRegion(
                    bbox=(x, y, w, h),
                    confidence=confidence,
                    method="morphological"
                ))
        
        return detected_regions
    
    def detect_gradient_based(self, image: np.ndarray) -> List[TextRegion]:
        """Detect text using gradient-based methods"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Calculate gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Calculate gradient magnitude
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        grad_magnitude = np.uint8(grad_magnitude / grad_magnitude.max() * 255)
        
        # Threshold and morphological operations
        _, binary = cv2.threshold(grad_magnitude, 50, 255, cv2.THRESH_BINARY)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
        morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter regions
            aspect_ratio = w / h if h > 0 else 0
            if (w > self.min_text_size and h > self.min_text_size and
                0.2 < aspect_ratio < 20):
                
                confidence = self._estimate_gradient_confidence(grad_magnitude[y:y+h, x:x+w])
                
                detected_regions.append(TextRegion(
                    bbox=(x, y, w, h),
                    confidence=confidence,
                    method="gradient"
                ))
        
        return detected_regions
    
    def _estimate_mser_confidence(self, region: np.ndarray, roi: np.ndarray) -> float:
        """Estimate confidence for MSER region"""
        base_confidence = 0.7
        
        # Adjust based on region size
        area = cv2.contourArea(region)
        if 100 < area < 10000:
            base_confidence *= 1.1
        
        # Adjust based on intensity variation
        if roi.size > 0:
            std_dev = np.std(roi)
            if std_dev > 30:  # Good contrast
                base_confidence *= 1.2
        
        return min(0.95, base_confidence)
    
    def _estimate_contour_confidence(self, contour: np.ndarray, roi: np.ndarray) -> float:
        """Estimate confidence for contour-based detection"""
        base_confidence = 0.6
        
        # Adjust based on contour properties
        perimeter = cv2.arcLength(contour, True)
        area = cv2.contourArea(contour)
        
        if area > 0:
            compactness = (perimeter * perimeter) / (4 * np.pi * area)
            if 1.0 < compactness < 4.0:  # Reasonable shape
                base_confidence *= 1.3
        
        return min(0.9, base_confidence)
    
    def _estimate_gradient_confidence(self, grad_roi: np.ndarray) -> float:
        """Estimate confidence for gradient-based detection"""
        base_confidence = 0.5
        
        if grad_roi.size > 0:
            mean_gradient = np.mean(grad_roi)
            if mean_gradient > 50:  # Strong edges
                base_confidence *= 1.4
        
        return min(0.85, base_confidence)

class AdvancedTextDetector:
    """Advanced text detection system combining multiple methods"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger("AdvancedTextDetector")
        
        # Detection parameters
        self.detection_method = DetectionMethod(config.get("method", "auto"))
        self.confidence_threshold = config.get("confidence_threshold", 0.5)
        self.nms_threshold = config.get("nms_threshold", 0.4)
        self.min_region_area = config.get("min_region_area", 100)
        self.max_region_area = config.get("max_region_area", 100000)
        
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
    
    def _initialize_deep_learning_detectors(self):
        """Initialize deep learning detectors if available"""
        if TORCH_AVAILABLE and self.config.get("enable_craft", True):
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
        """Main text detection method with intelligent method selection"""
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
            
            # Post-process regions
            regions = self._post_process_regions(regions, image.shape[:2])
            
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
        """Automatically select the best detection method based on image characteristics"""
        
        # Analyze image characteristics
        h, w = image.shape[:2]
        image_size = h * w
        
        # Convert to grayscale for analysis
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Calculate image complexity metrics
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
                return DetectionMethod.HYBRID
        
        elif noise_level > 0.2 or contrast_level < 20:  # Noisy or low contrast
            if self.craft_detector:
                return DetectionMethod.CRAFT
            else:
                return DetectionMethod.MSER
        
        else:  # Standard case
            if self.craft_detector:
                return DetectionMethod.CRAFT
            else:
                return DetectionMethod.HYBRID
    
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
        
        elif method == DetectionMethod.HYBRID:
            # Combine deep learning and traditional methods
            all_regions = []
            
            # Try CRAFT first
            if self.craft_detector:
                craft_regions = self.craft_detector.detect(image)
                all_regions.extend(craft_regions)
            
            # Add MSER regions
            mser_regions = self.traditional_detector.detect_mser(image)
            all_regions.extend(mser_regions)
            
            # Merge and filter
            return self._merge_similar_regions(all_regions)
        
        else:
            # Fallback to MSER
            return self.traditional_detector.detect_mser(image)
    
    def _post_process_regions(self, regions: List[TextRegion], 
                            image_shape: Tuple[int, int]) -> List[TextRegion]:
        """Post-process detected regions"""
        
        if not regions:
            return regions
        
        # Filter by confidence
        regions = [r for r in regions if r.confidence >= self.confidence_threshold]
        
        # Filter by area
        regions = [r for r in regions 
                  if self.min_region_area <= r.area <= self.max_region_area]
        
        # Validate bounding boxes
        h, w = image_shape
        validated_regions = []
        
        for region in regions:
            x, y, rw, rh = region.bbox
            
            # Ensure coordinates are within image bounds
            x = max(0, min(x, w - 1))
            y = max(0, min(y, h - 1))
            rw = max(1, min(rw, w - x))
            rh = max(1, min(rh, h - y))
            
            # Update region with validated coordinates
            validated_region = TextRegion(
                bbox=(x, y, rw, rh),
                confidence=region.confidence,
                angle=region.angle,
                text_type=region.text_type,
                method=region.method
            )
            
            validated_regions.append(validated_region)
        
        # Apply non-maximum suppression
        if len(validated_regions) > 1:
            validated_regions = self._apply_nms(validated_regions)
        
        # Sort by reading order (top-to-bottom, left-to-right)
        validated_regions.sort(key=lambda r: (r.bbox[1], r.bbox[0]))
        
        return validated_regions
    
    def _merge_similar_regions(self, regions: List[TextRegion]) -> List[TextRegion]:
        """Merge similar/overlapping regions from multiple methods"""
        if len(regions) <= 1:
            return regions
        
        # Group regions by overlap
        merged_regions = []
        used_indices = set()
        
        for i, region1 in enumerate(regions):
            if i in used_indices:
                continue
            
            # Find overlapping regions
            overlap_group = [region1]
            used_indices.add(i)
            
            for j, region2 in enumerate(regions[i+1:], i+1):
                if j in used_indices:
                    continue
                
                if self._regions_overlap(region1, region2):
                    overlap_group.append(region2)
                    used_indices.add(j)
            
            # Merge overlapping regions
            if len(overlap_group) > 1:
                merged_region = self._merge_region_group(overlap_group)
                merged_regions.append(merged_region)
            else:
                merged_regions.append(region1)
        
        return merged_regions
    
    def _merge_region_group(self, regions: List[TextRegion]) -> TextRegion:
        """Merge a group of overlapping regions"""
        if len(regions) == 1:
            return regions[0]
        
        # Calculate merged bounding box
        min_x = min(r.bbox[0] for r in regions)
        min_y = min(r.bbox[1] for r in regions)
        max_x = max(r.bbox[0] + r.bbox[2] for r in regions)
        max_y = max(r.bbox[1] + r.bbox[3] for r in regions)
        
        merged_bbox = (min_x, min_y, max_x - min_x, max_y - min_y)
        
        # Calculate weighted average confidence
        total_area = sum(r.area for r in regions)
        weighted_confidence = sum(r.confidence * r.area for r in regions) / total_area
        
        # Select best method (prefer deep learning methods)
        method_priority = {"craft": 4, "east": 3, "mser": 2, "morphological": 1, "gradient": 0}
        best_method = max(regions, key=lambda r: method_priority.get(r.method, 0)).method
        
        return TextRegion(
            bbox=merged_bbox,
            confidence=weighted_confidence,
            method=f"merged_{best_method}"
        )
    
    def _regions_overlap(self, region1: TextRegion, region2: TextRegion, 
                        overlap_threshold: float = 0.3) -> bool:
        """Check if two regions overlap significantly"""
        bbox1 = region1.bbox
        bbox2 = region2.bbox
        
        # Calculate intersection
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[0] + bbox1[2], bbox2[0] + bbox2[2])
        y2 = min(bbox1[1] + bbox1[3], bbox2[1] + bbox2[3])
        
        if x2 <= x1 or y2 <= y1:
            return False
        
        # Calculate IoU (Intersection over Union)
        intersection = (x2 - x1) * (y2 - y1)
        union = region1.area + region2.area - intersection
        
        iou = intersection / union if union > 0 else 0
        return iou > overlap_threshold
    
    def _apply_nms(self, regions: List[TextRegion]) -> List[TextRegion]:
        """Apply Non-Maximum Suppression to remove duplicate detections"""
        if len(regions) <= 1:
            return regions
        
        # Convert to format expected by cv2.dnn.NMSBoxes
        boxes = [r.bbox for r in regions]
        confidences = [r.confidence for r in regions]
        
        # Apply NMS
        indices = cv2.dnn.NMSBoxes(
            boxes, confidences, 
            self.confidence_threshold, 
            self.nms_threshold
        )
        
        # Return filtered regions
        if len(indices) > 0:
            return [regions[i] for i in indices.flatten()]
        else:
            return []
    
    def _update_stats(self, method: str, regions: List[TextRegion], processing_time: float):
        """Update detection statistics"""
        self.detection_stats['total_detections'] += len(regions)
        self.detection_stats['processing_times'].append(processing_time)
        
        if method not in self.detection_stats['method_usage']:
            self.detection_stats['method_usage'][method] = 0
        self.detection_stats['method_usage'][method] += 1
        
        if regions:
            avg_conf = sum(r.confidence for r in regions) / len(regions)
            total_count = sum(self.detection_stats['method_usage'].values())
            
            # Update running average
            self.detection_stats['avg_confidence'] = (
                (self.detection_stats['avg_confidence'] * (total_count - 1) + avg_conf) / total_count
            )
    
    def get_detection_stats(self) -> Dict[str, Any]:
        """Get detection performance statistics"""
        stats = self.detection_stats.copy()
        
        if stats['processing_times']:
            stats['avg_processing_time'] = sum(stats['processing_times']) / len(stats['processing_times'])
            stats['min_processing_time'] = min(stats['processing_times'])
            stats['max_processing_time'] = max(stats['processing_times'])
        
        return stats
    
    def visualize_detections(self, image: np.ndarray, regions: List[TextRegion],
                           output_path: Optional[str] = None) -> np.ndarray:
        """Visualize detected text regions on image"""
        vis_image = image.copy()
        
        # Define colors for different methods
        method_colors = {
            'craft': (0, 255, 0),      # Green
            'east': (255, 0, 0),       # Blue  
            'mser': (0, 0, 255),       # Red
            'morphological': (255, 255, 0),  # Cyan
            'gradient': (255, 0, 255), # Magenta
            'merged': (0, 255, 255)    # Yellow
        }
        
        for i, region in enumerate(regions):
            x, y, w, h = region.bbox
            
            # Get color based on method
            color = method_colors.get(region.method.split('_')[0], (128, 128, 128))
            
            # Draw bounding box
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 2)
            
            # Draw confidence score
            confidence_text = f"{region.confidence:.2f}"
            cv2.putText(vis_image, confidence_text, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Draw region number
            cv2.putText(vis_image, str(i), (x + 5, y + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Add legend
        legend_y = 30
        for method, color in method_colors.items():
            cv2.putText(vis_image, method, (10, legend_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            legend_y += 20
        
        if output_path:
            cv2.imwrite(output_path, vis_image)
        
        return vis_image
    
    def benchmark_methods(self, test_images: List[np.ndarray],
                         ground_truth: Optional[List[List[Tuple]]] = None) -> Dict[str, Dict]:
        """Benchmark different detection methods"""
        methods_to_test = [DetectionMethod.MSER, DetectionMethod.TRADITIONAL]
        
        if self.craft_detector:
            methods_to_test.append(DetectionMethod.CRAFT)
        if self.east_detector:
            methods_to_test.append(DetectionMethod.EAST)
        
        benchmark_results = {}
        
        for method in methods_to_test:
            self.logger.info(f"Benchmarking method: {method.value}")
            
            total_regions = 0
            total_time = 0.0
            total_confidence = 0.0
            
            method_results = []
            
            for i, image in enumerate(test_images):
                start_time = time.time()
                regions = self._detect_with_method(image, method)
                processing_time = time.time() - start_time
                
                total_regions += len(regions)
                total_time += processing_time
                
                if regions:
                    avg_confidence = sum(r.confidence for r in regions) / len(regions)
                    total_confidence += avg_confidence
                
                method_results.append({
                    'regions_count': len(regions),
                    'processing_time': processing_time,
                    'avg_confidence': avg_confidence if regions else 0.0
                })
            
            # Calculate aggregate metrics
            avg_regions = total_regions / len(test_images) if test_images else 0
            avg_time = total_time / len(test_images) if test_images else 0
            avg_confidence = total_confidence / len(test_images) if test_images else 0
            
            benchmark_results[method.value] = {
                'avg_regions_per_image': avg_regions,
                'avg_processing_time': avg_time,
                'avg_confidence': avg_confidence,
                'total_processing_time': total_time,
                'detailed_results': method_results
            }
        
        return benchmark_results
    
    def export_regions_to_json(self, regions: List[TextRegion], output_path: str):
        """Export detected regions to JSON format"""
        import json
        
        regions_data = []
        for i, region in enumerate(regions):
            region_data = {
                'id': i,
                'bbox': region.bbox,
                'confidence': region.confidence,
                'angle': region.angle,
                'text_type': region.text_type,
                'method': region.method,
                'area': region.area,
                'aspect_ratio': region.aspect_ratio,
                'center': region.center
            }
            regions_data.append(region_data)
        
        export_data = {
            'total_regions': len(regions),
            'detection_stats': self.get_detection_stats(),
            'regions': regions_data
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self.logger.info(f"Exported {len(regions)} regions to {output_path}")
    
    def load_regions_from_json(self, input_path: str) -> List[TextRegion]:
        """Load detected regions from JSON format"""
        import json
        
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        regions = []
        for region_data in data['regions']:
            region = TextRegion(
                bbox=tuple(region_data['bbox']),
                confidence=region_data['confidence'],
                angle=region_data.get('angle', 0.0),
                text_type=region_data.get('text_type', 'unknown'),
                method=region_data.get('method', 'unknown')
            )
            regions.append(region)
        
        self.logger.info(f"Loaded {len(regions)} regions from {input_path}")
        return regions
    
    def optimize_parameters(self, validation_images: List[np.ndarray],
                          ground_truth: List[List[Tuple]]) -> Dict[str, Any]:
        """Optimize detection parameters using validation data"""
        best_params = {}
        best_score = 0.0
        
        # Parameter ranges to test
        confidence_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
        nms_thresholds = [0.2, 0.3, 0.4, 0.5]
        
        self.logger.info("Starting parameter optimization...")
        
        for conf_thresh in confidence_thresholds:
            for nms_thresh in nms_thresholds:
                # Temporarily set parameters
                original_conf = self.confidence_threshold
                original_nms = self.nms_threshold
                
                self.confidence_threshold = conf_thresh
                self.nms_threshold = nms_thresh
                
                # Evaluate on validation set
                total_score = 0.0
                
                for i, (image, gt_regions) in enumerate(zip(validation_images, ground_truth)):
                    detected_regions = self.detect_text_regions(image)
                    score = self._calculate_detection_score(detected_regions, gt_regions)
                    total_score += score
                
                avg_score = total_score / len(validation_images) if validation_images else 0
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_params = {
                        'confidence_threshold': conf_thresh,
                        'nms_threshold': nms_thresh,
                        'score': avg_score
                    }
                
                # Restore original parameters
                self.confidence_threshold = original_conf
                self.nms_threshold = original_nms
        
        # Apply best parameters
        if best_params:
            self.confidence_threshold = best_params['confidence_threshold']
            self.nms_threshold = best_params['nms_threshold']
            self.logger.info(f"Optimized parameters: {best_params}")
        
        return best_params
    
    def _calculate_detection_score(self, detected: List[TextRegion], 
                                 ground_truth: List[Tuple]) -> float:
        """Calculate detection score against ground truth"""
        if not ground_truth:
            return 1.0 if not detected else 0.0
        
        if not detected:
            return 0.0
        
        # Convert ground truth to TextRegion format
        gt_regions = [TextRegion(bbox=bbox, confidence=1.0, method="ground_truth") 
                     for bbox in ground_truth]
        
        # Calculate precision and recall
        true_positives = 0
        
        for det_region in detected:
            for gt_region in gt_regions:
                if self._regions_overlap(det_region, gt_region, overlap_threshold=0.5):
                    true_positives += 1
                    break
        
        precision = true_positives / len(detected) if detected else 0
        recall = true_positives / len(gt_regions) if gt_regions else 0
        
        # F1 score
        if precision + recall == 0:
            return 0.0
        
        f1_score = 2 * (precision * recall) / (precision + recall)
        return f1_score
    
    def cleanup(self):
        """Cleanup detector resources"""
        if self.craft_detector and hasattr(self.craft_detector, 'model'):
            if self.craft_detector.model is not None:
                del self.craft_detector.model
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    torch.cuda.empty_cache()

        self.logger.info("Text detector cleanup completed")

    # Enhanced functionality from EnhancedTextDetector (merged for backward compatibility)

    def detect_text_regions_dict(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Enhanced detection method that returns results in dictionary format
        (for backward compatibility with EnhancedTextDetector interface)
        """
        try:
            # First, try the standard, optimal detection method
            detected_regions = self.detect_text_regions(image)

            # If standard detection fails (or returns a low number of regions), try the aggressive fallback
            if not detected_regions or len(detected_regions) < 3: # Added a low-count check
                self.logger.warning("Standard detection found no/few regions, trying aggressive mode")
                fallback_regions = self._aggressive_detection_fallback(image)
                # We should merge the results, not replace them
                detected_regions.extend(fallback_regions)
                # Now re-post-process the combined regions
                detected_regions = self._post_process_regions(detected_regions, image.shape[:2])

            # Convert TextRegion objects to dictionaries for compatibility
            result_regions = []
            for region in detected_regions:
                region_dict = {
                    'bbox': region.bbox,
                    'confidence': region.confidence,
                    'angle': region.angle,
                    'text_type': region.text_type,
                    'method': region.method
                }
                result_regions.append(region_dict)

            result = {
                'regions': result_regions,
                'total_regions': len(result_regions),
                'detection_method': 'enhanced',
                'success': True
            }

            self.logger.info(f"Enhanced detection found {len(result_regions)} regions")
            return result

        except Exception as e:
            self.logger.error(f"Enhanced detection failed: {e}")
            return {
                'regions': [],
                'total_regions': 0,
                'detection_method': 'failed',
                'success': False,
                'error': str(e)
            }

    def _aggressive_detection_fallback(self, image: np.ndarray) -> List[TextRegion]:
        """
        Aggressive fallback detection when standard methods fail
        """
        regions = []

        try:
            # Combine multiple aggressive methods

            # Method 1: Super aggressive MSER
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()

            # Use the consistent, modern MSER API with error handling
            if hasattr(cv2, 'MSER_create'):
                mser = cv2.MSER_create()
                mser.setMinArea(20)
                mser.setMaxArea(gray.shape[0] * gray.shape[1] // 4)
                mser.setMaxVariation(0.5)
            else:
                self.logger.warning("MSER_create not available in cv2, skipping aggressive MSER")
                return []

            mser_regions, bboxes = mser.detectRegions(gray)

            for i, bbox in enumerate(bboxes):
                x, y, w, h = bbox

                # Filter based on size
                if w > 5 and h > 5:
                    # Ensure bbox is within image bounds
                    x = max(0, min(x, gray.shape[1] - 1))
                    y = max(0, min(y, gray.shape[0] - 1))
                    w = min(w, gray.shape[1] - x)
                    h = min(h, gray.shape[0] - y)

                    if w > 0 and h > 0:
                        regions.append(TextRegion(
                            bbox=(x, y, w, h),
                            confidence=0.5,
                            method="aggressive_mser"
                        ))

            # Method 2: Contour-based detection
            contour_regions = self._aggressive_contour_detection(gray)
            regions.extend(contour_regions)

            # If after all aggressive methods we still have no regions, use grid fallback
            if not regions:
                self.logger.warning("All aggressive methods failed, using grid-based fallback detection")
                grid_regions = self._create_grid_regions(image)
                regions.extend(grid_regions)

            # Filter and merge
            # Note: We do a simplified post-processing here.
            if regions:
                regions = self._post_process_regions(regions, image.shape[:2])

            self.logger.info(f"Aggressive detection found {len(regions)} regions")
            return regions

        except Exception as e:
            self.logger.error(f"Aggressive detection failed: {e}")
            # Ultimate fallback - return full image as single region
            h, w = image.shape[:2]
            return [TextRegion(
                bbox=(0, 0, w, h),
                confidence=0.3,
                method="full_image_fallback"
            )]

    def _create_grid_regions(self, image: np.ndarray) -> List[TextRegion]:
        """Create grid-based regions as ultimate fallback"""
        h, w = image.shape[:2]
        regions = []

        # Adaptive grid size based on image dimensions
        if h > 800 or w > 600:
            rows, cols = 4, 3
        elif h > 400 or w > 300:
            rows, cols = 3, 2
        else:
            rows, cols = 2, 2

        cell_h = h // rows
        cell_w = w // cols

        for row in range(rows):
            for col in range(cols):
                x = col * cell_w
                y = row * cell_h

                # Adjust last cells to cover remainder
                width = cell_w if col < cols - 1 else w - x
                height = cell_h if row < rows - 1 else h - y

                if width > 30 and height > 20:
                    regions.append(TextRegion(
                        bbox=(x, y, width, height),
                        confidence=0.4,
                        method="grid_fallback"
                    ))

        return regions

    def _aggressive_contour_detection(self, gray: np.ndarray) -> List[TextRegion]:
        """Aggressive contour-based detection"""
        regions = []

        try:
            # Multiple threshold approaches
            thresholds = [
                cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
                cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)[1],
                cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            ]

            for binary in thresholds:
                # Find contours
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for contour in contours:
                    area = cv2.contourArea(contour)
                    if 50 < area < 10000:  # Size filter
                        x, y, w, h = cv2.boundingRect(contour)

                        # Aspect ratio filter
                        aspect_ratio = w / h if h > 0 else 0
                        if 0.1 < aspect_ratio < 20:
                            regions.append(TextRegion(
                                bbox=(x, y, w, h),
                                confidence=0.4,
                                method="aggressive_contour"
                            ))

        except Exception as e:
            self.logger.warning(f"Contour detection failed: {e}")

        return regions



    

