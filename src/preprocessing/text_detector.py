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
import multiprocessing
from concurrent.futures import ThreadPoolExecutor 
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
        self.min_text_size = config.get("min_text_size", 25)     # Increased from 20
        self.max_text_size = config.get("max_text_size", 150)    # Reduced from 200
        self.min_area_threshold = config.get("min_area_threshold", 500)   # Increased from 300
        self.aspect_ratio_min = config.get("aspect_ratio_min", 0.2)       # Increased from 0.15
        self.aspect_ratio_max = config.get("aspect_ratio_max", 12)        # Reduced from 15
        self.confidence_threshold = config.get("confidence_threshold", 0.55)  # New higher threshold
        self.logger = logging.getLogger("TextDetector.Traditional")
        
    def _create_mser_detector(self):
        """Helper to create a configured MSER detector instance - ULTRA OPTIMIZED VERSION"""
        try:
            if hasattr(cv2, 'MSER_create'):
                mser = cv2.MSER_create()
                # ULTRA RESTRICTIVE PARAMETERS for complex form documents
                try:
                    mser.setMinArea(100)                   # Increased from 50
                    mser.setMaxArea(self.max_text_size * 20)  # Reduced from * 30
                    mser.setMaxVariation(0.1)              # Reduced from 0.15 - very stable only
                    mser.setMinDiversity(0.5)              # Increased from 0.4 - high diversity
                    mser.setMaxEvolution(60)               # Reduced from 80 - minimal evolution
                    mser.setAreaThreshold(2.5)             # Increased from 1.8 - much stricter
                    mser.setMinMargin(0.015)               # Increased from 0.01 - larger margins
                    mser.setEdgeBlurSize(2)                # Reduced from 3 - minimal blur
                except AttributeError:
                    pass
                return mser
            else:
                self.logger.warning("MSER_create not available in cv2, using fallback detection")
                return None
        except Exception as e:
            self.logger.warning(f"MSER_create initialization failed: {e}")
            return None
        
    def _estimate_mser_confidence_enhanced(self, region: Optional[np.ndarray], 
                                        roi: np.ndarray, aspect_ratio: float, 
                                        area: int) -> float:
        """Enhanced confidence estimation for MSER regions"""
        base_confidence = 0.6
        
        # Size-based confidence adjustment
        if 500 < area < 5000:  # Sweet spot for text regions
            base_confidence *= 1.3
        elif area < 300:  # Very small regions
            base_confidence *= 0.7
        elif area > 10000:  # Very large regions
            base_confidence *= 0.8
        
        # Aspect ratio confidence adjustment
        if 0.3 < aspect_ratio < 8:  # Good text aspect ratios
            base_confidence *= 1.2
        elif aspect_ratio < 0.2 or aspect_ratio > 12:  # Poor aspect ratios
            base_confidence *= 0.6
        
        # ROI intensity analysis
        if roi.size > 0:
            std_dev = np.std(roi)
            mean_intensity = np.mean(roi)
            
            # Good contrast regions
            if std_dev > 35:  # Increased from 30
                base_confidence *= 1.25
            elif std_dev < 15:  # Low contrast
                base_confidence *= 0.7
            
            # Avoid very dark or very bright uniform regions
            if mean_intensity < 20 or mean_intensity > 235:
                base_confidence *= 0.8
        
        # Region point density (if available)
        if region is not None and len(region) > 0:
            density = len(region) / area if area > 0 else 0
            if 0.1 < density < 0.8:  # Good density
                base_confidence *= 1.15
        
        return min(0.95, base_confidence)

    def detect_mser(self, image: np.ndarray) -> List[TextRegion]:
        """Detect text using MSER - ULTRA OPTIMIZED for form documents"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        try:
            mser = self._create_mser_detector()
            if mser is None:
                return self._fallback_detection(gray)
            
            regions, bboxes = mser.detectRegions(gray)

            detected_regions = []
            for i, bbox in enumerate(bboxes):
                x, y, w, h = bbox

                # ULTRA AGGRESSIVE FILTERING for complex documents
                aspect_ratio = w / h if h > 0 else 0
                area = w * h
                
                # Skip tiny regions very aggressively
                if w < 25 or h < 15:  # Increased from 20,12
                    continue
                    
                # Skip large regions more aggressively
                if w > 300 or h > 100:  # More restrictive
                    continue
                
                # Much more restrictive aspect ratio filtering
                if aspect_ratio < 0.2 or aspect_ratio > 12:  # Was 0.15-15
                    continue
                
                # Skip small areas very aggressively
                if area < 500:  # Increased from 300
                    continue
                
                # Skip very thin or very wide regions (form lines/borders)
                if (w > h * 20) or (h > w * 15):  # More restrictive
                    continue

                # Additional form-specific filtering
                # Skip regions that look like form borders/lines
                if (w > 200 and h < 8) or (h > 80 and w < 8):
                    continue
                
                # Skip very small text (likely noise in forms)
                if w * h < 600:  # Minimum text area
                    continue

                # Ensure bbox is within image bounds
                x = max(0, min(x, gray.shape[1] - 1))
                y = max(0, min(y, gray.shape[0] - 1))
                w = min(w, gray.shape[1] - x)
                h = min(h, gray.shape[0] - y)

                if w > 0 and h > 0:
                    # Ultra-enhanced confidence estimation for forms
                    confidence = self._estimate_mser_confidence_ultra(
                        regions[i] if i < len(regions) else None, 
                        gray[y:y+h, x:x+w],
                        aspect_ratio,
                        area
                    )
                    
                    # Only keep very high-confidence regions
                    if confidence > 0.55:  # Increased from 0.4
                        detected_regions.append(TextRegion(
                            bbox=(x, y, w, h),
                            confidence=confidence,
                            method="ultra_optimized_mser"
                        ))

            self.logger.info(f"Ultra-optimized MSER detected {len(detected_regions)} regions")
            return detected_regions

        except Exception as e:
            self.logger.error(f"MSER detection failed: {e}")
            return self._fallback_detection(gray)
        
    def _estimate_mser_confidence_ultra(self, region: Optional[np.ndarray], 
                                    roi: np.ndarray, aspect_ratio: float, 
                                    area: int) -> float:
        """Ultra-enhanced confidence estimation for form documents"""
        base_confidence = 0.5  # Start lower for forms
        
        # Size-based confidence - prefer medium-sized regions for forms
        if 800 < area < 3000:  # Sweet spot for form text
            base_confidence *= 1.4
        elif 500 < area < 800:  # Acceptable size
            base_confidence *= 1.1
        elif area < 500:  # Too small for reliable text
            base_confidence *= 0.5
        elif area > 5000:  # Too large, likely form sections
            base_confidence *= 0.6
        
        # Aspect ratio confidence - forms prefer certain ratios
        if 0.4 < aspect_ratio < 6:  # Good form text aspect ratios
            base_confidence *= 1.3
        elif 0.25 < aspect_ratio < 0.4 or 6 < aspect_ratio < 10:  # Acceptable
            base_confidence *= 1.0
        else:  # Poor aspect ratios for text
            base_confidence *= 0.5
        
        # ROI analysis for form documents
        if roi.size > 0:
            std_dev = np.std(roi)
            mean_intensity = np.mean(roi)
            
            # Forms need good contrast
            if std_dev > 40:  # Excellent contrast for forms
                base_confidence *= 1.3
            elif std_dev > 25:  # Good contrast
                base_confidence *= 1.1
            elif std_dev < 15:  # Poor contrast - likely form background
                base_confidence *= 0.4
            
            # Avoid form backgrounds and borders
            if mean_intensity < 30 or mean_intensity > 220:  # Very dark/light
                base_confidence *= 0.6
            elif 50 < mean_intensity < 200:  # Good text range
                base_confidence *= 1.2
            
            # Check for uniform regions (likely form elements, not text)
            if std_dev < 10 and (mean_intensity < 40 or mean_intensity > 200):
                base_confidence *= 0.3
        
        # Region density analysis for forms
        if region is not None and len(region) > 0:
            density = len(region) / area if area > 0 else 0
            if 0.15 < density < 0.6:  # Good density for form text
                base_confidence *= 1.2
            elif density > 0.8:  # Too dense - likely noise
                base_confidence *= 0.5
        
        return min(0.98, base_confidence)

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
    
    def _estimate_mser_confidence(self, region: Optional[np.ndarray], roi: np.ndarray) -> float:
        """Estimate confidence for MSER region"""
        base_confidence = 0.7
        
        # Adjust based on region size (if region data available)
        if region is not None and len(region) > 0:
            area = len(region)  # MSER regions are point lists
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
    """Advanced text detection system combining multiple methods - MODERN SYSTEM"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger("AdvancedTextDetector")
        
        # Detection parameters - OPTIMIZED FOR MODERN PERFORMANCE
        self.detection_method = DetectionMethod(config.get("method", "auto"))
        self.confidence_threshold = config.get("confidence_threshold", 0.6)  # Higher threshold
        self.nms_threshold = config.get("nms_threshold", 0.3)  # More aggressive NMS
        self.min_region_area = config.get("min_region_area", 150)  # Larger minimum
        self.max_region_area = config.get("max_region_area", 50000)  # Reasonable max
        
        # MODERN DEDUPLICATION PARAMETERS
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
        self.parallel_threshold = config.get("parallel_threshold", 2000000)  # 2MP images
        self.tile_overlap = config.get("tile_overlap", 0.1)  # 10% overlap
        
        # READING ORDER DETECTION
        self.enable_reading_order = config.get("enable_reading_order", True)
        self.column_detection = config.get("column_detection", True)
        
        self.logger.info(f"Parallel processing: {self.enable_parallel}, Workers: {self.max_workers}")
    
    def detect_text_regions_parallel(self, image: np.ndarray) -> List[TextRegion]:
        """
        PARALLEL TEXT DETECTION - Process large images using multiple threads
        """
        start_time = time.time()
        
        if image is None or image.size == 0:
            return []
        
        h, w = image.shape[:2]
        image_size = h * w
        
        # Use parallel processing for large images
        if (self.enable_parallel and 
            image_size > self.parallel_threshold and 
            self.max_workers > 1):
            
            regions = self._parallel_detection_pipeline(image)
            self.logger.info(f"Parallel detection completed in {time.time() - start_time:.3f}s")
        else:
            # Use standard single-threaded detection
            regions = self.detect_text_regions(image)
            
        # Apply reading order detection if enabled
        if self.enable_reading_order and regions:
            regions = self._apply_reading_order(regions, image.shape[:2])
        
        return regions
    
    
    def _create_image_tiles(self, image: np.ndarray) -> Tuple[List[np.ndarray], List[Dict]]:
        """Create overlapping tiles for parallel processing"""
        h, w = image.shape[:2]
        
        # Determine tile size based on image size and worker count
        target_tiles = self.max_workers * 2  # 2 tiles per worker
        tile_area = (h * w) // target_tiles
        tile_size = int(np.sqrt(tile_area))
        
        # Ensure minimum tile size
        tile_size = max(512, min(tile_size, 1024))
        overlap_size = int(tile_size * self.tile_overlap)
        
        tiles = []
        tile_info = []
        
        # Create grid of overlapping tiles
        y_positions = list(range(0, h - tile_size + 1, tile_size - overlap_size))
        if y_positions[-1] + tile_size < h:
            y_positions.append(h - tile_size)
            
        x_positions = list(range(0, w - tile_size + 1, tile_size - overlap_size))
        if x_positions[-1] + tile_size < w:
            x_positions.append(w - tile_size)
        
        for y in y_positions:
            for x in x_positions:
                # Ensure we don't go out of bounds
                y_end = min(y + tile_size, h)
                x_end = min(x + tile_size, w)
                
                tile = image[y:y_end, x:x_end]
                info = {
                    'x_offset': x,
                    'y_offset': y,
                    'width': x_end - x,
                    'height': y_end - y,
                    'overlap_size': overlap_size
                }
                
                tiles.append(tile)
                tile_info.append(info)
        
        self.logger.info(f"Created {len(tiles)} tiles of size ~{tile_size}x{tile_size}")
        return tiles, tile_info
    
    def _process_tile(self, tile: np.ndarray, tile_info: Dict, tile_idx: int) -> List[TextRegion]:
        """Process a single tile for text detection"""
        try:
            # Use standard detection on tile
            regions = self._detect_with_method(tile, self._select_optimal_method(tile))
            
            # Adjust coordinates to global image space
            global_regions = []
            x_offset = tile_info['x_offset']
            y_offset = tile_info['y_offset']
            
            for region in regions:
                x, y, w, h = region.bbox
                global_bbox = (
                    x + x_offset,
                    y + y_offset,
                    w, h
                )
                
                global_region = TextRegion(
                    bbox=global_bbox,
                    confidence=region.confidence,
                    angle=region.angle,
                    text_type=region.text_type,
                    method=f"parallel_{region.method}"
                )
                global_regions.append(global_region)
            
            return global_regions
            
        except Exception as e:
            self.logger.error(f"Tile {tile_idx} processing failed: {e}")
            return []
    
    def _merge_tile_results(self, tile_results: Dict, image_shape: Tuple[int, int]) -> List[TextRegion]:
        """FAST tile result merging - optimized for overlapping tiles"""
        
        # Collect all regions quickly
        all_regions = []
        for tile_idx, (regions, tile_info) in tile_results.items():
            all_regions.extend(regions)
        
        if not all_regions:
            return []
        
        # Use the fast merging algorithm directly
        return self._grid_based_merging(all_regions)  # Your optimized algorithm
    
    def _merge_overlapping_tile_regions(self, regions: List[TextRegion]) -> List[TextRegion]:
        """ENHANCED tile region merging with better deduplication"""
        
        if len(regions) <= 1:
            return regions
        
        # Sort by confidence for priority merging
        regions = sorted(regions, key=lambda r: r.confidence, reverse=True)
        
        merged = []
        used_indices = set()
        
        for i, region in enumerate(regions):
            if i in used_indices:
                continue
            
            # Find overlapping regions from tile boundaries
            merge_candidates = [region]
            used_indices.add(i)
            
            for j, other_region in enumerate(regions[i+1:], i+1):
                if j in used_indices:
                    continue
                
                # More aggressive merging for tile boundaries
                iou = self._calculate_precise_iou(region.bbox, other_region.bbox)
                
                # Enhanced merging criteria for tiles
                should_merge = False
                
                # High overlap regions
                if iou > 0.2:  # Reduced from 0.3 for tile boundaries
                    should_merge = True
                
                # Adjacent regions with good alignment
                elif iou > 0.05 and self._regions_very_close(region, other_region):
                    should_merge = True
                
                if should_merge:
                    merge_candidates.append(other_region)
                    used_indices.add(j)
            
            # Create merged region
            if len(merge_candidates) == 1:
                merged.append(region)
            else:
                merged_region = self._merge_region_group_advanced(merge_candidates)
                merged.append(merged_region)
        
        return merged
    
    def _regions_very_close(self, region1: TextRegion, region2: TextRegion) -> bool:
        """Check if regions are very close (likely from tile boundaries)"""
        bbox1 = region1.bbox
        bbox2 = region2.bbox
        
        # Calculate minimum distance between regions
        x1_center = bbox1[0] + bbox1[2] // 2
        y1_center = bbox1[1] + bbox1[3] // 2
        x2_center = bbox2[0] + bbox2[2] // 2
        y2_center = bbox2[1] + bbox2[3] // 2
        
        distance = np.sqrt((x1_center - x2_center)**2 + (y1_center - y2_center)**2)
        
        # Average dimension for scale
        avg_size = (bbox1[2] + bbox1[3] + bbox2[2] + bbox2[3]) / 4
        
        # Very close if distance < 50% of average size
        return distance < avg_size * 0.5
    
    def _apply_reading_order(self, regions: List[TextRegion], 
                           image_shape: Tuple[int, int]) -> List[TextRegion]:
        """
        READING ORDER DETECTION - Sort regions in natural reading order
        """
        if not regions or len(regions) < 2:
            return regions
        
        try:
            # Detect document structure
            if self.column_detection:
                structured_regions = self._detect_reading_structure(regions, image_shape)
            else:
                structured_regions = self._simple_reading_order(regions)
            
            self.logger.info(f"Applied reading order to {len(structured_regions)} regions")
            return structured_regions
            
        except Exception as e:
            self.logger.warning(f"Reading order detection failed: {e}")
            return regions
    
    def _detect_reading_structure(self, regions: List[TextRegion], 
                                image_shape: Tuple[int, int]) -> List[TextRegion]:
        """Detect columns and reading flow"""
        
        if not regions:
            return regions
        
        h, w = image_shape
        
        # Group regions into horizontal bands (potential text lines)
        bands = self._group_into_horizontal_bands(regions)
        
        # Sort each band left-to-right
        sorted_bands = []
        for band in bands:
            # Sort by x-coordinate within each band
            band_sorted = sorted(band, key=lambda r: r.bbox[0])
            sorted_bands.append(band_sorted)
        
        # Sort bands top-to-bottom
        sorted_bands.sort(key=lambda band: min(r.bbox[1] for r in band))
        
        # Flatten back to single list
        reading_order_regions = []
        for band in sorted_bands:
            reading_order_regions.extend(band)
        
        return reading_order_regions
    
    def _group_into_horizontal_bands(self, regions: List[TextRegion]) -> List[List[TextRegion]]:
        """Group regions into horizontal bands (text lines)"""
        
        if not regions:
            return []
        
        # Sort by y-coordinate
        sorted_regions = sorted(regions, key=lambda r: r.bbox[1])
        
        bands = []
        current_band = [sorted_regions[0]]
        
        for region in sorted_regions[1:]:
            # Check if region is on same horizontal level as current band
            current_band_y = current_band[0].bbox[1] + current_band[0].bbox[3] // 2
            region_y = region.bbox[1] + region.bbox[3] // 2
            
            # Average height of current band
            avg_height = sum(r.bbox[3] for r in current_band) / len(current_band)
            
            # If y-difference is less than 50% of average height, same band
            if abs(region_y - current_band_y) < avg_height * 0.5:
                current_band.append(region)
            else:
                # Start new band
                bands.append(current_band)
                current_band = [region]
        
        # Add last band
        if current_band:
            bands.append(current_band)
        
        return bands
    
    def _simple_reading_order(self, regions: List[TextRegion]) -> List[TextRegion]:
        """Simple top-to-bottom, left-to-right reading order"""
        # Sort primarily by y-coordinate, secondarily by x-coordinate
        return sorted(regions, key=lambda r: (r.bbox[1], r.bbox[0]))
    
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
        """Main text detection method with intelligent method selection - MODERN OPTIMIZED VERSION"""
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
            
            # MODERN POST-PROCESSING PIPELINE
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
        
        # Decision logic - OPTIMIZED FOR MODERN PERFORMANCE
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
        
        else:  # Standard case - prefer MSER for reliability
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
    
    # ==========================================================================
    # MODERN POST-PROCESSING PIPELINE - COMPLETELY REWRITTEN FOR PERFORMANCE
    # ==========================================================================
    
    def _modern_post_process_regions(self, regions: List[TextRegion], 
                                   image_shape: Tuple[int, int]) -> List[TextRegion]:
        """MODERN POST-PROCESSING - Optimized for production systems"""
        
        if not regions:
            return regions
        
        self.logger.info(f"Starting modern post-processing: {len(regions)} raw regions")
        
        # Stage 1: Validate and normalize bounding boxes
        validated_regions = self._validate_bounding_boxes(regions, image_shape)
        self.logger.info(f"After validation: {len(validated_regions)} regions")
        
        # Stage 2: Smart filtering - remove obvious noise
        if self.enable_smart_filtering:
            filtered_regions = self._smart_filter_regions(validated_regions)
            self.logger.info(f"After smart filtering: {len(filtered_regions)} regions")
        else:
            filtered_regions = validated_regions
        
        # Stage 3: Advanced deduplication and merging
        merged_regions = self._advanced_region_merging(filtered_regions)
        self.logger.info(f"After advanced merging: {len(merged_regions)} regions")
        
        # Stage 4: Final quality filtering
        final_regions = self._final_quality_filter(merged_regions)
        self.logger.info(f"After final filtering: {len(final_regions)} regions")
        
        # Stage 5: Sort by reading order
        final_regions.sort(key=lambda r: (r.bbox[1], r.bbox[0]))
        
        self.logger.info(f"Modern post-processing complete: {len(regions)} → {len(final_regions)} regions")
        return final_regions
    
    def _validate_bounding_boxes(self, regions: List[TextRegion], 
                               image_shape: Tuple[int, int]) -> List[TextRegion]:
        """Validate and normalize bounding boxes"""
        h, w = image_shape
        validated_regions = []
        
        for region in regions:
            x, y, rw, rh = region.bbox
            
            # Clamp coordinates to image bounds
            x = max(0, min(x, w - 1))
            y = max(0, min(y, h - 1))
            rw = max(1, min(rw, w - x))
            rh = max(1, min(rh, h - y))
            
            # Only keep regions with positive dimensions
            if rw > 0 and rh > 0:
                validated_region = TextRegion(
                    bbox=(x, y, rw, rh),
                    confidence=region.confidence,
                    angle=region.angle,
                    text_type=region.text_type,
                    method=region.method
                )
                validated_regions.append(validated_region)
        
        return validated_regions
    
    def _smart_filter_regions(self, regions: List[TextRegion]) -> List[TextRegion]:
        """ENHANCED SMART FILTERING - More aggressive noise removal"""
        filtered = []
        
        for region in regions:
            x, y, w, h = region.bbox
            
            # More aggressive basic size filtering
            if region.area < 300:  # Increased from min_region_area (150)
                continue
            
            if region.area > 25000:  # Reduced from max_region_area (50000)
                continue
            
            # Higher confidence filtering
            if region.confidence < 0.5:  # Increased from confidence_threshold (0.6->0.5 but with better estimation)
                continue
            
            # More restrictive aspect ratio filtering
            aspect_ratio = region.aspect_ratio
            
            # Skip extremely thin regions more aggressively
            if aspect_ratio < 0.12 and h > w * 15:  # Was 0.05 and 20
                continue
            
            # Skip extremely wide regions more aggressively  
            if aspect_ratio > 25 and w > h * 20:  # Was 50 and 30
                continue
            
            # Skip small square regions (likely punctuation noise)
            if 0.7 < aspect_ratio < 1.4 and region.area < 600:  # Increased from 400
                continue
            
            # More restrictive dimension filtering
            if w < 20 or h < 10:  # Increased from 8 and 6
                continue
            
            # Skip very narrow tall regions (likely noise)
            if w < 8 and h > 40:
                continue
            
            # Skip very short wide regions (likely underlines)
            if h < 6 and w > 100:
                continue
            
            # Additional noise patterns
            # Skip regions that are too regular (likely generated graphics)
            if w == h and region.area < 1000:  # Perfect squares under 1000px
                continue
            
            filtered.append(region)
        
        return filtered
        
    # FAST REGION MERGING - Replace _advanced_region_merging method

    def _advanced_region_merging(self, regions: List[TextRegion]) -> List[TextRegion]:
        """OPTIMIZED REGION MERGING - Uses spatial indexing for O(n log n) performance"""
        
        if len(regions) <= 1:
            return regions
        
        start_time = time.time()
        
        # For very large region sets, use grid-based spatial optimization
        if len(regions) > 2000:
            merged_regions = self._grid_based_merging(regions)
        else:
            merged_regions = self._standard_merging(regions)
        
        merge_time = time.time() - start_time
        self.logger.info(f"Advanced merging: {len(regions)} → {len(merged_regions)} regions in {merge_time:.3f}s")
        
        return merged_regions

    def _grid_based_merging(self, regions: List[TextRegion]) -> List[TextRegion]:
        """GRID-BASED MERGING - O(n log n) algorithm for large region sets"""
        
        # Create spatial grid for fast neighbor lookup
        grid_size = 100  # 100x100 pixel grid cells
        region_grid = {}
        
        # Index regions by grid cells
        for i, region in enumerate(regions):
            x, y, w, h = region.bbox
            
            # Calculate grid coordinates (region can span multiple cells)
            start_x, start_y = x // grid_size, y // grid_size
            end_x, end_y = (x + w) // grid_size, (y + h) // grid_size
            
            # Add region to all grid cells it touches
            for gx in range(start_x, end_x + 1):
                for gy in range(start_y, end_y + 1):
                    if (gx, gy) not in region_grid:
                        region_grid[(gx, gy)] = []
                    region_grid[(gx, gy)].append((i, region))
        
        # Merge within each grid cell and neighbors
        merged = []
        used_indices = set()
        
        # Sort regions by confidence for priority merging
        sorted_regions = sorted(enumerate(regions), key=lambda x: x[1].confidence, reverse=True)
        
        for original_idx, region in sorted_regions:
            if original_idx in used_indices:
                continue
            
            # Find potential merge candidates in nearby grid cells
            merge_candidates = [region]
            used_indices.add(original_idx)
            
            x, y, w, h = region.bbox
            center_gx, center_gy = (x + w // 2) // grid_size, (y + h // 2) // grid_size
            
            # Check current and adjacent grid cells
            for dgx in [-1, 0, 1]:
                for dgy in [-1, 0, 1]:
                    check_gx, check_gy = center_gx + dgx, center_gy + dgy
                    
                    if (check_gx, check_gy) in region_grid:
                        for candidate_idx, candidate_region in region_grid[(check_gx, check_gy)]:
                            if candidate_idx in used_indices:
                                continue
                            
                            # Fast overlap check
                            if self._fast_should_merge(region, candidate_region):
                                merge_candidates.append(candidate_region)
                                used_indices.add(candidate_idx)
            
            # Create merged region
            if len(merge_candidates) == 1:
                merged.append(region)
            else:
                merged_region = self._merge_region_group_fast(merge_candidates)
                merged.append(merged_region)
        
        return merged

    def _standard_merging(self, regions: List[TextRegion]) -> List[TextRegion]:
        """STANDARD MERGING - For smaller region sets"""
        
        # Sort by confidence
        regions = sorted(regions, key=lambda r: r.confidence, reverse=True)
        
        merged = []
        used_indices = set()
        
        for i, region in enumerate(regions):
            if i in used_indices:
                continue
            
            merge_candidates = [region]
            used_indices.add(i)
            
            # Only check nearby regions (spatial optimization)
            for j, other_region in enumerate(regions[i+1:], i+1):
                if j in used_indices:
                    continue
                
                # Quick spatial distance check before expensive IoU
                if self._regions_spatially_close(region, other_region):
                    if self._fast_should_merge(region, other_region):
                        merge_candidates.append(other_region)
                        used_indices.add(j)
            
            # Create merged region
            if len(merge_candidates) == 1:
                merged.append(region)
            else:
                merged_region = self._merge_region_group_fast(merge_candidates)
                merged.append(merged_region)
        
        return merged

    def _regions_spatially_close(self, region1: TextRegion, region2: TextRegion) -> bool:
        """Quick spatial distance check to avoid expensive IoU calculations"""
        
        x1, y1, w1, h1 = region1.bbox
        x2, y2, w2, h2 = region2.bbox
        
        # Calculate centers
        c1_x, c1_y = x1 + w1 // 2, y1 + h1 // 2
        c2_x, c2_y = x2 + w2 // 2, y2 + h2 // 2
        
        # Maximum distance for potential merging
        max_size = max(w1, h1, w2, h2)
        max_distance = max_size * 2  # Regions can't be further than 2x their max dimension
        
        # Euclidean distance check
        distance = ((c1_x - c2_x) ** 2 + (c1_y - c2_y) ** 2) ** 0.5
        
        return distance <= max_distance

    def _fast_should_merge(self, region1: TextRegion, region2: TextRegion) -> bool:
        """Fast merge decision with optimized checks"""
        
        # Quick bounding box overlap check
        x1, y1, w1, h1 = region1.bbox
        x2, y2, w2, h2 = region2.bbox
        
        # Check if bounding boxes overlap at all
        if (x1 + w1 < x2 or x2 + w2 < x1 or 
            y1 + h1 < y2 or y2 + h2 < y1):
            # No overlap, check if they're adjacent and aligned
            return self._regions_adjacent_and_aligned(region1, region2)
        
        # Calculate IoU only if there's overlap
        iou = self._calculate_precise_iou(region1.bbox, region2.bbox)
        
        # Merge criteria
        if iou > self.iou_threshold:  # High IoU
            return True
        elif iou > 0.1 and self._regions_well_aligned(region1, region2):  # Medium IoU + alignment
            return True
        
        return False

    def _regions_adjacent_and_aligned(self, region1: TextRegion, region2: TextRegion) -> bool:
        """Check if regions are adjacent and well-aligned (for text lines)"""
        
        bbox1, bbox2 = region1.bbox, region2.bbox
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Check vertical alignment (same text line)
        y1_center = y1 + h1 // 2
        y2_center = y2 + h2 // 2
        
        avg_height = (h1 + h2) / 2
        vertical_alignment = abs(y1_center - y2_center) < avg_height * 0.3
        
        if not vertical_alignment:
            return False
        
        # Check horizontal proximity
        gap = min(abs((x1 + w1) - x2), abs((x2 + w2) - x1))
        avg_width = (w1 + w2) / 2
        
        # Adjacent if gap is small relative to average width
        return gap < avg_width * 0.5

    def _merge_region_group_fast(self, regions: List[TextRegion]) -> TextRegion:
        """Fast region group merging with minimal calculations"""
        
        if len(regions) == 1:
            return regions[0]
        
        # Calculate union bounding box efficiently
        min_x = min(r.bbox[0] for r in regions)
        min_y = min(r.bbox[1] for r in regions)
        max_x = max(r.bbox[0] + r.bbox[2] for r in regions)
        max_y = max(r.bbox[1] + r.bbox[3] for r in regions)
        
        merged_bbox = (min_x, min_y, max_x - min_x, max_y - min_y)
        
        # Fast confidence calculation (area-weighted average)
        total_area = sum(r.area for r in regions)
        if total_area > 0:
            weighted_confidence = sum(r.confidence * r.area for r in regions) / total_area
        else:
            weighted_confidence = sum(r.confidence for r in regions) / len(regions)
        
        # Confidence boost for consistent high-confidence regions
        high_conf_ratio = sum(1 for r in regions if r.confidence > 0.8) / len(regions)
        if high_conf_ratio > 0.7:
            weighted_confidence *= 1.1
        
        # Select best method
        best_region = max(regions, key=lambda r: r.confidence)
        
        return TextRegion(
            bbox=merged_bbox,
            confidence=min(0.99, weighted_confidence),
            angle=best_region.angle,
            method=f"fast_merged_{best_region.method}",
            text_type=best_region.text_type
        )
        
    def _calculate_precise_iou(self, bbox1: Tuple[int, int, int, int], 
                              bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate precise Intersection over Union (IoU)"""
        x1_1, y1_1, w1, h1 = bbox1
        x1_2, y1_2, w2, h2 = bbox2
        
        # Convert to (x1, y1, x2, y2) format
        x2_1, y2_1 = x1_1 + w1, y1_1 + h1
        x2_2, y2_2 = x1_2 + w2, y1_2 + h2
        
        # Calculate intersection coordinates
        ix1 = max(x1_1, x1_2)
        iy1 = max(y1_1, y1_2)
        ix2 = min(x2_1, x2_2)
        iy2 = min(y2_1, y2_2)
        
        # Check if there's an intersection
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        
        # Calculate intersection area
        intersection = (ix2 - ix1) * (iy2 - iy1)
        
        # Calculate union area
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        # Calculate IoU
        iou = intersection / union if union > 0 else 0.0
        return iou
    
    def _regions_well_aligned(self, region1: TextRegion, region2: TextRegion) -> bool:
        """Check if regions are well aligned (could be part of same text line)"""
        bbox1 = region1.bbox
        bbox2 = region2.bbox
        
        # Vertical overlap check
        y1_start, y1_end = bbox1[1], bbox1[1] + bbox1[3]
        y2_start, y2_end = bbox2[1], bbox2[1] + bbox2[3]
        
        vertical_overlap = max(0, min(y1_end, y2_end) - max(y1_start, y2_start))
        min_height = min(bbox1[3], bbox2[3])
        
        # Good vertical alignment if overlap > 50% of smaller region's height
        return vertical_overlap > min_height * 0.5
    
    def _regions_on_same_line(self, region1: TextRegion, region2: TextRegion) -> bool:
        """Check if regions are on the same text line"""
        bbox1 = region1.bbox
        bbox2 = region2.bbox
        
        # Centers should be roughly at same height
        center1_y = bbox1[1] + bbox1[3] // 2
        center2_y = bbox2[1] + bbox2[3] // 2
        
        height_diff = abs(center1_y - center2_y)
        avg_height = (bbox1[3] + bbox2[3]) / 2
        
        # On same line if height difference < 30% of average height
        if height_diff > avg_height * 0.3:
            return False
        
        # Check horizontal proximity
        x1_end = bbox1[0] + bbox1[2]
        x2_start = bbox2[0]
        x2_end = bbox2[0] + bbox2[2]
        x1_start = bbox1[0]
        
        # Horizontal gap between regions
        gap = min(abs(x1_end - x2_start), abs(x2_end - x1_start))
        avg_width = (bbox1[2] + bbox2[2]) / 2
        
        # On same line if gap < 100% of average width
        return gap < avg_width
    
    def _merge_region_group_advanced(self, regions: List[TextRegion]) -> TextRegion:
        """Advanced merging of region groups with intelligent weighting"""
        
        if len(regions) == 1:
            return regions[0]
        
        # Calculate union bounding box
        min_x = min(r.bbox[0] for r in regions)
        min_y = min(r.bbox[1] for r in regions)
        max_x = max(r.bbox[0] + r.bbox[2] for r in regions)
        max_y = max(r.bbox[1] + r.bbox[3] for r in regions)
        
        merged_bbox = (min_x, min_y, max_x - min_x, max_y - min_y)
        
        # Weighted confidence calculation
        total_area = sum(r.area for r in regions)
        total_conf_weighted = sum(r.confidence * r.area for r in regions)
        
        # Base weighted confidence
        weighted_confidence = total_conf_weighted / total_area if total_area > 0 else 0.5
        
        # Bonus for consistent high-confidence regions
        high_conf_count = sum(1 for r in regions if r.confidence > 0.8)
        if high_conf_count > len(regions) * 0.7:  # 70% are high confidence
            weighted_confidence *= 1.1
        
        # Method selection with priority
        method_priority = {
            "craft": 10, "east": 9,
            "merged_craft": 8, "merged_east": 8,
            "mser": 7, "merged_mser": 7,
            "morphological": 6, "gradient": 5,
            "aggressive_mser": 4, "aggressive_contour": 3,
            "fallback_contour": 2, "fallback_grid": 1,
            "full_image_fallback": 0
        }
        
        best_method_region = max(regions, key=lambda r: method_priority.get(r.method, 0))
        final_method = f"advanced_merged_{best_method_region.method}"
        
        # Calculate weighted angle
        total_angle_weight = sum(r.confidence for r in regions)
        weighted_angle = sum(r.angle * r.confidence for r in regions)
        avg_angle = weighted_angle / total_angle_weight if total_angle_weight > 0 else 0.0
        
        return TextRegion(
            bbox=merged_bbox,
            confidence=min(0.99, weighted_confidence),  # Cap at 0.99
            angle=avg_angle,
            method=final_method,
            text_type=best_method_region.text_type
        )
    
    def _final_quality_filter(self, regions: List[TextRegion]) -> List[TextRegion]:
        """Final quality filtering - remove remaining noise"""
        
        filtered = []
        
        # Calculate statistics for adaptive filtering
        if not regions:
            return filtered
        
        confidences = [r.confidence for r in regions]
        areas = [r.area for r in regions]
        
        median_confidence = np.median(confidences)
        median_area = np.median(areas)
        
        for region in regions:
            # Adaptive confidence threshold based on median
            adaptive_threshold = max(self.confidence_threshold, median_confidence * 0.7)
            
            if region.confidence < adaptive_threshold:
                continue
            
            # Remove regions that are way too small compared to median
            if region.area < median_area * 0.1:
                continue
            
            # Remove regions that are way too large (likely merged incorrectly)
            if region.area > median_area * 50:
                continue
            
            filtered.append(region)
        
        return filtered
    
    # ==========================================================================
    # LEGACY COMPATIBILITY METHODS
    # ==========================================================================
    
    def _post_process_regions(self, regions: List[TextRegion], 
                            image_shape: Tuple[int, int]) -> List[TextRegion]:
        """Legacy post-processing wrapper - calls modern version"""
        return self._modern_post_process_regions(regions, image_shape)
    
    def _merge_similar_regions(self, regions: List[TextRegion]) -> List[TextRegion]:
        """Legacy merge method - calls advanced version"""
        return self._advanced_region_merging(regions)
    
    def _apply_nms(self, regions: List[TextRegion]) -> List[TextRegion]:
        """Apply Non-Maximum Suppression to remove duplicate detections"""
        if len(regions) <= 1:
            return regions
        
        try:
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
                # If NMS filters out everything, return top regions by confidence
                sorted_regions = sorted(regions, key=lambda r: r.confidence, reverse=True)
                return sorted_regions[:max(1, len(regions) // 4)]
        
        except Exception as e:
            self.logger.warning(f"NMS failed: {e}, returning sorted regions")
            return sorted(regions, key=lambda r: r.confidence, reverse=True)
    
    # ==========================================================================
    # ENHANCED FUNCTIONALITY & UTILITIES
    # ==========================================================================
    
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
        """Visualize detected text regions on image with confidence-based coloring"""
        vis_image = image.copy()
        
        # Color mapping based on confidence levels
        def get_confidence_color(confidence: float) -> Tuple[int, int, int]:
            """Get color based on confidence level"""
            if confidence > 0.9:
                return (0, 255, 0)      # Bright green - very high confidence
            elif confidence > 0.8:
                return (0, 200, 50)     # Green - high confidence  
            elif confidence > 0.7:
                return (0, 150, 150)    # Yellow-green - good confidence
            elif confidence > 0.6:
                return (0, 100, 255)    # Orange - medium confidence
            else:
                return (0, 0, 255)      # Red - low confidence
        
        for i, region in enumerate(regions):
            x, y, w, h = region.bbox
            
            # Get color based on confidence
            color = get_confidence_color(region.confidence)
            
            # Draw bounding box with thickness based on confidence
            thickness = 3 if region.confidence > 0.8 else 2
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, thickness)
            
            # Draw confidence score
            confidence_text = f"{region.confidence:.2f}"
            cv2.putText(vis_image, confidence_text, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Draw region number
            cv2.putText(vis_image, str(i), (x + 5, y + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Add confidence legend
        legend_y = 30
        legend_items = [
            ("> 0.9", (0, 255, 0)),
            ("> 0.8", (0, 200, 50)),
            ("> 0.7", (0, 150, 150)),
            ("> 0.6", (0, 100, 255)),
            ("≤ 0.6", (0, 0, 255))
        ]
        
        for text, color in legend_items:
            cv2.putText(vis_image, text, (10, legend_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            legend_y += 20
        
        if output_path:
            cv2.imwrite(output_path, vis_image)
        
        return vis_image
    
    def export_regions_to_json(self, regions: List[TextRegion], output_path: str):
        """Export detected regions to JSON format"""
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
    
    def detect_text_regions_dict(self, image: np.ndarray) -> Dict[str, Any]:
        """Enhanced detection method that returns results in dictionary format"""
        try:
            # Use the modern optimized detection pipeline
            detected_regions = self.detect_text_regions(image)
            
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
                'detection_method': 'modern_optimized',
                'success': True
            }
            
            self.logger.info(f"Modern detection pipeline produced {len(result_regions)} regions")
            return result
            
        except Exception as e:
            self.logger.error(f"Modern detection failed: {e}")
            return {
                'regions': [],
                'total_regions': 0,
                'detection_method': 'failed',
                'success': False,
                'error': str(e)
            }
    
    def cleanup(self):
        """Cleanup detector resources"""
        if self.craft_detector and hasattr(self.craft_detector, 'model'):
            if self.craft_detector.model is not None:
                del self.craft_detector.model
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        self.logger.info("Modern text detector cleanup completed")
   
    def _parallel_detection_pipeline(self, image: np.ndarray) -> List[TextRegion]:
        """Execute parallel detection on image tiles with timeout protection"""
        
        # Create overlapping tiles
        tiles, tile_info = self._create_image_tiles(image)
        
        # Process tiles in parallel with timeout
        all_regions = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tiles for processing
            future_to_tile = {}
            for i, (tile, info) in enumerate(zip(tiles, tile_info)):
                future = executor.submit(self._process_tile, tile, info, i)
                future_to_tile[future] = (i, info)
            
            # Collect results with timeout
            tile_results = {}
            for future in future_to_tile:
                tile_idx, tile_info = future_to_tile[future]
                try:
                    # ADD TIMEOUT - prevent hanging
                    regions = future.result(timeout=30)  # 30 second timeout per tile
                    tile_results[tile_idx] = (regions, tile_info)
                except TimeoutError:
                    self.logger.warning(f"Tile {tile_idx} processing timed out")
                    tile_results[tile_idx] = ([], tile_info)
                except Exception as e:
                    self.logger.warning(f"Tile {tile_idx} processing failed: {e}")
                    tile_results[tile_idx] = ([], tile_info)
        
        # Merge results with progress logging
        self.logger.info(f"Starting merge of {len(tile_results)} tile results...")
        merged_regions = self._merge_tile_results(tile_results, image.shape[:2])
        
        self.logger.info(f"Parallel processing complete: {len(tiles)} tiles → {len(merged_regions)} regions")
        return merged_regions
    
    def detect_text_regions_with_boundary_detection(self, image: np.ndarray) -> List[TextRegion]:
        """Enhanced detection with document boundary masking"""
        
        start_time = time.time()
        
        if image is None or image.size == 0:
            return []
        
        try:
            # Step 1: Detect document boundaries
            boundary_detector = DocumentBoundaryDetector(self.config.get("boundary_detection", {}))
            document_mask = boundary_detector.detect_document_boundary(image)
            
            if document_mask is not None:
                # Step 2: Apply mask to image
                if len(image.shape) == 3:
                    # For color images
                    masked_image = image.copy()
                    background_pixels = document_mask == 0
                    masked_image[background_pixels] = [128, 128, 128]  # Gray background
                else:
                    # For grayscale
                    masked_image = image.copy()
                    masked_image[document_mask == 0] = 128
                    
                self.logger.info("Applied document boundary mask")
            else:
                masked_image = image
                document_mask = np.ones(image.shape[:2], dtype=np.uint8) * 255
                self.logger.info("No document boundary detected, using full image")
            
            # Step 3: Detect text regions on masked image
            regions = self.detect_text_regions(masked_image)
            
            # Step 4: Filter regions outside document boundary
            filtered_regions = []
            for region in regions:
                x, y, w, h = region.bbox
                
                # Check if region center is within document boundary
                center_x, center_y = x + w // 2, y + h // 2
                
                if (0 <= center_y < document_mask.shape[0] and 
                    0 <= center_x < document_mask.shape[1] and
                    document_mask[center_y, center_x] > 128):
                    
                    filtered_regions.append(region)
            
            processing_time = time.time() - start_time
            self.logger.info(f"Boundary-aware detection: {len(regions)} → {len(filtered_regions)} regions in {processing_time:.3f}s")
            
            return filtered_regions
            
        except Exception as e:
            self.logger.error(f"Boundary-aware detection failed: {e}")
            return self.detect_text_regions(image)
class DocumentBoundaryDetector:
    """Detect document boundaries to mask out background noise"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = logging.getLogger("DocumentBoundaryDetector")
        
        # Detection parameters
        self.min_document_area_ratio = config.get("min_document_area_ratio", 0.1)
        self.max_document_area_ratio = config.get("max_document_area_ratio", 0.8)
        self.gaussian_blur_size = config.get("gaussian_blur_size", 5)
        self.canny_low = config.get("canny_low", 50)
        self.canny_high = config.get("canny_high", 150)
        
    def detect_document_boundary(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect the main document boundary and return a mask
        Returns: Binary mask where 1 = document area, 0 = background
        """
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        h, w = gray.shape
        
        try:
            # Method 1: Edge-based document detection
            boundary = self._detect_by_edges(gray)
            if boundary is not None:
                return boundary
                
            # Method 2: Color/intensity based detection
            boundary = self._detect_by_intensity(gray)
            if boundary is not None:
                return boundary
                
            # Method 3: Fallback - assume document is center region
            return self._create_center_mask(gray)
            
        except Exception as e:
            self.logger.error(f"Document boundary detection failed: {e}")
            return self._create_center_mask(gray)
    
    def _detect_by_edges(self, gray: np.ndarray) -> Optional[np.ndarray]:
        """Detect document by finding strong rectangular edges"""
        
        # Blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (self.gaussian_blur_size, self.gaussian_blur_size), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, self.canny_low, self.canny_high)
        
        # Morphological operations to connect edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        h, w = gray.shape
        image_area = h * w
        
        # Find the largest rectangular contour
        best_contour = None
        best_score = 0
        
        for contour in contours:
            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Check if it's roughly rectangular (4-8 points)
            if len(approx) >= 4:
                area = cv2.contourArea(contour)
                area_ratio = area / image_area
                
                # Must be reasonable size for a document
                if (self.min_document_area_ratio < area_ratio < self.max_document_area_ratio):
                    # Score based on area and rectangularity
                    hull_area = cv2.contourArea(cv2.convexHull(contour))
                    rectangularity = area / hull_area if hull_area > 0 else 0
                    
                    score = area_ratio * rectangularity
                    
                    if score > best_score:
                        best_score = score
                        best_contour = contour
        
        if best_contour is not None:
            # Create mask from best contour
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.fillPoly(mask, [best_contour], 255)
            return mask
            
        return None
    
    def _detect_by_intensity(self, gray: np.ndarray) -> Optional[np.ndarray]:
        """Detect document by intensity differences from background"""
        
        h, w = gray.shape
        
        # Calculate intensity statistics
        mean_intensity = np.mean(gray)
        std_intensity = np.std(gray)
        
        # Threshold to separate document from background
        # Documents are usually brighter than wood background
        if mean_intensity > 128:  # Bright document
            threshold = mean_intensity - std_intensity * 0.5
            document_mask = gray > threshold
        else:  # Dark document
            threshold = mean_intensity + std_intensity * 0.5
            document_mask = gray < threshold
        
        # Clean up mask with morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        cleaned = cv2.morphologyEx(document_mask.astype(np.uint8) * 255, 
                                  cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        
        # Find largest connected component
        num_labels, labels = cv2.connectedComponents(cleaned)
        if num_labels < 2:
            return None
            
        # Find largest component (excluding background)
        largest_area = 0
        largest_label = 0
        
        for label in range(1, num_labels):
            area = np.sum(labels == label)
            if area > largest_area:
                largest_area = area
                largest_label = label
        
        # Check if largest component is reasonable size
        area_ratio = largest_area / (h * w)
        if self.min_document_area_ratio < area_ratio < self.max_document_area_ratio:
            return (labels == largest_label).astype(np.uint8) * 255
            
        return None
    
    def _create_center_mask(self, gray: np.ndarray) -> np.ndarray:
        """Fallback: create mask for center region of image"""
        h, w = gray.shape
        
        # Create mask for center 70% of image
        margin_h = int(h * 0.15)
        margin_w = int(w * 0.15)
        
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[margin_h:h-margin_h, margin_w:w-margin_w] = 255
        
        self.logger.info("Using fallback center mask for document boundary")
        return mask
    
# Alias for backward compatibility
TextDetector = AdvancedTextDetector