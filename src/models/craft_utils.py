# src/models/craft_utils.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple, List, Dict, Any
import logging

class VGG16FeatureExtractor(nn.Module):
    """VGG16 backbone for CRAFT text detector"""
    
    def __init__(self):
        super(VGG16FeatureExtractor, self).__init__()
        
        # VGG16 layers
        self.slice1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.slice2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.slice3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.slice4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.slice5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    
    def forward(self, x):
        h1 = self.slice1(x)      # 1/2
        h2 = self.slice2(h1)     # 1/4  
        h3 = self.slice3(h2)     # 1/8
        h4 = self.slice4(h3)     # 1/16
        h5 = self.slice5(h4)     # 1/32
        
        return [h1, h2, h3, h4, h5]

class FeatureFusionNetwork(nn.Module):
    """Feature Pyramid Network for multi-scale feature fusion"""
    
    def __init__(self):
        super(FeatureFusionNetwork, self).__init__()
        
        # Lateral connections
        self.lateral5 = nn.Conv2d(512, 256, kernel_size=1)
        self.lateral4 = nn.Conv2d(512, 256, kernel_size=1)  
        self.lateral3 = nn.Conv2d(256, 256, kernel_size=1)
        self.lateral2 = nn.Conv2d(128, 256, kernel_size=1)
        
        # Smooth layers
        self.smooth5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.smooth4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        
        # Final feature layers
        self.conv_cls = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 2, kernel_size=1)  # text + link
        )
    
    def forward(self, features):
        c2, c3, c4, c5 = features[1], features[2], features[3], features[4]
        
        # Top-down pathway
        p5 = self.lateral5(c5)
        p4 = self._upsample_add(p5, self.lateral4(c4))
        p3 = self._upsample_add(p4, self.lateral3(c3))
        p2 = self._upsample_add(p3, self.lateral2(c2))
        
        # Smooth layers
        p5 = self.smooth5(p5)
        p4 = self.smooth4(p4)
        p3 = self.smooth3(p3)  
        p2 = self.smooth2(p2)
        
        # Merge features
        feature = self._merge_features([p2, p3, p4, p5])
        
        # Final prediction
        output = self.conv_cls(feature)
        
        return output
    
    def _upsample_add(self, x, y):
        """Upsample x and add to y"""
        _, _, H, W = y.size()
        upsampled = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return upsampled + y
    
    def _merge_features(self, features):
        """Merge multi-scale features"""
        # Upsample all features to the size of p2
        _, _, H, W = features[0].size()
        
        upsampled_features = []
        for feature in features:
            if feature.size(2) != H or feature.size(3) != W:
                upsampled = F.interpolate(feature, size=(H, W), mode='bilinear', align_corners=True)
                upsampled_features.append(upsampled)
            else:
                upsampled_features.append(feature)
        
        # Concatenate and reduce dimensions
        merged = torch.cat(upsampled_features, dim=1)
        merged = F.conv2d(merged, torch.ones(256, merged.size(1), 1, 1).to(merged.device) / merged.size(1))
        
        return merged

class CRAFTModel(nn.Module):
    """Complete CRAFT text detection model"""
    
    def __init__(self, pretrained: bool = True):
        super(CRAFTModel, self).__init__()
        
        self.backbone = VGG16FeatureExtractor()
        self.fpn = FeatureFusionNetwork()
        
        # Initialize weights
        self._initialize_weights()
    
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        
        # Feature fusion and prediction
        output = self.fpn(features)
        
        # Split text and link predictions
        text_map = torch.sigmoid(output[:, 0:1, :, :])
        link_map = torch.sigmoid(output[:, 1:2, :, :])
        
        return text_map, link_map
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

class CRAFTPostProcessor:
    """Post-processing utilities for CRAFT model outputs"""
    
    def __init__(self, text_threshold: float = 0.7, link_threshold: float = 0.4,
                 low_text: float = 0.4, min_size: int = 10):
        self.text_threshold = text_threshold
        self.link_threshold = link_threshold
        self.low_text = low_text
        self.min_size = min_size
        
        self.logger = logging.getLogger("CRAFT.PostProcessor")
    
    def get_text_boxes(self, text_map: np.ndarray, link_map: np.ndarray,
                      text_threshold: float = None, link_threshold: float = None) -> List[np.ndarray]:
        """Extract text boxes from CRAFT output maps"""
        
        if text_threshold is None:
            text_threshold = self.text_threshold
        if link_threshold is None:
            link_threshold = self.link_threshold
        
        # Threshold maps
        text_mask = text_map > text_threshold
        link_mask = link_map > link_threshold
        
        # Combine text and link regions
        text_score_comb = np.clip(text_map + link_mask * text_map, 0, 1)
        
        # Find connected components
        labeled = self._watershed_detector(text_score_comb, text_mask)
        
        # Extract bounding boxes
        boxes = self._get_boxes_from_labeled_image(labeled, text_score_comb)
        
        return boxes
    
    def _watershed_detector(self, text_map: np.ndarray, text_mask: np.ndarray) -> np.ndarray:
        """Apply watershed algorithm for text region separation"""
        from skimage.segmentation import watershed
        from skimage.feature import peak_local_maxima
        
        # Find local maxima as markers
        coordinates = peak_local_maxima(text_map, min_distance=5, threshold_abs=self.low_text)
        markers = np.zeros_like(text_map, dtype=int)
        markers[tuple(coordinates.T)] = np.arange(1, len(coordinates) + 1)
        
        # Apply watershed
        labels = watershed(-text_map, markers, mask=text_mask)
        
        return labels
    
    def _get_boxes_from_labeled_image(self, labeled: np.ndarray, score_map: np.ndarray) -> List[np.ndarray]:
        """Extract bounding boxes from labeled image"""
        from skimage.measure import regionprops
        
        boxes = []
        
        for region in regionprops(labeled):
            if region.area < self.min_size:
                continue
            
            # Get region coordinates
            coords = region.coords
            
            # Calculate rotated bounding box
            try:
                rect = cv2.minAreaRect(coords.astype(np.float32))
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                
                # Validate box
                if self._is_valid_box(box, score_map.shape):
                    boxes.append(box)
                    
            except Exception as e:
                self.logger.warning(f"Failed to extract box for region: {e}")
                continue
        
        return boxes
    
    def _is_valid_box(self, box: np.ndarray, image_shape: Tuple[int, int]) -> bool:
        """Validate if box is within image bounds and has reasonable size"""
        h, w = image_shape
        
        # Check if all points are within bounds
        if np.any(box < 0) or np.any(box[:, 0] >= w) or np.any(box[:, 1] >= h):
            return False
        
        # Check minimum area
        area = cv2.contourArea(box)
        if area < self.min_size:
            return False
        
        return True
    
    def adjust_result_coordinates(self, polys: List[np.ndarray], 
                                ratio_w: float, ratio_h: float,
                                ratio_net: float = 2) -> List[np.ndarray]:
        """Adjust coordinates back to original image size"""
        
        adjusted_polys = []
        
        for poly in polys:
            poly = np.array(poly)
            poly *= ratio_net
            poly[:, 0] *= ratio_w
            poly[:, 1] *= ratio_h
            adjusted_polys.append(poly)
        
        return adjusted_polys

def download_craft_model(model_dir: str = "./models/craft") -> str:
    """Download pre-trained CRAFT model"""
    import os
    import urllib.request
    from pathlib import Path
    
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = model_dir / "craft_mlt_25k.pth"
    
    if model_path.exists():
        print(f"CRAFT model already exists at {model_path}")
        return str(model_path)
    
    # Download URL (this would be the actual CRAFT model URL)
    model_url = "https://example.com/craft_model.pth"  # Replace with actual URL
    
    try:
        print(f"Downloading CRAFT model to {model_path}")
        urllib.request.urlretrieve(model_url, model_path)
        print("CRAFT model downloaded successfully")
        return str(model_path)
    except Exception as e:
        print(f"Failed to download CRAFT model: {e}")
        print("Please download manually from the official CRAFT repository")
        return ""

def create_craft_config() -> Dict[str, Any]:
    """Create default CRAFT configuration"""
    return {
        "text_threshold": 0.7,
        "link_threshold": 0.4,
        "low_text": 0.4,
        "canvas_size": 1280,
        "mag_ratio": 1.5,
        "min_size": 10,
        "cuda": True,
        "model_path": "./models/craft/craft_mlt_25k.pth"
    }

def preprocess_image_for_craft(image: np.ndarray, canvas_size: int = 1280, 
                              mag_ratio: float = 1.5) -> Tuple[np.ndarray, float, float, float]:
    """Preprocess image for CRAFT inference"""
    
    # Get image dimensions
    img_height, img_width, _ = image.shape
    
    # Calculate target size
    target_size = canvas_size * mag_ratio
    
    # Calculate ratios
    ratio_h = target_size / img_height
    ratio_w = target_size / img_width
    
    # Use minimum ratio to maintain aspect ratio
    ratio = min(ratio_h, ratio_w)
    
    # Calculate new dimensions
    new_width = int(img_width * ratio)
    new_height = int(img_height * ratio)
    
    # Resize image
    resized_img = cv2.resize(image, (new_width, new_height))
    
    # Pad to target size
    target_width = int(target_size)
    target_height = int(target_size)
    
    padded_img = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    padded_img[:new_height, :new_width] = resized_img
    
    # Normalize for neural network
    normalized_img = padded_img.astype(np.float32) / 255.0
    normalized_img -= np.array([0.485, 0.456, 0.406])
    normalized_img /= np.array([0.229, 0.224, 0.225])
    
    # Convert to CHW format
    normalized_img = np.transpose(normalized_img, (2, 0, 1))
    
    return normalized_img, ratio_w, ratio_h, ratio