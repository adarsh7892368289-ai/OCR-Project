# src/preprocessing/text_detector.py
import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any
import skimage.restoration
from skimage import filters, morphology, measure
from scipy import ndimage

class TextDetector:
    """Detect text regions in images"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.min_text_size = config.get("min_text_size", 10)
        self.max_text_size = config.get("max_text_size", 300)
        
    def detect_text_regions(self, image: np.ndarray) -> list:
        """Detect text regions using MSER and morphological operations"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # MSER text detection
        mser = cv2.MSER_create(
            _min_area=self.min_text_size,
            _max_area=self.max_text_size * 100
        )
        
        regions, _ = mser.detectRegions(gray)
        
        # Filter and convert regions to bounding boxes
        text_regions = []
        for region in regions:
            x, y, w, h = cv2.boundingRect(region.reshape(-1, 1, 2))
            
            # Filter based on aspect ratio and size
            aspect_ratio = w / h
            if 0.1 < aspect_ratio < 10 and w > self.min_text_size and h > self.min_text_size:
                text_regions.append((x, y, w, h))
                
        # Merge overlapping regions
        merged_regions = self._merge_overlapping_regions(text_regions)
        
        return merged_regions
        
    def _merge_overlapping_regions(self, regions: list) -> list:
        """Merge overlapping text regions"""
        if not regions:
            return regions
            
        # Sort by x coordinate
        regions = sorted(regions, key=lambda r: r[0])
        merged = [regions[0]]
        
        for current in regions[1:]:
            last = merged[-1]
            
            # Check for overlap
            if self._regions_overlap(last, current):
                # Merge regions
                x1 = min(last[0], current[0])
                y1 = min(last[1], current[1])
                x2 = max(last[0] + last[2], current[0] + current[2])
                y2 = max(last[1] + last[3], current[1] + current[3])
                merged[-1] = (x1, y1, x2 - x1, y2 - y1)
            else:
                merged.append(current)
                
        return merged
        
    def _regions_overlap(self, region1: tuple, region2: tuple) -> bool:
        """Check if two regions overlap"""
        x1, y1, w1, h1 = region1
        x2, y2, w2, h2 = region2
        
        return not (x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1)