# src/advanced_ocr/preprocessing/content_classifier.py - FIXED VERSION
"""
Advanced OCR Content Classification Module
FIXED: Image format handling (PIL Image to numpy array conversion)
FIXED: Fallback when sklearn is not available
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional, Union
from dataclasses import dataclass
from PIL import Image

from ..utils.model_utils import ModelLoader
from ..utils.image_utils import ImageProcessor
from ..config import OCRConfig
from ..utils.logger import OCRLogger


@dataclass
class ContentClassification:
    content_type: str
    confidence_scores: Dict[str, float]
    dominant_type: str
    processing_time: float
    
    @property
    def confidence(self) -> float:
        """Overall confidence - highest score among content types"""
        return max(self.confidence_scores.values()) if self.confidence_scores else 0.0


class ContentClassifier:
    """
    Intelligent content type classifier for OCR optimization
    FIXED: Handles both PIL Images and numpy arrays
    FIXED: Graceful sklearn fallback
    """
    
    def __init__(self, config: OCRConfig):
        self.config = config
        self.logger = OCRLogger(__name__)
        # REMOVED: model_loader - not needed for heuristic classification
        
        # Classification parameters
        self.confidence_threshold = getattr(config.preprocessing, 'classification_confidence_threshold', 0.7)
        self.mixed_threshold = getattr(config.preprocessing, 'classification_mixed_threshold', 0.2)
        
        # FIXED: No model loading needed - using heuristic classification only
        self._model = None  # Always None - using heuristics
        self.logger.info("Content classifier initialized with heuristic classification")
        
    def classify_content(self, image: Union[np.ndarray, Image.Image]) -> ContentClassification:
        """
        Classify document content type using heuristic analysis only
        
        Args:
            image: Image from image_processor.py (PIL Image or numpy array)
            
        Returns:
            ContentClassification with scores and dominant type
        """
        start_time = self.logger.start_timer()
        
        try:
            # Convert PIL Image to numpy array if needed
            if isinstance(image, Image.Image):
                image_array = np.array(image)
            else:
                image_array = image
                
            # Extract classification features
            features = self._extract_features(image_array)
            
            # FIXED: Always use heuristic classification
            scores = self._heuristic_classify(features, image_array)
            
            # Determine dominant type and final classification
            dominant_type = max(scores, key=scores.get)
            content_type = self._determine_content_type(scores)
            
            processing_time = self.logger.end_timer(start_time)
            
            result = ContentClassification(
                content_type=content_type,
                confidence_scores=scores,
                dominant_type=dominant_type,
                processing_time=processing_time
            )
            
            self.logger.debug(
                f"Content classified as {content_type} "
                f"(dominant: {dominant_type}, scores: {scores})"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Content classification failed: {e}")
            # Return safe defaults
            return self._get_default_classification()
    
    def _extract_features(self, image: np.ndarray) -> Dict[str, float]:
        """Extract features for content type classification"""
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # FIXED: PIL uses RGB, cv2 expects RGB
        else:
            gray = image.copy()
        
        # Feature extraction for classification
        features = {}
        
        # 1. Edge density (printed text has more consistent edges)
        edges = cv2.Canny(gray, 50, 150)
        features['edge_density'] = np.sum(edges > 0) / edges.size
        
        # 2. Line regularity (printed text has more regular lines)
        features['line_regularity'] = self._calculate_line_regularity(gray)
        
        # 3. Character consistency (printed text more uniform)
        features['char_consistency'] = self._calculate_character_consistency(gray)
        
        # 4. Stroke width variation (handwritten varies more)
        features['stroke_variation'] = self._calculate_stroke_variation(gray)
        
        # 5. Text density patterns
        features['text_density'] = self._calculate_text_density(gray)
        
        return features
    
    def _calculate_line_regularity(self, image: np.ndarray) -> float:
        """Calculate line regularity score (higher = more printed-like)"""
        # Horizontal projection to find line patterns
        h_proj = np.sum(image < 128, axis=1)  # Sum of dark pixels per row
        
        # FIXED: Handle scipy import gracefully
        try:
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(h_proj, height=np.mean(h_proj))
        except ImportError:
            # Simple peak detection without scipy
            mean_val = np.mean(h_proj)
            peaks = []
            for i in range(1, len(h_proj) - 1):
                if h_proj[i] > mean_val and h_proj[i] > h_proj[i-1] and h_proj[i] > h_proj[i+1]:
                    peaks.append(i)
            peaks = np.array(peaks)
        
        if len(peaks) < 2:
            return 0.5  # Neutral score
        
        # Calculate line spacing consistency
        line_spaces = np.diff(peaks)
        if len(line_spaces) == 0:
            return 0.5
        
        # Higher consistency = more regular = more printed
        regularity = 1.0 - (np.std(line_spaces) / (np.mean(line_spaces) + 1e-6))
        return max(0.0, min(1.0, regularity))
    
    def _calculate_character_consistency(self, image: np.ndarray) -> float:
        """Calculate character consistency (higher = more printed-like)"""
        # Find connected components (characters)
        contours, _ = cv2.findContours(
            (image < 128).astype(np.uint8), 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        if len(contours) < 5:  # Need sufficient characters
            return 0.5
        
        # Calculate character size consistency
        areas = [cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 20]
        
        if len(areas) < 5:
            return 0.5
        
        # Higher consistency = more printed
        consistency = 1.0 - (np.std(areas) / (np.mean(areas) + 1e-6))
        return max(0.0, min(1.0, consistency))
    
    def _calculate_stroke_variation(self, image: np.ndarray) -> float:
        """Calculate stroke width variation (lower = more printed-like)"""
        # Simple stroke width analysis using distance transform
        binary = (image < 128).astype(np.uint8)
        
        if np.sum(binary) == 0:
            return 0.5
        
        # Distance transform to find stroke widths
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        
        # Find stroke centers (local maxima in distance transform)
        stroke_widths = dist[dist > 1].flatten()
        
        if len(stroke_widths) == 0:
            return 0.5
        
        # Calculate variation (lower = more consistent = more printed)
        variation = np.std(stroke_widths) / (np.mean(stroke_widths) + 1e-6)
        
        # Invert so lower variation gives higher score
        return max(0.0, min(1.0, 1.0 - variation))
    
    def _calculate_text_density(self, image: np.ndarray) -> float:
        """Calculate text density patterns"""
        binary = (image < 128).astype(np.uint8)
        text_ratio = np.sum(binary) / binary.size
        
        # Normalize to 0-1 range
        return min(1.0, text_ratio * 10)  # Scale factor for typical text density
    
    def _heuristic_classify(self, features: Dict[str, float], image: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Fallback heuristic classification"""
        
        # Heuristic scoring based on feature analysis
        printed_score = (
            features['line_regularity'] * 0.3 +
            features['char_consistency'] * 0.3 +
            features['stroke_variation'] * 0.2 +
            features['edge_density'] * 0.2
        )
        
        # Handwritten is inverse of many printed characteristics
        handwritten_score = (
            (1.0 - features['line_regularity']) * 0.3 +
            (1.0 - features['char_consistency']) * 0.3 +
            (1.0 - features['stroke_variation']) * 0.4
        )
        
        # Mixed content has moderate scores
        mixed_score = 1.0 - abs(printed_score - handwritten_score)
        
        # Normalize scores
        total = printed_score + handwritten_score + mixed_score
        if total == 0:
            total = 1.0
        
        return {
            'printed': printed_score / total,
            'handwritten': handwritten_score / total,
            'mixed': mixed_score / total
        }
    
    def _determine_content_type(self, scores: Dict[str, float]) -> str:
        """Determine final content type from scores"""
        max_score = max(scores.values())
        
        # Check for mixed content (no dominant type)
        if max_score < self.confidence_threshold:
            return 'mixed'
        
        # Check for mixed content (scores too close)
        sorted_scores = sorted(scores.values(), reverse=True)
        if len(sorted_scores) >= 2 and sorted_scores[0] - sorted_scores[1] < self.mixed_threshold:
            return 'mixed'
        
        # Return dominant type
        return max(scores, key=scores.get)
    
    def _get_default_classification(self) -> ContentClassification:
        """Return safe default classification on failure"""
        return ContentClassification(
            content_type='mixed',  # Safe default - enables all engines
            confidence_scores={'printed': 0.33, 'handwritten': 0.33, 'mixed': 0.34},
            dominant_type='mixed',
            processing_time=0.0
        )