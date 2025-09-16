"""
Advanced OCR Content Classification Module

This module provides intelligent content type classification for the advanced OCR system.
It analyzes document images to determine whether they contain printed text, handwritten text,
or a mixture of both, enabling intelligent engine selection and processing optimization.

The module focuses on:
- Lightweight ML-based content classification using CNN models
- Feature extraction from preprocessed images for content pattern analysis
- Confidence scoring for printed, handwritten, and mixed content types
- Fallback heuristic classification when ML models are unavailable
- Integration with engine coordinator for optimal OCR engine selection

Classes:
    ContentClassification: Data container for classification results and scores
    ContentClassifier: Main classifier with ML and heuristic classification methods

Functions:
    None (all functionality encapsulated in classes)

Example:
    >>> from advanced_ocr.preprocessing.content_classifier import ContentClassifier
    >>> from advanced_ocr.config import OCRConfig
    >>> classifier = ContentClassifier(OCRConfig())
    >>> result = classifier.classify_content(image)
    >>> print(f"Content type: {result.content_type}, Confidence: {result.confidence_scores}")

    >>> # Access detailed scores
    >>> scores = result.confidence_scores
    >>> print(f"Printed: {scores['printed']:.3f}, Handwritten: {scores['handwritten']:.3f}")
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

from ..utils.model_utils import ModelManager
from ..utils.image_utils import ImageProcessor
from ..config import OCRConfig
from ..utils.logger import Logger


@dataclass
class ContentClassification:
    """Content classification result container"""
    content_type: str  # 'printed', 'handwritten', 'mixed'
    confidence_scores: Dict[str, float]  # {'printed': 0.8, 'handwritten': 0.15, 'mixed': 0.05}
    dominant_type: str  # Primary content type
    processing_time: float


class ContentClassifier:
    """
    Intelligent content type classifier for OCR optimization
    
    Responsibilities:
    - Load classification model via model_utils
    - Analyze preprocessed images for content patterns  
    - Return detailed classification scores
    - Support engine_coordinator decision making
    """
    
    def __init__(self, config: OCRConfig):
        self.config = config
        self.logger = Logger(__name__)
        self.model_manager = ModelManager(config)
        
        # Classification parameters
        self.confidence_threshold = config.classification.confidence_threshold
        self.mixed_threshold = config.classification.mixed_threshold
        
        # Model lazy loading
        self._model = None
        self._feature_extractor = None
        
    def _load_model(self) -> None:
        """Lazy load classification model and feature extractor"""
        if self._model is None:
            try:
                # Load lightweight CNN model for content classification
                self._model = self.model_manager.load_model(
                    'content_classifier',
                    model_type='sklearn',  # Using sklearn for lightweight deployment
                    cache_key='content_classifier_v1'
                )
                self.logger.info("Content classifier model loaded successfully")
                
            except Exception as e:
                self.logger.error(f"Failed to load classification model: {e}")
                # Fallback to heuristic classification
                self._model = None
    
    def classify_content(self, image: np.ndarray) -> ContentClassification:
        """
        Classify document content type for intelligent engine selection
        
        Args:
            image: Preprocessed image from image_processor.py
            
        Returns:
            ContentClassification with scores and dominant type
        """
        start_time = self.logger.start_timer()
        
        try:
            # Ensure model is loaded
            self._load_model()
            
            # Extract classification features
            features = self._extract_features(image)
            
            if self._model is not None:
                # ML-based classification
                scores = self._ml_classify(features)
            else:
                # Fallback heuristic classification
                scores = self._heuristic_classify(features, image)
            
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
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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
        
        # Find peaks (text lines) and valleys (gaps)
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(h_proj, height=np.mean(h_proj))
        
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
            cv2.RETR_CHAIN_APPROX_SIMPLE
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
    
    def _ml_classify(self, features: Dict[str, float]) -> Dict[str, float]:
        """ML-based classification using trained model"""
        try:
            # Convert features to model input format
            feature_vector = np.array([
                features['edge_density'],
                features['line_regularity'], 
                features['char_consistency'],
                features['stroke_variation'],
                features['text_density']
            ]).reshape(1, -1)
            
            # Get prediction probabilities
            probabilities = self._model.predict_proba(feature_vector)[0]
            
            # Map to content types
            return {
                'printed': float(probabilities[0]),
                'handwritten': float(probabilities[1]), 
                'mixed': float(probabilities[2])
            }
            
        except Exception as e:
            self.logger.error(f"ML classification failed: {e}")
            return self._heuristic_classify(features, None)
    
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