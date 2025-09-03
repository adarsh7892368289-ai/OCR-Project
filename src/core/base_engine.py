# src/core/base_engine.py

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional, Union
import cv2
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
import time
import logging
from pathlib import Path

class TextType(Enum):
    """Text type classification"""
    PRINTED = "printed"
    HANDWRITTEN = "handwritten"
    MIXED = "mixed"
    UNKNOWN = "unknown"

class DetectionMethod(Enum):
    """Text detection method"""
    TRADITIONAL = "traditional"
    DEEP_LEARNING = "deep_learning"
    HYBRID = "hybrid"

@dataclass
class BoundingBox:
    """Enhanced bounding box with additional properties"""
    x: int
    y: int
    width: int
    height: int
    confidence: float = 0.0
    text_type: TextType = TextType.UNKNOWN
    rotation_angle: float = 0.0
    
    @property
    def center(self) -> Tuple[int, int]:
        return (self.x + self.width // 2, self.y + self.height // 2)
    
    @property
    def area(self) -> int:
        return self.width * self.height
    
    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height if self.height > 0 else 0
    
    def to_tuple(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.width, self.height)
    
    def intersects(self, other: 'BoundingBox') -> bool:
        """Check if this bbox intersects with another"""
        return not (self.x + self.width < other.x or 
                   other.x + other.width < self.x or
                   self.y + self.height < other.y or 
                   other.y + other.height < self.y)
    
    def iou(self, other: 'BoundingBox') -> float:
        """Calculate Intersection over Union"""
        if not self.intersects(other):
            return 0.0
        
        # Calculate intersection
        x1 = max(self.x, other.x)
        y1 = max(self.y, other.y)
        x2 = min(self.x + self.width, other.x + other.width)
        y2 = min(self.y + self.height, other.y + other.height)
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        union = self.area + other.area - intersection
        
        return intersection / union if union > 0 else 0.0

@dataclass
class TextRegion:
    """Enhanced text region with structure information"""
    bbox: BoundingBox
    text: str = ""
    confidence: float = 0.0
    text_type: TextType = TextType.UNKNOWN
    language: str = "en"
    reading_order: int = -1
    line_number: int = -1
    paragraph_id: int = -1
    structure_type: str = "text"  # text, header, footer, caption, etc.
    font_size: Optional[float] = None
    is_bold: bool = False
    is_italic: bool = False
    
    @property
    def is_valid(self) -> bool:
        """Check if region contains valid text"""
        return (len(self.text.strip()) > 0 and 
                self.confidence > 0.0 and 
                self.bbox.area > 0)

@dataclass 
class OCRResult:
    """Enhanced OCR result with detailed information"""
    text: str
    confidence: float
    bbox: BoundingBox
    text_regions: List[TextRegion] = field(default_factory=list)
    level: str = "line"  # char, word, line, block, page
    language: str = "en"
    text_type: TextType = TextType.UNKNOWN
    rotation_angle: float = 0.0
    processing_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        # Ensure bbox is BoundingBox object
        if isinstance(self.bbox, tuple):
            x, y, w, h = self.bbox
            self.bbox = BoundingBox(x, y, w, h, self.confidence)

@dataclass
class DocumentStructure:
    """Document structure analysis results"""
    headers: List[TextRegion] = field(default_factory=list)
    paragraphs: List[List[TextRegion]] = field(default_factory=list)
    lists: List[List[TextRegion]] = field(default_factory=list)
    tables: List[Dict[str, Any]] = field(default_factory=list)
    images: List[BoundingBox] = field(default_factory=list)
    reading_order: List[int] = field(default_factory=list)
    
@dataclass
class DocumentResult:
    """Comprehensive document OCR result"""
    full_text: str
    results: List[OCRResult]
    text_regions: List[TextRegion]
    document_structure: DocumentStructure
    processing_time: float
    engine_name: str
    image_stats: Dict[str, Any]
    confidence_score: float
    detected_languages: List[str] = field(default_factory=list)
    text_type: TextType = TextType.UNKNOWN
    preprocessing_steps: List[str] = field(default_factory=list)
    postprocessing_steps: List[str] = field(default_factory=list)
    
    @property
    def word_count(self) -> int:
        return len(self.full_text.split())
    
    @property
    def char_count(self) -> int:
        return len(self.full_text)
    
    @property
    def line_count(self) -> int:
        return len(self.full_text.split('\n'))

class BaseOCREngine(ABC):
    """Enhanced abstract base class for all OCR engines"""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}
        self.is_initialized = False
        self.supported_languages = []
        self.model_loaded = False
        self.logger = self._setup_logger()
        
        # Performance tracking
        self.processing_stats = {
            'total_processed': 0,
            'total_time': 0.0,
            'avg_confidence': 0.0,
            'errors': 0
        }
        
        # Engine capabilities
        self.supports_handwriting = False
        self.supports_multiple_languages = False
        self.supports_orientation_detection = False
        self.supports_structure_analysis = False
        self.max_image_size = (4096, 4096)
        self.min_image_size = (32, 32)
        
    def _setup_logger(self) -> logging.Logger:
        """Setup engine-specific logger"""
        logger = logging.getLogger(f"OCR.{self.name}")
        logger.setLevel(logging.INFO)
        return logger
        
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the OCR engine with enhanced error handling"""
        pass
        
    @abstractmethod
    def process_image(self, image: np.ndarray, **kwargs) -> DocumentResult:
        """Process an image and return comprehensive OCR results"""
        pass
        
    @abstractmethod
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages"""
        pass
        
    def preprocess_image(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Enhanced preprocessing with validation"""
        try:
            # Validate image
            if not self.validate_image(image):
                raise ValueError("Invalid image format or size")
            
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                if image.shape[2] == 4:  # RGBA
                    image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                elif image.shape[2] == 3:  # RGB/BGR
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            return image
            
        except Exception as e:
            self.logger.error(f"Preprocessing failed: {e}")
            raise
    
    def validate_image(self, image: np.ndarray) -> bool:
        """Validate image for processing"""
        if image is None:
            return False
        
        if len(image.shape) < 2 or len(image.shape) > 3:
            return False
        
        height, width = image.shape[:2]
        
        # Check size constraints
        if (width < self.min_image_size[0] or 
            height < self.min_image_size[1] or
            width > self.max_image_size[0] or 
            height > self.max_image_size[1]):
            return False
        
        return True
    
    def detect_text_type(self, image: np.ndarray) -> TextType:
        """Detect if text is handwritten or printed"""
        # Basic heuristic - can be overridden by specific engines
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Simple variance-based detection
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Lower variance might indicate handwriting
        if variance < 100:
            return TextType.HANDWRITTEN
        else:
            return TextType.PRINTED
    
    def detect_orientation(self, image: np.ndarray) -> float:
        """Detect text orientation angle"""
        # Basic orientation detection - can be enhanced by specific engines
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Use Hough line detection for orientation
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        if lines is not None:
            angles = []
            for rho, theta in lines[:10]:  # Use top 10 lines
                angle = np.degrees(theta) - 90
                angles.append(angle)
            
            if angles:
                return np.median(angles)
        
        return 0.0
    
    def calculate_confidence(self, results: List[OCRResult]) -> float:
        """Calculate overall confidence score with weighting"""
        if not results:
            return 0.0
        
        # Weight by text length and area
        total_weight = 0
        weighted_confidence = 0
        
        for result in results:
            text_length = len(result.text.strip())
            area = result.bbox.area
            weight = text_length * np.sqrt(area)  # Combined weight
            
            weighted_confidence += result.confidence * weight
            total_weight += weight
        
        return weighted_confidence / total_weight if total_weight > 0 else 0.0
    
    def validate_result(self, result: OCRResult) -> bool:
        """Enhanced result validation"""
        # Basic validation
        if result.confidence < self.config.get('min_confidence', 0.1):
            return False
        
        if len(result.text.strip()) == 0:
            return False
        
        if result.bbox.area <= 0:
            return False
        
        # Text quality checks
        text = result.text.strip()
        
        # Check for reasonable character distribution
        if len(text) > 2:
            alpha_ratio = sum(c.isalpha() for c in text) / len(text)
            if alpha_ratio < 0.1:  # Too few alphabetic characters
                return False
        
        return True
    
    def batch_process(self, images: List[np.ndarray], **kwargs) -> List[DocumentResult]:
        """Process multiple images efficiently"""
        results = []
        
        for i, image in enumerate(images):
            try:
                self.logger.info(f"Processing image {i+1}/{len(images)}")
                result = self.process_image(image, **kwargs)
                results.append(result)
                
                # Update stats
                self.processing_stats['total_processed'] += 1
                self.processing_stats['total_time'] += result.processing_time
                
            except Exception as e:
                self.logger.error(f"Failed to process image {i+1}: {e}")
                self.processing_stats['errors'] += 1
                
                # Create empty result for failed processing
                empty_result = DocumentResult(
                    full_text="",
                    results=[],
                    text_regions=[],
                    document_structure=DocumentStructure(),
                    processing_time=0.0,
                    engine_name=self.name,
                    image_stats={},
                    confidence_score=0.0
                )
                results.append(empty_result)
        
        return results
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get engine processing statistics"""
        stats = self.processing_stats.copy()
        
        if stats['total_processed'] > 0:
            stats['avg_processing_time'] = stats['total_time'] / stats['total_processed']
            stats['success_rate'] = (stats['total_processed'] - stats['errors']) / stats['total_processed']
        else:
            stats['avg_processing_time'] = 0.0
            stats['success_rate'] = 0.0
            
        return stats
    
    def save_model(self, path: str) -> bool:
        """Save model state (if applicable)"""
        try:
            # Base implementation - override in specific engines
            model_info = {
                'engine_name': self.name,
                'config': self.config,
                'stats': self.get_processing_stats(),
                'supported_languages': self.supported_languages
            }
            
            import json
            with open(path, 'w') as f:
                json.dump(model_info, f, indent=2)
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
            return False
    
    def load_model(self, path: str) -> bool:
        """Load model state (if applicable)"""
        try:
            # Base implementation - override in specific engines
            import json
            with open(path, 'r') as f:
                model_info = json.load(f)
            
            self.config.update(model_info.get('config', {}))
            self.processing_stats.update(model_info.get('stats', {}))
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False
    
    def cleanup(self):
        """Enhanced cleanup with resource tracking"""
        try:
            self.logger.info(f"Cleaning up {self.name} engine")
            self.is_initialized = False
            self.model_loaded = False
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")
    
    def __enter__(self):
        if not self.initialize():
            raise RuntimeError(f"Failed to initialize {self.name} engine")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
    
    def __str__(self) -> str:
        return f"{self.name}Engine(initialized={self.is_initialized}, languages={self.supported_languages})"
    
    def __repr__(self) -> str:
        return self.__str__()