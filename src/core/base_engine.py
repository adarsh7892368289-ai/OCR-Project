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

class OCREngineType(Enum):
    """OCR Engine types"""
    TESSERACT = "tesseract"
    EASYOCR = "easyocr"
    PADDLEOCR = "paddleocr"
    TROCR = "trocr"
    CRAFT = "craft"
    EAST = "east"

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
    text: str = ""
    confidence: float = 0.0
    bbox: Optional[BoundingBox] = None
    text_type: TextType = TextType.UNKNOWN
    language: str = "en"
    reading_order: int = -1
    line_number: int = -1
    paragraph_id: int = -1
    structure_type: str = "text"
    font_size: Optional[float] = None
    is_bold: bool = False
    is_italic: bool = False
    
    def __post_init__(self):
        # Create default bbox if none provided
        if self.bbox is None:
            self.bbox = BoundingBox(0, 0, 100, 30, self.confidence)
    
    @property
    def is_valid(self) -> bool:
        """Check if region contains valid text"""
        return (len(self.text.strip()) > 0 and 
                self.confidence > 0.0 and 
                self.bbox is not None and
                self.bbox.area > 0)
    
    @property
    def full_text(self) -> str:
        return self.text

@dataclass 
class OCRResult:
    """FIXED: OCR result with compatible constructor for post-processing pipeline"""
    # CRITICAL: Parameter order and naming fixed for pipeline compatibility
    text: str                                           # Main text (was full_text)
    confidence: float                                   # Confidence score
    regions: Optional[List[TextRegion]] = None         # Text regions (was text_regions)
    processing_time: float = 0.0                       # FIXED: Added missing parameter
    bbox: Optional[BoundingBox] = None                 # Bounding box (optional)
    level: str = "line"                               # Processing level
    language: str = "en"                              # Language
    text_type: TextType = TextType.UNKNOWN            # Text type
    rotation_angle: float = 0.0                       # Rotation angle
    metadata: Dict[str, Any] = field(default_factory=dict)  # Processing metadata (was processing_metadata)
    
    def __post_init__(self):
        # Handle regions parameter compatibility
        if self.regions is None:
            self.regions = []
        
        # Handle bbox parameter compatibility
        if self.bbox is None:
            if self.regions and len(self.regions) > 0:
                # Create bbox from first region
                first_region = self.regions[0]
                if first_region.bbox:
                    self.bbox = first_region.bbox
                else:
                    self.bbox = BoundingBox(0, 0, 100, 30, self.confidence)
            else:
                self.bbox = BoundingBox(0, 0, 100, 30, self.confidence)
        elif isinstance(self.bbox, tuple):
            # Convert tuple to BoundingBox
            if len(self.bbox) == 4:
                x, y, w, h = self.bbox
                self.bbox = BoundingBox(x, y, w, h, self.confidence)
            else:
                self.bbox = BoundingBox(0, 0, 100, 30, self.confidence)
    
    # Backward compatibility properties
    @property
    def full_text(self) -> str:
        """Backward compatibility for full_text"""
        return self.text
    
    @property
    def text_regions(self) -> List[TextRegion]:
        """Backward compatibility for text_regions"""
        return self.regions or []
    
    @property
    def processing_metadata(self) -> Dict[str, Any]:
        """Backward compatibility for processing_metadata"""
        return self.metadata

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
    """FIXED: Comprehensive document OCR result with compatible constructor"""
    # CRITICAL: Fixed parameter order and naming for pipeline compatibility
    pages: List[OCRResult] = field(default_factory=list)           # OCR results per page (was results)
    metadata: Dict[str, Any] = field(default_factory=dict)         # Document metadata
    processing_time: float = 0.0                                   # Total processing time
    engine_name: str = "unknown"                                   # Engine used
    confidence_score: float = 0.0                                  # Overall confidence
    text_type: TextType = TextType.UNKNOWN                         # Document text type
    detected_languages: List[str] = field(default_factory=list)    # Detected languages
    preprocessing_steps: List[str] = field(default_factory=list)   # Preprocessing applied
    postprocessing_steps: List[str] = field(default_factory=list)  # Postprocessing applied
    
    # Computed properties
    def __post_init__(self):
        # Ensure pages is not None
        if self.pages is None:
            self.pages = []
    
    @property
    def full_text(self) -> str:
        """Get full document text"""
        if not self.pages:
            return ""
        return "\n".join(page.text for page in self.pages)
    
    @property
    def text(self) -> str:
        """Alias for full_text"""
        return self.full_text
        
    @property
    def confidence(self) -> float:
        """Get overall confidence"""
        return self.confidence_score
        
    @property
    def best_engine(self) -> str:
        """Get best engine name"""
        return self.engine_name
    
    @property
    def word_count(self) -> int:
        """Get total word count"""
        return len(self.full_text.split())
    
    @property
    def char_count(self) -> int:
        """Get total character count"""
        return len(self.full_text)
    
    @property
    def line_count(self) -> int:
        """Get total line count"""
        return self.full_text.count('\n') + 1
    
    # Backward compatibility properties
    @property
    def results(self) -> List[OCRResult]:
        """Backward compatibility for results"""
        return self.pages
    
    @property
    def text_regions(self) -> List[TextRegion]:
        """Get all text regions from all pages"""
        regions = []
        for page in self.pages:
            regions.extend(page.regions or [])
        return regions
    
    @property
    def document_structure(self) -> DocumentStructure:
        """Get document structure (computed)"""
        # For now, return empty structure - can be enhanced later
        return DocumentStructure()
    
    @property
    def image_stats(self) -> Dict[str, Any]:
        """Get image statistics from metadata"""
        return self.metadata.get('image_stats', {})

class BaseOCREngine(ABC):
    """Enhanced abstract base class for all OCR engines"""
    
    def __init__(self, name: str = "", config: Optional[Dict[str, Any]] = None):
        # FIXED: Safer initialization to prevent None issues
        self.name = name if name else self.__class__.__name__
        self.config = config if config is not None else {}
        self.is_initialized = False
        self.supported_languages = []
        self.model_loaded = False
        self.logger = self._setup_logger()
        
        self.processing_stats = {
            'total_processed': 0,
            'total_time': 0.0,
            'avg_confidence': 0.0,
            'errors': 0
        }
        
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
            if not self.validate_image(image):
                raise ValueError("Invalid image format or size")
            
            if len(image.shape) == 3:
                if image.shape[2] == 4:
                    image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                elif image.shape[2] == 3:
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
        
        if (width < self.min_image_size[0] or 
            height < self.min_image_size[1] or
            width > self.max_image_size[0] or 
            height > self.max_image_size[1]):
            return False
        
        return True
    
    def detect_text_type(self, image: np.ndarray) -> TextType:
        """Detect if text is handwritten or printed"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        try:
            variance = cv2.Laplacian(gray, cv2.CV_64F).var()
            if variance < 100:
                return TextType.HANDWRITTEN
            else:
                return TextType.PRINTED
        except:
            return TextType.UNKNOWN
    
    def detect_orientation(self, image: np.ndarray) -> float:
        """Detect text orientation angle"""
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
            
            if lines is not None and len(lines) > 0:
                angles = [float(np.degrees(line[0][1]) - 90) for line in lines[:10]]
                if angles:
                    return float(np.median(angles))
        except:
            pass
        
        return 0.0
    
    def calculate_confidence(self, results: List[OCRResult]) -> float:
        """Calculate overall confidence score with weighting"""
        if not results:
            return 0.0
        
        total_weight = 0
        weighted_confidence = 0
        
        for result in results:
            text_length = len(result.text.strip())
            if result.bbox:
                area = result.bbox.area
            else:
                area = 100  # default area
            weight = text_length * np.sqrt(area)
            
            weighted_confidence += result.confidence * weight
            total_weight += weight
        
        return weighted_confidence / total_weight if total_weight > 0 else 0.0
    
    def validate_result(self, result: OCRResult) -> bool:
        """Enhanced result validation"""
        if result.confidence < self.config.get('min_confidence', 0.1):
            return False
        
        if len(result.text.strip()) == 0:
            return False
        
        if result.bbox and result.bbox.area <= 0:
            return False
        
        text = result.text.strip()
        
        if len(text) > 2:
            alpha_ratio = sum(c.isalpha() for c in text) / len(text)
            if alpha_ratio < 0.1:
                return False
        
        return True
    
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

# Alias for backward compatibility
OCREngine = BaseOCREngine

# Also export the main classes
__all__ = [
    'BaseOCREngine',
    'OCREngine', 
    'DocumentResult',
    'OCRResult',
    'TextRegion',
    'BoundingBox',
    'TextType',
    'DetectionMethod',
    'DocumentStructure',
    'OCREngineType'
]