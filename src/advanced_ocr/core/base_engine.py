# src/core/base_engine.py - Modern OCR Base Engine

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional, Union
import cv2
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
import time
import logging
from pathlib import Path
import json

class TextType(Enum):
    """Modern text type classification for OCR systems"""
    # Content-based classification
    PRINTED = "printed"
    HANDWRITTEN = "handwritten"
    MIXED = "mixed"
    UNKNOWN = "unknown"
    
    # Structural classification (what your test expects)
    PARAGRAPH = "paragraph"
    LINE = "line"
    WORD = "word"
    BLOCK = "block"
    TITLE = "title"
    HEADER = "header"
    FOOTER = "footer"
    CAPTION = "caption"
    TABLE_CELL = "table_cell"
    LIST_ITEM = "list_item"

class DetectionMethod(Enum):
    """Text detection methods"""
    TRADITIONAL = "traditional"
    DEEP_LEARNING = "deep_learning"
    HYBRID = "hybrid"
    TRANSFORMER = "transformer"
    CNN = "cnn"

class OCREngineType(Enum):
    """OCR Engine types"""
    TESSERACT = "tesseract"
    EASYOCR = "easyocr"
    PADDLEOCR = "paddleocr"
    TROCR = "trocr"
    CRAFT = "craft"
    EAST = "east"
    PPOCR = "ppocr"

class QualityLevel(Enum):
    """Image quality levels"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    UNUSABLE = "unusable"

@dataclass
class BoundingBox:
    """Enhanced bounding box with modern OCR features"""
    x: int
    y: int
    width: int
    height: int
    confidence: float = 0.0
    text_type: TextType = TextType.UNKNOWN
    rotation_angle: float = 0.0
    
    @property
    def center(self) -> Tuple[int, int]:
        """Get center coordinates"""
        return (self.x + self.width // 2, self.y + self.height // 2)
    
    @property
    def area(self) -> int:
        """Get bounding box area"""
        return self.width * self.height
    
    @property
    def aspect_ratio(self) -> float:
        """Get aspect ratio (width/height)"""
        return self.width / self.height if self.height > 0 else 0
    
    @property
    def corners(self) -> List[Tuple[int, int]]:
        """Get all four corner coordinates"""
        return [
            (self.x, self.y),
            (self.x + self.width, self.y),
            (self.x + self.width, self.y + self.height),
            (self.x, self.y + self.height)
        ]
    
    def to_tuple(self) -> Tuple[int, int, int, int]:
        """Convert to (x, y, width, height) tuple"""
        return (self.x, self.y, self.width, self.height)
    
    def to_xyxy(self) -> Tuple[int, int, int, int]:
        """Convert to (x1, y1, x2, y2) format"""
        return (self.x, self.y, self.x + self.width, self.y + self.height)
    
    def intersects(self, other: 'BoundingBox') -> bool:
        """Check if this bbox intersects with another"""
        return not (self.x + self.width <= other.x or 
                    other.x + other.width <= self.x or
                    self.y + self.height <= other.y or 
                    other.y + other.height <= self.y)
    
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
    
    def expand(self, padding: Union[int, Tuple[int, int, int, int]]) -> 'BoundingBox':
        """Expand bounding box by padding"""
        if isinstance(padding, int):
            return BoundingBox(
                max(0, self.x - padding),
                max(0, self.y - padding),
                self.width + 2 * padding,
                self.height + 2 * padding,
                self.confidence,
                self.text_type,
                self.rotation_angle
            )
        else:
            left, top, right, bottom = padding
            return BoundingBox(
                max(0, self.x - left),
                max(0, self.y - top),
                self.width + left + right,
                self.height + top + bottom,
                self.confidence,
                self.text_type,
                self.rotation_angle
            )

@dataclass
class TextRegion:
    """Modern text region with comprehensive metadata"""
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
    is_underlined: bool = False
    font_family: Optional[str] = None
    text_color: Optional[Tuple[int, int, int]] = None
    background_color: Optional[Tuple[int, int, int]] = None
    rotation: float = 0.0
    
    # Modern OCR features
    word_confidences: List[float] = field(default_factory=list)
    char_confidences: List[float] = field(default_factory=list)
    alternatives: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize with default values if needed"""
        if self.bbox is None:
            self.bbox = BoundingBox(0, 0, 100, 30, self.confidence, self.text_type)
    
    @property
    def is_valid(self) -> bool:
        """Check if region contains valid text"""
        return (len(self.text.strip()) > 0 and 
                self.confidence > 0.0 and 
                self.bbox is not None and
                self.bbox.area > 0)
    
    @property
    def full_text(self) -> str:
        """Get full text (alias for backward compatibility)"""
        return self.text
    
    @property
    def word_count(self) -> int:
        """Get word count"""
        return len(self.text.split()) if self.text else 0
    
    @property
    def char_count(self) -> int:
        """Get character count"""
        return len(self.text) if self.text else 0
    
    def get_words(self) -> List[str]:
        """Get individual words"""
        return self.text.split() if self.text else []

@dataclass 
class OCRResult:
    """Modern OCR result with comprehensive metadata"""
    text: str                                           
    confidence: float                                   
    regions: Optional[List[TextRegion]] = None        
    processing_time: float = 0.0                       
    bbox: Optional[BoundingBox] = None                
    level: str = "page"                               
    language: str = "en"                              
    text_type: TextType = TextType.UNKNOWN            
    rotation_angle: float = 0.0                       
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Modern OCR features
    engine_name: str = "unknown"
    engine_version: str = "unknown"
    quality_score: float = 0.0
    detected_languages: List[str] = field(default_factory=list)
    page_number: int = 1
    
    def __post_init__(self):
        """Initialize with safe defaults"""
        if self.regions is None:
            self.regions = []
        
        if self.bbox is None:
            if self.regions and len(self.regions) > 0:
                first_region = self.regions[0]
                if first_region.bbox:
                    self.bbox = first_region.bbox
                else:
                    self.bbox = BoundingBox(0, 0, 100, 30, self.confidence)
            else:
                self.bbox = BoundingBox(0, 0, 100, 30, self.confidence)
        elif isinstance(self.bbox, tuple):
            if len(self.bbox) == 4:
                x, y, w, h = self.bbox
                self.bbox = BoundingBox(x, y, w, h, self.confidence)
            else:
                self.bbox = BoundingBox(0, 0, 100, 30, self.confidence)
    
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
    
    @property
    def word_count(self) -> int:
        """Get total word count"""
        return len(self.text.split()) if self.text else 0
    
    @property
    def char_count(self) -> int:
        """Get total character count"""
        return len(self.text) if self.text else 0
    
    @property
    def line_count(self) -> int:
        """Get line count"""
        return len(self.text.split('\n')) if self.text else 0

@dataclass
class DocumentStructure:
    """Modern document structure analysis"""
    headers: List[TextRegion] = field(default_factory=list)
    paragraphs: List[List[TextRegion]] = field(default_factory=list)
    lists: List[List[TextRegion]] = field(default_factory=list)
    tables: List[Dict[str, Any]] = field(default_factory=list)
    images: List[BoundingBox] = field(default_factory=list)
    reading_order: List[int] = field(default_factory=list)
    columns: int = 1
    page_orientation: str = "portrait"
    
    @property
    def total_elements(self) -> int:
        """Get total structural elements"""
        return (len(self.headers) + len(self.paragraphs) + 
                len(self.lists) + len(self.tables) + len(self.images))

@dataclass
class DocumentResult:
    """Modern comprehensive document OCR result"""
    pages: List[OCRResult] = field(default_factory=list)           
    metadata: Dict[str, Any] = field(default_factory=dict)         
    processing_time: float = 0.0                                   
    engine_name: str = "unknown"                                   
    confidence_score: float = 0.0                                  
    text_type: TextType = TextType.UNKNOWN                         
    detected_languages: List[str] = field(default_factory=list)    
    preprocessing_steps: List[str] = field(default_factory=list)   
    postprocessing_steps: List[str] = field(default_factory=list)
    
    # Modern features
    quality_level: QualityLevel = QualityLevel.FAIR
    document_structure: Optional[DocumentStructure] = None
    
    def __post_init__(self):
        """Initialize with safe defaults"""
        if self.pages is None:
            self.pages = []
        if self.document_structure is None:
            self.document_structure = DocumentStructure()
    
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
    
    @property
    def page_count(self) -> int:
        """Get total page count"""
        return len(self.pages)
    
    # Backward compatibility
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
    def image_stats(self) -> Dict[str, Any]:
        """Get image statistics from metadata"""
        return self.metadata.get('image_stats', {})

class BaseOCREngine(ABC):
    """Modern abstract base class for OCR engines"""
    
    def __init__(self, name: str = "", config: Optional[Dict[str, Any]] = None):
        """Initialize OCR engine with modern defaults"""
        self.name = name if name else self.__class__.__name__.replace('Engine', '').lower()
        self.config = config if config is not None else {}
        self.is_initialized = False
        self.supported_languages = ['en']  # Default to English
        self.model_loaded = False
        self.logger = self._setup_logger()
        
        # Processing statistics
        self.processing_stats = {
            'total_processed': 0,
            'total_time': 0.0,
            'avg_confidence': 0.0,
            'errors': 0,
            'successful_extractions': 0
        }
        
        # Engine capabilities
        self.supports_handwriting = False
        self.supports_multiple_languages = False
        self.supports_orientation_detection = False
        self.supports_structure_analysis = False
        self.supports_table_detection = False
        self.max_image_size = (4096, 4096)
        self.min_image_size = (32, 32)
        
        # Performance settings
        self.batch_size = 1
        self.use_gpu = False
        self.num_threads = 1
        
    def _setup_logger(self) -> logging.Logger:
        """Setup engine-specific logger"""
        logger = logging.getLogger(f"OCR.{self.name}")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '[%(asctime)s] [%(levelname)8s] %(name)s.py:%(lineno)d - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger
    
    def is_available(self) -> bool:
        """Check if engine is available and ready to use"""
        try:
            return self.initialize()
        except Exception as e:
            self.logger.error(f"Engine availability check failed: {e}")
            return False
        
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the OCR engine"""
        pass
    
    def extract_text(self, image: np.ndarray, **kwargs) -> OCRResult:
        """Extract text from image - primary interface method"""
        start_time = time.time()
        
        try:
            if not self.is_initialized:
                if not self.initialize():
                    raise RuntimeError(f"Failed to initialize {self.name}")
            
            if not self.validate_image(image):
                raise ValueError("Invalid image format or dimensions")
            
            # Process the image
            result = self.process_image(image, **kwargs)
            
            # Convert DocumentResult to OCRResult if needed
            if isinstance(result, DocumentResult):
                if result.pages:
                    ocr_result = result.pages[0]  # Take first page
                else:
                    ocr_result = OCRResult("", 0.0, engine_name=self.name)
            else:
                ocr_result = result
            
            # Update processing stats
            processing_time = time.time() - start_time
            ocr_result.processing_time = processing_time
            ocr_result.engine_name = self.name
            
            self._update_stats(ocr_result, processing_time, success=True)
            
            return ocr_result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Text extraction failed: {e}")
            self._update_stats(None, processing_time, success=False)
            
            # Return empty result instead of raising
            return OCRResult(
                text="",
                confidence=0.0,
                processing_time=processing_time,
                engine_name=self.name,
                metadata={"error": str(e)}
            )
    
    @abstractmethod
    def process_image(self, image: np.ndarray, **kwargs) -> Union[OCRResult, DocumentResult]:
        """Process an image and return OCR results"""
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
            
            # Handle different image formats
            if len(image.shape) == 3:
                if image.shape[2] == 4:  # RGBA
                    image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                elif image.shape[2] == 3:  # BGR
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Resize if needed
            height, width = image.shape[:2]
            if width > self.max_image_size[0] or height > self.max_image_size[1]:
                scale = min(self.max_image_size[0]/width, self.max_image_size[1]/height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            
            return image
            
        except Exception as e:
            self.logger.error(f"Preprocessing failed: {e}")
            raise
    
    def validate_image(self, image: np.ndarray) -> bool:
        """Validate image for processing"""
        if image is None or not isinstance(image, np.ndarray):
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
    
    def _update_stats(self, result: Optional[OCRResult], processing_time: float, success: bool):
        """Update processing statistics"""
        self.processing_stats['total_processed'] += 1
        self.processing_stats['total_time'] += processing_time
        
        if success and result:
            self.processing_stats['successful_extractions'] += 1
            # Update average confidence
            total_conf = self.processing_stats['avg_confidence'] * (self.processing_stats['total_processed'] - 1)
            self.processing_stats['avg_confidence'] = (total_conf + result.confidence) / self.processing_stats['total_processed']
        else:
            self.processing_stats['errors'] += 1
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get comprehensive processing statistics"""
        stats = self.processing_stats.copy()
        
        if stats['total_processed'] > 0:
            stats['avg_processing_time'] = stats['total_time'] / stats['total_processed']
            stats['success_rate'] = stats['successful_extractions'] / stats['total_processed']
            stats['error_rate'] = stats['errors'] / stats['total_processed']
        else:
            stats['avg_processing_time'] = 0.0
            stats['success_rate'] = 0.0
            stats['error_rate'] = 0.0
            
        return stats
    
    def cleanup(self):
        """Cleanup resources"""
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

# Backward compatibility alias
OCREngine = BaseOCREngine

# Export all classes
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
    'OCREngineType',
    'QualityLevel'
]