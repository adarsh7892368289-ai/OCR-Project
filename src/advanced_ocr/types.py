# Data structures and type definitions for the Advanced OCR Library

"""
Data structures and type definitions for the Advanced OCR Library.
This module contains all data classes, enums, and type definitions used throughout the library.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union, Tuple, Any
from enum import Enum
import numpy as np


# === ENUMS ===

class ProcessingStrategy(Enum):
    """OCR processing strategy based on image quality"""
    MINIMAL = "minimal"      # High quality - minimal processing
    BALANCED = "balanced"    # Standard processing
    ENHANCED = "enhanced"    # Poor quality - heavy enhancement
    MULTI_ENGINE = "multi_engine"

class ImageType(Enum):
    """Image content type classification"""
    DOCUMENT = "document"
    HANDWRITTEN = "handwritten"
    PRINTED = "printed"
    MIXED = "mixed"
    TABLE = "table"
    FORM = "form"
    UNKNOWN = "unknown"


class ImageQuality(Enum):
    """Image quality levels"""
    EXCELLENT = "excellent"  # > 0.8
    GOOD = "good"           # 0.6 - 0.8
    FAIR = "fair"           # 0.4 - 0.6
    POOR = "poor"           # 0.2 - 0.4
    UNUSABLE = "unusable"   # < 0.2


class TextType(Enum):
    """Text structure classification"""
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
    PRINTED = "printed"
    HANDWRITTEN = "handwritten"
    MIXED = "mixed"
    UNKNOWN = "unknown"


# === DATA STRUCTURES ===

@dataclass
class BoundingBox:
    """Bounding box with coordinates and metadata"""
    x: int
    y: int
    width: int
    height: int
    confidence: float = 0.0
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
    def corners(self) -> List[Tuple[int, int]]:
        """Get all four corner coordinates"""
        return [
            (self.x, self.y),
            (self.x + self.width, self.y),
            (self.x + self.width, self.y + self.height),
            (self.x, self.y + self.height)
        ]


@dataclass
class QualityMetrics:
    """Image quality analysis results"""
    overall_score: float                    # 0.0 - 1.0
    sharpness_score: float                  # 0.0 - 1.0
    noise_level: float                      # 0.0 - 1.0 (higher = more noise)
    contrast_score: float                   # 0.0 - 1.0
    brightness_score: float                 # 0.0 - 1.0
    needs_enhancement: bool                 # Recommendation
    image_type: ImageType = ImageType.UNKNOWN
    quality_level: ImageQuality = ImageQuality.FAIR
    
    # Detailed metrics
    blur_variance: float = 0.0
    edge_density: float = 0.0
    text_region_count: int = 0
    estimated_dpi: int = 150
    color_channels: int = 1
    
    # Enhancement recommendations
    recommended_strategy: ProcessingStrategy = ProcessingStrategy.BALANCED
    enhancement_suggestions: List[str] = field(default_factory=list)
    
    @property
    def is_good_quality(self) -> bool:
        """Check if image has good quality for OCR"""
        return self.overall_score >= 0.6 and not self.needs_enhancement
    
    @property
    def quality_category(self) -> str:
        """Get quality category as string"""
        return self.quality_level.value


@dataclass
class TextRegion:
    """Text region with content and metadata"""
    text: str = ""
    confidence: float = 0.0
    bbox: Optional[BoundingBox] = None
    text_type: TextType = TextType.UNKNOWN
    language: str = "en"
    reading_order: int = -1
    
    # Text formatting
    font_size: Optional[float] = None
    is_bold: bool = False
    is_italic: bool = False
    
    # Alternative results
    alternatives: List[str] = field(default_factory=list)
    word_confidences: List[float] = field(default_factory=list)
    
    @property
    def is_valid(self) -> bool:
        """Check if region contains valid text"""
        return (len(self.text.strip()) > 0 and 
                self.confidence > 0.0 and 
                self.bbox is not None)
    
    @property
    def word_count(self) -> int:
        """Get word count"""
        return len(self.text.split()) if self.text else 0


@dataclass
class ProcessingOptions:
    """Configuration options for OCR processing"""
    # Engine selection
    engines: Optional[List[str]] = None  # ['paddleocr', 'easyocr', 'tesseract', 'trocr']
    strategy: Optional[ProcessingStrategy] = None  # Auto-detect if None
    
    # Processing settings
    enhance_image: bool = True
    detect_orientation: bool = True
    correct_rotation: bool = True
    
    # Quality thresholds
    min_confidence: float = 0.5
    early_termination: bool = True
    early_termination_threshold: float = 0.95
    
    # Performance settings
    max_processing_time: int = 120  # seconds
    use_parallel_processing: bool = True
    batch_size: int = 1
    
    # Language settings
    languages: List[str] = field(default_factory=lambda: ['en'])
    
    # Output settings
    include_regions: bool = False
    include_word_boxes: bool = False
    preserve_formatting: bool = False


@dataclass
class EnhancementResult:
    """Result from image enhancement process"""
    enhanced_image: np.ndarray
    original_image: np.ndarray
    enhancement_applied: str
    processing_time: float
    quality_improvement: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def was_enhanced(self) -> bool:
        """Check if enhancement was actually applied"""
        return self.enhancement_applied != "none"


@dataclass
class OCRResult:
    """Final result from OCR processing - matches your pipeline expectations"""
    text: str
    confidence: float
    processing_time: float = 0.0
    engine_used: str = "unknown"  # This matches pipeline.py usage
    
    # Required by pipeline.py
    quality_metrics: Optional[QualityMetrics] = None
    strategy_used: ProcessingStrategy = ProcessingStrategy.BALANCED
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Optional detailed results
    regions: List[TextRegion] = field(default_factory=list)
    bbox: Optional[BoundingBox] = None
    language: str = "en"
    detected_languages: List[str] = field(default_factory=list)
    
    # Engine compatibility (engine_name is alias for engine_used)
    @property
    def engine_name(self) -> str:
        """Compatibility property for engine_manager.py"""
        return self.engine_used
    
    @engine_name.setter
    def engine_name(self, value: str):
        """Compatibility setter for engine_manager.py"""
        self.engine_used = value
    
    @property
    def success(self) -> bool:
        """Check if OCR was successful"""
        return len(self.text.strip()) > 0 and self.confidence > 0.1
    
    @property
    def word_count(self) -> int:
        """Get total word count"""
        return len(self.text.split()) if self.text else 0
    
    @property
    def line_count(self) -> int:
        """Get line count"""
        return len(self.text.split('\n')) if self.text else 0
    
    @property
    def has_regions(self) -> bool:
        """Check if detailed region information is available"""
        return len(self.regions) > 0


@dataclass
class BatchResult:
    """Result from batch processing"""
    results: List[OCRResult]
    total_processing_time: float
    successful_count: int
    failed_count: int
    average_confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def success_rate(self) -> float:
        """Get success rate as percentage"""
        total = len(self.results)
        return (self.successful_count / total * 100) if total > 0 else 0.0
    
    @property
    def total_images(self) -> int:
        """Get total number of images processed"""
        return len(self.results)


# === TYPE ALIASES ===

# Common type aliases for better readability
ImageArray = np.ndarray
ConfigDict = Dict[str, Any]
EngineList = List[str]
LanguageList = List[str]

# Coordinate types
Point = Tuple[int, int]
Rectangle = Tuple[int, int, int, int]  # x, y, width, height
Polygon = List[Point]

# Result types
ResultList = List[OCRResult]
RegionList = List[TextRegion]


# === EXPORTED TYPES ===

__all__ = [
    # Enums
    'ProcessingStrategy',
    'ImageType', 
    'ImageQuality',
    'TextType',
    
    # Data structures
    'BoundingBox',
    'QualityMetrics',
    'TextRegion',
    'ProcessingOptions',
    'EnhancementResult',
    'OCRResult',
    'BatchResult',
    
    # Type aliases
    'ImageArray',
    'ConfigDict',
    'EngineList',
    'LanguageList',
    'Point',
    'Rectangle',
    'Polygon',
    'ResultList',
    'RegionList',
]