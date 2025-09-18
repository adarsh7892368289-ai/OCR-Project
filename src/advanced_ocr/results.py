# src/advanced_ocr/results.py
"""
Advanced OCR Results Library

This module provides comprehensive data structures for representing OCR results
with hierarchical text organization, spatial information, and confidence metrics.

Classes:
    OCRResult: Primary container for OCR extraction results
    BoundingBox: Flexible coordinate representation with format conversion
    ConfidenceMetrics: Multi-dimensional confidence scoring
    TextRegion: Base class for hierarchical text structures
    Word, Line, Paragraph, Block, Page: Hierarchical text elements
    ProcessingMetrics: Performance tracking and optimization metrics
    BatchResult: Container for multi-document processing results

Example:
    >>> result = OCRResult(
    ...     text="Hello World",
    ...     confidence=0.95,
    ...     engine_name="tesseract"
    ... )
    >>> print(f"Extracted: {result.text} (confidence: {result.confidence})")
    
    >>> bbox = BoundingBox((10, 20, 100, 50), BoundingBoxFormat.XYXY)
    >>> x, y, w, h = bbox.to_xywh()
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union, Tuple
from enum import Enum
import time
import json


class BoundingBoxFormat(Enum):
    XYXY = "xyxy"
    XYWH = "xywh" 
    CENTER_WH = "center_wh"
    POLYGON = "polygon"


class TextLevel(Enum):
    CHARACTER = "character"
    WORD = "word"
    LINE = "line"
    PARAGRAPH = "paragraph" 
    BLOCK = "block"
    PAGE = "page"
    DOCUMENT = "document"


class ContentType(Enum):
    PRINTED_TEXT = "printed_text"
    HANDWRITTEN_TEXT = "handwritten_text"
    MIXED_TEXT = "mixed_text"
    TABLE = "table"
    FORM_FIELD = "form_field"
    HEADER = "header"
    FOOTER = "footer"
    CAPTION = "caption"
    SIGNATURE = "signature"
    UNKNOWN = "unknown"


@dataclass
class BoundingBox:
    """Flexible bounding box with multiple coordinate format support - FIXED VERSION."""
    coordinates: Union[Tuple[float, float, float, float], List[Tuple[float, float]]]
    format: BoundingBoxFormat = BoundingBoxFormat.XYXY
    confidence: float = 1.0
    
    def __init__(self, *args, format: BoundingBoxFormat = BoundingBoxFormat.XYXY, confidence: float = 1.0):
        """
        Initialize BoundingBox with flexible input formats:
        
        # Single tuple/list of coordinates
        BoundingBox((x1, y1, x2, y2))
        BoundingBox([x, y, w, h])
        
        # Four separate coordinates
        BoundingBox(x1, y1, x2, y2)
        
        # Two points (for test compatibility)
        BoundingBox((x1, y1), (x2, y2))
        
        # Polygon points
        BoundingBox([(x1,y1), (x2,y2), (x3,y3), ...])
        """
        self.format = format
        self.confidence = confidence
        
        if len(args) == 1:
            # Single argument - could be tuple/list of coordinates or list of points
            arg = args[0]
            if isinstance(arg, (tuple, list)):
                if len(arg) == 4 and all(isinstance(x, (int, float)) for x in arg):
                    # Single tuple/list of 4 coordinates: (x1, y1, x2, y2) or (x, y, w, h)
                    self.coordinates = tuple(float(x) for x in arg)
                elif len(arg) > 0 and isinstance(arg[0], (tuple, list)):
                    # List of points: [(x1,y1), (x2,y2), ...]
                    self.coordinates = [tuple(float(coord) for coord in point) for point in arg]
                    self.format = BoundingBoxFormat.POLYGON
                else:
                    raise ValueError(f"Invalid single argument format: {arg}")
            else:
                raise ValueError(f"Single argument must be tuple/list, got {type(arg)}")
                
        elif len(args) == 2:
            # Two arguments - assume two points: (x1, y1), (x2, y2)
            point1, point2 = args
            if (isinstance(point1, (tuple, list)) and len(point1) == 2 and 
                isinstance(point2, (tuple, list)) and len(point2) == 2):
                x1, y1 = float(point1[0]), float(point1[1])
                x2, y2 = float(point2[0]), float(point2[1])
                self.coordinates = (x1, y1, x2, y2)
                self.format = BoundingBoxFormat.XYXY
            else:
                raise ValueError(f"Two arguments must be points (x,y), got {args}")
                
        elif len(args) == 4:
            # Four arguments - assume x1, y1, x2, y2
            self.coordinates = tuple(float(x) for x in args)
            self.format = BoundingBoxFormat.XYXY
            
        else:
            raise ValueError(f"Invalid number of arguments: {len(args)}. Expected 1, 2, or 4.")
    
    # ADDED: Direct property access for backward compatibility
    @property
    def x1(self) -> float:
        """X1 coordinate (left edge)."""
        x1, y1, x2, y2 = self.to_xyxy()
        return x1
    
    @property
    def y1(self) -> float:
        """Y1 coordinate (top edge)."""
        x1, y1, x2, y2 = self.to_xyxy()
        return y1
    
    @property
    def x2(self) -> float:
        """X2 coordinate (right edge)."""
        x1, y1, x2, y2 = self.to_xyxy()
        return x2
    
    @property
    def y2(self) -> float:
        """Y2 coordinate (bottom edge)."""
        x1, y1, x2, y2 = self.to_xyxy()
        return y2
    
    @property
    def x(self) -> float:
        """X coordinate (left edge) - alias for x1."""
        return self.x1
    
    @property
    def y(self) -> float:
        """Y coordinate (top edge) - alias for y1."""
        return self.y1
    
    @property
    def width(self) -> float:
        """Width of bounding box."""
        x, y, w, h = self.to_xywh()
        return w
    
    @property
    def height(self) -> float:
        """Height of bounding box."""
        x, y, w, h = self.to_xywh()
        return h
    
    def to_xyxy(self) -> Tuple[float, float, float, float]:
        """Convert to (x1, y1, x2, y2) format."""
        if self.format == BoundingBoxFormat.XYXY:
            return self.coordinates
        elif self.format == BoundingBoxFormat.XYWH:
            x, y, w, h = self.coordinates
            return (x, y, x + w, y + h)
        elif self.format == BoundingBoxFormat.CENTER_WH:
            cx, cy, w, h = self.coordinates
            return (cx - w/2, cy - h/2, cx + w/2, cy + h/2)
        elif self.format == BoundingBoxFormat.POLYGON:
            xs = [point[0] for point in self.coordinates]
            ys = [point[1] for point in self.coordinates]
            return (min(xs), min(ys), max(xs), max(ys))
    
    def to_xywh(self) -> Tuple[float, float, float, float]:
        """Convert to (x, y, width, height) format."""
        x1, y1, x2, y2 = self.to_xyxy()
        return (x1, y1, x2 - x1, y2 - y1)
    
    def area(self) -> float:
        """Calculate bounding box area."""
        if self.format == BoundingBoxFormat.POLYGON:
            coords = self.coordinates
            n = len(coords)
            area = 0.0
            for i in range(n):
                j = (i + 1) % n
                area += coords[i][0] * coords[j][1]
                area -= coords[j][0] * coords[i][1]
            return abs(area) / 2.0
        else:
            x, y, w, h = self.to_xywh()
            return w * h
    
    def iou(self, other: 'BoundingBox') -> float:
        """Calculate Intersection over Union with another bounding box."""
        x1, y1, x2, y2 = self.to_xyxy()
        ox1, oy1, ox2, oy2 = other.to_xyxy()
        
        ix1, iy1 = max(x1, ox1), max(y1, oy1)
        ix2, iy2 = min(x2, ox2), min(y2, oy2)
        
        if ix1 >= ix2 or iy1 >= iy2:
            return 0.0
        
        intersection = (ix2 - ix1) * (iy2 - iy1)
        union = self.area() + other.area() - intersection
        
        return intersection / union if union > 0 else 0.0

@dataclass
class ConfidenceMetrics:
    """Multi-dimensional confidence scoring for OCR results."""
    overall: float
    text_detection: float = 0.0
    text_recognition: float = 0.0
    layout_analysis: float = 0.0
    language_detection: float = 0.0
    char_confidences: Optional[List[float]] = None
    min_confidence: Optional[float] = None
    max_confidence: Optional[float] = None
    std_confidence: Optional[float] = None
    
    def __post_init__(self):
        """Calculate derived confidence metrics."""
        if self.char_confidences:
            self.min_confidence = min(self.char_confidences)
            self.max_confidence = max(self.char_confidences)
            if len(self.char_confidences) > 1:
                import statistics
                self.std_confidence = statistics.stdev(self.char_confidences)
            else:
                self.std_confidence = 0.0
    
    def __format__(self, format_spec: str) -> str:
        """Support formatting - returns overall confidence formatted."""
        if format_spec:
            return format(self.overall, format_spec)
        return str(self.overall)
    
    def __float__(self) -> float:
        """Convert to float - returns overall confidence."""
        return float(self.overall)
    
    def __str__(self) -> str:
        """String representation."""
        return f"{self.overall:.3f}"

@dataclass
class ProcessingMetrics:
    """Performance tracking for optimization requirements."""
    total_processing_time: float = 0.0
    preprocessing_time: float = 0.0
    ocr_processing_time: float = 0.0
    postprocessing_time: float = 0.0
    regions_detected: int = 0
    regions_after_filtering: int = 0
    region_filtering_time: float = 0.0
    engine_selection_time: float = 0.0
    engine_initialization_time: float = 0.0
    character_extraction_rate: float = 0.0
    peak_memory_usage: float = 0.0
    memory_efficiency: float = 0.0
    
    @property
    def text_detection_efficiency(self) -> float:
        """Calculate text detection filtering efficiency."""
        if self.regions_detected == 0:
            return 0.0
        return self.regions_after_filtering / self.regions_detected
    
    @property
    def speed_accuracy_ratio(self) -> float:
        """Speed vs accuracy performance metric."""
        if self.total_processing_time == 0:
            return 0.0
        return self.character_extraction_rate / self.total_processing_time


@dataclass
class TextRegion:
    """Base hierarchical text region with spatial and metadata information - FIXED VERSION."""
    text: str
    bbox: BoundingBox
    confidence: ConfidenceMetrics
    level: TextLevel
    element_id: Optional[str] = None
    content_type: ContentType = ContentType.PRINTED_TEXT
    language: str = "en"
    reading_order: Optional[int] = None
    font_size: Optional[float] = None
    font_family: Optional[str] = None
    is_bold: bool = False
    is_italic: bool = False
    text_color: Optional[Tuple[int, int, int]] = None
    background_color: Optional[Tuple[int, int, int]] = None
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    processing_time: float = 0.0
    engine_name: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # ADDED: coordinates property for backward compatibility
    @property
    def coordinates(self) -> Tuple[float, float, float, float]:
        """Get coordinates as (x, y, width, height) for backward compatibility."""
        return self.bbox.to_xywh()


@dataclass
class Word(TextRegion):
    """Word-level text element with character details."""
    level: TextLevel = field(default=TextLevel.WORD, init=False)
    char_bboxes: Optional[List[BoundingBox]] = None
    alternatives: List[str] = field(default_factory=list)
    is_numeric: bool = False
    contains_special_chars: bool = False
    
    def __post_init__(self):
        """Initialize word-level properties."""
        self.is_numeric = self.text.replace('.', '').replace(',', '').isdigit()
        self.contains_special_chars = not self.text.replace(' ', '').isalnum()


@dataclass
class Line(TextRegion):
    """Line-level text element containing words."""
    level: TextLevel = field(default=TextLevel.LINE, init=False)
    words: List[Word] = field(default_factory=list)
    baseline: Optional[Tuple[float, float, float, float]] = None
    text_direction: str = "ltr"
    line_height: Optional[float] = None
    
    def get_text(self, separator: str = " ") -> str:
        """Get full line text from constituent words."""
        return separator.join(word.text for word in self.words)
    
    def __post_init__(self):
        """Initialize line text from words."""
        if not self.text and self.words:
            self.text = self.get_text()


@dataclass
class Paragraph(TextRegion):
    """Paragraph-level text element containing lines."""
    level: TextLevel = field(default=TextLevel.PARAGRAPH, init=False)
    lines: List[Line] = field(default_factory=list)
    alignment: str = "left"
    indentation: float = 0.0
    line_spacing: Optional[float] = None
    
    def get_text(self, line_separator: str = "\n") -> str:
        """Get full paragraph text from constituent lines."""
        return line_separator.join(line.get_text() for line in self.lines)
    
    def __post_init__(self):
        """Initialize paragraph text from lines."""
        if not self.text and self.lines:
            self.text = self.get_text()


@dataclass
class Block(TextRegion):
    """Block-level text element containing paragraphs."""
    level: TextLevel = field(default=TextLevel.BLOCK, init=False)
    paragraphs: List[Paragraph] = field(default_factory=list)
    block_type: str = "text"
    column_index: int = 0
    
    def get_text(self, paragraph_separator: str = "\n\n") -> str:
        """Get full block text from constituent paragraphs."""
        return paragraph_separator.join(para.get_text() for para in self.paragraphs)
    
    def __post_init__(self):
        """Initialize block text from paragraphs."""
        if not self.text and self.paragraphs:
            self.text = self.get_text()


@dataclass
class Page:
    """Page-level container for OCR results with layout information."""
    page_number: int
    blocks: List[Block] = field(default_factory=list)
    width: float = 0.0
    height: float = 0.0
    dpi: float = 300.0
    rotation: float = 0.0
    confidence: ConfidenceMetrics = field(default_factory=lambda: ConfidenceMetrics(overall=0.0))
    detected_languages: List[str] = field(default_factory=lambda: ["en"])
    processing_time: float = 0.0
    reading_order: List[str] = field(default_factory=list)
    column_count: int = 1
    has_tables: bool = False
    has_images: bool = False
    
    def get_text(self, block_separator: str = "\n\n") -> str:
        """Get full page text from all blocks."""
        return block_separator.join(block.get_text() for block in self.blocks)
    
    def get_all_words(self) -> List[Word]:
        """Get all words on the page in a flat list."""
        words = []
        for block in self.blocks:
            for paragraph in block.paragraphs:
                for line in paragraph.lines:
                    words.extend(line.words)
        return words
    
    def word_count(self) -> int:
        """Get total word count for the page."""
        return len(self.get_all_words())


@dataclass
class OCRResult:
    """Primary OCR result container with hierarchical structure and metadata."""
    text: str
    confidence: float
    processing_time: float = 0.0
    engine_name: str = ""
    pages: List[Page] = field(default_factory=list)
    detected_languages: List[str] = field(default_factory=lambda: ["en"])
    dominant_language: str = "en"
    text_orientation: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    success: bool = True
    error_message: Optional[str] = None
    total_words: int = 0
    total_characters: int = 0
    avg_word_confidence: float = 0.0
    processing_metrics: Optional[ProcessingMetrics] = None
    
    def __post_init__(self):
        """Initialize derived properties."""
        if not self.text and self.pages:
            self.text = self.get_full_text()
        
        self.total_characters = len(self.text)
        self.total_words = self.get_word_count()
        self.avg_word_confidence = self.calculate_avg_word_confidence()
    
    def get_full_text(self, page_separator: str = "\n\n---\n\n") -> str:
        """Get complete document text from all pages."""
        return page_separator.join(page.get_text() for page in self.pages)
    
    def get_word_count(self) -> int:
        """Get total word count across all pages."""
        return sum(page.word_count() for page in self.pages)
    
    def get_all_words(self) -> List[Word]:
        """Get all words from all pages in a flat list."""
        words = []
        for page in self.pages:
            words.extend(page.get_all_words())
        return words
    
    def calculate_avg_word_confidence(self) -> float:
        """Calculate average confidence across all words."""
        words = self.get_all_words()
        if not words:
            return 0.0
        return sum(word.confidence.overall for word in words) / len(words)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'text': self.text,
            'confidence': self.confidence,
            'processing_time': self.processing_time,
            'engine_name': self.engine_name,
            'total_words': self.total_words,
            'total_characters': self.total_characters,
            'avg_word_confidence': self.avg_word_confidence,
            'detected_languages': self.detected_languages,
            'dominant_language': self.dominant_language,
            'success': self.success,
            'error_message': self.error_message,
            'timestamp': self.timestamp,
            'metadata': self.metadata,
            'pages': len(self.pages)
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Export to JSON format."""
        return json.dumps(self.to_dict(), indent=indent)
    
    def export_words_with_bbox(self) -> List[Dict[str, Any]]:
        """Export word-level results with bounding boxes."""
        results = []
        for word in self.get_all_words():
            x1, y1, x2, y2 = word.bbox.to_xyxy()
            results.append({
                'text': word.text,
                'confidence': word.confidence.overall,
                'bbox': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2},
                'language': word.language
            })
        return results


@dataclass
class BatchResult:
    """Container for multiple OCR results from batch processing."""
    results: List[OCRResult]
    total_pages: int = 0
    total_processing_time: float = 0.0
    avg_confidence: float = 0.0
    success_rate: float = 1.0
    engines_used: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Initialize batch-level properties."""
        if self.results:
            self.total_pages = len(self.results)
            self.total_processing_time = sum(r.processing_time for r in self.results)
            self.avg_confidence = sum(r.confidence for r in self.results) / len(self.results)
            self.success_rate = sum(1 for r in self.results if r.success) / len(self.results)
            self.engines_used = list(set(r.engine_name for r in self.results))

# Type aliases for convenience
BBox = BoundingBox

# Library exports
__all__ = [
    'OCRResult',
    'BatchResult',
    'Word',
    'Line',
    'Paragraph',
    'Block',
    'Page',
    'BoundingBox',
    'ConfidenceMetrics',
    'ProcessingMetrics',
    'BoundingBoxFormat',
    'TextLevel',
    'ContentType',
    'TextRegion',
    'BBox'
]