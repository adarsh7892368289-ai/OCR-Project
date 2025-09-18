"""
Modern data structures for OCR results with hierarchical text representation.
Provides clean, efficient containers without complex region handling.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple, Union
from enum import Enum
import json
import time


class CoordinateFormat(Enum):
    """Supported coordinate formats for bounding boxes."""
    XYXY = "xyxy"  # (x1, y1, x2, y2)
    XYWH = "xywh"  # (x, y, width, height)
    CENTER = "center"  # (center_x, center_y, width, height)


@dataclass
class BoundingBox:
    """Flexible bounding box with multiple coordinate format support."""
    
    coordinates: Tuple[float, float, float, float]
    format: CoordinateFormat = CoordinateFormat.XYXY
    
    def __post_init__(self):
        """Validate coordinates after initialization."""
        if len(self.coordinates) != 4:
            raise ValueError("Coordinates must be a tuple of 4 values")
        if any(coord < 0 for coord in self.coordinates):
            raise ValueError("Coordinates cannot be negative")
    
    @property
    def xyxy(self) -> Tuple[float, float, float, float]:
        """Get coordinates in (x1, y1, x2, y2) format."""
        if self.format == CoordinateFormat.XYXY:
            return self.coordinates
        elif self.format == CoordinateFormat.XYWH:
            x, y, w, h = self.coordinates
            return (x, y, x + w, y + h)
        else:  # CENTER format
            cx, cy, w, h = self.coordinates
            return (cx - w/2, cy - h/2, cx + w/2, cy + h/2)
    
    @property
    def xywh(self) -> Tuple[float, float, float, float]:
        """Get coordinates in (x, y, width, height) format."""
        if self.format == CoordinateFormat.XYWH:
            return self.coordinates
        elif self.format == CoordinateFormat.XYXY:
            x1, y1, x2, y2 = self.coordinates
            return (x1, y1, x2 - x1, y2 - y1)
        else:  # CENTER format
            cx, cy, w, h = self.coordinates
            return (cx - w/2, cy - h/2, w, h)
    
    @property
    def center(self) -> Tuple[float, float, float, float]:
        """Get coordinates in (center_x, center_y, width, height) format."""
        if self.format == CoordinateFormat.CENTER:
            return self.coordinates
        elif self.format == CoordinateFormat.XYWH:
            x, y, w, h = self.coordinates
            return (x + w/2, y + h/2, w, h)
        else:  # XYXY format
            x1, y1, x2, y2 = self.coordinates
            return ((x1 + x2)/2, (y1 + y2)/2, x2 - x1, y2 - y1)
    
    @property
    def area(self) -> float:
        """Calculate bounding box area."""
        _, _, w, h = self.xywh
        return w * h
    
    def contains_point(self, x: float, y: float) -> bool:
        """Check if point is inside bounding box."""
        x1, y1, x2, y2 = self.xyxy
        return x1 <= x <= x2 and y1 <= y <= y2
    
    def intersects(self, other: 'BoundingBox') -> bool:
        """Check if this box intersects with another box."""
        x1, y1, x2, y2 = self.xyxy
        ox1, oy1, ox2, oy2 = other.xyxy
        return not (x2 < ox1 or x1 > ox2 or y2 < oy1 or y1 > oy2)
    
    def scale(self, scale_factor: float) -> 'BoundingBox':
        """Scale bounding box by given factor."""
        if self.format == CoordinateFormat.XYXY:
            x1, y1, x2, y2 = self.coordinates
            return BoundingBox(
                (x1 * scale_factor, y1 * scale_factor, 
                 x2 * scale_factor, y2 * scale_factor),
                self.format
            )
        elif self.format == CoordinateFormat.XYWH:
            x, y, w, h = self.coordinates
            return BoundingBox(
                (x * scale_factor, y * scale_factor,
                 w * scale_factor, h * scale_factor),
                self.format
            )
        else:  # CENTER format
            cx, cy, w, h = self.coordinates
            return BoundingBox(
                (cx * scale_factor, cy * scale_factor,
                 w * scale_factor, h * scale_factor),
                self.format
            )


@dataclass
class ConfidenceMetrics:
    """Multi-dimensional confidence scoring for OCR results."""
    
    # Core confidence scores (0.0 to 1.0)
    character_level: float = 0.0
    word_level: float = 0.0
    line_level: float = 0.0
    layout_level: float = 0.0
    
    # Additional quality indicators
    text_quality: float = 0.0  # Linguistic quality assessment
    spatial_quality: float = 0.0  # Spatial arrangement quality
    
    # Engine-specific metadata
    engine_name: Optional[str] = None
    raw_confidence: Optional[float] = None
    processing_time: float = 0.0
    
    def __post_init__(self):
        """Validate confidence scores."""
        scores = [self.character_level, self.word_level, self.line_level, 
                 self.layout_level, self.text_quality, self.spatial_quality]
        for score in scores:
            if not 0.0 <= score <= 1.0:
                raise ValueError(f"Confidence scores must be between 0.0 and 1.0, got {score}")
    
    @property
    def overall_confidence(self) -> float:
        """Calculate weighted overall confidence score."""
        weights = {
            'character': 0.2,
            'word': 0.3,
            'line': 0.2,
            'layout': 0.1,
            'text_quality': 0.1,
            'spatial_quality': 0.1
        }
        
        return (
            self.character_level * weights['character'] +
            self.word_level * weights['word'] +
            self.line_level * weights['line'] +
            self.layout_level * weights['layout'] +
            self.text_quality * weights['text_quality'] +
            self.spatial_quality * weights['spatial_quality']
        )
    
    @property
    def quality_grade(self) -> str:
        """Get quality grade based on overall confidence."""
        confidence = self.overall_confidence
        if confidence >= 0.9:
            return "Excellent"
        elif confidence >= 0.8:
            return "Good"
        elif confidence >= 0.7:
            return "Fair"
        elif confidence >= 0.6:
            return "Poor"
        else:
            return "Very Poor"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'character_level': self.character_level,
            'word_level': self.word_level,
            'line_level': self.line_level,
            'layout_level': self.layout_level,
            'text_quality': self.text_quality,
            'spatial_quality': self.spatial_quality,
            'overall_confidence': self.overall_confidence,
            'quality_grade': self.quality_grade,
            'engine_name': self.engine_name,
            'raw_confidence': self.raw_confidence,
            'processing_time': self.processing_time
        }


@dataclass
class Word:
    """Individual word with position and confidence."""
    
    text: str
    bbox: BoundingBox
    confidence: ConfidenceMetrics
    char_confidences: List[float] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate word data."""
        if not self.text.strip():
            raise ValueError("Word text cannot be empty")
        
        # Ensure char_confidences length matches text length if provided
        if self.char_confidences and len(self.char_confidences) != len(self.text):
            self.char_confidences = []
    
    @property
    def length(self) -> int:
        """Get word length."""
        return len(self.text)
    
    @property
    def is_numeric(self) -> bool:
        """Check if word contains only numbers."""
        return self.text.replace('.', '').replace(',', '').isdigit()
    
    @property
    def is_alphabetic(self) -> bool:
        """Check if word contains only letters."""
        return self.text.replace('-', '').replace("'", '').isalpha()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'text': self.text,
            'bbox': {
                'coordinates': self.bbox.coordinates,
                'format': self.bbox.format.value
            },
            'confidence': self.confidence.to_dict(),
            'char_confidences': self.char_confidences,
            'metadata': {
                'length': self.length,
                'is_numeric': self.is_numeric,
                'is_alphabetic': self.is_alphabetic
            }
        }


@dataclass
class Line:
    """Text line containing multiple words."""
    
    words: List[Word] = field(default_factory=list)
    bbox: Optional[BoundingBox] = None
    confidence: Optional[ConfidenceMetrics] = None
    line_height: float = 0.0
    baseline: float = 0.0
    
    def __post_init__(self):
        """Calculate derived properties after initialization."""
        if self.words:
            self._calculate_bbox()
            self._calculate_confidence()
            self._calculate_line_metrics()
    
    def _calculate_bbox(self):
        """Calculate line bounding box from words."""
        if not self.words:
            return
        
        word_boxes = [word.bbox.xyxy for word in self.words]
        min_x = min(box[0] for box in word_boxes)
        min_y = min(box[1] for box in word_boxes)
        max_x = max(box[2] for box in word_boxes)
        max_y = max(box[3] for box in word_boxes)
        
        self.bbox = BoundingBox((min_x, min_y, max_x, max_y), CoordinateFormat.XYXY)
    
    def _calculate_confidence(self):
        """Calculate line confidence from word confidences."""
        if not self.words:
            return
        
        word_confidences = [word.confidence for word in self.words]
        
        # Weight by word length for better accuracy
        total_chars = sum(len(word.text) for word in self.words)
        if total_chars == 0:
            return
        
        weighted_char = sum(conf.character_level * len(word.text) 
                           for word, conf in zip(self.words, word_confidences)) / total_chars
        weighted_word = sum(conf.word_level * len(word.text) 
                           for word, conf in zip(self.words, word_confidences)) / total_chars
        
        avg_text_quality = sum(conf.text_quality for conf in word_confidences) / len(word_confidences)
        avg_spatial = sum(conf.spatial_quality for conf in word_confidences) / len(word_confidences)
        
        self.confidence = ConfidenceMetrics(
            character_level=weighted_char,
            word_level=weighted_word,
            line_level=weighted_word,  # Line level same as word level for now
            text_quality=avg_text_quality,
            spatial_quality=avg_spatial
        )
    
    def _calculate_line_metrics(self):
        """Calculate line height and baseline."""
        if not self.words or not self.bbox:
            return
        
        _, _, _, height = self.bbox.xywh
        self.line_height = height
        
        # Approximate baseline as 80% of line height from top
        _, y, _, _ = self.bbox.xywh
        self.baseline = y + height * 0.8
    
    @property
    def text(self) -> str:
        """Get combined text from all words."""
        return ' '.join(word.text for word in self.words)
    
    @property
    def word_count(self) -> int:
        """Get number of words in line."""
        return len(self.words)
    
    def add_word(self, word: Word):
        """Add word to line and recalculate properties."""
        self.words.append(word)
        self._calculate_bbox()
        self._calculate_confidence()
        self._calculate_line_metrics()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'text': self.text,
            'words': [word.to_dict() for word in self.words],
            'bbox': {
                'coordinates': self.bbox.coordinates if self.bbox else None,
                'format': self.bbox.format.value if self.bbox else None
            } if self.bbox else None,
            'confidence': self.confidence.to_dict() if self.confidence else None,
            'metadata': {
                'word_count': self.word_count,
                'line_height': self.line_height,
                'baseline': self.baseline
            }
        }


@dataclass
class Paragraph:
    """Paragraph containing multiple lines."""
    
    lines: List[Line] = field(default_factory=list)
    bbox: Optional[BoundingBox] = None
    confidence: Optional[ConfidenceMetrics] = None
    alignment: str = "unknown"  # left, center, right, justified, unknown
    line_spacing: float = 0.0
    
    def __post_init__(self):
        """Calculate derived properties after initialization."""
        if self.lines:
            self._calculate_bbox()
            self._calculate_confidence()
            self._calculate_layout_metrics()
    
    def _calculate_bbox(self):
        """Calculate paragraph bounding box from lines."""
        if not self.lines:
            return
        
        line_boxes = [line.bbox.xyxy for line in self.lines if line.bbox]
        if not line_boxes:
            return
        
        min_x = min(box[0] for box in line_boxes)
        min_y = min(box[1] for box in line_boxes)
        max_x = max(box[2] for box in line_boxes)
        max_y = max(box[3] for box in line_boxes)
        
        self.bbox = BoundingBox((min_x, min_y, max_x, max_y), CoordinateFormat.XYXY)
    
    def _calculate_confidence(self):
        """Calculate paragraph confidence from line confidences."""
        if not self.lines:
            return
        
        line_confidences = [line.confidence for line in self.lines if line.confidence]
        if not line_confidences:
            return
        
        # Weight by line length
        total_chars = sum(len(line.text) for line in self.lines)
        if total_chars == 0:
            return
        
        weighted_char = sum(conf.character_level * len(line.text) 
                           for line, conf in zip(self.lines, line_confidences)) / total_chars
        weighted_word = sum(conf.word_level * len(line.text) 
                           for line, conf in zip(self.lines, line_confidences)) / total_chars
        weighted_line = sum(conf.line_level * len(line.text) 
                           for line, conf in zip(self.lines, line_confidences)) / total_chars
        
        avg_text_quality = sum(conf.text_quality for conf in line_confidences) / len(line_confidences)
        avg_spatial = sum(conf.spatial_quality for conf in line_confidences) / len(line_confidences)
        
        self.confidence = ConfidenceMetrics(
            character_level=weighted_char,
            word_level=weighted_word,
            line_level=weighted_line,
            layout_level=weighted_line,  # Layout level same as line level for paragraphs
            text_quality=avg_text_quality,
            spatial_quality=avg_spatial
        )
    
    def _calculate_layout_metrics(self):
        """Calculate paragraph layout metrics."""
        if len(self.lines) < 2:
            return
        
        # Calculate average line spacing
        spacings = []
        for i in range(len(self.lines) - 1):
            if self.lines[i].bbox and self.lines[i+1].bbox:
                _, y1, _, y1_bottom = self.lines[i].bbox.xyxy
                _, y2, _, _ = self.lines[i+1].bbox.xyxy
                spacing = y2 - y1_bottom
                if spacing > 0:
                    spacings.append(spacing)
        
        if spacings:
            self.line_spacing = sum(spacings) / len(spacings)
        
        # Simple alignment detection based on x-coordinates
        if all(line.bbox for line in self.lines):
            left_edges = [line.bbox.xyxy[0] for line in self.lines]
            right_edges = [line.bbox.xyxy[2] for line in self.lines]
            
            left_var = max(left_edges) - min(left_edges)
            right_var = max(right_edges) - min(right_edges)
            
            if left_var < 5:  # Small variation in left edges
                self.alignment = "left"
            elif right_var < 5:  # Small variation in right edges
                self.alignment = "right"
            elif left_var < 10 and right_var < 10:
                self.alignment = "center"
    
    @property
    def text(self) -> str:
        """Get combined text from all lines."""
        return '\n'.join(line.text for line in self.lines)
    
    @property
    def line_count(self) -> int:
        """Get number of lines in paragraph."""
        return len(self.lines)
    
    @property
    def word_count(self) -> int:
        """Get total number of words in paragraph."""
        return sum(line.word_count for line in self.lines)
    
    def add_line(self, line: Line):
        """Add line to paragraph and recalculate properties."""
        self.lines.append(line)
        self._calculate_bbox()
        self._calculate_confidence()
        self._calculate_layout_metrics()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'text': self.text,
            'lines': [line.to_dict() for line in self.lines],
            'bbox': {
                'coordinates': self.bbox.coordinates if self.bbox else None,
                'format': self.bbox.format.value if self.bbox else None
            } if self.bbox else None,
            'confidence': self.confidence.to_dict() if self.confidence else None,
            'metadata': {
                'line_count': self.line_count,
                'word_count': self.word_count,
                'alignment': self.alignment,
                'line_spacing': self.line_spacing
            }
        }


@dataclass
class Page:
    """Complete page containing paragraphs and metadata."""
    
    paragraphs: List[Paragraph] = field(default_factory=list)
    page_number: int = 1
    image_dimensions: Tuple[int, int] = (0, 0)  # (width, height)
    confidence: Optional[ConfidenceMetrics] = None
    language: str = "unknown"
    processing_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate derived properties after initialization."""
        if self.paragraphs:
            self._calculate_confidence()
    
    def _calculate_confidence(self):
        """Calculate page confidence from paragraph confidences."""
        if not self.paragraphs:
            return
        
        para_confidences = [para.confidence for para in self.paragraphs if para.confidence]
        if not para_confidences:
            return
        
        # Weight by paragraph word count
        total_words = sum(para.word_count for para in self.paragraphs)
        if total_words == 0:
            return
        
        weighted_char = sum(conf.character_level * para.word_count 
                           for para, conf in zip(self.paragraphs, para_confidences)) / total_words
        weighted_word = sum(conf.word_level * para.word_count 
                           for para, conf in zip(self.paragraphs, para_confidences)) / total_words
        weighted_line = sum(conf.line_level * para.word_count 
                           for para, conf in zip(self.paragraphs, para_confidences)) / total_words
        
        avg_text_quality = sum(conf.text_quality for conf in para_confidences) / len(para_confidences)
        avg_spatial = sum(conf.spatial_quality for conf in para_confidences) / len(para_confidences)
        
        # Calculate layout quality based on paragraph organization
        layout_quality = min(0.9, len(self.paragraphs) * 0.1 + 0.5)  # More paragraphs = better layout
        
        self.confidence = ConfidenceMetrics(
            character_level=weighted_char,
            word_level=weighted_word,
            line_level=weighted_line,
            layout_level=layout_quality,
            text_quality=avg_text_quality,
            spatial_quality=avg_spatial
        )
    
    @property
    def text(self) -> str:
        """Get combined text from all paragraphs."""
        return '\n\n'.join(para.text for para in self.paragraphs)
    
    @property
    def paragraph_count(self) -> int:
        """Get number of paragraphs on page."""
        return len(self.paragraphs)
    
    @property
    def line_count(self) -> int:
        """Get total number of lines on page."""
        return sum(para.line_count for para in self.paragraphs)
    
    @property
    def word_count(self) -> int:
        """Get total number of words on page."""
        return sum(para.word_count for para in self.paragraphs)
    
    def add_paragraph(self, paragraph: Paragraph):
        """Add paragraph to page and recalculate confidence."""
        self.paragraphs.append(paragraph)
        self._calculate_confidence()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'text': self.text,
            'paragraphs': [para.to_dict() for para in self.paragraphs],
            'page_number': self.page_number,
            'image_dimensions': self.image_dimensions,
            'confidence': self.confidence.to_dict() if self.confidence else None,
            'language': self.language,
            'processing_metadata': self.processing_metadata,
            'statistics': {
                'paragraph_count': self.paragraph_count,
                'line_count': self.line_count,
                'word_count': self.word_count
            }
        }


@dataclass
class ProcessingMetrics:
    """Performance tracking for OCR pipeline stages."""
    
    stage_name: str
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    duration: Optional[float] = None
    memory_usage: Optional[float] = None  # MB
    gpu_usage: Optional[float] = None    # Percentage
    error_count: int = 0
    warning_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def finish(self):
        """Mark stage as finished and calculate duration."""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
    
    def add_error(self, error_msg: str):
        """Add error to metrics."""
        self.error_count += 1
        if 'errors' not in self.metadata:
            self.metadata['errors'] = []
        self.metadata['errors'].append(error_msg)
    
    def add_warning(self, warning_msg: str):
        """Add warning to metrics."""
        self.warning_count += 1
        if 'warnings' not in self.metadata:
            self.metadata['warnings'] = []
        self.metadata['warnings'].append(warning_msg)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'stage_name': self.stage_name,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.duration,
            'memory_usage': self.memory_usage,
            'gpu_usage': self.gpu_usage,
            'error_count': self.error_count,
            'warning_count': self.warning_count,
            'metadata': self.metadata
        }


@dataclass
class OCRResult:
    """Primary container for complete OCR results with hierarchical structure."""
    
    pages: List[Page] = field(default_factory=list)
    confidence: Optional[ConfidenceMetrics] = None
    processing_time: float = 0.0
    engine_info: Dict[str, Any] = field(default_factory=dict)
    processing_metrics: List[ProcessingMetrics] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate document-level confidence after initialization."""
        if self.pages:
            self._calculate_document_confidence()
    
    def _calculate_document_confidence(self):
        """Calculate document confidence from page confidences."""
        if not self.pages:
            return
        
        page_confidences = [page.confidence for page in self.pages if page.confidence]
        if not page_confidences:
            return
        
        # Weight by page word count
        total_words = sum(page.word_count for page in self.pages)
        if total_words == 0:
            return
        
        weighted_char = sum(conf.character_level * page.word_count 
                           for page, conf in zip(self.pages, page_confidences)) / total_words
        weighted_word = sum(conf.word_level * page.word_count 
                           for page, conf in zip(self.pages, page_confidences)) / total_words
        weighted_line = sum(conf.line_level * page.word_count 
                           for page, conf in zip(self.pages, page_confidences)) / total_words
        weighted_layout = sum(conf.layout_level * page.word_count 
                             for page, conf in zip(self.pages, page_confidences)) / total_words
        
        avg_text_quality = sum(conf.text_quality for conf in page_confidences) / len(page_confidences)
        avg_spatial = sum(conf.spatial_quality for conf in page_confidences) / len(page_confidences)
        
        self.confidence = ConfidenceMetrics(
            character_level=weighted_char,
            word_level=weighted_word,
            line_level=weighted_line,
            layout_level=weighted_layout,
            text_quality=avg_text_quality,
            spatial_quality=avg_spatial
        )
    
    @property
    def text(self) -> str:
        """Get combined text from all pages."""
        page_texts = []
        for i, page in enumerate(self.pages):
            if len(self.pages) > 1:
                page_texts.append(f"--- Page {page.page_number} ---\n{page.text}")
            else:
                page_texts.append(page.text)
        return '\n\n'.join(page_texts)
    
    @property
    def page_count(self) -> int:
        """Get number of pages in document."""
        return len(self.pages)
    
    @property
    def paragraph_count(self) -> int:
        """Get total number of paragraphs across all pages."""
        return sum(page.paragraph_count for page in self.pages)
    
    @property
    def line_count(self) -> int:
        """Get total number of lines across all pages."""
        return sum(page.line_count for page in self.pages)
    
    @property
    def word_count(self) -> int:
        """Get total number of words across all pages."""
        return sum(page.word_count for page in self.pages)
    
    def add_page(self, page: Page):
        """Add page to document and recalculate confidence."""
        self.pages.append(page)
        self._calculate_document_confidence()
    
    def add_processing_metric(self, metric: ProcessingMetrics):
        """Add processing metric to document."""
        self.processing_metrics.append(metric)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of processing performance."""
        total_duration = sum(m.duration for m in self.processing_metrics if m.duration)
        total_errors = sum(m.error_count for m in self.processing_metrics)
        total_warnings = sum(m.warning_count for m in self.processing_metrics)
        
        return {
            'total_processing_time': self.processing_time,
            'stage_processing_time': total_duration,
            'total_errors': total_errors,
            'total_warnings': total_warnings,
            'stages_completed': len(self.processing_metrics),
            'engine_info': self.engine_info
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert complete result to dictionary for serialization."""
        return {
            'text': self.text,
            'pages': [page.to_dict() for page in self.pages],
            'confidence': self.confidence.to_dict() if self.confidence else None,
            'processing_time': self.processing_time,
            'engine_info': self.engine_info,
            'processing_metrics': [metric.to_dict() for metric in self.processing_metrics],
            'metadata': self.metadata,
            'statistics': {
                'page_count': self.page_count,
                'paragraph_count': self.paragraph_count,
                'line_count': self.line_count,
                'word_count': self.word_count
            },
            'performance_summary': self.get_performance_summary()
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


@dataclass
class BatchResult:
    """Container for multi-document OCR processing results."""
    
    results: List[OCRResult] = field(default_factory=list)
    batch_id: Optional[str] = None
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    total_processing_time: Optional[float] = None
    failed_documents: List[Dict[str, Any]] = field(default_factory=list)
    batch_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def finish(self):
        """Mark batch processing as finished."""
        self.end_time = time.time()
        self.total_processing_time = self.end_time - self.start_time
    
    def add_result(self, result: OCRResult):
        """Add successful result to batch."""
        self.results.append(result)
    
    def add_failed_document(self, document_path: str, error: str):
        """Add failed document information."""
        self.failed_documents.append({
            'document_path': document_path,
            'error': error,
            'timestamp': time.time()
        })
    
    @property
    def success_count(self) -> int:
        """Get number of successfully processed documents."""
        return len(self.results)
    
    @property
    def failure_count(self) -> int:
        """Get number of failed documents."""
        return len(self.failed_documents)
    
    @property
    def total_documents(self) -> int:
        """Get total number of documents processed."""
        return self.success_count + self.failure_count
    
    @property
    def success_rate(self) -> float:
        """Get processing success rate."""
        if self.total_documents == 0:
            return 0.0
        return self.success_count / self.total_documents
    
    @property
    def total_pages(self) -> int:
        """Get total number of pages across all documents."""
        return sum(result.page_count for result in self.results)
    
    @property
    def total_words(self) -> int:
        """Get total number of words across all documents."""
        return sum(result.word_count for result in self.results)
    
    @property
    def average_confidence(self) -> float:
        """Get average confidence across all documents."""
        confidences = [result.confidence.overall_confidence 
                      for result in self.results if result.confidence]
        if not confidences:
            return 0.0
        return sum(confidences) / len(confidences)
    
    def get_batch_summary(self) -> Dict[str, Any]:
        """Get comprehensive batch processing summary."""
        return {
            'batch_id': self.batch_id,
            'processing_stats': {
                'total_documents': self.total_documents,
                'successful': self.success_count,
                'failed': self.failure_count,
                'success_rate': self.success_rate,
                'total_processing_time': self.total_processing_time
            },
            'content_stats': {
                'total_pages': self.total_pages,
                'total_words': self.total_words,
                'average_confidence': self.average_confidence
            },
            'failed_documents': self.failed_documents,
            'batch_metadata': self.batch_metadata
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert batch result to dictionary."""
        return {
            'results': [result.to_dict() for result in self.results],
            'batch_summary': self.get_batch_summary(),
            'processing_timeline': {
                'start_time': self.start_time,
                'end_time': self.end_time,
                'total_processing_time': self.total_processing_time
            }
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)