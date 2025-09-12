"""
Advanced OCR System - Result Data Classes
=========================================

Production-grade result classes for OCR operations with comprehensive metadata,
confidence scoring, and structured document representation.

Author: Production OCR Team
Version: 2.0.0
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime
import json
from pathlib import Path


@dataclass
class TextRegion:
    """
    Represents a detected text region with position and confidence information.
    
    This class stores information about individual text regions detected in an image,
    including their spatial coordinates, text content, and confidence metrics.
    """
    
    # Spatial coordinates (x, y, width, height)
    bbox: Tuple[int, int, int, int]
    
    # Extracted text content
    text: str
    
    # Confidence score (0.0 to 1.0)
    confidence: float
    
    # Engine that detected this region
    engine_name: str = ""
    
    # Additional properties
    font_size: Optional[float] = None
    is_handwritten: Optional[bool] = None
    language: Optional[str] = None
    text_direction: str = "ltr"  # left-to-right, right-to-left, top-to-bottom
    
    def __post_init__(self):
        """Validate region data after initialization."""
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")
        
        if len(self.bbox) != 4:
            raise ValueError(f"BBox must have 4 coordinates (x, y, w, h), got {len(self.bbox)}")
    
    @property
    def area(self) -> int:
        """Calculate the area of the text region."""
        return self.bbox[2] * self.bbox[3]
    
    @property
    def center(self) -> Tuple[int, int]:
        """Get the center coordinates of the text region."""
        x, y, w, h = self.bbox
        return (x + w // 2, y + h // 2)
    
    def overlaps_with(self, other: 'TextRegion', threshold: float = 0.5) -> bool:
        """Check if this region overlaps with another region."""
        x1, y1, w1, h1 = self.bbox
        x2, y2, w2, h2 = other.bbox
        
        # Calculate intersection
        left = max(x1, x2)
        top = max(y1, y2)
        right = min(x1 + w1, x2 + w2)
        bottom = min(y1 + h1, y2 + h2)
        
        if left >= right or top >= bottom:
            return False
        
        intersection_area = (right - left) * (bottom - top)
        union_area = self.area + other.area - intersection_area
        
        return (intersection_area / union_area) >= threshold


@dataclass
class QualityMetrics:
    """
    Image quality assessment metrics for OCR preprocessing decisions.
    
    Contains various image quality indicators that help determine
    the best preprocessing strategy and expected OCR accuracy.
    """
    
    # Overall quality score (0.0 to 1.0)
    overall_score: float
    
    # Individual quality components
    sharpness_score: float
    contrast_score: float
    brightness_score: float
    noise_level: float
    
    # Resolution information
    dpi: Optional[int] = None
    resolution: Optional[Tuple[int, int]] = None
    
    # Content analysis
    text_density: float = 0.0
    has_tables: bool = False
    has_images: bool = False
    skew_angle: float = 0.0
    
    def __post_init__(self):
        """Validate quality metrics after initialization."""
        scores = [self.overall_score, self.sharpness_score, self.contrast_score, 
                 self.brightness_score, self.noise_level, self.text_density]
        
        for score in scores:
            if not (0.0 <= score <= 1.0):
                raise ValueError(f"All quality scores must be between 0.0 and 1.0")
    
    @property
    def needs_enhancement(self) -> bool:
        """Determine if image needs enhancement based on quality metrics."""
        return (self.overall_score < 0.7 or 
                self.sharpness_score < 0.6 or 
                self.contrast_score < 0.6 or
                self.noise_level > 0.4)
    
    @property
    def recommended_preprocessing(self) -> List[str]:
        """Get recommended preprocessing operations based on quality metrics."""
        operations = []
        
        if self.sharpness_score < 0.6:
            operations.append("sharpen")
        
        if self.contrast_score < 0.6:
            operations.append("enhance_contrast")
        
        if self.noise_level > 0.4:
            operations.append("denoise")
        
        if abs(self.skew_angle) > 1.0:
            operations.append("deskew")
        
        if self.brightness_score < 0.4 or self.brightness_score > 0.8:
            operations.append("adjust_brightness")
        
        return operations


@dataclass
class ProcessingMetadata:
    """
    Comprehensive metadata about OCR processing pipeline execution.
    
    Tracks timing, engine usage, preprocessing steps, and system information
    for debugging and performance optimization.
    """
    
    # Timing information
    total_processing_time: float = 0.0
    preprocessing_time: float = 0.0
    ocr_processing_time: float = 0.0
    postprocessing_time: float = 0.0
    
    # Engine information
    engines_used: List[str] = field(default_factory=list)
    primary_engine: str = ""
    engine_selection_reason: str = ""
    
    # Processing pipeline
    preprocessing_steps: List[str] = field(default_factory=list)
    postprocessing_steps: List[str] = field(default_factory=list)
    
    # System information
    timestamp: datetime = field(default_factory=datetime.now)
    system_info: Dict[str, Any] = field(default_factory=dict)
    gpu_used: bool = False
    
    # Quality and performance metrics
    input_image_size: Optional[Tuple[int, int]] = None
    processed_image_size: Optional[Tuple[int, int]] = None
    total_regions_detected: int = 0
    regions_processed: int = 0
    
    def add_timing(self, operation: str, duration: float):
        """Add timing information for a specific operation."""
        if operation == "preprocessing":
            self.preprocessing_time += duration
        elif operation == "ocr":
            self.ocr_processing_time += duration
        elif operation == "postprocessing":
            self.postprocessing_time += duration
        
        self.total_processing_time += duration
    
    def add_preprocessing_step(self, step: str):
        """Add a preprocessing step to the pipeline log."""
        self.preprocessing_steps.append(step)
    
    def add_postprocessing_step(self, step: str):
        """Add a postprocessing step to the pipeline log."""
        self.postprocessing_steps.append(step)
    
    @property
    def processing_speed_chars_per_second(self) -> float:
        """Calculate processing speed in characters per second."""
        if self.total_processing_time == 0:
            return 0.0
        return getattr(self, '_total_chars', 0) / self.total_processing_time
    
    def set_total_characters(self, char_count: int):
        """Set total character count for speed calculation."""
        self._total_chars = char_count


@dataclass
class OCRResult:
    """
    Single OCR operation result with comprehensive metadata and analysis.
    
    This is the primary result class returned by OCR operations, containing
    extracted text, confidence metrics, spatial information, and processing metadata.
    """
    
    # Primary extracted content
    text: str
    
    # Confidence metrics
    overall_confidence: float
    
    # Spatial information
    regions: List[TextRegion] = field(default_factory=list)
    
    # Quality and metadata
    quality_metrics: Optional[QualityMetrics] = None
    processing_metadata: ProcessingMetadata = field(default_factory=ProcessingMetadata)
    
    # Content analysis
    detected_language: Optional[str] = None
    content_type: str = "mixed"  # printed, handwritten, mixed
    has_tables: bool = False
    has_mathematical_content: bool = False
    
    # Error and warning information
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate and initialize result after creation."""
        if not (0.0 <= self.overall_confidence <= 1.0):
            raise ValueError(f"Overall confidence must be between 0.0 and 1.0, got {self.overall_confidence}")
        
        # Set character count for processing speed calculation
        if hasattr(self.processing_metadata, 'set_total_characters'):
            self.processing_metadata.set_total_characters(len(self.text))
    
    @property
    def word_count(self) -> int:
        """Get the number of words in extracted text."""
        return len(self.text.split()) if self.text.strip() else 0
    
    @property
    def character_count(self) -> int:
        """Get the number of characters in extracted text."""
        return len(self.text)
    
    @property
    def line_count(self) -> int:
        """Get the number of lines in extracted text."""
        return len(self.text.splitlines()) if self.text.strip() else 0
    
    @property
    def is_high_quality(self) -> bool:
        """Determine if this is a high-quality OCR result."""
        return (self.overall_confidence >= 0.8 and 
                len(self.errors) == 0 and 
                self.character_count > 0)
    
    @property
    def has_warnings_or_errors(self) -> bool:
        """Check if result has any warnings or errors."""
        return len(self.warnings) > 0 or len(self.errors) > 0
    
    def add_warning(self, message: str):
        """Add a warning message to the result."""
        if message not in self.warnings:
            self.warnings.append(message)
    
    def add_error(self, message: str):
        """Add an error message to the result."""
        if message not in self.errors:
            self.errors.append(message)
    
    def get_text_by_confidence(self, min_confidence: float) -> str:
        """Get text from regions above specified confidence threshold."""
        high_confidence_regions = [
            region for region in self.regions 
            if region.confidence >= min_confidence
        ]
        
        # Sort by vertical position, then horizontal
        high_confidence_regions.sort(key=lambda r: (r.bbox[1], r.bbox[0]))
        
        return " ".join(region.text for region in high_confidence_regions)
    
    def get_regions_by_type(self, content_type: str) -> List[TextRegion]:
        """Get regions filtered by content type (printed/handwritten)."""
        if content_type == "printed":
            return [r for r in self.regions if r.is_handwritten is False]
        elif content_type == "handwritten":
            return [r for r in self.regions if r.is_handwritten is True]
        else:
            return self.regions.copy()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "text": self.text,
            "overall_confidence": self.overall_confidence,
            "word_count": self.word_count,
            "character_count": self.character_count,
            "line_count": self.line_count,
            "detected_language": self.detected_language,
            "content_type": self.content_type,
            "has_tables": self.has_tables,
            "has_mathematical_content": self.has_mathematical_content,
            "is_high_quality": self.is_high_quality,
            "processing_time": self.processing_metadata.total_processing_time,
            "engines_used": self.processing_metadata.engines_used,
            "warnings": self.warnings,
            "errors": self.errors,
            "regions_count": len(self.regions)
        }
    
    def to_json(self, include_regions: bool = False) -> str:
        """Convert result to JSON string."""
        data = self.to_dict()
        
        if include_regions:
            data["regions"] = [
                {
                    "bbox": region.bbox,
                    "text": region.text,
                    "confidence": region.confidence,
                    "engine_name": region.engine_name
                }
                for region in self.regions
            ]
        
        return json.dumps(data, indent=2, default=str)


@dataclass
class DocumentResult:
    """
    Multi-page document processing result with document-level analysis.
    
    Represents the result of processing an entire document (potentially multi-page)
    with document-level structure analysis and aggregated statistics.
    """
    
    # Page-level results
    pages: List[OCRResult]
    
    # Document-level aggregated content
    full_text: str = ""
    
    # Document structure analysis
    document_structure: Dict[str, Any] = field(default_factory=dict)
    table_of_contents: List[Dict[str, Any]] = field(default_factory=list)
    
    # Aggregated statistics
    total_pages: int = 0
    total_word_count: int = 0
    total_character_count: int = 0
    average_confidence: float = 0.0
    
    # Document-level metadata
    document_metadata: Dict[str, Any] = field(default_factory=dict)
    processing_summary: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate aggregated statistics after initialization."""
        self.total_pages = len(self.pages)
        
        if self.pages:
            # Aggregate statistics
            self.total_word_count = sum(page.word_count for page in self.pages)
            self.total_character_count = sum(page.character_count for page in self.pages)
            self.average_confidence = sum(page.overall_confidence for page in self.pages) / len(self.pages)
            
            # Combine full text if not provided
            if not self.full_text:
                self.full_text = "\n\n".join(page.text for page in self.pages)
            
            # Aggregate processing summary
            self._calculate_processing_summary()
    
    def _calculate_processing_summary(self):
        """Calculate document-level processing summary."""
        if not self.pages:
            return
        
        total_processing_time = sum(
            page.processing_metadata.total_processing_time for page in self.pages
        )
        
        all_engines = set()
        for page in self.pages:
            all_engines.update(page.processing_metadata.engines_used)
        
        all_warnings = []
        all_errors = []
        for page in self.pages:
            all_warnings.extend(page.warnings)
            all_errors.extend(page.errors)
        
        self.processing_summary = {
            "total_processing_time": total_processing_time,
            "average_processing_time_per_page": total_processing_time / len(self.pages),
            "engines_used": list(all_engines),
            "total_warnings": len(all_warnings),
            "total_errors": len(all_errors),
            "successful_pages": len([p for p in self.pages if p.is_high_quality]),
            "success_rate": len([p for p in self.pages if p.is_high_quality]) / len(self.pages)
        }
    
    @property
    def is_high_quality(self) -> bool:
        """Determine if the entire document processing was high quality."""
        return (self.average_confidence >= 0.8 and 
                self.processing_summary.get("success_rate", 0) >= 0.8)
    
    def get_page(self, page_number: int) -> Optional[OCRResult]:
        """Get specific page result (1-indexed)."""
        if 1 <= page_number <= len(self.pages):
            return self.pages[page_number - 1]
        return None
    
    def get_high_confidence_pages(self, min_confidence: float = 0.8) -> List[OCRResult]:
        """Get pages with confidence above threshold."""
        return [page for page in self.pages if page.overall_confidence >= min_confidence]
    
    def get_text_by_page_range(self, start_page: int, end_page: int) -> str:
        """Get text from specific page range (1-indexed, inclusive)."""
        if start_page < 1 or end_page > len(self.pages) or start_page > end_page:
            raise ValueError(f"Invalid page range: {start_page}-{end_page}")
        
        selected_pages = self.pages[start_page-1:end_page]
        return "\n\n".join(page.text for page in selected_pages)
    
    def save_results(self, output_path: Union[str, Path], format: str = "json"):
        """Save document results to file."""
        output_path = Path(output_path)
        
        if format.lower() == "json":
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.to_dict(), f, indent=2, default=str, ensure_ascii=False)
        
        elif format.lower() == "txt":
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(self.full_text)
        
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'json' or 'txt'")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert document result to dictionary for serialization."""
        return {
            "total_pages": self.total_pages,
            "total_word_count": self.total_word_count,
            "total_character_count": self.total_character_count,
            "average_confidence": self.average_confidence,
            "is_high_quality": self.is_high_quality,
            "processing_summary": self.processing_summary,
            "document_metadata": self.document_metadata,
            "pages": [page.to_dict() for page in self.pages]
        }


# Convenience functions for result creation
def create_empty_result(error_message: str = "") -> OCRResult:
    """Create an empty OCR result with optional error message."""
    result = OCRResult(
        text="",
        overall_confidence=0.0,
        processing_metadata=ProcessingMetadata()
    )
    
    if error_message:
        result.add_error(error_message)
    
    return result


def create_simple_result(text: str, confidence: float, engine_name: str = "") -> OCRResult:
    """Create a simple OCR result with basic information."""
    metadata = ProcessingMetadata()
    if engine_name:
        metadata.engines_used = [engine_name]
        metadata.primary_engine = engine_name
    
    return OCRResult(
        text=text,
        overall_confidence=confidence,
        processing_metadata=metadata
    )


def merge_results(results: List[OCRResult], strategy: str = "highest_confidence") -> OCRResult:
    """
    Merge multiple OCR results using specified strategy.
    
    Args:
        results: List of OCR results to merge
        strategy: Merging strategy ('highest_confidence', 'longest_text', 'consensus')
    
    Returns:
        Merged OCR result
    """
    if not results:
        return create_empty_result("No results to merge")
    
    if len(results) == 1:
        return results[0]
    
    if strategy == "highest_confidence":
        return max(results, key=lambda r: r.overall_confidence)
    
    elif strategy == "longest_text":
        return max(results, key=lambda r: len(r.text))
    
    elif strategy == "consensus":
        # Simple consensus: average confidence, combine unique text
        combined_text = " ".join(set(r.text.strip() for r in results if r.text.strip()))
        avg_confidence = sum(r.overall_confidence for r in results) / len(results)
        
        merged_metadata = ProcessingMetadata()
        all_engines = set()
        for result in results:
            all_engines.update(result.processing_metadata.engines_used)
        merged_metadata.engines_used = list(all_engines)
        
        return OCRResult(
            text=combined_text,
            overall_confidence=avg_confidence,
            processing_metadata=merged_metadata
        )
    
    else:
        raise ValueError(f"Unknown merge strategy: {strategy}")