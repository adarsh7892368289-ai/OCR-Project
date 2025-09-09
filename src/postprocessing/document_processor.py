# src/postprocessing/document_processor.py
"""
Document Processor - Missing component that other modules depend on
This provides basic document processing functionality needed by other post-processing components
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
import logging
import time

from ..core.base_engine import OCRResult, DocumentResult, TextRegion
from ..utils.logger import get_logger


@dataclass
class DocumentStructure:
    """Basic document structure representation"""
    blocks: Optional[List[Any]] = None
    lines: Optional[List[Any]] = None
    paragraphs: Optional[List[Any]] = None
    document_type: str = "unknown"
    reading_order: Optional[List[int]] = None
    
    def __post_init__(self):
        if self.blocks is None:
            self.blocks = []
        if self.lines is None:
            self.lines = []
        if self.paragraphs is None:
            self.paragraphs = []
        if self.reading_order is None:
            self.reading_order = []


@dataclass
class ProcessingContext:
    """Context information for document processing"""
    domain: Optional[str] = None
    language: str = "en"
    document_type: str = "general"
    quality_score: float = 0.0
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ProcessingResult:
    """
    Result of document processing operations
    This class was missing and causing import errors
    """
    original_text: str
    processed_text: str
    processing_time: float
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    statistics: Dict[str, Any] = field(default_factory=dict)
    document_structure: Optional[DocumentStructure] = None
    processing_context: Optional[ProcessingContext] = None
    
    @property
    def success(self) -> bool:
        """Check if processing was successful"""
        return self.confidence > 0.0 and bool(self.processed_text)
    
    @property
    def improvement_ratio(self) -> float:
        """Calculate improvement ratio"""
        if not self.original_text:
            return 0.0
        return abs(len(self.processed_text) - len(self.original_text)) / len(self.original_text)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'original_text': self.original_text,
            'processed_text': self.processed_text,
            'processing_time': self.processing_time,
            'confidence': self.confidence,
            'success': self.success,
            'improvement_ratio': self.improvement_ratio,
            'metadata': self.metadata,
            'statistics': self.statistics,
            'document_structure': self.document_structure.__dict__ if self.document_structure else None,
            'processing_context': self.processing_context.__dict__ if self.processing_context else None
        }


class DocumentProcessor:
    """
    Basic document processor for handling OCR results
    This is a minimal implementation to satisfy import dependencies
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        
    def process_document(self, ocr_result: OCRResult, context: Optional[ProcessingContext] = None) -> DocumentStructure:
        """
        Process OCR result into document structure
        Basic implementation - can be enhanced as needed
        """
        try:
            structure = DocumentStructure()
            
            if ocr_result.regions:
                # Convert OCR regions to basic document blocks
                structure.blocks = ocr_result.regions
                structure.lines = ocr_result.regions  # Simple mapping
                
                # Basic paragraph detection (group by similar Y coordinates)
                paragraphs = self._group_regions_into_paragraphs(ocr_result.regions)
                structure.paragraphs = paragraphs
                
                # Simple reading order (top to bottom, left to right)
                structure.reading_order = list(range(len(ocr_result.regions)))
            
            return structure
            
        except Exception as e:
            self.logger.error(f"Error processing document: {e}")
            return DocumentStructure()
    
    def create_processing_result(
        self,
        original_text: str,
        processed_text: str,
        processing_time: float = 0.0,
        confidence: float = 1.0,
        **kwargs
    ) -> ProcessingResult:
        """
        Create a ProcessingResult instance
        Helper method for other components
        """
        return ProcessingResult(
            original_text=original_text,
            processed_text=processed_text,
            processing_time=processing_time,
            confidence=confidence,
            metadata=kwargs.get('metadata', {}),
            statistics=kwargs.get('statistics', {}),
            document_structure=kwargs.get('document_structure'),
            processing_context=kwargs.get('processing_context')
        )
    
    def _group_regions_into_paragraphs(self, regions: List[Any]) -> List[List[Any]]:
        """Basic paragraph grouping"""
        if not regions:
            return []
        
        # Simple implementation - group regions that are close vertically
        paragraphs = []
        current_paragraph = [regions[0]]
        
        for i in range(1, len(regions)):
            # This is a very basic implementation
            # In practice, you'd use more sophisticated logic
            current_paragraph.append(regions[i])
        
        if current_paragraph:
            paragraphs.append(current_paragraph)
        
        return paragraphs
    
    def get_statistics(self) -> Dict[str, Any]:
        """Return basic statistics"""
        return {
            'processor_type': 'basic',
            'config': self.config
        }


# Commonly imported classes/functions that might be expected
def create_processing_context(domain: Optional[str] = None, language: str = "en") -> ProcessingContext:
    """Helper function to create processing context"""
    return ProcessingContext(domain=domain, language=language)


def create_processing_result(
    original_text: str,
    processed_text: str,
    processing_time: float = 0.0,
    confidence: float = 1.0,
    **kwargs
) -> ProcessingResult:
    """Helper function to create processing result"""
    return ProcessingResult(
        original_text=original_text,
        processed_text=processed_text,
        processing_time=processing_time,
        confidence=confidence,
        **kwargs
    )


def analyze_document_type(ocr_result: OCRResult) -> str:
    """Basic document type analysis"""
    # Very basic implementation
    text = ocr_result.full_text.lower() if ocr_result.full_text else ""
    
    if any(word in text for word in ['invoice', 'bill', 'total', 'amount']):
        return 'invoice'
    elif any(word in text for word in ['receipt', 'purchased', 'store']):
        return 'receipt'
    elif any(word in text for word in ['contract', 'agreement', 'terms']):
        return 'contract'
    else:
        return 'general'
from enum import Enum

class ProcessingStage(Enum):
    """Processing stages for post-processing pipeline"""
    TEXT_CORRECTION = "text_correction"
    CONFIDENCE_FILTERING = "confidence_filtering"
    LAYOUT_ANALYSIS = "layout_analysis"
    RESULT_FORMATTING = "result_formatting"
    QUALITY_ASSESSMENT = "quality_assessment"
    DOCUMENT_STRUCTURING = "document_structuring"

class DocumentType(Enum):
    """Document type classifications"""
    GENERAL = "general"
    INVOICE = "invoice"
    RECEIPT = "receipt"
    CONTRACT = "contract"
    FORM = "form"
    LETTER = "letter"
    REPORT = "report"
    BOOK = "book"
    ARTICLE = "article"
    HANDWRITTEN = "handwritten"
    MIXED = "mixed"
    TABLE = "table"
    DIAGRAM = "diagram"


# Keep your existing import aliases section and add these:

# Import alias for backward compatibility  
try:
    from .postprocessing_pipeline import PostProcessingPipeline
    # Create alias so other modules can import ProcessingPipeline from here
    ProcessingPipeline = PostProcessingPipeline
except ImportError as e:
    # Fallback if postprocessing_pipeline is not available
    ProcessingPipeline = None
    print(f"Warning: Could not import PostProcessingPipeline: {e}")

__all__ = [
    'DocumentProcessor',
    'DocumentStructure', 
    'ProcessingContext',
    'ProcessingResult',
    'ProcessingStage',
    'DocumentType',
    'ProcessingPipeline',
    'create_processing_context',
    'create_processing_result',
    'analyze_document_type'
]