"""
Structure Extractor - Document structure extraction component
This provides document structure analysis functionality needed by other post-processing components
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
import logging
from enum import Enum

from ..core.base_engine import OCRResult, TextRegion, BoundingBox
from ..utils.logger import get_logger


class StructureType(Enum):
    """Types of document structures"""
    PARAGRAPH = "paragraph"
    HEADING = "heading"
    LIST = "list"
    TABLE = "table"
    FOOTER = "footer"
    HEADER = "header"
    CAPTION = "caption"
    SIDEBAR = "sidebar"


@dataclass
class DocumentStructureElement:
    """Individual document structure element"""
    element_type: StructureType
    content: str
    bounding_box: Optional[BoundingBox] = None
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractedStructure:
    """Extracted document structure"""
    elements: List[DocumentStructureElement] = field(default_factory=list)
    reading_order: List[int] = field(default_factory=list)
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


class StructureExtractor:
    """
    Basic structure extractor for document analysis
    Minimal implementation to satisfy import dependencies
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        
    def extract_structure(self, ocr_result: OCRResult) -> ExtractedStructure:
        """
        Extract document structure from OCR result
        Basic implementation - can be enhanced as needed
        """
        try:
            elements = []
            
            if ocr_result.regions:
                for i, region in enumerate(ocr_result.regions):
                    element = DocumentStructureElement(
                        element_type=StructureType.PARAGRAPH,  # Default type
                        content=region.text if hasattr(region, 'text') else str(region),
                        bounding_box=region.bbox if hasattr(region, 'bbox') else None,
                        confidence=region.confidence if hasattr(region, 'confidence') else 1.0
                    )
                    elements.append(element)
            
            structure = ExtractedStructure(
                elements=elements,
                reading_order=list(range(len(elements))),
                confidence=1.0
            )
            
            return structure
            
        except Exception as e:
            self.logger.error(f"Error extracting structure: {e}")
            return ExtractedStructure()
    
    def analyze_element_type(self, text: str, bbox: Optional[BoundingBox] = None) -> StructureType:
        """Analyze what type of structure element this is"""
        text_lower = text.lower().strip()
        
        # Simple heuristics
        if len(text_lower) < 5 and text_lower.isupper():
            return StructureType.HEADING
        elif text_lower.startswith(('•', '-', '1.', '2.', 'a)', 'i)')):
            return StructureType.LIST
        elif 'table' in text_lower or '|' in text:
            return StructureType.TABLE
        else:
            return StructureType.PARAGRAPH
    
    def get_statistics(self) -> Dict[str, Any]:
        """Return basic statistics"""
        return {
            'extractor_type': 'basic',
            'config': self.config
        }


# Helper functions
def create_structure_element(
    element_type: StructureType,
    content: str,
    confidence: float = 1.0
) -> DocumentStructureElement:
    """Helper to create structure elements"""
    return DocumentStructureElement(
        element_type=element_type,
        content=content,
        confidence=confidence
    )


# Aliases for backward compatibility
DocumentStructure = ExtractedStructure
StructureElement = DocumentStructureElement
ElementType = StructureType
HierarchyLevel = StructureType

# Export commonly used items
__all__ = [
    'StructureExtractor',
    'DocumentStructureElement',
    'DocumentStructure',  
    'StructureElement',
    'ElementType',
    'HierarchyLevel',
    'ExtractedStructure',
    'StructureType',
    'create_structure_element'
]