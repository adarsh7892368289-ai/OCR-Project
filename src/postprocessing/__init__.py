 
"""
Enhanced Postprocessing Module
Advanced OCR post-processing with document structure understanding

This module provides comprehensive post-processing capabilities for OCR results:
- Layout analysis and document structure detection
- Intelligent text correction and enhancement
- Confidence-based filtering and validation
- Multi-format result formatting and export
"""

from .layout_analyzer import (
    EnhancedLayoutAnalyzer,
    LayoutAnalysis,
    TextBlock,
    TableStructure,
    LayoutType,
    TextBlockType
)

from .document_processor import (
    DocumentProcessor,
    ProcessingResult,
    ProcessingPipeline,
    ProcessingStage,
    DocumentType
)

from .structure_extractor import (
    StructureExtractor,
    DocumentStructure,
    StructureElement,
    ElementType,
    HierarchyLevel
)

from .quality_validator import (
    QualityValidator,
    QualityMetrics,
    ValidationResult,
    QualityLevel
)

from .export_manager import (
    ExportManager,
    ExportFormat,
    ExportOptions,
    DocumentExporter
)

__all__ = [
    # Layout Analysis
    'EnhancedLayoutAnalyzer',
    'LayoutAnalysis',
    'TextBlock', 
    'TableStructure',
    'LayoutType',
    'TextBlockType',
    
    # Document Processing
    'DocumentProcessor',
    'ProcessingResult',
    'ProcessingPipeline',
    'ProcessingStage',
    'DocumentType',
    
    # Structure Extraction
    'StructureExtractor',
    'DocumentStructure',
    'StructureElement',
    'ElementType',
    'HierarchyLevel',
    
    # Quality Validation
    'QualityValidator',
    'QualityMetrics',
    'ValidationResult',
    'QualityLevel',
    
    # Export Management
    'ExportManager',
    'ExportFormat',
    'ExportOptions',
    'DocumentExporter'
]

__version__ = "1.0.0"