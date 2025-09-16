# =============================================================================
# src/advanced_ocr/postprocessing/__init__.py - POSTPROCESSING API
# =============================================================================
"""
Postprocessing Package - Intelligent Result Enhancement Pipeline 
===============================================================

This package contains all text postprocessing and result fusion components:

Main Orchestrator:
- TextProcessor: Unified postprocessing pipeline coordinator

Specialized Processors:
- ResultFusion: Multi-engine result combination with confidence weighting
- LayoutReconstructor: Document structure and hierarchy reconstruction
- ConfidenceAnalyzer: Advanced multi-dimensional confidence analysis

All postprocessing is designed to maximize accuracy and preserve document structure.
"""

from .text_processor import TextProcessor
from .result_fusion import ResultFusion, FusionStrategy, FusionMethod, FusionDecision, FusionMetrics
from .layout_reconstructor import LayoutReconstructor, LayoutType, ReadingOrder, LayoutAnalysis, ReconstructionMetrics
from .confidence_analyzer import ConfidenceAnalyzer, ConfidenceFactors, ConsensusAnalysis, CharacterAnalysis

__all__ = [
    # Main coordinator
    'TextProcessor',

    # Result fusion
    'ResultFusion',
    'FusionStrategy',
    'FusionMethod',
    'FusionDecision',
    'FusionMetrics',

    # Layout reconstruction
    'LayoutReconstructor',
    'LayoutType',
    'ReadingOrder',
    'LayoutAnalysis',
    'ReconstructionMetrics',

    # Confidence analysis
    'ConfidenceAnalyzer',
    'ConfidenceFactors',
    'ConsensusAnalysis',
    'CharacterAnalysis'
]
