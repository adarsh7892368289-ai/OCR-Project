from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union
from enum import Enum
import numpy as np

# Import the comprehensive QualityMetrics from quality_analyzer
try:
    from .preprocessing.quality_analyzer import QualityMetrics, ImageType, ImageQuality
except ImportError:
    # Fallback for when running from different directories
    from preprocessing.quality_analyzer import QualityMetrics, ImageType, ImageQuality

class ProcessingStrategy(Enum):
    MINIMAL = "minimal"      # High quality - minimal processing
    BALANCED = "balanced"    # Standard processing
    ENHANCED = "enhanced"    # Poor quality - heavy enhancement

@dataclass
class ProcessingOptions:
    """Configuration options for OCR processing"""
    engines: Optional[List[str]] = None  # ['paddleocr', 'easyocr', 'tesseract', 'trocr']
    strategy: Optional[ProcessingStrategy] = None  # Auto-detect if None
    enhance_image: bool = True
    min_confidence: float = 0.5
    max_processing_time: int = 120  # seconds
    early_termination: bool = True
    early_termination_threshold: float = 0.95

@dataclass
class OCRResult:
    """Final result from OCR processing"""
    text: str
    confidence: float
    processing_time: float
    engine_used: str
    quality_metrics: QualityMetrics
    strategy_used: ProcessingStrategy
    metadata: Dict = field(default_factory=dict)
    
    @property
    def success(self) -> bool:
        return len(self.text.strip()) > 0 and self.confidence > 0.1