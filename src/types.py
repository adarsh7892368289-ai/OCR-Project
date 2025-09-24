from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union
from enum import Enum
import numpy as np

class ProcessingStrategy(Enum):
    MINIMAL = "minimal"      # High quality - minimal processing
    BALANCED = "balanced"    # Standard processing
    ENHANCED = "enhanced"    # Poor quality - heavy enhancement

@dataclass
class QualityMetrics:
    overall_score: float
    sharpness_score: float
    contrast_score: float
    brightness_score: float
    quality_level: str
    needs_enhancement: bool

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