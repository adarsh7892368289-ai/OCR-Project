# src/core/base_engine.py

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional
import cv2
import numpy as np
from dataclasses import dataclass
import time

@dataclass
class OCRResult:
    """Standard OCR result format"""
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    line_level: bool = False
    word_level: bool = False
    char_level: bool = False
    
@dataclass
class DocumentResult:
    """Complete document OCR result"""
    full_text: str
    results: List[OCRResult]
    processing_time: float
    engine_name: str
    image_stats: Dict[str, Any]
    confidence_score: float

class BaseOCREngine(ABC):
    """Abstract base class for all OCR engines"""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}
        self.is_initialized = False
        self.supported_languages = []
        self.model_loaded = False
        
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the OCR engine"""
        pass
        
    @abstractmethod
    def process_image(self, image: np.ndarray, **kwargs) -> DocumentResult:
        """Process an image and return OCR results"""
        pass
        
    @abstractmethod
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages"""
        pass
        
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Common preprocessing steps"""
        if len(image.shape) == 3:
            # Convert to grayscale if needed
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        return gray
        
    def calculate_confidence(self, results: List[OCRResult]) -> float:
        """Calculate overall confidence score"""
        if not results:
            return 0.0
            
        total_confidence = sum(result.confidence for result in results)
        return total_confidence / len(results)
        
    def validate_result(self, result: OCRResult) -> bool:
        """Validate OCR result quality"""
        # Basic validation rules
        if result.confidence < 0.3:
            return False
        if len(result.text.strip()) == 0:
            return False
        if result.bbox[2] <= 0 or result.bbox[3] <= 0:
            return False
            
        return True
        
    def cleanup(self):
        """Cleanup resources"""
        pass
        
    def __enter__(self):
        self.initialize()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()