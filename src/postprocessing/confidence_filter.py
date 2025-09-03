# src/postprocessing/confidence_filter.py
from typing import List, Dict, Any
from ..core.base_engine import OCRResult

class ConfidenceFilter:
    """Filter OCR results based on confidence scores"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.min_confidence = config.get("min_confidence", 0.3)  # Lowered for handwritten
        self.min_word_length = config.get("min_word_length", 1)  # Lowered for single chars
        
    def filter_results(self, results: List[OCRResult]) -> List[OCRResult]:
        """Filter results based on confidence and quality"""
        if not results:
            return results
            
        filtered = []
        for result in results:
            if self._should_keep_result(result):
                filtered.append(result)
                
        return filtered
        
    def _should_keep_result(self, result: OCRResult) -> bool:
        """Determine if result should be kept"""
        # Confidence check
        if result.confidence < self.min_confidence:
            return False
            
        # Text length check
        text = result.text.strip()
        if len(text) < self.min_word_length:
            return False
            
        # Check for suspicious patterns
        if self._is_suspicious_text(text):
            return False
            
        return True
        
    def _is_suspicious_text(self, text: str) -> bool:
        """Check if text contains suspicious patterns"""
        if len(text) == 0:
            return True
            
        # Too many special characters (but allow some for handwritten text)
        if len(text) > 3:
            special_char_ratio = sum(1 for c in text if not c.isalnum() and c not in ' .,!?-\'') / len(text)
            if special_char_ratio > 0.6:
                return True
            
        # All identical characters (but allow short sequences)
        if len(set(text.replace(' ', ''))) == 1 and len(text) > 3:
            return True
            
        return False