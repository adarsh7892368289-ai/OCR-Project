# src/postprocessing/confidence_filter.py
import re
from typing import List, Dict, Any, Tuple
from spellchecker import SpellChecker
import string
from collections import Counter

class ConfidenceFilter:
    """Filter OCR results based on confidence scores"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.min_confidence = config.get("min_confidence", 0.5)
        self.min_word_length = config.get("min_word_length", 2)
        self.suspicious_patterns = config.get("suspicious_patterns", [
            r'^[^a-zA-Z0-9\s]*$',  # Only special characters
            r'^.{1}$',              # Single character words (except common ones)
            r'[^\x20-\x7E]',        # Non-printable characters
        ])
        
    def filter_results(self, results: List[Any]) -> List[Any]:
        """Filter OCR results based on confidence and quality"""
        filtered = []
        
        for result in results:
            if self._is_valid_result(result):
                filtered.append(result)
                
        return filtered
        
    def _is_valid_result(self, result: Any) -> bool:
        """Check if result meets quality criteria"""
        # Check confidence threshold
        if hasattr(result, 'confidence') and result.confidence < self.min_confidence:
            return False
            
        # Check text content
        if hasattr(result, 'text'):
            text = result.text.strip()
            
            # Check minimum length
            if len(text) < self.min_word_length:
                # Allow single character if it's alphanumeric
                if len(text) == 1 and not text.isalnum():
                    return False
                    
            # Check for suspicious patterns
            for pattern in self.suspicious_patterns:
                if re.search(pattern, text):
                    return False
                    
        return True
        
    def calculate_text_quality_score(self, text: str) -> float:
        """Calculate quality score for text"""
        if not text.strip():
            return 0.0
            
        score = 1.0
        
        # Penalize too many special characters
        special_char_ratio = sum(1 for c in text if not c.isalnum() and c not in ' .,!?-') / len(text)
        score *= (1.0 - min(0.5, special_char_ratio))
        
        # Reward readable text patterns
        word_count = len(text.split())
        if word_count > 0:
            avg_word_length = len(text.replace(' ', '')) / word_count
            if 3 <= avg_word_length <= 8:  # Ideal word length range
                score *= 1.1
                
        # Penalize excessive punctuation
        punct_ratio = sum(1 for c in text if c in string.punctuation) / len(text)
        if punct_ratio > 0.3:
            score *= 0.8
            
        return min(1.0, score)