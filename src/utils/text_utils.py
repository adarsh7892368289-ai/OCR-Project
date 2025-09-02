# src/utils/text_utils.py
import re
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class TextProcessor:
    """Text processing utilities for OCR results"""
    
    def __init__(self):
        pass
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """Analyze text content and return characteristics"""
        if not text or not text.strip():
            return {
                'word_count': 0,
                'character_count': 0,
                'contains_numbers': False,
                'contains_special_characters': False,
                'text_type': 'Empty',
                'estimated_language': 'Unknown'
            }
        
        # Basic statistics
        words = text.split()
        word_count = len(words)
        char_count = len(text)
        
        # Content analysis
        has_numbers = bool(re.search(r'\d', text))
        has_special = bool(re.search(r'[!@#$%^&*(),.?":{}|<>]', text))
        
        # Text type classification
        text_type = self._classify_text_type(text)
        
        return {
            'word_count': word_count,
            'character_count': char_count,
            'contains_numbers': has_numbers,
            'contains_special_characters': has_special,
            'text_type': text_type,
            'estimated_language': 'English'
        }
    
    def _classify_text_type(self, text: str) -> str:
        """Classify the type of text content"""
        text_lower = text.lower()
        
        # Financial documents
        if re.search(r'\b(invoice|receipt|bill|total|amount|price|\$|€|£|\d+\.\d{2})\b', text_lower):
            return 'Financial Document'
        
        # Forms
        elif re.search(r'\b(name|address|phone|email|form)\b', text_lower):
            return 'Form/Personal Information'
        
        # Short text (signs, labels)
        elif len(text.split()) <= 5:
            return 'Sign/Label'
        
        # Default
        else:
            return 'General Text'


def clean_ocr_text(text: str) -> str:
    """Clean and normalize OCR text"""
    if not text:
        return ""
    
    # Remove extra whitespace
    cleaned = ' '.join(text.split())
    
    # Fix common OCR errors
    replacements = {
        '0': 'O',  # Sometimes zeros are mistaken for O
        'l': 'I',  # lowercase l for uppercase I
    }
    
    # Apply replacements cautiously
    # (You might want to make this more sophisticated)
    
    return cleaned.strip()


def extract_confidence_score(result: Dict) -> float:
    """Extract confidence score from OCR result"""
    return result.get('confidence', 0.0)