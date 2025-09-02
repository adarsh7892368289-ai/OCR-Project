# src/utils/text_utils.py

import re
import string
from typing import List, Dict, Set
from collections import Counter

class TextUtils:
    """Utility functions for text processing"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Basic text cleaning"""
        # Remove excessive whitespace
        cleaned = re.sub(r'\s+', ' ', text)
        
        # Remove non-printable characters
        cleaned = ''.join(char for char in cleaned if char.isprintable() or char in '\n\t')
        
        return cleaned.strip()
    
    @staticmethod
    def extract_words(text: str) -> List[str]:
        """Extract words from text"""
        # Split by whitespace and punctuation
        words = re.findall(r'\b\w+\b', text.lower())
        return words
    
    @staticmethod
    def calculate_text_metrics(text: str) -> Dict[str, int]:
        """Calculate text metrics"""
        words = TextUtils.extract_words(text)
        
        return {
            "character_count": len(text),
            "word_count": len(words),
            "line_count": text.count('\n') + 1,
            "paragraph_count": len([p for p in text.split('\n\n') if p.strip()]),
            "unique_words": len(set(words)),
            "average_word_length": sum(len(word) for word in words) / len(words) if words else 0
        }
    
    @staticmethod
    def detect_language(text: str) -> str:
        """Simple language detection based on character patterns"""
        # Very basic language detection - in production, use langdetect library
        
        # Count character types
        latin_chars = sum(1 for c in text if c.isascii() and c.isalpha())
        total_chars = sum(1 for c in text if c.isalpha())
        
        if total_chars == 0:
            return "unknown"
        
        latin_ratio = latin_chars / total_chars
        
        if latin_ratio > 0.8:
            return "en"  # Likely English or other Latin-script language
        else:
            return "other"  # Non-Latin script
    
    @staticmethod
    def extract_numbers(text: str) -> List[str]:
        """Extract numbers from text"""
        return re.findall(r'\b\d+\.?\d*\b', text)
    
    @staticmethod
    def extract_dates(text: str) -> List[str]:
        """Extract date patterns from text"""
        date_patterns = [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # MM/DD/YYYY or MM-DD-YYYY
            r'\b\d{1,2}\s+\w+\s+\d{2,4}\b',        # DD Month YYYY
            r'\b\w+\s+\d{1,2},\s+\d{2,4}\b'        # Month DD, YYYY
        ]
        
        dates = []
        for pattern in date_patterns:
            dates.extend(re.findall(pattern, text, re.IGNORECASE))
        
        return dates
    
    @staticmethod
    def extract_emails(text: str) -> List[str]:
        """Extract email addresses from text"""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        return re.findall(email_pattern, text)
    
    @staticmethod
    def extract_phone_numbers(text: str) -> List[str]:
        """Extract phone numbers from text"""
        phone_patterns = [
            r'\b\d{3}-\d{3}-\d{4}\b',           # XXX-XXX-XXXX
            r'\b\(\d{3}\)\s*\d{3}-\d{4}\b',     # (XXX) XXX-XXXX
            r'\b\d{10}\b'                       # XXXXXXXXXX
        ]
        
        phones = []
        for pattern in phone_patterns:
            phones.extend(re.findall(pattern, text))
        
        return phones