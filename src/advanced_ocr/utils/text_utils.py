# src/advanced_ocr/utils/text_utils.py
"""
Advanced OCR Text Utilities

This module provides comprehensive text processing and validation utilities for the
advanced OCR system. It handles text cleaning, normalization, quality assessment,
and merging operations specifically designed for OCR output processing.

The module focuses on:
- OCR artifact removal and text cleaning
- Unicode normalization and encoding fixes
- Text quality validation and scoring
- Multi-source text merging and deduplication
- Character pattern analysis and language detection

Classes:
    TextCleaner: Basic text cleaning operations for OCR output
    UnicodeNormalizer: Unicode normalization and encoding issue fixes
    TextValidator: Text validation, quality scoring, and language hints
    TextMerger: Text merging from multiple sources or regions

Functions:
    clean_ocr_text: Convenience function for comprehensive text cleaning
    validate_ocr_result: Validate OCR text quality with scoring
    merge_multiple_texts: Merge multiple text strings using various methods

Example:
    >>> from advanced_ocr.utils.text_utils import clean_ocr_text, validate_ocr_result
    >>> cleaned = clean_ocr_text("H3ll0 W0rld!", aggressive=True)
    >>> is_valid, score = validate_ocr_result(cleaned)
    >>> print(f"Text: '{cleaned}', Valid: {is_valid}, Score: {score:.2f}")

    >>> texts = ["Hello", "world", "Hello world"]
    >>> merged = merge_multiple_texts(texts, method="overlap")
    >>> print(f"Merged text: '{merged}'")
"""

import re
import unicodedata
from typing import List, Dict, Optional, Tuple, Union
import string
from collections import Counter
import difflib


class TextCleaner:
    """
    Provides basic text cleaning operations for OCR output.

    This class removes common OCR artifacts, control characters, and normalizes
    whitespace without semantic analysis. It handles character confusions, noise
    patterns, and formatting issues commonly encountered in OCR results.

    Attributes:
        OCR_ARTIFACTS (Dict[str, str]): Common OCR misrecognition patterns and replacements.
        WHITESPACE_PATTERNS (List[Tuple[str, str]]): Patterns for whitespace normalization.
    """
    
    # Common OCR misrecognition patterns
    OCR_ARTIFACTS = {
        # Common character confusions
        '|': 'I',  # Pipe to I
        '0': 'O',  # Zero to O (context-dependent)
        '5': 'S',  # Five to S (context-dependent)
        '1': 'l',  # One to lowercase l (context-dependent)
        
        # Common noise patterns
        r'[•·•]': '',  # Bullet points and middle dots
        r'[\x00-\x1f\x7f-\x9f]': '',  # Control characters
        r'[﻿‌‍]': '',  # Zero-width characters
    }
    
    # Whitespace normalization patterns
    WHITESPACE_PATTERNS = [
        (r'\s+', ' '),  # Multiple spaces to single space
        (r'\n\s*\n', '\n\n'),  # Multiple newlines to double newline
        (r'[ \t]+\n', '\n'),  # Trailing whitespace before newlines
        (r'\n[ \t]+', '\n'),  # Leading whitespace after newlines
    ]
    
    @staticmethod
    def remove_artifacts(text: str) -> str:
        """
        Remove common OCR artifacts and noise from text.
        
        Args:
            text (str): Input text with potential artifacts
            
        Returns:
            str: Cleaned text with artifacts removed
        """
        if not text:
            return ""
        
        cleaned_text = text
        
        # Remove control characters and non-printable characters
        cleaned_text = ''.join(char for char in cleaned_text 
                             if unicodedata.category(char) != 'Cc')
        
        # Apply artifact removal patterns
        for pattern, replacement in TextCleaner.OCR_ARTIFACTS.items():
            if pattern.startswith('r\''):
                # Regex pattern
                cleaned_text = re.sub(pattern[2:-1], replacement, cleaned_text)
            else:
                # Simple string replacement
                cleaned_text = cleaned_text.replace(pattern, replacement)
        
        return cleaned_text
    
    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """
        Normalize whitespace patterns in text.
        
        Args:
            text (str): Input text with irregular whitespace
            
        Returns:
            str: Text with normalized whitespace
        """
        if not text:
            return ""
        
        normalized_text = text
        
        # Apply whitespace normalization patterns
        for pattern, replacement in TextCleaner.WHITESPACE_PATTERNS:
            normalized_text = re.sub(pattern, replacement, normalized_text)
        
        # Strip leading and trailing whitespace
        normalized_text = normalized_text.strip()
        
        return normalized_text
    
    @staticmethod
    def clean_text(text: str, remove_artifacts: bool = True, 
                   normalize_whitespace: bool = True) -> str:
        """
        Comprehensive text cleaning operation.
        
        Args:
            text (str): Input text to clean
            remove_artifacts (bool): Whether to remove OCR artifacts
            normalize_whitespace (bool): Whether to normalize whitespace
            
        Returns:
            str: Cleaned text
        """
        if not text:
            return ""
        
        cleaned_text = text
        
        if remove_artifacts:
            cleaned_text = TextCleaner.remove_artifacts(cleaned_text)
        
        if normalize_whitespace:
            cleaned_text = TextCleaner.normalize_whitespace(cleaned_text)
        
        return cleaned_text


class UnicodeNormalizer:
    """
    Provides Unicode normalization and encoding issue fixes for OCR text.

    This class handles common Unicode normalization forms and fixes encoding
    artifacts that may occur during OCR processing, such as mojibake or
    improperly decoded characters from various character encodings.
    """
    
    @staticmethod
    def normalize_unicode(text: str, form: str = 'NFC') -> str:
        """
        Normalize Unicode characters to canonical form.
        
        Args:
            text (str): Input text with potential Unicode issues
            form (str): Unicode normalization form ('NFC', 'NFD', 'NFKC', 'NFKD')
            
        Returns:
            str: Unicode-normalized text
        """
        if not text:
            return ""
        
        return unicodedata.normalize(form, text)
    
    @staticmethod
    def fix_encoding_issues(text: str) -> str:
        """
        Fix common encoding issues in OCR text.
        
        Args:
            text (str): Text with potential encoding issues
            
        Returns:
            str: Text with encoding issues resolved
        """
        if not text:
            return ""
        
        # Common encoding issue patterns
        encoding_fixes = {
            'Ã¡': 'á', 'Ã©': 'é', 'Ã­': 'í', 'Ã³': 'ó', 'Ãº': 'ú',
            'Ã±': 'ñ', 'Ã¼': 'ü', 'Ã¤': 'ä', 'Ã¶': 'ö', 'Ã¨': 'è',
            'â€™': "'", 'â€œ': '"', 'â€�': '"', 'â€"': '–', 'â€"': '—',
        }
        
        fixed_text = text
        for wrong, correct in encoding_fixes.items():
            fixed_text = fixed_text.replace(wrong, correct)
        
        return fixed_text


class TextValidator:
    """
    Provides basic text validation utilities for OCR results.

    This class offers methods to validate OCR text quality, check basic validity
    criteria, calculate quality scores based on character patterns, and detect
    potential language hints from character sets. Useful for filtering out
    low-quality OCR results and providing basic text quality assessment.
    """
    
    @staticmethod
    def is_valid_text(text: str, min_length: int = 1, 
                     max_length: int = 100000) -> bool:
        """
        Check if text meets basic validity criteria.
        
        Args:
            text (str): Text to validate
            min_length (int): Minimum acceptable length
            max_length (int): Maximum acceptable length
            
        Returns:
            bool: True if text is valid, False otherwise
        """
        if not isinstance(text, str):
            return False
        
        if not text.strip():
            return False
        
        if not (min_length <= len(text) <= max_length):
            return False
        
        return True
    
    @staticmethod
    def calculate_text_quality_score(text: str) -> float:
        """
        Calculate a basic quality score for OCR text based on character patterns.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            float: Quality score between 0.0 and 1.0
        """
        if not text:
            return 0.0
        
        # Metrics for text quality
        total_chars = len(text)
        if total_chars == 0:
            return 0.0
        
        # Count different character types
        alpha_chars = sum(1 for c in text if c.isalpha())
        digit_chars = sum(1 for c in text if c.isdigit())
        space_chars = sum(1 for c in text if c.isspace())
        punct_chars = sum(1 for c in text if c in string.punctuation)
        other_chars = total_chars - alpha_chars - digit_chars - space_chars - punct_chars
        
        # Calculate ratios
        alpha_ratio = alpha_chars / total_chars
        valid_chars_ratio = (alpha_chars + digit_chars + space_chars + punct_chars) / total_chars
        
        # Quality score based on character distribution
        quality_score = (alpha_ratio * 0.6 + valid_chars_ratio * 0.4)
        
        # Penalize excessive repetitive characters
        char_counts = Counter(text)
        max_repetition = max(char_counts.values()) if char_counts else 0
        repetition_penalty = max(0, (max_repetition - total_chars * 0.3) / total_chars)
        
        quality_score = max(0.0, quality_score - repetition_penalty)
        
        return min(1.0, quality_score)
    
    @staticmethod
    def detect_language_hints(text: str) -> Dict[str, float]:
        """
        Detect basic language hints from character patterns.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            Dict[str, float]: Language hints with confidence scores
        """
        if not text:
            return {}
        
        hints = {}
        
        # Basic character set analysis
        has_latin = bool(re.search(r'[a-zA-Z]', text))
        has_cyrillic = bool(re.search(r'[а-яё]', text, re.IGNORECASE))
        has_arabic = bool(re.search(r'[\u0600-\u06FF]', text))
        has_chinese = bool(re.search(r'[\u4e00-\u9fff]', text))
        has_japanese = bool(re.search(r'[\u3040-\u309f\u30a0-\u30ff]', text))
        
        if has_latin:
            hints['latin'] = 0.8
        if has_cyrillic:
            hints['cyrillic'] = 0.8
        if has_arabic:
            hints['arabic'] = 0.8
        if has_chinese:
            hints['chinese'] = 0.8
        if has_japanese:
            hints['japanese'] = 0.8
        
        return hints


class TextMerger:
    """
    Provides utilities for merging text from multiple sources or regions.

    This class offers methods to combine text from different OCR results,
    handle overlapping content, and merge text regions with proper formatting.
    Useful for consolidating results from multiple OCR engines or processing
    text from different image regions.
    """
    
    @staticmethod
    def merge_text_lines(lines: List[str], separator: str = '\n') -> str:
        """
        Merge multiple text lines with specified separator.
        
        Args:
            lines (List[str]): List of text lines to merge
            separator (str): Separator between lines
            
        Returns:
            str: Merged text
        """
        if not lines:
            return ""
        
        # Filter out empty lines
        non_empty_lines = [line.strip() for line in lines if line.strip()]
        
        return separator.join(non_empty_lines)
    
    @staticmethod
    def merge_overlapping_text(text1: str, text2: str, 
                             overlap_threshold: float = 0.8) -> str:
        """
        Merge two text strings that may have overlapping content.
        
        Args:
            text1 (str): First text string
            text2 (str): Second text string
            overlap_threshold (float): Similarity threshold for overlap detection
            
        Returns:
            str: Merged text with overlaps resolved
        """
        if not text1:
            return text2
        if not text2:
            return text1
        
        # Check for overlap using sequence matching
        matcher = difflib.SequenceMatcher(None, text1, text2)
        
        # Find the longest common substring
        match = matcher.find_longest_match(0, len(text1), 0, len(text2))
        
        if match.size > 0:
            overlap_ratio = match.size / min(len(text1), len(text2))
            
            if overlap_ratio >= overlap_threshold:
                # Merge by combining unique parts
                before_overlap = text1[:match.a]
                overlap = text1[match.a:match.a + match.size]
                after_overlap = text2[match.b + match.size:]
                
                return before_overlap + overlap + after_overlap
        
        # No significant overlap found, concatenate with separator
        return f"{text1}\n{text2}"
    
    @staticmethod
    def merge_text_regions(regions: List[Dict[str, Union[str, float]]], 
                          sort_by_position: bool = True) -> str:
        """
        Merge text from multiple regions, optionally sorting by position.
        
        Args:
            regions (List[Dict]): List of text regions with 'text' and position info
            sort_by_position (bool): Whether to sort regions by position
            
        Returns:
            str: Merged text from all regions
        """
        if not regions:
            return ""
        
        # Extract text from regions
        texts = []
        for region in regions:
            if isinstance(region, dict) and 'text' in region:
                text = region['text']
                if text and text.strip():
                    texts.append(text.strip())
            elif isinstance(region, str):
                if region.strip():
                    texts.append(region.strip())
        
        return TextMerger.merge_text_lines(texts)


# Utility functions for external use
def clean_ocr_text(text: str, aggressive: bool = False) -> str:
    """
    Convenience function for cleaning OCR text.
    
    Args:
        text (str): Input OCR text
        aggressive (bool): Whether to apply aggressive cleaning
        
    Returns:
        str: Cleaned text
    """
    if not text:
        return ""
    
    # Apply Unicode normalization
    cleaned = UnicodeNormalizer.normalize_unicode(text)
    cleaned = UnicodeNormalizer.fix_encoding_issues(cleaned)
    
    # Apply basic cleaning
    cleaned = TextCleaner.clean_text(cleaned, 
                                   remove_artifacts=True, 
                                   normalize_whitespace=True)
    
    if aggressive:
        # Additional aggressive cleaning patterns
        aggressive_patterns = [
            (r'[^\w\s\.\,\!\?\-\'\"]', ''),  # Keep only basic punctuation
            (r'\b[a-zA-Z]\b', ''),  # Remove single characters
            (r'\d+[a-zA-Z]+\d+', ''),  # Remove mixed alphanumeric noise
        ]
        
        for pattern, replacement in aggressive_patterns:
            cleaned = re.sub(pattern, replacement, cleaned)
        
        cleaned = TextCleaner.normalize_whitespace(cleaned)
    
    return cleaned


def validate_ocr_result(text: str, min_quality_score: float = 0.3) -> Tuple[bool, float]:
    """
    Validate OCR result text quality.
    
    Args:
        text (str): OCR result text
        min_quality_score (float): Minimum acceptable quality score
        
    Returns:
        Tuple[bool, float]: (is_valid, quality_score)
    """
    if not TextValidator.is_valid_text(text):
        return False, 0.0
    
    quality_score = TextValidator.calculate_text_quality_score(text)
    is_valid = quality_score >= min_quality_score
    
    return is_valid, quality_score


def merge_multiple_texts(texts: List[str], method: str = 'lines') -> str:
    """
    Merge multiple text strings using specified method.
    
    Args:
        texts (List[str]): List of text strings to merge
        method (str): Merge method ('lines', 'overlap', 'regions')
        
    Returns:
        str: Merged text
    """
    if not texts:
        return ""
    
    if method == 'lines':
        return TextMerger.merge_text_lines(texts)
    elif method == 'overlap':
        result = texts[0] if texts else ""
        for text in texts[1:]:
            result = TextMerger.merge_overlapping_text(result, text)
        return result
    elif method == 'regions':
        # Convert strings to region format
        regions = [{'text': text} for text in texts]
        return TextMerger.merge_text_regions(regions)
    else:
        # Default to line merge
        return TextMerger.merge_text_lines(texts)

__all__ = [
    'TextCleaner', 'UnicodeNormalizer', 'TextValidator', 'TextMerger',
    'clean_ocr_text', 'validate_ocr_result', 'merge_multiple_texts'
]