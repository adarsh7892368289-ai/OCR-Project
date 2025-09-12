"""
Text processing utilities for OCR postprocessing.

This module provides production-grade utilities for language detection,
spell checking, text cleaning, confidence analysis, and layout reconstruction.

"""

import re
import unicodedata
import string
import math
import logging
from typing import List, Dict, Tuple, Optional, Any, Set, Union
from dataclasses import dataclass
from collections import Counter, defaultdict
from difflib import SequenceMatcher

# Import logger from utils
try:
    from .logger import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)


@dataclass
class TextStatistics:
    """Container for comprehensive text analysis statistics."""
    
    # Basic counts
    character_count: int
    word_count: int
    line_count: int
    sentence_count: int
    paragraph_count: int
    
    # Character analysis
    alpha_ratio: float
    digit_ratio: float
    punct_ratio: float
    whitespace_ratio: float
    
    # Language characteristics
    avg_word_length: float
    avg_sentence_length: float
    vocabulary_richness: float  # Unique words / Total words
    
    # Quality indicators
    confidence_score: float
    readability_score: float
    
    # Content analysis
    has_mathematical: bool = False
    has_tables: bool = False
    has_lists: bool = False
    detected_language: Optional[str] = None
    
    @property
    def quality_category(self) -> str:
        """Categorize text quality based on confidence score."""
        if self.confidence_score >= 0.9:
            return "excellent"
        elif self.confidence_score >= 0.7:
            return "good"
        elif self.confidence_score >= 0.5:
            return "fair"
        else:
            return "poor"


@dataclass
class TextRegion:
    """Container for text region with spatial and confidence information."""
    text: str
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    confidence: float
    language: Optional[str] = None
    is_rotated: bool = False
    rotation_angle: float = 0.0
    text_type: str = 'paragraph'  # 'paragraph', 'heading', 'list', 'table'
    
    @property
    def center_x(self) -> float:
        """Get horizontal center of the text region."""
        return self.bbox[0] + self.bbox[2] / 2
    
    @property
    def center_y(self) -> float:
        """Get vertical center of the text region."""
        return self.bbox[1] + self.bbox[3] / 2
    
    @property
    def area(self) -> int:
        """Get area of the text region."""
        return self.bbox[2] * self.bbox[3]


class TextProcessingError(Exception):
    """Exception raised during text processing operations."""
    pass


# Common word dictionaries for different languages
COMMON_ENGLISH_WORDS = {
    'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have',
    'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do',
    'at', 'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say',
    'her', 'she', 'or', 'an', 'will', 'my', 'one', 'all', 'would',
    'there', 'their', 'what', 'so', 'up', 'out', 'if', 'about',
    'who', 'get', 'which', 'go', 'me', 'when', 'make', 'can',
    'like', 'time', 'no', 'just', 'him', 'know', 'take', 'people',
    'into', 'year', 'your', 'good', 'some', 'could', 'them', 'see',
    'other', 'than', 'then', 'now', 'look', 'only', 'come', 'its',
    'over', 'think', 'also', 'back', 'after', 'use', 'two', 'how',
    'our', 'work', 'first', 'well', 'way', 'even', 'new', 'want',
    'because', 'any', 'these', 'give', 'day', 'most', 'us'
}

COMMON_SPANISH_WORDS = {
    'el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'es', 'se',
    'no', 'te', 'lo', 'le', 'da', 'su', 'por', 'son', 'con', 'para',
    'al', 'una', 'del', 'los', 'las', 'pero', 'más', 'como', 'muy', 'todo'
}

COMMON_FRENCH_WORDS = {
    'le', 'de', 'et', 'à', 'un', 'il', 'être', 'et', 'en', 'avoir',
    'que', 'pour', 'dans', 'ce', 'son', 'une', 'sur', 'avec', 'ne', 'se'
}

COMMON_GERMAN_WORDS = {
    'der', 'die', 'und', 'in', 'den', 'von', 'zu', 'das', 'mit', 'sich',
    'des', 'auf', 'für', 'ist', 'im', 'dem', 'nicht', 'ein', 'eine', 'als'
}

# Language word dictionaries
LANGUAGE_WORDS = {
    'en': COMMON_ENGLISH_WORDS,
    'es': COMMON_SPANISH_WORDS,
    'fr': COMMON_FRENCH_WORDS,
    'de': COMMON_GERMAN_WORDS
}

# Text cleaning patterns
NOISE_PATTERNS = [
    r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]\{\}\"\'\/\\@#$%&*+=<>|~`]',  # Non-standard characters
    r'\s{3,}',  # Multiple whitespace
    r'\.{4,}',  # Multiple dots (more than 3)
    r'\-{4,}',  # Multiple dashes (more than 3)
    r'_{4,}',   # Multiple underscores (more than 3)
    r'[^\S\r\n]{2,}',  # Multiple spaces (but preserve line breaks)
]

# Mathematical expression patterns
MATH_PATTERNS = [
    r'\d+[\+\-\*\/\=]\d+',  # Simple arithmetic
    r'[a-zA-Z]\s*[\+\-\*\/\=]\s*[a-zA-Z0-9]',  # Algebraic expressions
    r'\b(?:sin|cos|tan|log|ln|exp|sqrt|lim|int|sum|prod)\s*\(',  # Mathematical functions
    r'[a-zA-Z]\^[0-9]+',  # Exponents
    r'\b\d+\.\d+\b',   # Decimal numbers in math context
    r'[∫∑∏∆∇αβγδεζηθικλμνξοπρστυφχψω±≤≥≠≈∞]',  # Mathematical symbols
    r'\b(?:theorem|lemma|proof|equation|formula)\b',  # Math keywords
]

# Common OCR confusion pairs
OCR_CONFUSION_PAIRS = {
    # Letter-digit confusions
    'o': '0', 'O': '0', 'l': '1', 'I': '1', '1': 'l', 
    's': '5', 'S': '5', 'g': '9', 'q': '9', 'G': '6',
    'Z': '2', 'z': '2', 'B': '8', 'b': '6',
    
    # Letter confusions
    'rn': 'm', 'ri': 'n', 'cl': 'd', 'fi': 'fl', 'ffi': 'ffl',
    'vv': 'w', 'nn': 'n', 'ii': 'i', 'oo': 'o',
    
    # Common word corrections
    'teh': 'the', 'adn': 'and', 'taht': 'that', 'hte': 'the',
    'wiht': 'with', 'form': 'from', 'woudl': 'would', 'coudl': 'could',
    'shoudl': 'should', 'wont': "won't", 'cant': "can't",
    'dont': "don't", 'isnt': "isn't", 'wasnt': "wasn't"
}

# Context-aware corrections
CONTEXT_CORRECTIONS = [
    # Common phrase fixes
    (r'\bform\s+(?=the|a|an|this|that|these|those)', 'from '),
    (r'\bfrom\s+(?=is|was|are|were|will|would)', 'form '),
    (r'\bteh\s+', 'the '),
    (r'\badn\s+', 'and '),
    (r'\bno\s+(?=one|body|thing|where)', 'no '),
    (r'\b(?:of|if)\s+(?=course)\b', 'of course'),
]


def clean_text(text: str, 
               remove_noise: bool = True,
               normalize_whitespace: bool = True,
               normalize_unicode: bool = True,
               preserve_structure: bool = True,
               fix_encoding: bool = True) -> str:
    """
    Clean and normalize text for better OCR results.
    
    Args:
        text: Input text to clean
        remove_noise: Remove noise characters and patterns
        normalize_whitespace: Normalize whitespace characters
        normalize_unicode: Apply Unicode normalization
        preserve_structure: Preserve paragraph and line structure
        fix_encoding: Fix common encoding issues
    
    Returns:
        Cleaned text string
    
    Raises:
        TextProcessingError: If text cleaning fails
    """
    try:
        if not text or not isinstance(text, str):
            return ""
        
        cleaned = text
        
        # Fix common encoding issues
        if fix_encoding:
            # Fix common UTF-8 encoding issues
            encoding_fixes = [
                ('â€™', "'"),  # Right single quotation mark
                ('â€œ', '"'),  # Left double quotation mark
                ('â€\x9d', '"'),  # Right double quotation mark
                ('â€"', '–'),  # En dash
                ('â€"', '—'),  # Em dash
                ('Ã¡', 'á'), ('Ã©', 'é'), ('Ã­', 'í'), ('Ã³', 'ó'), ('Ãº', 'ú'),  # Accented vowels
                ('ï¿½', ''),   # Replacement character - remove
            ]
            
            for wrong, correct in encoding_fixes:
                cleaned = cleaned.replace(wrong, correct)
        
        # Unicode normalization
        if normalize_unicode:
            cleaned = unicodedata.normalize('NFKC', cleaned)
            
            # Convert smart quotes and dashes to standard versions
            smart_char_fixes = [
                (''', "'"), (''', "'"),  # Smart single quotes
                ('"', '"'), ('"', '"'),  # Smart double quotes
                ('–', '-'), ('—', '-'),  # En/em dashes to hyphens
                ('…', '...'),            # Ellipsis to three dots
            ]
            
            for smart, standard in smart_char_fixes:
                cleaned = cleaned.replace(smart, standard)
        
        # Remove noise patterns
        if remove_noise:
            for pattern in NOISE_PATTERNS:
                cleaned = re.sub(pattern, ' ', cleaned)
            
            # Remove isolated single characters that are likely OCR artifacts
            cleaned = re.sub(r'\b[a-zA-Z]\s+(?=[A-Z])', '', cleaned)
        
        # Normalize whitespace
        if normalize_whitespace:
            if preserve_structure:
                # Preserve paragraph breaks (double newlines)
                paragraphs = cleaned.split('\n\n')
                cleaned_paragraphs = []
                
                for paragraph in paragraphs:
                    # Clean within paragraph but preserve single line breaks
                    lines = paragraph.split('\n')
                    cleaned_lines = []
                    
                    for line in lines:
                        # Normalize spaces within lines
                        line = re.sub(r'\s+', ' ', line.strip())
                        if line:  # Keep non-empty lines
                            cleaned_lines.append(line)
                    
                    if cleaned_lines:
                        cleaned_paragraphs.append('\n'.join(cleaned_lines))
                
                cleaned = '\n\n'.join(cleaned_paragraphs)
            else:
                # Simple whitespace normalization
                cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # Final cleanup - remove empty lines and trim
        cleaned = cleaned.strip()
        
        if len(cleaned) != len(text):
            logger.debug(f"Text cleaned: {len(text)} -> {len(cleaned)} characters")
        
        return cleaned
    
    except Exception as e:
        raise TextProcessingError(f"Text cleaning failed: {str(e)}")


def detect_language(text: str, min_confidence: float = 0.6) -> Optional[str]:
    """
    Detect the primary language of text using character and word patterns.
    
    Args:
        text: Input text for language detection
        min_confidence: Minimum confidence threshold
    
    Returns:
        ISO 639-1 language code or None if detection fails
    
    Raises:
        TextProcessingError: If language detection fails
    """
    try:
        if not text or len(text.strip()) < 10:
            return None
        
        text_lower = text.lower()
        words = re.findall(r'\b[a-zA-Z]+\b', text_lower)
        
        if not words or len(words) < 3:
            return None
        
        # Calculate scores for each language
        language_scores = {}
        
        for lang_code, common_words in LANGUAGE_WORDS.items():
            matching_words = sum(1 for word in words if word in common_words)
            score = matching_words / len(words) if words else 0
            language_scores[lang_code] = score
        
        # Additional language-specific patterns
        pattern_scores = {}
        
        # Spanish patterns
        spanish_patterns = [
            r'ción\b', r'sión\b', r'mente\b', r'\bque\b', r'\bcon\b', 
            r'\bpara\b', r'\besta\b', r'\beste\b', r'ñ'
        ]
        spanish_score = sum(len(re.findall(pattern, text_lower)) for pattern in spanish_patterns)
        pattern_scores['es'] = min(0.3, spanish_score / len(words))
        
        # French patterns
        french_patterns = [
            r'tion\b', r'ment\b', r'\bque\b', r'\bdes\b', r'\bdans\b',
            r'\bavec\b', r'\bpour\b', r'ç', r'à', r'é', r'è'
        ]
        french_score = sum(len(re.findall(pattern, text_lower)) for pattern in french_patterns)
        pattern_scores['fr'] = min(0.3, french_score / len(words))
        
        # German patterns
        german_patterns = [
            r'ung\b', r'lich\b', r'keit\b', r'\bmit\b', r'\bfür\b',
            r'\bauch\b', r'\bwird\b', r'ß', r'ä', r'ö', r'ü'
        ]
        german_score = sum(len(re.findall(pattern, text_lower)) for pattern in german_patterns)
        pattern_scores['de'] = min(0.3, german_score / len(words))
        
        # English patterns (default, no extra patterns needed)
        pattern_scores['en'] = 0.0
        
        # Combine word-based and pattern-based scores
        final_scores = {}
        for lang_code in language_scores:
            final_scores[lang_code] = (
                language_scores[lang_code] * 0.7 + 
                pattern_scores.get(lang_code, 0) * 0.3
            )
        
        # Find language with highest confidence
        best_language = max(final_scores.items(), key=lambda x: x[1])
        
        if best_language[1] >= min_confidence:
            logger.debug(f"Language detected: {best_language[0]} (confidence: {best_language[1]:.3f})")
            return best_language[0]
        
        # If no language meets threshold, return English as default for Latin scripts
        if any(c.isalpha() and ord(c) < 256 for c in text):
            logger.debug("Defaulting to English for Latin script text")
            return 'en'
        
        return None
    
    except Exception as e:
        logger.warning(f"Language detection failed: {str(e)}")
        return None


def correct_common_ocr_errors(text: str, 
                             language: str = 'en',
                             confidence_threshold: float = 0.8) -> str:
    """
    Correct common OCR recognition errors using pattern matching and context.
    
    Args:
        text: Input text with potential OCR errors
        language: Language code for language-specific corrections
        confidence_threshold: Confidence threshold for applying corrections
    
    Returns:
        Text with common OCR errors corrected
    
    Raises:
        TextProcessingError: If correction fails
    """
    try:
        if not text:
            return ""
        
        corrected = text
        corrections_made = 0
        
        # Apply basic confusion pair corrections
        for wrong, correct in OCR_CONFUSION_PAIRS.items():
            if len(wrong) > 1:
                # Word boundary corrections for multi-character patterns
                pattern = r'\b' + re.escape(wrong) + r'\b'
                matches = len(re.findall(pattern, corrected, flags=re.IGNORECASE))
                if matches > 0:
                    corrected = re.sub(pattern, correct, corrected, flags=re.IGNORECASE)
                    corrections_made += matches
            else:
                # Single character corrections (more conservative)
                # Only apply in specific contexts to avoid false positives
                if wrong in '01lI':
                    # Digit/letter confusion - context matters
                    # Convert to digit if surrounded by digits
                    pattern = r'(?<=\d)' + re.escape(wrong) + r'(?=\d)'
                    corrected = re.sub(pattern, correct if correct.isdigit() else wrong, corrected)
                    # Convert to letter if surrounded by letters
                    pattern = r'(?<=[a-zA-Z])' + re.escape(wrong) + r'(?=[a-zA-Z])'
                    corrected = re.sub(pattern, correct if correct.isalpha() else wrong, corrected)
        
        # Apply context-aware corrections
        for pattern, replacement in CONTEXT_CORRECTIONS:
            matches = len(re.findall(pattern, corrected, flags=re.IGNORECASE))
            if matches > 0:
                corrected = re.sub(pattern, replacement, corrected, flags=re.IGNORECASE)
                corrections_made += matches
        
        # Language-specific corrections
        if language == 'en':
            english_corrections = [
                # Common word corrections with context
                (r'\b(?:recieve|receve)\b', 'receive'),
                (r'\b(?:occured|occured)\b', 'occurred'),
                (r'\b(?:seperate|seperate)\b', 'separate'),
                (r'\b(?:definitly|definatly)\b', 'definitely'),
                (r'\b(?:accomodate|acommodate)\b', 'accommodate'),
                
                # Grammar corrections
                (r'\bits\s+(?=own|self)', "it's "),  # its vs it's context
                (r"\bit's\s+(?=color|size|shape|own)", "its "),
                
                # Common phrase corrections
                (r'\ba\s+lot\b', 'a lot'),
                (r'\balot\b', 'a lot'),
                (r'\bwould\s+of\b', 'would have'),
                (r'\bcould\s+of\b', 'could have'),
                (r'\bshould\s+of\b', 'should have'),
            ]
            
            for pattern, replacement in english_corrections:
                matches = len(re.findall(pattern, corrected, flags=re.IGNORECASE))
                if matches > 0:
                    corrected = re.sub(pattern, replacement, corrected, flags=re.IGNORECASE)
                    corrections_made += matches
        
        # Fix common punctuation errors
        punctuation_corrections = [
            (r'\s+([,.!?;:])', r'\1'),  # Remove space before punctuation
            (r'([.!?])\s*([a-z])', r'\1 \2'),  # Ensure space after sentence end
            (r'\s*-\s*(?=\w)', '-'),  # Fix hyphen spacing
            (r'"\s*([^"]*?)\s*"', r'"\1"'),  # Fix quote spacing
            (r"'\s*([^']*?)\s*'", r"'\1'"),  # Fix apostrophe spacing
            (r'\(\s*([^)]*?)\s*\)', r'(\1)'),  # Fix parenthesis spacing
            (r'\[\s*([^\]]*?)\s*\]', r'[\1]'),  # Fix bracket spacing
        ]
        
        for pattern, replacement in punctuation_corrections:
            matches = len(re.findall(pattern, corrected))
            if matches > 0:
                corrected = re.sub(pattern, replacement, corrected)
                corrections_made += matches
        
        # Fix capitalization after sentence endings
        corrected = re.sub(r'([.!?])\s+([a-z])', 
                          lambda m: m.group(1) + ' ' + m.group(2).upper(), 
                          corrected)
        
        if corrections_made > 0:
            logger.debug(f"OCR errors corrected: {corrections_made} corrections made")
        
        return corrected
    
    except Exception as e:
        raise TextProcessingError(f"OCR error correction failed: {str(e)}")


def calculate_text_statistics(text: str, confidence_scores: Optional[List[float]] = None) -> TextStatistics:
    """
    Calculate comprehensive text analysis statistics.
    
    Args:
        text: Input text to analyze
        confidence_scores: Optional confidence scores for each word/character
    
    Returns:
        TextStatistics object with comprehensive analysis
    
    Raises:
        TextProcessingError: If statistics calculation fails
    """
    try:
        if not text:
            return TextStatistics(
                character_count=0, word_count=0, line_count=0,
                sentence_count=0, paragraph_count=0,
                alpha_ratio=0.0, digit_ratio=0.0, punct_ratio=0.0, whitespace_ratio=0.0,
                avg_word_length=0.0, avg_sentence_length=0.0, vocabulary_richness=0.0,
                confidence_score=0.0, readability_score=0.0
            )
        
        # Basic counts
        character_count = len(text)
        words = re.findall(r'\b\w+\b', text)
        word_count = len(words)
        lines = [line for line in text.split('\n') if line.strip()]
        line_count = len(lines)
        sentences = [sent.strip() for sent in re.split(r'[.!?]+', text) if sent.strip()]
        sentence_count = len(sentences)
        paragraphs = [para.strip() for para in text.split('\n\n') if para.strip()]
        paragraph_count = len(paragraphs)
        
        # Character analysis
        alpha_count = sum(1 for c in text if c.isalpha())
        digit_count = sum(1 for c in text if c.isdigit())
        punct_count = sum(1 for c in text if c in string.punctuation)
        whitespace_count = sum(1 for c in text if c.isspace())
        
        alpha_ratio = alpha_count / character_count if character_count > 0 else 0
        digit_ratio = digit_count / character_count if character_count > 0 else 0
        punct_ratio = punct_count / character_count if character_count > 0 else 0
        whitespace_ratio = whitespace_count / character_count if character_count > 0 else 0
        
        # Language characteristics
        avg_word_length = sum(len(word) for word in words) / word_count if word_count > 0 else 0
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        unique_words = set(word.lower() for word in words)
        vocabulary_richness = len(unique_words) / word_count if word_count > 0 else 0
        
        # Confidence score calculation
        if confidence_scores and len(confidence_scores) > 0:
            confidence_score = sum(confidence_scores) / len(confidence_scores)
        else:
            # Estimate confidence based on text characteristics
            confidence_indicators = []
            
            # Alpha ratio indicator (good: 0.7-0.9)
            alpha_score = min(1.0, max(0.0, (alpha_ratio - 0.5) * 2))
            confidence_indicators.append(alpha_score * 0.3)
            
            # Word length indicator (good: 3-8 characters average)
            if avg_word_length > 0:
                word_len_score = max(0.0, 1.0 - abs(avg_word_length - 5.5) / 5.5)
                confidence_indicators.append(word_len_score * 0.2)
            
            # Vocabulary richness indicator (good: 0.6-0.9)
            vocab_score = min(1.0, vocabulary_richness * 1.2)
            confidence_indicators.append(vocab_score * 0.2)
            
            # Sentence length indicator (good: 10-25 words)
            if avg_sentence_length > 0:
                sent_len_score = max(0.0, 1.0 - abs(avg_sentence_length - 17.5) / 17.5)
                confidence_indicators.append(sent_len_score * 0.1)
            
            # Punctuation ratio indicator (good: 0.05-0.15)
            punct_score = max(0.0, 1.0 - abs(punct_ratio - 0.1) / 0.1)
            confidence_indicators.append(punct_score * 0.2)
            
            confidence_score = sum(confidence_indicators) if confidence_indicators else 0.0
        
        # Readability score (simplified Flesch-like calculation)
        if sentence_count > 0 and word_count > 0:
            # Flesch Reading Ease approximation
            avg_sentence_len = word_count / sentence_count
            avg_syllables = avg_word_length * 0.5  # Rough syllable estimation
            
            flesch_score = 206.835 - (1.015 * avg_sentence_len) - (84.6 * avg_syllables)
            # Normalize to 0-1 scale
            readability_score = max(0.0, min(1.0, flesch_score / 100))
        else:
            readability_score = 0.0
        
        # Content analysis
        has_mathematical = any(re.search(pattern, text, re.IGNORECASE) for pattern in MATH_PATTERNS)
        
        # Table detection patterns
        table_patterns = [
            r'\|.*\|',  # Pipe-separated values
            r'\t.*\t',  # Tab-separated values
            r'^\s*\|?[\w\s]+\|[\w\s\|]+\|?\s*$',  # Table-like structure
        ]
        has_tables = any(re.search(pattern, text, re.MULTILINE) for pattern in table_patterns)
        
        # List detection
        list_patterns = [
            r'^\s*[-*•]\s+',  # Bullet lists
            r'^\s*\d+\.\s+',  # Numbered lists
            r'^\s*[a-zA-Z]\.\s+',  # Lettered lists
            r'^\s*\([a-zA-Z0-9]+\)\s+',  # Parenthetical lists
        ]
        has_lists = any(re.search(pattern, text, re.MULTILINE) for pattern in list_patterns)
        
        # Language detection
        detected_language = detect_language(text)
        
        stats = TextStatistics(
            character_count=character_count,
            word_count=word_count,
            line_count=line_count,
            sentence_count=sentence_count,
            paragraph_count=paragraph_count,
            alpha_ratio=alpha_ratio,
            digit_ratio=digit_ratio,
            punct_ratio=punct_ratio,
            whitespace_ratio=whitespace_ratio,
            avg_word_length=avg_word_length,
            avg_sentence_length=avg_sentence_length,
            vocabulary_richness=vocabulary_richness,
            confidence_score=confidence_score,
            readability_score=readability_score,
            has_mathematical=has_mathematical,
            has_tables=has_tables,
            has_lists=has_lists,
            detected_language=detected_language
        )
        
        logger.debug(f"Text statistics calculated: {word_count} words, "
                    f"{confidence_score:.3f} confidence, quality: {stats.quality_category}")
        
        return stats
    
    except Exception as e:
        raise TextProcessingError(f"Text statistics calculation failed: {str(e)}")


def reconstruct_layout(text_regions: List[Union[Dict[str, Any], TextRegion]], 
                      image_width: int, 
                      image_height: int,
                      reading_order: str = 'left_to_right') -> str:
    """
    Reconstruct document layout from text regions with spatial information.
    
    Args:
        text_regions: List of text regions with bbox and text info
        image_width: Original image width
        image_height: Original image height
        reading_order: Reading order ('left_to_right', 'right_to_left', 'top_to_bottom')
    
    Returns:
        Reconstructed text with proper layout
    
    Raises:
        TextProcessingError: If layout reconstruction fails
    """
    try:
        if not text_regions:
            return ""
        
        # Convert dict regions to TextRegion objects if needed
        regions = []
        for region in text_regions:
            if isinstance(region, dict):
                bbox = region.get('bbox', (0, 0, 0, 0))
                text = region.get('text', '').strip()
                confidence = region.get('confidence', 1.0)
                
                if text and len(bbox) == 4:
                    regions.append(TextRegion(
                        text=text,
                        bbox=tuple(bbox),
                        confidence=confidence,
                        language=region.get('language'),
                        text_type=region.get('text_type', 'paragraph')
                    ))
            elif isinstance(region, TextRegion):
                if region.text.strip():
                    regions.append(region)
        
        if not regions:
            return ""
        
        # Sort regions based on reading order
        if reading_order == 'left_to_right':
            # Sort by vertical position (top to bottom), then horizontal (left to right)
            sorted_regions = sorted(regions, key=lambda r: (r.bbox[1], r.bbox[0]))
        elif reading_order == 'right_to_left':
            # Sort by vertical position, then horizontal (right to left)
            sorted_regions = sorted(regions, key=lambda r: (r.bbox[1], -(r.bbox[0] + r.bbox[2])))
        else:  # top_to_bottom
            # Sort by horizontal position, then vertical
            sorted_regions = sorted(regions, key=lambda r: (r.bbox[0], r.bbox[1]))
        
        # Group regions into lines based on vertical overlap
        lines = []
        current_line = []
        line_threshold = 15  # Pixels tolerance for same line
        
        for region in sorted_regions:
            if not current_line:
                current_line = [region]
            else:
                # Check if this region overlaps vertically with current line
                current_line_top = min(r.bbox[1] for r in current_line)
                current_line_bottom = max(r.bbox[1] + r.bbox[3] for r in current_line)
                
                region_top = region.bbox[1]
                region_bottom = region.bbox[1] + region.bbox[3]
                
                # Check for vertical overlap
                overlap = min(current_line_bottom, region_bottom) - max(current_line_top, region_top)
                
                if overlap > line_threshold:
                    current_line.append(region)
                else:
                    # Start new line
                    if current_line:
                        lines.append(current_line)
                    current_line = [region]
        
        # Add last line
        if current_line:
            lines.append(current_line)
        
        # Reconstruct text with proper spacing and structure
        reconstructed_lines = []
        prev_line_bottom = None
        
        for line_regions in lines:
            # Sort regions within line by x-coordinate (left to right)
            if reading_order == 'right_to_left':
                line_regions.sort(key=lambda r: -(r.bbox[0] + r.bbox[2]))
            else:
                line_regions.sort(key=lambda r: r.bbox[0])
            
            # Combine text from regions in line
            line_text = ""
            prev_right = None
            
            for i, region in enumerate(line_regions):
                x, y, w, h = region.bbox
                text = region.text.strip()
                
                if not text:
                    continue
                
                # Add spacing between words based on horizontal distance
                if prev_right is not None and i > 0:
                    gap = x - prev_right
                    char_width = w / max(1, len(text))  # Approximate character width
                    
                    if gap > char_width * 3:  # Significant gap (more than 3 character widths)
                        line_text += "  "  # Double space for large gaps
                    elif gap > char_width * 1.5:  # Medium gap
                        line_text += " "   # Single space
                    elif gap < -char_width * 0.5:  # Overlapping text
                        pass  # No space for overlapping text
                    else:
                        line_text += " "   # Default single space
                
                line_text += text
                prev_right = x + w
            
            if line_text.strip():
                # Determine paragraph breaks based on vertical spacing and text type
                if prev_line_bottom is not None:
                    current_line_top = min(r.bbox[1] for r in line_regions)
                    vertical_gap = current_line_top - prev_line_bottom
                    
                    # Calculate average line height for context
                    avg_height = sum(r.bbox[3] for r in line_regions) / len(line_regions)
                    
                    # Check for heading or different text types
                    current_text_types = [r.text_type for r in line_regions]
                    is_heading = any(t == 'heading' for t in current_text_types)
                    
                    # Paragraph break criteria
                    should_break = False
                    
                    if vertical_gap > avg_height * 2.0:  # Large vertical gap
                        should_break = True
                    elif vertical_gap > avg_height * 1.2 and is_heading:  # Heading with moderate gap
                        should_break = True
                    elif any(t == 'list' for t in current_text_types):  # List items
                        should_break = True
                    
                    if should_break:
                        reconstructed_lines.append("")  # Empty line for paragraph break
                
                reconstructed_lines.append(line_text)
                prev_line_bottom = max(r.bbox[1] + r.bbox[3] for r in line_regions)
        
        # Join lines and clean up extra whitespace
        reconstructed_text = "\n".join(reconstructed_lines)
        
        # Clean up multiple consecutive empty lines
        reconstructed_text = re.sub(r'\n{3,}', '\n\n', reconstructed_text)
        
        # Final cleanup
        reconstructed_text = reconstructed_text.strip()
        
        logger.debug(f"Layout reconstructed: {len(text_regions)} regions -> {len(reconstructed_lines)} lines")
        return reconstructed_text
    
    except Exception as e:
        raise TextProcessingError(f"Layout reconstruction failed: {str(e)}")


def filter_by_confidence(text_regions: List[Union[Dict[str, Any], TextRegion]], 
                        min_confidence: float = 0.5,
                        adaptive_threshold: bool = True) -> List[Union[Dict[str, Any], TextRegion]]:
    """
    Filter text regions by confidence scores with adaptive thresholding.
    
    Args:
        text_regions: List of text regions with confidence scores
        min_confidence: Minimum confidence threshold
        adaptive_threshold: Use adaptive thresholding based on overall quality
    
    Returns:
        Filtered list of high-confidence text regions
    
    Raises:
        TextProcessingError: If filtering fails
    """
    try:
        if not text_regions:
            return []
        
        # Extract confidence scores
        confidences = []
        for region in text_regions:
            if isinstance(region, dict):
                confidences.append(region.get('confidence', 0.0))
            elif isinstance(region, TextRegion):
                confidences.append(region.confidence)
            else:
                confidences.append(0.0)
        
        if not any(confidences):
            logger.warning("No confidence scores found in text regions")
            return text_regions
        
        # Calculate adaptive threshold if enabled
        threshold = min_confidence
        if adaptive_threshold and confidences:
            valid_confidences = [c for c in confidences if c > 0]
            if valid_confidences:
                mean_conf = sum(valid_confidences) / len(valid_confidences)
                std_conf = math.sqrt(sum((c - mean_conf) ** 2 for c in valid_confidences) / len(valid_confidences))
                
                # Adaptive threshold: mean - 0.5 * std, but not below min_confidence
                adaptive_min = max(min_confidence, mean_conf - 0.5 * std_conf)
                threshold = adaptive_min
        
        # Filter regions based on threshold
        filtered_regions = []
        for i, region in enumerate(text_regions):
            confidence = confidences[i]
            if confidence >= threshold:
                filtered_regions.append(region)
        
        logger.debug(f"Confidence filtering: {len(text_regions)} -> {len(filtered_regions)} regions "
                    f"(threshold: {threshold:.3f})")
        
        return filtered_regions
    
    except Exception as e:
        raise TextProcessingError(f"Confidence filtering failed: {str(e)}")


def split_into_sentences(text: str, language: str = 'en') -> List[str]:
    """
    Split text into sentences using language-specific rules.
    
    Args:
        text: Input text to split
        language: Language code for language-specific rules
    
    Returns:
        List of sentences
    
    Raises:
        TextProcessingError: If sentence splitting fails
    """
    try:
        if not text.strip():
            return []
        
        # Language-specific abbreviations that shouldn't end sentences
        abbreviations = {
            'en': ['Dr', 'Mr', 'Mrs', 'Ms', 'Prof', 'Inc', 'Ltd', 'Corp', 'Co', 'etc', 'vs', 'eg', 'ie'],
            'es': ['Sr', 'Sra', 'Dr', 'Dra', 'Prof', 'etc', 'pág', 'cap'],
            'fr': ['M', 'Mme', 'Mlle', 'Dr', 'Prof', 'etc', 'p.ex', 'c.-à-d'],
            'de': ['Dr', 'Prof', 'Herr', 'Frau', 'etc', 'z.B', 'bzw', 'usw']
        }
        
        abbrevs = abbreviations.get(language, abbreviations['en'])
        
        # Create pattern for abbreviations
        abbrev_pattern = r'\b(?:' + '|'.join(re.escape(abbr) for abbr in abbrevs) + r')\.'
        
        # Replace abbreviations temporarily
        protected_text = text
        abbrev_placeholders = {}
        abbrev_matches = re.finditer(abbrev_pattern, text, re.IGNORECASE)
        
        for i, match in enumerate(abbrev_matches):
            placeholder = f"__ABBREV_{i}__"
            abbrev_placeholders[placeholder] = match.group()
            protected_text = protected_text.replace(match.group(), placeholder, 1)
        
        # Basic sentence splitting on sentence endings
        sentence_pattern = r'([.!?]+)(\s+|$)'
        parts = re.split(sentence_pattern, protected_text)
        
        sentences = []
        current_sentence = ""
        
        for i, part in enumerate(parts):
            if re.match(r'[.!?]+', part):
                # This is sentence ending punctuation
                current_sentence += part
                # Look ahead to see if there's whitespace or end of text
                if i + 1 < len(parts) and (parts[i + 1].strip() == "" or i + 1 == len(parts) - 1):
                    # Finalize sentence
                    sentence = current_sentence.strip()
                    if sentence:
                        # Restore abbreviations
                        for placeholder, original in abbrev_placeholders.items():
                            sentence = sentence.replace(placeholder, original)
                        sentences.append(sentence)
                    current_sentence = ""
            elif part.strip():
                current_sentence += part
        
        # Add any remaining text as final sentence
        if current_sentence.strip():
            sentence = current_sentence.strip()
            # Restore abbreviations
            for placeholder, original in abbrev_placeholders.items():
                sentence = sentence.replace(placeholder, original)
            sentences.append(sentence)
        
        # Filter out very short sentences (likely OCR artifacts)
        filtered_sentences = []
        for sent in sentences:
            # Remove abbreviation placeholders if any remain
            for placeholder, original in abbrev_placeholders.items():
                sent = sent.replace(placeholder, original)
            
            # Keep sentences with at least 2 words or 10 characters
            words = sent.split()
            if len(words) >= 2 or len(sent.strip()) >= 10:
                filtered_sentences.append(sent.strip())
        
        logger.debug(f"Text split into {len(filtered_sentences)} sentences")
        return filtered_sentences
    
    except Exception as e:
        raise TextProcessingError(f"Sentence splitting failed: {str(e)}")


def extract_numbers_and_dates(text: str) -> Dict[str, List[str]]:
    """
    Extract numbers, dates, and structured data from text.
    
    Args:
        text: Input text to analyze
    
    Returns:
        Dictionary with extracted structured data
    
    Raises:
        TextProcessingError: If extraction fails
    """
    try:
        extracted = {
            'numbers': [],
            'dates': [],
            'times': [],
            'phone_numbers': [],
            'emails': [],
            'urls': [],
            'currency': [],
            'percentages': [],
            'measurements': []
        }
        
        # Number patterns (integers, decimals, with/without commas)
        number_patterns = [
            r'\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b',  # Numbers with commas (1,234.56)
            r'\b\d+\.\d+\b',                      # Decimal numbers (123.45)
            r'\b\d{4,}\b',                        # Large integers (1234)
            r'\b\d{1,3}\b'                        # Small integers (1-999)
        ]
        
        for pattern in number_patterns:
            matches = re.findall(pattern, text)
            extracted['numbers'].extend(matches)
        
        # Date patterns (various formats)
        date_patterns = [
            # MM/DD/YYYY, MM-DD-YYYY, MM.DD.YYYY
            r'\b(?:0?[1-9]|1[0-2])[/\-\.] *(?:0?[1-9]|[12][0-9]|3[01])[/\-\.] *(?:19|20)?\d{2}\b',
            # DD/MM/YYYY, DD-MM-YYYY, DD.MM.YYYY
            r'\b(?:0?[1-9]|[12][0-9]|3[01])[/\-\.] *(?:0?[1-9]|1[0-2])[/\-\.] *(?:19|20)?\d{2}\b',
            # YYYY/MM/DD, YYYY-MM-DD
            r'\b(?:19|20)\d{2}[/\-\.] *(?:0?[1-9]|1[0-2])[/\-\.] *(?:0?[1-9]|[12][0-9]|3[01])\b',
            # Month DD, YYYY or DD Month YYYY
            r'\b(?:0?[1-9]|[12][0-9]|3[01]) *(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w* *,? *(?:19|20)?\d{2}\b',
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w* *(?:0?[1-9]|[12][0-9]|3[01]) *,? *(?:19|20)?\d{2}\b',
            # ISO format YYYY-MM-DD
            r'\b(?:19|20)\d{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01])\b'
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            extracted['dates'].extend(matches)
        
        # Time patterns
        time_patterns = [
            r'\b(?:0?[1-9]|1[0-2]):[0-5][0-9] *(?:AM|PM)\b',  # 12-hour format
            r'\b(?:[01]?[0-9]|2[0-3]):[0-5][0-9](?::[0-5][0-9])?\b',  # 24-hour format
        ]
        
        for pattern in time_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            extracted['times'].extend(matches)
        
        # Phone numbers (various formats)
        phone_patterns = [
            r'\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b',  # US format
            r'\+[1-9]\d{1,14}\b',  # International format
            r'\b\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b'  # Basic format
        ]
        
        for pattern in phone_patterns:
            matches = re.findall(pattern, text)
            extracted['phone_numbers'].extend(matches)
        
        # Email addresses
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        extracted['emails'] = re.findall(email_pattern, text)
        
        # URLs
        url_patterns = [
            r'https?://[^\s<>"{}|\\^`\[\]]+',  # HTTP/HTTPS URLs
            r'www\.[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?:/[^\s]*)?',  # WWW URLs
            r'[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?:/[^\s]*)?(?=\s|$)'  # Domain.extension
        ]
        
        for pattern in url_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            extracted['urls'].extend(matches)
        
        # Currency amounts
        currency_patterns = [
            r'[\$£€¥₹][\d,]+\.?\d*',  # Currency symbols before
            r'\b\d+\.?\d* *(?:USD|EUR|GBP|JPY|INR|dollars?|euros?|pounds?|rupees?|yen)\b',  # Currency codes/names after
        ]
        
        for pattern in currency_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            extracted['currency'].extend(matches)
        
        # Percentages
        percentage_pattern = r'\b\d+\.?\d*\s*%|\b\d+\.?\d* *percent\b'
        extracted['percentages'] = re.findall(percentage_pattern, text, re.IGNORECASE)
        
        # Measurements
        measurement_patterns = [
            r'\b\d+\.?\d* *(?:mm|cm|m|km|in|ft|yd|mi|kg|g|lb|oz|l|ml|gal)\b',  # Metric/Imperial
            r'\b\d+\.?\d* *(?:meters?|kilometers?|inches?|feet|yards?|miles?)\b',  # Full words
            r'\b\d+\.?\d* *(?:grams?|kilograms?|pounds?|ounces?)\b',  # Weight
            r'\b\d+\.?\d* *(?:liters?|litres?|milliliters?|gallons?)\b'  # Volume
        ]
        
        for pattern in measurement_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            extracted['measurements'].extend(matches)
        
        # Remove duplicates and empty matches
        for key in extracted:
            unique_matches = []
            seen = set()
            for match in extracted[key]:
                match_clean = match.strip()
                if match_clean and match_clean not in seen:
                    unique_matches.append(match_clean)
                    seen.add(match_clean)
            extracted[key] = unique_matches
        
        total_extracted = sum(len(v) for v in extracted.values())
        if total_extracted > 0:
            logger.debug(f"Extracted structured data: {total_extracted} items")
        
        return extracted
    
    except Exception as e:
        raise TextProcessingError(f"Data extraction failed: {str(e)}")


def calculate_text_similarity(text1: str, text2: str, method: str = 'jaccard') -> float:
    """
    Calculate similarity between two text strings.
    
    Args:
        text1: First text string
        text2: Second text string
        method: Similarity method ('jaccard', 'cosine', 'levenshtein')
    
    Returns:
        Similarity score between 0.0 and 1.0
    
    Raises:
        TextProcessingError: If similarity calculation fails
    """
    try:
        if not text1 or not text2:
            return 0.0
        
        if method == 'jaccard':
            # Jaccard similarity using word sets
            words1 = set(re.findall(r'\b\w+\b', text1.lower()))
            words2 = set(re.findall(r'\b\w+\b', text2.lower()))
            
            if not words1 and not words2:
                return 1.0  # Both empty
            if not words1 or not words2:
                return 0.0  # One empty
            
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            
            return intersection / union if union > 0 else 0.0
        
        elif method == 'cosine':
            # Cosine similarity using word frequency vectors
            words1 = re.findall(r'\b\w+\b', text1.lower())
            words2 = re.findall(r'\b\w+\b', text2.lower())
            
            if not words1 or not words2:
                return 0.0
            
            # Create frequency vectors
            all_words = set(words1 + words2)
            freq1 = Counter(words1)
            freq2 = Counter(words2)
            
            vector1 = [freq1.get(word, 0) for word in all_words]
            vector2 = [freq2.get(word, 0) for word in all_words]
            
            # Calculate cosine similarity
            dot_product = sum(a * b for a, b in zip(vector1, vector2))
            magnitude1 = math.sqrt(sum(a * a for a in vector1))
            magnitude2 = math.sqrt(sum(b * b for b in vector2))
            
            if magnitude1 == 0 or magnitude2 == 0:
                return 0.0
            
            return dot_product / (magnitude1 * magnitude2)
        
        elif method == 'levenshtein':
            # Normalized Levenshtein distance
            max_len = max(len(text1), len(text2))
            if max_len == 0:
                return 1.0
            
            # Calculate Levenshtein distance
            distance = _levenshtein_distance(text1.lower(), text2.lower())
            return 1.0 - (distance / max_len)
        
        else:
            raise ValueError(f"Unknown similarity method: {method}")
    
    except Exception as e:
        raise TextProcessingError(f"Text similarity calculation failed: {str(e)}")


def _levenshtein_distance(s1: str, s2: str) -> int:
    """
    Calculate Levenshtein distance between two strings.
    
    Args:
        s1: First string
        s2: Second string
    
    Returns:
        Levenshtein distance
    """
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def normalize_text_for_matching(text: str, 
                               remove_punctuation: bool = True,
                               normalize_case: bool = True,
                               remove_extra_spaces: bool = True) -> str:
    """
    Normalize text for fuzzy matching and comparison.
    
    Args:
        text: Input text to normalize
        remove_punctuation: Remove punctuation marks
        normalize_case: Convert to lowercase
        remove_extra_spaces: Normalize whitespace
    
    Returns:
        Normalized text
    
    Raises:
        TextProcessingError: If normalization fails
    """
    try:
        if not text:
            return ""
        
        normalized = text
        
        # Normalize case
        if normalize_case:
            normalized = normalized.lower()
        
        # Remove punctuation
        if remove_punctuation:
            normalized = re.sub(r'[^\w\s]', ' ', normalized)
        
        # Remove extra spaces
        if remove_extra_spaces:
            normalized = re.sub(r'\s+', ' ', normalized).strip()
        
        return normalized
    
    except Exception as e:
        raise TextProcessingError(f"Text normalization failed: {str(e)}")


def extract_text_features(text: str) -> Dict[str, Any]:
    """
    Extract comprehensive features from text for analysis and classification.
    
    Args:
        text: Input text to analyze
    
    Returns:
        Dictionary with extracted features
    
    Raises:
        TextProcessingError: If feature extraction fails
    """
    try:
        if not text:
            return {}
        
        # Get basic statistics
        stats = calculate_text_statistics(text)
        
        # Extract structured data
        structured_data = extract_numbers_and_dates(text)
        
        features = {
            # Basic metrics
            'character_count': stats.character_count,
            'word_count': stats.word_count,
            'sentence_count': stats.sentence_count,
            'paragraph_count': stats.paragraph_count,
            
            # Ratios
            'alpha_ratio': stats.alpha_ratio,
            'digit_ratio': stats.digit_ratio,
            'punct_ratio': stats.punct_ratio,
            'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text),
            
            # Language features
            'avg_word_length': stats.avg_word_length,
            'avg_sentence_length': stats.avg_sentence_length,
            'vocabulary_richness': stats.vocabulary_richness,
            'language': stats.detected_language,
            
            # Content indicators
            'has_mathematical': stats.has_mathematical,
            'has_tables': stats.has_tables,
            'has_lists': stats.has_lists,
            'has_urls': len(structured_data['urls']) > 0,
            'has_emails': len(structured_data['emails']) > 0,
            'has_phone_numbers': len(structured_data['phone_numbers']) > 0,
            'has_dates': len(structured_data['dates']) > 0,
            'has_currency': len(structured_data['currency']) > 0,
            
            # Quality metrics
            'confidence_score': stats.confidence_score,
            'readability_score': stats.readability_score,
            'quality_category': stats.quality_category,
            
            # Structure features
            'line_count': stats.line_count,
            'avg_line_length': stats.character_count / stats.line_count if stats.line_count > 0 else 0,
            'has_short_lines': any(len(line) < 20 for line in text.split('\n')),
            'has_long_lines': any(len(line) > 100 for line in text.split('\n')),
            
            # Special characters
            'has_special_chars': bool(re.search(r'[^\w\s\.\,\!\?\;\:\-\(\)]', text)),
            'newline_count': text.count('\n'),
            'tab_count': text.count('\t'),
            
            # Counts from structured data
            'number_count': len(structured_data['numbers']),
            'date_count': len(structured_data['dates']),
            'url_count': len(structured_data['urls']),
            'email_count': len(structured_data['emails']),
        }
        
        return features
    
    except Exception as e:
        raise TextProcessingError(f"Feature extraction failed: {str(e)}")


# Utility functions for batch processing
def process_text_batch(texts: List[str], 
                      operations: List[str] = None,
                      **kwargs) -> List[Dict[str, Any]]:
    """
    Process multiple texts in batch with specified operations.
    
    Args:
        texts: List of input texts
        operations: List of operations to perform ('clean', 'stats', 'language', 'correct')
        **kwargs: Additional arguments for processing functions
    
    Returns:
        List of processing results for each text
    
    Raises:
        TextProcessingError: If batch processing fails
    """
    try:
        if operations is None:
            operations = ['clean', 'stats']
        
        results = []
        
        for i, text in enumerate(texts):
            try:
                result = {'original_text': text, 'index': i}
                
                processed_text = text
                
                if 'clean' in operations:
                    processed_text = clean_text(processed_text, **kwargs)
                    result['cleaned_text'] = processed_text
                
                if 'correct' in operations:
                    language = kwargs.get('language', 'en')
                    processed_text = correct_common_ocr_errors(processed_text, language)
                    result['corrected_text'] = processed_text
                
                if 'stats' in operations:
                    stats = calculate_text_statistics(processed_text)
                    result['statistics'] = stats
                
                if 'language' in operations:
                    language = detect_language(processed_text)
                    result['detected_language'] = language
                
                if 'features' in operations:
                    features = extract_text_features(processed_text)
                    result['features'] = features
                
                if 'structured' in operations:
                    structured = extract_numbers_and_dates(processed_text)
                    result['structured_data'] = structured
                
                result['processed_text'] = processed_text
                result['success'] = True
                
            except Exception as e:
                result['error'] = str(e)
                result['success'] = False
                logger.warning(f"Failed to process text {i}: {str(e)}")
            
            results.append(result)
        
        logger.debug(f"Batch processed {len(texts)} texts with {len(operations)} operations")
        return results
    
    except Exception as e:
        raise TextProcessingError(f"Batch text processing failed: {str(e)}")


def validate_text_quality(text: str, 
                         min_confidence: float = 0.6,
                         min_word_count: int = 3,
                         max_noise_ratio: float = 0.3) -> Dict[str, Any]:
    """
    Validate text quality and provide recommendations for improvement.
    
    Args:
        text: Input text to validate
        min_confidence: Minimum acceptable confidence score
        min_word_count: Minimum word count for valid text
        max_noise_ratio: Maximum acceptable noise ratio
    
    Returns:
        Dictionary with validation results and recommendations
    
    Raises:
        TextProcessingError: If validation fails
    """
    try:
        if not text:
            return {
                'is_valid': False,
                'confidence': 0.0,
                'issues': ['Empty text'],
                'recommendations': ['Provide non-empty text input']
            }
        
        # Calculate statistics
        stats = calculate_text_statistics(text)
        
        issues = []
        recommendations = []
        
        # Check word count
        if stats.word_count < min_word_count:
            issues.append(f'Insufficient word count: {stats.word_count} < {min_word_count}')
            recommendations.append('Ensure text contains meaningful content with multiple words')
        
        # Check confidence score
        if stats.confidence_score < min_confidence:
            issues.append(f'Low confidence score: {stats.confidence_score:.3f} < {min_confidence}')
            recommendations.append('Consider image preprocessing or using higher quality source')
        
        # Check noise ratio (non-alphabetic characters excluding punctuation and whitespace)
        noise_chars = sum(1 for c in text if not (c.isalnum() or c.isspace() or c in string.punctuation))
        noise_ratio = noise_chars / len(text) if text else 0
        
        if noise_ratio > max_noise_ratio:
            issues.append(f'High noise ratio: {noise_ratio:.3f} > {max_noise_ratio}')
            recommendations.append('Apply text cleaning to remove noise characters')
        
        # Check character distribution
        if stats.alpha_ratio < 0.5:
            issues.append(f'Low alphabetic ratio: {stats.alpha_ratio:.3f}')
            recommendations.append('Text should contain more alphabetic characters')
        
        # Check for very short or very long words (potential OCR errors)
        words = re.findall(r'\b\w+\b', text)
        very_short_words = sum(1 for w in words if len(w) == 1)
        very_long_words = sum(1 for w in words if len(w) > 20)
        
        if words and very_short_words / len(words) > 0.3:
            issues.append('Too many single-character words (potential OCR errors)')
            recommendations.append('Apply OCR error correction')
        
        if very_long_words > 0:
            issues.append(f'Found {very_long_words} unusually long words')
            recommendations.append('Review for OCR concatenation errors')
        
        # Check sentence structure
        if stats.sentence_count > 0 and stats.avg_sentence_length > 50:
            issues.append('Very long average sentence length')
            recommendations.append('Check for missing punctuation or sentence boundaries')
        
        # Check for repeated patterns (potential OCR artifacts)
        repeated_chars = re.findall(r'(.)\1{4,}', text)  # Same char repeated 5+ times
        if repeated_chars:
            issues.append('Found repeated character patterns')
            recommendations.append('Remove repeated character artifacts')
        
        # Overall validation
        is_valid = (
            len(issues) == 0 or 
            (stats.confidence_score >= min_confidence and stats.word_count >= min_word_count)
        )
        
        result = {
            'is_valid': is_valid,
            'confidence': stats.confidence_score,
            'word_count': stats.word_count,
            'character_count': stats.character_count,
            'quality_category': stats.quality_category,
            'noise_ratio': noise_ratio,
            'issues': issues,
            'recommendations': recommendations,
            'statistics': stats
        }
        
        logger.debug(f"Text validation: {'PASS' if is_valid else 'FAIL'} "
                    f"(confidence: {stats.confidence_score:.3f}, {len(issues)} issues)")
        
        return result
    
    except Exception as e:
        raise TextProcessingError(f"Text quality validation failed: {str(e)}")


def merge_overlapping_regions(text_regions: List[Union[Dict[str, Any], TextRegion]], 
                             overlap_threshold: float = 0.5) -> List[Union[Dict[str, Any], TextRegion]]:
    """
    Merge overlapping text regions to avoid duplicate content.
    
    Args:
        text_regions: List of text regions with bounding boxes
        overlap_threshold: Minimum overlap ratio to trigger merge
    
    Returns:
        List of merged text regions
    
    Raises:
        TextProcessingError: If merging fails
    """
    try:
        if not text_regions or len(text_regions) <= 1:
            return text_regions
        
        # Convert to consistent format
        regions = []
        for region in text_regions:
            if isinstance(region, dict):
                if 'bbox' in region and 'text' in region:
                    regions.append(region)
            elif isinstance(region, TextRegion):
                regions.append({
                    'bbox': region.bbox,
                    'text': region.text,
                    'confidence': region.confidence,
                    'language': region.language,
                    'text_type': region.text_type
                })
        
        if not regions:
            return text_regions
        
        def calculate_overlap(bbox1, bbox2):
            """Calculate overlap ratio between two bounding boxes."""
            x1, y1, w1, h1 = bbox1
            x2, y2, w2, h2 = bbox2
            
            # Calculate intersection
            left = max(x1, x2)
            top = max(y1, y2)
            right = min(x1 + w1, x2 + w2)
            bottom = min(y1 + h1, y2 + h2)
            
            if left < right and top < bottom:
                intersection_area = (right - left) * (bottom - top)
                area1 = w1 * h1
                area2 = w2 * h2
                
                # Return ratio of intersection to smaller area
                smaller_area = min(area1, area2)
                return intersection_area / smaller_area if smaller_area > 0 else 0
            
            return 0.0
        
        def merge_regions(region1, region2):
            """Merge two overlapping regions."""
            bbox1 = region1['bbox']
            bbox2 = region2['bbox']
            
            # Calculate merged bounding box
            x1, y1, w1, h1 = bbox1
            x2, y2, w2, h2 = bbox2
            
            left = min(x1, x2)
            top = min(y1, y2)
            right = max(x1 + w1, x2 + w2)
            bottom = max(y1 + h1, y2 + h2)
            
            merged_bbox = (left, top, right - left, bottom - top)
            
            # Merge text content
            text1 = region1['text'].strip()
            text2 = region2['text'].strip()
            
            # Choose longer text or combine if significantly different
            if len(text1) > len(text2) * 1.5:
                merged_text = text1
            elif len(text2) > len(text1) * 1.5:
                merged_text = text2
            else:
                # Check similarity to decide whether to combine
                similarity = calculate_text_similarity(text1, text2, method='jaccard')
                if similarity > 0.7:
                    merged_text = text1 if len(text1) >= len(text2) else text2
                else:
                    # Combine different texts
                    merged_text = f"{text1} {text2}".strip()
            
            # Merge other properties
            merged_confidence = (region1.get('confidence', 1.0) + region2.get('confidence', 1.0)) / 2
            merged_language = region1.get('language') or region2.get('language')
            merged_type = region1.get('text_type', 'paragraph')
            
            return {
                'bbox': merged_bbox,
                'text': merged_text,
                'confidence': merged_confidence,
                'language': merged_language,
                'text_type': merged_type
            }
        
        # Iteratively merge overlapping regions
        merged_regions = regions.copy()
        changed = True
        
        while changed:
            changed = False
            new_regions = []
            merged_indices = set()
            
            for i, region1 in enumerate(merged_regions):
                if i in merged_indices:
                    continue
                
                merged_with = None
                for j, region2 in enumerate(merged_regions[i + 1:], i + 1):
                    if j in merged_indices:
                        continue
                    
                    overlap = calculate_overlap(region1['bbox'], region2['bbox'])
                    if overlap >= overlap_threshold:
                        # Merge these regions
                        merged_region = merge_regions(region1, region2)
                        merged_with = merged_region
                        merged_indices.add(j)
                        changed = True
                        break
                
                if merged_with:
                    new_regions.append(merged_with)
                    merged_indices.add(i)
                else:
                    new_regions.append(region1)
                    merged_indices.add(i)
            
            merged_regions = new_regions
        
        logger.debug(f"Region merging: {len(text_regions)} -> {len(merged_regions)} regions")
        
        # Convert back to original format if needed
        if text_regions and isinstance(text_regions[0], TextRegion):
            result = []
            for region in merged_regions:
                result.append(TextRegion(
                    text=region['text'],
                    bbox=region['bbox'],
                    confidence=region['confidence'],
                    language=region.get('language'),
                    text_type=region.get('text_type', 'paragraph')
                ))
            return result
        
        return merged_regions
    
    except Exception as e:
        raise TextProcessingError(f"Region merging failed: {str(e)}")


def format_text_for_output(text: str, 
                          format_type: str = 'plain',
                          preserve_structure: bool = True,
                          add_metadata: bool = False,
                          metadata: Optional[Dict[str, Any]] = None) -> str:
    """
    Format text for different output requirements.
    
    Args:
        text: Input text to format
        format_type: Output format ('plain', 'markdown', 'html', 'json')
        preserve_structure: Preserve original structure
        add_metadata: Include metadata in output
        metadata: Additional metadata to include
    
    Returns:
        Formatted text string
    
    Raises:
        TextProcessingError: If formatting fails
    """
    try:
        if not text:
            return ""
        
        if format_type == 'plain':
            # Simple plain text with basic cleanup
            formatted = clean_text(text, preserve_structure=preserve_structure)
            
        elif format_type == 'markdown':
            # Format as Markdown with structure preservation
            formatted = text
            
            # Convert headings (lines that are all caps or followed by many spaces)
            lines = formatted.split('\n')
            markdown_lines = []
            
            for i, line in enumerate(lines):
                stripped = line.strip()
                if not stripped:
                    markdown_lines.append('')
                    continue
                
                # Check for heading patterns
                if stripped.isupper() and len(stripped) > 5 and len(stripped) < 50:
                    markdown_lines.append(f'## {stripped.title()}')
                elif i < len(lines) - 1 and not lines[i + 1].strip():
                    # Line followed by empty line might be heading
                    words = stripped.split()
                    if len(words) <= 8 and not stripped.endswith('.'):
                        markdown_lines.append(f'### {stripped}')
                    else:
                        markdown_lines.append(stripped)
                else:
                    markdown_lines.append(stripped)
            
            formatted = '\n'.join(markdown_lines)
            
        elif format_type == 'html':
            # Basic HTML formatting
            formatted = clean_text(text, preserve_structure=True)
            
            # Convert paragraphs
            paragraphs = formatted.split('\n\n')
            html_paragraphs = []
            
            for para in paragraphs:
                if para.strip():
                    # Convert line breaks within paragraphs
                    para_html = para.replace('\n', '<br>')
                    html_paragraphs.append(f'<p>{para_html}</p>')
            
            formatted = '\n'.join(html_paragraphs)
            
        elif format_type == 'json':
            import json
            
            # Structure text data as JSON
            stats = calculate_text_statistics(text)
            structured_data = extract_numbers_and_dates(text)
            
            json_data = {
                'text': text,
                'statistics': {
                    'character_count': stats.character_count,
                    'word_count': stats.word_count,
                    'sentence_count': stats.sentence_count,
                    'confidence_score': stats.confidence_score,
                    'detected_language': stats.detected_language,
                    'quality_category': stats.quality_category
                },
                'structured_data': structured_data
            }
            
            if metadata:
                json_data['metadata'] = metadata
            
            formatted = json.dumps(json_data, indent=2, ensure_ascii=False)
            
        else:
            raise ValueError(f"Unknown format type: {format_type}")
        
        # Add metadata header if requested (for non-JSON formats)
        if add_metadata and format_type != 'json' and metadata:
            stats = calculate_text_statistics(text)
            header_lines = [
                f"Text Statistics:",
                f"- Characters: {stats.character_count}",
                f"- Words: {stats.word_count}",
                f"- Confidence: {stats.confidence_score:.3f}",
                f"- Language: {stats.detected_language or 'Unknown'}",
                f"- Quality: {stats.quality_category}",
                ""
            ]
            
            if metadata:
                header_lines.extend([f"- {k}: {v}" for k, v in metadata.items()])
                header_lines.append("")
            
            header = '\n'.join(header_lines)
            formatted = header + formatted
        
        return formatted
    
    except Exception as e:
        raise TextProcessingError(f"Text formatting failed: {str(e)}")


# Main utility function for comprehensive text processing
def process_ocr_text(text: str,
                    language: Optional[str] = None,
                    clean: bool = True,
                    correct_errors: bool = True,
                    validate: bool = True,
                    extract_features: bool = False) -> Dict[str, Any]:
    """
    Comprehensive OCR text processing pipeline.
    
    Args:
        text: Input OCR text
        language: Target language (auto-detect if None)
        clean: Apply text cleaning
        correct_errors: Apply error correction
        validate: Validate text quality
        extract_features: Extract text features
    
    Returns:
        Dictionary with comprehensive processing results
    
    Raises:
        TextProcessingError: If processing fails
    """
    try:
        if not text:
            return {'error': 'Empty input text'}
        
        result = {
            'original_text': text,
            'processing_steps': [],
            'success': True
        }
        
        processed_text = text
        
        # Language detection
        if not language:
            detected_lang = detect_language(processed_text)
            result['detected_language'] = detected_lang
            language = detected_lang or 'en'
            result['processing_steps'].append('language_detection')
        else:
            result['detected_language'] = language
        
        # Text cleaning
        if clean:
            processed_text = clean_text(
                processed_text,
                remove_noise=True,
                normalize_whitespace=True,
                normalize_unicode=True,
                preserve_structure=True
            )
            result['cleaned_text'] = processed_text
            result['processing_steps'].append('cleaning')
        
        # Error correction
        if correct_errors:
            corrected_text = correct_common_ocr_errors(processed_text, language)
            result['corrected_text'] = corrected_text
            processed_text = corrected_text
            result['processing_steps'].append('error_correction')
        
        # Final processed text
        result['processed_text'] = processed_text
        
        # Statistics calculation
        stats = calculate_text_statistics(processed_text)
        result['statistics'] = stats
        result['processing_steps'].append('statistics')
        
        # Quality validation
        if validate:
            validation = validate_text_quality(processed_text)
            result['validation'] = validation
            result['processing_steps'].append('validation')
        
        # Feature extraction
        if extract_features:
            features = extract_text_features(processed_text)
            result['features'] = features
            result['processing_steps'].append('feature_extraction')
        
        # Structured data extraction
        structured_data = extract_numbers_and_dates(processed_text)
        result['structured_data'] = structured_data
        result['processing_steps'].append('structured_extraction')
        
        # Processing summary
        result['summary'] = {
            'original_length': len(text),
            'processed_length': len(processed_text),
            'confidence_score': stats.confidence_score,
            'quality_category': stats.quality_category,
            'word_count': stats.word_count,
            'language': language,
            'processing_steps_count': len(result['processing_steps'])
        }
        
        logger.info(f"OCR text processing completed: {len(result['processing_steps'])} steps, "
                   f"quality: {stats.quality_category}, confidence: {stats.confidence_score:.3f}")
        
        return result
    
    except Exception as e:
        logger.error(f"OCR text processing failed: {str(e)}")
        return {
            'original_text': text,
            'error': str(e),
            'success': False
        }