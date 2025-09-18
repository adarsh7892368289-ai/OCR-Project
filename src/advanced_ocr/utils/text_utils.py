"""
Advanced OCR System - Text Processing Utilities
Modern text processing and cleaning utilities for OCR results.
"""

import re
import unicodedata
from typing import List, Dict, Tuple, Optional, Set, Any
from collections import Counter, defaultdict
import difflib
import logging
from dataclasses import dataclass

from ..config import OCRConfig
from ..results import OCRResult, Word, Line, Paragraph, ConfidenceMetrics


@dataclass
class TextQualityMetrics:
    """Comprehensive text quality assessment metrics."""
    
    meaningfulness_score: float = 0.0
    language_consistency: float = 0.0
    character_distribution: float = 0.0
    word_formation: float = 0.0
    punctuation_balance: float = 0.0
    overall_quality: float = 0.0
    detected_issues: List[str] = None
    
    def __post_init__(self):
        if self.detected_issues is None:
            self.detected_issues = []


class TextCleaner:
    """Advanced OCR artifact removal and text cleaning."""
    
    def __init__(self, config: Optional[OCRConfig] = None):
        self.config = config or OCRConfig()
        self.logger = logging.getLogger(__name__)
        
        # Common OCR error patterns
        self.ocr_patterns = {
            # Character substitution patterns (OCR confusion)
            'digit_letter': {
                '0': 'O', '1': 'I', '5': 'S', '8': 'B', '6': 'G',
                'O': '0', 'I': '1', 'S': '5', 'B': '8', 'G': '6'
            },
            
            # Common OCR artifacts
            'artifacts': [
                r'[|]{2,}',  # Multiple pipe characters
                r'[-_]{4,}',  # Long dashes/underscores
                r'[.]{4,}',   # Multiple dots
                r'[\s]{3,}',  # Excessive whitespace
            ],
            
            # Invalid character combinations
            'invalid_combinations': [
                r'[A-Za-z]{1}[0-9]{1}[A-Za-z]{1}',  # Letter-digit-letter
                r'[0-9]{1}[A-Za-z]{1}[0-9]{1}',     # Digit-letter-digit
            ]
        }
        
        # Language-specific character sets
        self.valid_chars = {
            'latin': set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'),
            'digits': set('0123456789'),
            'punctuation': set('.,;:!?()[]{}"\'-'),
            'whitespace': set(' \t\n\r'),
            'common_symbols': set('$%&*+/<=>@\\^_`|~')
        }
    
    def clean_text(self, text: str, aggressive: bool = False) -> str:
        """
        Clean OCR text with configurable aggressiveness.
        
        Args:
            text: Raw OCR text to clean
            aggressive: Apply more aggressive cleaning
            
        Returns:
            Cleaned text string
        """
        if not text or not text.strip():
            return ""
        
        cleaned = text
        
        try:
            # Basic cleaning
            cleaned = self._remove_artifacts(cleaned)
            cleaned = self._normalize_whitespace(cleaned)
            cleaned = self._fix_common_ocr_errors(cleaned)
            
            if aggressive:
                cleaned = self._aggressive_cleaning(cleaned)
                
            # Final normalization
            cleaned = self._final_normalization(cleaned)
            
            self.logger.debug(f"Text cleaned: '{text[:50]}...' -> '{cleaned[:50]}...'")
            
        except Exception as e:
            self.logger.warning(f"Text cleaning failed: {e}")
            return text  # Return original on failure
            
        return cleaned
    
    def _remove_artifacts(self, text: str) -> str:
        """Remove common OCR artifacts."""
        for pattern in self.ocr_patterns['artifacts']:
            text = re.sub(pattern, ' ', text)
        return text
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace characters."""
        # Replace various whitespace chars with standard space
        text = re.sub(r'[\t\r\n\f\v]+', ' ', text)
        # Collapse multiple spaces
        text = re.sub(r' {2,}', ' ', text)
        return text.strip()
    
    def _fix_common_ocr_errors(self, text: str) -> str:
        """Fix common OCR character recognition errors."""
        # Context-aware character substitution
        fixed = text
        
        # Fix obvious digit/letter confusions in word contexts
        words = fixed.split()
        corrected_words = []
        
        for word in words:
            if self._is_likely_word(word):
                corrected_word = self._fix_word_ocr_errors(word)
                corrected_words.append(corrected_word)
            else:
                corrected_words.append(word)
        
        return ' '.join(corrected_words)
    
    def _fix_word_ocr_errors(self, word: str) -> str:
        """Fix OCR errors within a word context."""
        if len(word) < 2:
            return word
            
        # Common word-level fixes
        fixes = {
            # Common OCR substitutions in word context
            'rn': 'm',  # 'rn' -> 'm' (like 'modern' -> 'modem')
            'cl': 'd',  # 'cl' -> 'd' in some contexts
        }
        
        corrected = word
        for error, fix in fixes.items():
            if error in corrected:
                # Only apply if it results in more letter-like word
                candidate = corrected.replace(error, fix)
                if self._is_more_word_like(candidate, corrected):
                    corrected = candidate
        
        return corrected
    
    def _is_likely_word(self, text: str) -> bool:
        """Check if text segment is likely a word."""
        if not text or len(text) < 2:
            return False
            
        # Should be mostly letters
        letter_ratio = sum(1 for c in text if c.isalpha()) / len(text)
        return letter_ratio > 0.6
    
    def _is_more_word_like(self, candidate: str, original: str) -> bool:
        """Check if candidate is more word-like than original."""
        candidate_letters = sum(1 for c in candidate if c.isalpha())
        original_letters = sum(1 for c in original if c.isalpha())
        
        return candidate_letters >= original_letters
    
    def _aggressive_cleaning(self, text: str) -> str:
        """Apply more aggressive cleaning."""
        # Remove isolated single characters (except I, a, A)
        words = text.split()
        filtered_words = []
        
        for word in words:
            if len(word) == 1 and word not in {'I', 'a', 'A'}:
                # Skip isolated characters unless they're meaningful
                continue
            filtered_words.append(word)
        
        return ' '.join(filtered_words)
    
    def _final_normalization(self, text: str) -> str:
        """Final text normalization."""
        # Ensure proper sentence spacing
        text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)
        
        # Clean up punctuation spacing
        text = re.sub(r'\s+([.,:;!?])', r'\1', text)
        text = re.sub(r'([.!?])\s*$', r'\1', text)
        
        return text.strip()


class UnicodeNormalizer:
    """Advanced Unicode normalization and encoding fixes."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Common encoding issues and fixes
        self.encoding_fixes = {
            'â€™': "'",     # Smart apostrophe
            'â€œ': '"',     # Smart quote open
            'â€': '"',      # Smart quote close
            'â€"': '—',     # Em dash
            'â€"': '–',     # En dash
            'Â': '',        # Non-breaking space artifact
        }
    
    def normalize(self, text: str) -> str:
        """
        Normalize Unicode text and fix encoding issues.
        
        Args:
            text: Text to normalize
            
        Returns:
            Normalized text
        """
        if not text:
            return ""
        
        try:
            # Fix common encoding artifacts
            normalized = self._fix_encoding_artifacts(text)
            
            # Unicode normalization
            normalized = unicodedata.normalize('NFKC', normalized)
            
            # Remove control characters except whitespace
            normalized = self._remove_control_characters(normalized)
            
            return normalized
            
        except Exception as e:
            self.logger.warning(f"Unicode normalization failed: {e}")
            return text
    
    def _fix_encoding_artifacts(self, text: str) -> str:
        """Fix common encoding artifacts."""
        fixed = text
        for artifact, replacement in self.encoding_fixes.items():
            fixed = fixed.replace(artifact, replacement)
        return fixed
    
    def _remove_control_characters(self, text: str) -> str:
        """Remove Unicode control characters except whitespace."""
        return ''.join(char for char in text 
                      if unicodedata.category(char)[0] != 'C' or char in ' \t\n\r')


class TextValidator:
    """Text quality scoring and meaningfulness assessment."""
    
    def __init__(self, config: Optional[OCRConfig] = None):
        self.config = config or OCRConfig()
        self.logger = logging.getLogger(__name__)
        
        # Load common words for validation (subset for basic validation)
        self.common_words = self._load_common_words()
        
        # Character frequency expectations for English
        self.expected_char_freq = {
            'e': 12.7, 't': 9.1, 'a': 8.2, 'o': 7.5, 'i': 7.0,
            'n': 6.7, 's': 6.3, 'h': 6.1, 'r': 6.0
        }
    
    def _load_common_words(self) -> Set[str]:
        """Load set of common English words."""
        # Basic set of most common English words
        common_words = {
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
        return common_words
    
    def assess_quality(self, text: str) -> TextQualityMetrics:
        """
        Comprehensive text quality assessment.
        
        Args:
            text: Text to assess
            
        Returns:
            TextQualityMetrics with detailed quality scores
        """
        if not text or not text.strip():
            return TextQualityMetrics()
        
        try:
            metrics = TextQualityMetrics()
            
            # Individual quality assessments
            metrics.meaningfulness_score = self._assess_meaningfulness(text)
            metrics.language_consistency = self._assess_language_consistency(text)
            metrics.character_distribution = self._assess_character_distribution(text)
            metrics.word_formation = self._assess_word_formation(text)
            metrics.punctuation_balance = self._assess_punctuation_balance(text)
            
            # Detect specific issues
            metrics.detected_issues = self._detect_issues(text)
            
            # Calculate overall quality (weighted average)
            weights = {
                'meaningfulness': 0.3,
                'language_consistency': 0.25,
                'character_distribution': 0.2,
                'word_formation': 0.15,
                'punctuation_balance': 0.1
            }
            
            metrics.overall_quality = (
                metrics.meaningfulness_score * weights['meaningfulness'] +
                metrics.language_consistency * weights['language_consistency'] +
                metrics.character_distribution * weights['character_distribution'] +
                metrics.word_formation * weights['word_formation'] +
                metrics.punctuation_balance * weights['punctuation_balance']
            )
            
            return metrics
            
        except Exception as e:
            self.logger.warning(f"Quality assessment failed: {e}")
            return TextQualityMetrics(overall_quality=0.5)  # Neutral score on error
    
    def _assess_meaningfulness(self, text: str) -> float:
        """Assess how meaningful the text appears."""
        words = text.lower().split()
        if not words:
            return 0.0
        
        # Calculate ratio of recognized common words
        recognized = sum(1 for word in words if word in self.common_words)
        recognition_ratio = recognized / len(words)
        
        # Penalize very short or very long words (OCR artifacts)
        avg_word_length = sum(len(word) for word in words) / len(words)
        length_score = 1.0 if 3 <= avg_word_length <= 8 else 0.7
        
        return min(recognition_ratio * length_score, 1.0)
    
    def _assess_language_consistency(self, text: str) -> float:
        """Assess language consistency."""
        if not text:
            return 0.0
        
        # Check for mixed scripts or unusual character combinations
        char_types = defaultdict(int)
        for char in text:
            if char.isalpha():
                if ord(char) < 128:  # ASCII letters
                    char_types['ascii'] += 1
                else:
                    char_types['non_ascii'] += 1
            elif char.isdigit():
                char_types['digits'] += 1
        
        total_chars = sum(char_types.values())
        if total_chars == 0:
            return 0.0
        
        # Prefer predominantly ASCII for English text
        ascii_ratio = char_types['ascii'] / total_chars
        return min(ascii_ratio * 1.2, 1.0)  # Slight bonus for ASCII
    
    def _assess_character_distribution(self, text: str) -> float:
        """Assess character frequency distribution."""
        if not text:
            return 0.0
        
        # Count character frequencies
        text_lower = text.lower()
        char_counts = Counter(c for c in text_lower if c.isalpha())
        
        if not char_counts:
            return 0.0
        
        total_chars = sum(char_counts.values())
        
        # Compare with expected English character frequencies
        score = 0.0
        for char, expected_pct in self.expected_char_freq.items():
            actual_pct = (char_counts.get(char, 0) / total_chars) * 100
            # Score based on how close actual frequency is to expected
            diff = abs(actual_pct - expected_pct)
            char_score = max(0, 1.0 - diff / expected_pct)
            score += char_score
        
        return min(score / len(self.expected_char_freq), 1.0)
    
    def _assess_word_formation(self, text: str) -> float:
        """Assess word formation quality."""
        words = text.split()
        if not words:
            return 0.0
        
        valid_words = 0
        for word in words:
            # Remove punctuation for assessment
            clean_word = re.sub(r'[^\w]', '', word)
            if self._is_well_formed_word(clean_word):
                valid_words += 1
        
        return valid_words / len(words)
    
    def _is_well_formed_word(self, word: str) -> bool:
        """Check if a word is well-formed."""
        if not word or len(word) < 1:
            return False
        
        # Should contain some vowels for longer words
        if len(word) > 3:
            vowels = sum(1 for c in word.lower() if c in 'aeiou')
            if vowels == 0:
                return False
        
        # Shouldn't have too many repeated characters
        for char in set(word):
            if word.count(char) > len(word) // 2:
                return False
        
        return True
    
    def _assess_punctuation_balance(self, text: str) -> float:
        """Assess punctuation usage balance."""
        if not text:
            return 1.0  # No punctuation issues if no text
        
        # Count punctuation
        punctuation_count = sum(1 for c in text if c in '.,;:!?()')
        char_count = len(text)
        
        if char_count == 0:
            return 1.0
        
        punct_ratio = punctuation_count / char_count
        
        # Ideal punctuation ratio is between 0.02 and 0.15
        if 0.02 <= punct_ratio <= 0.15:
            return 1.0
        elif punct_ratio < 0.02:
            return 0.8  # Too little punctuation
        else:
            return max(0.0, 1.0 - (punct_ratio - 0.15) * 2)  # Too much punctuation
    
    def _detect_issues(self, text: str) -> List[str]:
        """Detect specific text quality issues."""
        issues = []
        
        if not text or not text.strip():
            issues.append("Empty or whitespace-only text")
            return issues
        
        # Check for excessive repetition
        words = text.split()
        if len(set(words)) < len(words) * 0.5:
            issues.append("High word repetition")
        
        # Check for excessive punctuation
        punct_count = sum(1 for c in text if not c.isalnum() and c != ' ')
        if punct_count > len(text) * 0.2:
            issues.append("Excessive punctuation/symbols")
        
        # Check for unusual character patterns
        if re.search(r'[^\w\s.,;:!?()\-\'\"]{3,}', text):
            issues.append("Unusual character sequences")
        
        # Check for mixed case issues
        if re.search(r'[a-z][A-Z][a-z]', text):
            issues.append("Irregular capitalization")
        
        return issues


class TextMerger:
    """Intelligent combination of multiple OCR results."""
    
    def __init__(self, config: Optional[OCRConfig] = None):
        self.config = config or OCRConfig()
        self.logger = logging.getLogger(__name__)
        self.validator = TextValidator(config)
    
    def merge_texts(self, texts: List[str], confidences: List[float] = None) -> str:
        """
        Merge multiple OCR text results intelligently.
        
        Args:
            texts: List of OCR text results to merge
            confidences: Optional confidence scores for each text
            
        Returns:
            Best merged text result
        """
        if not texts:
            return ""
        
        if len(texts) == 1:
            return texts[0]
        
        # Filter out empty texts
        valid_texts = [(text, conf) for text, conf in 
                      zip(texts, confidences or [1.0] * len(texts))
                      if text and text.strip()]
        
        if not valid_texts:
            return ""
        
        if len(valid_texts) == 1:
            return valid_texts[0][0]
        
        try:
            # Method 1: Use highest confidence if available and reliable
            if confidences and max(confidences) > 0.8:
                best_idx = confidences.index(max(confidences))
                return texts[best_idx]
            
            # Method 2: Use quality-based selection
            best_text = self._select_by_quality(valid_texts)
            if best_text:
                return best_text
            
            # Method 3: Consensus-based merging for similar texts
            merged_text = self._consensus_merge(valid_texts)
            return merged_text
            
        except Exception as e:
            self.logger.warning(f"Text merging failed: {e}")
            # Fallback to longest text
            return max(texts, key=len)
    
    def _select_by_quality(self, texts_with_conf: List[Tuple[str, float]]) -> Optional[str]:
        """Select best text based on quality assessment."""
        quality_scores = []
        
        for text, conf in texts_with_conf:
            quality_metrics = self.validator.assess_quality(text)
            # Combine OCR confidence with text quality
            combined_score = (quality_metrics.overall_quality * 0.7 + conf * 0.3)
            quality_scores.append((text, combined_score))
        
        if quality_scores:
            best_text = max(quality_scores, key=lambda x: x[1])
            return best_text[0]
        
        return None
    
    def _consensus_merge(self, texts_with_conf: List[Tuple[str, float]]) -> str:
        """Merge texts using consensus approach."""
        texts = [text for text, _ in texts_with_conf]
        
        # For very similar texts, use character-level consensus
        if self._are_texts_similar(texts):
            return self._character_consensus(texts)
        
        # For different texts, return the best one
        return max(texts_with_conf, key=lambda x: x[1])[0]
    
    def _are_texts_similar(self, texts: List[str]) -> bool:
        """Check if texts are similar enough for consensus merging."""
        if len(texts) < 2:
            return False
        
        # Compare all pairs
        similarity_threshold = 0.7
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                similarity = difflib.SequenceMatcher(None, texts[i], texts[j]).ratio()
                if similarity < similarity_threshold:
                    return False
        
        return True
    
    def _character_consensus(self, texts: List[str]) -> str:
        """Create consensus text from similar texts."""
        if not texts:
            return ""
        
        # Use the longest text as base
        base_text = max(texts, key=len)
        
        # For each character position, use majority vote
        result = []
        for i in range(len(base_text)):
            chars_at_pos = []
            for text in texts:
                if i < len(text):
                    chars_at_pos.append(text[i])
            
            if chars_at_pos:
                # Use most common character at this position
                char_counts = Counter(chars_at_pos)
                most_common = char_counts.most_common(1)[0][0]
                result.append(most_common)
        
        return ''.join(result)
    
    def merge_ocr_results(self, results: List[OCRResult]) -> OCRResult:
        """
        Merge multiple OCRResult objects intelligently.
        
        Args:
            results: List of OCRResult objects to merge
            
        Returns:
            Merged OCRResult
        """
        if not results:
            return OCRResult()
        
        if len(results) == 1:
            return results[0]
        
        # Extract texts and confidences
        texts = [result.text for result in results if result.text]
        confidences = [result.confidence.overall for result in results if result.text]
        
        # Merge texts
        merged_text = self.merge_texts(texts, confidences)
        
        # Create merged result with best available metadata
        best_result = max(results, key=lambda r: r.confidence.overall)
        
        return OCRResult(
            text=merged_text,
            confidence=best_result.confidence,  # Use best confidence as base
            words=best_result.words,            # Use best structure
            lines=best_result.lines,
            paragraphs=best_result.paragraphs,
            processing_time=sum(r.processing_time for r in results),
            metadata={
                **best_result.metadata,
                'merged_from': len(results),
                'merge_method': 'intelligent_consensus'
            }
        )