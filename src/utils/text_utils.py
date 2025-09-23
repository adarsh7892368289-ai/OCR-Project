# Enhanced src/utils/text_utils.py - Additional functions for TrOCR

import re
import string
from typing import List, Dict, Set, Tuple
from collections import Counter
import unicodedata

class TextUtils:
    """Enhanced utility functions for text processing"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Enhanced text cleaning with OCR-specific improvements"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        cleaned = re.sub(r'\s+', ' ', text)
        
        # Remove non-printable characters but keep newlines and tabs
        cleaned = ''.join(char for char in cleaned if char.isprintable() or char in '\n\t')
        
        # Fix common OCR errors
        cleaned = TextUtils.fix_common_ocr_errors(cleaned)
        
        return cleaned.strip()
    
    @staticmethod
    def extract_words(text: str) -> List[str]:
        """Extract words from text"""
        # Split by whitespace and punctuation
        words = re.findall(r'\b\w+\b', text.lower())
        return words
    
    @staticmethod
    def calculate_text_metrics(text: str) -> Dict[str, int]:
        """Calculate comprehensive text metrics"""
        if not text:
            return {
                "character_count": 0,
                "word_count": 0,
                "line_count": 0,
                "paragraph_count": 0,
                "unique_words": 0,
                "average_word_length": 0,
                "sentence_count": 0,
                "alphanumeric_ratio": 0,
                "punctuation_count": 0
            }
        
        words = TextUtils.extract_words(text)
        sentences = TextUtils.split_sentences(text)
        
        # Calculate alphanumeric ratio
        alnum_chars = sum(1 for c in text if c.isalnum())
        total_chars = len(text)
        alnum_ratio = alnum_chars / total_chars if total_chars > 0 else 0
        
        # Count punctuation
        punct_count = sum(1 for c in text if c in string.punctuation)
        
        return {
            "character_count": len(text),
            "word_count": len(words),
            "line_count": text.count('\n') + 1,
            "paragraph_count": len([p for p in text.split('\n\n') if p.strip()]),
            "unique_words": len(set(words)),
            "average_word_length": sum(len(word) for word in words) / len(words) if words else 0,
            "sentence_count": len(sentences),
            "alphanumeric_ratio": alnum_ratio,
            "punctuation_count": punct_count
        }
    
    @staticmethod
    def detect_language(text: str) -> str:
        """Enhanced language detection"""
        if not text.strip():
            return "unknown"
        
        # Count different character types
        latin_chars = sum(1 for c in text if c.isascii() and c.isalpha())
        cyrillic_chars = sum(1 for c in text if '\u0400' <= c <= '\u04FF')
        cjk_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        arabic_chars = sum(1 for c in text if '\u0600' <= c <= '\u06FF')
        total_alpha = sum(1 for c in text if c.isalpha())
        
        if total_alpha == 0:
            return "unknown"
        
        # Determine dominant script
        latin_ratio = latin_chars / total_alpha
        cyrillic_ratio = cyrillic_chars / total_alpha
        cjk_ratio = cjk_chars / total_alpha
        arabic_ratio = arabic_chars / total_alpha
        
        if cjk_ratio > 0.5:
            return "zh"  # Chinese/Japanese/Korean
        elif arabic_ratio > 0.5:
            return "ar"  # Arabic
        elif cyrillic_ratio > 0.5:
            return "ru"  # Russian/Cyrillic
        elif latin_ratio > 0.8:
            return "en"  # English or other Latin script
        else:
            return "mixed"
    
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
    
    # NEW FUNCTIONS FOR TROCR AND OCR ENHANCEMENT
    
    @staticmethod
    def post_process_text(text: str, language: str = "en") -> str:
        """Enhanced post-processing for OCR text"""
        if not text:
            return ""
        
        # Basic cleaning
        text = TextUtils.clean_text(text)
        
        # Fix common OCR errors
        text = TextUtils.fix_common_ocr_errors(text)
        
        # Language-specific corrections
        if language == "en":
            text = TextUtils.fix_english_ocr_errors(text)
        
        # Normalize whitespace
        text = TextUtils.normalize_whitespace(text)
        
        return text
    
    @staticmethod
    def calculate_text_confidence(text: str, engine_confidence: float = 0.0) -> float:
        """Calculate enhanced confidence score based on text characteristics"""
        if not text.strip():
            return 0.0
        
        base_confidence = engine_confidence
        
        # Quality indicators
        quality_factors = []
        
        # 1. Word formation quality
        words = text.split()
        if words:
            valid_words = sum(1 for word in words if word.isalpha() and len(word) > 1)
            word_quality = valid_words / len(words)
            quality_factors.append(word_quality)
        
        # 2. Character distribution
        if len(text) > 0:
            alpha_ratio = sum(1 for c in text if c.isalpha()) / len(text)
            quality_factors.append(alpha_ratio)
        
        # 3. Reasonable punctuation
        punct_ratio = sum(1 for c in text if c in '.,!?;:') / len(text) if len(text) > 0 else 0
        # Good punctuation ratio is between 0.02 and 0.15
        punct_quality = 1.0 - abs(punct_ratio - 0.08) / 0.08
        punct_quality = max(0, min(1, punct_quality))
        quality_factors.append(punct_quality)
        
        # 4. Absence of suspicious character sequences
        suspicious_patterns = ['|||', '___', '...', '^^^', '~~~']
        has_suspicious = any(pattern in text for pattern in suspicious_patterns)
        suspicious_quality = 0.5 if has_suspicious else 1.0
        quality_factors.append(suspicious_quality)
        
        # Calculate combined quality score
        avg_quality = sum(quality_factors) / len(quality_factors) if quality_factors else 0.5
        
        # Combine with engine confidence
        if base_confidence > 0:
            final_confidence = (base_confidence + avg_quality) / 2
        else:
            final_confidence = avg_quality
        
        return min(1.0, max(0.0, final_confidence))
    
    @staticmethod
    def fix_common_ocr_errors(text: str) -> str:
        """Fix common OCR recognition errors"""
        if not text:
            return text
        
        # Common character substitutions
        corrections = {
            # Number/letter confusion
            'O': '0',  # Only in numeric contexts
            'l': '1',  # Only in numeric contexts
            'I': '1',  # Only in numeric contexts
            'S': '5',  # Only in numeric contexts
            'B': '8',  # Only in numeric contexts
            
            # Common misreadings
            'rn': 'm',  # rn often misread as m
            'ii': 'll',  # ii often misread as ll
            'cl': 'd',   # cl often misread as d
            'nn': 'n',   # double n sometimes incorrect
        }
        
        # Apply corrections contextually
        corrected = text
        
        # Fix obvious numeric contexts
        corrected = re.sub(r'\bO(\d)', r'0\1', corrected)  # O followed by digit
        corrected = re.sub(r'(\d)O\b', r'\10', corrected)  # Digit followed by O
        corrected = re.sub(r'\bI(\d)', r'1\1', corrected)  # I followed by digit
        corrected = re.sub(r'(\d)I\b', r'\11', corrected)  # Digit followed by I
        
        # Fix spacing issues
        corrected = re.sub(r'([a-z])([A-Z])', r'\1 \2', corrected)  # Add space between lowercase and uppercase
        corrected = re.sub(r'([.!?])([A-Z])', r'\1 \2', corrected)  # Add space after sentence punctuation
        
        return corrected
    
    @staticmethod
    def fix_english_ocr_errors(text: str) -> str:
        """Fix English-specific OCR errors"""
        if not text:
            return text
        
        # Common English word corrections
        word_corrections = {
            'teh': 'the',
            'adn': 'and',
            'tha': 'that',
            'wth': 'with',
            'fro': 'for',
            'you': 'you',
            'are': 'are',
            'ont': 'not',
            'can': 'can',
            'wil': 'will',
            'hav': 'have',
            'ths': 'this',
            'was': 'was',
            'but': 'but',
            'had': 'had',
            'his': 'his',
            'her': 'her',
            'she': 'she',
            'the': 'the'
        }
        
        # Apply word-level corrections
        words = text.split()
        corrected_words = []
        
        for word in words:
            # Remove punctuation for correction check
            clean_word = word.strip('.,!?;:()[]{}"\'-').lower()
            if clean_word in word_corrections:
                # Preserve original case and punctuation
                corrected = word.replace(clean_word, word_corrections[clean_word])
                corrected_words.append(corrected)
            else:
                corrected_words.append(word)
        
        return ' '.join(corrected_words)
    
    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """Normalize whitespace in text"""
        if not text:
            return text
        
        # Replace multiple spaces with single space
        text = re.sub(r' +', ' ', text)
        
        # Replace multiple newlines with maximum of two
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove trailing spaces from lines
        text = '\n'.join(line.rstrip() for line in text.split('\n'))
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    @staticmethod
    def split_sentences(text: str) -> List[str]:
        """Split text into sentences"""
        if not text:
            return []
        
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    @staticmethod
    def split_paragraphs(text: str) -> List[str]:
        """Split text into paragraphs"""
        if not text:
            return []
        
        paragraphs = text.split('\n\n')
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        return paragraphs
    
    @staticmethod
    def extract_structured_data(text: str) -> Dict[str, List[str]]:
        """Extract structured data from text"""
        return {
            'emails': TextUtils.extract_emails(text),
            'phone_numbers': TextUtils.extract_phone_numbers(text),
            'dates': TextUtils.extract_dates(text),
            'numbers': TextUtils.extract_numbers(text),
            'urls': TextUtils.extract_urls(text),
            'addresses': TextUtils.extract_addresses(text)
        }
    
    @staticmethod
    def extract_urls(text: str) -> List[str]:
        """Extract URLs from text"""
        url_pattern = r'https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:\w*))?)?'
        return re.findall(url_pattern, text, re.IGNORECASE)
    
    @staticmethod
    def extract_addresses(text: str) -> List[str]:
        """Extract potential addresses from text (basic implementation)"""
        # This is a simplified address extraction
        # In production, you'd use more sophisticated NLP
        
        address_patterns = [
            r'\d+\s+\w+\s+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd)',
            r'\d+\s+\w+\s+\w+\s+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd)'
        ]
        
        addresses = []
        for pattern in address_patterns:
            addresses.extend(re.findall(pattern, text, re.IGNORECASE))
        
        return addresses
    
    @staticmethod
    def calculate_reading_level(text: str) -> Dict[str, float]:
        """Calculate text reading level metrics"""
        if not text:
            return {'flesch_score': 0, 'reading_level': 'unknown'}
        
        words = TextUtils.extract_words(text)
        sentences = TextUtils.split_sentences(text)
        
        if not words or not sentences:
            return {'flesch_score': 0, 'reading_level': 'unknown'}
        
        # Count syllables (simplified)
        def count_syllables(word):
            word = word.lower()
            vowels = 'aeiouy'
            syllable_count = 0
            previous_char_was_vowel = False
            
            for char in word:
                if char in vowels and not previous_char_was_vowel:
                    syllable_count += 1
                previous_char_was_vowel = char in vowels
            
            # Adjust for silent 'e'
            if word.endswith('e'):
                syllable_count -= 1
            
            # Ensure at least one syllable
            return max(1, syllable_count)
        
        total_syllables = sum(count_syllables(word) for word in words)
        
        # Flesch Reading Ease Score
        avg_sentence_length = len(words) / len(sentences)
        avg_syllables_per_word = total_syllables / len(words)
        
        flesch_score = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        
        # Determine reading level
        if flesch_score >= 90:
            level = 'very_easy'
        elif flesch_score >= 80:
            level = 'easy'
        elif flesch_score >= 70:
            level = 'fairly_easy'
        elif flesch_score >= 60:
            level = 'standard'
        elif flesch_score >= 50:
            level = 'fairly_difficult'
        elif flesch_score >= 30:
            level = 'difficult'
        else:
            level = 'very_difficult'
        
        return {
            'flesch_score': flesch_score,
            'reading_level': level,
            'avg_sentence_length': avg_sentence_length,
            'avg_syllables_per_word': avg_syllables_per_word
        }
    
    @staticmethod
    def validate_text_quality(text: str, min_confidence: float = 0.7) -> Tuple[bool, Dict[str, any]]:
        """Validate text quality for OCR results"""
        if not text:
            return False, {'reason': 'empty_text'}
        
        metrics = TextUtils.calculate_text_metrics(text)
        confidence = TextUtils.calculate_text_confidence(text)
        
        issues = []
        
        # Check various quality indicators
        if confidence < min_confidence:
            issues.append('low_confidence')
        
        if metrics['alphanumeric_ratio'] < 0.5:
            issues.append('too_many_special_characters')
        
        if metrics['average_word_length'] < 2:
            issues.append('words_too_short')
        
        if metrics['average_word_length'] > 15:
            issues.append('words_too_long')
        
        # Check for suspicious patterns
        suspicious_patterns = ['|||', '___', r'\d{10,}', r'[^\w\s]{5,}']
        for pattern in suspicious_patterns:
            if re.search(pattern, text):
                issues.append(f'suspicious_pattern_{pattern}')
        
        is_valid = len(issues) == 0
        
        return is_valid, {
            'confidence': confidence,
            'issues': issues,
            'metrics': metrics
        }