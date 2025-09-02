# src/postprocessing/text_corrector.py

import re
from typing import List, Dict, Any, Tuple
from spellchecker import SpellChecker
import string
from collections import Counter

class TextCorrector:
    """Advanced text correction for OCR results"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.language = config.get("language", "en")
        self.spell_checker = SpellChecker(language=self.language)
        
        # Common OCR error patterns
        self.ocr_corrections = {
            # Common character confusions
            r'\b0\b': 'O',  # Zero to letter O
            r'\b1\b': 'I',  # One to letter I
            r'\b5\b': 'S',  # Five to letter S
            r'rn': 'm',     # rn to m
            r'cl': 'd',     # cl to d
            r'vv': 'w',     # vv to w
            r'll': 'll',    # ll patterns
            r'\\': '/',     # backslash to forward slash
            
            # Word boundary corrections
            r'\btbe\b': 'the',
            r'\btbis\b': 'this',
            r'\band\b': 'and',
            r'\bwith\b': 'with',
            r'\bfor\b': 'for',
            r'\byou\b': 'you',
            r'\bare\b': 'are',
        }
        
        # Load domain-specific vocabulary if provided
        self.domain_vocabulary = set(config.get("domain_vocabulary", []))
        
    def correct_text(self, text: str, confidence_threshold: float = 0.7) -> str:
        """Main text correction pipeline"""
        if not text.strip():
            return text
            
        corrected = text
        
        # Step 1: Basic OCR error corrections
        corrected = self._apply_ocr_corrections(corrected)
        
        # Step 2: Spell checking
        corrected = self._spell_check_text(corrected, confidence_threshold)
        
        # Step 3: Grammar and context corrections
        corrected = self._apply_context_corrections(corrected)
        
        # Step 4: Format cleaning
        corrected = self._clean_formatting(corrected)
        
        return corrected
        
    def _apply_ocr_corrections(self, text: str) -> str:
        """Apply common OCR error corrections"""
        corrected = text
        
        for pattern, replacement in self.ocr_corrections.items():
            corrected = re.sub(pattern, replacement, corrected, flags=re.IGNORECASE)
            
        return corrected
        
    def _spell_check_text(self, text: str, confidence_threshold: float) -> str:
        """Apply spell checking with confidence filtering"""
        words = text.split()
        corrected_words = []
        
        for word in words:
            # Skip words with numbers or special characters
            if any(char.isdigit() or char in string.punctuation for char in word):
                corrected_words.append(word)
                continue
                
            clean_word = word.strip(string.punctuation)
            
            # Skip if word is in domain vocabulary
            if clean_word.lower() in self.domain_vocabulary:
                corrected_words.append(word)
                continue
                
            # Check if word needs correction
            if clean_word.lower() not in self.spell_checker:
                # Get suggestions
                suggestions = self.spell_checker.candidates(clean_word.lower())
                
                if suggestions:
                    # Choose best suggestion based on edit distance
                    best_suggestion = min(suggestions, 
                                        key=lambda x: self._edit_distance(clean_word.lower(), x))
                    
                    # Only correct if confidence is high enough
                    if self._correction_confidence(clean_word, best_suggestion) > confidence_threshold:
                        # Preserve original capitalization
                        corrected_word = self._preserve_capitalization(word, clean_word, best_suggestion)
                        corrected_words.append(corrected_word)
                    else:
                        corrected_words.append(word)
                else:
                    corrected_words.append(word)
            else:
                corrected_words.append(word)
                
        return " ".join(corrected_words)
        
    def _edit_distance(self, s1: str, s2: str) -> int:
        """Calculate edit distance between two strings"""
        if len(s1) < len(s2):
            return self._edit_distance(s2, s1)
            
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
        
    def _correction_confidence(self, original: str, corrected: str) -> float:
        """Calculate confidence score for correction"""
        if original.lower() == corrected.lower():
            return 1.0
            
        # Calculate similarity based on edit distance
        max_len = max(len(original), len(corrected))
        edit_dist = self._edit_distance(original.lower(), corrected.lower())
        similarity = 1.0 - (edit_dist / max_len)
        
        # Boost confidence for common corrections
        if len(original) == len(corrected):
            similarity += 0.2
            
        return min(1.0, similarity)
        
    def _preserve_capitalization(self, original_word: str, clean_original: str, correction: str) -> str:
        """Preserve original capitalization pattern"""
        if clean_original.isupper():
            result = correction.upper()
        elif clean_original.istitle():
            result = correction.title()
        elif clean_original.islower():
            result = correction.lower()
        else:
            # Mixed case - try to preserve pattern
            result = ""
            for i, char in enumerate(correction):
                if i < len(clean_original):
                    if clean_original[i].isupper():
                        result += char.upper()
                    else:
                        result += char.lower()
                else:
                    result += char.lower()
                    
        # Add back punctuation
        prefix = original_word[:len(original_word) - len(clean_original)]
        suffix = original_word[len(prefix) + len(clean_original):]
        
        return prefix + result + suffix
        
    def _apply_context_corrections(self, text: str) -> str:
        """Apply context-aware corrections"""
        # Common word sequence corrections
        context_corrections = {
            r'\bof\s+the\s+the\b': 'of the',
            r'\bthe\s+the\b': 'the',
            r'\band\s+and\b': 'and',
            r'\ba\s+a\b': 'a',
            r'\bin\s+in\b': 'in',
            r'\bto\s+to\b': 'to',
        }
        
        corrected = text
        for pattern, replacement in context_corrections.items():
            corrected = re.sub(pattern, replacement, corrected, flags=re.IGNORECASE)
            
        return corrected
        
    def _clean_formatting(self, text: str) -> str:
        """Clean up formatting issues"""
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', text)
        
        # Fix punctuation spacing
        cleaned = re.sub(r'\s+([,.!?;:])', r'\1', cleaned)
        cleaned = re.sub(r'([,.!?;:])\s*([a-zA-Z])', r'\1 \2', cleaned)
        
        # Fix quotes
        cleaned = re.sub(r'\s*"\s*', '"', cleaned)
        cleaned = re.sub(r'\s*\'\s*', "'", cleaned)
        
        return cleaned.strip()