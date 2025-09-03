
try:
    from spellchecker import SpellChecker as PySpellChecker
    SPELLCHECKER_AVAILABLE = True
except ImportError:
    SPELLCHECKER_AVAILABLE = False
    print("Warning: pyspellchecker not installed. Spell checking disabled.")

import re
from typing import Dict, List, Any

class TextCorrector:
    """Text correction using spell checking and common OCR error patterns"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.language = config.get("language", "en")
        self.domain_vocabulary = set(config.get("domain_vocabulary", []))
        
        # Initialize spell checker if available
        self.spell = None
        if SPELLCHECKER_AVAILABLE:
            try:
                self.spell = PySpellChecker(language=self.language)
                # Add domain vocabulary
                if self.domain_vocabulary:
                    self.spell.word_frequency.load_words(self.domain_vocabulary)
            except Exception as e:
                print(f"Warning: Could not initialize spell checker: {e}")
                self.spell = None
            
        # Common OCR error patterns
        self.ocr_corrections = {
            'rn': 'm',
            'cl': 'd',
            'ﬁ': 'fi',
            'ﬂ': 'fl',
            ''': "'",
            ''': "'",
            '"': '"',
            '"': '"',
        }
        
        # Character-level corrections (context-sensitive)
        self.char_corrections = {
            '0': 'o',  # Only in word context
            '1': 'l',  # Only in word context
            '5': 's',  # Only in word context
            '8': 'B',  # Only at start of word
        }
        
    def correct_text(self, text: str) -> str:
        """Apply text corrections"""
        if not text.strip():
            return text
            
        # Basic cleanup
        corrected = self._basic_cleanup(text)
        
        # OCR-specific corrections
        corrected = self._fix_ocr_errors(corrected)
        
        # Spell checking (if available)
        if self.spell:
            corrected = self._spell_check(corrected)
        
        return corrected
        
    def _basic_cleanup(self, text: str) -> str:
        """Basic text cleanup"""
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common punctuation issues
        text = re.sub(r'\s+([,.!?;:])', r'\1', text)
        text = re.sub(r'([,.!?;:])\s*([A-Z])', r'\1 \2', text)
        
        return text.strip()
        
    def _fix_ocr_errors(self, text: str) -> str:
        """Fix common OCR errors"""
        # Replace common OCR mistakes
        for error, correction in self.ocr_corrections.items():
            text = text.replace(error, correction)
            
        # Context-aware character corrections
        words = text.split()
        corrected_words = []
        
        for word in words:
            corrected_word = word
            
            # Apply character-level corrections with context
            for char, replacement in self.char_corrections.items():
                if char in word:
                    if char == '8' and word.startswith(char):
                        # Replace 8 with B only at start of word
                        corrected_word = replacement + corrected_word[1:]
                    elif char in ['0', '1', '5'] and word.isalnum() and len(word) > 1:
                        # Replace in word context only
                        corrected_word = corrected_word.replace(char, replacement)
                        
            corrected_words.append(corrected_word)
            
        return ' '.join(corrected_words)
        
    def _spell_check(self, text: str) -> str:
        """Apply spell checking"""
        if not self.spell:
            return text
            
        words = text.split()
        corrected_words = []
        
        for word in words:
            # Skip words with numbers, symbols, or very short words
            clean_word = re.sub(r'[^\w]', '', word)
            if not clean_word.isalpha() or len(clean_word) < 3:
                corrected_words.append(word)
                continue
                
            # Check if word is in domain vocabulary
            if clean_word.lower() in self.domain_vocabulary:
                corrected_words.append(word)
                continue
                
            # Spell check
            try:
                if clean_word.lower() in self.spell:
                    corrected_words.append(word)
                else:
                    # Try to correct
                    correction = self.spell.correction(clean_word.lower())
                    if correction and correction != clean_word.lower():
                        # Preserve original case and punctuation
                        if clean_word.isupper():
                            correction = correction.upper()
                        elif clean_word[0].isupper():
                            correction = correction.capitalize()
                        
                        # Replace the clean word part while preserving punctuation
                        corrected_word = word.replace(clean_word, correction)
                        corrected_words.append(corrected_word)
                    else:
                        corrected_words.append(word)
            except Exception:
                # If spell checking fails for any reason, keep original
                corrected_words.append(word)
                    
        return ' '.join(corrected_words)

