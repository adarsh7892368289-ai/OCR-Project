"""
Enhanced Text Corrector with AI-Powered Contextual Correction
Step 5: Advanced Post-processing Implementation

Features:
- Multi-level text correction (spell, grammar, context)
- AI-powered contextual understanding
- Domain-specific correction rules
- Confidence-based correction decisions
- Performance monitoring and optimization
"""

import re
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import time
import json
from collections import defaultdict, Counter
from enum import Enum
import statistics
import difflib

try:
    import language_tool_python
    LANGUAGETOOL_AVAILABLE = True
except ImportError:
    LANGUAGETOOL_AVAILABLE = False

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForMaskedLM
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

from ..core.base_engine import OCRResult, TextRegion, BoundingBox
from ..utils.config import ConfigManager
from ..utils.logger import get_logger
from ..utils.text_utils import TextUtils

logger = get_logger(__name__)


class CorrectionType(Enum):
    """Types of text corrections"""
    SPELLING = "spelling"
    GRAMMAR = "grammar"
    PUNCTUATION = "punctuation"
    CAPITALIZATION = "capitalization"
    SPACING = "spacing"
    CONTEXT = "context"
    OCR_ARTIFACTS = "ocr_artifacts"
    FORMATTING = "formatting"


class CorrectionLevel(Enum):
    """Correction aggressiveness levels"""
    CONSERVATIVE = "conservative"  # Only high-confidence corrections
    MODERATE = "moderate"         # Balanced approach
    AGGRESSIVE = "aggressive"     # More corrections, higher risk


@dataclass
class Correction:
    """Individual text correction"""
    original_text: str
    corrected_text: str
    correction_type: CorrectionType
    confidence: float
    start_pos: int
    end_pos: int
    reason: str = ""
    context: str = ""
    
    @property
    def changed(self) -> bool:
        return self.original_text != self.corrected_text
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'original': self.original_text,
            'corrected': self.corrected_text,
            'type': self.correction_type.value,
            'confidence': self.confidence,
            'position': (self.start_pos, self.end_pos),
            'reason': self.reason,
            'context': self.context,
            'changed': self.changed
        }


@dataclass
class CorrectionResult:
    """Result of text correction process"""
    original_text: str
    corrected_text: str
    corrections: List[Correction]
    processing_time: float
    confidence: float
    statistics: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def correction_count(self) -> int:
        return len([c for c in self.corrections if c.changed])
    
    @property
    def improvement_ratio(self) -> float:
        if not self.original_text:
            return 0.0
        return self.correction_count / len(self.original_text.split())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'original_text': self.original_text,
            'corrected_text': self.corrected_text,
            'correction_count': self.correction_count,
            'improvement_ratio': self.improvement_ratio,
            'processing_time': self.processing_time,
            'confidence': self.confidence,
            'statistics': self.statistics,
            'corrections': [c.to_dict() for c in self.corrections]
        }


class EnhancedTextCorrector:
    """
    Advanced text corrector with AI-powered contextual understanding
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = ConfigManager(config_path).get_section('text_corrector', {})
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        
        # Configuration
        self.correction_level = CorrectionLevel(
            self.config.get('correction_level', 'moderate')
        )
        self.min_confidence_threshold = self.config.get('min_confidence_threshold', 0.7)
        self.enable_ai_correction = self.config.get('enable_ai_correction', True)
        self.enable_grammar_check = self.config.get('enable_grammar_check', True)
        self.enable_spell_check = self.config.get('enable_spell_check', True)
        self.enable_context_correction = self.config.get('enable_context_correction', True)
        
        # Domain-specific settings
        self.domain_rules = self.config.get('domain_rules', {})
        self.custom_dictionary = set(self.config.get('custom_dictionary', []))
        
        # Initialize components
        self.text_utils = TextUtils()
        self._initialize_correction_engines()
        self._load_correction_patterns()
        
        # Statistics
        self.stats = defaultdict(int)
        self.correction_history = []
        
        self.logger.info(f"Enhanced text corrector initialized with level: {self.correction_level.value}")
    
    def _initialize_correction_engines(self):
        """Initialize correction engines"""
        
        # Grammar checker
        self.grammar_tool = None
        if LANGUAGETOOL_AVAILABLE and self.enable_grammar_check:
            try:
                self.grammar_tool = language_tool_python.LanguageTool('en-US')
                self.logger.info("LanguageTool grammar checker initialized")
            except Exception as e:
                self.logger.warning(f"Could not initialize LanguageTool: {e}")
        
        # AI model for contextual correction
        self.ai_corrector = None
        if TRANSFORMERS_AVAILABLE and self.enable_ai_correction:
            try:
                model_name = self.config.get('ai_model', 'bert-base-uncased')
                self.ai_corrector = pipeline(
                    'fill-mask',
                    model=model_name,
                    device=0 if torch.cuda.is_available() else -1
                )
                self.logger.info(f"AI corrector initialized with model: {model_name}")
            except Exception as e:
                self.logger.warning(f"Could not initialize AI corrector: {e}")
    
    def _load_correction_patterns(self):
        """Load OCR-specific correction patterns"""
        
        # Common OCR error patterns
        self.ocr_patterns = [
            # Character substitutions
            (r'\b0(?=\w)', 'O'),      # 0 -> O in words
            (r'\bO(?=\d)', '0'),      # O -> 0 in numbers
            (r'\b1(?=\w)', 'l'),      # 1 -> l in words
            (r'\bl(?=\d)', '1'),      # l -> 1 in numbers
            (r'rn', 'm'),             # rn -> m
            (r'vv', 'w'),             # vv -> w
            (r'ii', 'll'),            # ii -> ll
            (r'\bB(?=\d)', '8'),      # B -> 8 in numbers
            (r'\b8(?=[a-zA-Z])', 'B'), # 8 -> B in words
            
            # Spacing issues
            (r'([a-z])([A-Z])', r'\1 \2'),  # Split camelCase
            (r'(\w)([.!?])(\w)', r'\1\2 \3'), # Add space after punctuation
            (r'\s+', ' '),                     # Multiple spaces to single
            
            # Common word corrections
            (r'\bteh\b', 'the'),
            (r'\band\b', 'and'),
            (r'\bwith\b', 'with'),
            (r'\bfrom\b', 'from'),
        ]
        
        # Load custom patterns from config
        custom_patterns = self.config.get('custom_patterns', [])
        for pattern, replacement in custom_patterns:
            self.ocr_patterns.append((pattern, replacement))
    
    def correct_text(
        self,
        text: str,
        context: Optional[str] = None,
        domain: Optional[str] = None
    ) -> CorrectionResult:
        """
        Perform comprehensive text correction
        
        Args:
            text: Text to correct
            context: Additional context for better corrections
            domain: Domain-specific correction rules to apply
            
        Returns:
            CorrectionResult with corrected text and details
        """
        start_time = time.time()
        
        if not text or not text.strip():
            return CorrectionResult(
                original_text=text,
                corrected_text=text,
                corrections=[],
                processing_time=time.time() - start_time,
                confidence=1.0
            )
        
        try:
            corrections = []
            current_text = text.strip()
            
            # Step 1: OCR artifact correction
            current_text, ocr_corrections = self._correct_ocr_artifacts(current_text)
            corrections.extend(ocr_corrections)
            
            # Step 2: Spacing and formatting
            current_text, spacing_corrections = self._correct_spacing(current_text)
            corrections.extend(spacing_corrections)
            
            # Step 3: Spell checking
            if self.enable_spell_check:
                current_text, spell_corrections = self._correct_spelling(current_text)
                corrections.extend(spell_corrections)
            
            # Step 4: Grammar checking
            if self.enable_grammar_check and self.grammar_tool:
                current_text, grammar_corrections = self._correct_grammar(current_text)
                corrections.extend(grammar_corrections)
            
            # Step 5: AI-powered contextual correction
            if self.enable_context_correction and self.ai_corrector:
                current_text, context_corrections = self._correct_context(
                    current_text, context
                )
                corrections.extend(context_corrections)
            
            # Step 6: Domain-specific corrections
            if domain and domain in self.domain_rules:
                current_text, domain_corrections = self._apply_domain_rules(
                    current_text, domain
                )
                corrections.extend(domain_corrections)
            
            # Step 7: Final cleanup
            current_text, cleanup_corrections = self._final_cleanup(current_text)
            corrections.extend(cleanup_corrections)
            
            processing_time = time.time() - start_time
            confidence = self._calculate_correction_confidence(corrections)
            
            # Generate statistics
            statistics_data = self._generate_correction_statistics(
                text, current_text, corrections
            )
            
            # Update global statistics
            self.stats['texts_corrected'] += 1
            self.stats['corrections_made'] += len([c for c in corrections if c.changed])
            for correction in corrections:
                self.stats[f'correction_{correction.correction_type.value}'] += 1
            
            result = CorrectionResult(
                original_text=text,
                corrected_text=current_text,
                corrections=corrections,
                processing_time=processing_time,
                confidence=confidence,
                statistics=statistics_data
            )
            
            # Store in history for analysis
            self.correction_history.append({
                'timestamp': time.time(),
                'original_length': len(text),
                'corrected_length': len(current_text),
                'correction_count': result.correction_count,
                'processing_time': processing_time,
                'confidence': confidence
            })
            
            self.logger.info(
                f"Text correction completed: {result.correction_count} corrections "
                f"in {processing_time:.3f}s"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in text correction: {e}")
            return CorrectionResult(
                original_text=text,
                corrected_text=text,
                corrections=[],
                processing_time=time.time() - start_time,
                confidence=0.0
            )
    
    def _correct_ocr_artifacts(self, text: str) -> Tuple[str, List[Correction]]:
        """Correct common OCR artifacts"""
        corrections = []
        current_text = text
        
        for pattern, replacement in self.ocr_patterns:
            matches = list(re.finditer(pattern, current_text, re.IGNORECASE))
            
            for match in reversed(matches):  # Reverse to maintain positions
                original = match.group(0)
                start, end = match.span()
                
                # Apply replacement
                if callable(replacement):
                    corrected = replacement(match)
                else:
                    corrected = re.sub(pattern, replacement, original, flags=re.IGNORECASE)
                
                if original != corrected:
                    correction = Correction(
                        original_text=original,
                        corrected_text=corrected,
                        correction_type=CorrectionType.OCR_ARTIFACTS,
                        confidence=0.9,  # High confidence for pattern-based corrections
                        start_pos=start,
                        end_pos=end,
                        reason=f"OCR artifact pattern: {pattern} -> {replacement}"
                    )
                    corrections.append(correction)
                    
                    # Apply correction to text
                    current_text = current_text[:start] + corrected + current_text[end:]
        
        return current_text, corrections
    
    def _correct_spacing(self, text: str) -> Tuple[str, List[Correction]]:
        """Correct spacing and formatting issues"""
        corrections = []
        current_text = text
        
        # Multiple spaces to single space
        multiple_spaces = re.findall(r'\s{2,}', current_text)
        current_text = re.sub(r'\s{2,}', ' ', current_text)
        
        if multiple_spaces:
            correction = Correction(
                original_text="multiple spaces",
                corrected_text="single space",
                correction_type=CorrectionType.SPACING,
                confidence=1.0,
                start_pos=0,
                end_pos=len(text),
                reason="Normalized multiple spaces to single spaces"
            )
            corrections.append(correction)
        
        # Fix missing spaces after punctuation
        punctuation_fixes = re.findall(r'([.!?])([A-Z])', current_text)
        current_text = re.sub(r'([.!?])([A-Z])', r'\1 \2', current_text)
        
        if punctuation_fixes:
            correction = Correction(
                original_text="missing spaces after punctuation",
                corrected_text="added spaces after punctuation",
                correction_type=CorrectionType.SPACING,
                confidence=0.9,
                start_pos=0,
                end_pos=len(text),
                reason="Added missing spaces after punctuation"
            )
            corrections.append(correction)
        
        # Remove spaces before punctuation
        spaces_before_punct = re.findall(r'\s+([.!?,:;])', current_text)
        current_text = re.sub(r'\s+([.!?,:;])', r'\1', current_text)
        
        if spaces_before_punct:
            correction = Correction(
                original_text="spaces before punctuation",
                corrected_text="no spaces before punctuation",
                correction_type=CorrectionType.SPACING,
                confidence=0.95,
                start_pos=0,
                end_pos=len(text),
                reason="Removed incorrect spaces before punctuation"
            )
            corrections.append(correction)
        
        return current_text.strip(), corrections
    
    def _correct_spelling(self, text: str) -> Tuple[str, List[Correction]]:
        """Perform spell checking and correction"""
        corrections = []
        current_text = text
        
        # Simple dictionary-based spell checking
        words = re.findall(r'\b\w+\b', current_text)
        
        for word in words:
            if (len(word) > 2 and 
                word.lower() not in self.custom_dictionary and
                not self.text_utils.is_valid_word(word)):
                
                # Get spell suggestions
                suggestions = self.text_utils.get_spell_suggestions(word)
                
                if suggestions:
                    best_suggestion = suggestions[0]
                    confidence = self._calculate_spelling_confidence(word, best_suggestion)
                    
                    if confidence >= self.min_confidence_threshold:
                        # Find and replace the word
                        pattern = r'\b' + re.escape(word) + r'\b'
                        matches = list(re.finditer(pattern, current_text, re.IGNORECASE))
                        
                        for match in reversed(matches):
                            start, end = match.span()
                            
                            correction = Correction(
                                original_text=word,
                                corrected_text=best_suggestion,
                                correction_type=CorrectionType.SPELLING,
                                confidence=confidence,
                                start_pos=start,
                                end_pos=end,
                                reason=f"Spell check suggestion: {word} -> {best_suggestion}"
                            )
                            corrections.append(correction)
                            
                            # Apply correction
                            current_text = current_text[:start] + best_suggestion + current_text[end:]
        
        return current_text, corrections
    
    def _correct_grammar(self, text: str) -> Tuple[str, List[Correction]]:
        """Perform grammar checking using LanguageTool"""
        corrections = []
        current_text = text
        
        if not self.grammar_tool:
            return current_text, corrections
        
        try:
            matches = self.grammar_tool.check(current_text)
            
            # Apply corrections in reverse order to maintain positions
            for match in reversed(matches):
                if (match.replacements and 
                    len(match.replacements[0]) > 0 and
                    match.replacements[0] != current_text[match.offset:match.offset + match.errorLength]):
                    
                    original = current_text[match.offset:match.offset + match.errorLength]
                    suggested = match.replacements[0]
                    
                    # Calculate confidence based on match category and context
                    confidence = self._calculate_grammar_confidence(match)
                    
                    if confidence >= self.min_confidence_threshold:
                        correction = Correction(
                            original_text=original,
                            corrected_text=suggested,
                            correction_type=CorrectionType.GRAMMAR,
                            confidence=confidence,
                            start_pos=match.offset,
                            end_pos=match.offset + match.errorLength,
                            reason=match.message
                        )
                        corrections.append(correction)
                        
                        # Apply correction
                        current_text = (current_text[:match.offset] + 
                                      suggested + 
                                      current_text[match.offset + match.errorLength:])
        
        except Exception as e:
            self.logger.warning(f"Grammar checking error: {e}")
        
        return current_text, corrections
    
    def _correct_context(self, text: str, context: Optional[str] = None) -> Tuple[str, List[Correction]]:
        """Perform AI-powered contextual correction"""
        corrections = []
        current_text = text
        
        if not self.ai_corrector:
            return current_text, corrections
        
        try:
            # Find potential context-based corrections
            sentences = re.split(r'[.!?]+', current_text)
            
            for i, sentence in enumerate(sentences):
                if len(sentence.strip()) < 5:
                    continue
                
                # Look for words that might need contextual correction
                words = sentence.split()
                for j, word in enumerate(words):
                    if self._needs_contextual_check(word):
                        # Create masked sentence for AI correction
                        masked_sentence = ' '.join(words[:j] + ['[MASK]'] + words[j+1:])
                        
                        try:
                            predictions = self.ai_corrector(masked_sentence)
                            if predictions and len(predictions) > 0:
                                best_pred = predictions[0]
                                suggested_word = best_pred['token_str']
                                confidence = best_pred['score']
                                
                                if (confidence >= 0.8 and 
                                    suggested_word.lower() != word.lower() and
                                    len(suggested_word) > 1):
                                    
                                    correction = Correction(
                                        original_text=word,
                                        corrected_text=suggested_word,
                                        correction_type=CorrectionType.CONTEXT,
                                        confidence=confidence,
                                        start_pos=0,  # Simplified for now
                                        end_pos=0,
                                        reason=f"Contextual AI correction: {word} -> {suggested_word}"
                                    )
                                    corrections.append(correction)
                                    
                                    # Apply correction to current text
                                    current_text = current_text.replace(word, suggested_word, 1)
                        
                        except Exception as e:
                            continue  # Skip this word if AI correction fails
        
        except Exception as e:
            self.logger.warning(f"Contextual correction error: {e}")
        
        return current_text, corrections
    
    def _apply_domain_rules(self, text: str, domain: str) -> Tuple[str, List[Correction]]:
        """Apply domain-specific correction rules"""
        corrections = []
        current_text = text
        
        if domain not in self.domain_rules:
            return current_text, corrections
        
        rules = self.domain_rules[domain]
        
        for rule in rules.get('patterns', []):
            pattern = rule.get('pattern', '')
            replacement = rule.get('replacement', '')
            confidence = rule.get('confidence', 0.8)
            
            if pattern and replacement:
                matches = list(re.finditer(pattern, current_text, re.IGNORECASE))
                
                for match in reversed(matches):
                    original = match.group(0)
                    start, end = match.span()
                    
                    correction = Correction(
                        original_text=original,
                        corrected_text=replacement,
                        correction_type=CorrectionType.FORMATTING,
                        confidence=confidence,
                        start_pos=start,
                        end_pos=end,
                        reason=f"Domain rule for {domain}: {pattern} -> {replacement}"
                    )
                    corrections.append(correction)
                    
                    current_text = current_text[:start] + replacement + current_text[end:]
        
        return current_text, corrections
    
    def _final_cleanup(self, text: str) -> Tuple[str, List[Correction]]:
        """Perform final text cleanup"""
        corrections = []
        current_text = text.strip()
        
        # Capitalize first letter of sentences
        sentences = re.split(r'([.!?]\s*)', current_text)
        cleaned_sentences = []
        
        for sentence in sentences:
            if sentence and sentence[0].islower() and sentence[0].isalpha():
                cleaned = sentence[0].upper() + sentence[1:]
                if cleaned != sentence:
                    correction = Correction(
                        original_text=sentence[:1],
                        corrected_text=cleaned[:1],
                        correction_type=CorrectionType.CAPITALIZATION,
                        confidence=0.95,
                        start_pos=0,
                        end_pos=1,
                        reason="Capitalized first letter of sentence"
                    )
                    corrections.append(correction)
                cleaned_sentences.append(cleaned)
            else:
                cleaned_sentences.append(sentence)
        
        current_text = ''.join(cleaned_sentences)
        
        return current_text, corrections
    
    def _needs_contextual_check(self, word: str) -> bool:
        """Determine if a word needs contextual checking"""
        # Check for common OCR confusion words
        confusion_words = {
            'to', 'too', 'two', 'there', 'their', 'they\'re',
            'your', 'you\'re', 'its', 'it\'s', 'then', 'than',
            'accept', 'except', 'affect', 'effect'
        }
        
        return word.lower() in confusion_words or len(word) < 3
    
    def _calculate_spelling_confidence(self, original: str, suggested: str) -> float:
        """Calculate confidence for spelling corrections"""
        # Use edit distance and other factors
        edit_distance = self.text_utils.calculate_edit_distance(original, suggested)
        max_len = max(len(original), len(suggested))
        
        if max_len == 0:
            return 0.0
        
        similarity = 1.0 - (edit_distance / max_len)
        
        # Boost confidence for common corrections
        common_corrections = {
            'teh': 'the', 'recieve': 'receive', 'seperate': 'separate'
        }
        
        if original.lower() in common_corrections:
            similarity = min(1.0, similarity + 0.2)
        
        return similarity
    
    def _calculate_grammar_confidence(self, match) -> float:
        """Calculate confidence for grammar corrections"""
        # Base confidence on match category
        category_confidence = {
            'TYPOS': 0.9,
            'GRAMMAR': 0.8,
            'STYLE': 0.6,
            'REDUNDANCY': 0.7,
            'CONFUSED_WORDS': 0.85
        }
        
        category = match.category if hasattr(match, 'category') else 'GRAMMAR'
        base_confidence = category_confidence.get(category, 0.7)
        
        # Adjust based on correction level
        if self.correction_level == CorrectionLevel.CONSERVATIVE:
            return base_confidence * 0.8
        elif self.correction_level == CorrectionLevel.AGGRESSIVE:
            return min(1.0, base_confidence * 1.2)
        
        return base_confidence
    
    def _calculate_correction_confidence(self, corrections: List[Correction]) -> float:
        """Calculate overall correction confidence"""
        if not corrections:
            return 1.0
        
        changed_corrections = [c for c in corrections if c.changed]
        if not changed_corrections:
            return 1.0
        
        # Weighted average of individual confidences
        total_weight = 0
        weighted_sum = 0
        
        for correction in changed_corrections:
            # Weight by correction type importance
            type_weights = {
                CorrectionType.SPELLING: 1.0,
                CorrectionType.GRAMMAR: 0.9,
                CorrectionType.OCR_ARTIFACTS: 1.1,
                CorrectionType.SPACING: 0.8,
                CorrectionType.CONTEXT: 0.7,
                CorrectionType.CAPITALIZATION: 0.6,
                CorrectionType.PUNCTUATION: 0.8,
                CorrectionType.FORMATTING: 0.5
            }
            
            weight = type_weights.get(correction.correction_type, 0.7)
            weighted_sum += correction.confidence * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 1.0
    
    def _generate_correction_statistics(
        self, 
        original: str, 
        corrected: str, 
        corrections: List[Correction]
    ) -> Dict[str, Any]:
        """Generate detailed correction statistics"""
        
        stats = {
            'original_length': len(original),
            'corrected_length': len(corrected),
            'total_corrections': len(corrections),
            'applied_corrections': len([c for c in corrections if c.changed]),
            'correction_types': {}
        }
        
        # Count corrections by type
        for correction in corrections:
            if correction.changed:
                type_name = correction.correction_type.value
                if type_name not in stats['correction_types']:
                    stats['correction_types'][type_name] = 0
                stats['correction_types'][type_name] += 1
        
        # Calculate text similarity
        similarity = difflib.SequenceMatcher(None, original, corrected).ratio()
        stats['text_similarity'] = similarity
        stats['change_ratio'] = 1.0 - similarity
        
        # Word-level statistics
        original_words = len(original.split()) if original else 0
        corrected_words = len(corrected.split()) if corrected else 0
        
        stats['word_count'] = {
            'original': original_words,
            'corrected': corrected_words,
            'change': corrected_words - original_words
        }
        
        return stats
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive correction statistics"""
        stats = dict(self.stats)
        
        if self.correction_history:
            recent_history = self.correction_history[-100:]
            
            stats['performance'] = {
                'avg_processing_time': statistics.mean([h['processing_time'] for h in recent_history]),
                'avg_corrections_per_text': statistics.mean([h['correction_count'] for h in recent_history]),
                'avg_confidence': statistics.mean([h['confidence'] for h in recent_history]),
                'total_processed': len(self.correction_history)
            }
            
            # Text length impact
            stats['text_analysis'] = {
                'avg_original_length': statistics.mean([h['original_length'] for h in recent_history]),
                'avg_corrected_length': statistics.mean([h['corrected_length'] for h in recent_history]),
                'avg_length_change': statistics.mean([
                    h['corrected_length'] - h['original_length'] for h in recent_history
                ])
            }
        
        stats['configuration'] = {
            'correction_level': self.correction_level.value,
            'min_confidence_threshold': self.min_confidence_threshold,
            'enable_ai_correction': self.enable_ai_correction,
            'enable_grammar_check': self.enable_grammar_check,
            'enable_spell_check': self.enable_spell_check,
            'components_available': {
                'languagetool': LANGUAGETOOL_AVAILABLE,
                'transformers': TRANSFORMERS_AVAILABLE,
                'grammar_tool': self.grammar_tool is not None,
                'ai_corrector': self.ai_corrector is not None
            }
        }
        
        return stats
    
    def add_custom_pattern(self, pattern: str, replacement: str, correction_type: CorrectionType = CorrectionType.OCR_ARTIFACTS):
        """Add custom correction pattern"""
        try:
            self.ocr_patterns.append((pattern, replacement))
            self.logger.info(f"Added custom pattern: {pattern} -> {replacement}")
            return True
        except Exception as e:
            self.logger.error(f"Error adding custom pattern: {e}")
            return False
    
    def add_domain_rules(self, domain: str, rules: Dict[str, Any]):
        """Add domain-specific correction rules"""
        try:
            self.domain_rules[domain] = rules
            self.logger.info(f"Added domain rules for: {domain}")
            return True
        except Exception as e:
            self.logger.error(f"Error adding domain rules: {e}")
            return False
    
    def export_corrections(self, correction_result: CorrectionResult, output_path: Union[str, Path]) -> bool:
        """Export correction analysis to file"""
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(correction_result.to_dict(), f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Corrections exported to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting corrections: {e}")
            return False
        
# Add alias for backward compatibility
TextCorrector = EnhancedTextCorrector
