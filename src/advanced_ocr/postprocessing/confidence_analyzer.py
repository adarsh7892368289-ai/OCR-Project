"""
Advanced OCR System - Confidence Analysis Module
ONLY JOB: Analyze and calculate confidence scores
DEPENDENCIES: results.py, config.py
USED BY: text_processor.py ONLY
"""

import numpy as np
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from ..results import OCRResult, ConfidenceMetrics, TextRegion, Word, Line
from ..config import OCRConfig


class ConfidenceFactors(Enum):
    """Factors that influence confidence scoring"""
    CONSENSUS = "consensus"
    CHARACTER_CLARITY = "character_clarity"
    WORD_DICTIONARY = "word_dictionary"
    SPATIAL_CONSISTENCY = "spatial_consistency"
    ENGINE_RELIABILITY = "engine_reliability"


@dataclass
class ConsensusAnalysis:
    """Analysis of consensus between multiple OCR results"""
    agreement_score: float
    conflicting_regions: List[Tuple[int, int]]  # Character positions with conflicts
    consensus_text: str
    engine_votes: Dict[str, float]


@dataclass  
class CharacterAnalysis:
    """Analysis of individual character confidence"""
    char_scores: List[float]
    problematic_chars: List[Tuple[int, str, float]]  # Position, char, score
    average_score: float
    min_score: float


class ConfidenceAnalyzer:
    """
    ONLY RESPONSIBILITY: Analyze and calculate confidence scores
    
    Receives OCRResult(s) from text_processor.py and returns enhanced confidence metrics.
    Does NOT perform text processing, result selection, or result fusion.
    """
    
    def __init__(self, config: OCRConfig):
        self.config = config
        self.min_confidence_threshold = config.postprocessing.confidence_threshold
        self.consensus_weight = 0.4
        self.clarity_weight = 0.3
        self.spatial_weight = 0.2
        self.reliability_weight = 0.1
        
        # Common English words for dictionary checking
        self._load_dictionary()
        
    def _load_dictionary(self):
        """Load common words for dictionary-based confidence"""
        # Basic English words - in production, load from file
        self.common_words = {
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have',
            'i', 'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you',
            'do', 'at', 'this', 'but', 'his', 'by', 'from', 'they',
            'she', 'or', 'an', 'will', 'my', 'one', 'all', 'would',
            'there', 'their', 'what', 'so', 'up', 'out', 'if', 'about',
            'who', 'get', 'which', 'go', 'me', 'when', 'make', 'can',
            'like', 'time', 'no', 'just', 'him', 'know', 'take',
            'people', 'into', 'year', 'your', 'good', 'some', 'could',
            'them', 'see', 'other', 'than', 'then', 'now', 'look',
            'only', 'come', 'its', 'over', 'think', 'also', 'back',
            'after', 'use', 'two', 'how', 'our', 'work', 'first',
            'well', 'way', 'even', 'new', 'want', 'because', 'any',
            'these', 'give', 'day', 'most', 'us'
        }
    
    def analyze_single_result(self, result: OCRResult) -> ConfidenceMetrics:
        """
        Analyze confidence for a single OCR result
        
        Args:
            result: Single OCR result to analyze
            
        Returns:
            Enhanced confidence metrics
        """
        # Character-level analysis
        char_analysis = self._analyze_character_confidence(result.text, result.confidence)
        
        # Word-level dictionary check
        word_scores = self._analyze_word_dictionary_match(result.text)
        
        # Spatial consistency check
        spatial_score = self._analyze_spatial_consistency(result.regions)
        
        # Engine reliability factor
        engine_reliability = self._get_engine_reliability(result.engine_name)
        
        # Combine all factors
        overall_confidence = self._calculate_weighted_confidence(
            char_analysis.average_score,
            word_scores,
            spatial_score,
            engine_reliability
        )
        
        return ConfidenceMetrics(
            overall=overall_confidence,
            word_level=[word_scores] * len(result.text.split()),
            char_level=char_analysis.char_scores,
            spatial_consistency=spatial_score,
            dictionary_match=word_scores,
            engine_reliability=engine_reliability,
            factors={
                ConfidenceFactors.CHARACTER_CLARITY.value: char_analysis.average_score,
                ConfidenceFactors.WORD_DICTIONARY.value: word_scores,
                ConfidenceFactors.SPATIAL_CONSISTENCY.value: spatial_score,
                ConfidenceFactors.ENGINE_RELIABILITY.value: engine_reliability
            }
        )
    
    def analyze_consensus(self, results: List[OCRResult]) -> ConfidenceMetrics:
        """
        Analyze confidence based on consensus between multiple results
        
        Args:
            results: List of OCR results to analyze consensus
            
        Returns:
            Enhanced confidence metrics with consensus analysis
        """
        if len(results) == 1:
            return self.analyze_single_result(results[0])
        
        # Perform consensus analysis
        consensus_analysis = self._perform_consensus_analysis(results)
        
        # Analyze the consensus text
        base_confidence = self.analyze_single_result(
            OCRResult(
                text=consensus_analysis.consensus_text,
                confidence=consensus_analysis.agreement_score,
                regions=results[0].regions,  # Use first result's regions
                engine_name="consensus"
            )
        )
        
        # Enhance with consensus factor
        consensus_enhanced = self._enhance_with_consensus(
            base_confidence, 
            consensus_analysis
        )
        
        return consensus_enhanced
    
    def _analyze_character_confidence(self, text: str, base_confidence: float) -> CharacterAnalysis:
        """Analyze confidence at character level"""
        char_scores = []
        problematic_chars = []
        
        for i, char in enumerate(text):
            score = self._calculate_character_score(char, i, text, base_confidence)
            char_scores.append(score)
            
            if score < self.min_confidence_threshold:
                problematic_chars.append((i, char, score))
        
        return CharacterAnalysis(
            char_scores=char_scores,
            problematic_chars=problematic_chars,
            average_score=np.mean(char_scores) if char_scores else 0.0,
            min_score=min(char_scores) if char_scores else 0.0
        )
    
    def _calculate_character_score(self, char: str, pos: int, text: str, base: float) -> float:
        """Calculate confidence score for individual character"""
        score = base
        
        # Boost confidence for common characters
        if char.isalpha() and char.lower() in 'etaoinshrdlu':
            score *= 1.1
        
        # Reduce confidence for easily confused characters
        confusing_chars = {'0O', '1lI', '5S', '8B', 'cl', 'rn', 'vv'}
        for pair in confusing_chars:
            if char in pair:
                score *= 0.9
                break
        
        # Context analysis - characters that fit well in context get boost
        if pos > 0 and pos < len(text) - 1:
            if self._is_contextually_appropriate(char, text[pos-1], text[pos+1]):
                score *= 1.05
        
        return min(1.0, max(0.0, score))
    
    def _is_contextually_appropriate(self, char: str, prev_char: str, next_char: str) -> bool:
        """Check if character fits well in its context"""
        # Simple heuristics for contextual appropriateness
        if char.isalpha():
            if prev_char.isalpha() and next_char.isalpha():
                return True
            if prev_char.isspace() and next_char.isalpha():
                return True
        elif char.isdigit():
            if prev_char.isdigit() or next_char.isdigit():
                return True
        elif char.isspace():
            if prev_char.isalpha() and next_char.isalpha():
                return True
        
        return False
    
    def _analyze_word_dictionary_match(self, text: str) -> float:
        """Analyze how many words match common dictionary"""
        words = re.findall(r'\b\w+\b', text.lower())
        if not words:
            return 0.5
        
        matches = sum(1 for word in words if word in self.common_words)
        return matches / len(words)
    
    def _analyze_spatial_consistency(self, regions: List[TextRegion]) -> float:
        """Analyze spatial consistency of text regions"""
        if not regions or len(regions) < 2:
            return 1.0
        
        # Check for reasonable spacing and alignment
        consistency_scores = []
        
        for i in range(len(regions) - 1):
            current = regions[i]
            next_region = regions[i + 1]
            
            # Check horizontal alignment for lines
            if hasattr(current, 'bbox') and hasattr(next_region, 'bbox'):
                alignment_score = self._calculate_alignment_score(
                    current.bbox, next_region.bbox
                )
                consistency_scores.append(alignment_score)
        
        return np.mean(consistency_scores) if consistency_scores else 1.0
    
    def _calculate_alignment_score(self, bbox1, bbox2) -> float:
        """Calculate alignment score between two bounding boxes"""
        # Simple alignment check - can be enhanced
        height_diff = abs(bbox1.height - bbox2.height)
        max_height = max(bbox1.height, bbox2.height)
        
        if max_height > 0:
            height_consistency = 1.0 - (height_diff / max_height)
        else:
            height_consistency = 1.0
        
        return max(0.0, height_consistency)
    
    def _get_engine_reliability(self, engine_name: str) -> float:
        """Get reliability score for specific engine"""
        # Engine reliability based on typical performance
        reliability_scores = {
            'tesseract': 0.85,
            'paddleocr': 0.90,
            'easyocr': 0.82,
            'trocr': 0.88,
            'consensus': 0.95
        }
        
        return reliability_scores.get(engine_name.lower(), 0.75)
    
    def _calculate_weighted_confidence(self, char_score: float, word_score: float, 
                                     spatial_score: float, reliability_score: float) -> float:
        """Calculate weighted overall confidence"""
        weighted = (
            char_score * self.clarity_weight +
            word_score * self.consensus_weight +  # Using consensus weight for word matching
            spatial_score * self.spatial_weight +
            reliability_score * self.reliability_weight
        )
        
        return min(1.0, max(0.0, weighted))
    
    def _perform_consensus_analysis(self, results: List[OCRResult]) -> ConsensusAnalysis:
        """Perform detailed consensus analysis between results"""
        texts = [r.text for r in results]
        engines = [r.engine_name for r in results]
        
        # Find consensus text using character-level voting
        consensus_text = self._build_consensus_text(texts)
        
        # Calculate agreement score
        agreement_score = self._calculate_agreement_score(texts, consensus_text)
        
        # Find conflicting regions
        conflicts = self._find_conflicting_regions(texts)
        
        # Calculate engine vote weights
        engine_votes = self._calculate_engine_votes(results, consensus_text)
        
        return ConsensusAnalysis(
            agreement_score=agreement_score,
            conflicting_regions=conflicts,
            consensus_text=consensus_text,
            engine_votes=engine_votes
        )
    
    def _build_consensus_text(self, texts: List[str]) -> str:
        """Build consensus text from multiple OCR results"""
        if not texts:
            return ""
        
        # For simplicity, use the longest common text as base
        # In production, implement more sophisticated consensus algorithm
        max_len_text = max(texts, key=len)
        
        # Simple consensus: character-by-character majority vote
        consensus_chars = []
        min_length = min(len(t) for t in texts)
        
        for i in range(min_length):
            chars_at_pos = [text[i] for text in texts]
            # Take most common character at this position
            consensus_char = max(set(chars_at_pos), key=chars_at_pos.count)
            consensus_chars.append(consensus_char)
        
        # Add remaining characters from longest text
        if len(max_len_text) > min_length:
            consensus_chars.extend(max_len_text[min_length:])
        
        return ''.join(consensus_chars)
    
    def _calculate_agreement_score(self, texts: List[str], consensus: str) -> float:
        """Calculate how well texts agree with consensus"""
        if not texts or not consensus:
            return 0.0
        
        agreement_scores = []
        for text in texts:
            # Calculate character-level similarity
            similarity = self._calculate_similarity(text, consensus)
            agreement_scores.append(similarity)
        
        return np.mean(agreement_scores)
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        if not text1 or not text2:
            return 0.0
        
        # Simple character-level similarity
        min_len = min(len(text1), len(text2))
        matches = sum(1 for i in range(min_len) if text1[i] == text2[i])
        
        max_len = max(len(text1), len(text2))
        return matches / max_len if max_len > 0 else 0.0
    
    def _find_conflicting_regions(self, texts: List[str]) -> List[Tuple[int, int]]:
        """Find character positions where texts conflict"""
        conflicts = []
        if len(texts) < 2:
            return conflicts
        
        min_len = min(len(t) for t in texts)
        
        for i in range(min_len):
            chars = set(text[i] for text in texts)
            if len(chars) > 1:  # Conflict found
                conflicts.append((i, i + 1))
        
        return conflicts
    
    def _calculate_engine_votes(self, results: List[OCRResult], consensus: str) -> Dict[str, float]:
        """Calculate how much each engine contributed to consensus"""
        votes = {}
        
        for result in results:
            similarity = self._calculate_similarity(result.text, consensus)
            reliability = self._get_engine_reliability(result.engine_name)
            vote_weight = similarity * reliability
            votes[result.engine_name] = vote_weight
        
        return votes
    
    def _enhance_with_consensus(self, base_confidence: ConfidenceMetrics, 
                               consensus: ConsensusAnalysis) -> ConfidenceMetrics:
        """Enhance confidence metrics with consensus information"""
        # Boost overall confidence based on consensus agreement
        consensus_boost = consensus.agreement_score * 0.2
        enhanced_overall = min(1.0, base_confidence.overall + consensus_boost)
        
        # Add consensus factor
        enhanced_factors = base_confidence.factors.copy()
        enhanced_factors[ConfidenceFactors.CONSENSUS.value] = consensus.agreement_score
        
        return ConfidenceMetrics(
            overall=enhanced_overall,
            word_level=base_confidence.word_level,
            char_level=base_confidence.char_level,
            spatial_consistency=base_confidence.spatial_consistency,
            dictionary_match=base_confidence.dictionary_match,
            engine_reliability=base_confidence.engine_reliability,
            factors=enhanced_factors
        )