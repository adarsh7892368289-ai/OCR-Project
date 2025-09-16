"""
Advanced OCR System - Result Fusion Module
ONLY JOB: Combine multiple engine results intelligently
DEPENDENCIES: results.py, confidence_analyzer.py, config.py
USED BY: text_processor.py ONLY
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from collections import Counter

from ..results import OCRResult, ConfidenceMetrics, TextRegion, Word, Line, BoundingBox
from ..config import OCRConfig
from .confidence_analyzer import ConfidenceAnalyzer


class FusionStrategy(Enum):
    """Different strategies for fusing OCR results"""
    WEIGHTED_VOTING = "weighted_voting"
    CONFIDENCE_BASED = "confidence_based"
    CONSENSUS_BUILDING = "consensus_building"
    ENGINE_PRIORITY = "engine_priority"


class FusionMethod(Enum):
    """Methods for combining text at different granularities"""
    CHARACTER_LEVEL = "character_level"
    WORD_LEVEL = "word_level"
    LINE_LEVEL = "line_level"
    REGION_LEVEL = "region_level"


@dataclass
class FusionDecision:
    """Decision made during fusion process"""
    position: int
    candidates: List[str]
    chosen: str
    reason: str
    confidence: float


@dataclass
class FusionMetrics:
    """Metrics from the fusion process"""
    num_inputs: int
    strategy_used: FusionStrategy
    decisions: List[FusionDecision]
    agreement_ratio: float
    fusion_confidence: float


class ResultFusion:
    """
    ONLY RESPONSIBILITY: Combine multiple engine results intelligently
    
    Receives multiple OCRResults from text_processor.py and uses confidence_analyzer.py
    for confidence scoring. Returns single fused OCRResult.
    Does NOT perform text cleaning, layout reconstruction, or engine selection.
    """
    
    def __init__(self, config: OCRConfig, confidence_analyzer: ConfidenceAnalyzer):
        self.config = config
        self.confidence_analyzer = confidence_analyzer
        self.fusion_strategy = FusionStrategy.WEIGHTED_VOTING
        self.fusion_method = FusionMethod.CHARACTER_LEVEL
        
        # Engine priority weights (higher = more trusted)
        self.engine_weights = {
            'paddleocr': 1.0,
            'tesseract': 0.9,
            'trocr': 0.95,
            'easyocr': 0.85
        }
        
        # Minimum agreement threshold for consensus
        self.min_consensus_ratio = 0.6
        
    def fuse_results(self, results: List[OCRResult]) -> Tuple[OCRResult, FusionMetrics]:
        """
        Fuse multiple OCR results into a single result
        
        Args:
            results: List of OCR results to fuse
            
        Returns:
            Tuple of (fused_result, fusion_metrics)
        """
        if not results:
            raise ValueError("No results provided for fusion")
        
        if len(results) == 1:
            # Single result - just enhance confidence
            enhanced_confidence = self.confidence_analyzer.analyze_single_result(results[0])
            fused_result = OCRResult(
                text=results[0].text,
                confidence=enhanced_confidence.overall,
                regions=results[0].regions,
                engine_name="single",
                confidence_metrics=enhanced_confidence
            )
            
            metrics = FusionMetrics(
                num_inputs=1,
                strategy_used=FusionStrategy.CONFIDENCE_BASED,
                decisions=[],
                agreement_ratio=1.0,
                fusion_confidence=enhanced_confidence.overall
            )
            
            return fused_result, metrics
        
        # Multiple results - perform fusion
        return self._perform_fusion(results)
    
    def _perform_fusion(self, results: List[OCRResult]) -> Tuple[OCRResult, FusionMetrics]:
        """Perform the actual fusion of multiple results"""
        # Analyze consensus to get enhanced confidence
        consensus_confidence = self.confidence_analyzer.analyze_consensus(results)
        
        # Choose fusion strategy based on results characteristics
        strategy = self._choose_fusion_strategy(results, consensus_confidence)
        
        # Perform fusion based on chosen strategy
        if strategy == FusionStrategy.WEIGHTED_VOTING:
            fused_result, decisions = self._weighted_voting_fusion(results)
        elif strategy == FusionStrategy.CONFIDENCE_BASED:
            fused_result, decisions = self._confidence_based_fusion(results)
        elif strategy == FusionStrategy.CONSENSUS_BUILDING:
            fused_result, decisions = self._consensus_building_fusion(results)
        else:  # ENGINE_PRIORITY
            fused_result, decisions = self._engine_priority_fusion(results)
        
        # Calculate fusion metrics
        agreement_ratio = self._calculate_agreement_ratio(results, fused_result.text)
        
        metrics = FusionMetrics(
            num_inputs=len(results),
            strategy_used=strategy,
            decisions=decisions,
            agreement_ratio=agreement_ratio,
            fusion_confidence=fused_result.confidence
        )
        
        # Enhance the fused result with consensus confidence
        fused_result.confidence_metrics = consensus_confidence
        
        return fused_result, metrics
    
    def _choose_fusion_strategy(self, results: List[OCRResult], 
                               consensus_confidence: ConfidenceMetrics) -> FusionStrategy:
        """Choose the best fusion strategy based on input characteristics"""
        # If high consensus agreement, use consensus building
        if consensus_confidence.factors.get('consensus', 0) > 0.8:
            return FusionStrategy.CONSENSUS_BUILDING
        
        # If results have very different confidence levels, use confidence-based
        confidences = [r.confidence for r in results]
        confidence_variance = np.var(confidences)
        if confidence_variance > 0.1:
            return FusionStrategy.CONFIDENCE_BASED
        
        # If we have known good engines, use engine priority
        known_engines = sum(1 for r in results if r.engine_name in self.engine_weights)
        if known_engines == len(results):
            return FusionStrategy.ENGINE_PRIORITY
        
        # Default to weighted voting
        return FusionStrategy.WEIGHTED_VOTING
    
    def _weighted_voting_fusion(self, results: List[OCRResult]) -> Tuple[OCRResult, List[FusionDecision]]:
        """Fuse results using weighted voting at character level"""
        decisions = []
        
        # Prepare weights based on engine reliability and confidence
        weights = []
        for result in results:
            engine_weight = self.engine_weights.get(result.engine_name, 0.5)
            confidence_weight = result.confidence
            combined_weight = engine_weight * confidence_weight
            weights.append(combined_weight)
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(results)] * len(results)
        
        # Perform character-level voting
        fused_text, char_decisions = self._character_level_voting(
            [r.text for r in results], weights
        )
        decisions.extend(char_decisions)
        
        # Calculate overall confidence
        overall_confidence = self._calculate_fusion_confidence(results, weights)
        
        # Use regions from highest confidence result
        best_result = max(results, key=lambda r: r.confidence)
        
        fused_result = OCRResult(
            text=fused_text,
            confidence=overall_confidence,
            regions=best_result.regions,
            engine_name="weighted_fusion"
        )
        
        return fused_result, decisions
    
    def _confidence_based_fusion(self, results: List[OCRResult]) -> Tuple[OCRResult, List[FusionDecision]]:
        """Fuse results by selecting highest confidence segments"""
        decisions = []
        
        # Select the result with highest overall confidence as base
        best_result = max(results, key=lambda r: r.confidence)
        
        # For areas where other results have significantly higher confidence,
        # consider replacing segments
        fused_text = best_result.text
        
        decision = FusionDecision(
            position=0,
            candidates=[r.text for r in results],
            chosen=fused_text,
            reason=f"Selected highest confidence result from {best_result.engine_name}",
            confidence=best_result.confidence
        )
        decisions.append(decision)
        
        fused_result = OCRResult(
            text=fused_text,
            confidence=best_result.confidence,
            regions=best_result.regions,
            engine_name="confidence_fusion"
        )
        
        return fused_result, decisions
    
    def _consensus_building_fusion(self, results: List[OCRResult]) -> Tuple[OCRResult, List[FusionDecision]]:
        """Build consensus by finding common elements across results"""
        decisions = []
        texts = [r.text for r in results]
        
        # Build consensus text character by character
        consensus_chars = []
        min_length = min(len(t) for t in texts) if texts else 0
        
        for i in range(min_length):
            chars_at_pos = [text[i] for text in texts]
            char_counter = Counter(chars_at_pos)
            
            # Find most common character
            most_common_char, count = char_counter.most_common(1)[0]
            consensus_ratio = count / len(chars_at_pos)
            
            if consensus_ratio >= self.min_consensus_ratio:
                chosen_char = most_common_char
                reason = f"Consensus {consensus_ratio:.2f}"
            else:
                # No clear consensus, use weighted approach
                result_weights = [self.engine_weights.get(r.engine_name, 0.5) * r.confidence 
                                for r in results]
                best_idx = np.argmax(result_weights)
                chosen_char = chars_at_pos[best_idx]
                reason = f"No consensus, used best engine: {results[best_idx].engine_name}"
            
            consensus_chars.append(chosen_char)
            
            decision = FusionDecision(
                position=i,
                candidates=list(set(chars_at_pos)),
                chosen=chosen_char,
                reason=reason,
                confidence=consensus_ratio if consensus_ratio >= self.min_consensus_ratio else 0.5
            )
            decisions.append(decision)
        
        # Handle remaining characters from longer texts
        if texts:
            longest_text = max(texts, key=len)
            if len(longest_text) > min_length:
                remaining = longest_text[min_length:]
                consensus_chars.extend(remaining)
                
                decision = FusionDecision(
                    position=min_length,
                    candidates=[remaining],
                    chosen=remaining,
                    reason="Extended from longest text",
                    confidence=0.7
                )
                decisions.append(decision)
        
        fused_text = ''.join(consensus_chars)
        
        # Calculate consensus confidence
        total_agreements = sum(1 for d in decisions if d.confidence > self.min_consensus_ratio)
        consensus_confidence = total_agreements / len(decisions) if decisions else 0.0
        
        # Use regions from result with best spatial consistency
        best_result = max(results, key=lambda r: r.confidence)
        
        fused_result = OCRResult(
            text=fused_text,
            confidence=consensus_confidence,
            regions=best_result.regions,
            engine_name="consensus_fusion"
        )
        
        return fused_result, decisions
    
    def _engine_priority_fusion(self, results: List[OCRResult]) -> Tuple[OCRResult, List[FusionDecision]]:
        """Fuse results based on predefined engine priorities"""
        decisions = []
        
        # Sort results by engine priority and confidence
        def priority_score(result):
            engine_weight = self.engine_weights.get(result.engine_name, 0.5)
            return engine_weight * result.confidence
        
        sorted_results = sorted(results, key=priority_score, reverse=True)
        best_result = sorted_results[0]
        
        decision = FusionDecision(
            position=0,
            candidates=[r.text for r in results],
            chosen=best_result.text,
            reason=f"Engine priority: {best_result.engine_name} (score: {priority_score(best_result):.3f})",
            confidence=best_result.confidence
        )
        decisions.append(decision)
        
        fused_result = OCRResult(
            text=best_result.text,
            confidence=best_result.confidence,
            regions=best_result.regions,
            engine_name="priority_fusion"
        )
        
        return fused_result, decisions
    
    def _character_level_voting(self, texts: List[str], 
                               weights: List[float]) -> Tuple[str, List[FusionDecision]]:
        """Perform character-level weighted voting"""
        decisions = []
        
        if not texts:
            return "", decisions
        
        # Find the maximum length
        max_length = max(len(text) for text in texts)
        fused_chars = []
        
        for i in range(max_length):
            # Get characters at this position (use space for shorter texts)
            chars_and_weights = []
            for j, text in enumerate(texts):
                char = text[i] if i < len(text) else ' '
                chars_and_weights.append((char, weights[j]))
            
            # Calculate weighted votes for each unique character
            char_votes = {}
            for char, weight in chars_and_weights:
                if char not in char_votes:
                    char_votes[char] = 0.0
                char_votes[char] += weight
            
            # Choose character with highest weighted vote
            if char_votes:
                chosen_char = max(char_votes.items(), key=lambda x: x[1])[0]
                confidence = char_votes[chosen_char]
                
                # Skip trailing spaces
                if chosen_char == ' ' and i >= len(max(texts, key=len).rstrip()):
                    break
                    
                fused_chars.append(chosen_char)
                
                decision = FusionDecision(
                    position=i,
                    candidates=list(char_votes.keys()),
                    chosen=chosen_char,
                    reason=f"Weighted vote: {confidence:.3f}",
                    confidence=confidence
                )
                decisions.append(decision)
        
        return ''.join(fused_chars), decisions
    
    def _calculate_fusion_confidence(self, results: List[OCRResult], 
                                   weights: List[float]) -> float:
        """Calculate overall confidence for fused result"""
        if not results or not weights:
            return 0.0
        
        # Weighted average of input confidences
        weighted_confidence = sum(r.confidence * w for r, w in zip(results, weights))
        
        # Boost confidence if results agree
        agreement_bonus = self._calculate_agreement_bonus(results)
        
        # Combine weighted confidence with agreement bonus
        final_confidence = min(1.0, weighted_confidence + agreement_bonus)
        
        return final_confidence
    
    def _calculate_agreement_bonus(self, results: List[OCRResult]) -> float:
        """Calculate bonus confidence based on agreement between results"""
        if len(results) < 2:
            return 0.0
        
        texts = [r.text for r in results]
        agreements = []
        
        # Calculate pairwise agreements
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                similarity = self._calculate_text_similarity(texts[i], texts[j])
                agreements.append(similarity)
        
        # Average agreement
        avg_agreement = np.mean(agreements) if agreements else 0.0
        
        # Convert to bonus (max 0.2 bonus for perfect agreement)
        return min(0.2, avg_agreement * 0.2)
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        if not text1 or not text2:
            return 0.0
        
        # Simple character-level similarity
        min_len = min(len(text1), len(text2))
        matches = sum(1 for i in range(min_len) if text1[i] == text2[i])
        
        max_len = max(len(text1), len(text2))
        return matches / max_len if max_len > 0 else 0.0
    
    def _calculate_agreement_ratio(self, results: List[OCRResult], fused_text: str) -> float:
        """Calculate how well the fused result represents input agreement"""
        if not results:
            return 0.0
        
        agreements = []
        for result in results:
            similarity = self._calculate_text_similarity(result.text, fused_text)
            agreements.append(similarity)
        
        return np.mean(agreements)
    
    def get_fusion_summary(self, fusion_metrics: FusionMetrics) -> Dict[str, any]:
        """
        Get a summary of the fusion process for debugging/analysis
        
        Args:
            fusion_metrics: Metrics from fusion process
            
        Returns:
            Dictionary containing fusion summary
        """
        decision_summary = {}
        if fusion_metrics.decisions:
            # Summarize decisions by reason
            reason_counts = {}
            confidence_sum = 0.0
            
            for decision in fusion_metrics.decisions:
                reason = decision.reason.split(':')[0]  # Get main reason
                if reason not in reason_counts:
                    reason_counts[reason] = 0
                reason_counts[reason] += 1
                confidence_sum += decision.confidence
            
            decision_summary = {
                'reason_distribution': reason_counts,
                'avg_decision_confidence': confidence_sum / len(fusion_metrics.decisions),
                'total_decisions': len(fusion_metrics.decisions)
            }
        
        return {
            'strategy_used': fusion_metrics.strategy_used.value,
            'num_input_results': fusion_metrics.num_inputs,
            'agreement_ratio': fusion_metrics.agreement_ratio,
            'fusion_confidence': fusion_metrics.fusion_confidence,
            'decision_summary': decision_summary
        }