"""
Enhanced Confidence Filter with Intelligent Filtering
Step 5: Advanced Post-processing Implementation

Features:
- Multi-level confidence scoring
- Context-aware filtering
- Quality assessment
- Adaptive thresholds
- Performance monitoring
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import time
import json
from collections import defaultdict, Counter
import statistics
import re

from ..core.base_engine import OCRResult, TextRegion, BoundingBox
from ..utils.config import ConfigManager
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ConfidenceAnalysis:
    """Confidence analysis result"""
    original_confidence: float
    adjusted_confidence: float
    quality_score: float
    factors: Dict[str, float] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    should_filter: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'original_confidence': self.original_confidence,
            'adjusted_confidence': self.adjusted_confidence,
            'quality_score': self.quality_score,
            'factors': self.factors,
            'recommendations': self.recommendations,
            'should_filter': self.should_filter
        }


@dataclass
class FilterResult:
    """Result of confidence filtering"""
    original_result: OCRResult
    filtered_result: OCRResult
    removed_regions: List[TextRegion]
    confidence_analysis: List[ConfidenceAnalysis]
    processing_time: float
    statistics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'original_regions_count': len(self.original_result.regions),
            'filtered_regions_count': len(self.filtered_result.regions),
            'removed_regions_count': len(self.removed_regions),
            'processing_time': self.processing_time,
            'statistics': self.statistics,
            'confidence_analysis': [analysis.to_dict() for analysis in self.confidence_analysis]
        }


class EnhancedConfidenceFilter:
    """
    Advanced confidence filtering with contextual understanding
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = ConfigManager(config_path).get_section('confidence_filter', {})
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        
        # Thresholds configuration
        self.base_confidence_threshold = self.config.get('base_confidence_threshold', 0.3)
        self.high_confidence_threshold = self.config.get('high_confidence_threshold', 0.8)
        self.adaptive_threshold = self.config.get('adaptive_threshold', True)
        self.context_boost = self.config.get('context_boost', 0.1)
        
        # Quality factors
        self.enable_length_factor = self.config.get('enable_length_factor', True)
        self.enable_dictionary_factor = self.config.get('enable_dictionary_factor', True)
        self.enable_context_factor = self.config.get('enable_context_factor', True)
        self.enable_pattern_factor = self.config.get('enable_pattern_factor', True)
        self.enable_geometric_factor = self.config.get('enable_geometric_factor', True)
        
        # Filter modes
        self.filter_mode = self.config.get('filter_mode', 'balanced')  # 'strict', 'balanced', 'lenient'
        self.preserve_high_confidence = self.config.get('preserve_high_confidence', True)
        self.min_region_size = self.config.get('min_region_size', 3)
        self.max_noise_ratio = self.config.get('max_noise_ratio', 0.5)
        
        # Initialize components
        self._initialize_dictionaries()
        self._initialize_patterns()
        
        # Statistics and performance tracking
        self.stats = defaultdict(int)
        self.confidence_history = []
        self.threshold_history = []
        
        self.logger.info(f"Enhanced confidence filter initialized with mode: {self.filter_mode}")
    
    def _initialize_dictionaries(self):
        """Initialize word dictionaries for quality scoring"""
        self.common_words = set()
        self.technical_terms = set()
        
        # Load common words
        try:
            dict_file = Path(self.config.get('common_words_file', 'data/common_words.txt'))
            if dict_file.exists():
                with open(dict_file, 'r', encoding='utf-8') as f:
                    self.common_words = {line.strip().lower() for line in f if line.strip()}
                self.logger.info(f"Loaded {len(self.common_words)} common words")
        except Exception as e:
            self.logger.warning(f"Could not load common words: {e}")
        
        # Fallback common words
        if not self.common_words:
            self.common_words = {
                'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'her',
                'was', 'one', 'our', 'had', 'by', 'word', 'oil', 'sit', 'now', 'find',
                'long', 'down', 'day', 'did', 'get', 'come', 'made', 'may', 'part',
                'time', 'very', 'when', 'much', 'first', 'water', 'been', 'call',
                'who', 'its', 'now', 'find', 'long', 'down', 'day', 'did', 'get'
            }
        
        # Load technical terms
        try:
            tech_file = Path(self.config.get('technical_terms_file', 'data/technical_terms.txt'))
            if tech_file.exists():
                with open(tech_file, 'r', encoding='utf-8') as f:
                    self.technical_terms = {line.strip().lower() for line in f if line.strip()}
                self.logger.info(f"Loaded {len(self.technical_terms)} technical terms")
        except Exception as e:
            self.logger.warning(f"Could not load technical terms: {e}")
    
    def _initialize_patterns(self):
        """Initialize text patterns for quality assessment"""
        self.valid_patterns = [
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # Dates
            r'\b\d{1,2}:\d{2}(?::\d{2})?\b',        # Times
            r'\b[A-Z]{1,3}\d+[A-Z]?\b',             # Codes (e.g., A123B)
            r'\b\d+(?:\.\d+)?%?\b',                 # Numbers/percentages
            r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', # Proper nouns
            r'\b[a-zA-Z]+@[a-zA-Z]+\.[a-zA-Z]+\b', # Email-like patterns
            r'\b\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b', # Phone numbers
        ]
        
        # FIXED: Corrected regex patterns with proper escaping
        self.noise_patterns = [
            r'^[^\w\s]+$',          # Only punctuation
            r'^\s*[|\\/_]+\s*$',    # Line artifacts
            r'^[.]{3,}$',           # Multiple dots
            r'^[-_]{3,}$',          # Multiple dashes/underscores
            r'^[^\x00-\x7F]+$',     # Non-ASCII characters only
            r'^\w{1,2}$',           # Very short fragments
        ]
    
    def filter_result(
        self,
        ocr_result: OCRResult,
        context: Optional[str] = None,
        custom_threshold: Optional[float] = None
    ) -> FilterResult:
        """
        Filter OCR result based on confidence and quality analysis
        
        Args:
            ocr_result: OCR result to filter
            context: Additional context for filtering decisions
            custom_threshold: Custom confidence threshold
            
        Returns:
            FilterResult with filtered regions
        """
        start_time = time.time()
        
        if not ocr_result.regions:
            return FilterResult(
                original_result=ocr_result,
                filtered_result=ocr_result,
                removed_regions=[],
                confidence_analysis=[],
                processing_time=time.time() - start_time
            )
        
        # Determine threshold
        threshold = custom_threshold or self._calculate_adaptive_threshold(ocr_result)
        
        # Analyze each region
        confidence_analyses = []
        filtered_regions = []
        removed_regions = []
        
        for i, region in enumerate(ocr_result.regions):
            analysis = self._analyze_region_confidence(
                region, ocr_result.regions, context, i
            )
            confidence_analyses.append(analysis)
            
            # Decide whether to keep the region
            should_keep = self._should_keep_region(region, analysis, threshold)
            
            if should_keep:
                # Update region with adjusted confidence
                updated_region = TextRegion(
                    text=getattr(region, 'text', getattr(region, 'full_text', str(region))),
                    bbox=region.bbox,
                    confidence=analysis.adjusted_confidence,
                    text_type=region.text_type,
                    language=region.language,
                    reading_order=region.reading_order
                )
                filtered_regions.append(updated_region)
            else:
                removed_regions.append(region)
        
        # Create filtered result
        filtered_result = OCRResult(
            text=' '.join(region.full_text for region in filtered_regions),
            confidence=(
                sum(region.confidence for region in filtered_regions) / len(filtered_regions)
                if filtered_regions else 0.0
            ),
            regions=filtered_regions,
            processing_time=ocr_result.processing_time,
            engine_name=f"{ocr_result.engine_name} + ConfidenceFilter",
            metadata={
                **ocr_result.metadata,
                'confidence_threshold': threshold,
                'regions_removed': len(removed_regions),
                'filter_mode': self.filter_mode
            }
        )
        
        processing_time = time.time() - start_time
        
        # Update statistics
        self.stats['total_processed'] += 1
        self.stats['regions_analyzed'] += len(ocr_result.regions)
        self.stats['regions_filtered'] += len(removed_regions)
        self.confidence_history.extend([analysis.adjusted_confidence for analysis in confidence_analyses])
        self.threshold_history.append(threshold)
        
        # Calculate statistics - FIXED: Properly handle statistics.mean function
        statistics_data = {
            'threshold_used': threshold,
            'removal_rate': len(removed_regions) / len(ocr_result.regions),
            'avg_original_confidence': self._safe_mean([a.original_confidence for a in confidence_analyses]),
            'avg_adjusted_confidence': self._safe_mean([a.adjusted_confidence for a in confidence_analyses]),
            'quality_distribution': self._calculate_quality_distribution(confidence_analyses)
        }
        
        result = FilterResult(
            original_result=ocr_result,
            filtered_result=filtered_result,
            removed_regions=removed_regions,
            confidence_analysis=confidence_analyses,
            processing_time=processing_time,
            statistics=statistics_data  # FIXED: Renamed to avoid name conflict
        )
        
        self.logger.info(
            f"Confidence filtering completed: {len(filtered_regions)} kept, "
            f"{len(removed_regions)} removed (threshold: {threshold:.3f})"
        )
        
        return result
    
    def _safe_mean(self, values: List[float]) -> float:
        """Safely calculate mean, handling empty lists"""
        if not values:
            return 0.0
        return statistics.mean(values)
    
    def _safe_stdev(self, values: List[float]) -> float:
        """Safely calculate standard deviation, handling edge cases"""
        if len(values) <= 1:
            return 0.0
        return statistics.stdev(values)
    
    def _calculate_adaptive_threshold(self, ocr_result: OCRResult) -> float:
        """Calculate adaptive threshold based on result characteristics"""
        if not self.adaptive_threshold:
            return self.base_confidence_threshold
        
        confidences = [region.confidence for region in ocr_result.regions]
        
        if not confidences:
            return self.base_confidence_threshold
        
        # Calculate statistics
        mean_confidence = self._safe_mean(confidences)
        std_confidence = self._safe_stdev(confidences)
        median_confidence = statistics.median(confidences)
        
        # Base threshold on distribution characteristics
        if self.filter_mode == 'strict':
            # Use higher threshold for strict mode
            threshold = max(
                self.base_confidence_threshold,
                median_confidence - 0.5 * std_confidence
            )
        elif self.filter_mode == 'lenient':
            # Use lower threshold for lenient mode
            threshold = min(
                self.base_confidence_threshold,
                mean_confidence - std_confidence
            )
        else:  # balanced
            # Balanced approach
            threshold = (
                0.4 * self.base_confidence_threshold +
                0.3 * mean_confidence +
                0.3 * median_confidence
            )
        
        # Ensure reasonable bounds
        threshold = max(0.1, min(0.9, threshold))
        
        return round(threshold, 3)
    
    def _analyze_region_confidence(
        self,
        region: TextRegion,
        all_regions: List[TextRegion],
        context: Optional[str],
        region_index: int
    ) -> ConfidenceAnalysis:
        """Analyze confidence of a single region"""
        factors = {}
        recommendations = []
        
        original_confidence = region.confidence
        adjusted_confidence = original_confidence
        
        # Length factor
        if self.enable_length_factor:
            length_factor = self._calculate_length_factor(region.full_text)
            factors['length'] = length_factor
            adjusted_confidence *= (1 + 0.1 * length_factor)
            
            if length_factor < -0.5:
                recommendations.append("Text too short, consider removal")
            elif length_factor > 0.5:
                recommendations.append("Good text length")
        
        # Dictionary factor
        if self.enable_dictionary_factor:
            dict_factor = self._calculate_dictionary_factor(region.full_text)
            factors['dictionary'] = dict_factor
            adjusted_confidence *= (1 + 0.15 * dict_factor)
            
            if dict_factor > 0.3:
                recommendations.append("Contains common words")
            elif dict_factor < -0.3:
                recommendations.append("Many unknown words")
        
        # Context factor
        if self.enable_context_factor and len(all_regions) > 1:
            context_factor = self._calculate_context_factor(
                region, all_regions, region_index
            )
            factors['context'] = context_factor
            adjusted_confidence *= (1 + 0.1 * context_factor)
            
            if context_factor > 0.2:
                recommendations.append("Consistent with surrounding text")
        
        # Pattern factor
        if self.enable_pattern_factor:
            pattern_factor = self._calculate_pattern_factor(region.full_text)
            factors['pattern'] = pattern_factor
            adjusted_confidence *= (1 + 0.1 * pattern_factor)
            
            if pattern_factor < -0.5:
                recommendations.append("Contains noise patterns")
        
        # Geometric factor
        if self.enable_geometric_factor:
            geometric_factor = self._calculate_geometric_factor(region)
            factors['geometric'] = geometric_factor
            adjusted_confidence *= (1 + 0.05 * geometric_factor)
            
            if geometric_factor < -0.3:
                recommendations.append("Unusual text region geometry")
        
        # Calculate quality score
        quality_score = sum(factors.values()) / len(factors) if factors else 0.0
        
        # Ensure confidence bounds
        adjusted_confidence = max(0.0, min(1.0, adjusted_confidence))
        
        # Determine if should filter
        should_filter = (
            adjusted_confidence < self.base_confidence_threshold or
            quality_score < -0.5 or
            len(region.full_text.strip()) < self.min_region_size
        )
        
        return ConfidenceAnalysis(
            original_confidence=original_confidence,
            adjusted_confidence=adjusted_confidence,
            quality_score=quality_score,
            factors=factors,
            recommendations=recommendations,
            should_filter=should_filter
        )
    
    def _calculate_length_factor(self, text: str) -> float:
        """Calculate factor based on text length"""
        text = text.strip()
        length = len(text)
        
        if length == 0:
            return -1.0
        elif length == 1:
            return -0.8
        elif length == 2:
            return -0.6
        elif 3 <= length <= 5:
            return -0.2
        elif 6 <= length <= 20:
            return 0.3
        elif 21 <= length <= 100:
            return 0.5
        else:
            return 0.2  # Very long text might be concatenated
    
    def _calculate_dictionary_factor(self, text: str) -> float:
        """Calculate factor based on dictionary word presence"""
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        
        if not words:
            return -0.5
        
        common_word_count = sum(1 for word in words if word in self.common_words)
        technical_word_count = sum(1 for word in words if word in self.technical_terms)
        
        common_ratio = common_word_count / len(words)
        technical_ratio = technical_word_count / len(words)
        
        # Boost for common words, smaller boost for technical terms
        factor = 0.8 * common_ratio + 0.4 * technical_ratio
        
        # Penalty for many unknown words
        unknown_ratio = 1 - (common_ratio + technical_ratio)
        if unknown_ratio > 0.7:
            factor -= 0.5 * unknown_ratio
        
        return min(1.0, max(-1.0, factor))
    
    def _calculate_context_factor(
        self,
        region: TextRegion,
        all_regions: List[TextRegion],
        region_index: int
    ) -> float:
        """Calculate factor based on context consistency"""
        if len(all_regions) < 2:
            return 0.0
        
        # Get neighboring regions
        neighbors = []
        if region_index > 0:
            neighbors.append(all_regions[region_index - 1])
        if region_index < len(all_regions) - 1:
            neighbors.append(all_regions[region_index + 1])
        
        if not neighbors:
            return 0.0
        
        # Calculate similarity with neighbors
        similarities = []
        for neighbor in neighbors:
            # Simple similarity based on character patterns
            region_chars = set(region.full_text.lower())
            neighbor_chars = set(neighbor.full_text.lower())
            
            if region_chars and neighbor_chars:
                similarity = len(region_chars & neighbor_chars) / len(region_chars | neighbor_chars)
                similarities.append(similarity)
        
        if not similarities:
            return 0.0
        
        avg_similarity = self._safe_mean(similarities)
        
        # Convert similarity to factor
        if avg_similarity > 0.3:
            return 0.5
        elif avg_similarity > 0.1:
            return 0.2
        else:
            return -0.2
    
    def _calculate_pattern_factor(self, text: str) -> float:
        """Calculate factor based on text patterns"""
        text = text.strip()
        
        if not text:
            return -1.0
        
        # Check for valid patterns
        valid_score = 0
        for pattern in self.valid_patterns:
            if re.search(pattern, text):
                valid_score += 0.2
        
        # Check for noise patterns
        noise_score = 0
        for pattern in self.noise_patterns:
            if re.search(pattern, text):
                noise_score += 0.3
        
        # Calculate final factor
        factor = valid_score - noise_score
        
        # Additional checks
        if len(text) < 3 and not re.search(r'\d', text):
            factor -= 0.3
        
        if re.search(r'[a-zA-Z]', text) and re.search(r'\d', text):
            factor += 0.1  # Mixed alphanumeric is often valid
        
        return min(1.0, max(-1.0, factor))
    
    def _calculate_geometric_factor(self, region: TextRegion) -> float:
        """Calculate factor based on region geometry"""
        if not region.bbox:
            return 0.0
        
        width = region.bbox.x2 - region.bbox.x1
        height = region.bbox.y2 - region.bbox.y1
        
        if width <= 0 or height <= 0:
            return -0.5
        
        aspect_ratio = width / height
        area = width * height
        
        # Reasonable aspect ratios for text
        if 0.1 <= aspect_ratio <= 50:
            geometric_factor = 0.2
        else:
            geometric_factor = -0.3
        
        # Very small regions are suspicious
        if area < 100:
            geometric_factor -= 0.3
        elif area > 10000:
            geometric_factor += 0.1
        
        return min(1.0, max(-1.0, geometric_factor))
    
    def _should_keep_region(
        self,
        region: TextRegion,
        analysis: ConfidenceAnalysis,
        threshold: float
    ) -> bool:
        """Determine if a region should be kept"""
        # Always keep high-confidence regions if enabled
        if self.preserve_high_confidence and analysis.original_confidence >= self.high_confidence_threshold:
            return True
        
        # Check adjusted confidence against threshold
        if analysis.adjusted_confidence < threshold:
            return False
        
        # Additional quality checks
        if analysis.quality_score < -0.7:
            return False
        
        # Minimum text length
        if len(region.full_text.strip()) < self.min_region_size:
            return False
        
        # Check noise ratio
        noise_chars = sum(1 for c in region.full_text if not c.isalnum() and not c.isspace())
        total_chars = len(region.full_text)
        if total_chars > 0 and noise_chars / total_chars > self.max_noise_ratio:
            return False
        
        return True
    
    def _calculate_quality_distribution(self, analyses: List[ConfidenceAnalysis]) -> Dict[str, int]:
        """Calculate quality score distribution"""
        distribution = {'high': 0, 'medium': 0, 'low': 0}
        
        for analysis in analyses:
            if analysis.quality_score > 0.3:
                distribution['high'] += 1
            elif analysis.quality_score > -0.3:
                distribution['medium'] += 1
            else:
                distribution['low'] += 1
        
        return distribution
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get filtering statistics"""
        stats = {
            'total_processed': self.stats['total_processed'],
            'regions_analyzed': self.stats['regions_analyzed'],
            'regions_filtered': self.stats['regions_filtered'],
            'filter_rate': (
                self.stats['regions_filtered'] / max(self.stats['regions_analyzed'], 1)
            ),
            'configuration': {
                'filter_mode': self.filter_mode,
                'base_threshold': self.base_confidence_threshold,
                'adaptive_threshold': self.adaptive_threshold,
                'preserve_high_confidence': self.preserve_high_confidence
            }
        }
        
        if self.confidence_history:
            stats['confidence_stats'] = {
                'mean': self._safe_mean(self.confidence_history),
                'median': statistics.median(self.confidence_history),
                'std': self._safe_stdev(self.confidence_history),
                'min': min(self.confidence_history),
                'max': max(self.confidence_history),
                'samples': len(self.confidence_history)
            }
        
        if self.threshold_history:
            stats['threshold_stats'] = {
                'mean': self._safe_mean(self.threshold_history),
                'min': min(self.threshold_history),
                'max': max(self.threshold_history),
                'samples': len(self.threshold_history)
            }
        
        return stats
    
    def reset_statistics(self):
        """Reset all statistics"""
        self.stats.clear()
        self.confidence_history.clear()
        self.threshold_history.clear()
        self.logger.info("Statistics reset")
    
    def optimize_threshold(self, validation_data: List[Tuple[OCRResult, List[bool]]]) -> float:
        """
        Optimize threshold based on validation data
        
        Args:
            validation_data: List of (OCRResult, ground_truth_labels) pairs
                           where ground_truth_labels[i] is True if region i should be kept
            
        Returns:
            Optimized threshold value
        """
        if not validation_data:
            return self.base_confidence_threshold
        
        best_threshold = self.base_confidence_threshold
        best_f1_score = 0.0
        
        # Test different thresholds
        for threshold in np.arange(0.1, 0.9, 0.05):
            true_positives = 0
            false_positives = 0
            true_negatives = 0
            false_negatives = 0
            
            for ocr_result, ground_truth in validation_data:
                filter_result = self.filter_result(ocr_result, custom_threshold=threshold)
                
                kept_indices = {i for i, region in enumerate(ocr_result.regions) 
                              if region in filter_result.filtered_result.regions}
                
                for i, should_keep in enumerate(ground_truth):
                    if i in kept_indices and should_keep:
                        true_positives += 1
                    elif i in kept_indices and not should_keep:
                        false_positives += 1
                    elif i not in kept_indices and should_keep:
                        false_negatives += 1
                    else:
                        true_negatives += 1
            
            # Calculate F1 score
            precision = true_positives / max(true_positives + false_positives, 1)
            recall = true_positives / max(true_positives + false_negatives, 1)
            f1_score = 2 * precision * recall / max(precision + recall, 1e-10)
            
            if f1_score > best_f1_score:
                best_f1_score = f1_score
                best_threshold = threshold
        
        self.logger.info(f"Optimized threshold: {best_threshold:.3f} (F1: {best_f1_score:.3f})")
        return best_threshold
AdvancedConfidenceFilter = EnhancedConfidenceFilter