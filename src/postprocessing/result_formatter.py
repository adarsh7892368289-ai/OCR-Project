"""
Advanced Confidence Filter with Multi-Level Quality Assessment
Step 5: Advanced Post-processing Implementation

Features:
- Multi-dimensional confidence scoring
- Adaptive threshold adjustment
- Quality-based text filtering
- Statistical confidence analysis
- Performance optimization
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import time
import json
from collections import defaultdict, Counter
from enum import Enum
import re
import statistics

from ..core.base_engine import OCRResult, TextRegion, BoundingBox
from ..utils.config import ConfigManager
from ..utils.logger import get_logger
from ..utils.text_utils import TextUtils

logger = get_logger(__name__)


class ConfidenceType(Enum):
    """Types of confidence scores"""
    CHARACTER_LEVEL = "character_level"
    WORD_LEVEL = "word_level"
    LINE_LEVEL = "line_level"
    BLOCK_LEVEL = "block_level"
    OVERALL = "overall"


class QualityLevel(Enum):
    """Quality assessment levels"""
    EXCELLENT = "excellent"      # > 0.95
    GOOD = "good"               # 0.8 - 0.95
    MODERATE = "moderate"       # 0.6 - 0.8
    POOR = "poor"               # 0.4 - 0.6
    VERY_POOR = "very_poor"     # < 0.4


@dataclass
class ConfidenceMetrics:
    """Comprehensive confidence metrics"""
    raw_confidence: float
    adjusted_confidence: float
    character_confidence: Optional[float] = None
    word_confidence: Optional[float] = None
    geometric_mean: Optional[float] = None
    harmonic_mean: Optional[float] = None
    quality_level: QualityLevel = QualityLevel.MODERATE
    confidence_variance: float = 0.0
    outlier_ratio: float = 0.0
    consistency_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'raw_confidence': self.raw_confidence,
            'adjusted_confidence': self.adjusted_confidence,
            'character_confidence': self.character_confidence,
            'word_confidence': self.word_confidence,
            'geometric_mean': self.geometric_mean,
            'harmonic_mean': self.harmonic_mean,
            'quality_level': self.quality_level.value,
            'confidence_variance': self.confidence_variance,
            'outlier_ratio': self.outlier_ratio,
            'consistency_score': self.consistency_score
        }


@dataclass
class FilterResult:
    """Result of confidence filtering operation"""
    original_regions: List[TextRegion]
    filtered_regions: List[TextRegion]
    rejected_regions: List[TextRegion]
    overall_confidence: float
    filter_statistics: Dict[str, Any]
    processing_time: float = 0.0
    
    @property
    def acceptance_rate(self) -> float:
        if not self.original_regions:
            return 0.0
        return len(self.filtered_regions) / len(self.original_regions)
    
    @property
    def rejection_rate(self) -> float:
        return 1.0 - self.acceptance_rate


class AdvancedConfidenceFilter:
    """
    Advanced confidence filter with multi-level quality assessment
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = ConfigManager(config_path).get_section('confidence_filter', {})
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        
        # Threshold configuration
        self.base_threshold = self.config.get('base_threshold', 0.5)
        self.character_threshold = self.config.get('character_threshold', 0.3)
        self.word_threshold = self.config.get('word_threshold', 0.4)
        self.adaptive_thresholds = self.config.get('enable_adaptive_thresholds', True)
        
        # Quality assessment
        self.quality_thresholds = {
            QualityLevel.EXCELLENT: self.config.get('excellent_threshold', 0.95),
            QualityLevel.GOOD: self.config.get('good_threshold', 0.8),
            QualityLevel.MODERATE: self.config.get('moderate_threshold', 0.6),
            QualityLevel.POOR: self.config.get('poor_threshold', 0.4)
        }
        
        # Advanced filtering options
        self.enable_outlier_detection = self.config.get('enable_outlier_detection', True)
        self.enable_consistency_check = self.config.get('enable_consistency_check', True)
        self.enable_context_filtering = self.config.get('enable_context_filtering', True)
        self.enable_length_weighting = self.config.get('enable_length_weighting', True)
        
        # Statistical parameters
        self.outlier_z_threshold = self.config.get('outlier_z_threshold', 2.0)
        self.min_word_length = self.config.get('min_word_length', 2)
        self.max_variance_threshold = self.config.get('max_variance_threshold', 0.3)
        
        # Initialize utilities
        self.text_utils = TextUtils()
        
        # Adaptive threshold history
        self.threshold_history = []
        self.performance_history = []
        
        # Statistics
        self.stats = defaultdict(int)
        self.quality_distribution = defaultdict(int)
        
        self.logger.info("Advanced confidence filter initialized")
    
    def filter_by_confidence(
        self,
        ocr_result: OCRResult,
        min_confidence: Optional[float] = None,
        adaptive: bool = None
    ) -> FilterResult:
        """
        Filter OCR results by confidence with advanced quality assessment
        
        Args:
            ocr_result: OCR result to filter
            min_confidence: Minimum confidence threshold (overrides config)
            adaptive: Use adaptive thresholds (overrides config)
            
        Returns:
            FilterResult with filtered regions and statistics
        """
        start_time = time.time()
        
        if not ocr_result.regions:
            return FilterResult(
                original_regions=[],
                filtered_regions=[],
                rejected_regions=[],
                overall_confidence=0.0,
                filter_statistics={},
                processing_time=time.time() - start_time
            )
        
        # Use provided parameters or defaults
        threshold = min_confidence if min_confidence is not None else self.base_threshold
        use_adaptive = adaptive if adaptive is not None else self.adaptive_thresholds
        
        try:
            # Step 1: Calculate enhanced confidence metrics for all regions
            enhanced_regions = []
            for region in ocr_result.regions:
                metrics = self._calculate_confidence_metrics(region)
                enhanced_regions.append((region, metrics))
            
            # Step 2: Apply adaptive threshold if enabled
            if use_adaptive:
                threshold = self._calculate_adaptive_threshold(enhanced_regions, threshold)
            
            # Step 3: Filter regions based on confidence and quality criteria
            filtered_regions = []
            rejected_regions = []
            
            for region, metrics in enhanced_regions:
                if self._should_accept_region(region, metrics, threshold):
                    filtered_regions.append(region)
                else:
                    rejected_regions.append(region)
            
            # Step 4: Calculate overall statistics
            overall_confidence = self._calculate_overall_confidence(filtered_regions)
            filter_stats = self._generate_filter_statistics(
                enhanced_regions, filtered_regions, rejected_regions, threshold
            )
            
            processing_time = time.time() - start_time
            
            # Update statistics
            self.stats['total_filtered'] += 1
            self.stats['regions_processed'] += len(ocr_result.regions)
            self.stats['regions_accepted'] += len(filtered_regions)
            self.stats['regions_rejected'] += len(rejected_regions)
            
            # Track threshold performance
            if use_adaptive:
                self.threshold_history.append({
                    'threshold': threshold,
                    'acceptance_rate': len(filtered_regions) / len(ocr_result.regions),
                    'timestamp': time.time()
                })
            
            result = FilterResult(
                original_regions=ocr_result.regions,
                filtered_regions=filtered_regions,
                rejected_regions=rejected_regions,
                overall_confidence=overall_confidence,
                filter_statistics=filter_stats,
                processing_time=processing_time
            )
            
            self.logger.info(
                f"Confidence filtering completed: {len(filtered_regions)}/{len(ocr_result.regions)} "
                f"regions accepted (threshold: {threshold:.3f})"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in confidence filtering: {e}")
            processing_time = time.time() - start_time
            return FilterResult(
                original_regions=ocr_result.regions,
                filtered_regions=[],
                rejected_regions=ocr_result.regions,
                overall_confidence=0.0,
                filter_statistics={'error': str(e)},
                processing_time=processing_time
            )
    
    def _calculate_confidence_metrics(self, region: TextRegion) -> ConfidenceMetrics:
        """Calculate comprehensive confidence metrics for a region"""
        raw_confidence = region.confidence
        
        # Initialize metrics
        metrics = ConfidenceMetrics(
            raw_confidence=raw_confidence,
            adjusted_confidence=raw_confidence
        )
        
        if not region.text or not region.text.strip():
            return metrics
        
        # Character-level analysis
        if hasattr(region, 'char_confidences') and region.char_confidences:
            char_confidences = region.char_confidences
            metrics.character_confidence = statistics.mean(char_confidences)
            metrics.confidence_variance = statistics.variance(char_confidences) if len(char_confidences) > 1 else 0.0
            
            # Detect outliers
            if self.enable_outlier_detection and len(char_confidences) > 3:
                mean_conf = statistics.mean(char_confidences)
                std_conf = statistics.stdev(char_confidences)
                outliers = [c for c in char_confidences if abs(c - mean_conf) > self.outlier_z_threshold * std_conf]
                metrics.outlier_ratio = len(outliers) / len(char_confidences)
        
        # Word-level analysis
        words = region.text.split()
        if hasattr(region, 'word_confidences') and region.word_confidences:
            word_confidences = region.word_confidences
            metrics.word_confidence = statistics.mean(word_confidences)
        else:
            # Estimate word confidence from character confidence
            if metrics.character_confidence is not None:
                metrics.word_confidence = metrics.character_confidence
        
        # Calculate alternative confidence measures
        if metrics.character_confidence is not None and metrics.word_confidence is not None:
            confidences = [metrics.character_confidence, metrics.word_confidence, raw_confidence]
            confidences = [c for c in confidences if c > 0]
            
            if confidences:
                # Geometric mean (more conservative)
                metrics.geometric_mean = statistics.geometric_mean(confidences)
                # Harmonic mean (even more conservative)
                metrics.harmonic_mean = statistics.harmonic_mean(confidences)
        
        # Text quality heuristics
        text_quality = self._assess_text_quality(region.text)
        
        # Length-based weighting
        if self.enable_length_weighting:
            length_factor = min(1.0, len(region.text.strip()) / 20.0)  # Longer text gets more weight
            metrics.adjusted_confidence = raw_confidence * (0.7 + 0.3 * length_factor)
        
        # Context-based adjustment
        if self.enable_context_filtering:
            context_factor = self._assess_context_quality(region.text)
            metrics.adjusted_confidence *= context_factor
        
        # Quality consistency check
        if self.enable_consistency_check:
            metrics.consistency_score = self._calculate_consistency_score(region)
            metrics.adjusted_confidence *= (0.8 + 0.2 * metrics.consistency_score)
        
        # Apply text quality factor
        metrics.adjusted_confidence *= text_quality
        
        # Determine quality level
        metrics.quality_level = self._determine_quality_level(metrics.adjusted_confidence)
        
        # Track quality distribution
        self.quality_distribution[metrics.quality_level] += 1
        
        return metrics
    
    def _assess_text_quality(self, text: str) -> float:
        """Assess text quality based on linguistic heuristics"""
        if not text or not text.strip():
            return 0.0
        
        quality_score = 1.0
        text = text.strip()
        
        # Check for gibberish patterns
        if len(text) < 2:
            quality_score *= 0.5
        
        # Check for excessive special characters
        special_char_ratio = len(re.findall(r'[^a-zA-Z0-9\s\-.,!?()"]', text)) / len(text)
        if special_char_ratio > 0.3:
            quality_score *= (1.0 - special_char_ratio)
        
        # Check for reasonable character distribution
        alpha_ratio = len(re.findall(r'[a-zA-Z]', text)) / len(text)
        if alpha_ratio < 0.3:  # Too few letters
            quality_score *= (0.5 + 0.5 * alpha_ratio / 0.3)
        
        # Check for excessive repetition
        if len(set(text.lower())) < len(text) * 0.3:  # Too repetitive
            unique_ratio = len(set(text.lower())) / len(text)
            quality_score *= (0.5 + 0.5 * unique_ratio / 0.3)
        
        # Check for word-like patterns
        words = text.split()
        if words:
            avg_word_length = statistics.mean(len(word) for word in words)
            if avg_word_length < 2:
                quality_score *= 0.6
            elif avg_word_length > 15:
                quality_score *= 0.8
        
        # Check for mixed case patterns (often indicates OCR errors)
        if re.search(r'[a-z][A-Z]|[A-Z][a-z][A-Z]', text):
            quality_score *= 0.9
        
        return max(0.1, min(1.0, quality_score))
    
    def _assess_context_quality(self, text: str) -> float:
        """Assess quality based on contextual factors"""
        if not text or not text.strip():
            return 0.5
        
        context_score = 1.0
        
        # Check for common OCR error patterns
        error_patterns = [
            r'\b[0-9]+[a-zA-Z]+[0-9]+\b',  # Mixed numbers and letters
            r'\b[a-zA-Z]{1,2}\b.*\b[a-zA-Z]{1,2}\b',  # Too many short words
            r'[!@#$%^&*]{2,}',  # Excessive special characters
            r'\b[BCDFGHJKLMNPQRSTVWXYZ]{4,}\b'  # Too many consonants
        ]
        
        for pattern in error_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                context_score *= 0.8
        
        # Boost score for dictionary words
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        if words and hasattr(self.text_utils, 'is_dictionary_word'):
            dict_ratio = sum(1 for word in words if self.text_utils.is_dictionary_word(word)) / len(words)
            context_score *= (0.7 + 0.3 * dict_ratio)
        
        return max(0.3, min(1.0, context_score))
    
    def _calculate_consistency_score(self, region: TextRegion) -> float:
        """Calculate consistency score for confidence values"""
        if not hasattr(region, 'char_confidences') or not region.char_confidences:
            return 1.0
        
        confidences = region.char_confidences
        if len(confidences) < 2:
            return 1.0
        
        # Calculate coefficient of variation (lower is more consistent)
        mean_conf = statistics.mean(confidences)
        std_conf = statistics.stdev(confidences)
        
        if mean_conf == 0:
            return 0.0
        
        cv = std_conf / mean_conf
        # Convert to consistency score (0-1, higher is better)
        consistency = max(0.0, 1.0 - cv)
        
        return consistency
    
    def _calculate_adaptive_threshold(
        self, 
        enhanced_regions: List[Tuple[TextRegion, ConfidenceMetrics]], 
        base_threshold: float
    ) -> float:
        """Calculate adaptive threshold based on confidence distribution"""
        if not enhanced_regions:
            return base_threshold
        
        confidences = [metrics.adjusted_confidence for _, metrics in enhanced_regions]
        
        # Calculate statistics
        mean_conf = statistics.mean(confidences)
        median_conf = statistics.median(confidences)
        std_conf = statistics.stdev(confidences) if len(confidences) > 1 else 0.1
        
        # Adaptive strategy based on distribution
        if mean_conf > 0.8:
            # High quality overall - be more selective
            adaptive_threshold = max(base_threshold, median_conf - 0.5 * std_conf)
        elif mean_conf < 0.4:
            # Low quality overall - be more lenient
            adaptive_threshold = min(base_threshold, mean_conf - std_conf)
        else:
            # Moderate quality - use median-based threshold
            adaptive_threshold = median_conf - 0.3 * std_conf
        
        # Ensure reasonable bounds
        adaptive_threshold = max(0.1, min(0.9, adaptive_threshold))
        
        # Smooth threshold changes if we have history
        if self.threshold_history:
            recent_threshold = statistics.mean([h['threshold'] for h in self.threshold_history[-5:]])
            adaptive_threshold = 0.7 * adaptive_threshold + 0.3 * recent_threshold
        
        return adaptive_threshold
    
    def _should_accept_region(
        self, 
        region: TextRegion, 
        metrics: ConfidenceMetrics, 
        threshold: float
    ) -> bool:
        """Determine if a region should be accepted based on comprehensive criteria"""
        
        # Primary confidence check
        if metrics.adjusted_confidence < threshold:
            return False
        
        # Quality level check
        if metrics.quality_level == QualityLevel.VERY_POOR:
            return False
        
        # Text content checks
        if not region.text or not region.text.strip():
            return False
        
        text = region.text.strip()
        
        # Minimum length check
        if len(text) < self.min_word_length:
            return False
        
        # Variance check (if available)
        if (metrics.confidence_variance > self.max_variance_threshold and 
            metrics.adjusted_confidence < 0.8):
            return False
        
        # Outlier ratio check
        if (metrics.outlier_ratio > 0.5 and 
            metrics.adjusted_confidence < 0.7):
            return False
        
        # Consistency check
        if (self.enable_consistency_check and 
            metrics.consistency_score < 0.3 and 
            metrics.adjusted_confidence < 0.7):
            return False
        
        return True
    
    def _determine_quality_level(self, confidence: float) -> QualityLevel:
        """Determine quality level based on confidence score"""
        if confidence >= self.quality_thresholds[QualityLevel.EXCELLENT]:
            return QualityLevel.EXCELLENT
        elif confidence >= self.quality_thresholds[QualityLevel.GOOD]:
            return QualityLevel.GOOD
        elif confidence >= self.quality_thresholds[QualityLevel.MODERATE]:
            return QualityLevel.MODERATE
        elif confidence >= self.quality_thresholds[QualityLevel.POOR]:
            return QualityLevel.POOR
        else:
            return QualityLevel.VERY_POOR
    
    def _calculate_overall_confidence(self, regions: List[TextRegion]) -> float:
        """Calculate overall confidence for filtered regions"""
        if not regions:
            return 0.0
        
        # Weight by text length
        weighted_sum = 0.0
        total_weight = 0.0
        
        for region in regions:
            weight = max(1.0, len(region.text.strip()) if region.text else 0)
            weighted_sum += region.confidence * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    
    def _generate_filter_statistics(
        self,
        enhanced_regions: List[Tuple[TextRegion, ConfidenceMetrics]],
        filtered_regions: List[TextRegion],
        rejected_regions: List[TextRegion],
        threshold: float
    ) -> Dict[str, Any]:
        """Generate comprehensive filtering statistics"""
        
        total_regions = len(enhanced_regions)
        accepted_count = len(filtered_regions)
        rejected_count = len(rejected_regions)
        
        # Confidence statistics
        all_confidences = [metrics.adjusted_confidence for _, metrics in enhanced_regions]
        accepted_confidences = [r.confidence for r in filtered_regions]
        rejected_confidences = [r.confidence for r in rejected_regions]
        
        stats = {
            'threshold_used': threshold,
            'total_regions': total_regions,
            'accepted_regions': accepted_count,
            'rejected_regions': rejected_count,
            'acceptance_rate': accepted_count / total_regions if total_regions > 0 else 0,
            'rejection_rate': rejected_count / total_regions if total_regions > 0 else 0,
            'confidence_stats': {
                'all': {
                    'mean': statistics.mean(all_confidences) if all_confidences else 0,
                    'median': statistics.median(all_confidences) if all_confidences else 0,
                    'std': statistics.stdev(all_confidences) if len(all_confidences) > 1 else 0,
                    'min': min(all_confidences) if all_confidences else 0,
                    'max': max(all_confidences) if all_confidences else 0
                },
                'accepted': {
                    'mean': statistics.mean(accepted_confidences) if accepted_confidences else 0,
                    'median': statistics.median(accepted_confidences) if accepted_confidences else 0,
                    'std': statistics.stdev(accepted_confidences) if len(accepted_confidences) > 1 else 0
                },
                'rejected': {
                    'mean': statistics.mean(rejected_confidences) if rejected_confidences else 0,
                    'median': statistics.median(rejected_confidences) if rejected_confidences else 0,
                    'std': statistics.stdev(rejected_confidences) if len(rejected_confidences) > 1 else 0
                }
            },
            'quality_distribution': dict(self.quality_distribution),
            'text_length_stats': {
                'accepted_avg_length': statistics.mean([
                    len(r.text) for r in filtered_regions if r.text
                ]) if filtered_regions else 0,
                'rejected_avg_length': statistics.mean([
                    len(r.text) for r in rejected_regions if r.text
                ]) if rejected_regions else 0
            }
        }
        
        return stats
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive filtering statistics"""
        stats = dict(self.stats)
        
        # Add performance metrics
        if self.threshold_history:
            recent_history = self.threshold_history[-100:]
            stats['adaptive_threshold_stats'] = {
                'avg_threshold': statistics.mean([h['threshold'] for h in recent_history]),
                'threshold_std': statistics.stdev([h['threshold'] for h in recent_history]) if len(recent_history) > 1 else 0,
                'avg_acceptance_rate': statistics.mean([h['acceptance_rate'] for h in recent_history]),
                'samples': len(recent_history)
            }
        
        stats['quality_distribution'] = dict(self.quality_distribution)
        
        stats['configuration'] = {
            'base_threshold': self.base_threshold,
            'adaptive_thresholds': self.adaptive_thresholds,
            'enable_outlier_detection': self.enable_outlier_detection,
            'enable_consistency_check': self.enable_consistency_check,
            'enable_context_filtering': self.enable_context_filtering
        }
        
        # Calculate efficiency metrics
        if self.stats['total_filtered'] > 0:
            stats['efficiency'] = {
                'avg_acceptance_rate': self.stats['regions_accepted'] / self.stats['regions_processed'],
                'avg_regions_per_document': self.stats['regions_processed'] / self.stats['total_filtered']
            }
        
        return stats
    
    def optimize_thresholds(
        self, 
        validation_data: List[Tuple[OCRResult, float]], 
        target_acceptance_rate: float = 0.8
    ) -> Dict[str, float]:
        """
        Optimize thresholds based on validation data
        
        Args:
            validation_data: List of (OCRResult, ground_truth_accuracy) pairs
            target_acceptance_rate: Target acceptance rate
            
        Returns:
            Dictionary of optimized thresholds
        """
        
        if not validation_data:
            return {
                'base_threshold': self.base_threshold,
                'character_threshold': self.character_threshold,
                'word_threshold': self.word_threshold
            }
        
        best_thresholds = {}
        best_score = 0.0
        
        # Grid search over threshold values
        threshold_range = np.arange(0.1, 0.95, 0.05)
        
        for base_thresh in threshold_range:
            total_accuracy = 0.0
            total_acceptance = 0.0
            
            for ocr_result, ground_truth_accuracy in validation_data:
                # Temporarily set threshold
                old_threshold = self.base_threshold
                self.base_threshold = base_thresh
                
                # Filter with this threshold
                filter_result = self.filter_by_confidence(ocr_result, base_thresh, adaptive=False)
                
                # Calculate metrics
                acceptance_rate = filter_result.acceptance_rate
                estimated_accuracy = filter_result.overall_confidence
                
                # Score combines accuracy and meeting target acceptance rate
                acceptance_penalty = abs(acceptance_rate - target_acceptance_rate)
                score = estimated_accuracy * ground_truth_accuracy - acceptance_penalty
                
                total_accuracy += score
                total_acceptance += acceptance_rate
                
                # Restore threshold
                self.base_threshold = old_threshold
            
            avg_score = total_accuracy / len(validation_data)
            avg_acceptance = total_acceptance / len(validation_data)
            
            # Penalize if too far from target acceptance rate
            final_score = avg_score - abs(avg_acceptance - target_acceptance_rate)
            
            if final_score > best_score:
                best_score = final_score
                best_thresholds = {
                    'base_threshold': base_thresh,
                    'character_threshold': base_thresh * 0.7,
                    'word_threshold': base_thresh * 0.8
                }
        
        self.logger.info(f"Threshold optimization completed: {best_thresholds}")
        return best_thresholds
    
    def export_filter_analysis(
        self, 
        filter_result: FilterResult, 
        output_path: Union[str, Path]
    ) -> bool:
        """Export detailed filter analysis to JSON"""
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            analysis_data = {
                'summary': {
                    'total_regions': len(filter_result.original_regions),
                    'accepted_regions': len(filter_result.filtered_regions),
                    'rejected_regions': len(filter_result.rejected_regions),
                    'acceptance_rate': filter_result.acceptance_rate,
                    'overall_confidence': filter_result.overall_confidence,
                    'processing_time': filter_result.processing_time
                },
                'statistics': filter_result.filter_statistics,
                'rejected_samples': [
                    {
                        'text': region.text[:100] + '...' if len(region.text) > 100 else region.text,
                        'confidence': region.confidence,
                        'length': len(region.text) if region.text else 0
                    }
                    for region in filter_result.rejected_regions[:20]  # First 20 samples
                ],
                'accepted_samples': [
                    {
                        'text': region.text[:100] + '...' if len(region.text) > 100 else region.text,
                        'confidence': region.confidence,
                        'length': len(region.text) if region.text else 0
                    }
                    for region in filter_result.filtered_regions[:20]  # First 20 samples
                ],
                'configuration': {
                    'base_threshold': self.base_threshold,
                    'adaptive_enabled': self.adaptive_thresholds,
                    'outlier_detection': self.enable_outlier_detection,
                    'consistency_check': self.enable_consistency_check
                }
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(analysis_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Filter analysis exported to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting filter analysis: {e}")
            return False