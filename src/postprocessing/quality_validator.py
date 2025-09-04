"""
Quality Validator - Missing component for OCR quality validation
This provides quality assessment functionality needed by other post-processing components
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
import logging
from enum import Enum
import re
import statistics

from ..core.base_engine import OCRResult, TextRegion, BoundingBox
from ..utils.logger import get_logger


class QualityMetric(Enum):
    """Types of quality metrics"""
    CONFIDENCE = "confidence"
    TEXT_COHERENCE = "text_coherence"
    LAYOUT_CONSISTENCY = "layout_consistency"
    CHARACTER_ACCURACY = "character_accuracy"
    WORD_ACCURACY = "word_accuracy"
    OVERALL = "overall"


class ValidationLevel(Enum):
    """Validation thoroughness levels"""
    BASIC = "basic"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"


@dataclass
class QualityReport:
    """Quality assessment report"""
    overall_score: float
    metric_scores: Dict[str, float] = field(default_factory=dict)
    issues_found: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    validation_level: ValidationLevel = ValidationLevel.STANDARD
    processing_time: float = 0.0
    
    @property
    def quality_grade(self) -> str:
        """Convert score to letter grade"""
        if self.overall_score >= 0.9:
            return "A"
        elif self.overall_score >= 0.8:
            return "B"
        elif self.overall_score >= 0.7:
            return "C"
        elif self.overall_score >= 0.6:
            return "D"
        else:
            return "F"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'overall_score': self.overall_score,
            'quality_grade': self.quality_grade,
            'metric_scores': self.metric_scores,
            'issues_found': self.issues_found,
            'recommendations': self.recommendations,
            'validation_level': self.validation_level.value,
            'processing_time': self.processing_time
        }


class QualityValidator:
    """
    Basic quality validator for OCR results
    Minimal implementation to satisfy import dependencies
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        
        # Quality thresholds
        self.confidence_threshold = self.config.get('confidence_threshold', 0.7)
        self.min_text_length = self.config.get('min_text_length', 5)
        self.max_special_char_ratio = self.config.get('max_special_char_ratio', 0.3)
        
    def validate_quality(
        self,
        ocr_result: OCRResult,
        validation_level: ValidationLevel = ValidationLevel.STANDARD
    ) -> QualityReport:
        """
        Validate OCR result quality
        Basic implementation - can be enhanced as needed
        """
        try:
            import time
            start_time = time.time()
            
            metric_scores = {}
            issues_found = []
            recommendations = []
            
            # Basic confidence check
            confidence_score = ocr_result.overall_confidence
            metric_scores[QualityMetric.CONFIDENCE.value] = confidence_score
            
            if confidence_score < self.confidence_threshold:
                issues_found.append(f"Low overall confidence: {confidence_score:.2%}")
                recommendations.append("Consider re-processing with different OCR settings")
            
            # Text coherence check
            text_score = self._assess_text_coherence(ocr_result.full_text or "")
            metric_scores[QualityMetric.TEXT_COHERENCE.value] = text_score
            
            if text_score < 0.6:
                issues_found.append("Text appears fragmented or incoherent")
                recommendations.append("Check for image quality issues or preprocessing needs")
            
            # Layout consistency (basic)
            layout_score = self._assess_layout_consistency(ocr_result.regions or [])
            metric_scores[QualityMetric.LAYOUT_CONSISTENCY.value] = layout_score
            
            # Calculate overall score
            overall_score = statistics.mean(metric_scores.values()) if metric_scores else 0.0
            
            processing_time = time.time() - start_time
            
            report = QualityReport(
                overall_score=overall_score,
                metric_scores=metric_scores,
                issues_found=issues_found,
                recommendations=recommendations,
                validation_level=validation_level,
                processing_time=processing_time
            )
            
            self.logger.info(f"Quality validation completed: {report.quality_grade} ({overall_score:.2%})")
            return report
            
        except Exception as e:
            self.logger.error(f"Error in quality validation: {e}")
            return QualityReport(
                overall_score=0.0,
                issues_found=[f"Validation error: {str(e)}"],
                validation_level=validation_level
            )
    
    def _assess_text_coherence(self, text: str) -> float:
        """Basic text coherence assessment"""
        if not text or not text.strip():
            return 0.0
        
        score = 1.0
        text = text.strip()
        
        # Check minimum length
        if len(text) < self.min_text_length:
            score *= 0.5
        
        # Check special character ratio
        special_chars = len(re.findall(r'[^a-zA-Z0-9\s\-.,!?()"]', text))
        special_ratio = special_chars / len(text) if text else 0
        
        if special_ratio > self.max_special_char_ratio:
            score *= (1.0 - special_ratio)
        
        # Check for reasonable word distribution
        words = text.split()
        if words:
            avg_word_length = statistics.mean(len(word) for word in words)
            if avg_word_length < 2 or avg_word_length > 15:
                score *= 0.8
        
        return max(0.0, min(1.0, score))
    
    def _assess_layout_consistency(self, regions: List[Any]) -> float:
        """Basic layout consistency assessment"""
        if not regions:
            return 1.0
        
        # Simple check - if we have regions, assume reasonable layout
        confidence_scores = []
        for region in regions:
            if hasattr(region, 'confidence'):
                confidence_scores.append(region.confidence)
        
        if confidence_scores:
            # Use coefficient of variation as consistency measure
            mean_conf = statistics.mean(confidence_scores)
            if len(confidence_scores) > 1:
                std_conf = statistics.stdev(confidence_scores)
                cv = std_conf / mean_conf if mean_conf > 0 else 1.0
                consistency = max(0.0, 1.0 - cv)  # Lower variation = higher consistency
            else:
                consistency = 1.0
        else:
            consistency = 0.5  # Default if no confidence data
        
        return consistency
    
    def get_statistics(self) -> Dict[str, Any]:
        """Return basic statistics"""
        return {
            'validator_type': 'basic',
            'config': self.config,
            'supported_metrics': [metric.value for metric in QualityMetric],
            'validation_levels': [level.value for level in ValidationLevel]
        }


# Helper functions
def create_quality_report(
    overall_score: float,
    validation_level: ValidationLevel = ValidationLevel.STANDARD
) -> QualityReport:
    """Helper function to create quality reports"""
    return QualityReport(
        overall_score=overall_score,
        validation_level=validation_level
    )


def validate_ocr_result(
    ocr_result: OCRResult,
    validator: Optional[QualityValidator] = None
) -> QualityReport:
    """Helper function to validate OCR results"""
    if validator is None:
        validator = QualityValidator()
    
    return validator.validate_quality(ocr_result)

# Add this alias to quality_validator.py
QualityMetrics = QualityReport
ValidationResult = QualityReport
QualityLevel = ValidationLevel

__all__ = [
    'QualityValidator',
    'QualityReport',
    'QualityMetrics',
    'ValidationResult',
    'QualityLevel',       # Add this alias
    'QualityMetric',
    'ValidationLevel',
    'create_quality_report',
    'validate_ocr_result'
]