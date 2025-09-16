"""
Advanced OCR System - Text Processing Orchestrator
ONLY JOB: Orchestrate all postprocessing steps
DEPENDENCIES: result_fusion.py, layout_reconstructor.py, confidence_analyzer.py, text_utils.py
USED BY: core.py ONLY
"""

from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum

from ..results import OCRResult, ConfidenceMetrics
from ..config import OCRConfig
from ..utils.text_utils import TextCleaner, TextValidator
from .result_fusion import ResultFusion, FusionMetrics
from .layout_reconstructor import LayoutReconstructor, ReconstructionMetrics
from .confidence_analyzer import ConfidenceAnalyzer


class ProcessingStage(Enum):
    """Stages of text postprocessing"""
    FUSION = "fusion"
    CLEANING = "cleaning"
    LAYOUT_RECONSTRUCTION = "layout_reconstruction"
    CONFIDENCE_ANALYSIS = "confidence_analysis"
    VALIDATION = "validation"


@dataclass
class ProcessingMetrics:
    """Comprehensive metrics from text processing"""
    fusion_metrics: Optional[FusionMetrics]
    reconstruction_metrics: Optional[ReconstructionMetrics]
    cleaning_metrics: Dict[str, any]
    final_confidence: ConfidenceMetrics
    processing_time: Dict[str, float]
    stages_completed: List[ProcessingStage]


class TextProcessor:
    """
    ONLY RESPONSIBILITY: Orchestrate all postprocessing steps
    
    Receives raw OCRResult(s) from core.py and orchestrates the complete postprocessing
    pipeline. Returns final polished OCRResult with enhanced confidence metrics.
    Does NOT perform engine coordination, image preprocessing, or engine selection.
    """
    
    def __init__(self, config: OCRConfig):
        self.config = config
        
        # Initialize all postprocessing components
        self.confidence_analyzer = ConfidenceAnalyzer(config)
        self.result_fusion = ResultFusion(config, self.confidence_analyzer)
        self.layout_reconstructor = LayoutReconstructor(config)
        self.text_cleaner = TextCleaner()
        self.text_validator = TextValidator()
        
        # Processing pipeline configuration
        self.enable_fusion = True
        self.enable_cleaning = config.postprocessing.enable_text_cleaning
        self.enable_layout_reconstruction = config.postprocessing.enable_layout_reconstruction
        self.enable_confidence_enhancement = True
        self.enable_validation = config.postprocessing.enable_validation
        
        # Performance tracking
        self.processing_times = {}
    
    def process_results(self, results: List[OCRResult]) -> Tuple[OCRResult, ProcessingMetrics]:
        """
        Process OCR results through complete postprocessing pipeline
        
        Args:
            results: List of raw OCR results from engines
            
        Returns:
            Tuple of (final_processed_result, processing_metrics)
        """
        if not results:
            raise ValueError("No OCR results provided for processing")
        
        import time
        
        stages_completed = []
        fusion_metrics = None
        reconstruction_metrics = None
        cleaning_metrics = {}
        
        # Stage 1: Result Fusion (if multiple results)
        start_time = time.time()
        if len(results) > 1 and self.enable_fusion:
            current_result, fusion_metrics = self.result_fusion.fuse_results(results)
            stages_completed.append(ProcessingStage.FUSION)
        else:
            # Single result - just use it directly
            current_result = results[0]
            if len(results) == 1:
                # Still analyze confidence for single result
                enhanced_confidence = self.confidence_analyzer.analyze_single_result(current_result)
                current_result.confidence_metrics = enhanced_confidence
        
        self.processing_times['fusion'] = time.time() - start_time
        
        # Stage 2: Text Cleaning
        start_time = time.time()
        if self.enable_cleaning:
            current_result, cleaning_metrics = self._perform_text_cleaning(current_result)
            stages_completed.append(ProcessingStage.CLEANING)
        
        self.processing_times['cleaning'] = time.time() - start_time
        
        # Stage 3: Layout Reconstruction
        start_time = time.time()
        if self.enable_layout_reconstruction:
            current_result, reconstruction_metrics = self.layout_reconstructor.reconstruct_layout(current_result)
            stages_completed.append(ProcessingStage.LAYOUT_RECONSTRUCTION)
        
        self.processing_times['layout_reconstruction'] = time.time() - start_time
        
        # Stage 4: Final Confidence Analysis
        start_time = time.time()
        if self.enable_confidence_enhancement:
            final_confidence = self._enhance_final_confidence(current_result, fusion_metrics)
            current_result.confidence_metrics = final_confidence
            current_result.confidence = final_confidence.overall
            stages_completed.append(ProcessingStage.CONFIDENCE_ANALYSIS)
        
        self.processing_times['confidence_analysis'] = time.time() - start_time
        
        # Stage 5: Validation
        start_time = time.time()
        if self.enable_validation:
            validation_results = self._validate_final_result(current_result)
            cleaning_metrics['validation'] = validation_results
            stages_completed.append(ProcessingStage.VALIDATION)
        
        self.processing_times['validation'] = time.time() - start_time
        
        # Compile comprehensive metrics
        processing_metrics = ProcessingMetrics(
            fusion_metrics=fusion_metrics,
            reconstruction_metrics=reconstruction_metrics,
            cleaning_metrics=cleaning_metrics,
            final_confidence=current_result.confidence_metrics or ConfidenceMetrics(overall=current_result.confidence),
            processing_time=self.processing_times.copy(),
            stages_completed=stages_completed
        )
        
        return current_result, processing_metrics
    
    def _perform_text_cleaning(self, result: OCRResult) -> Tuple[OCRResult, Dict[str, any]]:
        """Perform comprehensive text cleaning"""
        original_text = result.text
        cleaning_metrics = {
            'original_length': len(original_text),
            'operations_performed': []
        }
        
        # Clean the main text
        cleaned_text = self.text_cleaner.clean_ocr_text(original_text)
        
        # Track what was cleaned
        if cleaned_text != original_text:
            cleaning_metrics['operations_performed'].append('general_cleaning')
            cleaning_metrics['characters_changed'] = sum(
                1 for a, b in zip(original_text, cleaned_text) if a != b
            )
        
        # Normalize Unicode
        normalized_text = self.text_cleaner.normalize_unicode(cleaned_text)
        if normalized_text != cleaned_text:
            cleaning_metrics['operations_performed'].append('unicode_normalization')
        
        # Remove OCR artifacts
        artifact_free_text = self.text_cleaner.remove_ocr_artifacts(normalized_text)
        if artifact_free_text != normalized_text:
            cleaning_metrics['operations_performed'].append('artifact_removal')
        
        final_text = artifact_free_text
        
        # Update metrics
        cleaning_metrics['final_length'] = len(final_text)
        cleaning_metrics['length_change'] = len(final_text) - len(original_text)
        cleaning_metrics['cleaning_ratio'] = len(final_text) / len(original_text) if original_text else 1.0
        
        # Create cleaned result
        cleaned_result = OCRResult(
            text=final_text,
            confidence=result.confidence,
            regions=result.regions,
            engine_name=result.engine_name,
            confidence_metrics=result.confidence_metrics
        )
        
        # Clean text in hierarchical regions if they exist
        if result.regions:
            cleaned_result.regions = self._clean_hierarchical_regions(result.regions)
        
        return cleaned_result, cleaning_metrics
    
    def _clean_hierarchical_regions(self, regions):
        """Clean text in hierarchical regions recursively"""
        cleaned_regions = []
        
        for region in regions:
            if hasattr(region, 'text') and region.text:
                # Clean the region's text
                cleaned_text = self.text_cleaner.clean_ocr_text(region.text)
                
                # Create new region with cleaned text
                region_dict = region.__dict__.copy()
                region_dict['text'] = cleaned_text
                
                # Recursively clean child regions
                if hasattr(region, 'blocks') and region.blocks:
                    region_dict['blocks'] = self._clean_hierarchical_regions(region.blocks)
                elif hasattr(region, 'paragraphs') and region.paragraphs:
                    region_dict['paragraphs'] = self._clean_hierarchical_regions(region.paragraphs)
                elif hasattr(region, 'lines') and region.lines:
                    region_dict['lines'] = self._clean_hierarchical_regions(region.lines)
                elif hasattr(region, 'words') and region.words:
                    region_dict['words'] = self._clean_hierarchical_regions(region.words)
                
                # Create new region instance
                cleaned_region = type(region)(**region_dict)
                cleaned_regions.append(cleaned_region)
            else:
                cleaned_regions.append(region)
        
        return cleaned_regions
    
    def _enhance_final_confidence(self, result: OCRResult, 
                                 fusion_metrics: Optional[FusionMetrics]) -> ConfidenceMetrics:
        """Enhance confidence metrics with postprocessing context"""
        # Start with existing confidence or analyze fresh
        if result.confidence_metrics:
            base_confidence = result.confidence_metrics
        else:
            base_confidence = self.confidence_analyzer.analyze_single_result(result)
        
        # Enhancement factors based on postprocessing
        enhancement_factors = {}
        
        # Factor 1: Fusion quality (if applicable)
        if fusion_metrics:
            fusion_quality = fusion_metrics.agreement_ratio * fusion_metrics.fusion_confidence
            enhancement_factors['fusion_quality'] = fusion_quality
        else:
            enhancement_factors['fusion_quality'] = 1.0  # No fusion needed
        
        # Factor 2: Text quality after cleaning
        text_quality = self.text_validator.calculate_text_quality(result.text)
        enhancement_factors['text_quality'] = text_quality
        
        # Factor 3: Structural completeness
        if result.regions:
            structural_completeness = self._assess_structural_completeness(result.regions)
            enhancement_factors['structural_completeness'] = structural_completeness
        else:
            enhancement_factors['structural_completeness'] = 0.5  # No structure
        
        # Factor 4: Text validation score
        validation_score = self.text_validator.validate_text_content(result.text)
        enhancement_factors['validation_score'] = validation_score
        
        # Calculate enhanced overall confidence
        enhancement_weights = {
            'fusion_quality': 0.3,
            'text_quality': 0.3,
            'structural_completeness': 0.2,
            'validation_score': 0.2
        }
        
        weighted_enhancement = sum(
            factor * enhancement_weights[name] 
            for name, factor in enhancement_factors.items()
        )
        
        # Combine with base confidence (weighted average)
        enhanced_overall = (base_confidence.overall * 0.7 + weighted_enhancement * 0.3)
        enhanced_overall = min(1.0, max(0.0, enhanced_overall))
        
        # Create enhanced confidence metrics
        enhanced_factors = base_confidence.factors.copy()
        enhanced_factors.update(enhancement_factors)
        
        return ConfidenceMetrics(
            overall=enhanced_overall,
            word_level=base_confidence.word_level,
            char_level=base_confidence.char_level,
            spatial_consistency=base_confidence.spatial_consistency,
            dictionary_match=base_confidence.dictionary_match,
            engine_reliability=base_confidence.engine_reliability,
            factors=enhanced_factors
        )
    
    def _assess_structural_completeness(self, regions) -> float:
        """Assess how complete the hierarchical structure is"""
        structure_levels = {
            'pages': False,
            'blocks': False,
            'paragraphs': False,
            'lines': False,
            'words': False
        }
        
        # Check what levels are present in the hierarchy
        for region in regions:
            if hasattr(region, 'blocks'):  # Page
                structure_levels['pages'] = True
                for block in region.blocks:
                    structure_levels['blocks'] = True
                    if hasattr(block, 'paragraphs'):
                        for paragraph in block.paragraphs:
                            structure_levels['paragraphs'] = True
                            if hasattr(paragraph, 'lines'):
                                for line in paragraph.lines:
                                    structure_levels['lines'] = True
                                    if hasattr(line, 'words'):
                                        structure_levels['words'] = True
        
        # Calculate completeness score
        levels_present = sum(structure_levels.values())
        total_levels = len(structure_levels)
        
        return levels_present / total_levels
    
    def _validate_final_result(self, result: OCRResult) -> Dict[str, any]:
        """Validate the final processed result"""
        validation_results = {
            'text_valid': True,
            'structure_valid': True,
            'confidence_reasonable': True,
            'issues_found': []
        }
        
        # Text validation
        if not result.text or not result.text.strip():
            validation_results['text_valid'] = False
            validation_results['issues_found'].append('empty_text')
        
        # Text quality validation
        text_quality = self.text_validator.validate_text_content(result.text)
        if text_quality < 0.5:
            validation_results['text_valid'] = False
            validation_results['issues_found'].append('poor_text_quality')
        
        # Confidence validation
        if result.confidence < 0.1 or result.confidence > 1.0:
            validation_results['confidence_reasonable'] = False
            validation_results['issues_found'].append('invalid_confidence_range')
        
        # Structure validation
        if result.regions:
            hierarchy_validation = self.layout_reconstructor.validate_hierarchy(result.regions)
            if not hierarchy_validation.get('hierarchy_consistent', True):
                validation_results['structure_valid'] = False
                validation_results['issues_found'].append('inconsistent_hierarchy')
            
            if not hierarchy_validation.get('text_consistency', True):
                validation_results['structure_valid'] = False
                validation_results['issues_found'].append('text_hierarchy_mismatch')
            
            if not hierarchy_validation.get('bounding_boxes_valid', True):
                validation_results['structure_valid'] = False
                validation_results['issues_found'].append('invalid_bounding_boxes')
        
        # Overall validation
        validation_results['overall_valid'] = (
            validation_results['text_valid'] and 
            validation_results['structure_valid'] and 
            validation_results['confidence_reasonable']
        )
        
        return validation_results
    
    def get_processing_summary(self, metrics: ProcessingMetrics) -> Dict[str, any]:
        """
        Get a comprehensive summary of the processing pipeline
        
        Args:
            metrics: Metrics from processing pipeline
            
        Returns:
            Dictionary containing processing summary
        """
        summary = {
            'stages_completed': [stage.value for stage in metrics.stages_completed],
            'total_processing_time': sum(metrics.processing_time.values()),
            'stage_times': metrics.processing_time,
            'final_confidence': metrics.final_confidence.overall
        }
        
        # Add fusion summary if available
        if metrics.fusion_metrics:
            summary['fusion'] = self.result_fusion.get_fusion_summary(metrics.fusion_metrics)
        
        # Add layout summary if available
        if metrics.reconstruction_metrics:
            summary['layout'] = self.layout_reconstructor.get_layout_summary(metrics.reconstruction_metrics)
        
        # Add cleaning summary
        if metrics.cleaning_metrics:
            summary['cleaning'] = {
                'operations_performed': metrics.cleaning_metrics.get('operations_performed', []),
                'text_length_change': metrics.cleaning_metrics.get('length_change', 0),
                'cleaning_ratio': metrics.cleaning_metrics.get('cleaning_ratio', 1.0)
            }
        
        # Add confidence breakdown
        if metrics.final_confidence.factors:
            summary['confidence_factors'] = metrics.final_confidence.factors
        
        return summary
    
    def optimize_for_content_type(self, content_type: str):
        """
        Optimize processing pipeline for specific content types
        
        Args:
            content_type: Type of content ('handwritten', 'printed', 'mixed', 'form', 'table')
        """
        if content_type == 'handwritten':
            # More aggressive cleaning for handwritten text
            self.enable_cleaning = True
            # Handwritten text benefits less from strict layout reconstruction
            self.enable_layout_reconstruction = False
            
        elif content_type == 'printed':
            # Standard processing for printed text
            self.enable_cleaning = True
            self.enable_layout_reconstruction = True
            
        elif content_type == 'form':
            # Forms need careful layout reconstruction
            self.enable_layout_reconstruction = True
            self.enable_cleaning = True
            
        elif content_type == 'table':
            # Tables require precise layout reconstruction
            self.enable_layout_reconstruction = True
            self.enable_cleaning = False  # Avoid over-cleaning tabular data
            
        else:  # mixed or unknown
            # Conservative approach
            self.enable_cleaning = True
            self.enable_layout_reconstruction = True