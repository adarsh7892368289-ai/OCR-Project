"""
Post-processing Pipeline - Step 5 Integration
Orchestrates all post-processing components

File Location: src/postprocessing/postprocessing_pipeline.py

Features:
- Coordinates text correction, confidence filtering, layout analysis, and result formatting
- Configurable pipeline stages
- Performance monitoring
- Error handling and fallback mechanisms
"""

import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path
import logging

from ..core.base_engine import OCRResult, DocumentResult
from ..utils.config import ConfigManager
from ..utils.logger import get_logger

from .text_corrector import TextCorrector
from .confidence_filter import AdvancedConfidenceFilter
from .layout_analyzer import LayoutAnalyzer
from .result_formatter import EnhancedResultFormatter, OutputFormat, StructureLevel


@dataclass
class PipelineConfig:
    """Configuration for post-processing pipeline"""
    enable_text_correction: bool = True
    enable_confidence_filtering: bool = True
    enable_layout_analysis: bool = True
    enable_result_formatting: bool = True
    
    # Stage-specific settings
    correction_method: str = "auto"  # auto, spellcheck, context, ml
    confidence_threshold: float = 0.5
    adaptive_confidence: bool = True
    layout_detection_method: str = "hybrid"  # traditional, deep_learning, hybrid
    default_output_format: str = "json"
    structure_level: str = "enhanced"
    
    # Performance settings
    parallel_processing: bool = False
    max_workers: int = 4
    timeout_seconds: int = 300


@dataclass
class PipelineResult:
    """Result of post-processing pipeline"""
    original_result: OCRResult
    corrected_result: Optional[OCRResult] = None
    filtered_result: Optional[OCRResult] = None
    layout_analysis: Optional[Any] = None  # LayoutAnalysis type
    formatted_results: Dict[str, Any] = None
    
    processing_times: Dict[str, float] = None
    pipeline_stats: Dict[str, Any] = None
    total_processing_time: float = 0.0
    success: bool = True
    error_message: Optional[str] = None


class PostProcessingPipeline:
    """
    Comprehensive post-processing pipeline orchestrator
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_manager = ConfigManager(config_path)
        self.config = self._load_pipeline_config()
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize components
        self._init_components()
        
        # Statistics
        self.processing_stats = {
            'total_processed': 0,
            'successful_processing': 0,
            'failed_processing': 0,
            'avg_processing_time': 0.0
        }
        
        self.logger.info("Post-processing pipeline initialized")
    
    def _load_pipeline_config(self) -> PipelineConfig:
        """Load pipeline configuration"""
        pipeline_config = self.config_manager.get_section('postprocessing_pipeline', {})
        
        return PipelineConfig(
            enable_text_correction=pipeline_config.get('enable_text_correction', True),
            enable_confidence_filtering=pipeline_config.get('enable_confidence_filtering', True),
            enable_layout_analysis=pipeline_config.get('enable_layout_analysis', True),
            enable_result_formatting=pipeline_config.get('enable_result_formatting', True),
            correction_method=pipeline_config.get('correction_method', 'auto'),
            confidence_threshold=pipeline_config.get('confidence_threshold', 0.5),
            adaptive_confidence=pipeline_config.get('adaptive_confidence', True),
            layout_detection_method=pipeline_config.get('layout_detection_method', 'hybrid'),
            default_output_format=pipeline_config.get('default_output_format', 'json'),
            structure_level=pipeline_config.get('structure_level', 'enhanced'),
            parallel_processing=pipeline_config.get('parallel_processing', False),
            max_workers=pipeline_config.get('max_workers', 4),
            timeout_seconds=pipeline_config.get('timeout_seconds', 300)
        )
    
    def _init_components(self):
        """Initialize post-processing components"""
        try:
            # Text Corrector
            if self.config.enable_text_correction:
                self.text_corrector = TextCorrector()
                self.logger.info("Text corrector initialized")
            
            # Confidence Filter  
            if self.config.enable_confidence_filtering:
                self.confidence_filter = AdvancedConfidenceFilter()
                self.logger.info("Confidence filter initialized")
            
            # Layout Analyzer
            if self.config.enable_layout_analysis:
                self.layout_analyzer = LayoutAnalyzer()
                self.logger.info("Layout analyzer initialized")
            
            # Result Formatter
            if self.config.enable_result_formatting:
                self.result_formatter = EnhancedResultFormatter()
                self.logger.info("Result formatter initialized")
                
        except Exception as e:
            self.logger.error(f"Error initializing components: {e}")
            raise
    
    def process(
        self,
        ocr_result: OCRResult,
        domain: Optional[str] = None,
        output_formats: Optional[List[str]] = None,
        custom_config: Optional[Dict[str, Any]] = None
    ) -> PipelineResult:
        """
        Process OCR result through complete post-processing pipeline
        
        Args:
            ocr_result: Original OCR result
            domain: Document domain (invoice, receipt, etc.)
            output_formats: Desired output formats
            custom_config: Custom configuration overrides
            
        Returns:
            PipelineResult with all processing stages
        """
        start_time = time.time()
        pipeline_result = PipelineResult(original_result=ocr_result)
        processing_times = {}
        
        try:
            self.logger.info("Starting post-processing pipeline")
            
            # Apply custom configuration if provided
            if custom_config:
                self._apply_custom_config(custom_config)
            
            current_result = ocr_result
            
            # Stage 1: Text Correction
            if self.config.enable_text_correction:
                stage_start = time.time()
                try:
                    corrected_result = self._apply_text_correction(current_result, domain)
                    pipeline_result.corrected_result = corrected_result
                    current_result = corrected_result
                    processing_times['text_correction'] = time.time() - stage_start
                    self.logger.info(f"Text correction completed in {processing_times['text_correction']:.3f}s")
                except Exception as e:
                    self.logger.error(f"Text correction failed: {e}")
                    processing_times['text_correction'] = time.time() - stage_start
            
            # Stage 2: Confidence Filtering
            if self.config.enable_confidence_filtering:
                stage_start = time.time()
                try:
                    filter_result = self._apply_confidence_filtering(current_result)
                    # Create new OCR result with filtered regions
                    filtered_ocr_result = OCRResult(
                        text=' '.join([r.text for r in filter_result.filtered_regions]),
                        confidence=filter_result.overall_confidence,
                        regions=filter_result.filtered_regions,
                        processing_time=current_result.processing_time,
                        engine_name=current_result.engine_name,
                        metadata={**current_result.metadata, 'confidence_filtered': True}
                    )
                    pipeline_result.filtered_result = filtered_ocr_result
                    current_result = filtered_ocr_result
                    processing_times['confidence_filtering'] = time.time() - stage_start
                    self.logger.info(f"Confidence filtering completed in {processing_times['confidence_filtering']:.3f}s")
                except Exception as e:
                    self.logger.error(f"Confidence filtering failed: {e}")
                    processing_times['confidence_filtering'] = time.time() - stage_start
            
            # Stage 3: Layout Analysis
            if self.config.enable_layout_analysis:
                stage_start = time.time()
                try:
                    layout_analysis = self._apply_layout_analysis(current_result, domain)
                    pipeline_result.layout_analysis = layout_analysis
                    processing_times['layout_analysis'] = time.time() - stage_start
                    self.logger.info(f"Layout analysis completed in {processing_times['layout_analysis']:.3f}s")
                except Exception as e:
                    self.logger.error(f"Layout analysis failed: {e}")
                    processing_times['layout_analysis'] = time.time() - stage_start
            
            # Stage 4: Result Formatting
            if self.config.enable_result_formatting:
                stage_start = time.time()
                try:
                    formatted_results = self._apply_result_formatting(
                        current_result, 
                        pipeline_result.layout_analysis, 
                        domain, 
                        output_formats
                    )
                    pipeline_result.formatted_results = formatted_results
                    processing_times['result_formatting'] = time.time() - stage_start
                    self.logger.info(f"Result formatting completed in {processing_times['result_formatting']:.3f}s")
                except Exception as e:
                    self.logger.error(f"Result formatting failed: {e}")
                    processing_times['result_formatting'] = time.time() - stage_start
            
            # Calculate total processing time
            total_time = time.time() - start_time
            pipeline_result.processing_times = processing_times
            pipeline_result.total_processing_time = total_time
            
            # Generate pipeline statistics
            pipeline_result.pipeline_stats = self._generate_pipeline_stats(
                ocr_result, pipeline_result, processing_times
            )
            
            # Update global statistics
            self._update_statistics(pipeline_result, total_time)
            
            self.logger.info(f"Post-processing pipeline completed successfully in {total_time:.3f}s")
            
        except Exception as e:
            pipeline_result.success = False
            pipeline_result.error_message = str(e)
            pipeline_result.total_processing_time = time.time() - start_time
            self.logger.error(f"Post-processing pipeline failed: {e}")
            self.processing_stats['failed_processing'] += 1
        
        return pipeline_result
    
    def _apply_text_correction(self, ocr_result: OCRResult, domain: Optional[str]) -> OCRResult:
        """Apply text correction"""
        if self.config.correction_method == "auto":
            return self.text_corrector.correct_text(ocr_result, method="auto", domain=domain)
        else:
            return self.text_corrector.correct_text(ocr_result, method=self.config.correction_method, domain=domain)
    
    def _apply_confidence_filtering(self, ocr_result: OCRResult):
        """Apply confidence filtering"""
        return self.confidence_filter.filter_by_confidence(
            ocr_result,
            min_confidence=self.config.confidence_threshold,
            adaptive=self.config.adaptive_confidence
        )
    
    def _apply_layout_analysis(self, ocr_result: OCRResult, domain: Optional[str]):
        """Apply layout analysis"""
        return self.layout_analyzer.analyze_layout(
            ocr_result,
            method=self.config.layout_detection_method,
            domain=domain
        )
    
    def _apply_result_formatting(
        self, 
        ocr_result: OCRResult, 
        layout_analysis, 
        domain: Optional[str],
        output_formats: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Apply result formatting"""
        formatted_results = {}
        
        formats_to_generate = output_formats or [self.config.default_output_format]
        structure_level = StructureLevel(self.config.structure_level)
        
        for format_name in formats_to_generate:
            try:
                output_format = OutputFormat(format_name)
                result = self.result_formatter.format_result(
                    ocr_result=ocr_result,
                    layout_analysis=layout_analysis,
                    domain=domain,
                    output_format=output_format,
                    structure_level=structure_level
                )
                formatted_results[format_name] = result
            except ValueError:
                self.logger.warning(f"Unknown output format: {format_name}")
            except Exception as e:
                self.logger.error(f"Error formatting as {format_name}: {e}")
        
        return formatted_results
    
    def _apply_custom_config(self, custom_config: Dict[str, Any]):
        """Apply custom configuration overrides"""
        for key, value in custom_config.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                self.logger.debug(f"Applied custom config: {key} = {value}")
    
    def _generate_pipeline_stats(
        self, 
        original_result: OCRResult, 
        pipeline_result: PipelineResult,
        processing_times: Dict[str, float]
    ) -> Dict[str, Any]:
        """Generate comprehensive pipeline statistics"""
        
        stats = {
            'original_stats': {
                'text_length': len(original_result.text) if original_result.text else 0,
                'region_count': len(original_result.regions),
                'average_confidence': original_result.confidence,
                'processing_time': original_result.processing_time
            },
            'pipeline_performance': {
                'total_stages': sum(1 for stage in [
                    self.config.enable_text_correction,
                    self.config.enable_confidence_filtering, 
                    self.config.enable_layout_analysis,
                    self.config.enable_result_formatting
                ] if stage),
                'successful_stages': len([t for t in processing_times.values() if t > 0]),
                'stage_times': processing_times,
                'total_pipeline_time': pipeline_result.total_processing_time,
                'pipeline_overhead': pipeline_result.total_processing_time - sum(processing_times.values())
            }
        }
        
        # Add stage-specific stats
        if pipeline_result.corrected_result:
            stats['text_correction'] = {
                'corrections_made': self._count_text_differences(
                    original_result.text, pipeline_result.corrected_result.text
                ),
                'confidence_improvement': (
                    pipeline_result.corrected_result.confidence - original_result.confidence
                )
            }
        
        if pipeline_result.filtered_result:
            stats['confidence_filtering'] = {
                'regions_kept': len(pipeline_result.filtered_result.regions),
                'regions_rejected': len(original_result.regions) - len(pipeline_result.filtered_result.regions),
                'acceptance_rate': len(pipeline_result.filtered_result.regions) / len(original_result.regions) if original_result.regions else 0
            }
        
        if pipeline_result.layout_analysis:
            stats['layout_analysis'] = {
                'blocks_detected': len(pipeline_result.layout_analysis.blocks) if hasattr(pipeline_result.layout_analysis, 'blocks') else 0,
                'document_type': getattr(pipeline_result.layout_analysis, 'document_type', 'unknown'),
                'reading_order_length': len(getattr(pipeline_result.layout_analysis, 'reading_order', []))
            }
        
        if pipeline_result.formatted_results:
            stats['result_formatting'] = {
                'formats_generated': len(pipeline_result.formatted_results),
                'formats': list(pipeline_result.formatted_results.keys()),
                'total_output_size': sum(
                    len(str(result.content)) if hasattr(result, 'content') else 0 
                    for result in pipeline_result.formatted_results.values()
                )
            }
        
        return stats
    
    def _count_text_differences(self, original: str, corrected: str) -> int:
        """Count approximate differences between original and corrected text"""
        if not original or not corrected:
            return 0
        
        original_words = original.split()
        corrected_words = corrected.split()
        
        # Simple difference count - could be enhanced with edit distance
        return abs(len(original_words) - len(corrected_words))
    
    def _update_statistics(self, pipeline_result: PipelineResult, processing_time: float):
        """Update global processing statistics"""
        self.processing_stats['total_processed'] += 1
        
        if pipeline_result.success:
            self.processing_stats['successful_processing'] += 1
        else:
            self.processing_stats['failed_processing'] += 1
        
        # Update average processing time
        total_time = (self.processing_stats['avg_processing_time'] * 
                     (self.processing_stats['total_processed'] - 1) + processing_time)
        self.processing_stats['avg_processing_time'] = total_time / self.processing_stats['total_processed']
    
    def batch_process(
        self,
        ocr_results: List[OCRResult],
        domain: Optional[str] = None,
        output_formats: Optional[List[str]] = None,
        save_results: bool = False,
        output_dir: Optional[Union[str, Path]] = None
    ) -> List[PipelineResult]:
        """
        Process multiple OCR results in batch
        
        Args:
            ocr_results: List of OCR results to process
            domain: Document domain
            output_formats: Desired output formats
            save_results: Whether to save formatted results to files
            output_dir: Directory to save results
            
        Returns:
            List of PipelineResult objects
        """
        results = []
        
        self.logger.info(f"Starting batch processing of {len(ocr_results)} documents")
        
        for i, ocr_result in enumerate(ocr_results):
            try:
                self.logger.info(f"Processing document {i+1}/{len(ocr_results)}")
                
                pipeline_result = self.process(
                    ocr_result=ocr_result,
                    domain=domain,
                    output_formats=output_formats
                )
                
                # Save results if requested
                if save_results and output_dir and pipeline_result.formatted_results:
                    self._save_batch_results(pipeline_result, i, output_dir)
                
                results.append(pipeline_result)
                
            except Exception as e:
                self.logger.error(f"Error processing document {i+1}: {e}")
                # Create error result
                error_result = PipelineResult(
                    original_result=ocr_result,
                    success=False,
                    error_message=str(e)
                )
                results.append(error_result)
        
        self.logger.info(f"Batch processing completed: {len(results)} results")
        return results
    
    def _save_batch_results(self, pipeline_result: PipelineResult, index: int, output_dir: Union[str, Path]):
        """Save batch processing results to files"""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for format_name, formatted_result in pipeline_result.formatted_results.items():
            try:
                filename = f"document_{index:04d}.{format_name}"
                output_path = output_dir / filename
                
                if hasattr(formatted_result, 'save_to_file'):
                    formatted_result.save_to_file(output_path)
                else:
                    # Fallback for direct content
                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(str(formatted_result))
                
            except Exception as e:
                self.logger.error(f"Error saving {format_name} result for document {index}: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive pipeline statistics"""
        return {
            'processing_stats': self.processing_stats,
            'configuration': {
                'text_correction_enabled': self.config.enable_text_correction,
                'confidence_filtering_enabled': self.config.enable_confidence_filtering,
                'layout_analysis_enabled': self.config.enable_layout_analysis,
                'result_formatting_enabled': self.config.enable_result_formatting,
                'default_output_format': self.config.default_output_format,
                'structure_level': self.config.structure_level
            },
            'component_stats': {
                'text_corrector': getattr(self, 'text_corrector', {}).get_statistics() if hasattr(self, 'text_corrector') else {},
                'confidence_filter': getattr(self, 'confidence_filter', {}).get_statistics() if hasattr(self, 'confidence_filter') else {},
                'layout_analyzer': getattr(self, 'layout_analyzer', {}).get_statistics() if hasattr(self, 'layout_analyzer') else {},
                'result_formatter': getattr(self, 'result_formatter', {}).get_statistics() if hasattr(self, 'result_formatter') else {}
            }
        }
    
    def export_pipeline_report(self, output_path: Union[str, Path]) -> bool:
        """Export comprehensive pipeline performance report"""
        try:
            import json
            
            report_data = {
                'pipeline_statistics': self.get_statistics(),
                'configuration': vars(self.config),
                'export_timestamp': time.time()
            }
            
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            self.logger.info(f"Pipeline report exported to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting pipeline report: {e}")
            return False