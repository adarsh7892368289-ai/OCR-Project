# src/postprocessing/postprocessing_pipeline.py

import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import logging

from ..core.base_engine import OCRResult, DocumentResult, DocumentStructure, TextRegion
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
    # Changed from OCRResult to DocumentResult for consistency
    original_document_result: DocumentResult 
    corrected_results: List[OCRResult] = field(default_factory=list)
    filtered_results: List[OCRResult] = field(default_factory=list)
    layout_analysis: Optional[DocumentStructure] = None 
    formatted_results: Dict[str, Any] = field(default_factory=dict)
    
    processing_times: Dict[str, float] = field(default_factory=dict)
    pipeline_stats: Dict[str, Any] = field(default_factory=dict)
    total_processing_time: float = 0.0
    success: bool = True
    error_message: Optional[str] = None

class PostProcessingPipeline:
    """
    Comprehensive post-processing pipeline orchestrator
    """
    
    def __init__(self, config_path: Optional[str] = None):
        # Initialize logger FIRST
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        
        # 1. Determine config path
        self.config_path = config_path or "data/configs/postprocessing_config.yaml"
        
        # 2. Initialize and load the configuration manager
        self.config_manager = ConfigManager(self.config_path)
        
        # Load the configuration file into the ConfigManager instance
        try:
            self.config_manager.load_config(self.config_path)
            # Try to get the postprocessing_pipeline section
            pipeline_config_dict = self.config_manager.get('postprocessing_pipeline', {})

            if not pipeline_config_dict:
                self.logger.warning("Could not find postprocessing_pipeline section in config, using defaults")
                pipeline_config_dict = {}
            
        except Exception as e:
            self.logger.error(f"Error loading pipeline config: {e}")
            pipeline_config_dict = {}
        
        # 3. Create PipelineConfig dataclass from the loaded dictionary or defaults
        self.config = PipelineConfig(**pipeline_config_dict)
        
        # 4. Initialize components
        self._init_components()
        
        # 5. Statistics
        self.processing_stats = {
            'total_processed': 0,
            'successful_processing': 0,
            'failed_processing': 0,
            'avg_processing_time': 0.0
        }
        
        self.logger.info("Post-processing pipeline initialized")
    
    def _load_pipeline_config(self) -> PipelineConfig:
        """Load pipeline configuration"""
        try:
            # Load the config file first
            if hasattr(self, 'config_path') and self.config_path:
                self.config_manager.load_config(self.config_path)
            
            # Try to get the postprocessing_pipeline section using dot notation
            pipeline_config = self.config_manager.get('postprocessing_pipeline', {})
            
            # If that doesn't work, try getting the full config and extracting the section
            if not pipeline_config:
                full_config = self.config_manager.get_config()
                pipeline_config = full_config.get('postprocessing_pipeline', {})
            
            # If still empty, log warning and use empty dict (defaults will be used)
            if not pipeline_config:
                if hasattr(self, 'logger'):
                    self.logger.warning("Could not find postprocessing_pipeline section in config, using defaults")
                pipeline_config = {}
            
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.error(f"Error loading pipeline config: {e}")
            else:
                print(f"Error loading pipeline config: {e}")
            pipeline_config = {}
        
        # Create and return PipelineConfig object with loaded values or defaults
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
        document_result: DocumentResult,
        domain: Optional[str] = None,
        output_formats: Optional[List[str]] = None,
        custom_config: Optional[Dict[str, Any]] = None
    ) -> PipelineResult:
        """
        Process OCR result through a complete post-processing pipeline.

        Args:
            document_result: The complete DocumentResult object from the engine manager.
            domain: Document domain (invoice, receipt, etc.)
            output_formats: Desired output formats
            custom_config: Custom configuration overrides

        Returns:
            PipelineResult with all processing stages applied.
        """
        start_time = time.time()
        pipeline_result = PipelineResult(original_document_result=document_result)
        processing_times = {}

        try:
            self.logger.info("Starting post-processing pipeline")

            # Apply custom configuration if provided
            if custom_config:
                self._apply_custom_config(custom_config)

            current_results = document_result.results

            # Stage 1: Text Correction
            corrected_results = current_results
            if self.config.enable_text_correction and hasattr(self, 'text_corrector'):
                stage_start = time.time()
                corrected_results = [
                    self._apply_text_correction(ocr_r) for ocr_r in current_results
                ]
                pipeline_result.corrected_results = corrected_results
                processing_times['text_correction'] = time.time() - stage_start
                self.logger.info(f"Text correction completed in {processing_times['text_correction']:.3f}s")

            # Stage 2: Confidence Filtering
            filtered_results = corrected_results
            if self.config.enable_confidence_filtering and hasattr(self, 'confidence_filter'):
                stage_start = time.time()
                # We need to filter based on the document_result's overall confidence
                overall_confidence = document_result.confidence_score
                filtered_results = self._apply_confidence_filtering(corrected_results, overall_confidence)
                pipeline_result.filtered_results = filtered_results
                processing_times['confidence_filtering'] = time.time() - stage_start
                self.logger.info(f"Confidence filtering completed in {processing_times['confidence_filtering']:.3f}s")

            # Stage 3: Layout Analysis
            layout_analysis = document_result.document_structure
            if self.config.enable_layout_analysis and hasattr(self, 'layout_analyzer'):
                stage_start = time.time()
                # Pass the filtered results to the layout analyzer
                layout_analysis = self._apply_layout_analysis(filtered_results, domain)
                pipeline_result.layout_analysis = layout_analysis
                processing_times['layout_analysis'] = time.time() - stage_start
                self.logger.info(f"Layout analysis completed in {processing_times['layout_analysis']:.3f}s")

            # Stage 4: Result Formatting
            formatted_results = {}
            if self.config.enable_result_formatting and hasattr(self, 'result_formatter'):
                stage_start = time.time()
                formatted_results = self._apply_result_formatting(
                    filtered_results,
                    layout_analysis,
                    domain,
                    output_formats
                )
                pipeline_result.formatted_results = formatted_results
                processing_times['result_formatting'] = time.time() - stage_start
                self.logger.info(f"Result formatting completed in {processing_times['result_formatting']:.3f}s")

            # Calculate total processing time
            total_time = time.time() - start_time
            pipeline_result.processing_times = processing_times
            pipeline_result.total_processing_time = total_time

            # Generate pipeline statistics
            pipeline_result.pipeline_stats = self._generate_pipeline_stats(
                document_result, pipeline_result, processing_times
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

    def process_results(self, raw_ocr_results: List[OCRResult], quality_metrics: Optional[Any] = None) -> Dict[str, Any]:
        """
        Legacy method to process raw OCR results for backward compatibility.
        This method wraps the new process method.
        """
        # Create a dummy DocumentResult from raw OCR results
        from ..core.base_engine import DocumentResult
        document_result = DocumentResult(
            pages=raw_ocr_results,
            confidence_score=sum([r.confidence for r in raw_ocr_results]) / len(raw_ocr_results) if raw_ocr_results else 0.0,
            processing_time=0.0
        )
        pipeline_result = self.process(document_result)
        return pipeline_result.formatted_results

    def _apply_custom_config(self, custom_config: Dict[str, Any]):
        """Apply custom configuration overrides"""
        for key, value in custom_config.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                self.logger.info(f"Applied custom config: {key} = {value}")

    def _apply_text_correction(self, ocr_result: OCRResult) -> OCRResult:
        """Apply text correction to a single OCRResult - handles unified object structure."""
        try:
            # Check if the object has a 'full_text' attribute (your DocumentResult fix)
            text_to_correct = getattr(ocr_result, 'full_text', getattr(ocr_result, 'text', ''))
            
            if not text_to_correct.strip():
                self.logger.warning("No text content found in OCR result")
                return ocr_result
            
            correction_result = self.text_corrector.correct_text(text_to_correct)
            corrected_text = correction_result.corrected_text
            
            # Create a new OCRResult with the corrected text, preserving other attributes
            if corrected_text != text_to_correct:
                return OCRResult(
                    text=corrected_text,
                    confidence=ocr_result.confidence, # Confidence could be recalculated here
                    bbox=ocr_result.bbox,
                    regions=ocr_result.regions,
                    level=ocr_result.level,
                    language=ocr_result.language,
                    text_type=ocr_result.text_type,
                    rotation_angle=ocr_result.rotation_angle,
                    metadata=ocr_result.metadata
                )
            return ocr_result
        except Exception as e:
            self.logger.error(f"Text correction failed for single result: {e}")
            return ocr_result

    def _apply_confidence_filtering(self, ocr_results: List[OCRResult], overall_confidence: float) -> List[OCRResult]:
        """Apply confidence filtering to a list of OCR results."""
        filtered_results = []
        for ocr_result in ocr_results:
            # Check individual OCRResult confidence
            if ocr_result.confidence >= self.config.confidence_threshold:
                filtered_results.append(ocr_result)
        return filtered_results
    
    def _apply_layout_analysis(self, ocr_results: List[OCRResult], domain: Optional[str]) -> DocumentStructure:
        """Apply layout analysis to a list of OCR results."""
        try:
            if not ocr_results:
                return DocumentStructure() # Return an empty structure

            # Combine all OCR results into a single OCRResult for layout analysis
            combined_text = " ".join([r.text for r in ocr_results if r.text])
            combined_regions = []
            for r in ocr_results:
                if r.regions:
                    combined_regions.extend(r.regions)

            # Create a combined OCRResult
            from ..core.base_engine import OCRResult
            combined_ocr = OCRResult(
                text=combined_text,
                regions=combined_regions,
                confidence=sum([r.confidence for r in ocr_results]) / len(ocr_results) if ocr_results else 0.0
            )

            result = self.layout_analyzer.analyze_layout(combined_ocr)
            # Ensure we return a DocumentStructure
            if isinstance(result, DocumentStructure):
                return result
            else:
                return DocumentStructure()
        except Exception as e:
            self.logger.error(f"Layout analysis failed: {e}")
            return DocumentStructure()
            
    def _apply_result_formatting(self, ocr_results: List[OCRResult], layout_analysis: Optional[DocumentStructure], domain: Optional[str], output_formats: Optional[List[str]]) -> Dict[str, Any]:
        """Apply result formatting to a list of OCR results."""
        formats_to_generate = output_formats or [self.config.default_output_format]
        formatted_results = {}

        # Aggregate text from all results for formatting
        full_text = " ".join([r.full_text for r in ocr_results if r.full_text])

        # Combine all OCR results into a single OCRResult for formatting
        combined_regions = []
        for r in ocr_results:
            if r.regions:
                combined_regions.extend(r.regions)

        combined_ocr = OCRResult(
            text=full_text,
            regions=combined_regions,
            confidence=sum([r.confidence for r in ocr_results]) / len(ocr_results) if ocr_results else 0.0
        )

        for format_name in formats_to_generate:
            try:
                # Convert string format to OutputFormat enum
                from .result_formatter import OutputFormat, StructureLevel
                output_format_enum = getattr(OutputFormat, format_name.upper(), OutputFormat.JSON)
                structure_level_enum = getattr(StructureLevel, self.config.structure_level.upper(), StructureLevel.DETAILED)

                formatted_content = self.result_formatter.format_result(
                    ocr_result=combined_ocr,
                    output_format=output_format_enum,
                    structure_level=structure_level_enum
                )
                formatted_results[format_name] = formatted_content
            except Exception as e:
                self.logger.error(f"Error formatting as {format_name}: {e}")
                formatted_results[format_name] = {
                    'content': full_text,
                    'format': format_name,
                    'error': str(e),
                    'confidence': 0.0 # Fallback confidence
                }
        return formatted_results

    def _generate_pipeline_stats(
        self, 
        original_result: DocumentResult, 
        pipeline_result: PipelineResult,
        processing_times: Dict[str, float]
    ) -> Dict[str, Any]:
        """Generate comprehensive pipeline statistics"""
        
        stats = {
            'original_stats': {
                'text_length': len(original_result.full_text) if original_result.full_text else 0,
                'region_count': len(original_result.results),
                'average_confidence': original_result.confidence_score,
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
        if pipeline_result.corrected_results:
            original_full_text = " ".join([r.full_text for r in original_result.results])
            corrected_full_text = " ".join([r.full_text for r in pipeline_result.corrected_results])
            stats['text_correction'] = {
                'corrections_made': self._count_text_differences(original_full_text, corrected_full_text),
                'confidence_improvement': (
                    sum(r.confidence for r in pipeline_result.corrected_results) / len(pipeline_result.corrected_results) - original_result.confidence_score
                ) if pipeline_result.corrected_results else 0
            }
        
        if pipeline_result.filtered_results:
            stats['confidence_filtering'] = {
                'regions_kept': len(pipeline_result.filtered_results),
                'regions_rejected': len(original_result.results) - len(pipeline_result.filtered_results),
                'acceptance_rate': len(pipeline_result.filtered_results) / len(original_result.results) if original_result.results else 0
            }
        
        if pipeline_result.layout_analysis:
            stats['layout_analysis'] = {
                'blocks_detected': len(pipeline_result.layout_analysis.paragraphs) if hasattr(pipeline_result.layout_analysis, 'paragraphs') else 0,
                'document_type': getattr(pipeline_result.layout_analysis, 'document_type', 'unknown'),
                'reading_order_length': len(getattr(pipeline_result.layout_analysis, 'reading_order', []))
            }
        
        if pipeline_result.formatted_results:
            stats['result_formatting'] = {
                'formats_generated': len(pipeline_result.formatted_results),
                'formats': list(pipeline_result.formatted_results.keys()),
                'total_output_size': sum(
                    len(str(result['content'])) if isinstance(result, dict) and 'content' in result else 0
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
                
                # Create a DocumentResult from the OCRResult for processing
                from ..core.base_engine import DocumentResult
                document_result = DocumentResult(
                    pages=[ocr_result],
                    confidence_score=ocr_result.confidence,
                    processing_time=0.0
                )

                pipeline_result = self.process(
                    document_result=document_result,
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
                error_document_result = DocumentResult(
                    pages=[ocr_result],
                    confidence_score=0.0,
                    processing_time=0.0
                )
                error_result = PipelineResult(
                    original_document_result=error_document_result,
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
                'text_corrector': {'initialized': hasattr(self, 'text_corrector')},
                'confidence_filter': {'initialized': hasattr(self, 'confidence_filter')},
                'layout_analyzer': {'initialized': hasattr(self, 'layout_analyzer')},
                'result_formatter': {'initialized': hasattr(self, 'result_formatter')}
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