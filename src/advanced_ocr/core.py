"""
Advanced OCR System - Core Pipeline Orchestrator
===============================================

ONLY JOB: Coordinate the entire pipeline flow
DEPENDENCIES: image_processor.py, engine_coordinator.py, text_processor.py, config.py, logger.py
USED BY: __init__.py ONLY

CORRECT PIPELINE ORCHESTRATION:
1. Receive raw image from __init__.py
2. Call image_processor.py → get (enhanced_image, text_regions, quality_metrics)
3. Call engine_coordinator.py → get raw OCRResult(s)
4. Call text_processor.py → get final OCRResult
5. Return final result to __init__.py

WHAT IT SHOULD NOT DO:
❌ Any direct image processing
❌ Any direct text extraction  
❌ Any direct postprocessing
✅ Pipeline orchestration ONLY
"""

import time
from typing import Union, List, Optional, Dict, Any
from pathlib import Path
import numpy as np
from PIL import Image

# Core dependencies according to pipeline
from .config import OCRConfig
from .results import OCRResult, ProcessingMetrics, BatchResult
from .utils.logger import OCRLogger, ProcessingStageTimer
from .utils.image_utils import ImageLoader

# Pipeline components - EXACT dependencies from plan
from .preprocessing.image_processor import ImageProcessor, PreprocessingResult
from .engines.engine_coordinator import EngineCoordinator, CoordinationResult
from .postprocessing.text_processor import TextProcessor


class OCRPipelineError(Exception):
    """Exception raised during OCR pipeline execution."""
    pass


class OCRCore:
    """
    Main OCR Pipeline Orchestrator
    
    Coordinates the three main pipeline stages:
    1. Preprocessing (image_processor.py)
    2. Engine Coordination (engine_coordinator.py) 
    3. Postprocessing (text_processor.py)
    """
    
    def __init__(self, config: Optional[OCRConfig] = None):
        """
        Initialize OCR pipeline orchestrator.
        
        Args:
            config: OCR configuration object
        """
        self.config = config or OCRConfig()
        self.logger = OCRLogger("ocr_core")
        
        # Initialize pipeline components according to plan
        self._initialize_pipeline_components()
        
        # Performance tracking
        self.processing_metrics = ProcessingMetrics()
        
        self.logger.info("OCR Core initialized with pipeline components")
    
    def _initialize_pipeline_components(self):
        """Initialize the three main pipeline components."""
        try:
            # Stage 1: Preprocessing orchestrator
            self.image_processor = ImageProcessor(self.config.preprocessing)
            
            # Stage 2: Engine coordinator  
            self.engine_coordinator = EngineCoordinator(self.config.engines)
            
            # Stage 3: Postprocessing orchestrator
            self.text_processor = TextProcessor(self.config.postprocessing)
            
            self.logger.info("All pipeline components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize pipeline components: {str(e)}")
            raise OCRPipelineError(f"Pipeline initialization failed: {str(e)}")
    
    def extract_text(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
        config_override: Optional[Dict[str, Any]] = None
    ) -> OCRResult:
        """
        Main OCR extraction pipeline orchestration.
        
        Args:
            image_input: Input image (path, PIL Image, or numpy array)
            config_override: Optional configuration overrides
            
        Returns:
            OCRResult: Final processed OCR result
            
        Raises:
            OCRPipelineError: If any stage of the pipeline fails
        """
        pipeline_start = time.time()
        
        try:
            # Apply config overrides if provided
            effective_config = self._apply_config_overrides(config_override)
            
            with ProcessingStageTimer("total_pipeline") as total_timer:
                
                # STAGE 0: Load and validate input image
                raw_image = self._load_input_image(image_input)
                
                # STAGE 1: Preprocessing (image_processor.py)
                preprocessing_result = self._execute_preprocessing_stage(raw_image)
                
                # STAGE 2: Engine Coordination (engine_coordinator.py)  
                coordination_result = self._execute_engine_coordination_stage(
                    preprocessing_result
                )
                
                # STAGE 3: Postprocessing (text_processor.py)
                final_result = self._execute_postprocessing_stage(
                    coordination_result.ocr_results,
                    preprocessing_result
                )
                
                # Add pipeline metrics to result
                self._add_pipeline_metrics(final_result, total_timer.elapsed, coordination_result)
                
                self.logger.info(
                    f"OCR pipeline completed successfully in {total_timer.elapsed:.3f}s"
                )
                
                return final_result
                
        except Exception as e:
            self.logger.error(f"OCR pipeline failed: {str(e)}")
            raise OCRPipelineError(f"Pipeline execution failed: {str(e)}")
    
    def _load_input_image(self, image_input: Union[str, Path, Image.Image, np.ndarray]) -> Image.Image:
        """
        Load and validate input image using image_utils.
        
        Args:
            image_input: Input image in various formats
            
        Returns:
            PIL.Image: Loaded and validated image
        """
        with ProcessingStageTimer("image_loading") as timer:
            try:
                # Use ImageLoader from image_utils for consistent loading
                image_loader = ImageLoader(self.config.image_loading)
                raw_image = image_loader.load(image_input)
                
                self.logger.debug(f"Image loaded in {timer.elapsed:.3f}s: {raw_image.size}")
                return raw_image
                
            except Exception as e:
                self.logger.error(f"Failed to load image: {str(e)}")
                raise OCRPipelineError(f"Image loading failed: {str(e)}")
    
    def _execute_preprocessing_stage(self, raw_image: Image.Image) -> PreprocessingResult:
        """
        Execute Stage 1: Preprocessing via image_processor.py
        
        Args:
            raw_image: Raw input image
            
        Returns:
            PreprocessingResult: Enhanced image + text regions + quality metrics
        """
        with ProcessingStageTimer("preprocessing") as timer:
            try:
                # Call image_processor.py for complete preprocessing orchestration
                preprocessing_result = self.image_processor.process(raw_image)
                
                self.logger.info(
                    f"Preprocessing completed in {timer.elapsed:.3f}s - "
                    f"Found {len(preprocessing_result.text_regions)} text regions, "
                    f"Quality score: {preprocessing_result.quality_metrics.overall_score:.3f}"
                )
                
                return preprocessing_result
                
            except Exception as e:
                self.logger.error(f"Preprocessing stage failed: {str(e)}")
                raise OCRPipelineError(f"Preprocessing failed: {str(e)}")
    
    def _execute_engine_coordination_stage(
        self, 
        preprocessing_result: PreprocessingResult
    ) -> CoordinationResult:
        """
        Execute Stage 2: Engine Coordination via engine_coordinator.py
        
        Args:
            preprocessing_result: Result from preprocessing stage
            
        Returns:
            CoordinationResult: Raw OCR results from selected engines
        """
        with ProcessingStageTimer("engine_coordination") as timer:
            try:
                # Call engine_coordinator.py for intelligent engine selection and execution
                coordination_result = self.engine_coordinator.coordinate(
                    preprocessing_result.enhanced_image,
                    preprocessing_result.text_regions,
                    preprocessing_result.quality_metrics
                )
                
                engines_used = [engine.name for engine in coordination_result.engines_used]
                self.logger.info(
                    f"Engine coordination completed in {timer.elapsed:.3f}s - "
                    f"Used engines: {engines_used}, "
                    f"Results count: {len(coordination_result.ocr_results)}"
                )
                
                return coordination_result
                
            except Exception as e:
                self.logger.error(f"Engine coordination stage failed: {str(e)}")
                raise OCRPipelineError(f"Engine coordination failed: {str(e)}")
    
    def _execute_postprocessing_stage(
        self, 
        raw_ocr_results: List[OCRResult],
        preprocessing_result: PreprocessingResult
    ) -> OCRResult:
        """
        Execute Stage 3: Postprocessing via text_processor.py
        
        Args:
            raw_ocr_results: Raw results from engines
            preprocessing_result: Original preprocessing result for context
            
        Returns:
            OCRResult: Final processed and polished result
        """
        with ProcessingStageTimer("postprocessing") as timer:
            try:
                # Call text_processor.py for complete postprocessing orchestration
                final_result = self.text_processor.process(
                    raw_ocr_results,
                    preprocessing_result.quality_metrics
                )
                
                self.logger.info(
                    f"Postprocessing completed in {timer.elapsed:.3f}s - "
                    f"Final text length: {len(final_result.text)} chars, "
                    f"Final confidence: {final_result.confidence:.3f}"
                )
                
                return final_result
                
            except Exception as e:
                self.logger.error(f"Postprocessing stage failed: {str(e)}")
                raise OCRPipelineError(f"Postprocessing failed: {str(e)}")
    
    def _apply_config_overrides(
        self, 
        config_override: Optional[Dict[str, Any]]
    ) -> OCRConfig:
        """
        Apply configuration overrides to base config.
        
        Args:
            config_override: Dictionary of configuration overrides
            
        Returns:
            OCRConfig: Effective configuration with overrides applied
        """
        if not config_override:
            return self.config
            
        # Create a copy of the base config and apply overrides
        effective_config = self.config.copy()
        
        for key, value in config_override.items():
            if hasattr(effective_config, key):
                setattr(effective_config, key, value)
            else:
                self.logger.warning(f"Unknown config override: {key}")
        
        return effective_config
    
    def _add_pipeline_metrics(
        self,
        result: OCRResult,
        total_time: float,
        coordination_result: CoordinationResult
    ):
        """
        Add comprehensive pipeline metrics to the final result.
        
        Args:
            result: Final OCR result to enhance with metrics
            total_time: Total pipeline execution time
            coordination_result: Engine coordination result with engine metrics
        """
        # Update processing metrics
        result.processing_metrics.total_time = total_time
        result.processing_metrics.engines_used = [
            engine.name for engine in coordination_result.engines_used
        ]
        
        # Add engine-specific metrics
        for engine_result in coordination_result.ocr_results:
            if hasattr(engine_result, 'processing_metrics'):
                result.processing_metrics.engine_times[engine_result.engine_name] = \
                    engine_result.processing_metrics.extraction_time
    
    def batch_extract(
        self,
        image_inputs: List[Union[str, Path, Image.Image, np.ndarray]],
        config_override: Optional[Dict[str, Any]] = None
    ) -> BatchResult:
        """
        Process multiple images through the OCR pipeline.
        
        Args:
            image_inputs: List of input images
            config_override: Optional configuration overrides
            
        Returns:
            BatchResult: Container with all OCR results and batch metrics
        """
        batch_start = time.time()
        results = []
        failed_images = []
        
        self.logger.info(f"Starting batch processing of {len(image_inputs)} images")
        
        for i, image_input in enumerate(image_inputs):
            try:
                self.logger.debug(f"Processing image {i+1}/{len(image_inputs)}")
                result = self.extract_text(image_input, config_override)
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Failed to process image {i+1}: {str(e)}")
                failed_images.append((i, str(image_input), str(e)))
        
        batch_time = time.time() - batch_start
        
        batch_result = BatchResult(
            results=results,
            failed_images=failed_images,
            total_images=len(image_inputs),
            successful_images=len(results),
            total_processing_time=batch_time,
            average_time_per_image=batch_time / len(image_inputs) if image_inputs else 0
        )
        
        self.logger.info(
            f"Batch processing completed: {len(results)}/{len(image_inputs)} successful "
            f"in {batch_time:.3f}s"
        )
        
        return batch_result
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """
        Get current pipeline status and component health.
        
        Returns:
            Dict containing pipeline status information
        """
        return {
            "pipeline_initialized": all([
                hasattr(self, 'image_processor'),
                hasattr(self, 'engine_coordinator'),
                hasattr(self, 'text_processor')
            ]),
            "config": {
                "preprocessing_enabled": self.config.preprocessing.enabled,
                "engines_available": len(self.config.engines.available_engines),
                "postprocessing_enabled": self.config.postprocessing.enabled
            },
            "components": {
                "image_processor": getattr(self, 'image_processor', None) is not None,
                "engine_coordinator": getattr(self, 'engine_coordinator', None) is not None,
                "text_processor": getattr(self, 'text_processor', None) is not None
            }
        }