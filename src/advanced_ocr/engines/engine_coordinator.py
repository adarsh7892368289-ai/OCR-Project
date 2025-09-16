"""
Smart Engine Selection & Coordination Module for Advanced OCR System

Intelligently selects and coordinates OCR engines based on content analysis
and quality metrics for optimal text extraction performance.

Architecture:
- Uses content_classifier.py for intelligent content-based routing
- Coordinates multiple specialized engines based on content type
- Manages parallel/sequential engine execution
- Returns raw OCRResult(s) to core.py for postprocessing
- No result fusion - pure engine coordination

Author: Advanced OCR System
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum

from ..preprocessing.content_classifier import ContentClassifier, ContentClassification
from ..preprocessing.image_processor import PreprocessingResult  
from .base_engine import BaseOCREngine
from .tesseract_enhanced import TesseractEngine
from .paddleocr_optimized import PaddleOCREngine
from .easyocr_enhanced import EasyOCREngine
from .trocr_optimized import TrOCREngine
from ..results import OCRResult
from ..config import OCRConfig
from ..utils.logger import Logger


class EngineStrategy(Enum):
    """Engine selection strategies"""
    SINGLE_BEST = "single_best"        # Select single best engine
    MULTI_CONSENSUS = "multi_consensus" # Multiple engines for consensus
    HYBRID_ADAPTIVE = "hybrid_adaptive" # Adaptive based on confidence


@dataclass  
class EngineSelection:
    """Engine selection result"""
    primary_engines: List[str]      # Main engines to use
    fallback_engines: List[str]     # Backup engines if primary fails
    strategy: EngineStrategy        # Execution strategy
    confidence: float               # Selection confidence
    reasoning: str                  # Selection reasoning


@dataclass
class CoordinationResult:
    """Engine coordination result container"""
    results: List[OCRResult]        # OCR results from engines
    engine_selection: EngineSelection # Engine selection details
    execution_time: float           # Total coordination time
    engines_used: List[str]         # Actually executed engines
    performance_metrics: Dict       # Detailed performance metrics


class EngineCoordinator:
    """
    Intelligent OCR engine coordinator and orchestrator
    
    Responsibilities:
    - Analyze content type via content_classifier.py
    - Select optimal engine combination based on content + quality
    - Coordinate parallel/sequential engine execution
    - Handle engine failures and fallbacks
    - Return raw results to core.py (no fusion)
    """
    
    def __init__(self, config: OCRConfig):
        self.config = config
        self.logger = Logger(__name__)
        
        # Initialize content classifier
        self.content_classifier = ContentClassifier(config)
        
        # Initialize engines (lazy loading)
        self._engines: Dict[str, BaseOCREngine] = {}
        self._engine_configs = config.engines
        
        # Coordination parameters
        self.strategy = EngineStrategy(config.coordination.default_strategy)
        self.max_parallel_engines = config.coordination.max_parallel_engines
        self.engine_timeout = config.coordination.engine_timeout
        
        # Performance tracking
        self._coordination_stats = {
            'total_coordinations': 0,
            'engine_usage_counts': {},
            'avg_coordination_time': 0.0,
            'content_type_counts': {}
        }
    
    def coordinate_extraction(self, preprocessing_result: PreprocessingResult) -> CoordinationResult:
        """
        Coordinate OCR engines based on intelligent content analysis
        
        Args:
            preprocessing_result: Complete preprocessing results from image_processor.py
            
        Returns:
            CoordinationResult with raw OCR results and coordination metadata
        """
        start_time = self.logger.start_timer()
        
        try:
            # Step 1: Classify content type for intelligent routing
            self.logger.debug("Starting content classification")
            content_classification = self.content_classifier.classify_content(
                preprocessing_result.enhanced_image
            )
            
            # Step 2: Select optimal engine combination
            engine_selection = self._select_engines(
                content_classification, 
                preprocessing_result.quality_metrics
            )
            
            # Step 3: Execute selected engines
            ocr_results = self._execute_engines(
                engine_selection,
                preprocessing_result.enhanced_image,
                preprocessing_result.text_regions
            )
            
            coordination_time = self.logger.end_timer(start_time)
            
            # Step 4: Prepare coordination result
            result = CoordinationResult(
                results=ocr_results,
                engine_selection=engine_selection,
                execution_time=coordination_time,
                engines_used=[r.engine_name for r in ocr_results],
                performance_metrics=self._calculate_performance_metrics(ocr_results)
            )
            
            # Update stats
            self._update_coordination_stats(content_classification, result)
            
            self.logger.info(
                f"Engine coordination completed: {len(ocr_results)} results from "
                f"{engine_selection.primary_engines} engines ({coordination_time:.3f}s)"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Engine coordination failed: {e}")
            raise
    
    def _select_engines(self, classification: ContentClassification, quality_metrics) -> EngineSelection:
        """
        Intelligent engine selection based on content type and quality
        
        Content-Based Routing Strategy:
        - Handwritten: TrOCR + EasyOCR (specialized for handwriting)
        - Printed: PaddleOCR + Tesseract (optimized for printed text)  
        - Mixed: PaddleOCR + TrOCR (balanced approach)
        """
        
        content_type = classification.content_type
        confidence = max(classification.confidence_scores.values())
        
        # Engine selection logic based on content analysis
        if content_type == "handwritten":
            primary = ["trocr", "easyocr"]
            fallback = ["paddleocr"]
            strategy = EngineStrategy.MULTI_CONSENSUS if confidence > 0.7 else EngineStrategy.SINGLE_BEST
            reasoning = f"Handwritten content detected (conf: {confidence:.3f})"
            
        elif content_type == "printed":
            primary = ["paddleocr", "tesseract"]
            fallback = ["easyocr"]
            strategy = EngineStrategy.MULTI_CONSENSUS if confidence > 0.8 else EngineStrategy.SINGLE_BEST
            reasoning = f"Printed content detected (conf: {confidence:.3f})"
            
        else:  # mixed or uncertain
            primary = ["paddleocr", "trocr"]
            fallback = ["tesseract", "easyocr"]
            strategy = EngineStrategy.HYBRID_ADAPTIVE
            reasoning = f"Mixed/uncertain content (conf: {confidence:.3f})"
        
        # Quality-based adjustments
        if quality_metrics.overall_score < 0.5:  # Poor quality image
            # Add more engines for poor quality
            if "tesseract" not in primary and len(primary) < 3:
                primary.append("tesseract")
            strategy = EngineStrategy.MULTI_CONSENSUS  # Use consensus for poor quality
            reasoning += " + poor quality adjustment"
        
        return EngineSelection(
            primary_engines=primary,
            fallback_engines=fallback,
            strategy=strategy,
            confidence=confidence,
            reasoning=reasoning
        )
    
    def _execute_engines(self, selection: EngineSelection, image, text_regions) -> List[OCRResult]:
        """Execute selected engines with appropriate strategy"""
        
        if selection.strategy == EngineStrategy.SINGLE_BEST:
            return self._execute_single_engine(selection, image, text_regions)
        
        elif selection.strategy == EngineStrategy.MULTI_CONSENSUS:
            return self._execute_multiple_engines(selection, image, text_regions)
        
        else:  # HYBRID_ADAPTIVE
            return self._execute_adaptive_strategy(selection, image, text_regions)
    
    def _execute_single_engine(self, selection: EngineSelection, image, text_regions) -> List[OCRResult]:
        """Execute single best engine"""
        
        primary_engine = selection.primary_engines[0]
        
        try:
            engine = self._get_engine(primary_engine)
            result = engine.extract(image, text_regions)
            
            # Validate result quality
            if self._validate_result_quality(result):
                return [result]
            else:
                self.logger.warning(f"{primary_engine} produced poor quality result, trying fallback")
                # Try fallback engine
                if selection.fallback_engines:
                    fallback_engine = self._get_engine(selection.fallback_engines[0])
                    fallback_result = fallback_engine.extract(image, text_regions)
                    return [fallback_result]
                
        except Exception as e:
            self.logger.error(f"Primary engine {primary_engine} failed: {e}")
            # Try fallback
            if selection.fallback_engines:
                try:
                    fallback_engine = self._get_engine(selection.fallback_engines[0])
                    fallback_result = fallback_engine.extract(image, text_regions)
                    return [fallback_result]
                except Exception as fe:
                    self.logger.error(f"Fallback engine failed: {fe}")
        
        return []  # Return empty if all engines fail
    
    def _execute_multiple_engines(self, selection: EngineSelection, image, text_regions) -> List[OCRResult]:
        """Execute multiple engines in parallel for consensus"""
        
        engines_to_run = selection.primary_engines[:self.max_parallel_engines]
        results = []
        
        # Use ThreadPoolExecutor for parallel execution
        with ThreadPoolExecutor(max_workers=len(engines_to_run)) as executor:
            # Submit all engine tasks
            future_to_engine = {
                executor.submit(self._run_engine_safely, engine_name, image, text_regions): engine_name
                for engine_name in engines_to_run
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_engine, timeout=self.engine_timeout):
                engine_name = future_to_engine[future]
                try:
                    result = future.result()
                    if result and self._validate_result_quality(result):
                        results.append(result)
                        self.logger.debug(f"Engine {engine_name} completed successfully")
                    else:
                        self.logger.warning(f"Engine {engine_name} produced invalid result")
                        
                except Exception as e:
                    self.logger.error(f"Engine {engine_name} failed: {e}")
        
        # If no primary engines succeeded, try fallback
        if not results and selection.fallback_engines:
            self.logger.info("All primary engines failed, trying fallback")
            try:
                fallback_engine = self._get_engine(selection.fallback_engines[0])
                fallback_result = fallback_engine.extract(image, text_regions)
                if fallback_result:
                    results.append(fallback_result)
            except Exception as e:
                self.logger.error(f"Fallback engine failed: {e}")
        
        return results
    
    def _execute_adaptive_strategy(self, selection: EngineSelection, image, text_regions) -> List[OCRResult]:
        """Execute adaptive strategy - start with one, add more if needed"""
        
        results = []
        
        # Start with primary engine
        primary_engine = selection.primary_engines[0]
        try:
            engine = self._get_engine(primary_engine)
            primary_result = engine.extract(image, text_regions)
            
            if primary_result and self._validate_result_quality(primary_result):
                results.append(primary_result)
                
                # If primary result confidence is high enough, stop here
                if primary_result.confidence > 0.85:
                    return results
            
        except Exception as e:
            self.logger.error(f"Primary adaptive engine {primary_engine} failed: {e}")
        
        # If we need more engines, run additional ones
        additional_engines = selection.primary_engines[1:2]  # Run one more
        
        for engine_name in additional_engines:
            try:
                engine = self._get_engine(engine_name)
                result = engine.extract(image, text_regions)
                
                if result and self._validate_result_quality(result):
                    results.append(result)
                    
            except Exception as e:
                self.logger.error(f"Additional engine {engine_name} failed: {e}")
        
        return results
    
    def _run_engine_safely(self, engine_name: str, image, text_regions) -> Optional[OCRResult]:
        """Safely run a single engine with error handling"""
        try:
            engine = self._get_engine(engine_name)
            return engine.extract(image, text_regions)
        except Exception as e:
            self.logger.error(f"Engine {engine_name} execution failed: {e}")
            return None
    
    def _get_engine(self, engine_name: str) -> BaseOCREngine:
        """Get or initialize OCR engine (lazy loading)"""
        
        if engine_name not in self._engines:
            
            if engine_name == "tesseract":
                self._engines[engine_name] = TesseractEngine(self.config)
            elif engine_name == "paddleocr":
                self._engines[engine_name] = PaddleOCREngine(self.config)
            elif engine_name == "easyocr":
                self._engines[engine_name] = EasyOCREngine(self.config)
            elif engine_name == "trocr":
                self._engines[engine_name] = TrOCREngine(self.config)
            else:
                raise ValueError(f"Unknown engine: {engine_name}")
            
            self.logger.info(f"Initialized {engine_name} engine")
        
        return self._engines[engine_name]
    
    def _validate_result_quality(self, result: OCRResult) -> bool:
        """Validate OCR result quality"""
        if not result:
            return False
        
        # Basic quality checks
        if not result.text or len(result.text.strip()) == 0:
            return False
        
        if result.confidence < self.config.coordination.min_result_confidence:
            return False
        
        # Check for reasonable text length (avoid single character results)
        if len(result.text.strip()) < 2:
            return False
        
        return True
    
    def _calculate_performance_metrics(self, results: List[OCRResult]) -> Dict:
        """Calculate detailed performance metrics"""
        if not results:
            return {}
        
        return {
            'num_results': len(results),
            'avg_confidence': sum(r.confidence for r in results) / len(results),
            'total_text_length': sum(len(r.text) for r in results),
            'avg_processing_time': sum(r.processing_time for r in results) / len(results),
            'engines_used': list(set(r.engine_name for r in results))
        }
    
    def _update_coordination_stats(self, classification: ContentClassification, result: CoordinationResult):
        """Update coordination performance statistics"""
        stats = self._coordination_stats
        
        stats['total_coordinations'] += 1
        
        # Update content type counts
        content_type = classification.content_type
        if content_type not in stats['content_type_counts']:
            stats['content_type_counts'][content_type] = 0
        stats['content_type_counts'][content_type] += 1
        
        # Update engine usage counts
        for engine_name in result.engines_used:
            if engine_name not in stats['engine_usage_counts']:
                stats['engine_usage_counts'][engine_name] = 0
            stats['engine_usage_counts'][engine_name] += 1
        
        # Update average coordination time
        if stats['total_coordinations'] == 1:
            stats['avg_coordination_time'] = result.execution_time
        else:
            # Exponential moving average
            alpha = 0.1
            stats['avg_coordination_time'] = (
                alpha * result.execution_time + 
                (1 - alpha) * stats['avg_coordination_time']
            )
    
    def get_coordination_stats(self) -> Dict:
        """Get coordination performance statistics"""
        return self._coordination_stats.copy()