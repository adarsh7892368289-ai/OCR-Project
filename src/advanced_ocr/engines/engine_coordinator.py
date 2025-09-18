"""
Smart engine selection and coordination for optimal OCR results.
Implements adaptive strategies with parallel execution and fallback handling.
"""

import time
import logging
import threading
from typing import Dict, List, Optional, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from dataclasses import dataclass
from enum import Enum

from .base_engine import BaseOCREngine, EngineStatus
from ..results import OCRResult, ProcessingMetrics
from ..config import OCRConfig, EngineStrategy


@dataclass
class EngineSelection:
    """Information about engine selection decision."""
    
    primary_engine: str
    secondary_engines: List[str]
    strategy_used: EngineStrategy
    selection_reason: str
    confidence_threshold: float
    expected_processing_time: float


class ContentType(Enum):
    """Detected content types for engine selection."""
    PRINTED_TEXT = "printed_text"
    HANDWRITTEN = "handwritten"
    MIXED_CONTENT = "mixed_content"
    DOCUMENT = "document"
    RECEIPT = "receipt"
    FORM = "form"
    UNKNOWN = "unknown"


class EngineCoordinator:
    """
    Intelligent engine selection and coordination system.
    Implements adaptive strategies with performance-based optimization.
    """
    
    def __init__(self, config: OCRConfig):
        """
        Initialize engine coordinator.
        
        Args:
            config: OCR configuration
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Engine registry
        self.engines: Dict[str, BaseOCREngine] = {}
        self.engine_performance: Dict[str, Dict[str, Any]] = {}
        self._engine_lock = threading.RLock()
        
        # Thread pool for parallel execution
        max_workers = min(len(config.enabled_engines), config.performance.max_workers)
        self.executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix='EngineCoordinator'
        )
        
        # Performance tracking
        self._total_extractions = 0
        self._strategy_performance: Dict[str, Dict[str, Any]] = {
            strategy.value: {
                'used_count': 0,
                'success_count': 0,
                'total_time': 0.0,
                'avg_confidence': 0.0
            } for strategy in EngineStrategy
        }
        
        self.logger.info(f"Initialized engine coordinator with strategy: {config.engine_strategy.value}")
    
    def register_engine(self, engine: BaseOCREngine) -> bool:
        """
        Register an OCR engine.
        
        Args:
            engine: OCR engine instance
            
        Returns:
            True if registered successfully
        """
        with self._engine_lock:
            engine_name = engine.name
            
            if engine_name in self.engines:
                self.logger.warning(f"Engine {engine_name} already registered, replacing")
            
            # Initialize engine if needed
            if not engine.is_ready and not engine.initialize():
                self.logger.error(f"Failed to initialize engine: {engine_name}")
                return False
            
            self.engines[engine_name] = engine
            
            # Initialize performance tracking
            if engine_name not in self.engine_performance:
                self.engine_performance[engine_name] = {
                    'success_rate': 0.0,
                    'avg_processing_time': 0.0,
                    'avg_confidence': 0.0,
                    'content_type_performance': {},
                    'last_used': 0.0,
                    'total_uses': 0
                }
            
            self.logger.info(f"Registered engine: {engine_name}")
            return True
    
    def extract_text(self, image: Any) -> Optional[OCRResult]:
        """
        Extract text using optimal engine selection strategy.
        
        Args:
            image: Input image
            
        Returns:
            OCRResult or None if all engines failed
        """
        start_time = time.time()
        self._total_extractions += 1
        
        try:
            # Analyze image content for engine selection
            content_analysis = self._analyze_image_content(image)
            
            # Select engines based on strategy
            selection = self._select_engines(content_analysis)
            
            self.logger.info(
                f"Selected engines for {content_analysis['content_type'].value}: "
                f"primary={selection.primary_engine}, "
                f"secondary={selection.secondary_engines}, "
                f"strategy={selection.strategy_used.value}"
            )
            
            # Execute extraction based on strategy
            result = self._execute_extraction(image, selection)
            
            # Update performance metrics
            processing_time = time.time() - start_time
            self._update_strategy_performance(selection.strategy_used, result, processing_time)
            
            if result:
                # Add coordinator metadata
                result.metadata.update({
                    'engine_selection': {
                        'primary_engine': selection.primary_engine,
                        'secondary_engines': selection.secondary_engines,
                        'strategy_used': selection.strategy_used.value,
                        'selection_reason': selection.selection_reason,
                        'content_analysis': content_analysis
                    },
                    'coordinator_processing_time': processing_time
                })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Engine coordination failed: {e}")
            return None
    
    def _analyze_image_content(self, image: Any) -> Dict[str, Any]:
        """
        Analyze image content to inform engine selection.
        
        Args:
            image: Input image
            
        Returns:
            Content analysis dictionary
        """
        # For now, basic analysis - can be enhanced with ML models
        analysis = {
            'content_type': ContentType.PRINTED_TEXT,  # Default assumption
            'estimated_complexity': 'medium',
            'has_handwriting': False,
            'text_density': 'medium',
            'image_quality': 'good',
            'confidence': 0.7
        }
        
        try:
            # Basic image analysis
            from PIL import Image
            import numpy as np
            
            if isinstance(image, str):
                pil_image = Image.open(image)
            elif isinstance(image, Image.Image):
                pil_image = image
            else:
                # Assume it's a numpy array
                pil_image = Image.fromarray(image)
            
            # Image size analysis
            width, height = pil_image.size
            total_pixels = width * height
            
            if total_pixels < 100000:  # Small image
                analysis['estimated_complexity'] = 'low'
            elif total_pixels > 1000000:  # Large image
                analysis['estimated_complexity'] = 'high'
            
            # Convert to grayscale for analysis
            if pil_image.mode != 'L':
                gray_image = pil_image.convert('L')
            else:
                gray_image = pil_image
            
            # Basic quality assessment using variance
            img_array = np.array(gray_image)
            variance = np.var(img_array)
            
            if variance < 1000:
                analysis['image_quality'] = 'poor'
            elif variance > 5000:
                analysis['image_quality'] = 'excellent'
            
            self.logger.debug(f"Content analysis: {analysis}")
            
        except Exception as e:
            self.logger.warning(f"Content analysis failed: {e}, using defaults")
        
        return analysis
    
    def _select_engines(self, content_analysis: Dict[str, Any]) -> EngineSelection:
        """
        Select optimal engines based on strategy and content analysis.
        
        Args:
            content_analysis: Image content analysis
            
        Returns:
            EngineSelection with chosen engines and strategy
        """
        strategy = self.config.engine_strategy
        enabled_engines = self._get_available_engines()
        
        if not enabled_engines:
            raise RuntimeError("No engines available for text extraction")
        
        # Strategy-based selection
        if strategy == EngineStrategy.SINGLE:
            return self._select_single_engine(content_analysis, enabled_engines)
        elif strategy == EngineStrategy.DUAL:
            return self._select_dual_engines(content_analysis, enabled_engines)
        elif strategy == EngineStrategy.ADAPTIVE:
            return self._select_adaptive_engines(content_analysis, enabled_engines)
        else:  # ALL strategy
            return self._select_all_engines(content_analysis, enabled_engines)
    
    def _select_single_engine(self, content_analysis: Dict[str, Any], 
                             available_engines: List[str]) -> EngineSelection:
        """Select single best engine for content."""
        # Rank engines based on performance and content type
        best_engine = self._rank_engines_for_content(content_analysis, available_engines)[0]
        
        return EngineSelection(
            primary_engine=best_engine,
            secondary_engines=[],
            strategy_used=EngineStrategy.SINGLE,
            selection_reason=f"Best performing engine for {content_analysis['content_type'].value}",
            confidence_threshold=self.config.get_quality_threshold_value(),
            expected_processing_time=self._estimate_processing_time([best_engine])
        )
    
    def _select_dual_engines(self, content_analysis: Dict[str, Any], 
                            available_engines: List[str]) -> EngineSelection:
        """Select two complementary engines."""
        ranked_engines = self._rank_engines_for_content(content_analysis, available_engines)
        
        primary = ranked_engines[0]
        secondary = [ranked_engines[1]] if len(ranked_engines) > 1 else []
        
        return EngineSelection(
            primary_engine=primary,
            secondary_engines=secondary,
            strategy_used=EngineStrategy.DUAL,
            selection_reason="Dual engine fusion for improved accuracy",
            confidence_threshold=self.config.get_quality_threshold_value(),
            expected_processing_time=self._estimate_processing_time([primary] + secondary)
        )
    
    def _select_adaptive_engines(self, content_analysis: Dict[str, Any], 
                                available_engines: List[str]) -> EngineSelection:
        """Adaptively select engines based on content and performance."""
        content_type = content_analysis['content_type']
        complexity = content_analysis['estimated_complexity']
        
        # Adaptive logic based on content
        if content_type == ContentType.HANDWRITTEN:
            # Prioritize TrOCR for handwriting
            if 'trocr' in available_engines:
                primary = 'trocr'
                secondary = [e for e in ['paddleocr', 'easyocr'] if e in available_engines][:1]
                reason = "TrOCR optimized for handwritten text"
            else:
                ranked = self._rank_engines_for_content(content_analysis, available_engines)
                primary = ranked[0]
                secondary = ranked[1:2]
                reason = "Best available engine for handwritten text"
        
        elif complexity == 'high' or content_analysis['image_quality'] == 'poor':
            # Use multiple engines for complex/poor quality images
            ranked = self._rank_engines_for_content(content_analysis, available_engines)
            primary = ranked[0]
            secondary = ranked[1:3]  # Use up to 2 secondary engines
            reason = "Multiple engines for complex/poor quality image"
        
        else:
            # Use single best engine for simple cases
            best_engine = self._rank_engines_for_content(content_analysis, available_engines)[0]
            primary = best_engine
            secondary = []
            reason = "Single engine sufficient for simple content"
        
        return EngineSelection(
            primary_engine=primary,
            secondary_engines=secondary,
            strategy_used=EngineStrategy.ADAPTIVE,
            selection_reason=reason,
            confidence_threshold=self.config.get_quality_threshold_value(),
            expected_processing_time=self._estimate_processing_time([primary] + secondary)
        )
    
    def _select_all_engines(self, content_analysis: Dict[str, Any], 
                           available_engines: List[str]) -> EngineSelection:
        """Use all available engines for maximum accuracy."""
        ranked_engines = self._rank_engines_for_content(content_analysis, available_engines)
        
        primary = ranked_engines[0]
        secondary = ranked_engines[1:]
        
        return EngineSelection(
            primary_engine=primary,
            secondary_engines=secondary,
            strategy_used=EngineStrategy.ALL,
            selection_reason="All engines for maximum accuracy",
            confidence_threshold=self.config.get_quality_threshold_value(),
            expected_processing_time=self._estimate_processing_time(ranked_engines)
        )
    
    def _rank_engines_for_content(self, content_analysis: Dict[str, Any], 
                                 available_engines: List[str]) -> List[str]:
        """Rank engines based on expected performance for content type."""
        content_type = content_analysis['content_type']
        
        # Engine rankings for different content types
        rankings = {
            ContentType.PRINTED_TEXT: ['paddleocr', 'easyocr', 'tesseract', 'trocr'],
            ContentType.HANDWRITTEN: ['trocr', 'paddleocr', 'easyocr', 'tesseract'],
            ContentType.DOCUMENT: ['paddleocr', 'tesseract', 'easyocr', 'trocr'],
            ContentType.RECEIPT: ['paddleocr', 'easyocr', 'tesseract', 'trocr'],
            ContentType.MIXED_CONTENT: ['paddleocr', 'easyocr', 'trocr', 'tesseract']
        }
        
        # Get base ranking for content type
        base_ranking = rankings.get(content_type, rankings[ContentType.PRINTED_TEXT])
        
        # Filter to available engines and add performance weighting
        available_ranked = []
        for engine in base_ranking:
            if engine in available_engines:
                # Weight by historical performance
                performance = self.engine_performance.get(engine, {})
                success_rate = performance.get('success_rate', 0.5)
                
                # Add small random factor to break ties
                import random
                weight = success_rate + random.uniform(-0.01, 0.01)
                available_ranked.append((engine, weight))
        
        # Add any remaining engines not in base ranking
        for engine in available_engines:
            if engine not in [e[0] for e in available_ranked]:
                performance = self.engine_performance.get(engine, {})
                success_rate = performance.get('success_rate', 0.3)  # Lower default for unknown
                available_ranked.append((engine, success_rate))
        
        # Sort by weight (descending)
        available_ranked.sort(key=lambda x: x[1], reverse=True)
        
        return [engine for engine, _ in available_ranked]
    
    def _execute_extraction(self, image: Any, selection: EngineSelection) -> Optional[OCRResult]:
        """Execute text extraction with selected engines."""
        engines_to_use = [selection.primary_engine] + selection.secondary_engines
        
        if len(engines_to_use) == 1:
            # Single engine extraction
            return self._extract_with_single_engine(image, selection.primary_engine)
        else:
            # Multiple engine extraction with fusion
            return self._extract_with_multiple_engines(image, engines_to_use, selection)
    
    def _extract_with_single_engine(self, image: Any, engine_name: str) -> Optional[OCRResult]:
        """Extract text with single engine."""
        engine = self.engines.get(engine_name)
        if not engine:
            self.logger.error(f"Engine not found: {engine_name}")
            return None
        
        try:
            result = engine.extract(image)
            self._update_engine_performance(engine_name, result is not None, result)
            return result
        except Exception as e:
            self.logger.error(f"Single engine extraction failed: {e}")
            self._update_engine_performance(engine_name, False, None)
            return None
    
    def _extract_with_multiple_engines(self, image: Any, engine_names: List[str], 
                                     selection: EngineSelection) -> Optional[OCRResult]:
        """Extract text with multiple engines in parallel."""
        if not self.config.performance.enable_parallel_engines:
            # Sequential execution
            return self._extract_sequential(image, engine_names)
        
        # Parallel execution
        future_to_engine = {}
        
        for engine_name in engine_names:
            engine = self.engines.get(engine_name)
            if engine and engine.is_ready:
                future = self.executor.submit(engine.extract, image)
                future_to_engine[future] = engine_name
        
        if not future_to_engine:
            self.logger.error("No engines ready for parallel execution")
            return None
        
        # Collect results as they complete
        results = {}
        exceptions = {}
        
        try:
            for future in as_completed(future_to_engine, timeout=self.config.performance.default_timeout):
                engine_name = future_to_engine[future]
                try:
                    result = future.result(timeout=5)  # Quick timeout for individual results
                    results[engine_name] = result
                    self._update_engine_performance(engine_name, result is not None, result)
                    
                    if result:
                        self.logger.debug(f"Got result from {engine_name}")
                    else:
                        self.logger.warning(f"Engine {engine_name} returned no result")
                        
                except Exception as e:
                    exceptions[engine_name] = str(e)
                    self.logger.error(f"Engine {engine_name} failed: {e}")
                    self._update_engine_performance(engine_name, False, None)
        
        except Exception as e:
            self.logger.error(f"Parallel execution failed: {e}")
            # Cancel remaining futures
            for future in future_to_engine:
                future.cancel()
        
        # Return best result or fused result
        valid_results = {name: result for name, result in results.items() if result is not None}
        
        if not valid_results:
            self.logger.error("All engines failed to produce results")
            return None
        
        if len(valid_results) == 1:
            # Single valid result
            return list(valid_results.values())[0]
        
        # Multiple results - return primary if available, otherwise best confidence
        if selection.primary_engine in valid_results:
            primary_result = valid_results[selection.primary_engine]
            # Add metadata about other engines
            primary_result.metadata['alternative_engines'] = list(valid_results.keys())
            return primary_result
        
        # Return result with highest confidence
        best_result = max(valid_results.values(), 
                         key=lambda r: r.confidence.overall_confidence if r.confidence else 0.0)
        return best_result
    
    def _extract_sequential(self, image: Any, engine_names: List[str]) -> Optional[OCRResult]:
        """Extract text with engines sequentially (fallback approach)."""
        for engine_name in engine_names:
            engine = self.engines.get(engine_name)
            if not engine or not engine.is_ready:
                continue
            
            try:
                result = engine.extract(image)
                self._update_engine_performance(engine_name, result is not None, result)
                
                if result and result.confidence:
                    confidence = result.confidence.overall_confidence
                    if confidence >= self.config.get_quality_threshold_value():
                        self.logger.info(f"Sequential: {engine_name} met quality threshold")
                        return result
                    else:
                        self.logger.debug(
                            f"Sequential: {engine_name} below threshold "
                            f"({confidence:.2f} < {self.config.get_quality_threshold_value()})"
                        )
                
            except Exception as e:
                self.logger.error(f"Sequential engine {engine_name} failed: {e}")
                self._update_engine_performance(engine_name, False, None)
        
        self.logger.warning("All sequential engines failed or below threshold")
        return None
    
    def _get_available_engines(self) -> List[str]:
        """Get list of available and ready engines."""
        with self._engine_lock:
            available = []
            for name, engine in self.engines.items():
                if (name in self.config.enabled_engines and 
                    engine.status == EngineStatus.READY):
                    available.append(name)
            return available
    
    def _estimate_processing_time(self, engine_names: List[str]) -> float:
        """Estimate total processing time for engine combination."""
        if not engine_names:
            return 0.0
        
        if len(engine_names) == 1:
            # Single engine
            perf = self.engine_performance.get(engine_names[0], {})
            return perf.get('avg_processing_time', 5.0)  # Default 5 seconds
        
        # Multiple engines - assume parallel execution
        if self.config.performance.enable_parallel_engines:
            # Return time of slowest engine
            max_time = 0.0
            for engine_name in engine_names:
                perf = self.engine_performance.get(engine_name, {})
                engine_time = perf.get('avg_processing_time', 5.0)
                max_time = max(max_time, engine_time)
            return max_time
        else:
            # Sequential execution - sum all times
            total_time = 0.0
            for engine_name in engine_names:
                perf = self.engine_performance.get(engine_name, {})
                total_time += perf.get('avg_processing_time', 5.0)
            return total_time
    
    def _update_engine_performance(self, engine_name: str, success: bool, result: Optional[OCRResult]):
        """Update engine performance metrics."""
        if engine_name not in self.engine_performance:
            self.engine_performance[engine_name] = {
                'success_rate': 0.0,
                'avg_processing_time': 0.0,
                'avg_confidence': 0.0,
                'content_type_performance': {},
                'last_used': 0.0,
                'total_uses': 0
            }
        
        perf = self.engine_performance[engine_name]
        perf['total_uses'] += 1
        perf['last_used'] = time.time()
        
        # Update success rate (exponential moving average)
        alpha = 0.1  # Learning rate
        current_success = 1.0 if success else 0.0
        perf['success_rate'] = (1 - alpha) * perf['success_rate'] + alpha * current_success
        
        if result:
            # Update processing time
            if result.processing_time > 0:
                perf['avg_processing_time'] = (
                    (1 - alpha) * perf['avg_processing_time'] + 
                    alpha * result.processing_time
                )
            
            # Update confidence
            if result.confidence:
                confidence = result.confidence.overall_confidence
                perf['avg_confidence'] = (
                    (1 - alpha) * perf['avg_confidence'] + 
                    alpha * confidence
                )
    
    def _update_strategy_performance(self, strategy: EngineStrategy, result: Optional[OCRResult], 
                                   processing_time: float):
        """Update strategy performance metrics."""
        strategy_perf = self._strategy_performance[strategy.value]
        strategy_perf['used_count'] += 1
        strategy_perf['total_time'] += processing_time
        
        if result and result.confidence:
            strategy_perf['success_count'] += 1
            # Update average confidence (exponential moving average)
            alpha = 0.1
            new_confidence = result.confidence.overall_confidence
            strategy_perf['avg_confidence'] = (
                (1 - alpha) * strategy_perf['avg_confidence'] + 
                alpha * new_confidence
            )
    
    def get_engine_stats(self) -> Dict[str, Any]:
        """Get comprehensive engine performance statistics."""
        with self._engine_lock:
            stats = {
                'total_extractions': self._total_extractions,
                'registered_engines': len(self.engines),
                'available_engines': len(self._get_available_engines()),
                'engine_performance': self.engine_performance.copy(),
                'strategy_performance': self._strategy_performance.copy(),
                'engine_status': {
                    name: engine.status.value 
                    for name, engine in self.engines.items()
                }
            }
            
            # Calculate overall coordinator performance
            total_strategy_uses = sum(
                perf['used_count'] for perf in self._strategy_performance.values()
            )
            if total_strategy_uses > 0:
                overall_success_rate = sum(
                    perf['success_count'] for perf in self._strategy_performance.values()
                ) / total_strategy_uses
                
                stats['overall_performance'] = {
                    'success_rate': overall_success_rate,
                    'total_strategy_uses': total_strategy_uses,
                    'avg_processing_time': sum(
                        perf['total_time'] for perf in self._strategy_performance.values()
                    ) / total_strategy_uses
                }
            
            return stats
    
    def get_engine_recommendations(self) -> Dict[str, Any]:
        """Get recommendations for engine configuration optimization."""
        stats = self.get_engine_stats()
        recommendations = {
            'engine_recommendations': [],
            'strategy_recommendations': [],
            'configuration_suggestions': []
        }
        
        # Analyze engine performance
        for engine_name, perf in self.engine_performance.items():
            if perf['total_uses'] < 10:
                continue  # Not enough data
            
            if perf['success_rate'] < 0.5:
                recommendations['engine_recommendations'].append({
                    'engine': engine_name,
                    'issue': 'low_success_rate',
                    'value': perf['success_rate'],
                    'suggestion': 'Consider disabling or investigating configuration'
                })
            
            if perf['avg_processing_time'] > 30.0:
                recommendations['engine_recommendations'].append({
                    'engine': engine_name,
                    'issue': 'slow_processing',
                    'value': perf['avg_processing_time'],
                    'suggestion': 'Check GPU acceleration or reduce timeout'
                })
        
        # Analyze strategy performance
        best_strategy = None
        best_success_rate = 0.0
        
        for strategy, perf in self._strategy_performance.items():
            if perf['used_count'] > 0:
                success_rate = perf['success_count'] / perf['used_count']
                if success_rate > best_success_rate:
                    best_success_rate = success_rate
                    best_strategy = strategy
        
        if best_strategy and best_strategy != self.config.engine_strategy.value:
            recommendations['strategy_recommendations'].append({
                'current_strategy': self.config.engine_strategy.value,
                'recommended_strategy': best_strategy,
                'current_success_rate': self._strategy_performance[self.config.engine_strategy.value].get('success_count', 0) / max(1, self._strategy_performance[self.config.engine_strategy.value].get('used_count', 1)),
                'recommended_success_rate': best_success_rate,
                'reason': 'Higher success rate observed'
            })
        
        return recommendations
    
    def shutdown(self):
        """Shutdown coordinator and all engines."""
        self.logger.info("Shutting down engine coordinator")
        
        # Shutdown thread pool
        self.executor.shutdown(wait=True)
        
        # Shutdown all engines
        with self._engine_lock:
            for engine in self.engines.values():
                try:
                    engine.shutdown()
                except Exception as e:
                    self.logger.error(f"Error shutting down engine {engine.name}: {e}")
            
            self.engines.clear()
    
    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.shutdown()
        except:
            pass