# src/core/engine_manager.py

from typing import List, Dict, Any, Optional, Union
import numpy as np
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import time

from .base_engine import BaseOCREngine, DocumentResult, TextType, BoundingBox
from ..utils.config import Config

@dataclass
class EnginePerformance:
    """Track engine performance metrics"""
    engine_name: str
    avg_confidence: float
    avg_processing_time: float
    success_rate: float
    best_for_text_types: List[TextType]
    total_processed: int

class OCREngineManager:
    """Advanced OCR Engine Manager with intelligent engine selection"""
    
    def __init__(self, config: Config):
        self.config = config
        self.engines: Dict[str, BaseOCREngine] = {}
        self.initialized_engines: List[str] = []
        self.logger = logging.getLogger("OCR.EngineManager")
        self.performance_history: Dict[str, EnginePerformance] = {}
        
        # Engine selection strategy
        self.selection_strategy = config.get("engine.selection_strategy", "adaptive")
        self.parallel_processing = config.get("parallel_processing", True)
        self.max_workers = config.get("max_workers", 3)
        
        # Engine priorities for different scenarios
        self.engine_priorities = {
            TextType.PRINTED: ["paddleocr", "tesseract", "easyocr", "trocr"],
            TextType.HANDWRITTEN: ["trocr", "easyocr", "paddleocr"],
            TextType.MIXED: ["paddleocr", "easyocr", "trocr", "tesseract"],
            TextType.UNKNOWN: ["paddleocr", "easyocr", "trocr", "tesseract"]
        }
    
    def register_engine(self, engine: BaseOCREngine) -> bool:
        """Register an OCR engine"""
        try:
            if engine.name in self.engines:
                self.logger.warning(f"Engine {engine.name} already registered, overwriting")
            
            self.engines[engine.name] = engine
            
            # Initialize performance tracking
            self.performance_history[engine.name] = EnginePerformance(
                engine_name=engine.name,
                avg_confidence=0.0,
                avg_processing_time=0.0,
                success_rate=0.0,
                best_for_text_types=[],
                total_processed=0
            )
            
            self.logger.info(f"Registered engine: {engine.name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register engine {engine.name}: {e}")
            return False
    
    def initialize_engines(self, engine_names: Optional[List[str]] = None) -> Dict[str, bool]:
        """Initialize specified engines or all registered engines"""
        if engine_names is None:
            engine_names = list(self.engines.keys())
        
        results = {}
        
        for engine_name in engine_names:
            if engine_name not in self.engines:
                self.logger.error(f"Engine {engine_name} not registered")
                results[engine_name] = False
                continue
            
            try:
                engine = self.engines[engine_name]
                if engine.initialize():
                    self.initialized_engines.append(engine_name)
                    results[engine_name] = True
                    self.logger.info(f"Initialized engine: {engine_name}")
                else:
                    results[engine_name] = False
                    self.logger.error(f"Failed to initialize engine: {engine_name}")
                    
            except Exception as e:
                results[engine_name] = False
                self.logger.error(f"Exception initializing {engine_name}: {e}")
        
        return results
    
    def get_available_engines(self) -> List[str]:
        """Get list of available (initialized) engines"""
        return self.initialized_engines.copy()
    
    def select_best_engine(self, 
                          image: np.ndarray,
                          text_type: Optional[TextType] = None,
                          language: str = "en",
                          quality_priority: bool = True) -> Optional[str]:
        """Intelligently select the best engine for the given parameters"""
        
        if not self.initialized_engines:
            self.logger.error("No engines available")
            return None
        
        # Detect text type if not provided
        if text_type is None:
            text_type = self._detect_text_type(image)
        
        if self.selection_strategy == "adaptive":
            return self._adaptive_engine_selection(text_type, language, quality_priority)
        elif self.selection_strategy == "performance":
            return self._performance_based_selection(text_type, language)
        elif self.selection_strategy == "round_robin":
            return self._round_robin_selection()
        else:
            # Default to priority-based selection
            return self._priority_based_selection(text_type)
    
    def _detect_text_type(self, image: np.ndarray) -> TextType:
        """Detect text type from image characteristics"""
        # Use the first available engine's detection method
        if self.initialized_engines:
            engine_name = self.initialized_engines[0]
            engine = self.engines[engine_name]
            return engine.detect_text_type(image)
        
        return TextType.UNKNOWN
    
    def _adaptive_engine_selection(self, 
                                 text_type: TextType,
                                 language: str,
                                 quality_priority: bool) -> str:
        """Adaptive engine selection based on performance history"""
        
        available_engines = [name for name in self.engine_priorities[text_type] 
                           if name in self.initialized_engines]
        
        if not available_engines:
            return self.initialized_engines[0] if self.initialized_engines else None
        
        # Score engines based on performance and suitability
        engine_scores = {}
        
        for engine_name in available_engines:
            score = 0.0
            perf = self.performance_history[engine_name]
            
            # Base score from confidence and success rate
            if perf.total_processed > 0:
                confidence_score = perf.avg_confidence * 0.4
                success_score = perf.success_rate * 0.3
                
                # Speed score (inverse of processing time)
                if quality_priority:
                    speed_score = 0.1 / (perf.avg_processing_time + 0.1)
                else:
                    speed_score = 0.3 / (perf.avg_processing_time + 0.1)
                
                score = confidence_score + success_score + speed_score
            else:
                # New engine - give it a moderate score
                score = 0.5
            
            # Bonus for text type specialization
            engine = self.engines[engine_name]
            if text_type == TextType.HANDWRITTEN and engine.supports_handwriting:
                score *= 1.2
            
            # Language support bonus
            if language in engine.get_supported_languages():
                score *= 1.1
            
            engine_scores[engine_name] = score
        
        # Return the highest scoring engine
        return max(engine_scores.items(), key=lambda x: x[1])[0]
    
    def _performance_based_selection(self, text_type: TextType, language: str) -> str:
        """Select engine based on historical performance"""
        best_engine = None
        best_score = 0.0
        
        for engine_name in self.initialized_engines:
            perf = self.performance_history[engine_name]
            if perf.total_processed > 0:
                # Combined score of confidence and success rate
                score = (perf.avg_confidence * 0.7 + perf.success_rate * 0.3)
                
                if score > best_score:
                    best_score = score
                    best_engine = engine_name
        
        return best_engine or self.initialized_engines[0]
    
    def _priority_based_selection(self, text_type: TextType) -> str:
        """Select engine based on predefined priorities"""
        for engine_name in self.engine_priorities[text_type]:
            if engine_name in self.initialized_engines:
                return engine_name
        
        return self.initialized_engines[0]
    
    def _round_robin_selection(self) -> str:
        """Simple round-robin selection"""
        if not hasattr(self, '_round_robin_index'):
            self._round_robin_index = 0
        
        engine = self.initialized_engines[self._round_robin_index % len(self.initialized_engines)]
        self._round_robin_index += 1
        return engine
    
    def process_image(self, 
                     image: np.ndarray,
                     engine_name: Optional[str] = None,
                     **kwargs) -> DocumentResult:
        """Process image with specified or auto-selected engine"""
        
        if engine_name is None:
            engine_name = self.select_best_engine(image, **kwargs)
        
        if engine_name not in self.initialized_engines:
            raise ValueError(f"Engine {engine_name} not available")
        
        engine = self.engines[engine_name]
        start_time = time.time()
        
        try:
            result = engine.process_image(image, **kwargs)
            
            # Update performance tracking
            self._update_performance(engine_name, result, time.time() - start_time, True)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Engine {engine_name} failed: {e}")
            
            # Update performance tracking for failure
            self._update_performance(engine_name, None, time.time() - start_time, False)
            
            # Try fallback engine
            return self._try_fallback(image, engine_name, **kwargs)
    
    def process_image_multi_engine(self, 
                                 image: np.ndarray,
                                 engine_names: Optional[List[str]] = None,
                                 **kwargs) -> Dict[str, DocumentResult]:
        """Process image with multiple engines for comparison"""
        
        if engine_names is None:
            engine_names = self.initialized_engines[:3]  # Use top 3 engines
        
        results = {}
        
        if self.parallel_processing:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=min(len(engine_names), self.max_workers)) as executor:
                future_to_engine = {
                    executor.submit(self._process_with_engine, engine_name, image, **kwargs): engine_name
                    for engine_name in engine_names if engine_name in self.initialized_engines
                }
                
                for future in as_completed(future_to_engine):
                    engine_name = future_to_engine[future]
                    try:
                        result = future.result(timeout=30)  # 30 second timeout
                        results[engine_name] = result
                    except Exception as e:
                        self.logger.error(f"Engine {engine_name} failed in parallel processing: {e}")
        else:
            # Sequential processing
            for engine_name in engine_names:
                if engine_name in self.initialized_engines:
                    try:
                        result = self.process_image(image, engine_name, **kwargs)
                        results[engine_name] = result
                    except Exception as e:
                        self.logger.error(f"Engine {engine_name} failed: {e}")
        
        return results
    
    def _process_with_engine(self, engine_name: str, image: np.ndarray, **kwargs) -> DocumentResult:
        """Helper method for parallel processing"""
        return self.process_image(image, engine_name, **kwargs)
    
    def _try_fallback(self, image: np.ndarray, failed_engine: str, **kwargs) -> DocumentResult:
        """Try fallback engines when primary engine fails"""
        
        available_engines = [e for e in self.initialized_engines if e != failed_engine]
        
        if not available_engines:
            # Create empty result if no fallback available
            from .base_engine import DocumentResult, DocumentStructure
            return DocumentResult(
                full_text="",
                results=[],
                text_regions=[],
                document_structure=DocumentStructure(),
                processing_time=0.0,
                engine_name="failed",
                image_stats={},
                confidence_score=0.0
            )
        
        # Try the best alternative
        fallback_engine = available_engines[0]
        self.logger.info(f"Trying fallback engine: {fallback_engine}")
        
        try:
            return self.process_image(image, fallback_engine, **kwargs)
        except Exception as e:
            self.logger.error(f"Fallback engine {fallback_engine} also failed: {e}")
            # Return empty result
            from .base_engine import DocumentResult, DocumentStructure
            return DocumentResult(
                full_text="",
                results=[],
                text_regions=[],
                document_structure=DocumentStructure(),
                processing_time=0.0,
                engine_name="failed",
                image_stats={},
                confidence_score=0.0
            )
    
    def _update_performance(self, 
                          engine_name: str, 
                          result: Optional[DocumentResult],
                          processing_time: float,
                          success: bool):
        """Update engine performance statistics"""
        
        perf = self.performance_history[engine_name]
        
        # Update running averages
        total = perf.total_processed
        
        if success and result:
            # Update confidence average
            new_confidence = result.confidence_score
            perf.avg_confidence = ((perf.avg_confidence * total + new_confidence) / 
                                 (total + 1))
            
            # Update success rate
            perf.success_rate = ((perf.success_rate * total + 1.0) / (total + 1))
        else:
            # Update success rate for failure
            perf.success_rate = (perf.success_rate * total / (total + 1))
        
        # Update processing time average
        perf.avg_processing_time = ((perf.avg_processing_time * total + processing_time) / 
                                   (total + 1))
        
        perf.total_processed += 1
    
    def get_engine_performance(self) -> Dict[str, EnginePerformance]:
        """Get performance statistics for all engines"""
        return self.performance_history.copy()
    
    def get_best_engine_for_type(self, text_type: TextType) -> Optional[str]:
        """Get the best performing engine for a specific text type"""
        candidates = [name for name in self.engine_priorities[text_type] 
                     if name in self.initialized_engines]
        
        if not candidates:
            return None
        
        best_engine = None
        best_score = 0.0
        
        for engine_name in candidates:
            perf = self.performance_history[engine_name]
            if perf.total_processed > 0:
                score = perf.avg_confidence * perf.success_rate
                if score > best_score:
                    best_score = score
                    best_engine = engine_name
        
        return best_engine or candidates[0]
    
    def benchmark_engines(self, 
                         test_images: List[np.ndarray],
                         ground_truth: Optional[List[str]] = None) -> Dict[str, Dict[str, float]]:
        """Benchmark all engines against test images"""
        
        benchmark_results = {}
        
        for engine_name in self.initialized_engines:
            self.logger.info(f"Benchmarking engine: {engine_name}")
            
            results = []
            total_time = 0.0
            
            for i, image in enumerate(test_images):
                try:
                    start_time = time.time()
                    result = self.process_image(image, engine_name)
                    processing_time = time.time() - start_time
                    
                    results.append(result)
                    total_time += processing_time
                    
                except Exception as e:
                    self.logger.error(f"Benchmark failed for {engine_name} on image {i}: {e}")
                    continue
            
            if results:
                # Calculate metrics
                avg_confidence = sum(r.confidence_score for r in results) / len(results)
                avg_time = total_time / len(results)
                success_rate = len(results) / len(test_images)
                
                metrics = {
                    'avg_confidence': avg_confidence,
                    'avg_processing_time': avg_time,
                    'success_rate': success_rate,
                    'total_processed': len(results)
                }
                
                # Calculate accuracy if ground truth provided
                if ground_truth:
                    accuracy = self._calculate_accuracy(results, ground_truth)
                    metrics['accuracy'] = accuracy
                
                benchmark_results[engine_name] = metrics
        
        return benchmark_results
    
    def _calculate_accuracy(self, results: List[DocumentResult], ground_truth: List[str]) -> float:
        """Calculate accuracy against ground truth"""
        from difflib import SequenceMatcher
        
        accuracies = []
        
        for i, result in enumerate(results):
            if i < len(ground_truth):
                predicted = result.full_text.strip().lower()
                actual = ground_truth[i].strip().lower()
                
                # Use sequence matching for similarity
                similarity = SequenceMatcher(None, predicted, actual).ratio()
                accuracies.append(similarity)
        
        return sum(accuracies) / len(accuracies) if accuracies else 0.0
    
    def export_performance_report(self, output_path: str):
        """Export detailed performance report"""
        import json
        from datetime import datetime
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_engines': len(self.engines),
            'initialized_engines': len(self.initialized_engines),
            'performance_data': {}
        }
        
        for engine_name, perf in self.performance_history.items():
            report['performance_data'][engine_name] = {
                'avg_confidence': perf.avg_confidence,
                'avg_processing_time': perf.avg_processing_time,
                'success_rate': perf.success_rate,
                'total_processed': perf.total_processed,
                'best_for_text_types': [t.value for t in perf.best_for_text_types]
            }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Performance report exported to {output_path}")
    
    def cleanup_engines(self):
        """Cleanup all initialized engines"""
        for engine_name in self.initialized_engines:
            try:
                self.engines[engine_name].cleanup()
                self.logger.info(f"Cleaned up engine: {engine_name}")
            except Exception as e:
                self.logger.error(f"Failed to cleanup engine {engine_name}: {e}")
        
        self.initialized_engines.clear()
    
    def __enter__(self):
        # Initialize default engines
        default_engines = self.config.get("engines", {}).keys()
        self.initialize_engines(list(default_engines))
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup_engines()