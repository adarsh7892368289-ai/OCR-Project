import os
import time
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from .base_engine import BaseOCREngine, OCRResult, BoundingBox, TextRegion, DocumentResult

@dataclass
class EnginePerformance:
    """Track engine performance metrics"""
    engine_name: str
    total_processed: int = 0
    successful_processes: int = 0
    total_time: float = 0.0
    total_confidence: float = 0.0
    success_rate: float = 0.0
    avg_confidence: float = 0.0
    avg_processing_time: float = 0.0
    
    def update(self, processing_time: float, confidence: float, success: bool):
        """Update performance metrics"""
        self.total_processed += 1
        self.total_time += processing_time
        
        if success:
            self.successful_processes += 1
            self.total_confidence += confidence
        
        # Recalculate averages
        self.success_rate = self.successful_processes / self.total_processed
        if self.successful_processes > 0:
            self.avg_confidence = self.total_confidence / self.successful_processes
        self.avg_processing_time = self.total_time / self.total_processed

class EngineManager:
    """
    Modern Engine Manager - Multi-Engine Coordination
    
    Clean architecture for managing multiple OCR engines:
    - Engine registration and initialization
    - Intelligent engine selection
    - Multi-engine processing and comparison
    - Performance tracking and optimization
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize engine manager with optional configuration"""
        self.config = config or {}
        self.engines: Dict[str, BaseOCREngine] = {}
        self.performance: Dict[str, EnginePerformance] = {}
        self.logger = logging.getLogger(__name__)
        
        # Configuration with defaults
        self.max_workers = self.config.get('max_workers', 3)
        self.timeout = self.config.get('timeout', 300)
        self.enable_parallel = self.config.get('enable_parallel', True)
        
        # Engine selection strategy
        self.selection_strategy = self.config.get('selection_strategy', 'adaptive')
        
        # Engine priorities for different content types
        self.engine_priorities = {
            'printed': ['tesseract', 'paddleocr', 'easyocr', 'trocr'],
            'handwritten': ['trocr', 'easyocr', 'paddleocr', 'tesseract'],
            'mixed': ['paddleocr', 'easyocr', 'trocr', 'tesseract'],
            'document': ['paddleocr', 'tesseract', 'easyocr', 'trocr'],
            'default': ['paddleocr', 'easyocr', 'tesseract', 'trocr']
        }
        
    def register_engine(self, engine_name: str, engine: BaseOCREngine) -> bool:
        """Register and initialize an OCR engine"""
        try:
            if engine_name in self.engines:
                self.logger.warning(f"Engine {engine_name} already registered, overwriting")
            
            self.engines[engine_name] = engine
            self.performance[engine_name] = EnginePerformance(engine_name=engine_name)
            
            # Auto-initialize the engine
            try:
                if engine.initialize():
                    self.logger.info(f"Registered and initialized engine: {engine_name}")
                else:
                    self.logger.warning(f"Registered engine {engine_name} but initialization failed")
            except Exception as init_error:
                self.logger.warning(f"Engine {engine_name} registered but initialization failed: {init_error}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register engine {engine_name}: {e}")
            return False
    
    def get_available_engines(self) -> Dict[str, BaseOCREngine]:
        """Get all registered engines"""
        return self.engines.copy()
    
    def get_initialized_engines(self) -> Dict[str, BaseOCREngine]:
        """Get only initialized engines"""
        return {name: engine for name, engine in self.engines.items() 
                if engine.is_initialized}
    
    def select_best_engine(self, image: np.ndarray, content_type: str = 'default') -> str:
        """Select best engine based on content type and performance"""
        initialized_engines = self.get_initialized_engines()
        
        if not initialized_engines:
            raise RuntimeError("No initialized engines available")
        
        # Get priority list for content type
        priority_list = self.engine_priorities.get(content_type, 
                                                  self.engine_priorities['default'])
        
        # Find first available engine from priority list
        for engine_name in priority_list:
            if engine_name in initialized_engines:
                return engine_name
        
        # Fallback to first available engine
        return list(initialized_engines.keys())[0]
    
    def process_with_multiple_engines(self, image: np.ndarray, 
                                    engine_names: Optional[List[str]] = None) -> Dict[str, List[OCRResult]]:
        """Process image with multiple engines for comparison"""
        if engine_names is None:
            engine_names = list(self.get_initialized_engines().keys())
        
        results = {}
        
        if self.enable_parallel and len(engine_names) > 1:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(engine_names))) as executor:
                future_to_engine = {
                    executor.submit(self._process_single_engine, engine_name, image): engine_name
                    for engine_name in engine_names
                    if engine_name in self.engines and self.engines[engine_name].is_initialized
                }
                
                for future in as_completed(future_to_engine, timeout=self.timeout):
                    engine_name = future_to_engine[future]
                    try:
                        result = future.result()
                        results[engine_name] = result
                    except Exception as e:
                        self.logger.error(f"Engine {engine_name} failed: {e}")
                        results[engine_name] = []
        else:
            # Sequential processing
            for engine_name in engine_names:
                if engine_name in self.engines and self.engines[engine_name].is_initialized:
                    try:
                        result = self._process_single_engine(engine_name, image)
                        results[engine_name] = result
                    except Exception as e:
                        self.logger.error(f"Engine {engine_name} failed: {e}")
                        results[engine_name] = []
        
        return results
    
    def _process_single_engine(self, engine_name: str, image: np.ndarray) -> List[OCRResult]:
        """Process image with a single engine"""
        start_time = time.time()
        engine = self.engines[engine_name]
        
        try:
            # Process image
            ocr_results = engine.process_image(image)
            processing_time = time.time() - start_time
            
            # Handle different return types
            if isinstance(ocr_results, list):
                results = ocr_results
            else:
                # Single result - convert to list
                results = [ocr_results] if ocr_results else []
            
            # Update performance
            if results:
                avg_confidence = sum(r.confidence for r in results) / len(results)
                self.performance[engine_name].update(processing_time, avg_confidence, True)
            else:
                self.performance[engine_name].update(processing_time, 0.0, False)
            
            return results
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.performance[engine_name].update(processing_time, 0.0, False)
            raise e
    
    def compare_results(self, results: Dict[str, List[OCRResult]]) -> Dict[str, Any]:
        """Compare results from multiple engines"""
        if not results:
            return {}
        
        comparison = {
            'engine_count': len(results),
            'total_detections': {},
            'confidence_scores': {},
            'processing_quality': {},
            'best_engine': None,
            'consensus_text': None
        }
        
        # Calculate metrics for each engine
        best_score = 0.0
        best_engine = None
        
        for engine_name, ocr_results in results.items():
            if not ocr_results:
                comparison['total_detections'][engine_name] = 0
                comparison['confidence_scores'][engine_name] = 0.0
                comparison['processing_quality'][engine_name] = 'failed'
                continue
            
            # Calculate metrics
            total_detections = len(ocr_results)
            avg_confidence = sum(r.confidence for r in ocr_results) / total_detections
            total_text_length = sum(len(r.text) for r in ocr_results)
            
            # Quality score (combination of confidence and text length)
            quality_score = (avg_confidence * 0.7) + (min(total_text_length / 100, 1.0) * 0.3)
            
            comparison['total_detections'][engine_name] = total_detections
            comparison['confidence_scores'][engine_name] = avg_confidence
            comparison['processing_quality'][engine_name] = 'excellent' if quality_score > 0.8 else 'good' if quality_score > 0.6 else 'fair'
            
            if quality_score > best_score:
                best_score = quality_score
                best_engine = engine_name
        
        comparison['best_engine'] = best_engine
        
        # Generate consensus text from best engine
        if best_engine and results[best_engine]:
            comparison['consensus_text'] = ' '.join(r.text for r in results[best_engine])
        
        return comparison
    
    def select_best_result(self, results: Dict[str, List[OCRResult]]) -> Optional[OCRResult]:
        """Select the best result from multiple engine outputs"""
        if not results:
            return None
        
        best_result = None
        best_score = 0.0
        
        for engine_name, ocr_results in results.items():
            if not ocr_results:
                continue
            
            for result in ocr_results:
                # Score based on confidence and text length
                text_length_score = min(len(result.text) / 50, 1.0)  # Normalize to 0-1
                combined_score = (result.confidence * 0.8) + (text_length_score * 0.2)
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_result = result
        
        return best_result
    
    def batch_process(self, images: List[np.ndarray], 
                     engine_name: Optional[str] = None) -> List[List[OCRResult]]:
        """Process multiple images with specified or best engine"""
        if not images:
            return []
        
        # Select engine if not specified
        if engine_name is None:
            engine_name = self.select_best_engine(images[0])
        
        if engine_name not in self.engines or not self.engines[engine_name].is_initialized:
            raise ValueError(f"Engine {engine_name} not available")
        
        engine = self.engines[engine_name]
        
        # Use engine's batch processing if available
        if hasattr(engine, 'batch_process'):
            return engine.batch_process(images)
        else:
            # Manual batch processing
            results = []
            for image in images:
                try:
                    result = engine.process_image(image)
                    if isinstance(result, list):
                        results.append(result)
                    else:
                        results.append([result] if result else [])
                except Exception as e:
                    self.logger.error(f"Batch processing failed for image: {e}")
                    results.append([])
            return results
    
    def process_with_best_engine(self, image: np.ndarray, 
                               content_type: str = 'default') -> List[OCRResult]:
        """Process image with automatically selected best engine"""
        if image is None or image.size == 0:
            return []
        
        try:
            best_engine = self.select_best_engine(image, content_type)
            return self._process_single_engine(best_engine, image)
        except Exception as e:
            self.logger.error(f"Best engine processing failed: {e}")
            return []
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for all engines"""
        stats = {
            'total_engines': len(self.engines),
            'initialized_engines': len(self.get_initialized_engines()),
            'engine_performance': {},
            'system_stats': {
                'max_workers': self.max_workers,
                'parallel_enabled': self.enable_parallel,
                'timeout': self.timeout
            }
        }
        
        # Add individual engine stats
        total_processed = 0
        total_time = 0.0
        
        for engine_name, perf in self.performance.items():
            stats['engine_performance'][engine_name] = {
                'total_processed': perf.total_processed,
                'success_rate': perf.success_rate,
                'avg_confidence': perf.avg_confidence,
                'avg_processing_time': perf.avg_processing_time,
                'status': 'initialized' if self.engines[engine_name].is_initialized else 'not_initialized'
            }
            
            total_processed += perf.total_processed
            total_time += perf.total_time
        
        # System-wide stats
        stats['system_stats']['total_processed'] = total_processed
        stats['system_stats']['total_time'] = total_time
        if total_processed > 0:
            stats['system_stats']['average_time'] = total_time / total_processed
        
        return stats
    
    def cleanup(self):
        """Cleanup all engines"""
        for engine_name, engine in self.engines.items():
            try:
                if hasattr(engine, 'cleanup'):
                    engine.cleanup()
                self.logger.info(f"Cleaned up engine: {engine_name}")
            except Exception as e:
                self.logger.error(f"Failed to cleanup engine {engine_name}: {e}")
        
        self.engines.clear()
        self.performance.clear()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
    def process_with_best_engine_single(self, image: np.ndarray, 
                                   content_type: str = 'default') -> Optional[OCRResult]:
        """Process image with best engine and return single best result"""
        if image is None or image.size == 0:
            return None
        
        try:
            # Get all results from best engine
            results_list = self.process_with_best_engine(image, content_type)
            
            # Return single best result
            if results_list:
                # Find result with highest confidence
                return max(results_list, key=lambda x: x.confidence)
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Best engine single processing failed: {e}")
            return None
# Alias for backward compatibility
OCREngineManager = EngineManager

# Export main classes
__all__ = [
    'EngineManager',
    'OCREngineManager', 
    'EnginePerformance'
]