# src/core/engine_manager.py - Clean Engine Manager

import time
import numpy as np
import logging
from typing import Dict, Any, Optional, List
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from .base_engine import BaseOCREngine, OCRResult

@dataclass
class EnginePerformance:
    """Simple performance tracking per engine"""
    engine_name: str
    total_processed: int = 0
    successful_processes: int = 0
    total_time: float = 0.0
    avg_confidence: float = 0.0

class EngineManager:
    """
    Clean Engine Manager - Just manages engines, no complex logic
    
    Responsibilities:
    1. Register and initialize engines
    2. Provide available engines to pipeline
    3. Simple engine selection logic
    4. Basic performance tracking
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize engine manager"""
        self.config = config or {}
        self.engines: Dict[str, BaseOCREngine] = {}
        self.performance: Dict[str, EnginePerformance] = {}
        self.logger = logging.getLogger(__name__)
        
        # Simple engine priority (based on your test results)
        self.engine_priority = ['paddleocr', 'easyocr', 'tesseract', 'trocr']
        
    def register_engine(self, engine_name: str, engine: BaseOCREngine) -> bool:
        """Register an OCR engine"""
        try:
            if engine_name in self.engines:
                self.logger.warning(f"Engine {engine_name} already registered, overwriting")
            
            self.engines[engine_name] = engine
            self.performance[engine_name] = EnginePerformance(engine_name=engine_name)
            
            self.logger.info(f"Registered engine: {engine_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register engine {engine_name}: {e}")
            return False
    
    def get_available_engines(self) -> Dict[str, BaseOCREngine]:
        """Get all registered engines"""
        return self.engines.copy()
    
    def get_initialized_engines(self) -> Dict[str, BaseOCREngine]:
        """Get only initialized engines"""
        initialized = {}
        for name, engine in self.engines.items():
            try:
                if engine.is_available():  # This calls initialize() internally
                    initialized[name] = engine
            except Exception as e:
                self.logger.warning(f"Engine {name} not available: {e}")
        return initialized
    
    def extract_with_engine(self, image: np.ndarray, engine_name: str) -> OCRResult:
        """Extract text using specific engine - main interface for pipeline"""
        if engine_name not in self.engines:
            raise ValueError(f"Engine {engine_name} not registered")
        
        engine = self.engines[engine_name]
        start_time = time.time()
        
        try:
            # Use the engine's extract_text method (from base_engine.py)
            result = engine.extract_text(image)
            processing_time = time.time() - start_time
            
            # Update simple stats
            perf = self.performance[engine_name]
            perf.total_processed += 1
            perf.total_time += processing_time
            
            if result and result.text.strip():
                perf.successful_processes += 1
                # Update average confidence
                if perf.successful_processes == 1:
                    perf.avg_confidence = result.confidence
                else:
                    total_conf = perf.avg_confidence * (perf.successful_processes - 1)
                    perf.avg_confidence = (total_conf + result.confidence) / perf.successful_processes
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.performance[engine_name].total_time += processing_time
            self.logger.error(f"Engine {engine_name} failed: {e}")
            
            # Return empty result instead of raising
            return OCRResult(
                text="",
                confidence=0.0,
                processing_time=processing_time,
                engine_name=engine_name,
                metadata={"error": str(e)}
            )
    
    def extract_with_multiple_engines(self, image: np.ndarray, 
                                     engine_names: List[str],
                                     use_parallel: bool = True) -> Dict[str, OCRResult]:
        """Extract with multiple engines - returns results from each"""
        results = {}
        available_engines = self.get_initialized_engines()
        
        # Filter to available engines
        engines_to_use = [name for name in engine_names if name in available_engines]
        
        if not engines_to_use:
            self.logger.error("No engines available for multi-engine processing")
            return results
        
        if use_parallel and len(engines_to_use) > 1:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=min(3, len(engines_to_use))) as executor:
                future_to_engine = {
                    executor.submit(self.extract_with_engine, image, engine_name): engine_name
                    for engine_name in engines_to_use
                }
                
                for future in as_completed(future_to_engine, timeout=300):
                    engine_name = future_to_engine[future]
                    try:
                        result = future.result()
                        results[engine_name] = result
                    except Exception as e:
                        self.logger.error(f"Engine {engine_name} failed in parallel: {e}")
                        results[engine_name] = OCRResult("", 0.0, engine_name=engine_name)
        else:
            # Sequential processing
            for engine_name in engines_to_use:
                try:
                    result = self.extract_with_engine(image, engine_name)
                    results[engine_name] = result
                except Exception as e:
                    self.logger.error(f"Engine {engine_name} failed: {e}")
                    results[engine_name] = OCRResult("", 0.0, engine_name=engine_name)
        
        return results
    
    def select_best_engine(self, content_type: str = 'default') -> Optional[str]:
        """Select best engine based on priority and availability"""
        available_engines = self.get_initialized_engines()
        
        if not available_engines:
            return None
        
        # Content-specific priorities
        if content_type == 'handwritten':
            priority_list = ['trocr', 'easyocr', 'paddleocr', 'tesseract']
        elif content_type == 'table':
            priority_list = ['paddleocr', 'tesseract', 'easyocr', 'trocr']
        else:
            priority_list = self.engine_priority
        
        # Find first available engine from priority list
        for engine_name in priority_list:
            if engine_name in available_engines:
                return engine_name
        
        # Fallback to any available engine
        return list(available_engines.keys())[0]
    
    def select_best_result(self, results: Dict[str, OCRResult]) -> Optional[OCRResult]:
        """Select best result from multiple engine results"""
        if not results:
            return None
        
        best_result = None
        best_score = 0.0
        
        for engine_name, result in results.items():
            if not result or not result.text.strip():
                continue
            
            # Simple scoring: confidence (80%) + text length factor (20%)
            text_length_score = min(len(result.text) / 100, 1.0)  # Normalize to 0-1
            combined_score = (result.confidence * 0.8) + (text_length_score * 0.2)
            
            if combined_score > best_score:
                best_score = combined_score
                best_result = result
        
        return best_result
    
    def get_engine_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all engines"""
        info = {}
        for name, engine in self.engines.items():
            perf = self.performance[name]
            info[name] = {
                'available': engine.is_available(),
                'initialized': name in self.get_initialized_engines(),
                'total_processed': perf.total_processed,
                'success_rate': (perf.successful_processes / perf.total_processed 
                                if perf.total_processed > 0 else 0.0),
                'avg_confidence': perf.avg_confidence,
                'avg_time': (perf.total_time / perf.total_processed 
                           if perf.total_processed > 0 else 0.0)
            }
        return info
    
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

# Alias for backward compatibility
OCREngineManager = EngineManager