"""OCR Engine Manager for coordinating multiple OCR engines.

Manages registration, initialization, and execution of OCR engines with support
for both single-engine and parallel multi-engine processing. Provides intelligent
result combination and automatic engine discovery.

Examples
--------
    from advanced_ocr.core import EngineManager
    from advanced_ocr.types import ProcessingOptions
    
    # Basic usage with auto-discovery
    manager = EngineManager()
    manager.initialize_available_engines()
    
    # Execute single engine
    result = manager.execute_engine("paddleocr", image)
    print(f"Extracted: {result.text}")
    
    # Execute multiple engines in parallel
    engines = ["paddleocr", "easyocr", "tesseract"]
    results = manager.execute_multiple_engines(engines, image)
    combined = manager.combine_results(results)
    print(f"Combined confidence: {combined.confidence:.2f}")
"""

import logging
import time
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

from .base_engine import BaseOCREngine
from ..types import OCRResult, ProcessingOptions
from ..exceptions import EngineNotAvailableError, EngineInitializationError


class EngineManager:
    """Manages OCR engine lifecycle and coordinates execution strategies.
    
    Features:
    - Automatic engine discovery and registration
    - Single and multi-engine execution modes
    - Parallel processing for improved performance
    - Intelligent result combination with consensus detection
    - Engine availability monitoring
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize engine manager with optional configuration."""
        self.config = config or {}
        self.engines: Dict[str, BaseOCREngine] = {}
        self.logger = logging.getLogger(__name__)
    
    def register_engine(self, engine_name: str, engine: BaseOCREngine) -> None:
        """Register an OCR engine for use by the manager."""
        if not isinstance(engine, BaseOCREngine):
            raise ValueError(f"Engine must inherit from BaseOCREngine, got {type(engine)}")
        
        if engine_name in self.engines:
            self.logger.warning(f"Overwriting existing engine: {engine_name}")
        
        self.engines[engine_name] = engine
        self.logger.info(f"Registered engine: {engine_name}")
    
    def initialize_engines(self) -> Dict[str, bool]:
        """Initialize all registered engines.
        
        Returns:
            Dictionary mapping engine names to initialization success status
        """
        results = {}
        
        for name, engine in self.engines.items():
            try:
                success = engine.initialize()
                results[name] = success
                
                if success:
                    self.logger.info(f"Initialized engine: {name}")
                else:
                    self.logger.warning(f"Failed to initialize engine: {name}")
                    
            except Exception as e:
                results[name] = False
                self.logger.error(f"Engine {name} initialization error: {e}")
        
        return results
    
    def get_available_engines(self) -> List[str]:
        """Get list of engines that are initialized and available."""
        available = []
        
        for name, engine in self.engines.items():
            try:
                if engine.is_available():
                    available.append(name)
            except Exception as e:
                self.logger.debug(f"Engine {name} availability check failed: {e}")
        
        return available
    
    def execute_engine(self, engine_name: str, image: np.ndarray, 
                      options: Optional[ProcessingOptions] = None) -> OCRResult:
        """Execute a single OCR engine on an image."""
        if engine_name not in self.engines:
            raise EngineNotAvailableError(
                engine_name, 
                f"Engine not registered. Available: {list(self.engines.keys())}"
            )
        
        engine = self.engines[engine_name]
        
        if not engine.is_available():
            raise EngineNotAvailableError(engine_name, "Engine not initialized or unavailable")
        
        start_time = time.time()
        
        try:
            # Execute the engine
            result = engine.extract_text(image)
            processing_time = time.time() - start_time
            
            # Ensure result metadata is complete
            if not result.engine_used:
                result.engine_used = engine_name
            if not result.processing_time:
                result.processing_time = processing_time
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Engine {engine_name} execution failed: {e}")
            
            return OCRResult(
                text="",
                confidence=0.0,
                processing_time=processing_time,
                engine_used=engine_name,
                metadata={"error": str(e), "error_type": type(e).__name__}
            )
    
    def execute_multiple_engines(self, engine_names: List[str], image: np.ndarray,
                                options: Optional[ProcessingOptions] = None,
                                use_parallel: bool = True) -> Dict[str, OCRResult]:
        """Execute multiple OCR engines on the same image."""
        results = {}
        available_engines = self.get_available_engines()
        
        # Filter to only available engines
        engines_to_run = [name for name in engine_names if name in available_engines]
        
        if not engines_to_run:
            self.logger.warning("No requested engines are available")
            return results
        
        self.logger.info(f"Running {len(engines_to_run)} engines: {engines_to_run}")
        
        if use_parallel and len(engines_to_run) > 1:
            results = self._execute_engines_parallel(engines_to_run, image, options)
        else:
            results = self._execute_engines_sequential(engines_to_run, image, options)
        
        return results
    
    def _execute_engines_parallel(self, engines: List[str], image: np.ndarray,
                                options: Optional[ProcessingOptions]) -> Dict[str, OCRResult]:
        """Execute engines concurrently using thread pool."""
        results = {}
        max_workers = min(4, len(engines))  # Limit to 4 concurrent workers
        
        self.logger.info(f"Starting parallel execution with {max_workers} workers")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all engine tasks
            future_to_engine = {
                executor.submit(self.execute_engine, name, image, options): name
                for name in engines
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_engine):
                engine_name = future_to_engine[future]
                try:
                    result = future.result()
                    results[engine_name] = result
                    self.logger.debug(f"Completed {engine_name}: confidence {result.confidence:.3f}")
                        
                except Exception as e:
                    self.logger.error(f"Parallel execution error for {engine_name}: {e}")
                    results[engine_name] = OCRResult("", 0.0, engine_used=engine_name,
                                                   metadata={"error": str(e)})
        
        self.logger.info(f"Parallel execution completed for {len(results)} engines")
        return results
    
    def _execute_engines_sequential(self, engines: List[str], image: np.ndarray,
                                  options: Optional[ProcessingOptions]) -> Dict[str, OCRResult]:
        """Execute engines one after another in sequence."""
        results = {}
        
        self.logger.info(f"Starting sequential execution")
        
        for engine_name in engines:
            try:
                result = self.execute_engine(engine_name, image, options)
                results[engine_name] = result
                self.logger.debug(f"Completed {engine_name}: confidence {result.confidence:.3f}")
                    
            except Exception as e:
                self.logger.error(f"Sequential execution error for {engine_name}: {e}")
                results[engine_name] = OCRResult("", 0.0, engine_used=engine_name,
                                               metadata={"error": str(e)})
        
        self.logger.info(f"Sequential execution completed for {len(results)} engines")
        return results
    
    def combine_results(self, results: Dict[str, OCRResult]) -> OCRResult:
        """Combine results from multiple engines using intelligent selection.
        
        Uses confidence-based ranking with consensus detection. When multiple
        high-confidence engines agree on similar text, their combined confidence
        is boosted slightly to reflect the consensus.
        
        Returns:
            Single combined OCRResult with aggregated metadata
        """
        if not results:
            return OCRResult(text="", confidence=0.0)
        
        if len(results) == 1:
            return list(results.values())[0]
        
        self.logger.info(f"Combining results from {len(results)} engines")
        
        # Filter out failed results
        valid_results = {name: result for name, result in results.items() 
                        if result.text.strip() and result.confidence > 0.1}
        
        if not valid_results:
            # Return best of the failed results
            best_result = max(results.values(), key=lambda r: r.confidence)
            best_result.metadata["combination_method"] = "fallback_best_of_failed"
            self.logger.warning("No valid results found, using best failed result")
            return best_result
        
        if len(valid_results) == 1:
            result = list(valid_results.values())[0]
            result.metadata["combination_method"] = "single_valid"
            self.logger.info("Only one valid result found")
            return result
        
        # Multiple valid results - combine intelligently
        return self._combine_multiple_results(valid_results)
    
    def _combine_multiple_results(self, results: Dict[str, OCRResult]) -> OCRResult:
        """Combine multiple valid results with consensus detection."""
        self.logger.info(f"Combining {len(results)} valid results")
        
        # Sort by confidence descending
        sorted_results = sorted(results.items(), key=lambda x: x[1].confidence, reverse=True)
        
        # Get the best result as primary
        best_engine, best_result = sorted_results[0]
        
        # Check for text similarity among top results
        consensus_engines = [best_engine]
        consensus_confidences = [best_result.confidence]
        
        # Compare with other high-confidence results
        for engine_name, result in sorted_results[1:]:
            if result.confidence > 0.7:  # Only consider high-confidence results
                similarity = self._calculate_text_similarity(best_result.text, result.text)
                
                if similarity > 0.8:  # High similarity threshold
                    consensus_engines.append(engine_name)
                    consensus_confidences.append(result.confidence)
        
        # Create final result
        if len(consensus_engines) > 1:
            # Multiple engines agree - boost confidence slightly
            avg_confidence = sum(consensus_confidences) / len(consensus_confidences)
            boost = min(0.05, 0.01 * len(consensus_engines))  # Small boost
            final_confidence = min(1.0, avg_confidence + boost)
            combination_method = f"consensus_{len(consensus_engines)}_engines"
            self.logger.info(f"Consensus found among {len(consensus_engines)} engines")
        else:
            # Use best single result
            final_confidence = best_result.confidence
            combination_method = "best_single"
            self.logger.info("No consensus found, using best single result")
        
        # Create combined result
        combined_result = OCRResult(
            text=best_result.text,
            confidence=final_confidence,
            processing_time=sum(r.processing_time for _, r in sorted_results),
            engine_used=f"multi_engine({'+'.join(consensus_engines)})",
            regions=best_result.regions,
            bbox=best_result.bbox,
            language=best_result.language,
            metadata={
                "combination_method": combination_method,
                "engines_used": list(results.keys()),
                "consensus_engines": consensus_engines,
                "individual_confidences": {k: v.confidence for k, v in results.items()},
                "text_similarity_threshold": 0.8
            }
        )
        
        return combined_result
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple word-based text similarity."""
        if not text1 or not text2:
            return 0.0
        
        # Simple word-based similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 1.0 if text1.strip() == text2.strip() else 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def initialize_available_engines(self) -> Dict[str, bool]:
        """Auto-discover, register, and initialize available engines.
        
        Returns:
            Dictionary mapping engine names to initialization success status
        """
        results = {}
        
        # Get engine configurations from config
        engine_configs = self.config.get("engines", {})
        
        if not engine_configs:
            # Default configuration - enable all engines
            engine_configs = {
                "paddleocr": {"enabled": True},
                "easyocr": {"enabled": True},
                "tesseract": {"enabled": True},
                "trocr": {"enabled": True}
            }
            self.logger.info("Using default engine configuration")
        
        # Create and register each enabled engine
        for engine_name, engine_config in engine_configs.items():
            if not engine_config.get("enabled", False):
                self.logger.debug(f"Engine {engine_name} is disabled in config")
                continue
            
            try:
                engine_instance = self._create_engine_instance(engine_name, engine_config)
                
                if engine_instance:
                    self.register_engine(engine_name, engine_instance)
                else:
                    results[engine_name] = False
                    
            except Exception as e:
                results[engine_name] = False
                self.logger.error(f"Failed to setup {engine_name}: {e}")
        
        # Initialize all registered engines
        init_results = self.initialize_engines()
        results.update(init_results)
        
        successful_engines = [name for name, success in results.items() if success]
        self.logger.info(f"Successfully initialized {len(successful_engines)} engines: {successful_engines}")
        
        return results
    
    def _create_engine_instance(self, engine_name: str, engine_config: Dict[str, Any]) -> Optional[BaseOCREngine]:
        """Create an engine instance based on engine name and configuration."""
        try:
            if engine_name == "paddleocr":
                from ..engines.paddleocr import PaddleOCREngine
                return PaddleOCREngine(engine_config)
                
            elif engine_name == "easyocr":
                from ..engines.easyocr import EasyOCREngine
                return EasyOCREngine(engine_config)
                
            elif engine_name == "tesseract":
                from ..engines.tesseract import TesseractEngine
                return TesseractEngine(engine_config)
                
            elif engine_name == "trocr":
                from ..engines.trocr import TrOCREngine
                return TrOCREngine(engine_config)
                
            else:
                self.logger.warning(f"Unknown engine type: {engine_name}")
                return None
                
        except ImportError as e:
            self.logger.error(f"Engine {engine_name} dependencies not available: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error creating {engine_name} instance: {e}")
            return None
    
    def get_engine_info(self) -> Dict[str, Dict[str, Any]]:
        """Get basic information about all engines."""
        info = {}
        
        for name, engine in self.engines.items():
            try:
                available = engine.is_available()
                languages = engine.get_supported_languages() if available else []
            except Exception as e:
                available = False
                languages = []
                self.logger.debug(f"Error getting engine {name} info: {e}")
            
            info[name] = {
                'available': available,
                'supported_languages': languages
            }
        
        return info
    
    def cleanup(self) -> None:
        """Clean up all registered engines."""
        for name, engine in self.engines.items():
            try:
                if hasattr(engine, 'cleanup'):
                    engine.cleanup()
                self.logger.info(f"Cleaned up engine: {name}")
            except Exception as e:
                self.logger.error(f"Error cleaning up engine {name}: {e}")
        
        self.engines.clear()
        self.logger.info("Engine manager cleanup complete")