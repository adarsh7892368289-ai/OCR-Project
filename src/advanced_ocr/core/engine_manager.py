# src/advanced_ocr/core/engine_manager.py
"""
OCR Engine Manager - Manages OCR engines and coordinates their execution.

This module handles engine registration, initialization, and execution coordination.
It does NOT implement OCR algorithms, image preprocessing, or decision logic.
"""

import logging
import time
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import numpy as np

from .base_engine import BaseEngine
from ..types import OCRResult, ProcessingOptions
from ..exceptions import (
    EngineNotAvailableError, 
    EngineInitializationError,
    ProcessingTimeoutError
)

@dataclass
class EngineStats:
    """Simple performance statistics for engines"""
    engine_name: str
    total_calls: int = 0
    successful_calls: int = 0
    total_time: float = 0.0
    avg_confidence: float = 0.0
    
    @property
    def success_rate(self) -> float:
        return (self.successful_calls / self.total_calls * 100) if self.total_calls > 0 else 0.0
    
    @property 
    def avg_time(self) -> float:
        return self.total_time / self.total_calls if self.total_calls > 0 else 0.0


class EngineManager:
    """
    Manages OCR engines and coordinates their execution.
    
    Responsibilities:
    - Register and initialize OCR engines
    - Provide engine availability information
    - Execute engines with proper error handling
    - Track basic performance statistics
    
    Does NOT:
    - Decide which engine to use (pipeline's job)
    - Enhance images (preprocessor's job) 
    - Analyze quality (analyzer's job)
    - Make "best result" decisions (pipeline's job)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize engine manager with optional configuration"""
        self.config = config or {}
        self.engines: Dict[str, BaseEngine] = {}
        self.stats: Dict[str, EngineStats] = {}
        self.logger = logging.getLogger(__name__)
    
    def register_engine(self, engine_name: str, engine: BaseEngine) -> None:
        """
        Register an OCR engine.
        
        Args:
            engine_name: Unique identifier for the engine
            engine: Engine instance implementing BaseOCREngine
            
        Raises:
            ValueError: If engine is invalid
        """
        if not isinstance(engine, BaseEngine):
            raise ValueError(f"Engine must inherit from BaseOCREngine, got {type(engine)}")
        
        if engine_name in self.engines:
            self.logger.warning(f"Overwriting existing engine: {engine_name}")
        
        self.engines[engine_name] = engine
        self.stats[engine_name] = EngineStats(engine_name=engine_name)
        self.logger.info(f"Registered engine: {engine_name}")
    
    def initialize_engines(self) -> Dict[str, bool]:
        """
        Initialize all registered engines.
        
        Returns:
            Dict mapping engine names to initialization success status
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
        """
        Get names of engines that are available for use.
        
        Returns:
            List of available engine names
        """
        available = []
        
        for name, engine in self.engines.items():
            try:
                if engine.is_available():
                    available.append(name)
            except Exception as e:
                self.logger.debug(f"Engine {name} availability check failed: {e}")
        
        return available
    
    def get_engine_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about all engines including availability and stats.
        
        Returns:
            Dict with engine info including availability and performance stats
        """
        info = {}
        
        for name, engine in self.engines.items():
            stats = self.stats[name]
            
            try:
                available = engine.is_available()
                languages = engine.get_supported_languages() if available else []
            except Exception as e:
                available = False
                languages = []
                self.logger.debug(f"Error getting engine {name} info: {e}")
            
            info[name] = {
                'available': available,
                'supported_languages': languages,
                'total_calls': stats.total_calls,
                'success_rate': stats.success_rate,
                'avg_confidence': stats.avg_confidence,
                'avg_processing_time': stats.avg_time
            }
        
        return info
    
    def execute_engine(self, engine_name: str, image: np.ndarray, 
                      options: Optional[ProcessingOptions] = None) -> OCRResult:
        """
        Execute a specific engine on an image.
        
        Args:
            engine_name: Name of engine to execute
            image: Input image as numpy array
            options: Processing options to pass to engine
            
        Returns:
            OCRResult from the engine
            
        Raises:
            EngineNotAvailableError: If engine is not available
        """
        if engine_name not in self.engines:
            raise EngineNotAvailableError(
                engine_name, 
                f"Engine not registered. Available: {list(self.engines.keys())}"
            )
        
        engine = self.engines[engine_name]
        
        if not engine.is_available():
            raise EngineNotAvailableError(engine_name, "Engine not initialized or unavailable")
        
        start_time = time.time()
        stats = self.stats[engine_name]
        
        try:
            # Execute the engine
            result = engine.extract_text(image)
            processing_time = time.time() - start_time
            
            # Ensure result has required metadata
            if not result.engine_used:
                result.engine_used = engine_name
            if not result.processing_time:
                result.processing_time = processing_time
            
            # Update statistics
            stats.total_calls += 1
            stats.total_time += processing_time
            
            # Track success if we got meaningful text
            if result.text and result.text.strip() and result.confidence > 0:
                stats.successful_calls += 1
                # Update running average confidence
                if stats.successful_calls == 1:
                    stats.avg_confidence = result.confidence
                else:
                    n = stats.successful_calls
                    stats.avg_confidence = ((n-1) * stats.avg_confidence + result.confidence) / n
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            stats.total_calls += 1
            stats.total_time += processing_time
            
            self.logger.error(f"Engine {engine_name} execution failed: {e}")
            
            # Return empty result rather than raising - let pipeline handle
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
        """
        Execute multiple engines on the same image.
        
        Args:
            engine_names: List of engine names to execute
            image: Input image as numpy array  
            options: Processing options
            use_parallel: Whether to run engines in parallel
            
        Returns:
            Dict mapping engine names to their OCRResults
        """
        results = {}
        available_engines = self.get_available_engines()
        
        # Filter to only available engines
        engines_to_run = [name for name in engine_names if name in available_engines]
        
        if not engines_to_run:
            self.logger.warning("No requested engines are available")
            return results
        
        # Handle early termination if enabled
        early_termination = (options and options.early_termination) or False
        early_threshold = (options and options.early_termination_threshold) or 0.95
        
        if use_parallel and len(engines_to_run) > 1:
            # Parallel execution
            max_workers = min(4, len(engines_to_run))
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_engine = {
                    executor.submit(self.execute_engine, name, image, options): name
                    for name in engines_to_run
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_engine):
                    engine_name = future_to_engine[future]
                    try:
                        result = future.result()
                        results[engine_name] = result
                        
                        # Early termination check
                        if (early_termination and result.confidence >= early_threshold):
                            self.logger.info(f"Early termination triggered by {engine_name} "
                                           f"(confidence: {result.confidence:.3f})")
                            # Cancel remaining futures
                            for remaining_future in future_to_engine:
                                if remaining_future != future:
                                    remaining_future.cancel()
                            break
                            
                    except Exception as e:
                        self.logger.error(f"Parallel execution error for {engine_name}: {e}")
                        results[engine_name] = OCRResult("", 0.0, engine_used=engine_name)
        else:
            # Sequential execution
            for engine_name in engines_to_run:
                try:
                    result = self.execute_engine(engine_name, image, options)
                    results[engine_name] = result
                    
                    # Early termination check
                    if (early_termination and result.confidence >= early_threshold):
                        self.logger.info(f"Early termination after {engine_name} "
                                       f"(confidence: {result.confidence:.3f})")
                        break
                        
                except Exception as e:
                    self.logger.error(f"Sequential execution error for {engine_name}: {e}")
                    results[engine_name] = OCRResult("", 0.0, engine_used=engine_name)
        
        return results
    
    def cleanup(self) -> None:
        """Clean up all registered engines and reset state"""
        for name, engine in self.engines.items():
            try:
                if hasattr(engine, 'cleanup'):
                    engine.cleanup()
                self.logger.info(f"Cleaned up engine: {name}")
            except Exception as e:
                self.logger.error(f"Error cleaning up engine {name}: {e}")
        
        self.engines.clear()
        self.stats.clear()
        self.logger.info("Engine manager cleanup complete")
        
    def initialize_available_engines(self) -> Dict[str, bool]:
        """
        Auto-discover, register, and initialize available engines based on config.
        
        This method handles the complete engine setup process that the pipeline expects.
        
        Returns:
            Dict mapping engine names to initialization success status
        """
        results = {}
        
        # Get engine configurations from config
        engine_configs = self.config.get("engines", {})
        
        if not engine_configs:
            self.logger.warning("No engine configurations found")
            return results
        
        # Import and register each enabled engine
        for engine_name, engine_config in engine_configs.items():
            if not engine_config.get("enabled", False):
                self.logger.debug(f"Engine {engine_name} is disabled in config")
                continue
            
            try:
                # Import the appropriate engine class
                engine_instance = self._create_engine_instance(engine_name, engine_config)
                
                if engine_instance:
                    # Register the engine
                    self.register_engine(engine_name, engine_instance)
                    self.logger.info(f"Registered engine: {engine_name}")
                else:
                    results[engine_name] = False
                    self.logger.error(f"Failed to create {engine_name} instance")
                    
            except Exception as e:
                results[engine_name] = False
                self.logger.error(f"Failed to setup {engine_name}: {e}")
        
        # Initialize all registered engines
        init_results = self.initialize_engines()
        results.update(init_results)
        
        successful_engines = [name for name, success in results.items() if success]
        self.logger.info(f"Successfully initialized {len(successful_engines)} engines: {successful_engines}")
        
        return results

    def _create_engine_instance(self, engine_name: str, engine_config: Dict[str, Any]) -> Optional[BaseEngine]:
        """
        Create an engine instance based on engine name and configuration.
        
        Args:
            engine_name: Name of the engine to create
            engine_config: Configuration for the engine
            
        Returns:
            Engine instance or None if creation failed
        """
        try:
            if engine_name == "paddleocr":
                from ..engines.paddleocr import PaddleOCR
                return PaddleOCR(engine_config)
                
            elif engine_name == "easyocr":
                from ..engines.easyocr import EasyOCR
                return EasyOCR(engine_config)
                
            elif engine_name == "tesseract":
                from ..engines.tesseract import Tesseract
                return Tesseract(engine_config)
                
            elif engine_name == "trocr":
                from ..engines.trocr import TrOCR
                return TrOCR(engine_config)
                
            else:
                self.logger.warning(f"Unknown engine type: {engine_name}")
                return None
                
        except ImportError as e:
            self.logger.error(f"Engine {engine_name} dependencies not available: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Error creating {engine_name} instance: {e}")
            return None