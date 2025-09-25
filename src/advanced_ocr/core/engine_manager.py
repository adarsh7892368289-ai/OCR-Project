# src/advanced_ocr/core/engine_manager.py
"""
OCR Engine Manager - Manages OCR engines and coordinates their execution.

Enhanced version with smart engine selection, strategy-based routing,
and multi-engine result combination as expected by pipeline.py.
"""

import logging
import time
from typing import Dict, List, Optional, Any, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import numpy as np

from .base_engine import BaseOCREngine
from ..types import (
    OCRResult, ProcessingOptions, ProcessingStrategy, 
    QualityMetrics, ImageQuality, TextRegion
)
from ..exceptions import (
    EngineNotAvailableError, 
    EngineInitializationError,
    ProcessingTimeoutError
)

@dataclass
class EngineStats:
    """Performance statistics for engines"""
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
    Enhanced Engine Manager with Smart Selection & Result Combination
    
    NEW CAPABILITIES:
    - Strategy-based engine selection (MINIMAL/BALANCED/ENHANCED → specific engines)
    - Language-aware engine filtering
    - Multi-engine result combination with confidence weighting
    - Quality-based engine recommendations
    - Performance tracking and adaptive selection
    """
    
    # Strategy to engine priority mapping
    STRATEGY_ENGINE_PRIORITY = {
        ProcessingStrategy.MINIMAL: ["tesseract", "easyocr", "paddleocr", "trocr"],
        ProcessingStrategy.BALANCED: ["paddleocr", "easyocr", "tesseract", "trocr"], 
        ProcessingStrategy.ENHANCED: ["trocr", "paddleocr", "easyocr", "tesseract"]
    }
    
    # Engine language capabilities (comprehensive mapping)
    ENGINE_LANGUAGE_SUPPORT = {
        "paddleocr": {
            'en', 'ch', 'ta', 'te', 'ka', 'ja', 'ko', 'hi', 'ar', 'th', 'vi',
            'ms', 'ur', 'fa', 'bg', 'uk', 'be', 'ru', 'sr', 'hr', 'ro', 'hu',
            'pl', 'cs', 'sk', 'sl', 'et', 'lv', 'lt', 'is', 'da', 'no',
            'sv', 'fi', 'fr', 'de', 'es', 'pt', 'it', 'nl', 'ca'
        },
        "easyocr": {
            'en', 'ch_sim', 'ch_tra', 'ja', 'ko', 'vi', 'ar', 'th', 'la', 'cy',
            'fr', 'de', 'es', 'pt', 'ru', 'fa', 'ur', 'hi', 'bn', 'ta', 'te',
            'kn', 'ml', 'or', 'gu', 'mr', 'ne', 'pa', 'si', 'my'
        },
        "tesseract": {
            'eng', 'chi_sim', 'chi_tra', 'jpn', 'kor', 'fra', 'deu', 'spa',
            'por', 'rus', 'ara', 'hin', 'tha', 'vie', 'ita', 'nld', 'pol',
            'tur', 'swe', 'dan', 'nor', 'fin', 'ces', 'slk', 'slv', 'hrv',
            'bul', 'ron', 'hun', 'est', 'lav', 'lit', 'ukr', 'bel', 'mkd'
        },
        "trocr": {'en'}  # TrOCR primarily English, expandable
    }
    
    # Quality-based engine recommendations
    QUALITY_ENGINE_PREFERENCES = {
        ImageQuality.EXCELLENT: ["tesseract", "paddleocr"],
        ImageQuality.GOOD: ["paddleocr", "easyocr"], 
        ImageQuality.FAIR: ["paddleocr", "trocr", "easyocr"],
        ImageQuality.POOR: ["trocr", "paddleocr"],
        ImageQuality.UNUSABLE: ["trocr"]
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize enhanced engine manager"""
        self.config = config or {}
        self.engines: Dict[str, BaseOCREngine] = {}
        self.stats: Dict[str, EngineStats] = {}
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.performance_history: Dict[str, List[float]] = {}
        self.reliability_scores: Dict[str, float] = {}
    
    # ===== CORE ENGINE MANAGEMENT (Enhanced) =====
    
    def register_engine(self, engine_name: str, engine: BaseOCREngine) -> None:
        """Register an OCR engine with enhanced tracking"""
        if not isinstance(engine, BaseOCREngine):
            raise ValueError(f"Engine must inherit from BaseOCREngine, got {type(engine)}")
        
        if engine_name in self.engines:
            self.logger.warning(f"Overwriting existing engine: {engine_name}")
        
        self.engines[engine_name] = engine
        self.stats[engine_name] = EngineStats(engine_name=engine_name)
        self.performance_history[engine_name] = []
        self.reliability_scores[engine_name] = 0.8  # Default reliability
        
        self.logger.info(f"Registered engine: {engine_name}")
    
    def initialize_engines(self) -> Dict[str, bool]:
        """Initialize all registered engines with enhanced error handling"""
        results = {}
        
        for name, engine in self.engines.items():
            try:
                success = engine.initialize()
                results[name] = success
                
                if success:
                    self.logger.info(f"Initialized engine: {name}")
                    # Test basic functionality
                    if self._test_engine_basic_functionality(name):
                        self.reliability_scores[name] = 0.9
                    else:
                        self.reliability_scores[name] = 0.5
                        self.logger.warning(f"Engine {name} initialized but failed basic test")
                else:
                    self.logger.warning(f"Failed to initialize engine: {name}")
                    self.reliability_scores[name] = 0.0
                    
            except Exception as e:
                results[name] = False
                self.reliability_scores[name] = 0.0
                self.logger.error(f"Engine {name} initialization error: {e}")
        
        return results
    
    def get_available_engines(self) -> List[str]:
        """Get names of engines that are available and reliable"""
        available = []
        
        for name, engine in self.engines.items():
            try:
                if engine.is_available() and self.reliability_scores.get(name, 0) > 0.1:
                    available.append(name)
            except Exception as e:
                self.logger.debug(f"Engine {name} availability check failed: {e}")
        
        return available
    
    # ===== NEW: SMART ENGINE SELECTION =====
    
    def select_best_engine(self, 
                          strategy: ProcessingStrategy,
                          preferred_engines: Optional[List[str]] = None,
                          languages: Optional[List[str]] = None,
                          quality_metrics: Optional[QualityMetrics] = None) -> str:
        """
        Select the single best engine based on strategy and requirements.
        
        This is what pipeline.py expects for single-engine processing.
        """
        available_engines = self.get_available_engines()
        
        if not available_engines:
            raise EngineNotAvailableError("", "No OCR engines available")
        
        # Start with strategy-based priority
        strategy_priority = self.STRATEGY_ENGINE_PRIORITY.get(strategy, 
                                                            self.STRATEGY_ENGINE_PRIORITY[ProcessingStrategy.BALANCED])
        
        # Filter by availability
        candidates = [eng for eng in strategy_priority if eng in available_engines]
        
        # Apply language filtering if specified
        if languages:
            candidates = self._filter_engines_by_language(candidates, languages)
        
        # Apply preferred engines filter if specified
        if preferred_engines:
            preferred_available = [eng for eng in preferred_engines if eng in candidates]
            if preferred_available:
                candidates = preferred_available
        
        # Apply quality-based filtering if available
        if quality_metrics and quality_metrics.quality_level:
            quality_preferred = self.QUALITY_ENGINE_PREFERENCES.get(
                quality_metrics.quality_level, candidates
            )
            quality_candidates = [eng for eng in quality_preferred if eng in candidates]
            if quality_candidates:
                candidates = quality_candidates
        
        # Select best candidate based on reliability and performance
        if not candidates:
            # Fallback to any available engine
            candidates = available_engines
        
        best_engine = self._select_most_reliable_engine(candidates)
        
        self.logger.info(f"Selected engine '{best_engine}' for strategy {strategy.value}")
        return best_engine
    
    def select_engines_for_multi_engine(self, 
                                      preferred_engines: Optional[List[str]] = None,
                                      languages: Optional[List[str]] = None,
                                      max_engines: int = 3) -> List[str]:
        """
        Select multiple engines for MULTI_ENGINE strategy.
        
        Returns diverse set of engines for consensus building.
        """
        available_engines = self.get_available_engines()
        
        if not available_engines:
            raise EngineNotAvailableError("", "No OCR engines available")
        
        # Start with all available
        candidates = available_engines.copy()
        
        # Apply language filtering
        if languages:
            candidates = self._filter_engines_by_language(candidates, languages)
        
        # Apply preferred engines filter if specified
        if preferred_engines:
            preferred_available = [eng for eng in preferred_engines if eng in candidates]
            if preferred_available:
                candidates = preferred_available
        
        # Select diverse set of engines (different strengths)
        selected_engines = self._select_diverse_engine_set(candidates, max_engines)
        
        self.logger.info(f"Selected {len(selected_engines)} engines for multi-engine: {selected_engines}")
        return selected_engines
    
    # ===== NEW: RESULT COMBINATION =====
    
    def combine_results(self, results: Dict[str, OCRResult]) -> OCRResult:
        """
        Combine results from multiple engines using intelligent consensus.
        
        This is what pipeline.py expects for MULTI_ENGINE strategy.
        """
        if not results:
            return OCRResult(text="", confidence=0.0)
        
        if len(results) == 1:
            return list(results.values())[0]
        
        # Filter out failed results
        valid_results = {name: result for name, result in results.items() 
                        if result.text.strip() and result.confidence > 0.1}
        
        if not valid_results:
            # Return best of the failed results
            best_result = max(results.values(), key=lambda r: r.confidence)
            best_result.metadata["combination_method"] = "fallback_best_of_failed"
            return best_result
        
        if len(valid_results) == 1:
            result = list(valid_results.values())[0]
            result.metadata["combination_method"] = "single_valid"
            return result
        
        # Multi-result combination
        return self._combine_multiple_valid_results(valid_results)
    
    def _combine_multiple_valid_results(self, results: Dict[str, OCRResult]) -> OCRResult:
        """Combine multiple valid results using confidence weighting and text similarity"""
        # Calculate reliability-weighted confidence for each result
        weighted_results = []
        for engine_name, result in results.items():
            reliability = self.reliability_scores.get(engine_name, 0.5)
            weighted_confidence = result.confidence * reliability
            weighted_results.append((engine_name, result, weighted_confidence))
        
        # Sort by weighted confidence
        weighted_results.sort(key=lambda x: x[2], reverse=True)
        
        # Use highest weighted confidence result as primary
        primary_engine, primary_result, primary_weighted_conf = weighted_results[0]
        
        # Check for text similarity consensus
        consensus_texts = [primary_result.text]
        consensus_confidences = [primary_result.confidence]
        consensus_engines = [primary_engine]
        
        # Compare with other results
        for engine_name, result, weighted_conf in weighted_results[1:]:
            # Simple text similarity check (can be enhanced)
            similarity = self._calculate_text_similarity(primary_result.text, result.text)
            
            if similarity > 0.7:  # High similarity
                consensus_texts.append(result.text)
                consensus_confidences.append(result.confidence)
                consensus_engines.append(engine_name)
        
        # Create combined result
        if len(consensus_engines) > 1:
            # Consensus found - boost confidence
            avg_confidence = sum(consensus_confidences) / len(consensus_confidences)
            consensus_bonus = min(0.1, 0.02 * len(consensus_engines))
            final_confidence = min(1.0, avg_confidence + consensus_bonus)
            combination_method = f"consensus_{len(consensus_engines)}_engines"
        else:
            # No consensus - use primary result
            final_confidence = primary_result.confidence
            combination_method = "weighted_best_single"
        
        # Create final result
        combined_result = OCRResult(
            text=primary_result.text,
            confidence=final_confidence,
            processing_time=sum(r.processing_time for _, r, _ in weighted_results),
            engine_used=f"multi_engine_{'+'.join(consensus_engines)}",
            regions=primary_result.regions,
            bbox=primary_result.bbox,
            language=primary_result.language,
            metadata={
                "combination_method": combination_method,
                "engines_used": consensus_engines,
                "individual_results": {
                    engine: {"text": result.text, "confidence": result.confidence}
                    for engine, result, _ in weighted_results
                },
                "consensus_engines": consensus_engines,
                "text_similarity_threshold": 0.7
            }
        )
        
        return combined_result
    
    # ===== ENHANCED EXECUTION =====
    
    def execute_engine(self, engine_name: str, image: np.ndarray, 
                      options: Optional[ProcessingOptions] = None) -> OCRResult:
        """Execute single engine with enhanced tracking and error handling"""
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
            
            # Ensure result metadata is complete
            if not result.engine_used:
                result.engine_used = engine_name
            if not result.processing_time:
                result.processing_time = processing_time
            
            # Update performance tracking
            self._update_engine_performance(engine_name, result, processing_time)
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            stats.total_calls += 1
            stats.total_time += processing_time
            
            # Update reliability score on failure
            self.reliability_scores[engine_name] *= 0.95
            
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
        """Execute multiple engines with enhanced coordination"""
        results = {}
        available_engines = self.get_available_engines()
        
        # Filter to only available engines
        engines_to_run = [name for name in engine_names if name in available_engines]
        
        if not engines_to_run:
            self.logger.warning("No requested engines are available")
            return results
        
        # Enhanced early termination logic
        early_termination = (options and options.early_termination) or False
        early_threshold = (options and options.early_termination_threshold) or 0.95
        
        if use_parallel and len(engines_to_run) > 1:
            results = self._execute_engines_parallel(engines_to_run, image, options, 
                                                   early_termination, early_threshold)
        else:
            results = self._execute_engines_sequential(engines_to_run, image, options,
                                                     early_termination, early_threshold)
        
        return results
    
    # ===== HELPER METHODS =====
    
    def _filter_engines_by_language(self, engines: List[str], languages: List[str]) -> List[str]:
        """Filter engines that support required languages"""
        if not languages:
            return engines
        
        filtered_engines = []
        for engine in engines:
            engine_langs = self.ENGINE_LANGUAGE_SUPPORT.get(engine, set())
            
            # Check if engine supports any of the required languages
            if any(lang in engine_langs for lang in languages):
                filtered_engines.append(engine)
        
        return filtered_engines or engines  # Fallback to original list if no matches
    
    def _select_most_reliable_engine(self, candidates: List[str]) -> str:
        """Select most reliable engine from candidates"""
        if not candidates:
            raise EngineNotAvailableError("", "No candidate engines available")
        
        # Score engines by reliability and recent performance
        scored_engines = []
        for engine in candidates:
            reliability = self.reliability_scores.get(engine, 0.5)
            recent_performance = self._get_recent_performance_score(engine)
            overall_score = reliability * 0.6 + recent_performance * 0.4
            scored_engines.append((engine, overall_score))
        
        # Select best scoring engine
        best_engine = max(scored_engines, key=lambda x: x[1])[0]
        return best_engine
    
    def _select_diverse_engine_set(self, candidates: List[str], max_engines: int) -> List[str]:
        """Select diverse set of engines for multi-engine processing"""
        if len(candidates) <= max_engines:
            return candidates
        
        # Prioritize engines with different strengths
        diversity_priority = ["paddleocr", "trocr", "easyocr", "tesseract"]
        
        selected = []
        for engine in diversity_priority:
            if engine in candidates and len(selected) < max_engines:
                selected.append(engine)
        
        # Fill remaining slots with highest reliability
        remaining_candidates = [e for e in candidates if e not in selected]
        if remaining_candidates and len(selected) < max_engines:
            remaining_needed = max_engines - len(selected)
            remaining_scored = [(e, self.reliability_scores.get(e, 0.5)) 
                              for e in remaining_candidates]
            remaining_scored.sort(key=lambda x: x[1], reverse=True)
            
            for engine, _ in remaining_scored[:remaining_needed]:
                selected.append(engine)
        
        return selected
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity score"""
        if not text1 or not text2:
            return 0.0
        
        # Simple character-based similarity (can be enhanced with more sophisticated methods)
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 1.0 if text1.strip() == text2.strip() else 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _update_engine_performance(self, engine_name: str, result: OCRResult, processing_time: float):
        """Update engine performance tracking"""
        stats = self.stats[engine_name]
        stats.total_calls += 1
        stats.total_time += processing_time
        
        # Track success
        if result.text and result.text.strip() and result.confidence > 0:
            stats.successful_calls += 1
            
            # Update running average confidence
            if stats.successful_calls == 1:
                stats.avg_confidence = result.confidence
            else:
                n = stats.successful_calls
                stats.avg_confidence = ((n-1) * stats.avg_confidence + result.confidence) / n
            
            # Update reliability score positively
            self.reliability_scores[engine_name] = min(1.0, 
                self.reliability_scores[engine_name] * 0.98 + 0.02)
        
        # Track recent performance
        self.performance_history[engine_name].append(result.confidence)
        if len(self.performance_history[engine_name]) > 10:
            self.performance_history[engine_name].pop(0)
    
    def _get_recent_performance_score(self, engine_name: str) -> float:
        """Get recent performance score for engine"""
        history = self.performance_history.get(engine_name, [])
        if not history:
            return 0.5  # Default score
        
        return sum(history) / len(history)
    
    def _test_engine_basic_functionality(self, engine_name: str) -> bool:
        """Test basic engine functionality with a simple test image"""
        try:
            # Create a simple test image (white background with black text)
            test_image = np.ones((100, 300, 3), dtype=np.uint8) * 255
            # Add some simple text (this is a very basic test)
            
            engine = self.engines[engine_name]
            result = engine.extract_text(test_image)
            
            # Consider it working if it doesn't crash and returns a result
            return isinstance(result, OCRResult)
            
        except Exception as e:
            self.logger.debug(f"Basic functionality test failed for {engine_name}: {e}")
            return False
    
    def _execute_engines_parallel(self, engines: List[str], image: np.ndarray,
                                options: Optional[ProcessingOptions],
                                early_termination: bool, early_threshold: float) -> Dict[str, OCRResult]:
        """Execute engines in parallel"""
        results = {}
        max_workers = min(4, len(engines))
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_engine = {
                executor.submit(self.execute_engine, name, image, options): name
                for name in engines
            }
            
            for future in as_completed(future_to_engine):
                engine_name = future_to_engine[future]
                try:
                    result = future.result()
                    results[engine_name] = result
                    
                    # Enhanced early termination
                    if (early_termination and result.confidence >= early_threshold 
                        and result.text.strip()):
                        self.logger.info(f"Early termination triggered by {engine_name} "
                                       f"(confidence: {result.confidence:.3f})")
                        
                        # Cancel remaining futures
                        for remaining_future in future_to_engine:
                            if remaining_future != future:
                                remaining_future.cancel()
                        break
                        
                except Exception as e:
                    self.logger.error(f"Parallel execution error for {engine_name}: {e}")
                    results[engine_name] = OCRResult("", 0.0, engine_used=engine_name,
                                                   metadata={"error": str(e)})
        
        return results
    
    def _execute_engines_sequential(self, engines: List[str], image: np.ndarray,
                                  options: Optional[ProcessingOptions],
                                  early_termination: bool, early_threshold: float) -> Dict[str, OCRResult]:
        """Execute engines sequentially"""
        results = {}
        
        for engine_name in engines:
            try:
                result = self.execute_engine(engine_name, image, options)
                results[engine_name] = result
                
                # Early termination check
                if (early_termination and result.confidence >= early_threshold 
                    and result.text.strip()):
                    self.logger.info(f"Early termination after {engine_name} "
                                   f"(confidence: {result.confidence:.3f})")
                    break
                    
            except Exception as e:
                self.logger.error(f"Sequential execution error for {engine_name}: {e}")
                results[engine_name] = OCRResult("", 0.0, engine_used=engine_name,
                                               metadata={"error": str(e)})
        
        return results
    
    # ===== PIPELINE INTEGRATION METHODS =====
    
    def initialize_available_engines(self) -> Dict[str, bool]:
        """
        Auto-discover, register, and initialize available engines.
        Enhanced version for pipeline integration.
        """
        results = {}
        
        # Get engine configurations from config
        engine_configs = self.config.get("engines", {})
        
        if not engine_configs:
            # Default engine configuration if none provided
            engine_configs = {
                "paddleocr": {"enabled": True},
                "easyocr": {"enabled": True},
                "tesseract": {"enabled": True},
                "trocr": {"enabled": True}
            }
            self.logger.info("Using default engine configuration")
        
        # Import and register each enabled engine
        for engine_name, engine_config in engine_configs.items():
            if not engine_config.get("enabled", False):
                self.logger.debug(f"Engine {engine_name} is disabled in config")
                continue
            
            try:
                engine_instance = self._create_engine_instance(engine_name, engine_config)
                
                if engine_instance:
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

    def _create_engine_instance(self, engine_name: str, engine_config: Dict[str, Any]) -> Optional[BaseOCREngine]:
        """Create an engine instance based on engine name and configuration"""
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
        """Get comprehensive information about all engines"""
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
                'avg_processing_time': stats.avg_time,
                'reliability_score': self.reliability_scores.get(name, 0.0),
                'recent_performance': self._get_recent_performance_score(name)
            }
        
        return info
    
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
        self.performance_history.clear()
        self.reliability_scores.clear()
        self.logger.info("Engine manager cleanup complete")