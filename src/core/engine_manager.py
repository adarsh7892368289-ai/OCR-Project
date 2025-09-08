# src/core/engine_manager.py

from typing import List, Dict, Any, Optional, Union
import numpy as np
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import time

from .base_engine import BaseOCREngine, DocumentResult, TextType, BoundingBox, OCRResult, DocumentStructure, TextRegion
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
        
        # FIXED: Get configuration values with proper defaults
        self.selection_strategy = config.get("engine_selection_strategy", "adaptive")
        self.parallel_processing = config.get("system.parallel_processing", True)
        self.max_workers = config.get("system.max_workers", 3)
        
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
                                 quality_priority: bool) -> Optional[str]:
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
        best_engine = max(engine_scores.items(), key=lambda x: x[1])[0]
        return best_engine
    
    def _performance_based_selection(self, text_type: TextType, language: str) -> Optional[str]:
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
        
        return best_engine or (self.initialized_engines[0] if self.initialized_engines else None)
    
    def _priority_based_selection(self, text_type: TextType) -> Optional[str]:
        """Select engine based on predefined priorities"""
        for engine_name in self.engine_priorities[text_type]:
            if engine_name in self.initialized_engines:
                return engine_name
        
        return self.initialized_engines[0] if self.initialized_engines else None
    
    def _round_robin_selection(self) -> Optional[str]:
        """Simple round-robin selection"""
        if not self.initialized_engines:
            return None
            
        if not hasattr(self, '_round_robin_index'):
            self._round_robin_index = 0
        
        engine = self.initialized_engines[self._round_robin_index % len(self.initialized_engines)]
        self._round_robin_index += 1
        return engine
    
    def process_image(self, 
                     image: np.ndarray, 
                     engine_name: Optional[str] = None,
                     **kwargs) -> DocumentResult:
        """
        FIXED: Process a single image with compatible return structure
        """
        if not self.initialized_engines:
            self.logger.error("No engines initialized")
            raise RuntimeError("No OCR engines available")
        
        # Select engine
        selected_engine_name = engine_name or self.select_best_engine(image, **kwargs)
        if not selected_engine_name:
            raise RuntimeError("Failed to select OCR engine")
        
        selected_engine = self.engines[selected_engine_name]
        
        start_time = time.time()
        
        try:
            # Process with engine
            result = selected_engine.process_image(image, **kwargs)
            processing_time = time.time() - start_time
            
            # FIXED: Ensure result compatibility
            if result is None:
                # Create empty result
                result = DocumentResult(
                    pages=[],
                    metadata={"engine": selected_engine_name},
                    processing_time=processing_time,
                    engine_name=selected_engine_name,
                    confidence_score=0.0
                )
            
            # Update performance
            self._update_performance(selected_engine_name, result, processing_time, True)
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"Engine {selected_engine_name} failed: {e}")
            
            # Update performance for failure
            self._update_performance(selected_engine_name, None, processing_time, False)
            
            # Try fallback
            return self._try_fallback(image, selected_engine_name, **kwargs)
    
    def process_regions(self, 
                        regions_data: List[Dict[str, Any]],
                        full_image_stats: Dict[str, Any],
                        text_type: Optional[TextType] = None,
                        engine_name: Optional[str] = None,
                        **kwargs) -> DocumentResult:
        """
        FIXED: Process regions with proper error handling and return structure
        """
        if not regions_data:
            self.logger.warning("No regions to process")
            return DocumentResult(
                pages=[],
                metadata={"engine": engine_name or "none", "image_stats": full_image_stats},
                processing_time=0.0,
                engine_name=engine_name or "none",
                confidence_score=0.0
            )
        
        start_time = time.time()
        
        # Select engine
        selected_engine_name = engine_name or self.select_best_engine(
            regions_data[0]['image'], text_type, **kwargs
        )
        
        if not selected_engine_name or selected_engine_name not in self.initialized_engines:
            raise ValueError(f"Engine {selected_engine_name} not available")
            
        selected_engine = self.engines[selected_engine_name]
        
        self.logger.info(f"Processing {len(regions_data)} regions with {selected_engine_name}")
        
        all_ocr_results: List[OCRResult] = []
        total_confidence = 0.0
        successful_regions = 0
        
        # FIXED: Process regions with proper error handling
        for i, region_data in enumerate(regions_data):
            try:
                region_image = region_data['image']
                region_metadata = region_data.get('metadata', {})
                
                # Process single region
                ocr_result = self._process_single_region(selected_engine, region_image, region_metadata)
                
                if ocr_result and ocr_result.text.strip():
                    # Create text region for compatibility
                    bbox_data = region_metadata.get('bbox', {})
                    bbox = BoundingBox(
                        x=bbox_data.get('x', 0),
                        y=bbox_data.get('y', 0),
                        width=bbox_data.get('width', region_image.shape[1]),
                        height=bbox_data.get('height', region_image.shape[0]),
                        confidence=ocr_result.confidence
                    )
                    
                    # FIXED: Create OCRResult with correct constructor parameters
                    final_result = OCRResult(
                        text=ocr_result.text,
                        confidence=ocr_result.confidence,
                        regions=[TextRegion(
                            text=ocr_result.text,
                            confidence=ocr_result.confidence,
                            bbox=bbox
                        )],
                        processing_time=ocr_result.processing_time,
                        bbox=bbox,
                        metadata={"region_index": i, "engine": selected_engine_name}
                    )
                    
                    all_ocr_results.append(final_result)
                    total_confidence += ocr_result.confidence
                    successful_regions += 1
                    
            except Exception as e:
                self.logger.error(f"Failed to process region {i}: {e}")
                continue
        
        processing_time = time.time() - start_time
        
        # FIXED: Create properly formatted DocumentResult
        full_text = " ".join([r.text for r in all_ocr_results])
        avg_confidence = total_confidence / successful_regions if successful_regions > 0 else 0.0
        
        final_result = DocumentResult(
            pages=all_ocr_results,  # FIXED: Use 'pages' parameter name
            metadata={
                "engine": selected_engine_name,
                "image_stats": full_image_stats,
                "regions_processed": len(regions_data),
                "successful_regions": successful_regions
            },
            processing_time=processing_time,
            engine_name=selected_engine_name,
            confidence_score=avg_confidence
        )
        
        # Update performance
        self._update_performance(selected_engine_name, final_result, processing_time, successful_regions > 0)
        
        return final_result
        
    def _process_single_region(self, engine: BaseOCREngine, image: np.ndarray, metadata: Dict[str, Any]) -> Optional[OCRResult]:
        """FIXED: Process single region with proper error handling"""
        try:
            # Process image with engine
            doc_result = engine.process_image(image)
            
            if doc_result and hasattr(doc_result, 'pages') and doc_result.pages:
                return doc_result.pages[0]  # Return first OCR result
            elif doc_result and hasattr(doc_result, 'results') and doc_result.results:
                return doc_result.results[0]  # Backward compatibility
            else:
                # Create minimal result if engine returns nothing useful
                return OCRResult(
                    text="",
                    confidence=0.0,
                    regions=[],
                    processing_time=0.0,
                    metadata={"engine": engine.name, "status": "no_text"}
                )
                
        except Exception as e:
            self.logger.error(f"Engine {engine.name} failed on region: {e}")
            return None
    
    def process_regions_multi_engine(self,
                                 regions_data: List[Dict[str, Any]],
                                 full_image_stats: Dict[str, Any],
                                 engine_names: Optional[List[str]] = None,
                                 **kwargs) -> Dict[str, DocumentResult]:
        """Process regions with multiple engines for comparison"""
        if not regions_data:
            return {}

        if engine_names is None:
            engine_names = self.initialized_engines[:3]  # Use top 3 engines
        
        results = {}
        
        for engine_name in engine_names:
            if engine_name in self.initialized_engines:
                try:
                    result = self.process_regions(
                        regions_data,
                        full_image_stats,
                        engine_name=engine_name,
                        **kwargs
                    )
                    results[engine_name] = result
                except Exception as e:
                    self.logger.error(f"Multi-engine processing failed for {engine_name}: {e}")
        
        return results
    
    def _try_fallback(self, image: np.ndarray, failed_engine: str, **kwargs) -> DocumentResult:
        """FIXED: Try fallback engines with proper return structure"""
        available_engines = [e for e in self.initialized_engines if e != failed_engine]
        
        if not available_engines:
            # Return empty result
            return DocumentResult(
                pages=[],
                metadata={"engine": "failed", "fallback_attempted": True},
                processing_time=0.0,
                engine_name="failed",
                confidence_score=0.0
            )
        
        # Try the best alternative
        fallback_engine = available_engines[0]
        self.logger.info(f"Trying fallback engine: {fallback_engine}")
        
        try:
            return self.process_image(image, fallback_engine, **kwargs)
        except Exception as e:
            self.logger.error(f"Fallback engine {fallback_engine} also failed: {e}")
            return DocumentResult(
                pages=[],
                metadata={"engine": "failed", "fallback_failed": True},
                processing_time=0.0,
                engine_name="failed",
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
        # FIXED: Get engine list from config properly
        engine_configs = self.config.get("engines", {})
        enabled_engines = [name for name, config in engine_configs.items() 
                          if config.get("enabled", True)]
        
        if enabled_engines:
            self.initialize_engines(enabled_engines)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup_engines()

# Alias for backward compatibility  
EngineManager = OCREngineManager

# Export main classes
__all__ = [
    'OCREngineManager',
    'EngineManager',
    'EnginePerformance'
]