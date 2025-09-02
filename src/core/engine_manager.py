# src/core/engine_manager.py

import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import concurrent.futures
import time
import threading
from collections import defaultdict
import statistics

from .base_engine import BaseOCREngine, OCRResult, DocumentResult
from ..engines.tesseract_engine import TesseractEngine
from ..engines.easyocr_engine import EasyOCREngine
from ..engines.trocr_engine import TrOCREngine

class EngineManager:
    """Manages multiple OCR engines and combines their results"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.engines: Dict[str, BaseOCREngine] = {}
        self.engine_weights = {}
        self.parallel_processing = self.config.get("parallel_processing", True)
        self.max_workers = self.config.get("max_workers", 3)
        self.confidence_threshold = self.config.get("confidence_threshold", 0.5)
        self.voting_strategy = self.config.get("voting_strategy", "weighted_confidence")
        
    def initialize_engines(self, engine_configs: Dict[str, Dict[str, Any]]) -> Dict[str, bool]:
        """Initialize specified OCR engines"""
        initialization_results = {}
        
        for engine_name, engine_config in engine_configs.items():
            try:
                print(f"Initializing {engine_name}...")
                
                if engine_name.lower() == "tesseract":
                    engine = TesseractEngine(engine_config)
                elif engine_name.lower() == "easyocr":
                    engine = EasyOCREngine(engine_config)
                elif engine_name.lower() == "trocr":
                    engine = TrOCREngine(engine_config)
                else:
                    print(f"Unknown engine: {engine_name}")
                    initialization_results[engine_name] = False
                    continue
                    
                success = engine.initialize()
                if success:
                    self.engines[engine_name] = engine
                    # Set default weights based on engine characteristics
                    self.engine_weights[engine_name] = self._get_default_weight(engine_name)
                    
                initialization_results[engine_name] = success
                print(f"{engine_name} initialization: {'Success' if success else 'Failed'}")
                
            except Exception as e:
                print(f"Error initializing {engine_name}: {e}")
                initialization_results[engine_name] = False
                
        return initialization_results
        
    def _get_default_weight(self, engine_name: str) -> float:
        """Get default weight for engine based on its characteristics"""
        weights = {
            "tesseract": 1.2,  # High weight for printed text
            "easyocr": 1.0,    # Balanced weight
            "trocr": 1.5,      # Higher weight for handwritten text
            "paddleocr": 1.1,  # Good for multilingual
            "keras_ocr": 0.9   # Lower weight, used as backup
        }
        return weights.get(engine_name.lower(), 1.0)
        
    def process_image(self, image: np.ndarray, **kwargs) -> DocumentResult:
        """Process image with all available engines"""
        if not self.engines:
            raise RuntimeError("No OCR engines initialized")
            
        start_time = time.time()
        
        # Analyze image to determine optimal engine selection
        image_analysis = self._analyze_image(image)
        selected_engines = self._select_engines(image_analysis, **kwargs)
        
        print(f"Selected engines: {list(selected_engines.keys())}")
        
        # Process with selected engines
        if self.parallel_processing and len(selected_engines) > 1:
            engine_results = self._process_parallel(image, selected_engines, **kwargs)
        else:
            engine_results = self._process_sequential(image, selected_engines, **kwargs)
            
        # Combine results
        final_result = self._combine_results(engine_results, image_analysis)
        
        # Update processing time
        total_time = time.time() - start_time
        final_result.processing_time = total_time
        
        print(f"Total processing time: {total_time:.2f}s")
        print(f"Final confidence: {final_result.confidence_score:.3f}")
        
        return final_result
        
    def _analyze_image(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze image characteristics to guide engine selection"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        analysis = {
            "width": image.shape[1],
            "height": image.shape[0],
            "aspect_ratio": image.shape[1] / image.shape[0],
            "mean_brightness": np.mean(gray),
            "brightness_std": np.std(gray),
            "has_handwritten": self._detect_handwritten_probability(gray),
            "text_density": self._estimate_text_density(gray),
            "image_quality": self._assess_image_quality(gray)
        }
        
        return analysis
        
    def _detect_handwritten_probability(self, gray_image: np.ndarray) -> float:
        """Estimate probability of handwritten text presence"""
        # Use edge detection and contour analysis
        edges = cv2.Canny(gray_image, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return 0.0
            
        # Analyze contour characteristics
        irregularity_scores = []
        for contour in contours:
            if cv2.contourArea(contour) > 100:  # Filter small contours
                # Calculate contour irregularity
                perimeter = cv2.arcLength(contour, True)
                area = cv2.contourArea(contour)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    irregularity = 1.0 - circularity
                    irregularity_scores.append(irregularity)
                    
        if irregularity_scores:
            avg_irregularity = np.mean(irregularity_scores)
            # Higher irregularity suggests handwritten text
            handwritten_prob = min(1.0, avg_irregularity * 2)
        else:
            handwritten_prob = 0.5  # Default when uncertain
            
        return handwritten_prob
        
    def _estimate_text_density(self, gray_image: np.ndarray) -> float:
        """Estimate text density in the image"""
        # Use morphological operations to detect text regions
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morph = cv2.morphologyEx(gray_image, cv2.MORPH_CLOSE, kernel)
        
        # Threshold
        _, thresh = cv2.threshold(morph, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Calculate text area ratio
        text_pixels = np.sum(thresh < 128)  # Assuming dark text on light background
        total_pixels = thresh.size
        
        return text_pixels / total_pixels
        
    def _assess_image_quality(self, gray_image: np.ndarray) -> float:
        """Assess overall image quality for OCR"""
        # Calculate Laplacian variance (focus measure)
        laplacian_var = cv2.Laplacian(gray_image, cv2.CV_64F).var()
        
        # Normalize to 0-1 range (higher is better)
        quality_score = min(1.0, laplacian_var / 1000.0)
        
        return quality_score
        
    def _select_engines(self, image_analysis: Dict[str, Any], **kwargs):
        selected = {}
        handwritten_prob = image_analysis.get("has_handwritten", 0.5)
        
        # Always include TrOCR for any text (it works well for printed text too)
        if "trocr" in self.engines:
            selected["trocr"] = self.engines["trocr"]
        
        # Add EasyOCR for mixed content
        if "easyocr" in self.engines:
            selected["easyocr"] = self.engines["easyocr"]
        
        # Add Tesseract only for clearly printed text
        if handwritten_prob < 0.5 and "tesseract" in self.engines:
            selected["tesseract"] = self.engines["tesseract"]
        
        return selected
        
    def _process_parallel(self, image: np.ndarray, engines: Dict[str, BaseOCREngine], **kwargs) -> Dict[str, DocumentResult]:
        """Process image with multiple engines in parallel"""
        results = {}
        
        def process_engine(engine_name: str, engine: BaseOCREngine) -> Tuple[str, DocumentResult]:
            try:
                print(f"Processing with {engine_name}...")
                result = engine.process_image(image, **kwargs)
                print(f"{engine_name} completed in {result.processing_time:.2f}s")
                return engine_name, result
            except Exception as e:
                print(f"Error in {engine_name}: {e}")
                return engine_name, DocumentResult(
                    full_text="", results=[], processing_time=0.0,
                    engine_name=engine_name, image_stats={}, confidence_score=0.0
                )
                
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_engine = {
                executor.submit(process_engine, name, engine): name 
                for name, engine in engines.items()
            }
            
            for future in concurrent.futures.as_completed(future_to_engine):
                engine_name, result = future.result()
                results[engine_name] = result
                
        return results
        
    def _process_sequential(self, image: np.ndarray, engines: Dict[str, BaseOCREngine], **kwargs) -> Dict[str, DocumentResult]:
        """Process image with engines sequentially"""
        results = {}
        
        for engine_name, engine in engines.items():
            try:
                print(f"Processing with {engine_name}...")
                result = engine.process_image(image, **kwargs)
                results[engine_name] = result
                print(f"{engine_name} completed in {result.processing_time:.2f}s")
            except Exception as e:
                print(f"Error in {engine_name}: {e}")
                results[engine_name] = DocumentResult(
                    full_text="", results=[], processing_time=0.0,
                    engine_name=engine_name, image_stats={}, confidence_score=0.0
                )
                
        return results
        
    def _combine_results(self, engine_results: Dict[str, DocumentResult], image_analysis: Dict[str, Any]) -> DocumentResult:
        """Combine results from multiple engines using intelligent voting"""
        if not engine_results:
            return DocumentResult("", [], 0.0, "combined", {}, 0.0)
            
        # Filter out failed results
        valid_results = {
            name: result for name, result in engine_results.items()
            if result.confidence_score > 0.1 and result.full_text.strip()
        }
        
        if not valid_results:
            # Return best available result even if low confidence
            best_result = max(engine_results.values(), key=lambda r: r.confidence_score)
            return best_result
            
        if len(valid_results) == 1:
            return list(valid_results.values())[0]
            
        # Use voting strategy
        if self.voting_strategy == "weighted_confidence":
            combined_result = self._weighted_confidence_voting(valid_results, image_analysis)
        elif self.voting_strategy == "best_confidence":
            combined_result = max(valid_results.values(), key=lambda r: r.confidence_score)
        else:
            combined_result = self._simple_majority_voting(valid_results)
            
        return combined_result
        
    def _weighted_confidence_voting(self, results: Dict[str, DocumentResult], image_analysis: Dict[str, Any]) -> DocumentResult:
        """Combine results using weighted confidence voting"""
        # Adjust weights based on image characteristics
        adjusted_weights = self._adjust_weights_for_image(image_analysis)
        
        # Calculate weighted scores
        weighted_scores = {}
        for engine_name, result in results.items():
            base_weight = adjusted_weights.get(engine_name, 1.0)
            confidence_weight = result.confidence_score
            weighted_scores[engine_name] = base_weight * confidence_weight
            
        # Select best result based on weighted score
        best_engine = max(weighted_scores.keys(), key=lambda e: weighted_scores[e])
        best_result = results[best_engine]
        
        # Create combined result with metadata from all engines
        processing_times = [r.processing_time for r in results.values()]
        
        return DocumentResult(
            full_text=best_result.full_text,
            results=best_result.results,
            processing_time=sum(processing_times),
            engine_name=f"combined({best_engine})",
            image_stats=best_result.image_stats,
            confidence_score=best_result.confidence_score
        )
        
    def _adjust_weights_for_image(self, image_analysis: Dict[str, Any]) -> Dict[str, float]:
        """Adjust engine weights based on image characteristics"""
        adjusted = self.engine_weights.copy()
        
        handwritten_prob = image_analysis.get("has_handwritten", 0.5)
        
        # Boost TrOCR weight for handwritten text
        if "trocr" in adjusted:
            adjusted["trocr"] *= (1.0 + handwritten_prob)
            
        # Boost Tesseract for printed text
        if "tesseract" in adjusted:
            adjusted["tesseract"] *= (1.0 + (1.0 - handwritten_prob))
            
        return adjusted
        
    def _simple_majority_voting(self, results: Dict[str, DocumentResult]) -> DocumentResult:
        """Simple majority voting based on confidence"""
        return max(results.values(), key=lambda r: r.confidence_score)
        
    def get_engine_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about loaded engines"""
        info = {}
        for name, engine in self.engines.items():
            info[name] = {
                "name": engine.name,
                "initialized": engine.is_initialized,
                "supported_languages": engine.get_supported_languages(),
                "weight": self.engine_weights.get(name, 1.0)
            }
        return info
        
    def cleanup(self):
        """Cleanup all engines"""
        for engine in self.engines.values():
            try:
                engine.cleanup()
            except Exception as e:
                print(f"Error during cleanup: {e}")
        self.engines.clear()