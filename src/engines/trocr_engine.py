# src/engines/trocr_engine.py

import cv2
import numpy as np
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from typing import List, Dict, Any, Tuple
import time
import warnings
warnings.filterwarnings("ignore")

from ..core.base_engine import BaseOCREngine, OCRResult, DocumentResult

class TrOCREngine(BaseOCREngine):
    """TrOCR Engine - Transformer-based OCR, excellent for handwritten text"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("TrOCR", config)
        self.processor = None
        self.model = None
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = config.get("model_name", "microsoft/trocr-base-handwritten")
        self.max_new_tokens = config.get("max_new_tokens", 128)
        self.batch_size = config.get("batch_size", 4)
        
    def initialize(self) -> bool:
        """Initialize TrOCR model and processor"""
        try:
            print(f"Loading TrOCR model: {self.model_name}")
            
            # Load processor and model
            self.processor = TrOCRProcessor.from_pretrained(self.model_name)
            self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name)
            
            # Move model to device
            self.model.to(self.device)
            self.model.eval()
            
            # Supported languages (TrOCR primarily supports English)
            self.supported_languages = ["en"]
            
            self.is_initialized = True
            self.model_loaded = True
            print(f"TrOCR initialized successfully on {self.device}")
            return True
            
        except Exception as e:
            print(f"Failed to initialize TrOCR: {e}")
            return False
            
    def get_supported_languages(self) -> List[str]:
        """Get supported languages"""
        return self.supported_languages
        
    def process_image(self, image: np.ndarray, **kwargs) -> DocumentResult:
        """Process image with TrOCR"""
        if not self.is_initialized:
            raise RuntimeError("TrOCR engine not initialized")
            
        start_time = time.time()
        
        try:
            # Detect text regions first
            text_regions = self._detect_text_regions(image)
            
            if not text_regions:
                # If no regions detected, process entire image
                text_regions = [self._get_full_image_region(image)]
            
            # Process each text region
            results = []
            for region in text_regions:
                region_results = self._process_text_region(image, region)
                results.extend(region_results)
            
            # Get full text
            full_text = self._extract_full_text(results)
            
            # Calculate statistics
            processing_time = time.time() - start_time
            confidence_score = self.calculate_confidence(results)
            image_stats = self._calculate_image_stats(image)
            
            return DocumentResult(
                full_text=full_text,
                results=results,
                processing_time=processing_time,
                engine_name=self.name,
                image_stats=image_stats,
                confidence_score=confidence_score
            )
            
        except Exception as e:
            print(f"TrOCR processing error: {e}")
            return DocumentResult(
                full_text="",
                results=[],
                processing_time=time.time() - start_time,
                engine_name=self.name,
                image_stats={},
                confidence_score=0.0
            )
            
    def _detect_text_regions(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect text regions in the image using OpenCV methods"""
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Apply morphological operations to find text regions
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 3))
        dilated = cv2.dilate(gray, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        regions = []
        min_area = 500  # Minimum area threshold
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                x, y, w, h = cv2.boundingRect(contour)
                # Add some padding
                padding = 10
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(image.shape[1] - x, w + 2 * padding)
                h = min(image.shape[0] - y, h + 2 * padding)
                regions.append((x, y, w, h))
                
        return regions
        
    def _get_full_image_region(self, image: np.ndarray) -> Tuple[int, int, int, int]:
        """Get region covering the entire image"""
        return (0, 0, image.shape[1], image.shape[0])
        
    def _process_text_region(self, image: np.ndarray, region: Tuple[int, int, int, int]) -> List[OCRResult]:
        """Process a single text region with TrOCR"""
        x, y, w, h = region
        
        # Extract region from image
        region_image = image[y:y+h, x:x+w]
        
        # Preprocess for TrOCR
        processed_region = self._preprocess_for_trocr(region_image)
        
        # Convert to PIL Image
        if len(processed_region.shape) == 3:
            pil_image = Image.fromarray(cv2.cvtColor(processed_region, cv2.COLOR_BGR2RGB))
        else:
            pil_image = Image.fromarray(processed_region).convert('RGB')
            
        try:
            # Process with TrOCR
            pixel_values = self.processor(pil_image, return_tensors="pt").pixel_values.to(self.device)
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    pixel_values, 
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,  # Use greedy decoding for consistency
                    num_beams=1
                )
                
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Create OCR result
            if generated_text.strip():
                # Estimate confidence (TrOCR doesn't provide confidence scores)
                confidence = self._estimate_confidence(generated_text, region_image)
                
                result = OCRResult(
                    text=generated_text.strip(),
                    confidence=confidence,
                    bbox=(x, y, w, h),
                    line_level=True
                )
                
                if self.validate_result(result):
                    return [result]
                    
        except Exception as e:
            print(f"Error processing region with TrOCR: {e}")
            
        return []
        
    def _preprocess_for_trocr(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for optimal TrOCR performance"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Resize to optimal size for TrOCR (384x384 is typical)
        target_height = 384
        aspect_ratio = image.shape[1] / image.shape[0]
        target_width = int(target_height * aspect_ratio)
        
        # Ensure minimum width
        if target_width < 384:
            target_width = 384
            
        resized = cv2.resize(gray, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
        
        # Normalize and enhance
        normalized = cv2.normalize(resized, None, 0, 255, cv2.NORM_MINMAX)
        
        # Apply slight Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(normalized, (3, 3), 0)
        
        # Convert back to 3-channel for consistency
        result = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)
        
        return result
        
    def _estimate_confidence(self, text: str, image: np.ndarray) -> float:
        """Estimate confidence score for TrOCR result"""
        # Simple heuristic-based confidence estimation
        base_confidence = 0.8
        
        # Penalize very short or very long outputs
        text_len = len(text.strip())
        if text_len < 3:
            base_confidence *= 0.5
        elif text_len > 200:
            base_confidence *= 0.8
            
        # Boost confidence for common words/patterns
        common_words = {'the', 'and', 'is', 'in', 'to', 'of', 'a', 'that', 'it', 'with', 'for', 'as', 'was', 'on', 'are', 'you'}
        words = text.lower().split()
        if words and len([w for w in words if w in common_words]) / len(words) > 0.3:
            base_confidence = min(0.95, base_confidence * 1.1)
            
        # Check for suspicious patterns (too many special characters)
        special_char_ratio = sum(1 for c in text if not c.isalnum() and c not in ' .,!?-') / len(text) if text else 0
        if special_char_ratio > 0.3:
            base_confidence *= 0.7
            
        return min(0.95, base_confidence)
        
    def _extract_full_text(self, results: List[OCRResult]) -> str:
        """Extract full text from results"""
        if not results:
            return ""
            
        # Sort by position (top to bottom, left to right)
        sorted_results = sorted(results, key=lambda r: (r.bbox[1], r.bbox[0]))
        return "\n".join(result.text for result in sorted_results)
        
    def _calculate_image_stats(self, image: np.ndarray) -> Dict[str, Any]:
        """Calculate image statistics"""
        return {
            "width": image.shape[1],
            "height": image.shape[0],
            "channels": len(image.shape),
            "mean_brightness": np.mean(image),
            "std_brightness": np.std(image)
        }
        
    def process_handwritten_lines(self, image: np.ndarray) -> DocumentResult:
        """Specialized method for processing handwritten lines"""
        # Use line-by-line processing for better handwritten text recognition
        return self.process_image(image, line_detection=True)
        
    def cleanup(self):
        """Cleanup TrOCR resources"""
        if self.model:
            del self.model
            self.model = None
        if self.processor:
            del self.processor
            self.processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.model_loaded = False