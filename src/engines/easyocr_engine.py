# src/engines/easyocr_engine.py

import cv2
import numpy as np
import easyocr
from typing import List, Dict, Any, Tuple
import time
import os

from ..core.base_engine import BaseOCREngine, OCRResult, DocumentResult

class EasyOCREngine(BaseOCREngine):
    """EasyOCR Engine - Good balance for handwritten and printed text"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("EasyOCR", config)
        self.reader = None
        self.languages = config.get("languages", ["en"])
        self.gpu = config.get("gpu", True)
        self.model_storage_directory = config.get("model_dir", None)
        
    def initialize(self) -> bool:
        """Initialize EasyOCR reader"""
        try:
            # Initialize EasyOCR reader
            self.reader = easyocr.Reader(
                self.languages,
                gpu=self.gpu,
                model_storage_directory=self.model_storage_directory,
                download_enabled=True,
                detector=True,
                recognizer=True,
                verbose=False
            )
            
            self.supported_languages = self.languages
            self.is_initialized = True
            self.model_loaded = True
            return True
            
        except Exception as e:
            print(f"Failed to initialize EasyOCR: {e}")
            return False
            
    def get_supported_languages(self) -> List[str]:
        """Get supported languages"""
        # EasyOCR supported languages
        return [
            'en', 'ch_sim', 'ch_tra', 'ja', 'ko', 'th', 'vi', 'ar', 'hi', 'ur', 
            'fa', 'ru', 'bg', 'uk', 'be', 'te', 'kn', 'ta', 'bn', 'as', 'mr', 
            'ne', 'si', 'my', 'km', 'lo', 'sa', 'fr', 'de', 'es', 'pt', 'it',
            'nl', 'sv', 'da', 'no', 'fi', 'lt', 'lv', 'et', 'pl', 'cs', 'sk',
            'sl', 'hu', 'ro', 'hr', 'sr', 'bs', 'mk', 'sq', 'mt', 'cy', 'ga',
            'tr', 'az', 'uz', 'mn'
        ]
        
    def process_image(self, image: np.ndarray, **kwargs) -> DocumentResult:
        """Process image with EasyOCR"""
        if not self.is_initialized:
            raise RuntimeError("EasyOCR engine not initialized")
            
        start_time = time.time()
        
        try:
            # Enhanced preprocessing
            processed_image = self._preprocess_for_easyocr(image)
            
            # EasyOCR processing
            ocr_results = self.reader.readtext(
                processed_image,
                detail=1,  # Get detailed results with bounding boxes
                paragraph=kwargs.get("paragraph", False),
                width_ths=kwargs.get("width_ths", 0.7),
                height_ths=kwargs.get("height_ths", 0.7),
                decoder=kwargs.get("decoder", "greedy"),
                beamWidth=kwargs.get("beamWidth", 5),
                batch_size=kwargs.get("batch_size", 1)
            )
            
            # Parse results
            results = self._parse_easyocr_results(ocr_results)
            
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
            print(f"EasyOCR processing error: {e}")
            return DocumentResult(
                full_text="",
                results=[],
                processing_time=time.time() - start_time,
                engine_name=self.name,
                image_stats={},
                confidence_score=0.0
            )
            
    def _preprocess_for_easyocr(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image optimized for EasyOCR"""
        # Convert to RGB if needed (EasyOCR expects RGB)
        if len(image.shape) == 3:
            if image.shape[2] == 3:
                # Assume BGR and convert to RGB
                processed = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                processed = image
        else:
            # Convert grayscale to RGB
            processed = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
        # Resize if image is too large (EasyOCR works better with reasonable sizes)
        height, width = processed.shape[:2]
        max_dim = 2048
        
        if max(height, width) > max_dim:
            scale = max_dim / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            processed = cv2.resize(processed, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
            
        # Enhance contrast and sharpening for better text detection
        processed = self._enhance_for_ocr(processed)
        
        return processed
        
    def _enhance_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """Enhance image for better OCR results"""
        # Convert to LAB color space for better contrast adjustment
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge back
            enhanced = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        else:
            enhanced = image
            
        # Slight sharpening
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel * 0.1)
        
        # Blend original and sharpened
        result = cv2.addWeighted(enhanced, 0.7, sharpened, 0.3, 0)
        
        return result
        
    def _parse_easyocr_results(self, ocr_results: List) -> List[OCRResult]:
        """Parse EasyOCR results into standardized format"""
        results = []
        
        for detection in ocr_results:
            bbox_points, text, confidence = detection
            
            if text.strip() and confidence > 0.1:  # Filter low confidence
                # Convert polygon to bounding box
                bbox = self._polygon_to_bbox(bbox_points)
                
                result = OCRResult(
                    text=text.strip(),
                    confidence=confidence,
                    bbox=bbox,
                    word_level=True
                )
                
                if self.validate_result(result):
                    results.append(result)
                    
        return results
        
    def _polygon_to_bbox(self, points: List[List[float]]) -> Tuple[int, int, int, int]:
        """Convert polygon points to bounding box"""
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        return (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))
        
    def _extract_full_text(self, results: List[OCRResult]) -> str:
        """Extract full text preserving reading order"""
        if not results:
            return ""
            
        # Sort by Y coordinate first, then X coordinate
        sorted_results = sorted(results, key=lambda r: (r.bbox[1], r.bbox[0]))
        
        # Group into lines
        lines = []
        current_line = []
        last_y = -1
        y_threshold = 20  # pixels
        
        for result in sorted_results:
            y_pos = result.bbox[1]
            
            if last_y == -1 or abs(y_pos - last_y) <= y_threshold:
                current_line.append(result)
                last_y = y_pos
            else:
                if current_line:
                    lines.append(current_line)
                current_line = [result]
                last_y = y_pos
                
        if current_line:
            lines.append(current_line)
            
        # Combine lines into full text
        full_text_lines = []
        for line in lines:
            # Sort words in line by X position
            line_sorted = sorted(line, key=lambda r: r.bbox[0])
            line_text = " ".join(result.text for result in line_sorted)
            full_text_lines.append(line_text)
            
        return "\n".join(full_text_lines)
        
    def _calculate_image_stats(self, image: np.ndarray) -> Dict[str, Any]:
        """Calculate image statistics"""
        return {
            "width": image.shape[1],
            "height": image.shape[0],
            "channels": len(image.shape),
            "mean_brightness": np.mean(image),
            "std_brightness": np.std(image)
        }
        
    def cleanup(self):
        """Cleanup EasyOCR resources"""
        if self.reader:
            # EasyOCR doesn't have explicit cleanup, but we can clear the reference
            self.reader = None
            self.model_loaded = False