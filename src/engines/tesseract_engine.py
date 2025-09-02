# src/engines/tesseract_engine.py

import cv2
import numpy as np
import pytesseract
from typing import List, Dict, Any
import time
import re
from PIL import Image
import os

from ..core.base_engine import BaseOCREngine, OCRResult, DocumentResult

class TesseractEngine(BaseOCREngine):
    """Tesseract OCR Engine - Excellent for printed text"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("Tesseract", config)
        self.tesseract_config = self._build_tesseract_config()
        
    def _build_tesseract_config(self) -> str:
        """Build Tesseract configuration string"""
        config_parts = []
        
        # OCR Engine Mode (LSTM only)
        config_parts.append("--oem 1")
        
        # Page Segmentation Mode
        psm = self.config.get("psm", 6)  # Default: uniform block of text
        config_parts.append(f"--psm {psm}")
        
        # Language
        lang = self.config.get("lang", "eng")
        if isinstance(lang, list):
            lang = "+".join(lang)
        config_parts.append(f"-l {lang}")
        
        # Additional configurations
        if self.config.get("whitelist"):
            config_parts.append(f"-c tessedit_char_whitelist={self.config['whitelist']}")
        if self.config.get("blacklist"):
            config_parts.append(f"-c tessedit_char_blacklist={self.config['blacklist']}")
            
        return " ".join(config_parts)
        
    def initialize(self) -> bool:
        """Initialize Tesseract engine"""
        try:
            # Test if Tesseract is available
            pytesseract.get_tesseract_version()
            
            # Set Tesseract command path if provided
            if "tesseract_cmd" in self.config:
                pytesseract.pytesseract.tesseract_cmd = self.config["tesseract_cmd"]
                
            self.supported_languages = self._get_available_languages()
            self.is_initialized = True
            return True
        except Exception as e:
            print(f"Failed to initialize Tesseract: {e}")
            return False
            
    def _get_available_languages(self) -> List[str]:
        """Get available Tesseract languages"""
        try:
            langs = pytesseract.get_languages()
            return [lang for lang in langs if lang != 'osd']
        except:
            return ["eng"]  # Default fallback
            
    def get_supported_languages(self) -> List[str]:
        """Get supported languages"""
        return self.supported_languages
        
    def process_image(self, image: np.ndarray, **kwargs) -> DocumentResult:
        """Process image with Tesseract OCR"""
        start_time = time.time()
        
        # Preprocess image
        processed_image = self.preprocess_image(image)
        
        # Enhanced preprocessing for better OCR
        processed_image = self._enhance_image(processed_image)
        
        try:
            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(processed_image)
            
            # Get detailed OCR data
            data = pytesseract.image_to_data(
                pil_image, 
                config=self.tesseract_config,
                output_type=pytesseract.Output.DICT
            )
            
            # Parse results
            results = self._parse_tesseract_data(data)
            
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
            print(f"Tesseract processing error: {e}")
            return DocumentResult(
                full_text="",
                results=[],
                processing_time=time.time() - start_time,
                engine_name=self.name,
                image_stats={},
                confidence_score=0.0
            )
            
    def _enhance_image(self, image: np.ndarray) -> np.ndarray:
        """Enhanced image preprocessing for better OCR"""
        # Noise reduction
        denoised = cv2.fastNlMeansDenoising(image)
        
        # Morphological operations to clean text
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        morphed = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(morphed)
        
        # Threshold the image
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return thresh
        
    def _parse_tesseract_data(self, data: Dict) -> List[OCRResult]:
        """Parse Tesseract output data into OCRResult objects"""
        results = []
        
        for i in range(len(data['text'])):
            text = data['text'][i].strip()
            conf = float(data['conf'][i])
            
            if text and conf > 0:  # Filter empty text and invalid confidence
                bbox = (
                    int(data['left'][i]),
                    int(data['top'][i]),
                    int(data['width'][i]),
                    int(data['height'][i])
                )
                
                result = OCRResult(
                    text=text,
                    confidence=conf / 100.0,  # Convert to 0-1 range
                    bbox=bbox,
                    word_level=True
                )
                
                if self.validate_result(result):
                    results.append(result)
                    
        return results
        
    def _extract_full_text(self, results: List[OCRResult]) -> str:
        """Extract full text from OCR results"""
        if not results:
            return ""
            
        # Sort results by position (top to bottom, left to right)
        sorted_results = sorted(results, key=lambda r: (r.bbox[1], r.bbox[0]))
        
        # Group results by lines based on Y position
        lines = []
        current_line = []
        current_y = -1
        line_threshold = 10  # pixels
        
        for result in sorted_results:
            y_pos = result.bbox[1]
            
            if current_y == -1 or abs(y_pos - current_y) <= line_threshold:
                current_line.append(result)
                current_y = y_pos
            else:
                if current_line:
                    lines.append(current_line)
                current_line = [result]
                current_y = y_pos
                
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
        
    def extract_with_layout(self, image: np.ndarray) -> DocumentResult:
        """Extract text while preserving layout structure"""
        # Use PSM 6 for uniform block of text with layout
        original_config = self.tesseract_config
        self.tesseract_config = self.tesseract_config.replace("--psm 6", "--psm 6")
        
        result = self.process_image(image)
        self.tesseract_config = original_config
        
        return result