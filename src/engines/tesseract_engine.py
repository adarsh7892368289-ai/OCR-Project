import cv2
import numpy as np
import pytesseract
from typing import List, Dict, Any, Optional
import time
import re
from PIL import Image
import os
import subprocess

from ..core.base_engine import (
    BaseOCREngine, 
    OCRResult, 
    DocumentResult, 
    TextRegion,
    BoundingBox,
    DocumentStructure,
    TextType
)

def find_tesseract():
    """Find Tesseract installation automatically"""
    possible_paths = [
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Users\adbm\AppData\Local\Programs\Tesseract-OCR\tesseract.exe',
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        r'C:\Users\{}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'.format(os.getenv('USERNAME', '')),
        'tesseract',  # Linux/Mac default
        '/usr/bin/tesseract',
        '/usr/local/bin/tesseract'
    ]
    
    for path in possible_paths:
        try:
            if os.name == 'nt':  # Windows
                if os.path.exists(path):
                    return path
            else:  # Linux/Mac
                result = subprocess.run(['which', 'tesseract'], capture_output=True, text=True)
                if result.returncode == 0:
                    return result.stdout.strip()
        except:
            continue
    return None

class TesseractEngine(BaseOCREngine):
    """Tesseract OCR Engine - Excellent for printed text"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
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
        
        return " ".join(config_parts)
        
    def initialize(self) -> bool:
        """Initialize Tesseract engine"""
        try:
            # Auto-find Tesseract
            tesseract_cmd = find_tesseract()
            if tesseract_cmd:
                pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
                print(f"Found Tesseract at: {tesseract_cmd}")
            else:
                print("Tesseract not found. Please install from: https://github.com/UB-Mannheim/tesseract/wiki")
                return False
            
            # Test if Tesseract works
            pytesseract.get_tesseract_version()
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
        return getattr(self, 'supported_languages', ["eng"])
        
    def process_image(self, image, **kwargs) -> DocumentResult:
        """Process image with Tesseract OCR"""
        start_time = time.time()
        
        try:
            # Handle different input types
            if hasattr(image, 'shape'):  # numpy array
                processed_image = self._enhance_image(image)
            elif hasattr(image, 'mode'):  # PIL Image
                processed_image = self._enhance_image(np.array(image))
            else:
                raise ValueError(f"Unsupported image type: {type(image)}")
            
            # Convert numpy array to PIL Image
            if len(processed_image.shape) == 3:
                pil_image = Image.fromarray(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
            else:
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
            
            # Create text regions from results
            text_regions = self._create_text_regions(results)
            
            # Calculate statistics
            processing_time = time.time() - start_time
            confidence_score = self.calculate_confidence(results)
            
            # Create OCR result for the page
            page_result = OCRResult(
                text=full_text,
                confidence=confidence_score,
                regions=text_regions,
                processing_time=processing_time,
                bbox=None,
                level="page"
            )
            
            # Create document result with correct constructor
            return DocumentResult(
                pages=[page_result],
                metadata={'image_stats': {}},
                processing_time=processing_time,
                engine_name=self.name,
                confidence_score=confidence_score
            )
            
        except Exception as e:
            print(f"Tesseract processing error: {e}")
            
            # Return properly constructed empty result
            empty_page = OCRResult(
                text="",
                confidence=0.0,
                regions=[],
                processing_time=time.time() - start_time
            )
            
            return DocumentResult(
                pages=[empty_page],
                metadata={'error': str(e)},
                processing_time=time.time() - start_time,
                engine_name=self.name,
                confidence_score=0.0
            )
            
    def _enhance_image(self, image: np.ndarray) -> np.ndarray:
        """Enhanced image preprocessing for better OCR"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        # Noise reduction
        denoised = cv2.fastNlMeansDenoising(gray)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
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
                bbox = BoundingBox(
                    x=int(data['left'][i]),
                    y=int(data['top'][i]),
                    width=int(data['width'][i]),
                    height=int(data['height'][i]),
                    confidence=conf / 100.0
                )
                
                # Create OCRResult with correct parameter names
                result = OCRResult(
                    text=text,
                    confidence=conf / 100.0,
                    bbox=bbox,
                    level="word"
                )
                
                # Check text attribute
                if hasattr(result, 'text') and result.text.strip():
                    results.append(result)
                    
        return results

    def _create_text_regions(self, results: List[OCRResult]) -> List[TextRegion]:
        """Create text regions from OCR results"""
        text_regions = []
        
        for i, result in enumerate(results):
            region = TextRegion(
                text=result.text,
                confidence=result.confidence,
                bbox=result.bbox,
                text_type=TextType.PRINTED,
                reading_order=i
            )
            text_regions.append(region)
            
        return text_regions
        
    def _extract_full_text(self, results: List[OCRResult]) -> str:
        """Extract full text from OCR results"""
        if not results:
            return ""
            
        # Sort results by position (top to bottom, left to right)
        sorted_results = sorted(results, key=lambda r: (r.bbox.y if r.bbox else 0, r.bbox.x if r.bbox else 0))
        
        # Group results by lines based on Y position
        lines = []
        current_line = []
        current_y = -1
        line_threshold = 10  # pixels
        
        for result in sorted_results:
            if result.bbox:  # Check if bbox exists
                y_pos = result.bbox.y
                
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
            line_sorted = sorted(line, key=lambda r: r.bbox.x if r.bbox else 0)
            line_text = " ".join(result.text for result in line_sorted if result.text)
            if line_text.strip():  # Only add non-empty lines
                full_text_lines.append(line_text)
            
        return "\n".join(full_text_lines)
        
    def calculate_confidence(self, results: List[OCRResult]) -> float:
        """Calculate overall confidence score"""
        if not results:
            return 0.0
        
        confidences = [result.confidence for result in results if result.confidence > 0]
        if not confidences:
            return 0.0
            
        return sum(confidences) / len(confidences)