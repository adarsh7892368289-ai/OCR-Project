import numpy as np
import pytesseract
from typing import List, Dict, Any, Optional
import time
from PIL import Image
import os
import subprocess

from ..core.base_engine import (
    BaseOCREngine, 
    OCRResult, 
    DocumentResult, 
    TextRegion,
    BoundingBox,
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
    """
    Modern Tesseract OCR Engine - AI-Style Architecture
    
    Clean separation of concerns:
    - Takes preprocessed images as input
    - Performs pure OCR extraction 
    - Returns structured results for postprocessing
    - No internal preprocessing or postprocessing
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("Tesseract", config)
        self.tesseract_config = self._build_tesseract_config()
        self.supports_handwriting = False
        self.supports_multiple_languages = True
        self.supports_orientation_detection = True
        self.supports_structure_analysis = True
        
    def _build_tesseract_config(self) -> str:
        """Build optimized Tesseract configuration"""
        config_parts = ["--oem 1"]  # LSTM neural net mode
        
        # Page segmentation mode
        psm = self.config.get("psm", 6)
        config_parts.append(f"--psm {psm}")
        
        # Language configuration
        lang = self.config.get("lang", "eng")
        if isinstance(lang, list):
            lang = "+".join(lang)
        config_parts.append(f"-l {lang}")
        
        return " ".join(config_parts)
        
    def initialize(self) -> bool:
        """Initialize Tesseract engine"""
        try:
            tesseract_cmd = find_tesseract()
            if not tesseract_cmd:
                self.logger.error("Tesseract not found")
                return False
                
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
            version = pytesseract.get_tesseract_version()
            self.supported_languages = self._get_available_languages()
            
            self.is_initialized = True
            self.model_loaded = True
            
            self.logger.info(f"Tesseract {version} initialized with {len(self.supported_languages)} languages")
            return True
            
        except Exception as e:
            self.logger.error(f"Tesseract initialization failed: {e}")
            return False
            
    def _get_available_languages(self) -> List[str]:
        """Get available languages"""
        try:
            langs = pytesseract.get_languages()
            return [lang for lang in langs if lang != 'osd']
        except:
            return ["eng"]
            
    def get_supported_languages(self) -> List[str]:
        """Get supported languages"""
        return getattr(self, 'supported_languages', ["eng"])
        
    def process_image(self, preprocessed_image: np.ndarray, **kwargs) -> List[OCRResult]:
        """
        Process preprocessed image and extract text
        
        Args:
            preprocessed_image: Image from preprocessing pipeline
            **kwargs: Additional parameters
            
        Returns:
            List[OCRResult]: Raw OCR results for postprocessing
        """
        start_time = time.time()
        
        try:
            if not self.validate_image(preprocessed_image):
                raise ValueError("Invalid preprocessed image")
            
            # Convert to PIL Image for Tesseract
            pil_image = self._numpy_to_pil(preprocessed_image)
            
            # Extract OCR data
            ocr_data = pytesseract.image_to_data(
                pil_image,
                config=self.tesseract_config,
                output_type=pytesseract.Output.DICT
            )
            
            # Convert to OCRResult objects
            results = self._extract_ocr_results(ocr_data)
            
            # Update stats
            processing_time = time.time() - start_time
            self.processing_stats['total_processed'] += 1
            self.processing_stats['total_time'] += processing_time
            
            return results
            
        except Exception as e:
            self.logger.error(f"OCR extraction failed: {e}")
            self.processing_stats['errors'] += 1
            return []
    
    def _numpy_to_pil(self, image: np.ndarray) -> Image.Image:
        """Convert numpy array to PIL Image"""
        if len(image.shape) == 3:
            if image.shape[2] == 3:
                # Assume BGR, convert to RGB
                image_rgb = image[:, :, ::-1]  # BGR to RGB
                return Image.fromarray(image_rgb)
            else:
                return Image.fromarray(image)
        else:
            # Grayscale
            return Image.fromarray(image)
    
    def _extract_ocr_results(self, data: Dict) -> List[OCRResult]:
        """Extract OCRResult objects from Tesseract data"""
        results = []
        
        for i in range(len(data['text'])):
            text = data['text'][i].strip()
            conf = float(data['conf'][i])
            
            if not text or conf <= 0:
                continue
                
            bbox = BoundingBox(
                x=int(data['left'][i]),
                y=int(data['top'][i]),
                width=int(data['width'][i]),
                height=int(data['height'][i]),
                confidence=conf / 100.0
            )
            
            result = OCRResult(
                text=text,
                confidence=conf / 100.0,
                bbox=bbox,
                level="word",
                metadata={
                    'tesseract_level': data.get('level', [0])[i] if 'level' in data else 5,
                    'block_num': data.get('block_num', [0])[i] if 'block_num' in data else 0,
                    'par_num': data.get('par_num', [0])[i] if 'par_num' in data else 0,
                    'line_num': data.get('line_num', [0])[i] if 'line_num' in data else 0,
                    'word_num': data.get('word_num', [0])[i] if 'word_num' in data else 0
                }
            )
            
            results.append(result)
            
        return results
    
    def batch_process(self, images: List[np.ndarray], **kwargs) -> List[List[OCRResult]]:
        """Process multiple images efficiently"""
        results = []
        for image in images:
            image_results = self.process_image(image, **kwargs)
            results.append(image_results)
        return results
    
    def get_engine_info(self) -> Dict[str, Any]:
        """Get engine information"""
        info = {
            'name': self.name,
            'version': None,
            'type': 'traditional_ocr',
            'supported_languages': self.get_supported_languages(),
            'capabilities': {
                'handwriting': self.supports_handwriting,
                'multiple_languages': self.supports_multiple_languages,
                'orientation_detection': self.supports_orientation_detection,
                'structure_analysis': self.supports_structure_analysis
            },
            'optimal_for': ['printed_text', 'documents', 'books', 'forms'],
            'performance_profile': {
                'accuracy': 'high',
                'speed': 'medium',
                'memory_usage': 'low',
                'gpu_required': False
            }
        }
        
        try:
            info['version'] = str(pytesseract.get_tesseract_version())
        except:
            info['version'] = 'unknown'
            
        return info