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
    Modern Tesseract OCR Engine - Aligned with Pipeline
    
    Clean integration with your pipeline:
    - Takes preprocessed images from YOUR preprocessing pipeline
    - Returns single OCRResult compatible with YOUR base engine
    - Works with YOUR engine manager and postprocessing
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
        
    def process_image(self, preprocessed_image: np.ndarray, **kwargs) -> OCRResult:
        """
        FIXED: Process preprocessed image and return single OCRResult
        
        Args:
            preprocessed_image: Image from YOUR preprocessing pipeline
            **kwargs: Additional parameters
            
        Returns:
            OCRResult: Single result compatible with YOUR base engine
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
            
            # FIXED: Combine all Tesseract detections into single OCRResult
            result = self._combine_tesseract_results(ocr_data)
            
            # Set processing time and engine info
            processing_time = time.time() - start_time
            result.processing_time = processing_time
            result.engine_name = self.name
            
            # Update stats
            self.processing_stats['total_processed'] += 1
            self.processing_stats['total_time'] += processing_time
            if result.text.strip():
                self.processing_stats['successful_extractions'] += 1
            
            return result
            
        except Exception as e:
            self.logger.error(f"OCR extraction failed: {e}")
            processing_time = time.time() - start_time
            self.processing_stats['errors'] += 1
            
            # Return empty result instead of raising
            return OCRResult(
                text="",
                confidence=0.0,
                processing_time=processing_time,
                engine_name=self.name,
                metadata={"error": str(e)}
            )
    
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
    
    def _combine_tesseract_results(self, data: Dict) -> OCRResult:
        """
        FIXED: Combine all Tesseract detections into single OCRResult
        """
        regions = []
        all_text_parts = []
        total_confidence = 0.0
        detection_count = 0
        
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
                confidence=conf / 100.0,
                text_type=TextType.PRINTED
            )
            
            # Create text region
            region = TextRegion(
                text=text,
                confidence=conf / 100.0,
                bbox=bbox,
                text_type=TextType.PRINTED,
                language="en"
            )
            
            regions.append(region)
            all_text_parts.append(text)
            total_confidence += conf / 100.0
            detection_count += 1
        
        # Combine all text parts with space separation
        combined_text = " ".join(all_text_parts)
        overall_confidence = total_confidence / detection_count if detection_count > 0 else 0.0
        
        # Create overall bounding box from all regions
        overall_bbox = self._calculate_overall_bbox(regions) if regions else BoundingBox(0, 0, 100, 30)
        
        # Return single OCRResult combining all detections
        return OCRResult(
            text=combined_text,
            confidence=overall_confidence,
            regions=regions,
            bbox=overall_bbox,
            level="page",
            engine_name=self.name,
            text_type=TextType.PRINTED,
            metadata={
                'detection_method': 'tesseract',
                'detection_count': detection_count,
                'tesseract_config': self.tesseract_config,
                'individual_confidences': [r.confidence for r in regions]
            }
        )
    
    def _calculate_overall_bbox(self, regions: List[TextRegion]) -> BoundingBox:
        """Calculate overall bounding box from all text regions"""
        if not regions:
            return BoundingBox(0, 0, 100, 30)
        
        min_x = min(r.bbox.x for r in regions)
        min_y = min(r.bbox.y for r in regions)
        max_x = max(r.bbox.x + r.bbox.width for r in regions)
        max_y = max(r.bbox.y + r.bbox.height for r in regions)
        
        return BoundingBox(
            x=min_x,
            y=min_y,
            width=max_x - min_x,
            height=max_y - min_y,
            confidence=sum(r.confidence for r in regions) / len(regions)
        )
    
    def batch_process(self, images: List[np.ndarray], **kwargs) -> List[OCRResult]:
        """Process multiple images - returns single result per image"""
        results = []
        for image in images:
            result = self.process_image(image, **kwargs)
            results.append(result)
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
            },
            'pipeline_integration': {
                'uses_preprocessing_pipeline': True,
                'returns_single_result': True,
                'compatible_with_base_engine': True,
                'works_with_engine_manager': True
            }
        }
        
        try:
            info['version'] = str(pytesseract.get_tesseract_version())
        except:
            info['version'] = 'unknown'
            
        return info