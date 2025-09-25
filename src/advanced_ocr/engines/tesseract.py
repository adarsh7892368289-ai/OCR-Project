# src/advanced_ocr/engines/tesseract.py
"""
Tesseract OCR Engine Implementation

Responsibility: ONLY convert images to text using Tesseract
- Initialize Tesseract with configuration
- Convert preprocessed images to text regions
- Return OCRResult with basic text regions and bounding boxes
- Handle Tesseract-specific settings (PSM, OEM, etc.)

Does NOT do (handled by pipeline):
- Image enhancement
- Layout reconstruction
- Quality analysis  
- Engine management
"""

import numpy as np
import pytesseract
from typing import List, Dict, Any, Optional
import time
from PIL import Image

from ..core.base_engine import BaseOCREngine
from ..types import OCRResult, TextRegion, BoundingBox, TextType


class TesseractEngine(BaseOCREngine):
    """
    Tesseract Engine - Focused OCR text extraction only
    
    Clean integration with pipeline:
    - Receives preprocessed images from ImageEnhancer
    - Returns raw OCRResult with text regions
    - Pipeline handles layout reconstruction
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("Tesseract", config)
        
        # Tesseract specific settings
        self.psm = self.config.get("psm", 6)  # Page segmentation mode
        self.oem = self.config.get("oem", 1)  # OCR engine mode
        self.lang = self.config.get("lang", "eng")
        
        # Build Tesseract config string
        self.tesseract_config = self._build_config()
        
    def initialize(self) -> bool:
        """Initialize Tesseract engine"""
        try:
            # Test Tesseract availability
            version = pytesseract.get_tesseract_version()
            self.supported_languages = self._get_available_languages()
            
            self.is_initialized = True
            self.logger.info(f"Tesseract {version} initialized with {len(self.supported_languages)} languages")
            return True
            
        except Exception as e:
            self.logger.error(f"Tesseract initialization failed: {e}")
            return False
            
    def is_available(self) -> bool:
        """Check if Tesseract is available"""
        try:
            pytesseract.get_tesseract_version()
            return self.is_initialized
        except:
            return False
            
    def extract_text(self, image: np.ndarray) -> OCRResult:
        """
        Core OCR extraction using Tesseract
        
        Args:
            image: Preprocessed image from ImageEnhancer
            
        Returns:
            OCRResult: Raw text regions (pipeline handles layout)
        """
        start_time = time.time()
        
        try:
            if not self.validate_image(image):
                raise ValueError("Invalid image")
            
            # Convert to PIL format
            pil_image = self._prepare_image(image)
            
            # Tesseract extraction with word-level data
            data = pytesseract.image_to_data(
                pil_image,
                config=self.tesseract_config,
                output_type=pytesseract.Output.DICT
            )
            
            # Convert to our format
            regions = self._parse_results(data)
            
            # Basic concatenation (pipeline handles proper layout)
            text = " ".join(r.text for r in regions if r.text.strip())
            confidence = sum(r.confidence for r in regions) / len(regions) if regions else 0.0
            
            processing_time = time.time() - start_time
            
            # Update stats
            self.processing_stats['total_processed'] += 1
            self.processing_stats['total_time'] += processing_time
            if text.strip():
                self.processing_stats['successful_extractions'] += 1
            
            return OCRResult(
                text=text,
                confidence=confidence,
                processing_time=processing_time,
                engine_used=self.name,
                regions=regions,
                metadata={
                    'detection_count': len(regions),
                    'tesseract_config': self.tesseract_config
                }
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.processing_stats['errors'] += 1
            self.logger.error(f"Tesseract failed: {e}")
            
            return OCRResult(
                text="",
                confidence=0.0,
                processing_time=processing_time,
                engine_used=self.name,
                metadata={"error": str(e)}
            )
    
    def get_supported_languages(self) -> List[str]:
        """Get Tesseract supported languages"""
        return getattr(self, 'supported_languages', ["eng"])
    
    def _build_config(self) -> str:
        """Build Tesseract configuration string"""
        config_parts = [f"--oem {self.oem}", f"--psm {self.psm}"]
        
        # Language configuration
        if isinstance(self.lang, list):
            lang_str = "+".join(self.lang)
        else:
            lang_str = self.lang
        config_parts.append(f"-l {lang_str}")
        
        return " ".join(config_parts)
        
    def _get_available_languages(self) -> List[str]:
        """Get available Tesseract languages"""
        try:
            langs = pytesseract.get_languages()
            return [lang for lang in langs if lang != 'osd']
        except:
            return ["eng"]
            
    def _prepare_image(self, image: np.ndarray) -> Image.Image:
        """Convert numpy array to PIL Image"""
        if len(image.shape) == 3:
            if image.shape[2] == 3:
                # BGR to RGB
                image_rgb = image[:, :, ::-1]
                return Image.fromarray(image_rgb)
            return Image.fromarray(image)
        else:
            # Grayscale
            return Image.fromarray(image)
    
    def _parse_results(self, data: Dict) -> List[TextRegion]:
        """Convert Tesseract results to TextRegions"""
        regions = []
        
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
            
            region = TextRegion(
                text=text,
                confidence=conf / 100.0,
                bbox=bbox,
                text_type=TextType.PRINTED,
                language=self.lang if isinstance(self.lang, str) else self.lang[0]
            )
            
            regions.append(region)
            
        return regions
    
Tesseract=TesseractEngine
