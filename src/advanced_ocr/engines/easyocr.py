# src/advanced_ocr/engines/easyocr.py
"""
EasyOCR Engine Implementation

Responsibility: ONLY convert images to text using EasyOCR
- Initialize EasyOCR reader
- Convert preprocessed images to text regions
- Return OCRResult with basic text regions and bounding boxes
- Handle EasyOCR-specific configuration

Does NOT do (handled by pipeline):
- Image enhancement
- Layout reconstruction  
- Quality analysis
- Engine management
"""

import cv2
import numpy as np
import time
from typing import List, Dict, Any, Optional

from ..core.base_engine import BaseOCREngine
from ..types import OCRResult, TextRegion, BoundingBox, TextType


class EasyOCREngine(BaseOCREngine):
    """
    EasyOCR Engine - Focused OCR text extraction only
    
    Clean integration with pipeline:
    - Receives preprocessed images from ImageEnhancer
    - Returns raw OCRResult with text regions
    - Pipeline handles layout reconstruction
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("EasyOCR", config)
        self.reader = None
        
        # EasyOCR specific settings
        self.languages = self.config.get("languages", ["en"])
        self.gpu = self.config.get("gpu", True)
        self.model_storage_directory = self.config.get("model_dir", None)
        
    def initialize(self) -> bool:
        """Initialize EasyOCR reader"""
        try:
            import easyocr
            
            self.reader = easyocr.Reader(
                self.languages,
                gpu=self.gpu,
                model_storage_directory=self.model_storage_directory,
                download_enabled=True,
                verbose=False
            )
            
            self.is_initialized = True
            self.logger.info(f"EasyOCR initialized (GPU: {self.gpu})")
            return True
            
        except Exception as e:
            self.logger.error(f"EasyOCR initialization failed: {e}")
            return False
            
    def is_available(self) -> bool:
        """Check if EasyOCR is available"""
        try:
            import easyocr
            return self.is_initialized
        except ImportError:
            return False
    
    def extract_text(self, image: np.ndarray) -> OCRResult:
        """
        Core OCR extraction using EasyOCR
        
        Args:
            image: Preprocessed image from ImageEnhancer
            
        Returns:
            OCRResult: Raw text regions (pipeline handles layout)
        """
        start_time = time.time()
        
        try:
            if not self.is_initialized and not self.initialize():
                raise RuntimeError("EasyOCR not initialized")
            
            if not self.validate_image(image):
                raise ValueError("Invalid image")
            
            # Convert to EasyOCR format (only format conversion)
            ocr_image = self._prepare_image(image)
            
            # EasyOCR extraction
            detections = self.reader.readtext(
                ocr_image,
                detail=1,
                paragraph=False,
                width_ths=0.7,
                height_ths=0.7,
                decoder="greedy"
            )
            
            # Convert to our format
            regions = self._parse_results(detections)
            
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
                metadata={'detection_count': len(regions)}
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.processing_stats['errors'] += 1
            self.logger.error(f"EasyOCR failed: {e}")
            
            return OCRResult(
                text="",
                confidence=0.0,
                processing_time=processing_time,
                engine_used=self.name,
                metadata={"error": str(e)}
            )
            
    def get_supported_languages(self) -> List[str]:
        """Get EasyOCR supported languages"""
        return [
            'en', 'ch_sim', 'ch_tra', 'ja', 'ko', 'th', 'vi', 'ar', 'hi', 'ur', 
            'fa', 'ru', 'bg', 'uk', 'be', 'te', 'kn', 'ta', 'bn', 'as', 'mr', 
            'ne', 'si', 'my', 'km', 'lo', 'sa', 'fr', 'de', 'es', 'pt', 'it',
            'nl', 'sv', 'da', 'no', 'fi', 'lt', 'lv', 'et', 'pl', 'cs', 'sk',
            'sl', 'hu', 'ro', 'hr', 'sr', 'bs', 'mk', 'sq', 'mt', 'cy', 'ga',
            'tr', 'az', 'uz', 'mn'
        ]
    
    def _prepare_image(self, image: np.ndarray) -> np.ndarray:
        """Convert to EasyOCR RGB format"""
        if len(image.shape) == 3:
            if image.shape[2] == 3:
                return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
        else:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    def _parse_results(self, detections: List) -> List[TextRegion]:
        """Convert EasyOCR results to TextRegions"""
        regions = []
        
        for detection in detections:
            try:
                if len(detection) >= 3:
                    points, text, confidence = detection
                    
                    if not text.strip() or confidence <= 0.1:
                        continue
                    
                    # Convert polygon to bbox
                    x_coords = [p[0] for p in points]
                    y_coords = [p[1] for p in points]
                    
                    bbox = BoundingBox(
                        x=int(min(x_coords)),
                        y=int(min(y_coords)),
                        width=int(max(x_coords) - min(x_coords)),
                        height=int(max(y_coords) - min(y_coords)),
                        confidence=confidence
                    )
                    
                    region = TextRegion(
                        text=text.strip(),
                        confidence=confidence,
                        bbox=bbox,
                        text_type=TextType.MIXED,
                        language=self.languages[0] if self.languages else "en"
                    )
                    
                    regions.append(region)
                    
            except Exception as e:
                self.logger.warning(f"Skipping detection: {e}")
                continue
        
        return regions
    
EasyOCR=EasyOCREngine