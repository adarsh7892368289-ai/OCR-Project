# src/advanced_ocr/engines/tesseract_engine.py
"""
Advanced OCR Tesseract Engine - SIMPLIFIED FOR NEW PIPELINE

CRITICAL UPDATE: Interface changed to extract(image) - NO text_regions parameter
Processes full images using Tesseract's built-in text detection

ARCHITECTURAL COMPLIANCE:
- Inherits from BaseOCREngine with new simplified interface
- Lazy initialization (no blocking in __init__)
- Processes full images (Tesseract handles detection internally)
- Returns simple OCRResult (no postprocessing)
- Optimal PSM mode selection for full image processing

PIPELINE INTEGRATION:
Receives: preprocessed numpy array from engine_coordinator.py
Returns: Raw OCRResult to engine_coordinator.py
"""

import numpy as np
from typing import Optional, Dict, Any
import os

from .base_engine import BaseOCREngine
from ..results import OCRResult
from ..config import OCRConfig

try:
    import pytesseract
    from PIL import Image
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    pytesseract = None
    Image = None


class TesseractEngine(BaseOCREngine):
    """
    Tesseract-based OCR engine - SIMPLIFIED INTERFACE
    
    UPDATED RESPONSIBILITIES:
    - Extract text from FULL IMAGES using Tesseract
    - Use optimal PSM modes for full page processing
    - Return RAW OCRResult with text and confidence
    - NO external text region processing
    - NO layout analysis beyond basic PSM modes
    
    REMOVED RESPONSIBILITIES:
    - Text region cropping and processing
    - Complex coordinate handling
    - Multi-region coordination
    """
    
    def __init__(self, config: OCRConfig):
        """Initialize configuration ONLY - NO model loading required for Tesseract"""
        super().__init__(config)
        
        if not TESSERACT_AVAILABLE:
            raise ImportError("PyTesseract not installed. Install with: pip install pytesseract")
        
        # Configuration extraction - safe access
        try:
            self.tesseract_config = getattr(config.engines, 'tesseract', {})
        except AttributeError:
            self.tesseract_config = {}
        
        # Tesseract parameters
        self.language = self.tesseract_config.get('language', 'eng')
        self.psm = self.tesseract_config.get('psm', 6)  # Default: uniform block of text
        self.oem = self.tesseract_config.get('oem', 3)  # Default: both neural nets and legacy
        
        # Custom Tesseract path if specified
        tesseract_cmd = self.tesseract_config.get('tesseract_cmd', None)
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        
        self.logger.info(f"Tesseract engine configured: lang={self.language}, psm={self.psm}, oem={self.oem}")
    
    def _initialize_implementation(self):
        """
        Initialize and verify Tesseract installation
        
        Tesseract doesn't require model loading, just verify it's working
        """
        try:
            # Test Tesseract availability and version
            version = pytesseract.get_tesseract_version()
            self.logger.info(f"Tesseract version: {version}")
            
            # Test with simple image
            test_image = np.ones((50, 100, 3), dtype=np.uint8) * 255
            pil_image = Image.fromarray(test_image)
            
            # Quick test extraction
            test_result = pytesseract.image_to_string(
                pil_image, 
                lang=self.language,
                config=f'--psm {self.psm} --oem {self.oem}'
            )
            
            self.logger.info("Tesseract initialized and verified successfully")
            
        except Exception as e:
            self.logger.error(f"Tesseract initialization failed: {e}")
            raise
    
    def _extract_implementation(self, image: np.ndarray) -> OCRResult:
        """
        UPDATED: Main extraction method - SIMPLIFIED INTERFACE
        
        CRITICAL CHANGE: Only takes image parameter (no text_regions)
        Uses Tesseract's built-in text detection on full image
        
        Args:
            image: PREPROCESSED numpy array from engine_coordinator.py
            
        Returns:
            Raw OCRResult for text_processor.py
        """
        try:
            # Convert numpy array to PIL Image
            pil_image = self._prepare_image_for_tesseract(image)
            
            # Extract text using optimal PSM mode for full image
            extracted_data = self._extract_full_image(pil_image)
            
            return OCRResult(
                text=extracted_data.get('text', ''),
                confidence=extracted_data.get('confidence', 0.0),
                engine_name=self.engine_name,
                success=True,
                metadata={
                    'extraction_method': 'full_image_tesseract',
                    'psm_mode': self.psm,
                    'oem_mode': self.oem,
                    'language': self.language,
                    'image_shape': image.shape,
                    **extracted_data
                }
            )
            
        except Exception as e:
            self.logger.error(f"Tesseract text extraction failed: {e}")
            return self._create_error_result(f"Text extraction failed: {e}")
    
    def _prepare_image_for_tesseract(self, image: np.ndarray) -> Image.Image:
        """Convert numpy array to PIL Image for Tesseract"""
        if image is None or image.size == 0:
            raise ValueError("Invalid input image")
        
        # Handle different image formats
        if len(image.shape) == 2:
            # Grayscale
            mode = 'L'
        elif len(image.shape) == 3:
            if image.shape[2] == 3:
                # RGB
                mode = 'RGB'
            elif image.shape[2] == 4:
                # RGBA
                mode = 'RGBA'
            else:
                raise ValueError(f"Unsupported image channels: {image.shape[2]}")
        else:
            raise ValueError(f"Unsupported image dimensions: {image.shape}")
        
        # Ensure uint8 format
        if image.dtype != np.uint8:
            if image.dtype in [np.float32, np.float64]:
                image = np.clip(image * 255, 0, 255).astype(np.uint8)
            else:
                image = np.clip(image, 0, 255).astype(np.uint8)
        
        return Image.fromarray(image, mode=mode)
    
    def _extract_full_image(self, pil_image: Image.Image) -> Dict[str, Any]:
        """
        Extract text from full image using Tesseract
        
        UPDATED: Uses optimal PSM modes for full page processing
        """
        try:
            # Build Tesseract configuration
            config = f'--psm {self.psm} --oem {self.oem}'
            
            # Extract text
            text = pytesseract.image_to_string(
                pil_image,
                lang=self.language,
                config=config
            ).strip()
            
            # Get detailed data with confidence scores
            try:
                detailed_data = pytesseract.image_to_data(
                    pil_image,
                    lang=self.language,
                    config=config,
                    output_type=pytesseract.Output.DICT
                )
                
                # Calculate average confidence from word-level confidences
                confidences = [
                    int(conf) for conf in detailed_data['conf'] 
                    if int(conf) > 0  # Filter out -1 confidence values
                ]
                
                avg_confidence = np.mean(confidences) / 100.0 if confidences else 0.0
                word_count = len([w for w in detailed_data['text'] if w.strip()])
                
            except Exception as e:
                self.logger.debug(f"Failed to get detailed data: {e}")
                avg_confidence = 0.8 if text else 0.0
                word_count = len(text.split()) if text else 0
            
            return {
                'text': text,
                'confidence': float(avg_confidence),
                'word_count': word_count,
                'psm_used': self.psm,
                'extraction_success': True
            }
            
        except Exception as e:
            self.logger.error(f"Tesseract extraction failed: {e}")
            return {
                'text': '',
                'confidence': 0.0,
                'word_count': 0,
                'extraction_success': False,
                'error': str(e)
            }
    
    def _create_error_result(self, error_message: str) -> OCRResult:
        """Create error result for failed operations"""
        return OCRResult(
            text="",
            confidence=0.0,
            engine_name=self.engine_name,
            success=False,
            error_message=error_message,
            metadata={'error_type': 'extraction_failed'}
        )
    
    def get_optimal_psm_for_content(self, image_characteristics: Dict[str, Any]) -> int:
        """
        Suggest optimal PSM mode based on image characteristics
        
        This could be used by preprocessing if image analysis is available
        """
        # PSM mode selection logic
        height, width = image_characteristics.get('dimensions', (0, 0))
        
        if width > height * 3:
            # Very wide image - likely single line
            return 8  # Single word
        elif height > width * 2:
            # Very tall image - likely single column
            return 4  # Single column
        elif width > 1000 and height > 1000:
            # Large image - likely full page
            return 3  # Fully automatic page segmentation
        else:
            # Standard image
            return 6  # Uniform block of text
    
    def get_engine_info(self) -> Dict[str, Any]:
        """Get comprehensive engine information for debugging"""
        try:
            tesseract_version = str(pytesseract.get_tesseract_version())
        except:
            tesseract_version = "unknown"
        
        return {
            'engine_type': 'tesseract',
            'engine_name': self.engine_name,
            'language': self.language,
            'psm_mode': self.psm,
            'oem_mode': self.oem,
            'tesseract_available': TESSERACT_AVAILABLE,
            'tesseract_version': tesseract_version,
            'tesseract_cmd': getattr(pytesseract.pytesseract, 'tesseract_cmd', 'default'),
            'status': self.get_status().value,
            'interface_version': 'simplified_full_image',
            'metrics': {
                'total_extractions': self.get_metrics().total_extractions,
                'success_rate': self.get_metrics().success_rate,
                'avg_confidence': self.get_metrics().avg_confidence
            }
        }