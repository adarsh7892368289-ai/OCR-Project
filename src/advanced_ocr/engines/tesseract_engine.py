# src/advanced_ocr/engines/tesseract_engine.py
"""
Advanced OCR Tesseract Engine

This module provides the Tesseract-based OCR engine implementation for the advanced OCR
system. Tesseract is a widely-used open-source OCR engine that excels at recognizing
printed text with good accuracy and supports multiple languages.

The module focuses on:
- Printed text recognition with high accuracy
- Multi-language OCR support (100+ languages)
- Optimized configuration for different text types
- Region-based and full-image OCR processing
- Confidence scoring and text validation
- Tesseract-specific image preprocessing enhancements

Classes:
    TesseractEngine: Main Tesseract-based OCR engine implementation
    TesseractEnhancedEngine: Enhanced variant with additional preprocessing

Functions:
    _extract_implementation: Core OCR extraction logic
    _extract_full_image: Full image OCR processing
    _extract_from_regions: Region-based OCR processing
    _process_tesseract_results: Result processing and formatting

Example:
    >>> engine = TesseractEngine(config)
    >>> engine.initialize()
    >>> result = engine.extract(image, text_regions)
    >>> print(f"Extracted text: {result.text}")

"""


import cv2
import numpy as np
import pytesseract
from typing import List, Optional, Dict, Any, Tuple
from PIL import Image
import logging

from .base_engine import BaseOCREngine, EngineStatus
from ..results import OCRResult, TextRegion, Word, Line, BoundingBox, ConfidenceMetrics
from ..utils.image_utils import ImageProcessor, CoordinateTransformer
from ..utils.text_utils import TextCleaner, TextValidator
from ..config import EngineConfig


class TesseractEngine(BaseOCREngine):
    """
    Tesseract OCR Engine with optimized configuration.
    
    Focuses purely on text extraction from preprocessed images and text regions.
    No preprocessing, no postprocessing - just efficient Tesseract text extraction.
    """
    
    def __init__(self, config: EngineConfig):
        """Initialize Tesseract engine with optimized settings."""
        super().__init__("tesseract", config)
        
        # Initialize pipeline utilities correctly
        self.image_processor = ImageProcessor()
        self.text_cleaner = TextCleaner()
        self.text_validator = TextValidator()
        self.coord_transformer = CoordinateTransformer()
        
        # Tesseract-specific configurations from config
        self.tesseract_config = self._build_tesseract_config()
        self.oem_mode = getattr(config, 'oem_mode', 3)  # LSTM OCR Engine Mode
        self.psm_mode = getattr(config, 'psm_mode', 6)  # Assume uniform block of text
        
        # Performance optimizations from config
        self.dpi = getattr(config, 'dpi', 300)
        self.timeout = getattr(config, 'timeout', 30)  # seconds
        
        # Language support from config
        self.languages = getattr(config, 'languages', 'eng')
        
        self._validate_tesseract_installation()
        
    def _validate_tesseract_installation(self) -> None:
        """Validate that Tesseract is properly installed."""
        try:
            version = pytesseract.get_tesseract_version()
            self.logger.info(f"Tesseract version: {version}")
            self._status = EngineStatus.READY
        except Exception as e:
            self.logger.error(f"Tesseract validation failed: {e}")
            self._status = EngineStatus.ERROR
            raise RuntimeError(f"Tesseract installation invalid: {e}")
    
    def _build_tesseract_config(self) -> str:
        """Build optimized Tesseract configuration string."""
        config_parts = []
        
        # Basic configuration
        config_parts.extend([
            f'--oem {self.oem_mode}',
            f'--psm {self.psm_mode}'
        ])
        
        # Advanced options
        config_options = [
            'preserve_interword_spaces=1',
            'tessedit_do_invert=0',
            'tessedit_char_blacklist=',
            'load_system_dawg=1',
            'load_freq_dawg=1'
        ]
        
        # Add character whitelist if specified in config
        if hasattr(self.config, 'char_whitelist') and self.config.char_whitelist:
            config_options.append(f'tessedit_char_whitelist={self.config.char_whitelist}')
        
        # Add config options with -c prefix
        for option in config_options:
            config_parts.append(f'-c {option}')
            
        return ' '.join(config_parts)
    
    def extract(self, image: np.ndarray, text_regions: List[BoundingBox]) -> OCRResult:
        """
        Extract text from preprocessed image using provided text regions.
        
        Args:
            image: ALREADY PREPROCESSED image from image_processor.py
            text_regions: Text regions from text_detector.py (via image_processor.py)
            
        Returns:
            OCRResult with extracted text and basic confidence scores
        """
        if self._status != EngineStatus.READY:
            raise RuntimeError(f"Engine not ready: {self._status}")
            
        try:
            self._status = EngineStatus.PROCESSING
            
            # Convert numpy array to PIL Image for Tesseract
            if isinstance(image, np.ndarray):
                # Handle different channel formats
                if len(image.shape) == 3 and image.shape[2] == 3:
                    # BGR to RGB conversion
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(image)
            else:
                pil_image = image
            
            # Extract text from each region
            extracted_words = []
            total_confidence = 0.0
            processed_regions = 0
            
            for region in text_regions:
                try:
                    # Extract region from image
                    region_image = self._extract_region(pil_image, region)
                    
                    if region_image is None:
                        continue
                    
                    # Run Tesseract on region
                    region_result = self._extract_from_region(region_image, region)
                    
                    if region_result:
                        extracted_words.extend(region_result['words'])
                        total_confidence += region_result['confidence']
                        processed_regions += 1
                        
                except Exception as e:
                    self.logger.warning(f"Failed to process region {region}: {e}")
                    continue
            
            # Calculate overall confidence
            overall_confidence = total_confidence / max(processed_regions, 1)
            
            # Create confidence metrics
            confidence_metrics = ConfidenceMetrics(
                overall_confidence=overall_confidence,
                engine_confidence=overall_confidence,
                consensus_confidence=0.0,  # Single engine, no consensus
                validation_confidence=self._calculate_validation_confidence(extracted_words)
            )
            
            # Build result
            result = OCRResult(
                engine_name=self.name,
                text=self._combine_words_to_text(extracted_words),
                words=extracted_words,
                confidence=confidence_metrics,
                processing_time=0.0,  # Will be set by base class
                metadata={
                    'tesseract_version': str(pytesseract.get_tesseract_version()),
                    'config_used': self.tesseract_config,
                    'regions_processed': processed_regions,
                    'total_regions': len(text_regions)
                }
            )
            
            self._status = EngineStatus.READY
            return result
            
        except Exception as e:
            self._status = EngineStatus.ERROR
            self.logger.error(f"Tesseract extraction failed: {e}")
            raise RuntimeError(f"Text extraction failed: {e}")
    
    def _extract_region(self, image: Image.Image, region: BoundingBox) -> Optional[Image.Image]:
        """Extract a region from the image."""
        try:
            # Convert coordinates to integers
            x1, y1, x2, y2 = region.to_xyxy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Validate coordinates
            if x1 >= x2 or y1 >= y2:
                return None
                
            # Ensure coordinates are within image bounds
            img_width, img_height = image.size
            x1 = max(0, min(x1, img_width))
            y1 = max(0, min(y1, img_height))
            x2 = max(x1, min(x2, img_width))
            y2 = max(y1, min(y2, img_height))
            
            # Extract region
            region_image = image.crop((x1, y1, x2, y2))
            
            # Skip tiny regions
            if region_image.width < 10 or region_image.height < 10:
                return None
                
            return region_image
            
        except Exception as e:
            self.logger.warning(f"Failed to extract region {region}: {e}")
            return None
    
    def _extract_from_region(self, region_image: Image.Image, region_bbox: BoundingBox) -> Optional[Dict[str, Any]]:
        """Extract text from a single region using Tesseract."""
        try:
            # Get detailed data from Tesseract
            data = pytesseract.image_to_data(
                region_image,
                lang=self.languages,
                config=self.tesseract_config,
                output_type=pytesseract.Output.DICT,
                timeout=self.timeout
            )
            
            # Process Tesseract output
            words = []
            confidences = []
            
            for i in range(len(data['text'])):
                text = data['text'][i].strip()
                
                if not text or len(text) < 1:
                    continue
                
                # Clean text using text_utils.py
                cleaned_text = self.text_cleaner.clean_ocr_text(text)
                if not cleaned_text:
                    continue
                
                # Validate text using text_utils.py  
                if not self.text_validator.is_valid_text(cleaned_text):
                    continue
                
                confidence = float(data['conf'][i])
                if confidence < 0:  # Tesseract uses -1 for no confidence
                    confidence = 0.0
                
                # Convert relative coordinates to absolute
                rel_x = data['left'][i]
                rel_y = data['top'][i]
                rel_width = data['width'][i]
                rel_height = data['height'][i]
                
                # Transform coordinates using image_utils.py
                abs_x = region_bbox.x + rel_x
                abs_y = region_bbox.y + rel_y
                
                word_bbox = BoundingBox(
                    x=abs_x,
                    y=abs_y,
                    width=rel_width,
                    height=rel_height
                )
                
                # Create Word object
                word = Word(
                    text=cleaned_text,
                    bbox=word_bbox,
                    confidence=confidence / 100.0  # Convert to 0-1 range
                )
                
                words.append(word)
                confidences.append(confidence)
            
            if not words:
                return None
            
            # Calculate region confidence
            region_confidence = sum(confidences) / len(confidences) if confidences else 0.0
            
            return {
                'words': words,
                'confidence': region_confidence / 100.0  # Convert to 0-1 range
            }
            
        except Exception as e:
            self.logger.warning(f"Tesseract processing failed for region: {e}")
            return None
    
    def _calculate_validation_confidence(self, words: List[Word]) -> float:
        """Calculate validation confidence based on text characteristics."""
        if not words:
            return 0.0
        
        total_score = 0.0
        
        for word in words:
            # Length score (longer words generally more reliable)
            length_score = min(len(word.text) / 10.0, 1.0)
            
            # Character validity score
            valid_chars = sum(1 for c in word.text if c.isalnum() or c.isspace())
            char_score = valid_chars / max(len(word.text), 1)
            
            # Dictionary validation using text_utils.py
            dict_score = 1.0 if self.text_validator.is_valid_text(word.text) else 0.5
            
            word_score = (length_score * 0.3 + char_score * 0.4 + dict_score * 0.3)
            total_score += word_score
        
        return total_score / len(words)
    
    def _combine_words_to_text(self, words: List[Word]) -> str:
        """Combine words into readable text."""
        if not words:
            return ""
        
        # Sort words by position (top-to-bottom, left-to-right)
        sorted_words = sorted(words, key=lambda w: (w.bbox.y, w.bbox.x))
        
        # Group words into lines based on Y position
        lines = []
        current_line = []
        last_y = None
        y_threshold = 10  # pixels
        
        for word in sorted_words:
            if last_y is None or abs(word.bbox.y - last_y) > y_threshold:
                # New line
                if current_line:
                    lines.append(current_line)
                current_line = [word]
                last_y = word.bbox.y
            else:
                current_line.append(word)
                
        if current_line:
            lines.append(current_line)
        
        # Join words in each line, then join lines
        text_lines = []
        for line in lines:
            # Sort words in line by X position
            line_words = sorted(line, key=lambda w: w.bbox.x)
            line_text = ' '.join(word.text for word in line_words)
            text_lines.append(line_text)
        
        return '\n'.join(text_lines)
    
    def get_engine_info(self) -> Dict[str, Any]:
        """Get engine information."""
        try:
            version = str(pytesseract.get_tesseract_version())
        except:
            version = "unknown"
            
        return {
            'engine': 'tesseract',
            'version': version,
            'languages': self.languages,
            'oem_mode': self.oem_mode,
            'psm_mode': self.psm_mode,
            'status': self._status.value
        }
    
    def cleanup(self) -> None:
        """Cleanup engine resources."""
        # Tesseract doesn't require explicit cleanup
        self._status = EngineStatus.READY
        self.logger.info("Tesseract engine cleaned up")


class TesseractEnhancedEngine(TesseractEngine):
    """
    Enhanced Tesseract engine with advanced preprocessing for specific use cases.
    
    This variant applies Tesseract-specific optimizations before text extraction.
    Use this when you need Tesseract-specific image enhancements.
    """
    
    def __init__(self, config: EngineConfig):
        super().__init__(config)
        self.enhance_for_tesseract = getattr(config, 'enhance_for_tesseract', True)
    
    def _extract_region(self, image: Image.Image, region: BoundingBox) -> Optional[Image.Image]:
        """Extract and enhance region specifically for Tesseract."""
        region_image = super()._extract_region(image, region)
        
        if region_image is None or not self.enhance_for_tesseract:
            return region_image
        
        try:
            # Convert to numpy for processing
            img_array = np.array(region_image)
            
            # Tesseract-specific enhancements
            # 1. Ensure sufficient contrast
            img_array = self._enhance_contrast(img_array)
            
            # 2. Resize if too small (Tesseract works better with larger images)
            if img_array.shape[0] < 32 or img_array.shape[1] < 32:
                scale_factor = max(32 / img_array.shape[0], 32 / img_array.shape[1])
                new_height = int(img_array.shape[0] * scale_factor)
                new_width = int(img_array.shape[1] * scale_factor)
                img_array = cv2.resize(img_array, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            
            # 3. Ensure it's grayscale (Tesseract often works better with grayscale)
            if len(img_array.shape) == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            
            return Image.fromarray(img_array)
            
        except Exception as e:
            self.logger.warning(f"Region enhancement failed, using original: {e}")
            return region_image
    
    def _enhance_contrast(self, img_array: np.ndarray) -> np.ndarray:
        """Apply CLAHE for better contrast."""
        try:
            if len(img_array.shape) == 3:
                # Convert to LAB color space for better contrast enhancement
                lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                lab[:,:,0] = clahe.apply(lab[:,:,0])
                return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            else:
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                return clahe.apply(img_array)
        except Exception:
            return img_array