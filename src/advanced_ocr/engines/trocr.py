"""
TrOCR Engine Implementation

This module implements the TrOCR (Transformer-based OCR) engine for the advanced OCR library.
TrOCR is particularly effective for handwritten text and complex document layouts.

Responsibilities:
- Implement TrOCR-specific text extraction
- Handle TrOCR model initialization and configuration
- Convert TrOCR results to standard OCRResult format
- Manage TrOCR model loading and inference

Does NOT handle:
- Image enhancement or preprocessing (handled by ImageEnhancer)
- Quality analysis (handled by QualityAnalyzer)
- Engine management or selection (handled by EngineManager)
- Layout reconstruction (handled by pipeline)
"""

import logging
import time
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
from PIL import Image
import torch
import cv2

# Check for transformers availability
try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    TrOCRProcessor = None
    VisionEncoderDecoderModel = None

from ..core.base_engine import BaseOCREngine
from ..types import OCRResult, BoundingBox, TextRegion, TextType
from ..exceptions import EngineNotAvailableError, EngineInitializationError, ImageProcessingError


logger = logging.getLogger(__name__)


class TrOCREngine(BaseOCREngine):
    """
    TrOCR (Transformer-based OCR) engine implementation.
    
    TrOCR uses vision transformers for text recognition and is particularly
    effective for handwritten text, mathematical expressions, and complex layouts.
    
    Integrates with pipeline architecture:
    - Receives preprocessed images from pipeline
    - Returns raw OCR results to pipeline
    - Pipeline handles layout reconstruction
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize TrOCR engine.
        
        Args:
            config: Engine configuration dictionary
        """
        super().__init__("TrOCR", config)
        
        # Default configuration
        self._default_config = {
            'model_name': 'microsoft/trocr-base-printed',  # or 'microsoft/trocr-base-handwritten'
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'max_length': 384,
            'num_beams': 5,
            'early_stopping': True,
            'confidence_threshold': 0.3,
            'min_text_height': 15,
            'max_text_height': 200,
            'text_detection_method': 'contours',  # 'contours' or 'sliding_window'
            'region_padding': 5,
            'min_region_area': 100
        }
        
        # Merge with provided config
        self.config.update(self._default_config)
        if config:
            self.config.update(config)
        
        self._processor = None
        self._model = None
        self._device = self.config['device']
        
        # Engine capabilities
        self.supports_handwriting = True
        self.supports_multiple_languages = False  # TrOCR is typically English-focused
        self.supports_orientation_detection = False
        
        logger.info(f"TrOCR engine initialized with device: {self._device}")

    def is_available(self) -> bool:
        """Check if TrOCR dependencies are available"""
        if not TRANSFORMERS_AVAILABLE:
            return False
        
        try:
            import torch
            return True
        except ImportError:
            return False

    def initialize(self) -> bool:
        """
        Initialize TrOCR model and processor.
        
        Returns:
            bool: True if initialization successful
            
        Raises:
            EngineInitializationError: If model loading fails
        """
        if self.is_initialized:
            return True
        
        if not self.is_available():
            raise EngineNotAvailableError(
                "TrOCR", 
                "transformers library not installed. Install with: pip install transformers torch"
            )
            
        try:
            logger.info(f"Loading TrOCR model: {self.config['model_name']}")
            start_time = time.time()
            
            # Load processor and model
            self._processor = TrOCRProcessor.from_pretrained(
                self.config['model_name']
            )
            self._model = VisionEncoderDecoderModel.from_pretrained(
                self.config['model_name']
            )
            
            # Move model to appropriate device
            self._model.to(self._device)
            self._model.eval()
            
            load_time = time.time() - start_time
            logger.info(f"TrOCR model loaded successfully in {load_time:.2f}s")
            
            self.is_initialized = True
            return True
            
        except Exception as e:
            error_msg = f"Failed to initialize TrOCR model: {str(e)}"
            logger.error(error_msg)
            raise EngineInitializationError(error_msg, "TrOCR", e) from e

    def extract_text(self, image: np.ndarray) -> OCRResult:
        """
        Extract text from image using TrOCR - ONLY OCR processing.
        
        Args:
            image: Preprocessed image from pipeline (BGR format from OpenCV)
            
        Returns:
            OCRResult: Raw OCR results with text regions
        """
        start_time = time.time()
        
        try:
            # Validate initialization
            if not self.is_initialized:
                if not self.initialize():
                    raise RuntimeError("TrOCR engine not initialized")
            
            # Validate input
            if not self.validate_image(image):
                raise ValueError("Invalid input image")
            
            # Convert image format for TrOCR
            pil_image = self._prepare_image_for_trocr(image)
            
            # Detect text regions first
            text_regions = self._detect_text_regions(image)
            
            if not text_regions:
                # If no regions detected, process entire image
                text_regions = [(0, 0, image.shape[1], image.shape[0])]
            
            # Process each detected text region
            regions = []
            for region_coords in text_regions:
                try:
                    x, y, w, h = region_coords
                    
                    # Skip regions that are too small
                    if (w < self.config['min_text_height'] or 
                        h < self.config['min_text_height'] or
                        w * h < self.config['min_region_area']):
                        continue
                    
                    # Crop region from PIL image
                    region_image = pil_image.crop((x, y, x + w, y + h))
                    
                    # Extract text from region using TrOCR
                    region_text, confidence = self._extract_text_from_region(region_image)
                    
                    if region_text.strip() and confidence >= self.config['confidence_threshold']:
                        # Create text region
                        bbox = BoundingBox(
                            x=x, 
                            y=y, 
                            width=w, 
                            height=h,
                            confidence=0.9  # High confidence for bounding box detection
                        )
                        
                        text_region = TextRegion(
                            text=region_text.strip(),
                            confidence=confidence,
                            bbox=bbox,
                            text_type=TextType.HANDWRITTEN if 'handwritten' in self.config['model_name'].lower() else TextType.PRINTED,
                            language="en"
                        )
                        regions.append(text_region)
                        
                except Exception as e:
                    logger.warning(f"Failed to process text region {region_coords}: {e}")
                    continue
            
            # Create OCRResult - simple concatenation, no layout logic
            text = self._simple_text_concatenation(regions)
            confidence = self._calculate_average_confidence(regions)
            
            processing_time = time.time() - start_time
            
            result = OCRResult(
                text=text,
                confidence=confidence,
                processing_time=processing_time,
                engine_used=self.name,
                regions=regions,
                metadata={
                    'detection_count': len(regions),
                    'model_name': self.config['model_name'],
                    'device': self._device,
                    'total_regions_detected': len(text_regions)
                }
            )
            
            # Update statistics
            self.processing_stats['total_processed'] += 1
            self.processing_stats['total_time'] += processing_time
            if result.success:
                self.processing_stats['successful_extractions'] += 1
            else:
                self.processing_stats['errors'] += 1
            
            logger.info(f"TrOCR extracted {len(text)} chars, {len(regions)} regions in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"TrOCR failed: {e}")
            
            self.processing_stats['total_processed'] += 1
            self.processing_stats['total_time'] += processing_time
            self.processing_stats['errors'] += 1
            
            return OCRResult(
                text="",
                confidence=0.0,
                processing_time=processing_time,
                engine_used=self.name,
                metadata={"error": str(e)}
            )

    def get_supported_languages(self) -> List[str]:
        """Get TrOCR supported languages"""
        # TrOCR models are typically English-focused
        # Handwritten models may support some multilingual text
        if 'multilingual' in self.config['model_name'].lower():
            return ['en', 'de', 'fr', 'it', 'pt', 'es']
        return ['en']

    def _prepare_image_for_trocr(self, image: np.ndarray) -> Image.Image:
        """
        Convert image format for TrOCR compatibility.
        ONLY format conversion - no enhancement.
        """
        # Convert BGR to RGB for PIL
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif len(image.shape) == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image_rgb = image
            
        return Image.fromarray(image_rgb)

    def _detect_text_regions(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect text regions in image for TrOCR processing.
        ONLY detection - no enhancement or analysis.
        
        Returns:
            List of (x, y, width, height) tuples
        """
        try:
            if self.config['text_detection_method'] == 'contours':
                return self._detect_regions_by_contours(image)
            else:
                return self._detect_regions_by_sliding_window(image)
        except Exception as e:
            logger.warning(f"Text region detection failed: {e}")
            return []

    def _detect_regions_by_contours(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect text regions using contour analysis"""
        # Convert to grayscale for processing
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Simple thresholding to find text regions
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter regions by size
            if (w >= self.config['min_text_height'] and 
                h >= self.config['min_text_height'] and
                w <= self.config['max_text_height'] * 10 and  # reasonable width limit
                h <= self.config['max_text_height'] and
                w * h >= self.config['min_region_area']):
                
                # Add padding
                padding = self.config['region_padding']
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(image.shape[1] - x, w + 2 * padding)
                h = min(image.shape[0] - y, h + 2 * padding)
                
                regions.append((x, y, w, h))
        
        return regions

    def _detect_regions_by_sliding_window(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect text regions using sliding window approach"""
        regions = []
        height, width = image.shape[:2]
        
        # Define window sizes based on expected text height
        window_heights = [
            self.config['min_text_height'] * 2,
            self.config['min_text_height'] * 4,
            self.config['max_text_height']
        ]
        
        for win_h in window_heights:
            for y in range(0, height - win_h, win_h // 2):
                for x in range(0, width - win_h, win_h):
                    w = min(win_h * 4, width - x)  # Aspect ratio consideration
                    h = win_h
                    
                    if w * h >= self.config['min_region_area']:
                        regions.append((x, y, w, h))
        
        return regions

    def _extract_text_from_region(self, region_image: Image.Image) -> Tuple[str, float]:
        """
        Extract text from a single image region using TrOCR.
        
        Returns:
            Tuple of (extracted_text, confidence_score)
        """
        try:
            # Process image with TrOCR processor
            pixel_values = self._processor(region_image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(self._device)
            
            # Generate text with TrOCR model
            with torch.no_grad():
                generated_ids = self._model.generate(
                    pixel_values,
                    max_length=self.config['max_length'],
                    num_beams=self.config['num_beams'],
                    early_stopping=self.config['early_stopping']
                )
            
            # Decode generated text
            generated_text = self._processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Estimate confidence (TrOCR doesn't provide direct confidence scores)
            confidence = self._estimate_confidence(generated_text, region_image)
            
            return generated_text, confidence
            
        except Exception as e:
            logger.warning(f"TrOCR text extraction failed for region: {e}")
            return "", 0.0

    def _estimate_confidence(self, text: str, image: Image.Image) -> float:
        """
        Estimate confidence score for extracted text.
        TrOCR doesn't provide direct confidence, so we use heuristics.
        """
        if not text or not text.strip():
            return 0.0
        
        confidence = 0.5  # Base confidence
        
        # Increase confidence based on text characteristics
        if len(text.strip()) > 3:
            confidence += 0.1
        
        if any(char.isalnum() for char in text):
            confidence += 0.1
        
        if len(text.split()) > 1:
            confidence += 0.1
        
        # Decrease confidence for very short or suspicious text
        if len(text.strip()) < 2:
            confidence -= 0.2
        
        if text.count('?') / len(text) > 0.3:  # Many uncertain characters
            confidence -= 0.2
        
        return max(0.0, min(1.0, confidence))

    def _simple_text_concatenation(self, regions: List[TextRegion]) -> str:
        """
        Simple text concatenation - NO layout reconstruction.
        Pipeline will handle proper layout reconstruction.
        """
        if not regions:
            return ""
        
        # Sort by reading order (top to bottom, left to right) and concatenate
        sorted_regions = sorted(regions, key=lambda r: (
            r.bbox.y if r.bbox else 0,
            r.bbox.x if r.bbox else 0
        ))
        
        # Simple space-separated concatenation
        return " ".join(region.text for region in sorted_regions if region.text.strip())

    def _calculate_average_confidence(self, regions: List[TextRegion]) -> float:
        """Calculate average confidence from regions"""
        if not regions:
            return 0.0
        
        confidences = [r.confidence for r in regions if r.confidence > 0]
        return sum(confidences) / len(confidences) if confidences else 0.0

    def cleanup(self):
        """Clean up TrOCR resources"""
        if self._model is not None:
            del self._model
            self._model = None
        
        if self._processor is not None:
            del self._processor  
            self._processor = None
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        super().cleanup()
        logger.debug("TrOCR engine resources cleaned up")


# Export alias for consistency
TrOCR = TrOCREngine