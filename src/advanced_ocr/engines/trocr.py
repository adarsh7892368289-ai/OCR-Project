"""
TrOCR Engine Implementation - DEVICE ISSUE FIXED

This module implements the TrOCR (Transformer-based OCR) engine for the advanced OCR library.
TrOCR is particularly effective for handwritten text and complex document layouts.

FIXED: Device configuration error resolved
"""

import logging
import time
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
from PIL import Image
import cv2

# Check for transformers availability
try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    TrOCRProcessor = None
    VisionEncoderDecoderModel = None
    torch = None

from ..core.base_engine import BaseOCREngine
from ..types import OCRResult, BoundingBox, TextRegion, TextType


class TrOCREngine(BaseOCREngine):
    """
    TrOCR (Transformer-based OCR) engine implementation.
    
    FIXED: Device configuration and initialization issues resolved
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("TrOCR", config)
        
        # FIXED: Proper device configuration
        if TRANSFORMERS_AVAILABLE and torch and torch.cuda.is_available():
            self.device = self.config.get("device", "cuda")
        else:
            self.device = "cpu"  # Always fallback to CPU, never "auto"
        
        self.model_name = self.config.get("model_name", "microsoft/trocr-base-printed")
        self.max_length = self.config.get("max_length", 384)
        
        self._processor = None
        self._model = None
        
        # Engine capabilities
        self.supports_handwriting = True
        self.supports_multiple_languages = False
        self.supports_orientation_detection = False
        self.supports_structure_analysis = False

    def is_available(self) -> bool:
        """Check if TrOCR dependencies are available"""
        return TRANSFORMERS_AVAILABLE and torch is not None

    def initialize(self) -> bool:
        """
        FIXED: TrOCR initialization with proper device handling
        """
        if not self.is_available():
            self.logger.error("TrOCR requires transformers library: pip install transformers torch")
            return False
            
        try:
            self.logger.info(f"Loading TrOCR model: {self.model_name} on device: {self.device}")
            
            # FIXED: Simple initialization like PaddleOCR/EasyOCR
            self._processor = TrOCRProcessor.from_pretrained(self.model_name)
            self._model = VisionEncoderDecoderModel.from_pretrained(self.model_name)
            
            # FIXED: Proper device handling
            if self.device == "cuda" and torch.cuda.is_available():
                self._model = self._model.to("cuda")
                self.logger.info("TrOCR using CUDA")
            else:
                self._model = self._model.to("cpu")
                self.device = "cpu"  # Ensure consistency
                self.logger.info("TrOCR using CPU")
            
            self._model.eval()
            
            self.is_initialized = True
            self.logger.info("TrOCR initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"TrOCR initialization failed: {e}")
            return False

    def get_supported_languages(self) -> List[str]:
        """Get TrOCR supported languages"""
        return ['en']  # TrOCR is primarily English-focused

    def extract_text(self, image: np.ndarray) -> OCRResult:
        """
        FIXED: Extract text from preprocessed image - matches PaddleOCR interface exactly
        """
        start_time = time.time()
        
        try:
            # Validate initialization
            if not self.is_initialized or self._model is None:
                if not self.initialize():
                    raise RuntimeError("TrOCR engine not initialized")
            
            # Validate preprocessed input
            if not self.validate_image(image):
                raise ValueError("Invalid preprocessed image")
            
            # Convert image for TrOCR
            pil_image = self._prepare_for_trocr(image)
            
            # Call TrOCR - SIMPLE approach like other engines
            try:
                extracted_text, confidence = self._extract_with_trocr(pil_image)
            except Exception as e:
                self.logger.error(f"TrOCR extraction failed: {e}")
                raise e
            
            # Create result in same format as other engines
            result = self._create_ocr_result(extracted_text, confidence, image.shape)
            
            # Set processing time and engine name
            processing_time = time.time() - start_time
            result.processing_time = processing_time
            result.engine_used = self.name  # Match PaddleOCR field name
            
            # Update stats like other engines
            self.processing_stats['total_processed'] += 1
            self.processing_stats['total_time'] += processing_time
            if result.text.strip():
                self.processing_stats['successful_extractions'] += 1
                self.logger.info(f"SUCCESS: TrOCR extracted {len(result.text)} chars (conf: {result.confidence:.3f}) in {processing_time:.2f}s")
            else:
                self.logger.warning(f"NO TEXT: TrOCR found no text in {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"TrOCR failed: {e}")
            self.processing_stats['errors'] += 1
            
            return OCRResult(
                text="",
                confidence=0.0,
                processing_time=processing_time,
                engine_used=self.name,
                metadata={"error": str(e)}
            )

    def _prepare_for_trocr(self, image: np.ndarray) -> Image.Image:
        """
        FIXED: Minimal conversion for TrOCR compatibility - matches other engines
        Only format conversion - YOUR preprocessing pipeline handles enhancement
        """
        # Convert BGR to RGB for PIL (like PaddleOCR does)
        if len(image.shape) == 3:
            if image.shape[2] == 3:
                # Assume BGR from OpenCV, convert to RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
        else:
            # Convert grayscale to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
        return Image.fromarray(image_rgb)

    def _extract_with_trocr(self, pil_image: Image.Image) -> Tuple[str, float]:
        """
        FIXED: Simple TrOCR extraction without complex region detection
        """
        try:
            # Process entire image with TrOCR
            pixel_values = self._processor(pil_image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(self.device)
            
            # Generate text
            with torch.no_grad():
                generated_ids = self._model.generate(
                    pixel_values,
                    max_length=self.max_length,
                    num_beams=5,
                    early_stopping=True
                )
            
            # Decode text
            generated_text = self._processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            # Simple confidence estimation (TrOCR doesn't provide direct confidence)
            confidence = 0.8 if generated_text.strip() else 0.0
            
            return generated_text, confidence
            
        except Exception as e:
            self.logger.error(f"TrOCR extraction error: {e}")
            return "", 0.0

    def _create_ocr_result(self, text: str, confidence: float, image_shape: Tuple[int, int, int]) -> OCRResult:
        """
        FIXED: Create OCRResult in same format as other engines
        """
        regions = []
        
        if text.strip():
            # Create single region for entire image (like simple OCR)
            height, width = image_shape[:2]
            bbox = BoundingBox(
                x=0, y=0, width=width, height=height,
                confidence=confidence
            )
            
            region = TextRegion(
                text=text.strip(),
                confidence=confidence,
                bbox=bbox,
                text_type=TextType.HANDWRITTEN if 'handwritten' in self.model_name.lower() else TextType.PRINTED,
                language="en"
            )
            regions.append(region)
        
        # Calculate overall bbox like other engines
        overall_bbox = self._calculate_overall_bbox(regions, image_shape) if regions else BoundingBox(0, 0, 100, 30)
        
        return OCRResult(
            text=text,
            confidence=confidence,
            regions=regions,
            bbox=overall_bbox,
            engine_used=self.name,
            metadata={
                'detection_method': 'trocr',
                'model_name': self.model_name,
                'device': self.device,
                'detection_count': len(regions)
            }
        )

    def _calculate_overall_bbox(self, regions: List[TextRegion], image_shape: Tuple[int, int, int]) -> BoundingBox:
        """Calculate overall bounding box - matches other engines"""
        if not regions:
            return BoundingBox(0, 0, image_shape[1], image_shape[0])
        
        # For single region, return that region's bbox
        if len(regions) == 1:
            return regions[0].bbox
        
        # For multiple regions, calculate combined bbox
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

    def cleanup(self):
        """Clean up TrOCR resources"""
        if self._model is not None:
            del self._model
            self._model = None
        
        if self._processor is not None:
            del self._processor  
            self._processor = None
            
        if torch and torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        super().cleanup()


# Export alias for consistency
TrOCR = TrOCREngine