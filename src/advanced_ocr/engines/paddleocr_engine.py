
# src/advanced_ocr/engines/paddleocr_engine.py
"""
Advanced OCR PaddleOCR Engine

This module provides the PaddleOCR-based OCR engine implementation for the advanced OCR
system. PaddleOCR is optimized for printed text recognition and layout preservation,
offering excellent performance on structured documents and multi-language support.

The module focuses on:
- Printed text recognition with high accuracy
- Layout preservation and document structure understanding
- Multi-language OCR support (80+ languages)
- Robust detection and recognition pipeline
- Structured document processing capabilities
- Region-based and full-image OCR processing

Classes:
    PaddleOCREngine: PaddleOCR-based OCR engine implementation

Functions:
    _extract_implementation: Core OCR extraction logic
    _extract_full_image: Full image OCR processing
    _extract_from_regions: Region-based OCR processing
    _process_paddle_results: Result processing and formatting

Example:
    >>> engine = PaddleOCREngine(config)
    >>> engine.initialize()
    >>> result = engine.extract(image, text_regions)
    >>> print(f"Extracted text: {result.text}")

"""

import numpy as np
from typing import List, Optional, Tuple, Any
import warnings

from advanced_ocr.engines.base_engine import BaseOCREngine, EngineStatus
from advanced_ocr.results import OCRResult, BoundingBox, TextRegion
from advanced_ocr.config import OCRConfig
from advanced_ocr.utils.model_utils import ModelLoader, cached_model_load
from advanced_ocr.utils.image_utils import ImageProcessor
from advanced_ocr.utils.text_utils import TextCleaner

# Suppress PaddleOCR warnings
warnings.filterwarnings('ignore', category=UserWarning, module='paddle')

try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False
    # Create a dummy class for type hints when PaddleOCR is not available
    class PaddleOCR:
        pass


class PaddleOCREngine(BaseOCREngine):
    """
    PaddleOCR-based OCR engine optimized for printed text and layout preservation
    
    Strengths:
    - Excellent for printed text and documents
    - Good layout preservation
    - Multiple language support
    - Robust detection and recognition pipeline
    - Works well with structured documents
    
    Pipeline Integration:
    - Receives preprocessed image from engine_coordinator.py
    - Processes provided text regions or full image
    - Returns raw OCRResult for postprocessing
    """
    
    def __init__(self, config: OCRConfig):
        """Initialize PaddleOCR engine"""
        super().__init__(config)
        
        if not PADDLEOCR_AVAILABLE:
            raise ImportError("PaddleOCR not installed. Install with: pip install paddlepaddle paddleocr")
        
        # PaddleOCR-specific configuration
        self.paddle_config = getattr(config.engines, 'paddleocr', {})
        
        # Language support
        self.language = self.paddle_config.get('language', 'en')
        self.use_gpu = self.paddle_config.get('use_gpu', False)
        
        # Performance settings
        self.use_angle_cls = self.paddle_config.get('use_angle_cls', True)
        self.det_max_side_len = self.paddle_config.get('det_max_side_len', 1920)
        self.rec_batch_num = self.paddle_config.get('rec_batch_num', 8)
        
        # Detection thresholds
        self.det_db_thresh = self.paddle_config.get('det_db_thresh', 0.3)
        self.det_db_box_thresh = self.paddle_config.get('det_db_box_thresh', 0.6)
        
        # Model components
        self._paddle_ocr: Optional[PaddleOCR] = None
        self._model_loader = ModelLoader()
        self._image_processor = ImageProcessor()
        self._text_cleaner = TextCleaner()
        
        self.logger.info(f"PaddleOCR engine configured: lang={self.language}, gpu={self.use_gpu}")
    
    def _initialize_implementation(self):
        """Initialize PaddleOCR model"""
        self.logger.info("Loading PaddleOCR model...")
        
        try:
            # Load PaddleOCR with configuration
            self._paddle_ocr = cached_model_load(
                model_key=f"paddleocr_{self.language}_{self.use_gpu}",
                load_func=self._load_paddle_model,
                cache_timeout=3600  # 1 hour cache
            )
            
            # Test model with dummy input
            test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
            _ = self._paddle_ocr.ocr(test_image, cls=False)
            
            self.logger.info("PaddleOCR model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load PaddleOCR: {e}")
            raise
    
    def _load_paddle_model(self) -> PaddleOCR:
        """Load PaddleOCR model with configuration"""
        return PaddleOCR(
            lang=self.language,
            use_gpu=self.use_gpu,
            use_angle_cls=self.use_angle_cls,
            det_max_side_len=self.det_max_side_len,
            rec_batch_num=self.rec_batch_num,
            det_db_thresh=self.det_db_thresh,
            det_db_box_thresh=self.det_db_box_thresh,
            show_log=False  # Suppress logs
        )
    
    def _extract_implementation(self, image: np.ndarray, text_regions: List[TextRegion]) -> OCRResult:
        """
        Extract text using PaddleOCR
        
        Args:
            image: Preprocessed image from image_processor.py
            text_regions: Text regions from text_detector.py (can be empty)
            
        Returns:
            Raw OCRResult with extracted text and bounding boxes
        """
        
        # Ensure model is loaded
        if self._paddle_ocr is None:
            self.initialize()
        
        # Convert image format if needed
        ocr_image = self._prepare_image_for_paddle(image)
        
        # Extract text based on available regions
        if text_regions and len(text_regions) > 0:
            return self._extract_from_regions(ocr_image, text_regions)
        else:
            return self._extract_full_image(ocr_image)
    
    def _prepare_image_for_paddle(self, image: np.ndarray) -> np.ndarray:
        """Prepare image format for PaddleOCR"""
        
        # PaddleOCR expects RGB format
        if len(image.shape) == 2:
            # Grayscale to RGB
            image = self._image_processor.convert_grayscale_to_rgb(image)
        elif len(image.shape) == 3 and image.shape[2] == 4:
            # RGBA to RGB
            image = self._image_processor.convert_rgba_to_rgb(image)
        elif len(image.shape) == 3 and image.shape[2] == 3:
            # Already RGB, check if BGR (OpenCV format)
            # PaddleOCR expects RGB, but sometimes gets BGR
            pass
        
        # Ensure uint8 format
        if image.dtype != np.uint8:
            image = self._image_processor.normalize_to_uint8(image)
        
        return image
    
    def _extract_full_image(self, image: np.ndarray) -> OCRResult:
        """Extract text from full image using PaddleOCR"""
        
        try:
            # Run PaddleOCR on full image
            results = self._paddle_ocr.ocr(image, cls=self.use_angle_cls)
            
            if not results or not results[0]:
                return OCRResult(
                    text="",
                    confidence=0.0,
                    bounding_boxes=[],
                    metadata={'paddle_results': 'empty'}
                )
            
            # Process PaddleOCR results
            return self._process_paddle_results(results[0])
            
        except Exception as e:
            self.logger.error(f"PaddleOCR full image extraction failed: {e}")
            raise RuntimeError(f"PaddleOCR extraction error: {e}")
    
    def _extract_from_regions(self, image: np.ndarray, text_regions: List[TextRegion]) -> OCRResult:
        """Extract text from specific regions using PaddleOCR"""
        
        all_text_parts = []
        all_boxes = []
        confidence_scores = []
        
        for region in text_regions:
            try:
                # Extract region from image
                region_image = self._extract_region_image(image, region)
                
                if region_image.size == 0:
                    continue
                
                # Run PaddleOCR on region
                results = self._paddle_ocr.ocr(region_image, cls=self.use_angle_cls)
                
                if results and results[0]:
                    # Process region results
                    region_result = self._process_paddle_results(results[0])
                    
                    if region_result.text.strip():
                        all_text_parts.append(region_result.text.strip())
                        
                        # Adjust bounding boxes to global coordinates
                        for bbox in region_result.bounding_boxes:
                            adjusted_bbox = BoundingBox(
                                x=bbox.x + region.x,
                                y=bbox.y + region.y,
                                width=bbox.width,
                                height=bbox.height,
                                confidence=bbox.confidence
                            )
                            all_boxes.append(adjusted_bbox)
                        
                        confidence_scores.append(region_result.confidence)
                
            except Exception as e:
                self.logger.warning(f"Failed to process region {region}: {e}")
                continue
        
        # Combine results
        combined_text = ' '.join(all_text_parts)
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
        
        return OCRResult(
            text=combined_text,
            confidence=float(avg_confidence),
            bounding_boxes=all_boxes,
            metadata={
                'regions_processed': len(text_regions),
                'successful_extractions': len(all_text_parts)
            }
        )
    
    def _extract_region_image(self, image: np.ndarray, region: TextRegion) -> np.ndarray:
        """Extract region from image with bounds checking"""
        
        # Ensure coordinates are within image bounds
        y1 = max(0, int(region.y))
        y2 = min(image.shape[0], int(region.y + region.height))
        x1 = max(0, int(region.x))
        x2 = min(image.shape[1], int(region.x + region.width))
        
        if y1 >= y2 or x1 >= x2:
            return np.array([])  # Empty array for invalid region
        
        return image[y1:y2, x1:x2]
    
    def _process_paddle_results(self, paddle_results: List[Any]) -> OCRResult:
        """Process raw PaddleOCR results into OCRResult format"""
        
        if not paddle_results:
            return OCRResult(text="", confidence=0.0, bounding_boxes=[])
        
        text_parts = []
        bounding_boxes = []
        confidence_scores = []
        
        for line in paddle_results:
            if not line or len(line) != 2:
                continue
            
            coords, (text, confidence) = line
            
            if not text or not text.strip():
                continue
            
            # Clean text using text_utils
            cleaned_text = self._text_cleaner.clean_ocr_text(text)
            if not cleaned_text.strip():
                continue
            
            text_parts.append(cleaned_text)
            confidence_scores.append(confidence)
            
            # Convert coordinates to BoundingBox
            if coords and len(coords) == 4:
                bbox = self._coords_to_bounding_box(coords, confidence)
                if bbox:
                    bounding_boxes.append(bbox)
        
        # Combine text parts
        combined_text = ' '.join(text_parts)
        avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.0
        
        return OCRResult(
            text=combined_text,
            confidence=float(avg_confidence),
            bounding_boxes=bounding_boxes,
            metadata={
                'paddle_lines_detected': len(paddle_results),
                'valid_extractions': len(text_parts)
            }
        )
    
    def _coords_to_bounding_box(self, coords: List[List[float]], confidence: float) -> Optional[BoundingBox]:
        """Convert PaddleOCR coordinates to BoundingBox"""
        
        try:
            # PaddleOCR returns 4 corner points: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            x_coords = [point[0] for point in coords]
            y_coords = [point[1] for point in coords]
            
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)
            
            return BoundingBox(
                x=int(x_min),
                y=int(y_min),
                width=int(x_max - x_min),
                height=int(y_max - y_min),
                confidence=confidence
            )
            
        except (IndexError, TypeError, ValueError):
            return None
    
    def _cleanup_implementation(self):
        """Clean up PaddleOCR resources"""
        if self._paddle_ocr is not None:
            # PaddleOCR doesn't have explicit cleanup, but we can clear references
            self._paddle_ocr = None
            self.logger.debug("PaddleOCR model reference cleared")
    
    def get_engine_info(self) -> dict:
        """Get PaddleOCR-specific engine information"""
        return {
            'engine_type': 'paddleocr',
            'language': self.language,
            'use_gpu': self.use_gpu,
            'use_angle_cls': self.use_angle_cls,
            'det_max_side_len': self.det_max_side_len,
            'available': PADDLEOCR_AVAILABLE,
            'status': self.get_status().value
        }