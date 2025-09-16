
# src/advanced_ocr/engines/easyocr_engine.py
"""
Advanced OCR EasyOCR Engine

This module provides the EasyOCR-based OCR engine implementation for the advanced OCR
system. EasyOCR is optimized for handwritten text recognition and supports multiple
languages with GPU acceleration capabilities.

The module focuses on:
- Handwritten text recognition with high accuracy
- Multi-language OCR support (80+ languages)
- GPU acceleration for improved performance
- Robust processing of varied text orientations
- Natural scene text recognition capabilities
- Region-based and full-image OCR processing

Classes:
    EasyOCREngine: EasyOCR-based OCR engine implementation

Functions:
    _extract_implementation: Core OCR extraction logic
    _extract_full_image: Full image OCR processing
    _extract_from_regions: Region-based OCR processing
    _process_easyocr_results: Result processing and formatting

Example:
    >>> engine = EasyOCREngine(config)
    >>> engine.initialize()
    >>> result = engine.extract(image, text_regions)
    >>> print(f"Extracted text: {result.text}")

"""

import numpy as np
from typing import List, Optional, Tuple, Any
import warnings

from ..engines.base_engine import BaseOCREngine, EngineStatus
from ..results import OCRResult, BoundingBox, TextRegion
from ..config import OCRConfig
from ..utils.model_utils import ModelLoader, cached_model_load
from ..utils.image_utils import ImageProcessor
from ..utils.text_utils import TextCleaner

# Suppress EasyOCR warnings
warnings.filterwarnings('ignore', category=UserWarning, module='easyocr')

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    # Create a dummy class for type hints when EasyOCR is not available
    class easyocr:
        class Reader:
            pass


class EasyOCREngine(BaseOCREngine):
    """
    EasyOCR-based OCR engine optimized for handwritten text and multiple languages
    
    Strengths:
    - Excellent for handwritten text
    - Strong multi-language support
    - Good GPU acceleration
    - Robust for varied text orientations
    - Works well with natural scene text
    
    Pipeline Integration:
    - Receives preprocessed image from engine_coordinator.py
    - Processes provided text regions or full image
    - Returns raw OCRResult for postprocessing
    """
    
    def __init__(self, config: OCRConfig):
        """Initialize EasyOCR engine"""
        super().__init__(config)
        
        if not EASYOCR_AVAILABLE:
            raise ImportError("EasyOCR not installed. Install with: pip install easyocr")
        
        # EasyOCR-specific configuration
        self.easy_config = getattr(config.engines, 'easyocr', {})
        
        # Language support
        self.languages = self.easy_config.get('languages', ['en'])
        if isinstance(self.languages, str):
            self.languages = [self.languages]
        
        # GPU configuration
        self.use_gpu = self.easy_config.get('use_gpu', False)
        self.gpu_device = self.easy_config.get('gpu_device', 0)
        
        # Performance settings
        self.batch_size = self.easy_config.get('batch_size', 8)
        self.workers = self.easy_config.get('workers', 1)
        
        # Detection and recognition thresholds
        self.text_threshold = self.easy_config.get('text_threshold', 0.7)
        self.low_text = self.easy_config.get('low_text', 0.4)
        self.link_threshold = self.easy_config.get('link_threshold', 0.4)
        
        # Model components
        self._easyocr_reader: Optional[easyocr.Reader] = None
        self._model_loader = ModelLoader()
        self._image_processor = ImageProcessor()
        self._text_cleaner = TextCleaner()
        
        self.logger.info(f"EasyOCR engine configured: langs={self.languages}, gpu={self.use_gpu}")
    
    def _initialize_implementation(self):
        """Initialize EasyOCR reader"""
        self.logger.info("Loading EasyOCR model...")
        
        try:
            # Load EasyOCR with configuration
            self._easyocr_reader = cached_model_load(
                model_key=f"easyocr_{'_'.join(self.languages)}_{self.use_gpu}",
                load_func=self._load_easyocr_reader,
                cache_timeout=3600  # 1 hour cache
            )
            
            # Test model with dummy input
            test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
            _ = self._easyocr_reader.readtext(test_image, batch_size=1)
            
            self.logger.info("EasyOCR model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load EasyOCR: {e}")
            raise
    
    def _load_easyocr_reader(self) -> easyocr.Reader:
        """Load EasyOCR reader with configuration"""
        return easyocr.Reader(
            lang_list=self.languages,
            gpu=self.use_gpu,
            model_storage_directory=None,  # Use default
            user_network_directory=None,  # Use default
            recog_network='TRBC',  # Use default recognition network
            detect_network='craft',  # Use CRAFT for detection
            verbose=False  # Suppress logs
        )
    
    def _extract_implementation(self, image: np.ndarray, text_regions: List[TextRegion]) -> OCRResult:
        """
        Extract text using EasyOCR
        
        Args:
            image: Preprocessed image from image_processor.py
            text_regions: Text regions from text_detector.py (can be empty)
            
        Returns:
            Raw OCRResult with extracted text and bounding boxes
        """
        
        # Ensure model is loaded
        if self._easyocr_reader is None:
            self.initialize()
        
        # Convert image format if needed
        ocr_image = self._prepare_image_for_easyocr(image)
        
        # Extract text based on available regions
        if text_regions and len(text_regions) > 0:
            return self._extract_from_regions(ocr_image, text_regions)
        else:
            return self._extract_full_image(ocr_image)
    
    def _prepare_image_for_easyocr(self, image: np.ndarray) -> np.ndarray:
        """Prepare image format for EasyOCR"""
        
        # EasyOCR expects RGB format
        if len(image.shape) == 2:
            # Grayscale to RGB
            image = self._image_processor.convert_grayscale_to_rgb(image)
        elif len(image.shape) == 3 and image.shape[2] == 4:
            # RGBA to RGB
            image = self._image_processor.convert_rgba_to_rgb(image)
        elif len(image.shape) == 3 and image.shape[2] == 3:
            # Already RGB
            pass
        
        # Ensure uint8 format
        if image.dtype != np.uint8:
            image = self._image_processor.normalize_to_uint8(image)
        
        return image
    
    def _extract_full_image(self, image: np.ndarray) -> OCRResult:
        """Extract text from full image using EasyOCR"""
        
        try:
            # Run EasyOCR on full image
            results = self._easyocr_reader.readtext(
                image,
                batch_size=self.batch_size,
                workers=self.workers,
                text_threshold=self.text_threshold,
                low_text=self.low_text,
                link_threshold=self.link_threshold,
                canvas_size=2560,  # Max canvas size for processing
                mag_ratio=1.0,  # Magnification ratio
                slope_ths=0.1,  # Slope threshold for text line detection
                ycenter_ths=0.5,  # Y-center threshold for text grouping
                height_ths=0.7,  # Height threshold for text filtering
                width_ths=0.5,  # Width threshold for text filtering
                add_margin=0.1  # Add margin to detected text boxes
            )
            
            if not results:
                return OCRResult(
                    text="",
                    confidence=0.0,
                    bounding_boxes=[],
                    metadata={'easyocr_results': 'empty'}
                )
            
            # Process EasyOCR results
            return self._process_easyocr_results(results)
            
        except Exception as e:
            self.logger.error(f"EasyOCR full image extraction failed: {e}")
            raise RuntimeError(f"EasyOCR extraction error: {e}")
    
    def _extract_from_regions(self, image: np.ndarray, text_regions: List[TextRegion]) -> OCRResult:
        """Extract text from specific regions using EasyOCR"""
        
        all_text_parts = []
        all_boxes = []
        confidence_scores = []
        
        for region in text_regions:
            try:
                # Extract region from image
                region_image = self._extract_region_image(image, region)
                
                if region_image.size == 0:
                    continue
                
                # Run EasyOCR on region
                results = self._easyocr_reader.readtext(
                    region_image,
                    batch_size=self.batch_size,
                    workers=self.workers,
                    text_threshold=self.text_threshold,
                    low_text=self.low_text,
                    link_threshold=self.link_threshold
                )
                
                if results:
                    # Process region results
                    region_result = self._process_easyocr_results(results)
                    
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
    
    def _process_easyocr_results(self, easyocr_results: List[Any]) -> OCRResult:
        """Process raw EasyOCR results into OCRResult format"""
        
        if not easyocr_results:
            return OCRResult(text="", confidence=0.0, bounding_boxes=[])
        
        text_parts = []
        bounding_boxes = []
        confidence_scores = []
        
        for detection in easyocr_results:
            if not detection or len(detection) != 3:
                continue
            
            bbox_coords, text, confidence = detection
            
            if not text or not text.strip():
                continue
            
            # Clean text using text_utils
            cleaned_text = self._text_cleaner.clean_ocr_text(text)
            if not cleaned_text.strip():
                continue
            
            text_parts.append(cleaned_text)
            confidence_scores.append(confidence)
            
            # Convert coordinates to BoundingBox
            if bbox_coords and len(bbox_coords) == 4:
                bbox = self._coords_to_bounding_box(bbox_coords, confidence)
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
                'easyocr_detections': len(easyocr_results),
                'valid_extractions': len(text_parts)
            }
        )
    
    def _coords_to_bounding_box(self, coords: List[List[float]], confidence: float) -> Optional[BoundingBox]:
        """Convert EasyOCR coordinates to BoundingBox"""
        
        try:
            # EasyOCR returns 4 corner points: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
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
        """Clean up EasyOCR resources"""
        if self._easyocr_reader is not None:
            # EasyOCR doesn't have explicit cleanup, but we can clear references
            self._easyocr_reader = None
            self.logger.debug("EasyOCR reader reference cleared")
    
    def get_engine_info(self) -> dict:
        """Get EasyOCR-specific engine information"""
        return {
            'engine_type': 'easyocr',
            'languages': self.languages,
            'use_gpu': self.use_gpu,
            'batch_size': self.batch_size,
            'text_threshold': self.text_threshold,
            'available': EASYOCR_AVAILABLE,
            'status': self.get_status().value
        }