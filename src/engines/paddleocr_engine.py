import os
import cv2
import numpy as np
import time
import warnings
from typing import List, Dict, Any, Tuple, Optional

# Suppress PaddlePaddle warnings
warnings.filterwarnings('ignore', category=UserWarning, module='paddle')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from ..core.base_engine import (
    BaseOCREngine, 
    OCRResult, 
    BoundingBox, 
    TextRegion, 
    DocumentResult, 
    TextType
)

class PaddleOCREngine(BaseOCREngine):
    """
    Modern PaddleOCR Engine - Aligned with Pipeline Architecture
    
    Clean integration with your existing pipeline:
    - Takes preprocessed images from YOUR preprocessing pipeline
    - Returns single OCRResult compatible with YOUR base engine
    - Works with YOUR engine manager and postprocessing
    - Excellent for printed text and document analysis
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("PaddleOCR", config)
        self.ocr = None
        self.languages = self.config.get("languages", ["en"])
        self.gpu = self.config.get("gpu", True)
        self.model_storage_directory = self.config.get("model_dir", None)
        
        # Engine capabilities
        self.supports_handwriting = False
        self.supports_multiple_languages = True
        self.supports_orientation_detection = True
        self.supports_structure_analysis = True
        
    def initialize(self) -> bool:
        """Initialize PaddleOCR with robust error handling"""
        try:
            self.logger.info(f"Initializing PaddleOCR with languages: {self.languages}")
            
            # Import PaddleOCR
            from paddleocr import PaddleOCR
            
            # Initialize with modern configuration
            self.ocr = PaddleOCR(
                use_angle_cls=True,  # Enable angle classification
                lang='en',           # Default language
                show_log=False       # Suppress verbose logs
            )
            
            self.supported_languages = self.get_supported_languages()
            self.is_initialized = True
            self.model_loaded = True
            
            self.logger.info("PaddleOCR initialized successfully")
            return True
            
        except ImportError as e:
            self.logger.error(f"PaddleOCR not installed: {e}")
            self.logger.info("Install with: pip install paddlepaddle paddleocr")
            return False
        except Exception as e:
            self.logger.error(f"PaddleOCR initialization failed: {e}")
            self.ocr = None
            self.is_initialized = False
            self.model_loaded = False
            return False
            
    def get_supported_languages(self) -> List[str]:
        """Get comprehensive list of supported languages"""
        return [
            'en', 'ch', 'ta', 'te', 'ka', 'ja', 'ko', 'hi', 'ar', 'th', 'vi', 
            'ms', 'ur', 'fa', 'bg', 'uk', 'be', 'ru', 'sr', 'hr', 'ro', 'hu', 
            'pl', 'cs', 'sk', 'sl', 'hr', 'et', 'lv', 'lt', 'is', 'da', 'no', 
            'sv', 'fi', 'fr', 'de', 'es', 'pt', 'it', 'nl', 'ca', 'eu', 'gl'
        ]
        
    def process_image(self, preprocessed_image: np.ndarray, **kwargs) -> OCRResult:
        """
        Process preprocessed image with PaddleOCR - FIXED: Returns single OCRResult
        
        Args:
            preprocessed_image: Image from YOUR preprocessing pipeline
            **kwargs: Additional parameters
            
        Returns:
            OCRResult: Single result compatible with YOUR base engine
        """
        start_time = time.time()
        
        try:
            # Validate initialization
            if not self.is_initialized or self.ocr is None:
                if not self.initialize():
                    raise RuntimeError("PaddleOCR engine not initialized")
            
            # Validate preprocessed input
            if not self.validate_image(preprocessed_image):
                raise ValueError("Invalid preprocessed image")
            
            # Convert preprocessed image to PaddleOCR format
            paddle_image = self._prepare_for_paddleocr(preprocessed_image)
            
            # Extract OCR data with angle classification
            paddle_results = self.ocr.ocr(paddle_image, cls=True)
            
            # Convert to single OCRResult - FIXED: Combines all detections
            result = self._combine_ocr_results(paddle_results)
            
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
            self.logger.error(f"PaddleOCR extraction failed: {e}")
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
    
    def _prepare_for_paddleocr(self, image: np.ndarray) -> np.ndarray:
        """
        Minimal conversion for PaddleOCR compatibility
        Only format conversion - YOUR preprocessing pipeline handles enhancement
        """
        # PaddleOCR expects RGB format
        if len(image.shape) == 3:
            if image.shape[2] == 3:
                # Assume BGR from OpenCV, convert to RGB
                return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                return image
        else:
            # Convert grayscale to RGB
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    def _combine_ocr_results(self, paddle_results: List) -> OCRResult:
        """
        FIXED: Combine all PaddleOCR detections into single OCRResult
        This aligns with YOUR base engine's expect_text() interface
        """
        regions = []
        all_text_parts = []
        total_confidence = 0.0
        detection_count = 0
        
        # Handle empty results
        if not paddle_results or not paddle_results[0]:
            return OCRResult(
                text="",
                confidence=0.0,
                regions=[],
                engine_name=self.name,
                metadata={"detection_count": 0}
            )
        
        # Process each detection
        for detection in paddle_results[0]:
            if len(detection) >= 2:
                bbox_points, text_info = detection
                
                # Extract text and confidence
                text = text_info[0] if text_info and len(text_info) > 0 else ""
                confidence = text_info[1] if text_info and len(text_info) > 1 else 0.0
                
                # Filter very low confidence results
                if not text.strip() or confidence <= 0.1:
                    continue
                
                # Convert polygon to bounding box
                x, y, w, h = self._polygon_to_bbox(bbox_points)
                bbox = BoundingBox(
                    x=x, y=y, width=w, height=h, 
                    confidence=confidence,
                    text_type=TextType.PRINTED
                )
                
                # Create text region
                region = TextRegion(
                    text=text.strip(),
                    confidence=confidence,
                    bbox=bbox,
                    text_type=TextType.PRINTED,
                    language="en"
                )
                
                regions.append(region)
                all_text_parts.append(text.strip())
                total_confidence += confidence
                detection_count += 1
        
        # Combine all text parts
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
                'detection_method': 'paddleocr',
                'detection_count': detection_count,
                'has_angle_classification': True,
                'individual_confidences': [r.confidence for r in regions]
            }
        )
    
    def _polygon_to_bbox(self, points: List[List[float]]) -> Tuple[int, int, int, int]:
        """Convert polygon points to bounding box coordinates"""
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        return (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))
    
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
        """Get comprehensive engine information"""
        info = {
            'name': self.name,
            'version': None,
            'type': 'deep_learning_ocr',
            'supported_languages': self.get_supported_languages(),
            'capabilities': {
                'handwriting': self.supports_handwriting,
                'multiple_languages': self.supports_multiple_languages,
                'orientation_detection': self.supports_orientation_detection,
                'structure_analysis': self.supports_structure_analysis
            },
            'optimal_for': ['printed_text', 'documents', 'receipts', 'forms', 'structured_text'],
            'performance_profile': {
                'accuracy': 'high',
                'speed': 'fast',
                'memory_usage': 'high' if self.gpu else 'medium',
                'gpu_required': False,
                'gpu_recommended': True
            },
            'model_info': {
                'detection_model': 'DB',
                'recognition_model': 'CRNN', 
                'angle_classification': True,
                'languages_loaded': self.languages,
                'gpu_enabled': self.gpu
            },
            'pipeline_integration': {
                'uses_preprocessing_pipeline': True,
                'returns_single_result': True,
                'compatible_with_base_engine': True,
                'works_with_engine_manager': True
            }
        }
        
        try:
            # Try to get PaddleOCR version if available
            import paddle
            info['version'] = getattr(paddle, '__version__', 'unknown')
        except:
            info['version'] = 'unknown'
            
        return info