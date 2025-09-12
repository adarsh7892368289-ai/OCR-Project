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
        
        # Layout preservation settings
        self.line_height_threshold = self.config.get("line_height_threshold", 15)
        self.word_spacing_threshold = self.config.get("word_spacing_threshold", 20)
        
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
        Process preprocessed image with PaddleOCR - FIXED: Returns single OCRResult with preserved layout
        
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
            
            # FIXED: Convert to single OCRResult with layout preservation
            result = self._combine_ocr_results_with_layout(paddle_results)
            
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
    
    def _combine_ocr_results_with_layout(self, paddle_results: List) -> OCRResult:
        """
        FIXED: Combine all PaddleOCR detections with preserved document layout
        """
        regions = []
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
                
                # Create text region with spatial information
                region = TextRegion(
                    text=text.strip(),
                    confidence=confidence,
                    bbox=bbox,
                    text_type=TextType.PRINTED,
                    language="en"
                )
                
                regions.append(region)
                detection_count += 1
        
        # FIXED: Reconstruct text with proper layout based on spatial positioning
        formatted_text = self._reconstruct_document_layout(regions)
        
        # Calculate overall confidence
        overall_confidence = sum(r.confidence for r in regions) / len(regions) if regions else 0.0
        
        # Create overall bounding box from all regions
        overall_bbox = self._calculate_overall_bbox(regions) if regions else BoundingBox(0, 0, 100, 30)
        
        # Return single OCRResult with preserved layout
        return OCRResult(
            text=formatted_text,
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
                'individual_confidences': [r.confidence for r in regions],
                'layout_preserved': True
            }
        )
    
    def _reconstruct_document_layout(self, regions: List[TextRegion]) -> str:
        """
        CORE FIX: Reconstruct document text with proper line breaks and formatting
        """
        if not regions:
            return ""
        
        # Sort regions by vertical position (top to bottom), then horizontal (left to right)
        sorted_regions = sorted(regions, key=lambda r: (r.bbox.y, r.bbox.x))
        
        # Group regions into lines based on Y-coordinate proximity
        lines = []
        current_line = []
        current_line_y = None
        
        for region in sorted_regions:
            region_center_y = region.bbox.y + region.bbox.height // 2
            
            if current_line_y is None:
                # First region
                current_line_y = region_center_y
                current_line.append(region)
            else:
                # Check if this region is on the same line
                y_distance = abs(region_center_y - current_line_y)
                
                if y_distance <= self.line_height_threshold:
                    # Same line - add to current line
                    current_line.append(region)
                else:
                    # New line - finish current line and start new one
                    if current_line:
                        lines.append(current_line)
                    current_line = [region]
                    current_line_y = region_center_y
        
        # Add the last line
        if current_line:
            lines.append(current_line)
        
        # Build the formatted text
        formatted_lines = []
        
        for line_regions in lines:
            # Sort regions in the line by X-coordinate (left to right)
            line_regions.sort(key=lambda r: r.bbox.x)
            
            # Combine text from regions in the line with appropriate spacing
            line_text_parts = []
            prev_region = None
            
            for region in line_regions:
                if prev_region is not None:
                    # Calculate horizontal distance between regions
                    horizontal_gap = region.bbox.x - (prev_region.bbox.x + prev_region.bbox.width)
                    
                    # Add extra spaces for large gaps (likely separate words/columns)
                    if horizontal_gap > self.word_spacing_threshold:
                        # Multiple spaces for large gaps
                        spaces = " " * min(4, max(2, horizontal_gap // 10))
                        line_text_parts.append(spaces)
                    else:
                        # Single space for normal word separation
                        line_text_parts.append(" ")
                
                line_text_parts.append(region.text)
                prev_region = region
            
            # Join the line parts and add to formatted lines
            line_text = "".join(line_text_parts).strip()
            if line_text:  # Only add non-empty lines
                formatted_lines.append(line_text)
        
        # Join all lines with newlines to preserve document structure
        return "\n".join(formatted_lines)
    
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
                'structure_analysis': self.supports_structure_analysis,
                'layout_preservation': True  # NEW: Added layout preservation capability
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
                'works_with_engine_manager': True,
                'preserves_document_layout': True  # NEW: Layout preservation feature
            }
        }
        
        try:
            # Try to get PaddleOCR version if available
            import paddle
            info['version'] = getattr(paddle, '__version__', 'unknown')
        except:
            info['version'] = 'unknown'
            
        return info