# src/advanced_ocr/engines/easyocr.py
"""
EasyOCR Engine Implementation - Aligned with PaddleOCR Quality and Pipeline Architecture

Fixed to match PaddleOCR's layout reconstruction and filtering quality while
maintaining clean pipeline integration.
"""

import cv2
import numpy as np
import time
import re
from typing import List, Dict, Any, Optional, Tuple
import logging

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    easyocr = None

from ..core.base_engine import BaseOCREngine
from ..types import OCRResult, TextRegion, BoundingBox, TextType


class EasyOCREngine(BaseOCREngine):
    """
    EasyOCR Engine - High Quality Layout Reconstruction
    
    Fixed to provide quality results matching PaddleOCR:
    - Intelligent text filtering and quality assessment
    - Sophisticated layout reconstruction with line grouping
    - Proper spatial analysis for document structure
    - Clean pipeline integration
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize EasyOCR engine with enhanced configuration"""
        super().__init__("EasyOCR", config)
        
        # Check availability first
        if not EASYOCR_AVAILABLE:
            self.logger.error("EasyOCR not installed. Install with: pip install easyocr")
            
        self.reader = None
        
        # Configuration
        self.languages = self.config.get("languages", ["en"])
        self.gpu = self.config.get("gpu", True)
        self.model_storage_directory = self.config.get("model_dir", None)
        self.download_enabled = self.config.get("download_enabled", True)
        
        # Enhanced detection parameters for better quality
        self.width_ths = self.config.get("width_ths", 0.7)
        self.height_ths = self.config.get("height_ths", 0.7)
        self.decoder = self.config.get("decoder", "greedy")
        self.beam_width = self.config.get("beam_width", 5)
        self.batch_size = self.config.get("batch_size", 1)
        
        # Enhanced quality thresholds (stricter filtering like PaddleOCR)
        self.min_confidence = self.config.get("min_confidence", 0.3)  # Raised from 0.1
        self.min_text_length = self.config.get("min_text_length", 2)  # Raised from 1
        self.min_bbox_area = self.config.get("min_bbox_area", 100)  # New: filter tiny boxes
        
        # Layout reconstruction settings (matching PaddleOCR approach)
        self.line_height_threshold = self.config.get("line_height_threshold", 15)
        self.word_spacing_threshold = self.config.get("word_spacing_threshold", 20)
        
        # Engine capabilities
        self.model_loaded = False
        self.supported_languages = []
    
    def initialize(self) -> bool:
        """Initialize EasyOCR reader with enhanced error handling"""
        if not EASYOCR_AVAILABLE:
            self.logger.error("EasyOCR not available - cannot initialize")
            return False
        
        try:
            self.logger.info(f"Initializing EasyOCR with languages: {self.languages}")
            
            # Initialize EasyOCR reader with optimized parameters
            self.reader = easyocr.Reader(
                self.languages,
                gpu=self.gpu,
                model_storage_directory=self.model_storage_directory,
                download_enabled=self.download_enabled,
                detector=True,
                recognizer=True,
                verbose=False
            )
            
            self.supported_languages = self.languages.copy()
            self.is_initialized = True
            self.model_loaded = True
            
            self.logger.info(f"EasyOCR initialized successfully (GPU: {self.gpu})")
            return True
            
        except Exception as e:
            self.logger.error(f"EasyOCR initialization failed: {e}")
            self.reader = None
            self.is_initialized = False
            self.model_loaded = False
            return False
    
    def is_available(self) -> bool:
        """Check if EasyOCR is available and initialized"""
        return (
            EASYOCR_AVAILABLE and 
            self.is_initialized and 
            self.reader is not None and 
            self.model_loaded
        )
    
    def extract_text(self, image: np.ndarray) -> OCRResult:
        """
        Extract text with enhanced quality matching PaddleOCR approach
        
        Args:
            image: Preprocessed image from ImageEnhancer (numpy array)
            
        Returns:
            OCRResult: High-quality text extraction with layout reconstruction
        """
        start_time = time.time()
        
        try:
            # Validate inputs
            if not self.is_available():
                raise RuntimeError("EasyOCR engine not available")
            
            if not self.validate_image(image):
                raise ValueError("Invalid image for text extraction")
            
            # Prepare image for EasyOCR
            easyocr_image = self._convert_image_format(image)
            
            # Extract text using EasyOCR with optimized parameters
            detections = self.reader.readtext(
                easyocr_image,
                detail=1,  # Get detailed results with bounding boxes
                paragraph=False,  # Word/line level for better control
                width_ths=self.width_ths,
                height_ths=self.height_ths,
                decoder=self.decoder,
                beamWidth=self.beam_width,
                batch_size=self.batch_size
            )
            
            # Apply enhanced filtering and quality assessment
            filtered_detections = self._filter_and_enhance_detections(detections)
            
            # Convert to OCRResult with sophisticated layout reconstruction
            result = self._combine_detections_with_advanced_layout(filtered_detections)
            
            # Set engine metadata
            processing_time = time.time() - start_time
            result.processing_time = processing_time
            result.engine_used = self.name
            
            # Update performance stats
            self._update_stats(result, processing_time)
            
            # Log results similar to PaddleOCR
            if result.text.strip():
                self.logger.info(f"SUCCESS: EasyOCR extracted {len(result.text)} chars "
                               f"(conf: {result.confidence:.3f}) in {processing_time:.2f}s")
            else:
                self.logger.warning(f"NO TEXT: EasyOCR found no text in {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.processing_stats['errors'] += 1
            self.processing_stats['total_time'] += processing_time
            
            self.logger.error(f"EasyOCR text extraction failed: {e}")
            
            return OCRResult(
                text="",
                confidence=0.0,
                processing_time=processing_time,
                engine_used=self.name,
                metadata={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "extraction_failed": True
                }
            )
    
    def get_supported_languages(self) -> List[str]:
        """Get list of languages supported by EasyOCR"""
        return [
            'en', 'ch_sim', 'ch_tra', 'ja', 'ko', 'th', 'vi', 'ar', 'hi', 'ur', 
            'fa', 'ru', 'bg', 'uk', 'be', 'te', 'kn', 'ta', 'bn', 'as', 'mr', 
            'ne', 'si', 'my', 'km', 'lo', 'sa', 'fr', 'de', 'es', 'pt', 'it',
            'nl', 'sv', 'da', 'no', 'fi', 'lt', 'lv', 'et', 'pl', 'cs', 'sk',
            'sl', 'hu', 'ro', 'hr', 'sr', 'bs', 'mk', 'sq', 'mt', 'cy', 'ga',
            'tr', 'az', 'uz', 'mn'
        ]
    
    def _convert_image_format(self, image: np.ndarray) -> np.ndarray:
        """Convert image to EasyOCR compatible format"""
        # EasyOCR expects RGB format
        if len(image.shape) == 3:
            if image.shape[2] == 3:
                # Assume BGR from OpenCV preprocessing, convert to RGB
                return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif image.shape[2] == 4:
                # RGBA to RGB
                return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            else:
                return image
        else:
            # Grayscale to RGB
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    def _filter_and_enhance_detections(self, raw_detections: List) -> List:
        """
        Apply sophisticated filtering to improve detection quality
        Similar to PaddleOCR's quality approach
        """
        if not raw_detections:
            return []
        
        enhanced_detections = []
        
        for detection in raw_detections:
            if len(detection) >= 3:
                bbox_points, text, confidence = detection
                
                # Basic text validation
                if not text or not isinstance(text, str):
                    continue
                
                text = text.strip()
                if len(text) < self.min_text_length:
                    continue
                
                # Confidence filtering
                if confidence < self.min_confidence:
                    continue
                
                # Calculate bounding box area for size filtering
                bbox_area = self._calculate_bbox_area(bbox_points)
                if bbox_area < self.min_bbox_area:
                    continue
                
                # Text quality assessment
                if not self._is_valid_text_content(text):
                    continue
                
                # Aspect ratio check (avoid extremely thin/wide boxes)
                if not self._is_reasonable_aspect_ratio(bbox_points):
                    continue
                
                enhanced_detections.append(detection)
        
        self.logger.info(f"Filtered {len(raw_detections)} raw detections to "
                        f"{len(enhanced_detections)} high-quality detections")
        
        return enhanced_detections
    
    def _calculate_bbox_area(self, bbox_points: List[List[float]]) -> float:
        """Calculate area of bounding box from polygon points"""
        if len(bbox_points) < 4:
            return 0.0
        
        # Simple rectangular area approximation
        x_coords = [p[0] for p in bbox_points]
        y_coords = [p[1] for p in bbox_points]
        
        width = max(x_coords) - min(x_coords)
        height = max(y_coords) - min(y_coords)
        
        return width * height
    
    def _is_valid_text_content(self, text: str) -> bool:
        """
        Assess text quality to filter out noise
        Similar validation to PaddleOCR approach
        """
        # Filter out purely numeric noise with very short length
        if len(text) <= 2 and text.isdigit():
            return False
        
        # Filter out single character noise (except common single chars)
        if len(text) == 1:
            valid_single_chars = set('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz$%&@#')
            return text in valid_single_chars
        
        # Filter out purely special character strings
        special_char_ratio = len([c for c in text if not c.isalnum() and not c.isspace()]) / len(text)
        if special_char_ratio > 0.8:
            return False
        
        # Filter out repetitive character patterns that look like noise
        if len(set(text.replace(' ', ''))) == 1 and len(text) > 3:
            return False
        
        return True
    
    def _is_reasonable_aspect_ratio(self, bbox_points: List[List[float]]) -> bool:
        """Check if bounding box has reasonable aspect ratio"""
        if len(bbox_points) < 4:
            return False
        
        x_coords = [p[0] for p in bbox_points]
        y_coords = [p[1] for p in bbox_points]
        
        width = max(x_coords) - min(x_coords)
        height = max(y_coords) - min(y_coords)
        
        if width <= 0 or height <= 0:
            return False
        
        aspect_ratio = width / height
        
        # Allow reasonable aspect ratios (text can be wide but not extremely thin/wide)
        return 0.1 <= aspect_ratio <= 20.0
    
    def _combine_detections_with_advanced_layout(self, detections: List) -> OCRResult:
        """
        Combine detections with sophisticated layout reconstruction
        Matching PaddleOCR's approach to layout preservation
        """
        if not detections:
            return OCRResult(
                text="",
                confidence=0.0,
                engine_used=self.name,
                metadata={"detection_count": 0}
            )
        
        # Convert detections to TextRegions
        regions = []
        for detection in detections:
            bbox_points, text, confidence = detection
            
            # Convert polygon to bounding box
            bbox = self._polygon_to_bbox(bbox_points, confidence)
            
            # Create text region
            region = TextRegion(
                text=text.strip(),
                confidence=confidence,
                bbox=bbox,
                text_type=TextType.MIXED,
                language=self.languages[0] if self.languages else "en"
            )
            
            regions.append(region)
        
        # Apply sophisticated layout reconstruction (matching PaddleOCR)
        formatted_text = self._reconstruct_document_layout_advanced(regions)
        
        # Calculate overall confidence
        overall_confidence = sum(r.confidence for r in regions) / len(regions) if regions else 0.0
        
        # Calculate overall bounding box
        overall_bbox = self._calculate_overall_bbox(regions)
        
        return OCRResult(
            text=formatted_text,
            confidence=overall_confidence,
            regions=regions,
            bbox=overall_bbox,
            engine_used=self.name,
            metadata={
                "detection_count": len(regions),
                "gpu_enabled": self.gpu,
                "languages": self.languages,
                "layout_preserved": True,
                "filtering_applied": True,
                "extraction_method": "easyocr_enhanced"
            }
        )
    
    def _reconstruct_document_layout_advanced(self, regions: List[TextRegion]) -> str:
        """
        Advanced layout reconstruction matching PaddleOCR's sophisticated approach
        Groups text into lines and preserves document structure
        """
        if not regions:
            return ""
        
        # Sort regions by vertical position first, then horizontal
        sorted_regions = sorted(regions, key=lambda r: (
            r.bbox.y + r.bbox.height // 2,  # Use center Y for better grouping
            r.bbox.x
        ))
        
        # Group regions into lines using adaptive threshold
        lines = []
        current_line = []
        
        for i, region in enumerate(sorted_regions):
            region_center_y = region.bbox.y + region.bbox.height // 2
            
            if not current_line:
                current_line = [region]
            else:
                # Check if region should be grouped with current line
                should_group = False
                
                for line_region in current_line:
                    line_center_y = line_region.bbox.y + line_region.bbox.height // 2
                    y_distance = abs(region_center_y - line_center_y)
                    
                    # Adaptive threshold based on text height
                    adaptive_threshold = min(
                        self.line_height_threshold,
                        min(region.bbox.height, line_region.bbox.height) * 0.7
                    )
                    
                    if y_distance <= adaptive_threshold:
                        should_group = True
                        break
                
                if should_group:
                    current_line.append(region)
                else:
                    if current_line:
                        lines.append(current_line)
                    current_line = [region]
            
            # Handle last region
            if i == len(sorted_regions) - 1 and current_line:
                lines.append(current_line)
        
        # Reconstruct text with intelligent spacing
        formatted_lines = []
        
        for line_regions in lines:
            if not line_regions:
                continue
            
            # Sort regions within line by X coordinate
            line_regions.sort(key=lambda r: r.bbox.x)
            
            line_parts = []
            
            for i, region in enumerate(line_regions):
                text = region.text.strip()
                if not text:
                    continue
                
                if i > 0:
                    prev_region = line_regions[i-1]
                    horizontal_gap = region.bbox.x - (prev_region.bbox.x + prev_region.bbox.width)
                    
                    # Intelligent spacing based on gap size
                    if horizontal_gap > self.word_spacing_threshold * 2:
                        spaces = "    "  # Large gap - likely column/section separation
                    elif horizontal_gap > self.word_spacing_threshold:
                        spaces = "  "   # Medium gap - likely word/item separation  
                    elif horizontal_gap > 5:
                        spaces = " "    # Small gap - normal word separation
                    else:
                        # Very small gap - check if space needed
                        spaces = " " if not prev_region.text.strip().endswith(' ') else ""
                    
                    line_parts.append(spaces)
                
                line_parts.append(text)
            
            # Assemble line
            if line_parts:
                line_text = "".join(line_parts).strip()
                if line_text:
                    formatted_lines.append(line_text)
        
        # Handle vertical spacing between text blocks (like PaddleOCR)
        final_text_lines = []
        prev_line_bottom = None
        
        for i, (formatted_line, line_regions) in enumerate(zip(formatted_lines, lines)):
            current_line_top = min(r.bbox.y for r in line_regions)
            
            # Add extra spacing for significant vertical gaps
            if prev_line_bottom is not None:
                vertical_gap = current_line_top - prev_line_bottom
                avg_line_height = sum(r.bbox.height for r in line_regions) / len(line_regions)
                
                # Add blank line for section separation
                if vertical_gap > avg_line_height * 1.5:
                    final_text_lines.append("")
            
            final_text_lines.append(formatted_line)
            prev_line_bottom = max(r.bbox.y + r.bbox.height for r in line_regions)
        
        # Join with newlines and clean up
        result = "\n".join(final_text_lines)
        
        # Clean up excessive consecutive newlines
        result = re.sub(r'\n{3,}', '\n\n', result)
        
        return result.strip()
    
    def _polygon_to_bbox(self, points: List[List[float]], confidence: float) -> BoundingBox:
        """Convert polygon points to BoundingBox object"""
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        return BoundingBox(
            x=int(x_min),
            y=int(y_min),
            width=int(x_max - x_min),
            height=int(y_max - y_min),
            confidence=confidence
        )
    
    def _calculate_overall_bbox(self, regions: List[TextRegion]) -> BoundingBox:
        """Calculate overall bounding box from all regions"""
        if not regions:
            return BoundingBox(0, 0, 100, 30, confidence=0.0)
        
        min_x = min(r.bbox.x for r in regions)
        min_y = min(r.bbox.y for r in regions)
        max_x = max(r.bbox.x + r.bbox.width for r in regions)
        max_y = max(r.bbox.y + r.bbox.height for r in regions)
        
        avg_confidence = sum(r.confidence for r in regions) / len(regions)
        
        return BoundingBox(
            x=min_x,
            y=min_y,
            width=max_x - min_x,
            height=max_y - min_y,
            confidence=avg_confidence
        )
    
    def _update_stats(self, result: OCRResult, processing_time: float):
        """Update engine performance statistics"""
        self.processing_stats['total_processed'] += 1
        self.processing_stats['total_time'] += processing_time
        
        if result.text and result.text.strip() and result.confidence > 0:
            self.processing_stats['successful_extractions'] += 1
    
    def get_engine_info(self) -> Dict[str, Any]:
        """Get comprehensive engine information for EngineManager"""
        return {
            'name': self.name,
            'version': self._get_easyocr_version(),
            'type': 'neural_ocr',
            'available': self.is_available(),
            'supported_languages': self.get_supported_languages(),
            'loaded_languages': self.supported_languages,
            'capabilities': {
                'handwriting_recognition': True,
                'multiple_languages': True,
                'scene_text': True,
                'orientation_detection': True,
                'confidence_scoring': True,
                'layout_preservation': True,
                'quality_filtering': True
            },
            'performance_profile': {
                'accuracy': 'high',
                'speed': 'medium',
                'memory_usage': 'high' if self.gpu else 'medium',
                'gpu_acceleration': self.gpu,
                'optimal_for': [
                    'handwritten_text', 
                    'scene_text', 
                    'multilingual_documents',
                    'mixed_content'
                ]
            },
            'technical_details': {
                'detection_model': 'CRAFT',
                'recognition_model': 'CRNN',
                'framework': 'PyTorch',
                'gpu_enabled': self.gpu,
                'model_storage': self.model_storage_directory,
                'enhanced_filtering': True,
                'layout_reconstruction': 'advanced'
            }
        }
    
    def _get_easyocr_version(self) -> str:
        """Get EasyOCR version if available"""
        try:
            if EASYOCR_AVAILABLE:
                return getattr(easyocr, '__version__', 'unknown')
            return 'not_installed'
        except:
            return 'unknown'
    
    def cleanup(self):
        """Clean up engine resources"""
        if self.reader is not None:
            self.reader = None
        
        self.is_initialized = False
        self.model_loaded = False
        self.supported_languages = []
        
        self.logger.info(f"EasyOCR engine cleaned up")
    
    def __str__(self) -> str:
        return f"EasyOCR(available={self.is_available()}, languages={self.languages})"
    
    def __repr__(self) -> str:
        return self.__str__()