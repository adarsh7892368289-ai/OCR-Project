# src/advanced_ocr/engines/easyocr.py
"""EasyOCR engine implementation for text extraction.

Provides OCR capabilities using EasyOCR with layout-aware text reconstruction.
Excellent for handwriting recognition and scene text extraction.
"""

import cv2
import numpy as np
import time
import re
from typing import List, Dict, Any, Optional, Tuple

from ..core.base_engine import BaseOCREngine
from ..types import OCRResult, TextRegion, BoundingBox, TextType


class EasyOCREngine(BaseOCREngine):
    """EasyOCR engine for text extraction from images.
    
    Features:
    - Neural network-based recognition
    - Handwriting support
    - Multiple language support
    - Layout-aware text reconstruction
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize EasyOCR engine with configuration."""
        super().__init__("EasyOCR", config)
        self.reader = None
        
        # Configuration
        self.languages = self.config.get("languages", ["en"])
        self.gpu = self.config.get("gpu", True)
        self.model_storage_directory = self.config.get("model_dir", None)
        self.download_enabled = self.config.get("download_enabled", True)
        
        # Detection parameters
        self.width_ths = self.config.get("width_ths", 0.5)
        self.height_ths = self.config.get("height_ths", 0.5)
        self.decoder = self.config.get("decoder", "greedy")
        self.beamwidth = self.config.get("beamwidth", 5)
        
        # Layout reconstruction settings
        self.line_height_threshold = self.config.get("line_height_threshold", 15)
        self.word_spacing_threshold = self.config.get("word_spacing_threshold", 20)
        
        # Engine capabilities
        self.supports_handwriting = True
        self.supports_multiple_languages = True
        self.supports_orientation_detection = True
        self.supports_structure_analysis = True
    
    def initialize(self) -> bool:
        """Initialize EasyOCR reader."""
        try:
            import easyocr
            
            self.logger.info(f"Initializing EasyOCR with languages: {self.languages}")
            
            self.reader = easyocr.Reader(
                self.languages,
                gpu=self.gpu,
                model_storage_directory=self.model_storage_directory,
                download_enabled=self.download_enabled,
                verbose=False
            )
            
            self.is_initialized = True
            self.logger.info(f"EasyOCR initialized successfully (GPU: {self.gpu})")
            return True
            
        except ImportError:
            self.logger.error("EasyOCR not installed. Install with: pip install easyocr")
            return False
        except Exception as e:
            self.logger.error(f"EasyOCR initialization failed: {e}")
            return False
    
    def is_available(self) -> bool:
        """Check if EasyOCR is available and initialized."""
        try:
            import easyocr
            return self.is_initialized and self.reader is not None
        except ImportError:
            return False
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        return [
            'en', 'ch_sim', 'ch_tra', 'ja', 'ko', 'th', 'vi', 'ar', 'hi', 'ur',
            'fa', 'ru', 'bg', 'uk', 'be', 'te', 'kn', 'ta', 'bn', 'as', 'mr',
            'ne', 'si', 'my', 'km', 'lo', 'sa', 'fr', 'de', 'es', 'pt', 'it',
            'nl', 'sv', 'da', 'no', 'fi', 'lt', 'lv', 'et', 'pl', 'cs', 'sk',
            'sl', 'hu', 'ro', 'hr', 'sr', 'bs', 'mk', 'sq', 'mt', 'cy', 'ga',
            'tr', 'az', 'uz', 'mn'
        ]
    
    def extract_text(self, image: np.ndarray) -> OCRResult:
        """Extract text from preprocessed image."""
        start_time = time.time()
        
        try:
            # Validate initialization
            if not self.is_initialized or self.reader is None:
                if not self.initialize():
                    raise RuntimeError("EasyOCR engine not initialized")
            
            if not self.validate_image(image):
                raise ValueError("Invalid preprocessed image")
            
            # Prepare image for EasyOCR (BGR to RGB conversion)
            easyocr_image = self._prepare_for_easyocr(image)
            
            # Call EasyOCR
            detections = self.reader.readtext(
                easyocr_image,
                detail=1,
                paragraph=False,
                width_ths=self.width_ths,
                height_ths=self.height_ths,
                decoder=self.decoder,
                beamWidth=self.beamwidth
            )
            
            # Parse and reconstruct layout
            easyocr_detections = self._parse_easyocr_results(detections)
            result = self._combine_ocr_results_with_layout(easyocr_detections)
            
            # Set metadata
            processing_time = time.time() - start_time
            result.processing_time = processing_time
            result.engine_used = self.name
            
            # Update statistics
            self.processing_stats['total_processed'] += 1
            self.processing_stats['total_time'] += processing_time
            if result.text.strip():
                self.processing_stats['successful_extractions'] += 1
                self.logger.info(f"SUCCESS: EasyOCR extracted {len(result.text)} chars "
                               f"(conf: {result.confidence:.3f}) in {processing_time:.2f}s")
            else:
                self.logger.warning(f"NO TEXT: EasyOCR found no text in {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"EasyOCR failed: {e}")
            self.processing_stats['errors'] += 1
            
            return OCRResult(
                text="",
                confidence=0.0,
                processing_time=processing_time,
                engine_used=self.name,
                metadata={"error": str(e)}
            )
    
    def _parse_easyocr_results(self, results) -> List:
        """Parse EasyOCR results into standard detection format.
        
        Returns:
            List of detections: [[bbox_points, text, confidence], ...]
        """
        detections = []
        
        if not results:
            return detections
        
        # EasyOCR format: [bbox_points, text, confidence]
        for detection in results:
            try:
                if len(detection) >= 3:
                    bbox_points, text, confidence = detection
                    
                    if not text or not isinstance(text, str):
                        continue
                    
                    text = text.strip()
                    if not text or confidence <= 0.1:
                        continue
                    
                    # Validate bbox format: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                    if not bbox_points or len(bbox_points) < 4:
                        continue
                    
                    standard_detection = [bbox_points, text, float(confidence)]
                    detections.append(standard_detection)
                    
            except Exception as e:
                self.logger.warning(f"Skipping problematic detection: {e}")
                continue
        
        self.logger.info(f"Parsed {len(detections)} valid detections from EasyOCR")
        return detections
    
    def _prepare_for_easyocr(self, image: np.ndarray) -> np.ndarray:
        """Convert image to RGB format for EasyOCR."""
        if len(image.shape) == 3:
            if image.shape[2] == 3:
                # Convert BGR to RGB
                return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif image.shape[2] == 4:
                # Convert RGBA to RGB
                return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            else:
                return image
        else:
            # Convert grayscale to RGB
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    
    def _combine_ocr_results_with_layout(self, easyocr_detections: List) -> OCRResult:
        """Combine detections into final result with preserved layout."""
        regions = []
        detection_count = 0
        
        if not easyocr_detections:
            return OCRResult(
                text="",
                confidence=0.0,
                regions=[],
                engine_used=self.name,
                metadata={"detection_count": 0}
            )
        
        # Convert detections to TextRegions
        for detection in easyocr_detections:
            try:
                if len(detection) >= 3:
                    bbox_points, text, confidence = detection
                    
                    if not text.strip() or confidence <= 0.1:
                        continue
                    
                    # Convert polygon to bounding box
                    x, y, w, h = self._polygon_to_bbox(bbox_points)
                    bbox = BoundingBox(
                        x=x, y=y, width=w, height=h,
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
                    detection_count += 1
                    
            except Exception as e:
                self.logger.warning(f"Skipping problematic detection: {e}")
                continue
        
        # Reconstruct document layout
        formatted_text = self._reconstruct_document_layout(regions)
        
        # Calculate overall metrics
        overall_confidence = sum(r.confidence for r in regions) / len(regions) if regions else 0.0
        overall_bbox = self._calculate_overall_bbox(regions) if regions else BoundingBox(0, 0, 100, 30)
        
        return OCRResult(
            text=formatted_text,
            confidence=overall_confidence,
            regions=regions,
            bbox=overall_bbox,
            engine_used=self.name,
            metadata={
                'detection_method': 'easyocr',
                'detection_count': detection_count,
                'individual_confidences': [r.confidence for r in regions],
                'layout_preserved': True
            }
        )
    
    def _reconstruct_document_layout(self, regions: List[TextRegion]) -> str:
        """Reconstruct document layout by grouping regions into lines."""
        if not regions:
            return ""
        
        # Sort by vertical position, then horizontal
        sorted_regions = sorted(regions, key=lambda r: (
            r.bbox.y + r.bbox.height // 2,  # Center Y for better grouping
            r.bbox.x
        ))
        
        # Group regions into lines based on vertical proximity
        lines = []
        current_line = []
        
        for i, region in enumerate(sorted_regions):
            region_center_y = region.bbox.y + region.bbox.height // 2
            
            if not current_line:
                current_line = [region]
            else:
                # Check if region belongs to current line
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
            
            # Add last line
            if i == len(sorted_regions) - 1 and current_line:
                lines.append(current_line)
        
        # Assemble text with intelligent spacing
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
                    # Calculate horizontal gap and add appropriate spacing
                    prev_region = line_regions[i-1]
                    horizontal_gap = region.bbox.x - (prev_region.bbox.x + prev_region.bbox.width)
                    
                    if horizontal_gap > self.word_spacing_threshold * 2:
                        spaces = "    "  # Column separation
                    elif horizontal_gap > self.word_spacing_threshold:
                        spaces = "  "   # Item/value separation
                    elif horizontal_gap > 5:
                        spaces = " "    # Word separation
                    else:
                        spaces = " " if not prev_region.text.strip().endswith(' ') else ""
                    
                    line_parts.append(spaces)
                
                line_parts.append(text)
            
            if line_parts:
                line_text = "".join(line_parts).strip()
                if line_text:
                    formatted_lines.append(line_text)
        
        # Add vertical spacing for text blocks
        final_text_lines = []
        prev_line_bottom = None
        
        for i, (formatted_line, line_regions) in enumerate(zip(formatted_lines, lines)):
            current_line_top = min(r.bbox.y for r in line_regions)
            
            # Add blank line for significant vertical gaps
            if prev_line_bottom is not None:
                vertical_gap = current_line_top - prev_line_bottom
                avg_line_height = sum(r.bbox.height for r in line_regions) / len(line_regions)
                
                if vertical_gap > avg_line_height * 1.5:
                    final_text_lines.append("")
            
            final_text_lines.append(formatted_line)
            prev_line_bottom = max(r.bbox.y + r.bbox.height for r in line_regions)
        
        # Join and clean up
        result = "\n".join(final_text_lines)
        
        # Limit consecutive newlines to maximum of 2
        result = re.sub(r'\n{3,}', '\n\n', result)
        
        return result.strip()
    
    def _polygon_to_bbox(self, points: List[List[float]]) -> Tuple[int, int, int, int]:
        """Convert polygon points to bounding box (x, y, width, height)."""
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        return (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))
    
    def _calculate_overall_bbox(self, regions: List[TextRegion]) -> BoundingBox:
        """Calculate overall bounding box encompassing all regions."""
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


# Alias for compatibility
EasyOCR = EasyOCREngine