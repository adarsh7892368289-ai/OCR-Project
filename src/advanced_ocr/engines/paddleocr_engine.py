"""
PaddleOCR Enhanced Engine - Production Grade Implementation

This module provides an enhanced PaddleOCR engine with layout preservation,
optimized performance, and intelligent text region handling.

Key Features:
- Layout-aware text extraction with proper positioning
- Optimized preprocessing for better accuracy
- GPU acceleration with fallback support
- Confidence-based filtering and region merging
- Memory efficient processing
- Comprehensive error handling and logging

Author: Advanced OCR System
Version: 2.0.0
"""

import cv2
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
import time
from pathlib import Path

try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False
    logging.warning("PaddleOCR not available. Install with: pip install paddleocr")

from .base_engine import BaseOCREngine
from ..results import OCRResult, TextRegion
from ..utils.image_utils import ImageUtils
from ..utils.text_utils import TextUtils
from ..config import OCRConfig


@dataclass
class PaddleOCRRegion:
    """
    Represents a text region detected by PaddleOCR with enhanced metadata.
    """
    bbox: List[List[int]]  # Bounding box coordinates [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    text: str              # Extracted text content
    confidence: float      # Detection confidence (0.0 to 1.0)
    line_id: int          # Line identifier for layout reconstruction
    word_spacing: float   # Average character spacing in this region
    font_size: float      # Estimated font size
    is_vertical: bool     # Whether text orientation is vertical
    text_type: str        # 'printed', 'handwritten', or 'mixed'


class PaddleOCREnhanced(BaseOCREngine):
    """
    Enhanced PaddleOCR engine with layout preservation and performance optimizations.
    
    This engine provides:
    - Intelligent text region detection and merging
    - Layout-aware text extraction with proper ordering
    - GPU acceleration with automatic fallback
    - Confidence-based filtering
    - Memory efficient batch processing
    - Comprehensive error handling
    """
    
    def __init__(self, config: OCRConfig):
        """
        Initialize the enhanced PaddleOCR engine.
        
        Args:
            config: OCR configuration object containing engine settings
            
        Raises:
            ImportError: If PaddleOCR is not installed
            RuntimeError: If engine initialization fails
        """
        super().__init__(config)
        
        if not PADDLEOCR_AVAILABLE:
            raise ImportError(
                "PaddleOCR is not installed. Please install it using: "
                "pip install paddleocr"
            )
        
        self.logger = logging.getLogger(__name__)
        self.engine_name = "paddleocr_enhanced"
        
        # Engine configuration
        self._setup_engine_config()
        
        # Initialize PaddleOCR engine
        self._initialize_engine()
        
        # Performance metrics
        self._reset_metrics()
        
        self.logger.info(f"PaddleOCR Enhanced engine initialized successfully")
    
    def _setup_engine_config(self) -> None:
        """Configure engine-specific settings."""
        engine_config = self.config.engines.get('paddleocr', {})
        
        # Core settings
        self.use_gpu = engine_config.get('use_gpu', self.config.gpu_enabled)
        self.use_angle_cls = engine_config.get('use_angle_cls', True)
        self.lang = engine_config.get('language', 'en')
        self.show_log = engine_config.get('show_log', False)
        
        # Performance settings
        self.max_text_length = engine_config.get('max_text_length', 25)
        self.det_db_thresh = engine_config.get('det_db_thresh', 0.3)
        self.det_db_box_thresh = engine_config.get('det_db_box_thresh', 0.6)
        self.rec_batch_num = engine_config.get('rec_batch_num', 6)
        
        # Layout preservation settings
        self.preserve_layout = engine_config.get('preserve_layout', True)
        self.merge_nearby_regions = engine_config.get('merge_nearby_regions', True)
        self.line_height_threshold = engine_config.get('line_height_threshold', 1.5)
        self.word_spacing_threshold = engine_config.get('word_spacing_threshold', 10)
        
        # Quality settings
        self.min_confidence = engine_config.get('min_confidence', 0.5)
        self.min_text_length = engine_config.get('min_text_length', 2)
        self.filter_noise = engine_config.get('filter_noise', True)
    
    def _initialize_engine(self) -> None:
        """Initialize the PaddleOCR engine with optimized settings."""
        try:
            self.paddle_ocr = PaddleOCR(
                use_angle_cls=self.use_angle_cls,
                lang=self.lang,
                use_gpu=self.use_gpu,
                show_log=self.show_log,
                max_text_length=self.max_text_length,
                det_db_thresh=self.det_db_thresh,
                det_db_box_thresh=self.det_db_box_thresh,
                rec_batch_num=self.rec_batch_num,
            )
            
            # Test engine with a small dummy image
            self._test_engine()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize PaddleOCR: {str(e)}")
            if self.use_gpu:
                self.logger.info("Attempting to fall back to CPU mode...")
                try:
                    self.use_gpu = False
                    self.paddle_ocr = PaddleOCR(
                        use_angle_cls=self.use_angle_cls,
                        lang=self.lang,
                        use_gpu=False,
                        show_log=self.show_log,
                        max_text_length=self.max_text_length,
                        det_db_thresh=self.det_db_thresh,
                        det_db_box_thresh=self.det_db_box_thresh,
                        rec_batch_num=self.rec_batch_num,
                    )
                    self.logger.info("Successfully fell back to CPU mode")
                except Exception as cpu_error:
                    raise RuntimeError(f"Failed to initialize PaddleOCR in both GPU and CPU modes: {str(cpu_error)}")
            else:
                raise RuntimeError(f"Failed to initialize PaddleOCR: {str(e)}")
    
    def _test_engine(self) -> None:
        """Test the engine with a dummy image to ensure it's working."""
        try:
            # Create a small test image
            test_image = np.ones((100, 200, 3), dtype=np.uint8) * 255
            cv2.putText(test_image, "TEST", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            
            # Test OCR
            result = self.paddle_ocr.ocr(test_image, cls=self.use_angle_cls)
            
            if result is None or (isinstance(result, list) and len(result) == 0):
                self.logger.warning("PaddleOCR test returned empty result, but engine appears functional")
            else:
                self.logger.debug("PaddleOCR engine test completed successfully")
                
        except Exception as e:
            raise RuntimeError(f"PaddleOCR engine test failed: {str(e)}")
    
    def _reset_metrics(self) -> None:
        """Reset performance metrics."""
        self.metrics = {
            'total_processing_time': 0.0,
            'detection_time': 0.0,
            'recognition_time': 0.0,
            'postprocessing_time': 0.0,
            'images_processed': 0,
            'regions_detected': 0,
            'regions_filtered': 0,
            'average_confidence': 0.0
        }
    
    def process_image(self, image: np.ndarray) -> OCRResult:
        """
        Process an image and extract text with layout preservation.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            OCRResult: Processed result with text and metadata
            
        Raises:
            ValueError: If image is invalid
            RuntimeError: If processing fails
        """
        if image is None or image.size == 0:
            raise ValueError("Input image is None or empty")
        
        start_time = time.time()
        
        try:
            # Validate and preprocess image
            processed_image = self._preprocess_image(image)
            
            # Extract text regions using PaddleOCR
            detection_start = time.time()
            raw_results = self._extract_text_regions(processed_image)
            self.metrics['detection_time'] = time.time() - detection_start
            
            # Process and enhance results
            postprocess_start = time.time()
            enhanced_regions = self._enhance_regions(raw_results, processed_image)
            
            # Filter and merge regions
            filtered_regions = self._filter_regions(enhanced_regions)
            
            if self.merge_nearby_regions:
                merged_regions = self._merge_nearby_text_regions(filtered_regions)
            else:
                merged_regions = filtered_regions
            
            # Reconstruct layout and create final text
            final_text, structured_regions = self._reconstruct_layout(merged_regions)
            
            self.metrics['postprocessing_time'] = time.time() - postprocess_start
            
            # Update metrics
            total_time = time.time() - start_time
            self._update_metrics(total_time, len(merged_regions))
            
            # Create result object
            result = self._create_result(
                text=final_text,
                regions=structured_regions,
                processing_time=total_time,
                image_shape=image.shape
            )
            
            self.logger.debug(
                f"PaddleOCR processed image in {total_time:.3f}s, "
                f"found {len(merged_regions)} regions"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing image with PaddleOCR: {str(e)}")
            # Return empty result on error
            return OCRResult(
                text="",
                confidence=0.0,
                processing_time=time.time() - start_time,
                engine_name=self.engine_name,
                regions=[],
                metadata={
                    'error': str(e),
                    'engine': self.engine_name,
                    'status': 'failed'
                }
            )
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for optimal PaddleOCR performance.
        
        Args:
            image: Input image
            
        Returns:
            Preprocessed image
        """
        try:
            # Convert to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                # Assume BGR and convert to RGB
                processed_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                processed_image = image.copy()
            
            # Resize if image is too large (PaddleOCR works best with moderate sizes)
            height, width = processed_image.shape[:2]
            max_dimension = 2048
            
            if max(height, width) > max_dimension:
                if width > height:
                    new_width = max_dimension
                    new_height = int(height * max_dimension / width)
                else:
                    new_height = max_dimension
                    new_width = int(width * max_dimension / height)
                
                processed_image = cv2.resize(
                    processed_image, 
                    (new_width, new_height), 
                    interpolation=cv2.INTER_LANCZOS4
                )
                
                self.logger.debug(f"Resized image from {width}x{height} to {new_width}x{new_height}")
            
            # Ensure image has proper data type
            if processed_image.dtype != np.uint8:
                processed_image = processed_image.astype(np.uint8)
            
            return processed_image
            
        except Exception as e:
            self.logger.error(f"Error in image preprocessing: {str(e)}")
            return image
    
    def _extract_text_regions(self, image: np.ndarray) -> List[List]:
        """
        Extract text regions using PaddleOCR.
        
        Args:
            image: Preprocessed image
            
        Returns:
            Raw PaddleOCR results
        """
        try:
            # Run PaddleOCR
            results = self.paddle_ocr.ocr(image, cls=self.use_angle_cls)
            
            if results is None:
                return []
            
            # Handle different result formats
            if isinstance(results, list):
                if len(results) == 0:
                    return []
                # If results is a list of lists (multi-page), take first page
                if isinstance(results[0], list):
                    return results[0] if len(results) > 0 else []
                else:
                    return results
            
            return []
            
        except Exception as e:
            self.logger.error(f"Error in text extraction: {str(e)}")
            return []
    
    def _enhance_regions(self, raw_results: List[List], image: np.ndarray) -> List[PaddleOCRRegion]:
        """
        Enhance raw PaddleOCR results with additional metadata.
        
        Args:
            raw_results: Raw PaddleOCR detection results
            image: Source image
            
        Returns:
            List of enhanced region objects
        """
        enhanced_regions = []
        
        for idx, item in enumerate(raw_results):
            try:
                if len(item) < 2:
                    continue
                
                bbox, (text, confidence) = item[0], item[1]
                
                if not text or not text.strip():
                    continue
                
                # Create enhanced region
                region = PaddleOCRRegion(
                    bbox=bbox,
                    text=text.strip(),
                    confidence=float(confidence),
                    line_id=idx,
                    word_spacing=self._calculate_word_spacing(bbox, text),
                    font_size=self._estimate_font_size(bbox),
                    is_vertical=self._is_vertical_text(bbox),
                    text_type=self._classify_text_type(text)
                )
                
                enhanced_regions.append(region)
                
            except Exception as e:
                self.logger.warning(f"Error enhancing region {idx}: {str(e)}")
                continue
        
        return enhanced_regions
    
    def _calculate_word_spacing(self, bbox: List[List[int]], text: str) -> float:
        """Calculate average character spacing in a text region."""
        try:
            if len(text) <= 1:
                return 0.0
            
            # Calculate region width
            x_coords = [point[0] for point in bbox]
            width = max(x_coords) - min(x_coords)
            
            # Estimate character spacing
            char_spacing = width / len(text) if len(text) > 0 else 0.0
            return float(char_spacing)
            
        except Exception:
            return 0.0
    
    def _estimate_font_size(self, bbox: List[List[int]]) -> float:
        """Estimate font size from bounding box."""
        try:
            # Calculate region height
            y_coords = [point[1] for point in bbox]
            height = max(y_coords) - min(y_coords)
            
            # Font size is approximately the height of the bounding box
            return float(height)
            
        except Exception:
            return 12.0  # Default font size
    
    def _is_vertical_text(self, bbox: List[List[int]]) -> bool:
        """Determine if text orientation is vertical."""
        try:
            # Calculate width and height
            x_coords = [point[0] for point in bbox]
            y_coords = [point[1] for point in bbox]
            
            width = max(x_coords) - min(x_coords)
            height = max(y_coords) - min(y_coords)
            
            # Consider vertical if height is significantly greater than width
            return height > width * 1.5
            
        except Exception:
            return False
    
    def _classify_text_type(self, text: str) -> str:
        """Classify text as printed, handwritten, or mixed."""
        # This is a simple heuristic - in production, you might use ML models
        if TextUtils.contains_special_characters(text):
            return 'mixed'
        elif TextUtils.has_consistent_spacing(text):
            return 'printed'
        else:
            return 'handwritten'
    
    def _filter_regions(self, regions: List[PaddleOCRRegion]) -> List[PaddleOCRRegion]:
        """
        Filter regions based on confidence and quality criteria.
        
        Args:
            regions: List of enhanced regions
            
        Returns:
            Filtered regions
        """
        filtered = []
        
        for region in regions:
            try:
                # Filter by confidence
                if region.confidence < self.min_confidence:
                    self.metrics['regions_filtered'] += 1
                    continue
                
                # Filter by text length
                if len(region.text) < self.min_text_length:
                    self.metrics['regions_filtered'] += 1
                    continue
                
                # Filter noise if enabled
                if self.filter_noise and self._is_noise_region(region):
                    self.metrics['regions_filtered'] += 1
                    continue
                
                filtered.append(region)
                
            except Exception as e:
                self.logger.warning(f"Error filtering region: {str(e)}")
                continue
        
        return filtered
    
    def _is_noise_region(self, region: PaddleOCRRegion) -> bool:
        """Determine if a region contains noise/artifacts."""
        text = region.text.strip()
        
        # Check for common noise patterns
        if len(text) == 1 and text in '.,;:|()[]{}':
            return True
        
        # Check for excessive punctuation
        punct_ratio = sum(1 for c in text if c in '.,;:|()[]{}!?@#$%^&*') / len(text)
        if punct_ratio > 0.7:
            return True
        
        # Check for very low confidence with short text
        if region.confidence < 0.3 and len(text) < 5:
            return True
        
        return False
    
    def _merge_nearby_text_regions(self, regions: List[PaddleOCRRegion]) -> List[PaddleOCRRegion]:
        """
        Merge nearby text regions that likely belong to the same line.
        
        Args:
            regions: List of text regions
            
        Returns:
            List of merged regions
        """
        if not regions:
            return regions
        
        try:
            # Sort regions by vertical position, then horizontal
            sorted_regions = sorted(regions, key=lambda r: (
                min(point[1] for point in r.bbox),  # Top y-coordinate
                min(point[0] for point in r.bbox)   # Left x-coordinate
            ))
            
            merged = []
            current_group = [sorted_regions[0]]
            
            for region in sorted_regions[1:]:
                # Check if region should be merged with current group
                if self._should_merge_regions(current_group[-1], region):
                    current_group.append(region)
                else:
                    # Finalize current group and start new one
                    if len(current_group) > 1:
                        merged_region = self._merge_region_group(current_group)
                        merged.append(merged_region)
                    else:
                        merged.append(current_group[0])
                    current_group = [region]
            
            # Handle last group
            if len(current_group) > 1:
                merged_region = self._merge_region_group(current_group)
                merged.append(merged_region)
            else:
                merged.append(current_group[0])
            
            return merged
            
        except Exception as e:
            self.logger.error(f"Error merging regions: {str(e)}")
            return regions
    
    def _should_merge_regions(self, region1: PaddleOCRRegion, region2: PaddleOCRRegion) -> bool:
        """Determine if two regions should be merged."""
        try:
            # Get bounding boxes
            bbox1, bbox2 = region1.bbox, region2.bbox
            
            # Calculate centers and dimensions
            center1_y = sum(point[1] for point in bbox1) / 4
            center2_y = sum(point[1] for point in bbox2) / 4
            
            height1 = max(point[1] for point in bbox1) - min(point[1] for point in bbox1)
            height2 = max(point[1] for point in bbox2) - min(point[1] for point in bbox2)
            
            avg_height = (height1 + height2) / 2
            
            # Check vertical alignment (same line)
            vertical_distance = abs(center1_y - center2_y)
            if vertical_distance > avg_height * self.line_height_threshold:
                return False
            
            # Check horizontal proximity
            right1 = max(point[0] for point in bbox1)
            left2 = min(point[0] for point in bbox2)
            
            horizontal_gap = left2 - right1
            if horizontal_gap > self.word_spacing_threshold:
                return False
            
            # Check if gap is reasonable for word spacing
            avg_char_spacing = (region1.word_spacing + region2.word_spacing) / 2
            if horizontal_gap > avg_char_spacing * 3:
                return False
            
            return True
            
        except Exception:
            return False
    
    def _merge_region_group(self, regions: List[PaddleOCRRegion]) -> PaddleOCRRegion:
        """Merge a group of regions into a single region."""
        try:
            # Combine texts with appropriate spacing
            texts = []
            for i, region in enumerate(regions):
                if i > 0:
                    # Add space between words
                    prev_region = regions[i-1]
                    right_prev = max(point[0] for point in prev_region.bbox)
                    left_curr = min(point[0] for point in region.bbox)
                    gap = left_curr - right_prev
                    
                    # Determine spacing based on gap size
                    if gap > prev_region.word_spacing * 2:
                        texts.append('  ')  # Double space for larger gaps
                    else:
                        texts.append(' ')   # Single space for normal gaps
                
                texts.append(region.text)
            
            combined_text = ''.join(texts)
            
            # Calculate combined bounding box
            all_points = []
            for region in regions:
                all_points.extend(region.bbox)
            
            min_x = min(point[0] for point in all_points)
            min_y = min(point[1] for point in all_points)
            max_x = max(point[0] for point in all_points)
            max_y = max(point[1] for point in all_points)
            
            combined_bbox = [[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]]
            
            # Calculate average confidence weighted by text length
            total_chars = sum(len(r.text) for r in regions)
            weighted_confidence = sum(r.confidence * len(r.text) for r in regions) / total_chars
            
            # Create merged region
            merged_region = PaddleOCRRegion(
                bbox=combined_bbox,
                text=combined_text,
                confidence=weighted_confidence,
                line_id=regions[0].line_id,
                word_spacing=sum(r.word_spacing for r in regions) / len(regions),
                font_size=sum(r.font_size for r in regions) / len(regions),
                is_vertical=any(r.is_vertical for r in regions),
                text_type=regions[0].text_type  # Use first region's classification
            )
            
            return merged_region
            
        except Exception as e:
            self.logger.error(f"Error merging region group: {str(e)}")
            return regions[0]  # Return first region as fallback
    
    def _reconstruct_layout(self, regions: List[PaddleOCRRegion]) -> Tuple[str, List[TextRegion]]:
        """
        Reconstruct document layout from text regions.
        
        Args:
            regions: List of processed regions
            
        Returns:
            Tuple of (final_text, structured_regions)
        """
        if not regions:
            return "", []
        
        try:
            # Sort regions for reading order (top-to-bottom, left-to-right)
            if self.preserve_layout:
                sorted_regions = self._sort_regions_by_reading_order(regions)
            else:
                sorted_regions = regions
            
            # Build final text and region list
            text_parts = []
            structured_regions = []
            
            for region in sorted_regions:
                text_parts.append(region.text)
                
                # Create TextRegion object
                text_region = TextRegion(
                    bbox=region.bbox,
                    text=region.text,
                    confidence=region.confidence,
                    metadata={
                        'engine': self.engine_name,
                        'line_id': region.line_id,
                        'word_spacing': region.word_spacing,
                        'font_size': region.font_size,
                        'is_vertical': region.is_vertical,
                        'text_type': region.text_type
                    }
                )
                structured_regions.append(text_region)
            
            # Join text with appropriate separators
            if self.preserve_layout:
                final_text = self._join_with_layout_preservation(text_parts, sorted_regions)
            else:
                final_text = ' '.join(text_parts)
            
            return final_text, structured_regions
            
        except Exception as e:
            self.logger.error(f"Error reconstructing layout: {str(e)}")
            # Fallback: simple concatenation
            simple_text = ' '.join(r.text for r in regions)
            simple_regions = [
                TextRegion(
                    bbox=r.bbox,
                    text=r.text,
                    confidence=r.confidence,
                    metadata={'engine': self.engine_name}
                ) for r in regions
            ]
            return simple_text, simple_regions
    
    def _sort_regions_by_reading_order(self, regions: List[PaddleOCRRegion]) -> List[PaddleOCRRegion]:
        """Sort regions in natural reading order."""
        try:
            # Group regions into lines
            lines = self._group_regions_into_lines(regions)
            
            # Sort lines vertically
            sorted_lines = sorted(lines, key=lambda line: min(
                min(point[1] for point in region.bbox) for region in line
            ))
            
            # Sort regions within each line horizontally
            sorted_regions = []
            for line in sorted_lines:
                sorted_line = sorted(line, key=lambda r: min(point[0] for point in r.bbox))
                sorted_regions.extend(sorted_line)
            
            return sorted_regions
            
        except Exception as e:
            self.logger.error(f"Error sorting regions: {str(e)}")
            return regions
    
    def _group_regions_into_lines(self, regions: List[PaddleOCRRegion]) -> List[List[PaddleOCRRegion]]:
        """Group regions into text lines."""
        if not regions:
            return []
        
        lines = []
        remaining_regions = regions.copy()
        
        while remaining_regions:
            # Start new line with first remaining region
            current_line = [remaining_regions.pop(0)]
            
            # Find regions that belong to the same line
            i = 0
            while i < len(remaining_regions):
                region = remaining_regions[i]
                
                # Check if region belongs to current line
                if self._regions_on_same_line(current_line[0], region):
                    current_line.append(remaining_regions.pop(i))
                else:
                    i += 1
            
            lines.append(current_line)
        
        return lines
    
    def _regions_on_same_line(self, region1: PaddleOCRRegion, region2: PaddleOCRRegion) -> bool:
        """Check if two regions are on the same text line."""
        try:
            # Calculate vertical centers
            center1_y = sum(point[1] for point in region1.bbox) / 4
            center2_y = sum(point[1] for point in region2.bbox) / 4
            
            # Calculate average height
            height1 = max(point[1] for point in region1.bbox) - min(point[1] for point in region1.bbox)
            height2 = max(point[1] for point in region2.bbox) - min(point[1] for point in region2.bbox)
            avg_height = (height1 + height2) / 2
            
            # Check if centers are within reasonable vertical distance
            vertical_distance = abs(center1_y - center2_y)
            return vertical_distance < avg_height * 0.7
            
        except Exception:
            return False
    
    def _join_with_layout_preservation(self, text_parts: List[str], regions: List[PaddleOCRRegion]) -> str:
        """Join text parts while preserving document layout."""
        if not text_parts:
            return ""
        
        if len(text_parts) == 1:
            return text_parts[0]
        
        try:
            result_parts = [text_parts[0]]
            
            for i in range(1, len(text_parts)):
                current_region = regions[i]
                prev_region = regions[i-1]
                
                # Determine separator based on layout
                separator = self._determine_separator(prev_region, current_region)
                result_parts.append(separator)
                result_parts.append(text_parts[i])
            
            return ''.join(result_parts)
            
        except Exception as e:
            self.logger.error(f"Error in layout preservation: {str(e)}")
            # Fallback to simple space separation
            return ' '.join(text_parts)
    
    def _determine_separator(self, prev_region: PaddleOCRRegion, current_region: PaddleOCRRegion) -> str:
        """Determine appropriate separator between two text regions."""
        try:
            # Calculate positions
            prev_bottom = max(point[1] for point in prev_region.bbox)
            current_top = min(point[1] for point in current_region.bbox)
            
            prev_right = max(point[0] for point in prev_region.bbox)
            current_left = min(point[0] for point in current_region.bbox)
            
            # Calculate vertical and horizontal gaps
            vertical_gap = current_top - prev_bottom
            horizontal_gap = current_left - prev_right
            
            # Determine if regions are on different lines
            avg_height = (prev_region.font_size + current_region.font_size) / 2
            
            if vertical_gap > avg_height * 0.5:  # Different lines
                # Check for paragraph breaks (larger vertical gaps)
                if vertical_gap > avg_height * 2:
                    return '\n\n'  # Paragraph break
                else:
                    return '\n'    # Line break
            else:  # Same line
                # Determine word spacing
                avg_char_spacing = (prev_region.word_spacing + current_region.word_spacing) / 2
                
                if horizontal_gap > avg_char_spacing * 3:
                    return '  '    # Double space for larger gaps
                elif horizontal_gap > 0:
                    return ' '     # Normal word space
                else:
                    return ''      # Adjacent text (no space)
            
        except Exception:
            return ' '  # Default separator
    
    def _create_result(self, text: str, regions: List[TextRegion], 
                      processing_time: float, image_shape: Tuple[int, ...]) -> OCRResult:
        """Create final OCRResult object."""
        try:
            # Calculate overall confidence
            if regions:
                overall_confidence = sum(r.confidence for r in regions) / len(regions)
            else:
                overall_confidence = 0.0
            
            # Create metadata
            metadata = {
                'engine': self.engine_name,
                'version': '2.0.0',
                'image_shape': image_shape,
                'num_regions': len(regions),
                'processing_time': processing_time,
                'gpu_used': self.use_gpu,
                'language': self.lang,
                'settings': {
                    'preserve_layout': self.preserve_layout,
                    'merge_nearby_regions': self.merge_nearby_regions,
                    'min_confidence': self.min_confidence,
                    'use_angle_cls': self.use_angle_cls
                },
                'performance_metrics': self.metrics.copy()
            }
            
            # Add text statistics
            if text:
                metadata['text_stats'] = {
                    'length': len(text),
                    'words': len(text.split()),
                    'lines': text.count('\n') + 1,
                    'avg_word_length': sum(len(word) for word in text.split()) / len(text.split()) if text.split() else 0
                }
            
            return OCRResult(
                text=text,
                confidence=overall_confidence,
                processing_time=processing_time,
                engine_name=self.engine_name,
                regions=regions,
                metadata=metadata
            )
            
        except Exception as e:
            self.logger.error(f"Error creating result: {str(e)}")
            return OCRResult(
                text=text or "",
                confidence=0.0,
                processing_time=processing_time,
                engine_name=self.engine_name,
                regions=regions or [],
                metadata={'error': str(e), 'engine': self.engine_name}
            )
    
    def _update_metrics(self, processing_time: float, num_regions: int) -> None:
        """Update performance metrics."""
        try:
            self.metrics['images_processed'] += 1
            self.metrics['total_processing_time'] += processing_time
            self.metrics['regions_detected'] += num_regions
            
            # Calculate averages
            if self.metrics['images_processed'] > 0:
                self.metrics['avg_processing_time'] = (
                    self.metrics['total_processing_time'] / self.metrics['images_processed']
                )
                self.metrics['avg_regions_per_image'] = (
                    self.metrics['regions_detected'] / self.metrics['images_processed']
                )
        
        except Exception as e:
            self.logger.warning(f"Error updating metrics: {str(e)}")
    
    def batch_process(self, images: List[np.ndarray]) -> List[OCRResult]:
        """
        Process multiple images in batch for improved efficiency.
        
        Args:
            images: List of input images
            
        Returns:
            List of OCR results
        """
        if not images:
            return []
        
        self.logger.info(f"Starting batch processing of {len(images)} images")
        
        results = []
        batch_start_time = time.time()
        
        try:
            for i, image in enumerate(images):
                try:
                    result = self.process_image(image)
                    results.append(result)
                    
                    if (i + 1) % 10 == 0:  # Log progress every 10 images
                        self.logger.info(f"Processed {i + 1}/{len(images)} images")
                        
                except Exception as e:
                    self.logger.error(f"Error processing image {i}: {str(e)}")
                    # Add error result
                    error_result = OCRResult(
                        text="",
                        confidence=0.0,
                        processing_time=0.0,
                        engine_name=self.engine_name,
                        regions=[],
                        metadata={'error': str(e), 'image_index': i}
                    )
                    results.append(error_result)
            
            total_batch_time = time.time() - batch_start_time
            self.logger.info(
                f"Batch processing completed in {total_batch_time:.2f}s. "
                f"Average: {total_batch_time/len(images):.3f}s per image"
            )
            
        except Exception as e:
            self.logger.error(f"Fatal error in batch processing: {str(e)}")
        
        return results
    
    def get_engine_info(self) -> Dict[str, Any]:
        """
        Get comprehensive engine information and capabilities.
        
        Returns:
            Dictionary containing engine information
        """
        return {
            'name': self.engine_name,
            'version': '2.0.0',
            'type': 'deep_learning',
            'backend': 'paddleocr',
            'capabilities': {
                'languages': ['en', 'ch', 'ta', 'te', 'ka', 'ja', 'ko', 'hi', 'ar', 'fr', 'de', 'es', 'pt', 'ru'],
                'text_types': ['printed', 'handwritten', 'mixed'],
                'orientations': ['horizontal', 'vertical', 'rotated'],
                'features': [
                    'layout_preservation',
                    'region_merging', 
                    'confidence_filtering',
                    'gpu_acceleration',
                    'batch_processing',
                    'angle_classification'
                ]
            },
            'settings': {
                'gpu_enabled': self.use_gpu,
                'language': self.lang,
                'use_angle_cls': self.use_angle_cls,
                'preserve_layout': self.preserve_layout,
                'merge_nearby_regions': self.merge_nearby_regions,
                'min_confidence': self.min_confidence,
                'det_db_thresh': self.det_db_thresh,
                'det_db_box_thresh': self.det_db_box_thresh
            },
            'performance_metrics': self.metrics.copy(),
            'memory_efficient': True,
            'thread_safe': False,  # PaddleOCR is not thread-safe
            'recommended_use_cases': [
                'Document digitization',
                'Form processing', 
                'Invoice extraction',
                'Book/magazine scanning',
                'Handwritten notes',
                'Multi-language documents'
            ]
        }
    
    def supports_language(self, language_code: str) -> bool:
        """
        Check if the engine supports a specific language.
        
        Args:
            language_code: Language code (e.g., 'en', 'ch', 'fr')
            
        Returns:
            True if language is supported
        """
        supported_languages = [
            'en', 'ch', 'ta', 'te', 'ka', 'ja', 'ko', 'hi', 'ar', 
            'fr', 'de', 'es', 'pt', 'ru', 'oc', 'rs', 'bg', 'uk', 'be', 'ur', 'fa'
        ]
        return language_code.lower() in supported_languages
    
    def cleanup(self) -> None:
        """Clean up resources and free memory."""
        try:
            if hasattr(self, 'paddle_ocr'):
                # PaddleOCR doesn't have explicit cleanup, but we can clear the reference
                del self.paddle_ocr
                
            self.logger.info("PaddleOCR Enhanced engine cleaned up successfully")
            
        except Exception as e:
            self.logger.warning(f"Error during cleanup: {str(e)}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup()
    
    def __del__(self):
        """Destructor with cleanup."""
        try:
            self.cleanup()
        except Exception:
            pass  # Ignore errors during destruction


# Utility functions for standalone usage
def create_paddleocr_engine(config: Optional[OCRConfig] = None) -> PaddleOCREnhanced:
    """
    Factory function to create a PaddleOCR Enhanced engine.
    
    Args:
        config: Optional OCR configuration. If None, uses default config.
        
    Returns:
        Configured PaddleOCR Enhanced engine
        
    Example:
        >>> engine = create_paddleocr_engine()
        >>> result = engine.process_image(image)
        >>> print(result.text)
    """
    if config is None:
        # Create default configuration optimized for PaddleOCR
        config = OCRConfig()
        config.engines = {
            'paddleocr': {
                'use_gpu': True,
                'use_angle_cls': True,
                'language': 'en',
                'preserve_layout': True,
                'merge_nearby_regions': True,
                'min_confidence': 0.5,
                'det_db_thresh': 0.3,
                'det_db_box_thresh': 0.6
            }
        }
    
    return PaddleOCREnhanced(config)


def quick_extract_text(image: np.ndarray, language: str = 'en', 
                      preserve_layout: bool = True) -> str:
    """
    Quick text extraction using PaddleOCR with minimal configuration.
    
    Args:
        image: Input image as numpy array
        language: Language code for OCR
        preserve_layout: Whether to preserve document layout
        
    Returns:
        Extracted text string
        
    Example:
        >>> text = quick_extract_text(image, language='en')
        >>> print(text)
    """
    config = OCRConfig()
    config.engines = {
        'paddleocr': {
            'language': language,
            'preserve_layout': preserve_layout,
            'use_gpu': True,
            'min_confidence': 0.5
        }
    }
    
    with PaddleOCREnhanced(config) as engine:
        result = engine.process_image(image)
        return result.text


# Module-level constants
__version__ = "2.0.0"
__author__ = "Advanced OCR System"
__engine_name__ = "paddleocr_enhanced"

# Export main classes and functions
__all__ = [
    'PaddleOCREnhanced',
    'PaddleOCRRegion', 
    'create_paddleocr_engine',
    'quick_extract_text'
]