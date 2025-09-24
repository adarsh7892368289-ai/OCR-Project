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

class PaddleOCR(BaseOCREngine):
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
        """Simple PaddleOCR initialization"""
        try:
            from paddleocr import PaddleOCR
            
            # SIMPLE: Most basic initialization that usually works
            self.ocr = PaddleOCR(
                use_angle_cls=True,
                lang='en',
            )
            
            self.is_initialized = True
            self.model_loaded = True
            self.logger.info("PaddleOCR initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"PaddleOCR initialization failed: {e}")
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
        FIXED: Process preprocessed image with PaddleOCR - Now uses proper layout reconstruction
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
            
            # Convert image for PaddleOCR
            paddle_image = self._prepare_for_paddleocr(preprocessed_image)
            
            # Call PaddleOCR
            try:
                results = self.ocr.ocr(paddle_image, cls=True)
            except Exception as e1:
                try:
                    results = self.ocr.ocr(paddle_image)
                except Exception as e2:
                    self.logger.error(f"Both PaddleOCR calls failed: {e1}, {e2}")
                    raise e2
            
            # FIXED: Parse results and use the same layout reconstruction as EasyOCR
            paddle_detections = self._parse_paddleocr_results(results)
            
            # Use the SAME layout reconstruction method as EasyOCR
            result = self._combine_ocr_results_with_layout(paddle_detections)
            
            # Set processing time
            processing_time = time.time() - start_time
            result.processing_time = processing_time
            result.engine_name = self.name
            
            # Update stats
            self.processing_stats['total_processed'] += 1
            self.processing_stats['total_time'] += processing_time
            if result.text.strip():
                self.processing_stats['successful_extractions'] += 1
                self.logger.info(f"SUCCESS: PaddleOCR extracted {len(result.text)} chars (conf: {result.confidence:.3f}) in {processing_time:.2f}s")
            else:
                self.logger.warning(f"NO TEXT: PaddleOCR found no text in {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"PaddleOCR failed: {e}")
            self.processing_stats['errors'] += 1
            
            return OCRResult(
                text="",
                confidence=0.0,
                processing_time=processing_time,
                engine_name=self.name,
                metadata={"error": str(e)}
            )

    def _parse_paddleocr_results(self, results) -> List:
        """
        FIXED: Parse PaddleOCR results into standard detection format
        Returns format compatible with layout reconstruction: [bbox_points, text, confidence]
        """
        detections = []
        
        if not results or not results[0]:
            return detections
        
        # Handle different PaddleOCR result formats
        page_results = results[0]
        
        if isinstance(page_results, dict):
            # New structured format
            rec_texts = page_results.get('rec_texts', [])
            rec_scores = page_results.get('rec_scores', [])
            rec_polys = page_results.get('rec_polys', [])
            
            for i, text in enumerate(rec_texts):
                if text and text.strip():
                    confidence = float(rec_scores[i]) if i < len(rec_scores) else 0.5
                    
                    if confidence > 0.1:  # Filter low confidence
                        if i < len(rec_polys):
                            bbox_points = rec_polys[i]
                        else:
                            # Create dummy bbox if missing
                            bbox_points = [[0, 0], [100, 0], [100, 30], [0, 30]]
                        
                        # Convert to standard detection format: [bbox_points, text, confidence]
                        detection = [bbox_points, text.strip(), confidence]
                        detections.append(detection)
        
        elif isinstance(page_results, list):
            # Standard format: list of detections
            for detection in page_results:
                try:
                    if isinstance(detection, (list, tuple)) and len(detection) >= 2:
                        bbox_points = detection[0]
                        text_info = detection[1]
                        
                        # Extract text and confidence
                        if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                            text = str(text_info[0]) if text_info[0] else ""
                            confidence = float(text_info[1]) if text_info[1] else 0.0
                        else:
                            text = str(text_info) if text_info else ""
                            confidence = 0.5
                        
                        if text.strip() and confidence > 0.1:
                            # Convert to standard format: [bbox_points, text, confidence]
                            standard_detection = [bbox_points, text.strip(), confidence]
                            detections.append(standard_detection)
                            
                except Exception as e:
                    self.logger.warning(f"Skipping problematic detection: {detection}, error: {e}")
                    continue
        
        self.logger.info(f"Parsed {len(detections)} valid detections from PaddleOCR")
        return detections

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
    
    def _combine_ocr_results_with_layout(self, paddle_detections: List) -> OCRResult:
        """
        FIXED: Use the SAME layout reconstruction logic as EasyOCR
        """
        regions = []
        detection_count = 0
        
        # Handle empty results
        if not paddle_detections:
            return OCRResult(
                text="",
                confidence=0.0,
                regions=[],
                engine_name=self.name,
                metadata={"detection_count": 0}
            )
        
        # Convert PaddleOCR detections to TextRegions (same as EasyOCR)
        for detection in paddle_detections:
            try:
                if len(detection) >= 3:
                    bbox_points, text, confidence = detection
                    
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
                    
            except Exception as e:
                self.logger.warning(f"Skipping problematic detection: {detection}, error: {e}")
                continue
        
        # Use the SAME layout reconstruction as EasyOCR
        formatted_text = self._reconstruct_document_layout(regions)
        
        # Calculate overall confidence
        overall_confidence = sum(r.confidence for r in regions) / len(regions) if regions else 0.0
        
        # Create overall bounding box from all regions
        overall_bbox = self._calculate_overall_bbox(regions) if regions else BoundingBox(0, 0, 100, 30)
        
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
        IMPROVED: Enhanced layout reconstruction for complex documents like receipts
        Handles multi-column layouts, preserves vertical spacing, and maintains text order
        """
        if not regions:
            return ""
        
        # Sort regions primarily by vertical position, with more refined logic
        sorted_regions = sorted(regions, key=lambda r: (
            r.bbox.y + r.bbox.height // 2,  # Use center Y for better line grouping
            r.bbox.x  # Then by X position
        ))
        
        # IMPROVED: More sophisticated line grouping
        lines = []
        current_line = []
        
        for i, region in enumerate(sorted_regions):
            region_center_y = region.bbox.y + region.bbox.height // 2
            
            if not current_line:
                # First region
                current_line = [region]
            else:
                # Check if this region should be on the same line as previous regions
                should_group = False
                
                # Compare with all regions in current line, not just the first one
                for line_region in current_line:
                    line_center_y = line_region.bbox.y + line_region.bbox.height // 2
                    y_distance = abs(region_center_y - line_center_y)
                    
                    # More conservative line height threshold for receipts
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
                    # Finalize current line and start new one
                    if current_line:
                        lines.append(current_line)
                    current_line = [region]
            
            # Handle the last region
            if i == len(sorted_regions) - 1 and current_line:
                lines.append(current_line)
        
        # IMPROVED: Better text assembly with column awareness
        formatted_lines = []
        
        for line_regions in lines:
            if not line_regions:
                continue
                
            # Sort regions in line by X coordinate
            line_regions.sort(key=lambda r: r.bbox.x)
            
            # IMPROVED: Detect column breaks and handle spacing more intelligently
            line_parts = []
            
            for i, region in enumerate(line_regions):
                text = region.text.strip()
                if not text:
                    continue
                    
                if i > 0:
                    prev_region = line_regions[i-1]
                    horizontal_gap = region.bbox.x - (prev_region.bbox.x + prev_region.bbox.width)
                    
                    # More nuanced spacing logic for receipts
                    if horizontal_gap > self.word_spacing_threshold * 2:
                        # Large gap - likely different columns, use multiple spaces
                        spaces = "    "  # 4 spaces for column separation
                    elif horizontal_gap > self.word_spacing_threshold:
                        # Medium gap - separate items/prices
                        spaces = "  "  # 2 spaces
                    elif horizontal_gap > 5:
                        # Small gap - word separation
                        spaces = " "
                    else:
                        # Very small or no gap - might be continuous text
                        spaces = " " if not prev_region.text.strip().endswith(' ') else ""
                    
                    line_parts.append(spaces)
                
                line_parts.append(text)
            
            # Assemble the line
            if line_parts:
                line_text = "".join(line_parts).strip()
                if line_text:
                    formatted_lines.append(line_text)
        
        # IMPROVED: Handle vertical spacing between text blocks
        final_text_lines = []
        prev_line_bottom = None
        
        for i, (formatted_line, line_regions) in enumerate(zip(formatted_lines, lines)):
            current_line_top = min(r.bbox.y for r in line_regions)
            
            # Add extra line break for significant vertical gaps (new sections)
            if prev_line_bottom is not None:
                vertical_gap = current_line_top - prev_line_bottom
                avg_line_height = sum(r.bbox.height for r in line_regions) / len(line_regions)
                
                # If gap is significantly larger than typical line height, add extra spacing
                if vertical_gap > avg_line_height * 1.5:
                    final_text_lines.append("")  # Add blank line for section separation
            
            final_text_lines.append(formatted_line)
            prev_line_bottom = max(r.bbox.y + r.bbox.height for r in line_regions)
        
        # Join with newlines, removing any excessive blank lines
        result = "\n".join(final_text_lines)
        
        # Clean up multiple consecutive newlines (max 2 in a row)
        import re
        result = re.sub(r'\n{3,}', '\n\n', result)
        
        return result.strip()
    
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
                'layout_preservation': True
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
                'preserves_document_layout': True
            }
        }
        
        try:
            # Try to get PaddleOCR version if available
            import paddle
            info['version'] = getattr(paddle, '__version__', 'unknown')
        except:
            info['version'] = 'unknown'
            
        return info