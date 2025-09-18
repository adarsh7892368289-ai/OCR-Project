"""
PaddleOCR Engine Implementation
High-performance OCR using PaddleOCR with optimal settings for modern AI workflow.
"""

import time
import logging
from typing import Any, Optional, List, Tuple, Dict
import numpy as np
from PIL import Image

from .base_engine import BaseOCREngine, EngineStatus
from ..results import (
    OCRResult, Page, Paragraph, Line, Word, 
    BoundingBox, ConfidenceMetrics, CoordinateFormat, ProcessingMetrics
)
from ..config import OCRConfig
from ..utils.model_utils import get_model_loader
from ..utils.image_utils import ImageLoader


class PaddleOCREngine(BaseOCREngine):
    """
    PaddleOCR implementation with full image processing and hierarchical text extraction.
    Optimized for printed text with strong performance on diverse document types.
    """
    
    def __init__(self, config: OCRConfig):
        """
        Initialize PaddleOCR engine.
        
        Args:
            config: OCR configuration
        """
        super().__init__("paddleocr", config)
        self.paddleocr = None
        self.image_loader = ImageLoader()
        
        # PaddleOCR specific settings
        self.use_angle_cls = True
        self.use_gpu = self._detect_gpu()
        self.lang = 'en'  # Default to English
        
        self.logger.info(f"Initialized PaddleOCR engine (GPU: {self.use_gpu})")
    
    def _detect_gpu(self) -> bool:
        """Detect if GPU acceleration is available."""
        try:
            import paddle
            return paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0
        except ImportError:
            try:
                import torch
                return torch.cuda.is_available()
            except ImportError:
                return False
    
    def initialize(self) -> bool:
        """
        Initialize PaddleOCR model with optimized settings.
        
        Returns:
            True if initialization successful
        """
        if self.status == EngineStatus.READY:
            return True
        
        self.status = EngineStatus.INITIALIZING
        start_time = time.time()
        
        try:
            # Get model loader
            model_loader = get_model_loader(self.config)
            
            # Load PaddleOCR model
            self.paddleocr = model_loader.load_model('paddleocr')
            
            if self.paddleocr is None:
                # Fallback: try direct initialization
                self.logger.warning("Model loader failed, trying direct PaddleOCR initialization")
                self.paddleocr = self._initialize_paddleocr_direct()
            
            if self.paddleocr is None:
                raise RuntimeError("Failed to initialize PaddleOCR")
            
            # Test with small image
            test_result = self._test_engine()
            if not test_result:
                raise RuntimeError("Engine test failed")
            
            init_time = time.time() - start_time
            self.status = EngineStatus.READY
            self.logger.info(f"PaddleOCR initialized successfully in {init_time:.2f}s")
            return True
            
        except Exception as e:
            self.status = EngineStatus.ERROR
            self.logger.error(f"PaddleOCR initialization failed: {e}")
            return False
    
    def _initialize_paddleocr_direct(self) -> Optional[Any]:
        """Direct PaddleOCR initialization as fallback."""
        try:
            from paddleocr import PaddleOCR
            
            return PaddleOCR(
                use_angle_cls=self.use_angle_cls,
                lang=self.lang,
                use_gpu=self.use_gpu,
                show_log=False,
                # Optimization settings
                det_algorithm='DB',
                rec_algorithm='SVTR_LCNet',
                use_space_char=True,
                drop_score=0.3  # Lower threshold for better recall
            )
        except ImportError:
            self.logger.error("PaddleOCR not installed. Install with: pip install paddleocr")
            return None
        except Exception as e:
            self.logger.error(f"Direct PaddleOCR initialization failed: {e}")
            return None
    
    def _test_engine(self) -> bool:
        """Test engine with a simple image."""
        try:
            # Create small test image
            test_image = Image.new('RGB', (100, 50), color='white')
            test_array = np.array(test_image)
            
            # Quick test
            result = self.paddleocr.ocr(test_array, cls=False)
            return True
        except Exception as e:
            self.logger.error(f"Engine test failed: {e}")
            return False
    
    def extract(self, image: Any) -> Optional[OCRResult]:
        """
        Extract text from image using PaddleOCR.
        
        Args:
            image: Input image (file path, PIL Image, or numpy array)
            
        Returns:
            OCRResult with hierarchical text structure or None if failed
        """
        if self.status != EngineStatus.READY:
            self.logger.error("Engine not ready for extraction")
            return None
        
        self.status = EngineStatus.BUSY
        start_time = time.time()
        metrics = ProcessingMetrics("paddleocr_extraction")
        
        try:
            # Load and prepare image
            pil_image, img_array = self._prepare_image(image)
            if pil_image is None:
                return None
            
            # Perform OCR extraction
            ocr_results = self._perform_ocr(img_array)
            if not ocr_results:
                self.logger.warning("No OCR results from PaddleOCR")
                return self._create_empty_result(pil_image, metrics, start_time)
            
            # Process results into hierarchical structure
            ocr_result = self._process_ocr_results(ocr_results, pil_image, metrics, start_time)
            
            metrics.finish()
            if ocr_result:
                ocr_result.add_processing_metric(metrics)
            
            return ocr_result
            
        except Exception as e:
            metrics.add_error(f"PaddleOCR extraction failed: {str(e)}")
            metrics.finish()
            self.logger.error(f"PaddleOCR extraction error: {e}")
            return None
        finally:
            self.status = EngineStatus.READY
    
    def _prepare_image(self, image: Any) -> Tuple[Optional[Image.Image], Optional[np.ndarray]]:
        """Prepare image for PaddleOCR processing."""
        try:
            # Load image using image loader
            pil_image = self.image_loader.load_image(image)
            if pil_image is None:
                return None, None
            
            # Convert to RGB if needed
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Convert to numpy array
            img_array = np.array(pil_image)
            
            return pil_image, img_array
            
        except Exception as e:
            self.logger.error(f"Image preparation failed: {e}")
            return None, None
    
    def _perform_ocr(self, img_array: np.ndarray) -> Optional[List]:
        """Perform OCR using PaddleOCR."""
        try:
            # PaddleOCR returns: [[[bbox], (text, confidence)], ...]
            results = self.paddleocr.ocr(img_array, cls=self.use_angle_cls)
            
            # PaddleOCR returns list of lists for multiple images, get first result
            if isinstance(results, list) and len(results) > 0:
                return results[0] if results[0] is not None else []
            
            return []
            
        except Exception as e:
            self.logger.error(f"PaddleOCR processing failed: {e}")
            return None
    
    def _process_ocr_results(self, ocr_results: List, pil_image: Image.Image, 
                           metrics: ProcessingMetrics, start_time: float) -> Optional[OCRResult]:
        """Process PaddleOCR results into hierarchical structure."""
        try:
            if not ocr_results:
                return self._create_empty_result(pil_image, metrics, start_time)
            
            # Extract words from PaddleOCR results
            words = self._extract_words(ocr_results)
            if not words:
                return self._create_empty_result(pil_image, metrics, start_time)
            
            # Group words into lines based on vertical proximity
            lines = self._group_words_into_lines(words)
            
            # Group lines into paragraphs based on spacing
            paragraphs = self._group_lines_into_paragraphs(lines)
            
            # Create page
            page = Page(
                paragraphs=paragraphs,
                page_number=1,
                image_dimensions=(pil_image.width, pil_image.height),
                language=self.lang,
                processing_metadata={
                    'engine': 'paddleocr',
                    'words_detected': len(words),
                    'lines_detected': len(lines),
                    'paragraphs_detected': len(paragraphs)
                }
            )
            
            # Create final result
            processing_time = time.time() - start_time
            result = OCRResult(
                pages=[page],
                processing_time=processing_time,
                engine_info={
                    'name': 'paddleocr',
                    'version': self._get_paddleocr_version(),
                    'gpu_used': self.use_gpu,
                    'language': self.lang
                },
                metadata={
                    'total_words': len(words),
                    'processing_time_per_word': processing_time / max(1, len(words)),
                    'image_size': f"{pil_image.width}x{pil_image.height}"
                }
            )
            
            self.logger.info(
                f"PaddleOCR extracted {len(words)} words in {processing_time:.2f}s"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Result processing failed: {e}")
            metrics.add_error(f"Result processing failed: {str(e)}")
            return None
    
    def _extract_words(self, ocr_results: List) -> List[Word]:
        """Extract Word objects from PaddleOCR results."""
        words = []
        
        for line_result in ocr_results:
            try:
                if not line_result or len(line_result) != 2:
                    continue
                
                bbox_coords, (text, confidence) = line_result
                
                # Skip empty text
                if not text or not text.strip():
                    continue
                
                # Convert bbox coordinates
                bbox = self._create_bounding_box(bbox_coords)
                if bbox is None:
                    continue
                
                # Create confidence metrics
                conf_metrics = ConfidenceMetrics(
                    character_level=confidence,
                    word_level=confidence,
                    line_level=confidence,
                    text_quality=self._assess_text_quality(text),
                    spatial_quality=0.8,  # PaddleOCR generally good spatial accuracy
                    engine_name='paddleocr',
                    raw_confidence=confidence
                )
                
                # Create Word object
                word = Word(
                    text=text.strip(),
                    bbox=bbox,
                    confidence=conf_metrics
                )
                
                words.append(word)
                
            except Exception as e:
                self.logger.warning(f"Failed to process word result: {e}")
                continue
        
        return words
    
    def _create_bounding_box(self, bbox_coords: List) -> Optional[BoundingBox]:
        """Create BoundingBox from PaddleOCR coordinates."""
        try:
            # PaddleOCR returns [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            if len(bbox_coords) != 4:
                return None
            
            # Extract all x and y coordinates
            x_coords = [point[0] for point in bbox_coords]
            y_coords = [point[1] for point in bbox_coords]
            
            # Calculate bounding rectangle
            min_x = min(x_coords)
            min_y = min(y_coords)
            max_x = max(x_coords)
            max_y = max(y_coords)
            
            return BoundingBox(
                coordinates=(min_x, min_y, max_x, max_y),
                format=CoordinateFormat.XYXY
            )
            
        except Exception as e:
            self.logger.warning(f"Failed to create bounding box: {e}")
            return None
    
    def _assess_text_quality(self, text: str) -> float:
        """Assess text quality based on content characteristics."""
        if not text:
            return 0.0
        
        quality_score = 0.5  # Base score
        
        # Check for common OCR errors
        error_indicators = ['|', '~', '`', '@#$', '|||']
        if not any(indicator in text for indicator in error_indicators):
            quality_score += 0.2
        
        # Check for reasonable character distribution
        if text.isalnum() or any(c.isalnum() for c in text):
            quality_score += 0.2
        
        # Check length (very short or very long might be errors)
        if 1 <= len(text) <= 50:
            quality_score += 0.1
        
        return min(1.0, quality_score)
    
    def _group_words_into_lines(self, words: List[Word]) -> List[Line]:
        """Group words into lines based on vertical proximity."""
        if not words:
            return []
        
        # Sort words by vertical position (top to bottom)
        sorted_words = sorted(words, key=lambda w: w.bbox.xyxy[1])
        
        lines = []
        current_line_words = [sorted_words[0]]
        current_line_y = sorted_words[0].bbox.xyxy[1]
        
        # Group words with similar y-coordinates
        line_height_threshold = 10  # pixels
        
        for word in sorted_words[1:]:
            word_y = word.bbox.xyxy[1]
            
            if abs(word_y - current_line_y) <= line_height_threshold:
                # Same line
                current_line_words.append(word)
            else:
                # New line
                if current_line_words:
                    # Sort current line words by x-coordinate (left to right)
                    current_line_words.sort(key=lambda w: w.bbox.xyxy[0])
                    lines.append(Line(words=current_line_words))
                
                current_line_words = [word]
                current_line_y = word_y
        
        # Add the last line
        if current_line_words:
            current_line_words.sort(key=lambda w: w.bbox.xyxy[0])
            lines.append(Line(words=current_line_words))
        
        return lines
    
    def _group_lines_into_paragraphs(self, lines: List[Line]) -> List[Paragraph]:
        """Group lines into paragraphs based on spacing."""
        if not lines:
            return []
        
        paragraphs = []
        current_paragraph_lines = [lines[0]]
        
        # Calculate spacing threshold based on average line height
        if len(lines) > 1:
            line_heights = []
            for line in lines:
                if line.bbox:
                    _, _, _, height = line.bbox.xywh
                    line_heights.append(height)
            
            avg_line_height = sum(line_heights) / len(line_heights) if line_heights else 20
            spacing_threshold = avg_line_height * 1.5  # 1.5x line height indicates paragraph break
        else:
            spacing_threshold = 30  # Default threshold
        
        # Group lines with reasonable spacing
        for i in range(1, len(lines)):
            prev_line = lines[i-1]
            curr_line = lines[i]
            
            if prev_line.bbox and curr_line.bbox:
                # Calculate vertical spacing
                prev_bottom = prev_line.bbox.xyxy[3]
                curr_top = curr_line.bbox.xyxy[1]
                spacing = curr_top - prev_bottom
                
                if spacing <= spacing_threshold:
                    # Same paragraph
                    current_paragraph_lines.append(curr_line)
                else:
                    # New paragraph
                    if current_paragraph_lines:
                        paragraphs.append(Paragraph(lines=current_paragraph_lines))
                    current_paragraph_lines = [curr_line]
            else:
                # If no bbox info, assume same paragraph
                current_paragraph_lines.append(curr_line)
        
        # Add the last paragraph
        if current_paragraph_lines:
            paragraphs.append(Paragraph(lines=current_paragraph_lines))
        
        return paragraphs
    
    def _create_empty_result(self, pil_image: Image.Image, metrics: ProcessingMetrics, 
                           start_time: float) -> OCRResult:
        """Create empty result for images with no text."""
        processing_time = time.time() - start_time
        
        # Create empty page
        page = Page(
            paragraphs=[],
            page_number=1,
            image_dimensions=(pil_image.width, pil_image.height),
            language=self.lang,
            processing_metadata={
                'engine': 'paddleocr',
                'words_detected': 0,
                'lines_detected': 0,
                'paragraphs_detected': 0
            }
        )
        
        # Create empty confidence
        empty_confidence = ConfidenceMetrics(
            character_level=0.0,
            word_level=0.0,
            line_level=0.0,
            layout_level=0.0,
            text_quality=0.0,
            spatial_quality=0.0,
            engine_name='paddleocr'
        )
        
        result = OCRResult(
            pages=[page],
            confidence=empty_confidence,
            processing_time=processing_time,
            engine_info={
                'name': 'paddleocr',
                'version': self._get_paddleocr_version(),
                'gpu_used': self.use_gpu,
                'language': self.lang
            },
            metadata={
                'total_words': 0,
                'image_size': f"{pil_image.width}x{pil_image.height}",
                'no_text_detected': True
            }
        )
        
        return result
    
    def _get_paddleocr_version(self) -> str:
        """Get PaddleOCR version."""
        try:
            import paddleocr
            return getattr(paddleocr, '__version__', 'unknown')
        except:
            return 'unknown'
    
    def shutdown(self) -> None:
        """Shutdown PaddleOCR engine and cleanup resources."""
        self.logger.info("Shutting down PaddleOCR engine")
        
        if self.paddleocr is not None:
            try:
                # PaddleOCR doesn't have explicit cleanup, but we can clear the reference
                del self.paddleocr
                self.paddleocr = None
            except Exception as e:
                self.logger.error(f"Error during PaddleOCR cleanup: {e}")
        
        self.status = EngineStatus.SHUTDOWN
        
        # Force garbage collection
        import gc
        gc.collect()
    
    @property
    def is_ready(self) -> bool:
        """Check if engine is ready for processing."""
        return self.status == EngineStatus.READY and self.paddleocr is not None
    
    def get_engine_info(self) -> Dict[str, Any]:
        """Get comprehensive engine information."""
        return {
            'name': self.name,
            'version': self._get_paddleocr_version(),
            'status': self.status.value,
            'gpu_enabled': self.use_gpu,
            'language': self.lang,
            'use_angle_classification': self.use_angle_cls,
            'capabilities': [
                'text_detection',
                'text_recognition',
                'angle_classification',
                'multilingual_support',
                'gpu_acceleration'
            ],
            'supported_formats': [
                'jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp'
            ],
            'optimal_for': [
                'printed_text',
                'documents',
                'receipts',
                'forms',
                'mixed_content'
            ]
        }