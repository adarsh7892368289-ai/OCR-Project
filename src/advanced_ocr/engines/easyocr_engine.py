"""
EasyOCR Engine Implementation
Complementary OCR engine using EasyOCR for robust text extraction and result fusion.
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


class EasyOCREngine(BaseOCREngine):
    """
    EasyOCR implementation for complementary OCR results and fusion scenarios.
    Strong performance on diverse fonts and text styles with good multilingual support.
    """
    
    def __init__(self, config: OCRConfig):
        """
        Initialize EasyOCR engine.
        
        Args:
            config: OCR configuration
        """
        super().__init__("easyocr", config)
        self.reader = None
        self.image_loader = ImageLoader()
        
        # EasyOCR specific settings
        self.languages = ['en']  # Default to English
        self.use_gpu = self._detect_gpu()
        self.detail_level = 1  # 0=simple, 1=detailed
        self.paragraph_grouping = True
        
        self.logger.info(f"Initialized EasyOCR engine (GPU: {self.use_gpu})")
    
    def _detect_gpu(self) -> bool:
        """Detect if GPU acceleration is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def initialize(self) -> bool:
        """
        Initialize EasyOCR reader with optimized settings.
        
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
            
            # Load EasyOCR model
            self.reader = model_loader.load_model('easyocr')
            
            if self.reader is None:
                # Fallback: try direct initialization
                self.logger.warning("Model loader failed, trying direct EasyOCR initialization")
                self.reader = self._initialize_easyocr_direct()
            
            if self.reader is None:
                raise RuntimeError("Failed to initialize EasyOCR")
            
            # Test with small image
            test_result = self._test_engine()
            if not test_result:
                raise RuntimeError("Engine test failed")
            
            init_time = time.time() - start_time
            self.status = EngineStatus.READY
            self.logger.info(f"EasyOCR initialized successfully in {init_time:.2f}s")
            return True
            
        except Exception as e:
            self.status = EngineStatus.ERROR
            self.logger.error(f"EasyOCR initialization failed: {e}")
            return False
    
    def _initialize_easyocr_direct(self) -> Optional[Any]:
        """Direct EasyOCR initialization as fallback."""
        try:
            import easyocr
            
            return easyocr.Reader(
                self.languages,
                gpu=self.use_gpu,
                verbose=False,
                quantize=True,  # Use quantized models for better performance
                download_enabled=True
            )
        except ImportError:
            self.logger.error("EasyOCR not installed. Install with: pip install easyocr")
            return None
        except Exception as e:
            self.logger.error(f"Direct EasyOCR initialization failed: {e}")
            return None
    
    def _test_engine(self) -> bool:
        """Test engine with a simple image."""
        try:
            # Create small test image
            test_image = Image.new('RGB', (100, 50), color='white')
            test_array = np.array(test_image)
            
            # Quick test
            result = self.reader.readtext(test_array, detail=0)
            return True
        except Exception as e:
            self.logger.error(f"Engine test failed: {e}")
            return False
    
    def extract(self, image: Any) -> Optional[OCRResult]:
        """
        Extract text from image using EasyOCR.
        
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
        metrics = ProcessingMetrics("easyocr_extraction")
        
        try:
            # Load and prepare image
            pil_image, img_array = self._prepare_image(image)
            if pil_image is None:
                return None
            
            # Perform OCR extraction
            ocr_results = self._perform_ocr(img_array)
            if not ocr_results:
                self.logger.warning("No OCR results from EasyOCR")
                return self._create_empty_result(pil_image, metrics, start_time)
            
            # Process results into hierarchical structure
            ocr_result = self._process_ocr_results(ocr_results, pil_image, metrics, start_time)
            
            metrics.finish()
            if ocr_result:
                ocr_result.add_processing_metric(metrics)
            
            return ocr_result
            
        except Exception as e:
            metrics.add_error(f"EasyOCR extraction failed: {str(e)}")
            metrics.finish()
            self.logger.error(f"EasyOCR extraction error: {e}")
            return None
        finally:
            self.status = EngineStatus.READY
    
    def _prepare_image(self, image: Any) -> Tuple[Optional[Image.Image], Optional[np.ndarray]]:
        """Prepare image for EasyOCR processing."""
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
        """Perform OCR using EasyOCR."""
        try:
            # EasyOCR readtext returns: [(bbox, text, confidence), ...]
            results = self.reader.readtext(
                img_array,
                detail=self.detail_level,
                paragraph=self.paragraph_grouping,
                width_ths=0.7,  # Text width threshold
                height_ths=0.7,  # Text height threshold
                # Optimized settings for better accuracy
                decoder='greedy',
                beamWidth=5,
                batch_size=1
            )
            
            return results if results else []
            
        except Exception as e:
            self.logger.error(f"EasyOCR processing failed: {e}")
            return None
    
    def _process_ocr_results(self, ocr_results: List, pil_image: Image.Image, 
                           metrics: ProcessingMetrics, start_time: float) -> Optional[OCRResult]:
        """Process EasyOCR results into hierarchical structure."""
        try:
            if not ocr_results:
                return self._create_empty_result(pil_image, metrics, start_time)
            
            # Extract words from EasyOCR results
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
                language=self.languages[0] if self.languages else 'unknown',
                processing_metadata={
                    'engine': 'easyocr',
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
                    'name': 'easyocr',
                    'version': self._get_easyocr_version(),
                    'gpu_used': self.use_gpu,
                    'languages': self.languages
                },
                metadata={
                    'total_words': len(words),
                    'processing_time_per_word': processing_time / max(1, len(words)),
                    'image_size': f"{pil_image.width}x{pil_image.height}"
                }
            )
            
            self.logger.info(
                f"EasyOCR extracted {len(words)} words in {processing_time:.2f}s"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Result processing failed: {e}")
            metrics.add_error(f"Result processing failed: {str(e)}")
            return None
    
    def _extract_words(self, ocr_results: List) -> List[Word]:
        """Extract Word objects from EasyOCR results."""
        words = []
        
        for result in ocr_results:
            try:
                if len(result) < 3:
                    continue
                
                bbox_coords, text, confidence = result
                
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
                    spatial_quality=0.85,  # EasyOCR generally has good spatial accuracy
                    engine_name='easyocr',
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
        """Create BoundingBox from EasyOCR coordinates."""
        try:
            # EasyOCR returns [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
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
        error_indicators = ['|', '~', '`', '@#', '|||', '§', '¶']
        if any(indicator in text for indicator in error_indicators):
            quality_score -= 0.2
        
        # Check for reasonable character distribution
        if text.isalnum() or any(c.isalnum() for c in text):
            quality_score += 0.2
        
        # Penalize very short single character results (often errors)
        if len(text) == 1 and not text.isalnum():
            quality_score -= 0.2
        elif 2 <= len(text) <= 50:
            quality_score += 0.1
        
        return max(0.0, min(1.0, quality_score))
    
    def _group_words_into_lines(self, words: List[Word]) -> List[Line]:
        """Group words into lines based on vertical proximity."""
        if not words:
            return []
        
        # Sort words by vertical position (top to bottom)
        sorted_words = sorted(words, key=lambda w: w.bbox.xyxy[1])
        
        lines = []
        current_line_words = [sorted_words[0]]
        current_line_center_y = (sorted_words[0].bbox.xyxy[1] + sorted_words[0].bbox.xyxy[3]) / 2
        
        # Group words with similar y-coordinates
        for word in sorted_words[1:]:
            word_center_y = (word.bbox.xyxy[1] + word.bbox.xyxy[3]) / 2
            word_height = word.bbox.xyxy[3] - word.bbox.xyxy[1]
            
            # Dynamic threshold based on word height
            line_threshold = max(word_height * 0.5, 8)  # At least 8 pixels
            
            if abs(word_center_y - current_line_center_y) <= line_threshold:
                # Same line
                current_line_words.append(word)
                # Update center Y to average of current line
                all_centers = [(w.bbox.xyxy[1] + w.bbox.xyxy[3]) / 2 for w in current_line_words]
                current_line_center_y = sum(all_centers) / len(all_centers)
            else:
                # New line
                if current_line_words:
                    # Sort current line words by x-coordinate (left to right)
                    current_line_words.sort(key=lambda w: w.bbox.xyxy[0])
                    lines.append(Line(words=current_line_words))
                
                current_line_words = [word]
                current_line_center_y = word_center_y
        
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
            spacing_threshold = avg_line_height * 2.0  # 2x line height indicates paragraph break
        else:
            spacing_threshold = 40  # Default threshold
        
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
            language=self.languages[0] if self.languages else 'unknown',
            processing_metadata={
                'engine': 'easyocr',
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
            engine_name='easyocr'
        )
        
        result = OCRResult(
            pages=[page],
            confidence=empty_confidence,
            processing_time=processing_time,
            engine_info={
                'name': 'easyocr',
                'version': self._get_easyocr_version(),
                'gpu_used': self.use_gpu,
                'languages': self.languages
            },
            metadata={
                'total_words': 0,
                'image_size': f"{pil_image.width}x{pil_image.height}",
                'no_text_detected': True
            }
        )
        
        return result
    
    def _get_easyocr_version(self) -> str:
        """Get EasyOCR version."""
        try:
            import easyocr
            return getattr(easyocr, '__version__', 'unknown')
        except:
            return 'unknown'
    
    def shutdown(self) -> None:
        """Shutdown EasyOCR engine and cleanup resources."""
        self.logger.info("Shutting down EasyOCR engine")
        
        if self.reader is not None:
            try:
                # EasyOCR doesn't have explicit cleanup, but we can clear the reference
                del self.reader
                self.reader = None
            except Exception as e:
                self.logger.error(f"Error during EasyOCR cleanup: {e}")
        
        self.status = EngineStatus.SHUTDOWN
        
        # Force garbage collection
        import gc
        gc.collect()
    
    @property
    def is_ready(self) -> bool:
        """Check if engine is ready for processing."""
        return self.status == EngineStatus.READY and self.reader is not None
    
    def get_engine_info(self) -> Dict[str, Any]:
        """Get comprehensive engine information."""
        return {
            'name': self.name,
            'version': self._get_easyocr_version(),
            'status': self.status.value,
            'gpu_enabled': self.use_gpu,
            'languages': self.languages,
            'detail_level': self.detail_level,
            'paragraph_grouping': self.paragraph_grouping,
            'capabilities': [
                'text_detection',
                'text_recognition',
                'multilingual_support',
                'gpu_acceleration',
                'paragraph_detection'
            ],
            'supported_formats': [
                'jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp'
            ],
            'optimal_for': [
                'diverse_fonts',
                'multilingual_text',
                'scene_text',
                'stylized_text',
                'complementary_ocr'
            ]
        }