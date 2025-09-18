"""
TrOCR Engine Implementation
Specialized handwritten text recognition using transformer models.
Optimized for both printed and handwritten text with high accuracy.
"""

import time
import logging
from typing import Any, Optional, List, Tuple, Dict
import numpy as np
from PIL import Image
import torch

from .base_engine import BaseOCREngine, EngineStatus
from ..results import (
    OCRResult, Page, Paragraph, Line, Word, 
    BoundingBox, ConfidenceMetrics, CoordinateFormat, ProcessingMetrics
)
from ..config import EngineConfig
from ..utils.model_utils import get_model_loader
from ..utils.image_utils import ImageLoader


class TrOCREngine(BaseOCREngine):
    """
    TrOCR implementation for advanced handwritten and printed text recognition.
    Uses transformer-based models for high accuracy on challenging text.
    """
    
    def __init__(self, config: EngineConfig):
        """
        Initialize TrOCR engine.
        
        Args:
            config: Engine configuration
        """
        super().__init__(config)
        self.processor = None
        self.model = None
        self.image_loader = ImageLoader()
        
        # TrOCR specific settings
        self.model_name = "microsoft/trocr-base-printed"  # Default model
        self.device = self._detect_device()
        self.max_length = 384  # Maximum sequence length
        self.batch_size = 1    # Process one image at a time
        
        # Model variants based on content type
        self.model_variants = {
            'printed': 'microsoft/trocr-base-printed',
            'handwritten': 'microsoft/trocr-base-handwritten',
            'large_printed': 'microsoft/trocr-large-printed',
            'large_handwritten': 'microsoft/trocr-large-handwritten'
        }
        
        self.logger.info(f"Initialized TrOCR engine (Device: {self.device})")
    
    def _detect_device(self) -> str:
        """Detect best available device for inference."""
        if self.config.gpu_enabled:
            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda"
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    return "mps"  # Apple Silicon
            except ImportError:
                pass
        return "cpu"
    
    def _initialize_engine(self) -> bool:
        """
        Initialize TrOCR model and processor.
        
        Returns:
            True if initialization successful
        """
        try:
            # Get model loader
            model_loader = get_model_loader(self.config)
            
            # Try to load from model cache first
            model_data = model_loader.load_model('trocr')
            
            if model_data and isinstance(model_data, dict):
                self.processor = model_data.get('processor')
                self.model = model_data.get('model')
            
            if self.processor is None or self.model is None:
                # Fallback: direct initialization
                self.logger.warning("Model loader failed, trying direct TrOCR initialization")
                success = self._initialize_trocr_direct()
                if not success:
                    return False
            
            # Move model to appropriate device
            if self.model and hasattr(self.model, 'to'):
                self.model = self.model.to(self.device)
                self.model.eval()
            
            # Test engine
            if not self._test_engine():
                raise RuntimeError("Engine test failed")
            
            self.logger.info(f"TrOCR initialized successfully with model: {self.model_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"TrOCR initialization failed: {e}")
            return False
    
    def _initialize_trocr_direct(self) -> bool:
        """Direct TrOCR initialization as fallback."""
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            
            # Load processor and model
            self.processor = TrOCRProcessor.from_pretrained(self.model_name)
            self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name)
            
            return True
            
        except ImportError:
            self.logger.error(
                "TrOCR dependencies not installed. "
                "Install with: pip install transformers torch pillow"
            )
            return False
        except Exception as e:
            self.logger.error(f"Direct TrOCR initialization failed: {e}")
            return False
    
    def _test_engine(self) -> bool:
        """Test engine with a simple image."""
        try:
            # Create small test image with text
            test_image = Image.new('RGB', (200, 50), color='white')
            
            # Quick processing test
            pixel_values = self.processor(test_image, return_tensors="pt").pixel_values
            if self.device != "cpu":
                pixel_values = pixel_values.to(self.device)
            
            with torch.no_grad():
                generated_ids = self.model.generate(pixel_values, max_length=10)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Engine test failed: {e}")
            return False
    
    def _extract_text(self, image: Any) -> Optional[OCRResult]:
        """
        Extract text from image using TrOCR.
        
        Args:
            image: Input image (file path, PIL Image, or numpy array)
            
        Returns:
            OCRResult with extracted text or None if failed
        """
        start_time = time.time()
        metrics = ProcessingMetrics("trocr_extraction")
        
        try:
            # Load and prepare image
            pil_image, img_array = self._prepare_image(image)
            if pil_image is None:
                return None
            
            # Detect if image likely contains handwritten text
            text_type = self._detect_text_type(img_array)
            
            # Use appropriate model if available
            if text_type == 'handwritten' and self._has_handwritten_model():
                self._switch_model('handwritten')
            
            # Perform text recognition
            extracted_text, confidence = self._perform_recognition(pil_image)
            
            if not extracted_text or not extracted_text.strip():
                self.logger.warning("No text extracted from image")
                return self._create_empty_result(pil_image, metrics, start_time)
            
            # Create OCR result with hierarchical structure
            ocr_result = self._create_ocr_result(
                extracted_text, confidence, pil_image, text_type, metrics, start_time
            )
            
            metrics.finish()
            if ocr_result:
                ocr_result.add_processing_metric(metrics)
            
            return ocr_result
            
        except Exception as e:
            metrics.add_error(f"TrOCR extraction failed: {str(e)}")
            metrics.finish()
            self.logger.error(f"TrOCR extraction error: {e}")
            return None
    
    def _prepare_image(self, image: Any) -> Tuple[Optional[Image.Image], Optional[np.ndarray]]:
        """Prepare image for TrOCR processing."""
        try:
            # Load image
            pil_image = self.image_loader.load_image(image)
            if pil_image is None:
                return None, None
            
            # Convert to RGB if needed
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # TrOCR works best with reasonably sized images
            # Resize if too large while preserving aspect ratio
            max_size = 2048
            if max(pil_image.size) > max_size:
                pil_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            # Convert to numpy for additional analysis
            img_array = np.array(pil_image)
            
            return pil_image, img_array
            
        except Exception as e:
            self.logger.error(f"Image preparation failed: {e}")
            return None, None
    
    def _detect_text_type(self, img_array: np.ndarray) -> str:
        """
        Detect if image likely contains printed or handwritten text.
        Simple heuristic based on edge characteristics.
        
        Args:
            img_array: Image array
            
        Returns:
            'printed' or 'handwritten'
        """
        try:
            import cv2
            
            # Convert to grayscale
            if len(img_array.shape) == 3:
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = img_array
            
            # Apply edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Calculate edge characteristics
            total_pixels = gray.size
            edge_pixels = np.sum(edges > 0)
            edge_density = edge_pixels / total_pixels
            
            # Apply morphological operations to detect text structure
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            dilated = cv2.dilate(edges, kernel, iterations=1)
            
            # Find contours
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Analyze contour regularity (printed text tends to be more regular)
            if len(contours) > 5:
                areas = [cv2.contourArea(c) for c in contours if cv2.contourArea(c) > 10]
                if areas:
                    area_variance = np.var(areas) / (np.mean(areas) + 1e-6)
                    
                    # High variance suggests handwritten, low variance suggests printed
                    if area_variance > 2.0 or edge_density > 0.15:
                        return 'handwritten'
            
            return 'printed'
            
        except Exception:
            # Default to printed if detection fails
            return 'printed'
    
    def _has_handwritten_model(self) -> bool:
        """Check if handwritten model variant is available."""
        # For now, assume we only have the base model
        # In production, this would check if handwritten model is loaded
        return False
    
    def _switch_model(self, text_type: str) -> bool:
        """Switch to appropriate model variant if available."""
        target_model = self.model_variants.get(text_type)
        if target_model and target_model != self.model_name:
            try:
                # In production, this would switch models
                # For now, just log the intention
                self.logger.info(f"Would switch to model: {target_model}")
                return True
            except Exception as e:
                self.logger.warning(f"Failed to switch model: {e}")
                return False
        return True
    
    def _perform_recognition(self, pil_image: Image.Image) -> Tuple[str, float]:
        """
        Perform text recognition on image.
        
        Args:
            pil_image: PIL Image to process
            
        Returns:
            Tuple of (extracted_text, confidence_score)
        """
        try:
            # Process image
            pixel_values = self.processor(pil_image, return_tensors="pt").pixel_values
            
            # Move to device
            if self.device != "cpu":
                pixel_values = pixel_values.to(self.device)
            
            # Generate text
            with torch.no_grad():
                generated_ids = self.model.generate(
                    pixel_values,
                    max_length=self.max_length,
                    num_beams=4,
                    early_stopping=True,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id
                )
            
            # Decode generated text
            generated_text = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0]
            
            # Calculate confidence (simplified - in production, use beam scores)
            confidence = self._estimate_confidence(generated_text, pil_image)
            
            return generated_text, confidence
            
        except Exception as e:
            self.logger.error(f"Recognition failed: {e}")
            return "", 0.0
    
    def _estimate_confidence(self, text: str, image: Image.Image) -> float:
        """
        Estimate confidence score for extracted text.
        
        Args:
            text: Extracted text
            image: Source image
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not text or not text.strip():
            return 0.0
        
        confidence = 0.5  # Base confidence
        
        # Text length factor (reasonable length suggests better extraction)
        text_len = len(text.strip())
        if 5 <= text_len <= 200:
            confidence += 0.2
        elif text_len > 200:
            confidence += 0.1
        
        # Character quality (alphanumeric vs special characters)
        alpha_ratio = sum(c.isalnum() or c.isspace() for c in text) / len(text)
        confidence += alpha_ratio * 0.2
        
        # Word formation quality
        words = text.split()
        if words:
            valid_words = sum(1 for word in words if len(word) > 1 and word.isalnum())
            word_quality = valid_words / len(words)
            confidence += word_quality * 0.1
        
        return min(1.0, confidence)
    
    def _create_ocr_result(self, text: str, confidence: float, pil_image: Image.Image,
                          text_type: str, metrics: ProcessingMetrics, 
                          start_time: float) -> OCRResult:
        """Create structured OCR result from extracted text."""
        try:
            processing_time = time.time() - start_time
            
            # Since TrOCR doesn't provide spatial information, create simple structure
            # Split text into logical units
            paragraphs_text = self._split_into_paragraphs(text)
            
            paragraphs = []
            for para_idx, para_text in enumerate(paragraphs_text):
                lines_text = para_text.split('\n')
                lines = []
                
                for line_idx, line_text in enumerate(lines_text):
                    if not line_text.strip():
                        continue
                    
                    words_text = line_text.split()
                    words = []
                    
                    # Create words without spatial information
                    for word_idx, word_text in enumerate(words_text):
                        if not word_text.strip():
                            continue
                        
                        # Create dummy bounding box (TrOCR doesn't provide coordinates)
                        # In production, you might combine with a separate text detection model
                        word_bbox = BoundingBox(
                            (word_idx * 50, line_idx * 25, (word_idx + 1) * 50, (line_idx + 1) * 25),
                            CoordinateFormat.XYXY
                        )
                        
                        word_confidence = ConfidenceMetrics(
                            character_level=confidence,
                            word_level=confidence,
                            line_level=confidence,
                            text_quality=self._assess_text_quality(word_text),
                            spatial_quality=0.3,  # Low since we don't have real spatial info
                            engine_name='trocr',
                            raw_confidence=confidence
                        )
                        
                        word = Word(
                            text=word_text,
                            bbox=word_bbox,
                            confidence=word_confidence
                        )
                        words.append(word)
                    
                    if words:
                        lines.append(Line(words=words))
                
                if lines:
                    paragraphs.append(Paragraph(lines=lines))
            
            # Create page
            page = Page(
                paragraphs=paragraphs,
                page_number=1,
                image_dimensions=(pil_image.width, pil_image.height),
                language='auto',
                processing_metadata={
                    'engine': 'trocr',
                    'text_type': text_type,
                    'model_name': self.model_name,
                    'device': self.device,
                    'words_detected': len(text.split()),
                    'lines_detected': len([line for para in paragraphs for line in para.lines]),
                    'paragraphs_detected': len(paragraphs)
                }
            )
            
            # Create final result
            result = OCRResult(
                pages=[page],
                processing_time=processing_time,
                engine_info={
                    'name': 'trocr',
                    'version': self._get_trocr_version(),
                    'model_name': self.model_name,
                    'device': self.device,
                    'text_type_detected': text_type
                },
                metadata={
                    'total_words': len(text.split()),
                    'processing_time_per_word': processing_time / max(1, len(text.split())),
                    'image_size': f"{pil_image.width}x{pil_image.height}",
                    'full_text': text
                }
            )
            
            self.logger.info(
                f"TrOCR extracted {len(text.split())} words in {processing_time:.2f}s"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Result creation failed: {e}")
            return None
    
    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text into logical paragraphs."""
        # Split on double newlines or significant spacing
        paragraphs = []
        current_para = []
        
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                if current_para:
                    paragraphs.append('\n'.join(current_para))
                    current_para = []
            else:
                current_para.append(line)
        
        if current_para:
            paragraphs.append('\n'.join(current_para))
        
        # If no paragraph breaks found, treat as single paragraph
        if not paragraphs and text.strip():
            paragraphs = [text.strip()]
        
        return paragraphs
    
    def _assess_text_quality(self, text: str) -> float:
        """Assess text quality based on content characteristics."""
        if not text:
            return 0.0
        
        quality_score = 0.5  # Base score
        
        # Check for common OCR errors
        error_indicators = ['�', '|', '~', '`', '|||', '@#$']
        if not any(indicator in text for indicator in error_indicators):
            quality_score += 0.2
        
        # Check for reasonable character distribution
        if text.isalnum() or any(c.isalnum() for c in text):
            quality_score += 0.2
        
        # Check length
        if 1 <= len(text) <= 50:
            quality_score += 0.1
        
        return min(1.0, quality_score)
    
    def _create_empty_result(self, pil_image: Image.Image, metrics: ProcessingMetrics, 
                           start_time: float) -> OCRResult:
        """Create empty result for images with no text."""
        processing_time = time.time() - start_time
        
        page = Page(
            paragraphs=[],
            page_number=1,
            image_dimensions=(pil_image.width, pil_image.height),
            language='auto',
            processing_metadata={
                'engine': 'trocr',
                'words_detected': 0,
                'lines_detected': 0,
                'paragraphs_detected': 0
            }
        )
        
        empty_confidence = ConfidenceMetrics(
            character_level=0.0,
            word_level=0.0,
            line_level=0.0,
            layout_level=0.0,
            text_quality=0.0,
            spatial_quality=0.0,
            engine_name='trocr'
        )
        
        result = OCRResult(
            pages=[page],
            confidence=empty_confidence,
            processing_time=processing_time,
            engine_info={
                'name': 'trocr',
                'version': self._get_trocr_version(),
                'model_name': self.model_name,
                'device': self.device
            },
            metadata={
                'total_words': 0,
                'image_size': f"{pil_image.width}x{pil_image.height}",
                'no_text_detected': True
            }
        )
        
        return result
    
    def _get_trocr_version(self) -> str:
        """Get TrOCR/transformers version."""
        try:
            import transformers
            return getattr(transformers, '__version__', 'unknown')
        except:
            return 'unknown'
    
    def _cleanup_engine(self) -> None:
        """Cleanup TrOCR engine resources."""
        self.logger.info("Shutting down TrOCR engine")
        
        try:
            # Clear CUDA cache if using GPU
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Clear model references
            if self.model is not None:
                del self.model
                self.model = None
            
            if self.processor is not None:
                del self.processor
                self.processor = None
            
        except Exception as e:
            self.logger.error(f"Error during TrOCR cleanup: {e}")
        
        # Force garbage collection
        import gc
        gc.collect()
    
    @property
    def is_ready(self) -> bool:
        """Check if engine is ready for processing."""
        return (self.status == EngineStatus.READY and 
                self.processor is not None and 
                self.model is not None)
    
    def get_engine_info(self) -> Dict[str, Any]:
        """Get comprehensive engine information."""
        return {
            'name': self.name,
            'version': self._get_trocr_version(),
            'status': self.status.value,
            'model_name': self.model_name,
            'device': self.device,
            'available_variants': list(self.model_variants.keys()),
            'capabilities': [
                'handwritten_text',
                'printed_text',
                'transformer_based',
                'high_accuracy',
                'multilingual_support'
            ],
            'supported_formats': [
                'jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp'
            ],
            'optimal_for': [
                'handwritten_text',
                'challenging_text',
                'historical_documents',
                'forms',
                'notes'
            ],
            'limitations': [
                'no_spatial_coordinates',
                'single_text_block_processing',
                'requires_separate_text_detection'
            ]
        }