# src/advanced_ocr/engines/trocr_engine.py
"""
TrOCR Engine Implementation - COMPLETELY FIXED VERSION

This module provides the TrOCR engine implementation following the project's
architectural principles. It implements the abstract base class methods and
focuses solely on text extraction from preprocessed images and provided regions.

CRITICAL FIXES APPLIED:
- Fixed numpy array input handling per base_engine.py interface
- Proper PIL Image conversion for TrOCR processing
- Handles TextRegion input correctly (not BoundingBox)
- Proper coordinate extraction from BoundingBox objects
- Correct OCRResult and TextRegion constructors
- Robust error handling for all edge cases
"""

import logging
import numpy as np
import torch
from PIL import Image
from typing import List, Optional, Dict, Any
import cv2
import warnings

from .base_engine import BaseOCREngine
from ..results import (
    OCRResult, TextRegion, BoundingBox, ConfidenceMetrics, 
    Word, Line, TextLevel, ContentType, BoundingBoxFormat
)
from ..utils.image_utils import ImageProcessor, CoordinateTransformer
from ..utils.text_utils import TextCleaner, UnicodeNormalizer
from ..config import OCRConfig
from ..utils.logger import OCRLogger

# Suppress transformer warnings
warnings.filterwarnings('ignore', category=UserWarning, module='transformers')

logger = OCRLogger(__name__)


class TrOCREngine(BaseOCREngine):
    """
    TrOCR-based OCR engine for handwritten text recognition
    
    RESPONSIBILITIES (following project architecture):
    - Extract text using TrOCR from PREPROCESSED image + text_regions
    - Return raw OCRResult with confidence scores
    - Handle TrOCR model loading directly
    - Use text_utils.py for basic text operations
    
    WHAT IT SHOULD NOT DO:
    - Image preprocessing (already done by image_processor.py)
    - Text region detection (regions provided by text_detector.py)
    - Result postprocessing (done by text_processor.py)
    """
    
    def __init__(self, config: OCRConfig):
        super().__init__(config)
        self.engine_name = "trocr"
        
        # Model configuration handling
        engine_config = config.engines.get('trocr', None)
        if engine_config:
            self.model_name = (
                getattr(engine_config, 'model_path', None) or 
                engine_config.custom_params.get('model_name', None) or
                'microsoft/trocr-base-handwritten'
            )
            self.batch_size = getattr(engine_config, 'batch_size', 4)
            self.max_length = getattr(engine_config, 'max_length', 256)
        else:
            self.model_name = 'microsoft/trocr-base-handwritten'
            self.batch_size = 4
            self.max_length = 256
        
        self.device = self._determine_device(engine_config)
        
        # Model components (lazy loaded)
        self.processor = None
        self.model = None
        
        # Utility classes from project architecture
        self.text_cleaner = TextCleaner()
        self.unicode_normalizer = UnicodeNormalizer()
        
        logger.info(f"TrOCR engine initialized: {self.model_name} on {self.device}")
    
    def _determine_device(self, engine_config) -> str:
        """Determine optimal device for TrOCR"""
        if engine_config and getattr(engine_config, 'force_cpu', False):
            return 'cpu'
        
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'
    
    def _extract_implementation(self, image: np.ndarray, text_regions: List[TextRegion]) -> OCRResult:
        """
        COMPLETELY FIXED: Handle numpy array input correctly per base_engine.py interface
        
        Core TrOCR text extraction from preprocessed image and detected regions.
        This method is called by the base class extract() method.
        
        Args:
            image: PREPROCESSED numpy array (from base_engine.py interface)
            text_regions: List of TextRegion objects (from preprocessing pipeline)
            
        Returns:
            OCRResult with extracted text and confidence scores
        """
        logger.debug(f"TrOCR extracting from {len(text_regions)} TextRegion objects")
        
        # CRITICAL FIX: Convert numpy array to PIL Image for TrOCR processing
        try:
            if isinstance(image, np.ndarray):
                if len(image.shape) == 2:  # Grayscale
                    pil_image = Image.fromarray(image, mode='L')
                    logger.debug(f"Converted grayscale numpy array to PIL: {image.shape}")
                elif len(image.shape) == 3:
                    if image.shape[2] == 3:  # RGB
                        pil_image = Image.fromarray(image, mode='RGB')
                        logger.debug(f"Converted RGB numpy array to PIL: {image.shape}")
                    elif image.shape[2] == 4:  # RGBA
                        pil_image = Image.fromarray(image, mode='RGBA')
                        logger.debug(f"Converted RGBA numpy array to PIL: {image.shape}")
                    else:
                        raise ValueError(f"Unsupported number of channels: {image.shape[2]}")
                else:
                    raise ValueError(f"Invalid image dimensions: {len(image.shape)}")
            elif hasattr(image, 'size'):  # Already PIL Image
                pil_image = image
                logger.debug("Image is already PIL Image")
            else:
                raise ValueError(f"Unsupported image type: {type(image)}")
                
        except Exception as e:
            logger.error(f"Failed to convert image for TrOCR processing: {e}")
            return OCRResult(
                text="",
                confidence=0.0,
                processing_time=0.0,
                engine_name=self.engine_name,
                success=False,
                error_message=f"Image conversion failed: {e}"
            )
        
        # Ensure models are loaded
        try:
            self._ensure_models_loaded()
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            return OCRResult(
                text="",
                confidence=0.0,
                processing_time=0.0,
                engine_name=self.engine_name,
                success=False,
                error_message=f"Model loading failed: {e}"
            )
        
        # Process regions in batches for efficiency
        all_text_regions = []
        total_confidence = 0.0
        processed_count = 0
        
        try:
            for batch_start in range(0, len(text_regions), self.batch_size):
                batch_end = min(batch_start + self.batch_size, len(text_regions))
                batch_regions = text_regions[batch_start:batch_end]
                
                # Use PIL image for processing
                batch_results = self._process_region_batch(pil_image, batch_regions)
                all_text_regions.extend(batch_results)
                
                # Accumulate confidence scores
                for region in batch_results:
                    if hasattr(region, 'confidence') and region.confidence:
                        if hasattr(region.confidence, 'overall'):
                            total_confidence += region.confidence.overall
                            processed_count += 1
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            # Continue with partial results
        
        # Calculate overall confidence
        overall_confidence = total_confidence / processed_count if processed_count > 0 else 0.0
        
        # Combine text from all regions
        combined_text = self._combine_region_texts(all_text_regions)
        
        # FIXED: Use exact OCRResult constructor from results.py
        result = OCRResult(
            text=combined_text,
            confidence=overall_confidence,
            processing_time=0.0,  # Will be set by base class
            engine_name=self.engine_name,
            success=True,
            metadata={
                'regions_processed': len(all_text_regions),
                'input_regions': len(text_regions),
                'image_conversion': 'numpy_to_pil'
            }
        )
        
        logger.info(f"TrOCR extraction completed: {len(all_text_regions)} regions, "
                   f"confidence: {overall_confidence:.3f}")
        
        return result
    
    def _load_models(self):
        """Load TrOCR models directly"""
        logger.info(f"Loading TrOCR model: {self.model_name}")
        
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        except ImportError as e:
            raise RuntimeError(f"Transformers library required for TrOCR: {e}")
        
        try:
            # Load processor
            processor = TrOCRProcessor.from_pretrained(self.model_name)
            
            # Load model directly  
            model = VisionEncoderDecoderModel.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32
            )
            
            # Move to device and set eval mode
            model = model.to(self.device)
            model.eval()
            
            logger.info(f"TrOCR model loaded successfully on {self.device}")
            return processor, model
            
        except Exception as e:
            logger.error(f"TrOCR model loading failed: {e}")
            raise RuntimeError(f"Failed to load TrOCR model: {e}")
    
    def _ensure_models_loaded(self):
        """Ensure TrOCR models are loaded before processing"""
        if self.processor is None or self.model is None:
            self.processor, self.model = self._load_models()
    
    def _process_region_batch(self, image: Image.Image, regions: List[TextRegion]) -> List[TextRegion]:
        """
        FIXED: Process TextRegion objects through TrOCR
        
        Args:
            image: Source PIL Image (preprocessed)
            regions: List of TextRegion objects to process
            
        Returns:
            List of TextRegion objects with extracted text
        """
        batch_results = []
        
        try:
            # Extract region images from TextRegion objects
            region_images = []
            valid_regions = []
            
            for region in regions:
                # FIXED: Extract BoundingBox from TextRegion
                bbox = region.bbox
                region_img = self._extract_region_image(image, bbox)
                if region_img is not None:
                    region_images.append(region_img)
                    valid_regions.append(region)
                else:
                    logger.debug(f"Skipping invalid region: {getattr(region, 'element_id', 'unknown')}")
            
            if not region_images:
                logger.warning("No valid region images extracted from batch")
                return batch_results
            
            # Process through TrOCR model
            with torch.no_grad():
                # Prepare batch inputs
                pixel_values = self.processor(
                    images=region_images,
                    return_tensors="pt"
                ).pixel_values.to(self.device)
                
                # Generate text with beam search
                generated_ids = self.model.generate(
                    pixel_values,
                    max_length=self.max_length,
                    num_beams=4,
                    early_stopping=True,
                    do_sample=False,
                    pad_token_id=self.processor.tokenizer.pad_token_id
                )
                
                # Decode generated text
                generated_texts = self.processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True
                )
            
            # FIXED: Create TextRegion objects with correct constructor
            for region, text in zip(valid_regions, generated_texts):
                cleaned_text = self._clean_extracted_text(text)
                
                if cleaned_text.strip():  # Only include non-empty results
                    confidence_score = self._estimate_confidence(cleaned_text, region.bbox)
                    
                    # Create ConfidenceMetrics using exact constructor
                    confidence_metrics = ConfidenceMetrics(
                        overall=confidence_score,
                        text_detection=confidence_score,
                        text_recognition=confidence_score,
                        layout_analysis=0.0,
                        language_detection=0.0
                    )
                    
                    # FIXED: Use exact TextRegion constructor parameters
                    text_region = TextRegion(
                        text=cleaned_text,
                        bbox=region.bbox,  # Reuse the original bbox
                        confidence=confidence_metrics,
                        level=TextLevel.WORD,  # Required parameter
                        element_id=f"trocr_{len(batch_results)}",
                        content_type=ContentType.PRINTED_TEXT,
                        language="en",
                        engine_name=self.engine_name
                    )
                    
                    batch_results.append(text_region)
                else:
                    logger.debug(f"Empty text extracted for region {getattr(region, 'element_id', 'unknown')}")
        
        except Exception as e:
            logger.error(f"TrOCR batch processing failed: {e}")
            import traceback
            logger.error(f"TrOCR batch processing traceback: {traceback.format_exc()}")
            # Return partial results rather than failing completely
        
        logger.info(f"TrOCR batch processing: {len(batch_results)} valid results from {len(regions)} regions")
        return batch_results
    
    def _extract_region_image(self, image: Image.Image, bbox: BoundingBox) -> Optional[Image.Image]:
        """COMPLETELY FIXED: Handle both PIL Image and numpy array inputs safely"""
        try:
            # CRITICAL FIX: Ensure we have a PIL Image
            if isinstance(image, np.ndarray):
                # Convert numpy array to PIL Image
                if len(image.shape) == 2:  # Grayscale
                    pil_image = Image.fromarray(image, mode='L')
                elif len(image.shape) == 3:
                    if image.shape[2] == 3:  # RGB
                        pil_image = Image.fromarray(image, mode='RGB')
                    elif image.shape[2] == 4:  # RGBA
                        pil_image = Image.fromarray(image, mode='RGBA')
                    else:
                        logger.error(f"Unsupported array shape: {image.shape}")
                        return None
                else:
                    logger.error(f"Unsupported array dimensions: {len(image.shape)}")
                    return None
            elif hasattr(image, 'size'):
                # Already PIL Image
                pil_image = image
            else:
                logger.error(f"Unsupported image type: {type(image)}")
                return None
            
            # NOW safe to call .size
            img_width, img_height = pil_image.size
            
            # SAFE coordinate extraction - handle all BoundingBox formats
            try:
                if bbox.format == BoundingBoxFormat.XYWH:
                    coords = bbox.coordinates
                    # Manual unpacking to avoid errors
                    if len(coords) == 4:
                        x, y, w, h = float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3])
                        x1, y1, x2, y2 = x, y, x + w, y + h
                    else:
                        logger.error(f"Invalid XYWH coordinates length: {len(coords)}")
                        return None
                elif bbox.format == BoundingBoxFormat.XYXY:
                    coords = bbox.coordinates
                    if len(coords) == 4:
                        x1, y1, x2, y2 = float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3])
                    else:
                        logger.error(f"Invalid XYXY coordinates length: {len(coords)}")
                        return None
                else:
                    # Try the built-in method for other formats
                    try:
                        x1, y1, x2, y2 = bbox.to_xyxy()
                    except Exception as e:
                        logger.error(f"bbox.to_xyxy() failed: {e}")
                        return None
                    
            except Exception as e:
                logger.error(f"Coordinate conversion failed: {e}")
                logger.error(f"bbox.coordinates: {bbox.coordinates}")
                logger.error(f"bbox.format: {bbox.format}")
                return None
            
            # Convert to integers and clamp to image bounds
            x1 = max(0, min(int(x1), img_width - 1))
            y1 = max(0, min(int(y1), img_height - 1))
            x2 = max(x1 + 1, min(int(x2), img_width))
            y2 = max(y1 + 1, min(int(y2), img_height))
            
            # Validate region size
            if x2 <= x1 or y2 <= y1:
                logger.debug(f"Invalid region bounds: ({x1},{y1},{x2},{y2})")
                return None
            
            # Extract region
            region_img = pil_image.crop((x1, y1, x2, y2))
            
            # Check minimum size
            if region_img.width < 8 or region_img.height < 8:
                logger.debug(f"Region too small: {region_img.width}x{region_img.height}")
                return None
            
            logger.debug(f"Successfully extracted region: {region_img.width}x{region_img.height}")
            return region_img
            
        except Exception as e:
            logger.error(f"Region extraction failed for bbox {bbox}: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return None
    
    def _clean_extracted_text(self, text: str) -> str:
        """Clean extracted text using project's text utilities"""
        if not text:
            return ""
        
        try:
            # Use project's TextCleaner for basic cleaning
            cleaned = self.text_cleaner.clean_text(text)
            
            # Use project's UnicodeNormalizer
            normalized = self.unicode_normalizer.normalize_unicode(cleaned)
            
            return normalized.strip()
        except Exception as e:
            logger.warning(f"Text cleaning failed: {e}, returning original text")
            return text.strip()
    
    def _estimate_confidence(self, text: str, bbox: BoundingBox) -> float:
        """
        Estimate confidence for TrOCR results
        
        Since TrOCR doesn't provide direct confidence scores, estimate based on:
        - Text characteristics
        - Region properties
        - Content patterns
        """
        if not text.strip():
            return 0.0
        
        base_confidence = 0.75  # TrOCR is generally reliable
        
        # Text length adjustment
        text_len = len(text.strip())
        if text_len >= 3:
            base_confidence += 0.1
        elif text_len == 1:
            base_confidence -= 0.2
        
        # Character variety (real text vs noise)
        unique_chars = len(set(text.lower().replace(' ', '')))
        if unique_chars >= 3:
            base_confidence += 0.05
        
        # Alphanumeric ratio
        alphanumeric_chars = sum(1 for c in text if c.isalnum())
        if alphanumeric_chars > 0:
            alphanumeric_ratio = alphanumeric_chars / len(text)
            base_confidence += alphanumeric_ratio * 0.1
        
        # Region size consideration (safely handle different BoundingBox formats)
        try:
            if bbox.format == BoundingBoxFormat.XYWH:
                coords = bbox.coordinates
                w, h = coords[2], coords[3]
            elif bbox.format == BoundingBoxFormat.XYXY:
                coords = bbox.coordinates
                w = coords[2] - coords[0]
                h = coords[3] - coords[1]
            else:
                x, y, w, h = bbox.to_xywh()
                
            region_area = w * h
            
            if region_area < 50:  # Very small regions often noise
                base_confidence -= 0.3
            elif region_area > 500:  # Larger regions usually better
                base_confidence += 0.05
                
        except Exception:
            pass  # Skip region size adjustment if format is unclear
        
        return max(0.0, min(1.0, base_confidence))
    
    def _combine_region_texts(self, regions: List[TextRegion]) -> str:
        """FIXED: Combine text from TextRegion objects maintaining reading order"""
        if not regions:
            return ""
        
        # Sort regions by reading order (top to bottom, left to right)
        try:
            # Safe sorting with fallback
            sorted_regions = []
            for region in regions:
                try:
                    # Get y1 coordinate safely
                    if region.bbox.format == BoundingBoxFormat.XYXY:
                        y1 = region.bbox.coordinates[1]
                        x1 = region.bbox.coordinates[0]
                    elif region.bbox.format == BoundingBoxFormat.XYWH:
                        y1 = region.bbox.coordinates[1] 
                        x1 = region.bbox.coordinates[0]
                    else:
                        x1, y1, x2, y2 = region.bbox.to_xyxy()
                    
                    sorted_regions.append((region, y1, x1))
                except Exception:
                    sorted_regions.append((region, 0, 0))  # Fallback position
            
            # Sort by y1, then x1
            sorted_regions.sort(key=lambda item: (item[1], item[2]))
            regions = [item[0] for item in sorted_regions]
            
        except Exception:
            # Fallback if sorting fails completely
            pass
        
        combined_parts = []
        prev_y = None
        
        for region in regions:
            try:
                # Get current y coordinate safely
                if region.bbox.format == BoundingBoxFormat.XYXY:
                    current_y = region.bbox.coordinates[1]
                elif region.bbox.format == BoundingBoxFormat.XYWH:
                    current_y = region.bbox.coordinates[1]
                else:
                    x1, current_y, x2, y2 = region.bbox.to_xyxy()
                
                # Add line break for significant vertical separation
                if prev_y is not None and current_y - prev_y > 15:
                    combined_parts.append('\n')
                elif combined_parts and not combined_parts[-1].endswith('\n'):
                    combined_parts.append(' ')
                
                combined_parts.append(region.text)
                prev_y = current_y
                
            except Exception:
                # Just add text if position handling fails
                if combined_parts:
                    combined_parts.append(' ')
                combined_parts.append(region.text)
        
        return ''.join(combined_parts).strip()
    
    def get_engine_status(self) -> Dict[str, Any]:
        """Get current engine status and configuration"""
        return {
            'engine_name': self.engine_name,
            'model_name': self.model_name,
            'device': self.device,
            'is_loaded': self.model is not None,
            'batch_size': self.batch_size,
            'specialization': 'handwritten_text'
        }