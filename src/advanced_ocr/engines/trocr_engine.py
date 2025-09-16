# src/advanced_ocr/engines/trocr_engine.py
"""
Advanced OCR TrOCR Engine

This module provides the TrOCR (Transformer-based Optical Character Recognition)
engine implementation for the advanced OCR system. TrOCR combines vision transformer
encoders with text transformer decoders, making it particularly effective for
handwritten text, degraded documents, and challenging fonts.

The module focuses on:
- Handwritten text recognition with high accuracy
- Degraded document processing capabilities
- Transformer-based architecture for complex text patterns
- Multi-language support through transformer models
- Region-based and full-image OCR processing
- Confidence estimation and text validation

Classes:
    TrOCREngine: TrOCR-based OCR engine implementation

Functions:
    _extract_implementation: Core OCR extraction logic
    _process_batch: Batch processing for efficiency
    _estimate_confidence: Confidence score estimation
    _combine_region_text: Text combination from multiple regions

Example:
    >>> engine = TrOCREngine(config)
    >>> engine.initialize()
    >>> result = engine.extract(image, text_regions)
    >>> print(f"Extracted text: {result.text}")

"""


import logging
import numpy as np
import torch
from PIL import Image
from typing import List, Optional, Dict, Any
import cv2
import warnings

from ..engines.base_engine import BaseOCREngine
from ..results import OCRResult, TextRegion, BoundingBox, ConfidenceMetrics
from ..utils.image_utils import ImageProcessor, CoordinateTransformer
from ..utils.text_utils import TextCleaner, UnicodeNormalizer
from ..utils.model_utils import ModelCache, cached_model_load
from ..config import OCRConfig, EngineConfig

# Suppress transformer warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='transformers')

logger = logging.getLogger(__name__)


class TrOCREngine(BaseOCREngine):
    """
    TrOCR-based OCR engine optimized for handwritten and challenging text.
    
    TrOCR combines a vision transformer encoder with a text transformer decoder,
    making it particularly effective for:
    - Handwritten text recognition
    - Degraded or low-quality documents  
    - Unusual fonts and artistic text
    - Documents with complex layouts
    
    The engine processes preprocessed images and text regions provided by
    the preprocessing pipeline, focusing solely on text extraction.
    """
    
    ENGINE_NAME = "trocr"
    
    # Default model configurations - can be overridden in config
    DEFAULT_MODELS = {
        'handwritten': 'microsoft/trocr-base-handwritten',
        'printed': 'microsoft/trocr-base-printed',
        'large': 'microsoft/trocr-large-printed'
    }
    
    def __init__(self, config: OCRConfig):
        """
        Initialize TrOCR engine with configuration.
        
        Args:
            config: OCR configuration containing TrOCR-specific settings
        """
        super().__init__(config)
        
        # Extract TrOCR-specific configuration
        self.engine_config: EngineConfig = config.engines.get('trocr', EngineConfig())
        
        # Model selection based on configuration or defaults
        self.model_name = self.engine_config.model_path or self.DEFAULT_MODELS['handwritten']
        self.device = self._determine_device()
        
        # Initialize model components (lazy loading via ModelCache)
        self.processor = None
        self.model = None
        
        # Processing utilities
        self.image_processor = ImageProcessor()
        self.text_cleaner = TextCleaner()
        self.unicode_normalizer = UnicodeNormalizer()
        self.coord_transformer = CoordinateTransformer()
        
        # Performance settings
        self.batch_size = self.engine_config.batch_size or 4
        self.max_length = self.engine_config.max_length or 256
        self.confidence_threshold = self.engine_config.confidence_threshold or 0.5
        
        logger.info(f"TrOCR engine initialized with model: {self.model_name}")
    
    def _determine_device(self) -> str:
        """
        Determine optimal device for TrOCR processing.
        
        Returns:
            Device string ('cuda', 'mps', or 'cpu')
        """
        if self.engine_config.force_cpu:
            return 'cpu'
            
        if torch.cuda.is_available():
            return 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    
    @cached_model_load("trocr", "transformers")
    def _load_model(self):
        """
        Load TrOCR processor and model using your ModelCache system.
        
        The @cached_model_load decorator from model_utils.py handles:
        - Model caching to prevent repeated loading
        - Memory management and cleanup
        - Version compatibility checking
        
        Returns:
            Tuple of (processor, model) ready for inference
        """
        logger.info(f"Loading TrOCR model via ModelCache: {self.model_name}")
        
        # The actual transformers import happens here, only when needed
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
        except ImportError as e:
            raise RuntimeError(f"Transformers library required for TrOCR: {e}")
        
        try:
            # Load processor (handles image preprocessing and tokenization)
            processor = TrOCRProcessor.from_pretrained(self.model_name)
            
            # Load model with appropriate configuration
            model = VisionEncoderDecoderModel.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32
            )
            
            # Move model to device and set to evaluation mode
            model = model.to(self.device)
            model.eval()
            
            logger.info(f"TrOCR model loaded successfully on device: {self.device}")
            return processor, model
            
        except Exception as e:
            logger.error(f"Failed to load TrOCR model {self.model_name}: {e}")
            raise RuntimeError(f"TrOCR model loading failed: {e}")
    
    def _ensure_models_loaded(self) -> None:
        """Ensure models are loaded before processing."""
        if self.processor is None or self.model is None:
            self.processor, self.model = self._load_model()
    
    def extract(self, image: np.ndarray, text_regions: List[BoundingBox]) -> OCRResult:
        """
        Extract text from preprocessed image using TrOCR.
        
        This is the main extraction method that processes text regions
        identified by the preprocessing pipeline.
        
        Args:
            image: Preprocessed image as numpy array (from image_processor.py)
            text_regions: List of text bounding boxes (from text_detector.py)
            
        Returns:
            OCRResult containing extracted text and metadata
            
        Note:
            - Image should already be preprocessed (enhanced, noise-reduced, etc.)
            - Text regions should already be detected and filtered
            - This method focuses solely on text extraction from provided regions
        """
        logger.debug(f"TrOCR processing {len(text_regions)} text regions")
        
        # Ensure models are loaded
        self._ensure_models_loaded()
        
        # Convert numpy array to PIL Image for TrOCR processing
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3 and image.shape[2] == 3:
                # BGR to RGB conversion if needed
                image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                image_pil = Image.fromarray(image)
        else:
            image_pil = image
        
        # Extract text from each region
        extracted_regions = []
        processing_metadata = {
            'total_regions': len(text_regions),
            'processed_regions': 0,
            'failed_regions': 0,
            'avg_confidence': 0.0
        }
        
        # Process regions in batches for efficiency
        for batch_start in range(0, len(text_regions), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(text_regions))
            batch_regions = text_regions[batch_start:batch_end]
            
            batch_results = self._process_batch(image_pil, batch_regions)
            extracted_regions.extend(batch_results)
            
            # Update metadata
            processing_metadata['processed_regions'] = len(extracted_regions)
        
        # Calculate overall statistics
        confidences = [region.confidence_metrics.overall_confidence 
                      for region in extracted_regions 
                      if region.confidence_metrics.overall_confidence > 0]
        
        processing_metadata['avg_confidence'] = (
            sum(confidences) / len(confidences) if confidences else 0.0
        )
        processing_metadata['failed_regions'] = (
            len(text_regions) - len(extracted_regions)
        )
        
        # Create overall result
        result = OCRResult(
            text=self._combine_region_text(extracted_regions),
            confidence=processing_metadata['avg_confidence'],
            text_regions=extracted_regions,
            processing_metadata=processing_metadata,
            engine_name=self.ENGINE_NAME
        )
        
        logger.info(f"TrOCR extraction completed: {len(extracted_regions)} regions, "
                   f"avg confidence: {processing_metadata['avg_confidence']:.3f}")
        
        return result
    
    def _process_batch(self, image: Image.Image, regions: List[BoundingBox]) -> List[TextRegion]:
        """
        Process a batch of text regions efficiently.
        
        Args:
            image: PIL Image to extract text from
            regions: List of bounding boxes to process
            
        Returns:
            List of TextRegion objects with extracted text
        """
        batch_results = []
        
        try:
            # Extract region images
            region_images = []
            valid_regions = []
            
            for region in regions:
                try:
                    region_img = self._extract_region_image(image, region)
                    if region_img is not None:
                        region_images.append(region_img)
                        valid_regions.append(region)
                except Exception as e:
                    logger.warning(f"Failed to extract region {region}: {e}")
                    continue
            
            if not region_images:
                return batch_results
            
            # Process batch through TrOCR
            with torch.no_grad():
                # Prepare inputs
                pixel_values = self.processor(
                    images=region_images, 
                    return_tensors="pt"
                ).pixel_values.to(self.device)
                
                # Generate text
                generated_ids = self.model.generate(
                    pixel_values,
                    max_length=self.max_length,
                    num_beams=4,
                    early_stopping=True,
                    do_sample=False
                )
                
                # Decode results
                generated_texts = self.processor.batch_decode(
                    generated_ids, 
                    skip_special_tokens=True
                )
            
            # Create TextRegion objects
            for region, text in zip(valid_regions, generated_texts):
                if text.strip():  # Only include non-empty results
                    # Clean and normalize text
                    cleaned_text = self.text_cleaner.clean_text(text)
                    normalized_text = self.unicode_normalizer.normalize(cleaned_text)
                    
                    # Calculate confidence (TrOCR doesn't provide direct confidence)
                    confidence = self._estimate_confidence(normalized_text, region)
                    
                    # Create text region
                    text_region = TextRegion(
                        text=normalized_text,
                        bounding_box=region,
                        confidence_metrics=ConfidenceMetrics(
                            overall_confidence=confidence,
                            character_confidence=[confidence] * len(normalized_text),
                            word_confidence=[confidence] * len(normalized_text.split())
                        )
                    )
                    
                    batch_results.append(text_region)
                    
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            # Continue with empty results rather than failing completely
            
        return batch_results
    
    def _extract_region_image(self, image: Image.Image, region: BoundingBox) -> Optional[Image.Image]:
        """
        Extract a region image from the full image.
        
        Args:
            image: Source PIL Image
            region: Bounding box coordinates
            
        Returns:
            Cropped PIL Image or None if extraction fails
        """
        try:
            # Ensure coordinates are within image bounds
            img_width, img_height = image.size
            
            # Convert region to standard format and clamp to image bounds
            x1 = max(0, min(region.x1, img_width - 1))
            y1 = max(0, min(region.y1, img_height - 1))
            x2 = max(x1 + 1, min(region.x2, img_width))
            y2 = max(y1 + 1, min(region.y2, img_height))
            
            # Extract region
            region_img = image.crop((x1, y1, x2, y2))
            
            # Ensure minimum size for TrOCR processing
            if region_img.width < 10 or region_img.height < 10:
                return None
                
            return region_img
            
        except Exception as e:
            logger.warning(f"Region extraction failed: {e}")
            return None
    
    def _estimate_confidence(self, text: str, region: BoundingBox) -> float:
        """
        Estimate confidence score for extracted text.
        
        Since TrOCR doesn't provide direct confidence scores, we estimate
        based on text characteristics and region properties.
        
        Args:
            text: Extracted text
            region: Source bounding box
            
        Returns:
            Estimated confidence score (0.0 to 1.0)
        """
        confidence = 0.7  # Base confidence for TrOCR
        
        # Adjust based on text characteristics
        if len(text.strip()) == 0:
            return 0.0
        
        # Length-based adjustment
        if len(text) >= 3:
            confidence += 0.1
        
        # Character variety (indicates real text vs noise)
        unique_chars = len(set(text.lower().replace(' ', '')))
        if unique_chars >= 3:
            confidence += 0.1
        
        # Alphanumeric content
        alphanumeric_ratio = sum(c.isalnum() for c in text) / len(text)
        confidence += alphanumeric_ratio * 0.1
        
        # Region size (very small regions are often noise)
        region_area = (region.x2 - region.x1) * (region.y2 - region.y1)
        if region_area < 100:  # Very small regions
            confidence -= 0.2
        elif region_area > 1000:  # Larger regions often have better text
            confidence += 0.1
        
        return max(0.0, min(1.0, confidence))
    
    def _combine_region_text(self, regions: List[TextRegion]) -> str:
        """
        Combine text from multiple regions into a single string.
        
        Args:
            regions: List of TextRegion objects with extracted text
            
        Returns:
            Combined text string with appropriate spacing
        """
        if not regions:
            return ""
        
        # Sort regions by vertical position, then horizontal
        sorted_regions = sorted(regions, key=lambda r: (r.bounding_box.y1, r.bounding_box.x1))
        
        combined_text = []
        prev_y = None
        
        for region in sorted_regions:
            current_y = region.bounding_box.y1
            
            # Add line break for new lines (significant vertical gap)
            if prev_y is not None and current_y - prev_y > 20:
                combined_text.append('\n')
            elif combined_text and not combined_text[-1].endswith('\n'):
                combined_text.append(' ')
            
            combined_text.append(region.text)
            prev_y = current_y
        
        return ''.join(combined_text).strip()
    
    def get_engine_info(self) -> Dict[str, Any]:
        """
        Get engine information and current configuration.
        
        Returns:
            Dictionary containing engine metadata
        """
        return {
            'engine_name': self.ENGINE_NAME,
            'model_name': self.model_name,
            'device': self.device,
            'batch_size': self.batch_size,
            'max_length': self.max_length,
            'confidence_threshold': self.confidence_threshold,
            'is_loaded': self.model is not None,
            'specialization': 'handwritten_text'
        }
    
    def cleanup(self) -> None:
        """
        Clean up resources - let the @cached_model_load decorator handle caching.
        
        The ModelCache system from model_utils.py manages memory automatically,
        so we just clear local references.
        """
        # Clear local references - ModelCache handles the rest
        self.model = None
        self.processor = None
        
        logger.info("TrOCR engine cleanup completed")