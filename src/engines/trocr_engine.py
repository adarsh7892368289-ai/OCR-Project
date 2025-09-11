import os
import cv2
import numpy as np
import time
import warnings
from typing import List, Dict, Any, Tuple, Optional
from PIL import Image

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from ..core.base_engine import (
    BaseOCREngine, 
    OCRResult, 
    BoundingBox, 
    TextRegion, 
    DocumentResult, 
    TextType
)

class TrOCREngine(BaseOCREngine):
    """
    FIXED TrOCR Engine - Returns single OCRResult compatible with base engine
    
    Key Optimizations:
    - Intelligent image resizing before processing
    - Early stopping for sufficient text extraction
    - Reduced segmentation overhead
    - FIXED: Returns single OCRResult (not List[OCRResult])
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("TrOCR", config)
        self.model = None
        self.processor = None
        self.device = self.config.get("device", "cpu")
        self.model_name = self.config.get("model_name", "microsoft/trocr-base-printed")
        
        # Performance optimization settings
        self.max_image_dimension = self.config.get("max_dimension", 800)
        self.enable_segmentation = self.config.get("enable_segmentation", True)
        self.max_strips = self.config.get("max_strips", 3)
        self.min_confidence = self.config.get("min_confidence", 0.4)
        
        # Modern engine capabilities
        self.supports_handwriting = True
        self.supports_multiple_languages = True
        self.supports_orientation_detection = False
        self.supports_structure_analysis = True
        
    def initialize(self) -> bool:
        """Initialize TrOCR with robust error handling"""
        try:
            self.logger.info(f"Initializing TrOCR with model: {self.model_name}")
            
            # Import TrOCR components
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            import torch
            
            # Initialize processor and model
            self.processor = TrOCRProcessor.from_pretrained(self.model_name)
            self.model = VisionEncoderDecoderModel.from_pretrained(self.model_name)
            
            # Set device
            if self.device == "cuda" and torch.cuda.is_available():
                self.model = self.model.to("cuda")
                self.logger.info("TrOCR using CUDA")
            else:
                self.model = self.model.to("cpu")
                self.device = "cpu"
                self.logger.info("TrOCR using CPU")
            
            self.model.eval()  # Set to evaluation mode
            
            self.supported_languages = self.get_supported_languages()
            self.is_initialized = True
            self.model_loaded = True
            
            self.logger.info(f"TrOCR initialized (max_dim: {self.max_image_dimension}, max_strips: {self.max_strips})")
            return True
            
        except ImportError as e:
            self.logger.error(f"TrOCR dependencies not installed: {e}")
            self.logger.info("Install with: pip install transformers torch torchvision")
            self.is_initialized = False
            self.model_loaded = False
            return False
        except Exception as e:
            self.logger.error(f"TrOCR initialization failed: {e}")
            self.model = None
            self.processor = None
            self.is_initialized = False
            self.model_loaded = False
            return False
            
    def get_supported_languages(self) -> List[str]:
        """Get supported languages based on model"""
        if "multilingual" in self.model_name.lower():
            return ['en', 'de', 'fr', 'it', 'pt', 'es', 'nl', 'ru', 'ja', 'ko', 'zh', 'ar']
        else:
            return ['en', 'de', 'fr', 'it', 'pt']
        
    def process_image(self, preprocessed_image: np.ndarray, **kwargs) -> OCRResult:
        """
        FIXED: Process preprocessed image with TrOCR - Returns single OCRResult
        
        Args:
            preprocessed_image: Preprocessed image from image enhancement
            **kwargs: Additional parameters
            
        Returns:
            OCRResult: Single OCR result compatible with base engine
        """
        start_time = time.time()
        
        try:
            # Validate initialization
            if not self.is_initialized or self.model is None:
                if not self.initialize():
                    raise RuntimeError("TrOCR engine failed to initialize")
            
            # Validate preprocessed input from pipeline
            if not self.validate_image(preprocessed_image):
                raise ValueError("Invalid preprocessed image from pipeline")
            
            original_shape = preprocessed_image.shape
            self.logger.info(f"Processing preprocessed image: {original_shape}")
            
            # OPTIMIZATION 1: Resize if too large (major speedup)
            optimized_image = self._optimize_for_speed(preprocessed_image)
            
            # OPTIMIZATION 2: Try fast full-image extraction first
            text, confidence = self._fast_extract_text(optimized_image)
            
            # OPTIMIZATION 3: Only use segmentation if needed and enabled
            if (not text.strip() or len(text.strip()) < 10 or confidence < self.min_confidence) and self.enable_segmentation:
                self.logger.info("Full image extraction insufficient, using limited segmentation")
                text, confidence = self._limited_segmentation(optimized_image)
            
            # Create single OCRResult - FIXED!
            result = self._create_pipeline_result(text, confidence, original_shape)
            
            # Update processing stats
            processing_time = time.time() - start_time
            if result:
                result.processing_time = processing_time
                result.engine_name = self.name
                
                self.processing_stats['total_processed'] += 1
                self.processing_stats['total_time'] += processing_time
                
                if result.text.strip():
                    self.processing_stats['successful_extractions'] += 1
                
                self.logger.info(f"TrOCR extracted {len(text)} chars (conf: {confidence:.3f}) in {processing_time:.2f}s")
            else:
                # Return empty result instead of None
                result = OCRResult(
                    text="",
                    confidence=0.0,
                    processing_time=processing_time,
                    engine_name=self.name,
                    metadata={"no_text_found": True}
                )
                self.logger.warning(f"TrOCR: No meaningful text extracted in {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.logger.error(f"TrOCR extraction failed after {processing_time:.2f}s: {e}")
            self.processing_stats['errors'] += 1
            
            # Return empty result instead of raising
            return OCRResult(
                text="",
                confidence=0.0,
                processing_time=processing_time,
                engine_name=self.name,
                metadata={"error": str(e)}
            )
    
    def _optimize_for_speed(self, preprocessed_image: np.ndarray) -> np.ndarray:
        """
        CRITICAL OPTIMIZATION: Resize large images for 5-10x speedup
        """
        h, w = preprocessed_image.shape[:2]
        max_dim = max(h, w)
        
        # Only resize if significantly larger than target
        if max_dim > self.max_image_dimension:
            scale = self.max_image_dimension / max_dim
            new_h, new_w = int(h * scale), int(w * scale)
            
            # Use INTER_AREA for downscaling (better quality)
            optimized = cv2.resize(preprocessed_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            self.logger.info(f"Resized {w}x{h} → {new_w}x{new_h} for {scale:.2f}x speedup")
            return optimized
        
        return preprocessed_image
    
    def _fast_extract_text(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Fast single-pass text extraction
        """
        try:
            # Convert to PIL for TrOCR
            pil_image = self._prepare_for_trocr(image)
            
            # Extract text with optimized settings
            return self._extract_text_with_confidence(pil_image, fast_mode=True)
            
        except Exception as e:
            self.logger.error(f"Fast text extraction failed: {e}")
            return "", 0.0
    
    def _limited_segmentation(self, image: np.ndarray) -> Tuple[str, float]:
        """
        OPTIMIZED: Limited segmentation for speed
        """
        try:
            h, w = image.shape[:2]
            
            # OPTIMIZATION: Use larger strips, fewer iterations
            strip_height = h // self.max_strips
            if strip_height < 50:
                strip_height = h // 2  # Fallback to half-image strips
            
            all_text = []
            all_confidences = []
            strips_processed = 0
            
            for y in range(0, h, strip_height):
                if strips_processed >= self.max_strips:
                    break
                
                y_end = min(y + strip_height, h)
                strip = image[y:y_end, :]
                
                # Skip tiny strips
                if strip.shape[0] < 30 or strip.shape[1] < 50:
                    continue
                
                # Quick content check
                if self._has_text_content(strip):
                    try:
                        pil_strip = self._prepare_for_trocr(strip)
                        text, conf = self._extract_text_with_confidence(pil_strip, fast_mode=True)
                        
                        if text.strip() and len(text.strip()) > 2 and conf > self.min_confidence:
                            all_text.append(text.strip())
                            all_confidences.append(conf)
                            strips_processed += 1
                            
                            # OPTIMIZATION: Early stopping if we have enough text
                            if len(" ".join(all_text)) > 100:
                                break
                                
                    except Exception as e:
                        self.logger.debug(f"Strip processing failed: {e}")
                        continue
            
            # Combine results
            if all_text:
                combined_text = " ".join(all_text)
                avg_confidence = sum(all_confidences) / len(all_confidences)
                
                self.logger.info(f"Limited segmentation: {len(all_text)} strips, {len(combined_text)} chars")
                return combined_text, avg_confidence
            else:
                return "", 0.0
                
        except Exception as e:
            self.logger.error(f"Limited segmentation failed: {e}")
            return "", 0.0
    
    def _prepare_for_trocr(self, image: np.ndarray) -> Image.Image:
        """Convert preprocessed image to PIL format for TrOCR"""
        try:
            # Convert to PIL Image (preprocessed image should be RGB already)
            if len(image.shape) == 3:
                if image.shape[2] == 3:
                    # Preprocessed images are typically RGB
                    return Image.fromarray(image)
                else:
                    return Image.fromarray(image)
            else:
                # Convert grayscale to RGB
                rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                return Image.fromarray(rgb_image)
        except Exception as e:
            self.logger.error(f"Image preparation for TrOCR failed: {e}")
            raise
    
    def _extract_text_with_confidence(self, pil_image: Image.Image, fast_mode: bool = False) -> Tuple[str, float]:
        """Extract text with TrOCR - optimized for speed"""
        try:
            import torch
            
            # Process image with TrOCR processor
            inputs = self.processor(images=pil_image, return_tensors="pt")
            pixel_values = inputs.pixel_values.to(self.device)
            
            # OPTIMIZATION: Reduced beam search for speed
            max_length = 200 if fast_mode else 384
            num_beams = 2 if fast_mode else 4
            
            # Generate text
            with torch.no_grad():
                generated_ids = self.model.generate(
                    pixel_values,
                    max_length=max_length,
                    num_beams=num_beams,
                    early_stopping=True,
                    do_sample=False,
                    return_dict_in_generate=True,
                    output_scores=True
                )
            
            # Decode generated text
            text = self.processor.batch_decode(generated_ids.sequences, skip_special_tokens=True)[0]
            
            # Calculate confidence
            confidence = self._calculate_confidence(generated_ids)
            
            return text.strip(), confidence
            
        except Exception as e:
            self.logger.error(f"TrOCR text extraction failed: {e}")
            return "", 0.0
    
    def _has_text_content(self, strip: np.ndarray) -> bool:
        """Quick check if image strip likely contains text"""
        try:
            if len(strip.shape) == 3:
                gray = cv2.cvtColor(strip, cv2.COLOR_RGB2GRAY)
            else:
                gray = strip
            
            # Quick statistical checks
            mean_val = np.mean(gray)
            std_val = np.std(gray)
            
            # Skip if too uniform or extreme values
            return not (mean_val > 250 or mean_val < 10 or std_val < 15)
            
        except Exception:
            return True
    
    def _calculate_confidence(self, generated_output) -> float:
        """Calculate confidence score from TrOCR generation output"""
        try:
            import torch
            
            if hasattr(generated_output, 'sequences_scores') and generated_output.sequences_scores is not None:
                score = torch.softmax(generated_output.sequences_scores, dim=0)[0].item()
                return min(score, 0.95)
            
            if hasattr(generated_output, 'scores') and generated_output.scores:
                token_probs = []
                for score in generated_output.scores[:5]:  # Only check first 5 tokens for speed
                    if score.shape[0] > 0:
                        token_prob = torch.softmax(score[0], dim=0).max().item()
                        token_probs.append(token_prob)
                
                if token_probs:
                    return min(sum(token_probs) / len(token_probs), 0.95)
            
            return 0.75  # Default confidence for TrOCR
            
        except Exception:
            return 0.70
    
    def _create_pipeline_result(self, text: str, confidence: float, original_shape: Tuple) -> Optional[OCRResult]:
        """Create OCRResult compatible with postprocessing pipeline"""
        if not text.strip():
            return None
        
        try:
            # Create bounding box for original image dimensions
            height, width = original_shape[:2]
            bbox = BoundingBox(
                x=0, y=0, width=width, height=height, 
                confidence=confidence,
                text_type=TextType.PRINTED
            )
            
            # Create text region
            region = TextRegion(
                text=text.strip(),
                confidence=confidence,
                bbox=bbox,
                text_type=TextType.PRINTED,
                language="en"
            )
            
            result = OCRResult(
                text=text.strip(),
                confidence=confidence,
                regions=[region],
                bbox=bbox,
                level="page",
                engine_name=self.name,
                text_type=TextType.PRINTED,
                metadata={
                    'detection_method': 'trocr',
                    'model_name': self.model_name,
                    'device': self.device,
                    'transformer_based': True,
                    'supports_handwriting': self.supports_handwriting,
                    'engine_name': 'TrOCR',
                    'optimized': True,
                    'max_dimension': self.max_image_dimension,
                    'segmentation_used': self.enable_segmentation
                }
            )
            
            return result
        except Exception as e:
            self.logger.error(f"Pipeline result creation failed: {e}")
            return None
    
    def batch_process(self, images: List[np.ndarray], **kwargs) -> List[OCRResult]:
        """Process multiple preprocessed images efficiently"""
        results = []
        for i, image in enumerate(images):
            self.logger.info(f"Processing image {i+1}/{len(images)} with optimized TrOCR")
            image_result = self.process_image(image, **kwargs)
            results.append(image_result)
        return results
    
    def validate_image(self, image: np.ndarray) -> bool:
        """Validate preprocessed image from pipeline"""
        try:
            if image is None or not isinstance(image, np.ndarray):
                return False
            if len(image.shape) not in [2, 3]:
                return False
            if image.shape[0] == 0 or image.shape[1] == 0:
                return False
            return True
        except:
            return False
    
    def get_engine_info(self) -> Dict[str, Any]:
        """Get comprehensive engine information"""
        info = {
            'name': self.name,
            'version': None,
            'type': 'transformer_ocr',
            'supported_languages': self.get_supported_languages(),
            'capabilities': {
                'handwriting': self.supports_handwriting,
                'multiple_languages': self.supports_multiple_languages,
                'orientation_detection': self.supports_orientation_detection,
                'structure_analysis': self.supports_structure_analysis
            },
            'optimal_for': ['handwritten_text', 'printed_text', 'mixed_text', 'documents', 'forms'],
            'performance_profile': {
                'accuracy': 'very_high',
                'speed': 'optimized',
                'memory_usage': 'medium',
                'gpu_required': False,
                'gpu_recommended': True
            },
            'optimization_settings': {
                'max_dimension': self.max_image_dimension,
                'enable_segmentation': self.enable_segmentation,
                'max_strips': self.max_strips,
                'min_confidence': self.min_confidence
            },
            'model_info': {
                'model_name': self.model_name,
                'architecture': 'Vision Encoder-Decoder',
                'transformer_based': True,
                'device': self.device,
                'supports_handwriting': self.supports_handwriting
            },
            'pipeline_integration': {
                'uses_preprocessing_pipeline': True,
                'returns_single_result': True,
                'compatible_with_base_engine': True,
                'works_with_engine_manager': True
            }
        }
        
        try:
            import transformers
            info['version'] = transformers.__version__
        except:
            info['version'] = 'unknown'
            
        return info