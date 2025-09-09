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
    Modern TrOCR Engine - AI-Style Pipeline Architecture
    
    Clean separation of concerns:
    - Takes preprocessed images from preprocessing pipeline
    - Performs pure OCR extraction with TrOCR transformer models
    - Returns structured results for postprocessing pipeline
    - Excellent for handwritten text and transformer-based recognition
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("TrOCR", config)
        self.model = None
        self.processor = None
        self.device = self.config.get("device", "cpu")
        self.model_name = self.config.get("model_name", "microsoft/trocr-base-printed")
        
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
            
            self.logger.info("TrOCR initialized successfully")
            return True
            
        except ImportError as e:
            self.logger.error(f"TrOCR dependencies not installed: {e}")
            self.logger.info("Install with: pip install transformers torch torchvision")
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
        # TrOCR models support multiple languages but primarily trained on English
        if "multilingual" in self.model_name.lower():
            return ['en', 'de', 'fr', 'it', 'pt', 'es', 'nl', 'ru', 'ja', 'ko', 'zh', 'ar']
        else:
            return ['en', 'de', 'fr', 'it', 'pt']
        
    def process_image(self, preprocessed_image: np.ndarray, **kwargs) -> List[OCRResult]:
        """
        Process preprocessed image with TrOCR
        
        Args:
            preprocessed_image: Image from preprocessing pipeline
            **kwargs: Additional parameters
            
        Returns:
            List[OCRResult]: Raw OCR results for postprocessing
        """
        start_time = time.time()
        
        try:
            # Validate initialization
            if not self.is_initialized or self.model is None:
                if not self.initialize():
                    raise RuntimeError("TrOCR engine not initialized")
            
            # Validate preprocessed input
            if not self.validate_image(preprocessed_image):
                raise ValueError("Invalid preprocessed image")
            
            # Convert preprocessed image to TrOCR format
            pil_image = self._prepare_for_trocr(preprocessed_image)
            
            # Extract OCR data with transformer model
            text, confidence = self._extract_text_with_confidence(pil_image)
            
            # Create OCRResult
            result = self._create_ocr_result(text, confidence, preprocessed_image.shape)
            
            # Update processing stats
            processing_time = time.time() - start_time
            self.processing_stats['total_processed'] += 1
            self.processing_stats['total_time'] += processing_time
            
            return [result] if result else []
            
        except Exception as e:
            self.logger.error(f"TrOCR extraction failed: {e}")
            self.processing_stats['errors'] += 1
            return []
    
    def _prepare_for_trocr(self, image: np.ndarray) -> Image.Image:
        """
        Minimal conversion for TrOCR compatibility
        Only format conversion, no enhancement (done by preprocessing pipeline)
        """
        # Convert to PIL Image
        if len(image.shape) == 3:
            if image.shape[2] == 3:
                # Assume BGR from OpenCV, convert to RGB
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                return Image.fromarray(rgb_image)
            else:
                return Image.fromarray(image)
        else:
            # Convert grayscale to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            return Image.fromarray(rgb_image)
    
    def _extract_text_with_confidence(self, pil_image: Image.Image) -> Tuple[str, float]:
        """
        Extract text and confidence using TrOCR transformer model
        Minimal processing - just format conversion
        """
        try:
            import torch
            
            # Process image with TrOCR processor
            pixel_values = self.processor(pil_image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(self.device)
            
            # Generate text with confidence scoring
            with torch.no_grad():
                generated_ids = self.model.generate(
                    pixel_values,
                    max_length=384,
                    num_beams=4,
                    early_stopping=True,
                    do_sample=False,
                    return_dict_in_generate=True,
                    output_scores=True
                )
            
            # Decode generated text
            text = self.processor.batch_decode(generated_ids.sequences, skip_special_tokens=True)[0]
            
            # Calculate confidence from generation scores
            confidence = self._calculate_confidence(generated_ids)
            
            # Clean text
            text = text.strip()
            
            return text, confidence
            
        except Exception as e:
            self.logger.error(f"TrOCR text extraction failed: {e}")
            return "", 0.0
    
    def _calculate_confidence(self, generated_output) -> float:
        """Calculate confidence score from TrOCR generation output"""
        try:
            import torch
            
            # Use sequence scores if available
            if hasattr(generated_output, 'sequences_scores') and generated_output.sequences_scores is not None:
                score = torch.softmax(generated_output.sequences_scores, dim=0)[0].item()
                return min(score, 0.95)  # Cap at 0.95
            
            # Alternative: use average token scores
            if hasattr(generated_output, 'scores') and generated_output.scores:
                token_probs = []
                for score in generated_output.scores:
                    if score.shape[0] > 0:
                        token_prob = torch.softmax(score[0], dim=0).max().item()
                        token_probs.append(token_prob)
                
                if token_probs:
                    avg_prob = sum(token_probs) / len(token_probs)
                    return min(avg_prob, 0.95)
            
            # Default confidence for TrOCR
            return 0.75
            
        except Exception as e:
            self.logger.warning(f"Confidence calculation failed: {e}")
            return 0.70
    
    def _create_ocr_result(self, text: str, confidence: float, image_shape: Tuple) -> OCRResult:
        """Create structured OCRResult from TrOCR output"""
        if not text.strip():
            return None
        
        # Create bounding box for full image
        height, width = image_shape[:2]
        bbox = BoundingBox(
            x=0, y=0, width=width, height=height, 
            confidence=confidence
        )
        
        result = OCRResult(
            text=text.strip(),
            confidence=confidence,
            bbox=bbox,
            level="page",
            metadata={
                'detection_method': 'trocr',
                'model_name': self.model_name,
                'device': self.device,
                'transformer_based': True,
                'supports_handwriting': self.supports_handwriting
            }
        )
        
        return result
    
    def batch_process(self, images: List[np.ndarray], **kwargs) -> List[List[OCRResult]]:
        """Process multiple images efficiently"""
        results = []
        for image in images:
            image_results = self.process_image(image, **kwargs)
            results.append(image_results)
        return results
    
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
                'speed': 'medium',
                'memory_usage': 'high',
                'gpu_required': False,
                'gpu_recommended': True
            },
            'model_info': {
                'model_name': self.model_name,
                'architecture': 'Vision Encoder-Decoder',
                'transformer_based': True,
                'device': self.device,
                'supports_handwriting': self.supports_handwriting
            }
        }
        
        try:
            # Try to get transformers version
            import transformers
            info['version'] = transformers.__version__
        except:
            info['version'] = 'unknown'
            
        return info