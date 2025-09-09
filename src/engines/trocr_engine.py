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
        # Use a more robust model for printed text
        self.model_name = self.config.get("model_name", "microsoft/trocr-large-printed")
        # Add image resizing parameters
        self.max_image_size = self.config.get("max_image_size", 2240)  # Max dimension for TrOCR
        self.target_height = self.config.get("target_height", 384)  # TrOCR expected height
        self.target_width = self.config.get("target_width", 384)   # TrOCR expected width

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

            # Debug: Log extracted text and confidence
            self.logger.info(f"TrOCR extracted text: {text}")
            self.logger.info(f"TrOCR confidence: {confidence}")

            # Create multiple OCRResult objects from the extracted text
            results = self._create_multiple_ocr_results(text, confidence, preprocessed_image.shape)

            # Update processing stats
            processing_time = time.time() - start_time
            self.processing_stats['total_processed'] += 1
            self.processing_stats['total_time'] += processing_time

            return results

        except Exception as e:
            self.logger.error(f"TrOCR extraction failed: {e}")
            self.processing_stats['errors'] += 1
            return []
    
    def _prepare_for_trocr(self, image: np.ndarray) -> Image.Image:
        """
        Prepare image for TrOCR with proper resizing and format conversion
        """
        # Convert to PIL Image
        if len(image.shape) == 3:
            if image.shape[2] == 3:
                # Assume BGR from OpenCV, convert to RGB
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_image)
            else:
                pil_image = Image.fromarray(image)
        else:
            # Convert grayscale to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            pil_image = Image.fromarray(rgb_image)

        # Resize image if it's too large for TrOCR
        width, height = pil_image.size
        max_dim = max(width, height)

        if max_dim > self.max_image_size:
            # Calculate scaling factor
            scale_factor = self.max_image_size / max_dim
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)

            # Resize image
            pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            self.logger.info(f"Resized image from {width}x{height} to {new_width}x{new_height}")

        return pil_image
    
    def _extract_text_with_confidence(self, pil_image: Image.Image) -> Tuple[str, float]:
        """
        Extract text and confidence using TrOCR transformer model
        Based on working minimal test approach
        """
        try:
            import torch

            # Process image with TrOCR processor
            pixel_values = self.processor(pil_image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(self.device)

            # Generate text with confidence scoring - optimized parameters
            with torch.no_grad():
                generated_output = self.model.generate(
                    pixel_values,
                    max_length=128,  # Reasonable length for documents
                    min_length=1,    # Allow short text
                    num_beams=4,     # Balanced beam search
                    do_sample=False, # Deterministic for consistency
                    return_dict_in_generate=True,
                    output_scores=True
                )

            # Decode generated text - simplified approach
            text = ""
            if hasattr(generated_output, 'sequences') and generated_output.sequences is not None:
                if isinstance(generated_output.sequences, torch.Tensor):
                    # Standard case: tensor with shape [batch_size, seq_len]
                    text = self.processor.batch_decode(generated_output.sequences, skip_special_tokens=True)[0]
                else:
                    self.logger.warning(f"Unexpected sequences type: {type(generated_output.sequences)}")
            else:
                self.logger.warning("No sequences in generation output")

            # Calculate confidence from generation scores
            confidence = self._calculate_confidence(generated_output)

            # Clean text
            text = text.strip()

            return text, confidence

        except Exception as e:
            self.logger.error(f"TrOCR text extraction failed: {e}")
            # Don't print full traceback in production
            return "", 0.0
    
    def _calculate_confidence(self, generated_output) -> float:
        """Calculate confidence score from TrOCR generation output - Simplified version"""
        try:
            import torch

            # Try sequence scores first (most reliable)
            if hasattr(generated_output, 'sequences_scores') and generated_output.sequences_scores is not None:
                if isinstance(generated_output.sequences_scores, torch.Tensor):
                    if generated_output.sequences_scores.numel() > 0:
                        try:
                            # Convert to numpy and get the score
                            score_array = generated_output.sequences_scores.detach().cpu().numpy()
                            if score_array.ndim == 0:
                                # Scalar
                                score_val = float(score_array)
                            elif hasattr(score_array, 'flat') and len(score_array.flat) > 0:
                                # Array with flat attribute
                                score_val = float(score_array.flat[0])
                            elif hasattr(score_array, '__getitem__') and len(score_array) > 0:
                                # Array-like object
                                score_val = float(score_array[0])
                            else:
                                # Fallback
                                score_val = float(score_array)

                            # Convert to probability-like value
                            confidence = 1.0 / (1.0 + abs(score_val))
                            return min(max(confidence, 0.1), 0.95)
                        except Exception as e:
                            self.logger.debug(f"Error processing sequences_scores: {e}")

            # Fallback: try token scores
            if hasattr(generated_output, 'scores') and generated_output.scores:
                try:
                    # Get the last few token scores for average
                    token_scores = []
                    num_scores = min(len(generated_output.scores), 5)  # Last 5 tokens

                    for i in range(len(generated_output.scores) - num_scores, len(generated_output.scores)):
                        score_tensor = generated_output.scores[i]
                        if isinstance(score_tensor, torch.Tensor) and score_tensor.numel() > 0:
                            try:
                                # Convert to numpy and get max probability
                                score_array = score_tensor.detach().cpu().numpy()
                                if hasattr(score_array, 'max'):
                                    max_prob = float(score_array.max())
                                else:
                                    max_prob = float(score_array)
                                token_scores.append(max_prob)
                            except:
                                continue

                    if token_scores:
                        avg_score = sum(token_scores) / len(token_scores)
                        return min(max(avg_score, 0.1), 0.95)

                except Exception as e:
                    self.logger.debug(f"Error processing token scores: {e}")

            # Default confidence
            return 0.75

        except Exception as e:
            self.logger.warning(f"Confidence calculation failed: {e}")
            # Don't print full traceback to avoid spam
            return 0.70
    
    def _create_multiple_ocr_results(self, text: str, confidence: float, image_shape: Tuple) -> List[OCRResult]:
        """Create multiple OCRResult objects by splitting text into words/phrases"""
        if not text.strip():
            return []

        # Split text into words/phrases using various delimiters
        words = []
        temp_text = text.strip()

        # Try splitting by spaces first
        if ' ' in temp_text:
            words = temp_text.split()
        else:
            # If no spaces, split by common patterns or treat as single word
            words = [temp_text]

        # If we have very few words, try to split by other patterns
        if len(words) <= 2:
            # Try splitting by uppercase letters (common in receipts)
            import re
            words = re.findall(r'[A-Z][a-z]*|[A-Z]+(?=[A-Z]|$)', temp_text)
            if len(words) <= 2:
                words = [temp_text]  # Keep as single result

        # Ensure we have at least some results
        if len(words) == 0:
            words = [text.strip()]

        results = []
        height, width = image_shape[:2]
        word_width = width // len(words) if words else width

        for i, word_text in enumerate(words):
            if not word_text.strip():
                continue

            # Calculate bounding box for this word
            x_start = i * word_width
            x_end = min((i + 1) * word_width, width)

            # Create BoundingBox with proper integer values
            bbox = BoundingBox(
                x=int(x_start),
                y=int(0),
                width=int(x_end - x_start),
                height=int(height),
                confidence=float(confidence)
            )

            # Adjust confidence slightly for each word
            word_confidence = confidence * (0.8 + 0.2 * (len(word_text.strip()) / max(len(text), 1)))

            result = OCRResult(
                text=word_text.strip(),
                confidence=min(float(word_confidence), 0.95),
                bbox=bbox,
                level="word",
                metadata={
                    'detection_method': 'trocr',
                    'model_name': self.model_name,
                    'device': self.device,
                    'transformer_based': True,
                    'supports_handwriting': self.supports_handwriting,
                    'word_number': i + 1,
                    'total_words': len(words),
                    'original_text': text.strip()
                }
            )

            results.append(result)

        return results

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