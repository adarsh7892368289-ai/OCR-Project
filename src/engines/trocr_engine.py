from PIL import Image
import torch
from typing import List, Dict, Any
import logging
import numpy as np
import cv2

class TrOCREngine:
    """TrOCR engine for handwritten text recognition with improved processing"""
    
    def __init__(self, use_gpu: bool = True, config: Dict = None):
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = "cuda" if self.use_gpu else "cpu"
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        self.processor = None
        self.model = None
        self._initialize()
    
    def _initialize(self):
        """Initialize TrOCR model"""
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            
            model_name = self.config.get('model_name', 'microsoft/trocr-base-handwritten')
            
            self.processor = TrOCRProcessor.from_pretrained(model_name)
            self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
            
            if self.use_gpu:
                self.model = self.model.to(self.device)
                
        except ImportError:
            raise ImportError("Transformers not installed. Run: pip install transformers torch")
        except Exception as e:
            self.logger.error(f"TrOCR initialization failed: {e}")
            raise
    
    def extract_text(self, image) -> List[Dict[str, Any]]:
        """Extract text using TrOCR with better preprocessing"""
        
        if self.processor is None or self.model is None:
            return []
        
        try:
            # Convert numpy array to PIL Image if needed
            if isinstance(image, np.ndarray):
                if len(image.shape) == 3:
                    # Convert BGR to RGB
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                elif len(image.shape) == 2:
                    # Convert grayscale to RGB
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                pil_image = Image.fromarray(image)
            else:
                pil_image = image
            
            # Ensure proper size and format
            pil_image = pil_image.convert('RGB')
            
            # Process with TrOCR
            pixel_values = self.processor(pil_image, return_tensors="pt").pixel_values
            
            if self.use_gpu:
                pixel_values = pixel_values.to(self.device)
            
            # Generate text
            with torch.no_grad():
                generated_ids = self.model.generate(
                    pixel_values, 
                    max_length=100,
                    num_beams=4,
                    early_stopping=True
                )
                generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            if generated_text.strip():
                confidence = 85.0  # TrOCR doesn't provide confidence scores, use default
                self.logger.info(f"TrOCR extracted: '{generated_text[:50]}...'")
                
                return [{
                    'text': generated_text.strip(),
                    'confidence': confidence,
                    'bbox': [],
                    'engine': 'TROCR'
                }]
            else:
                return []
                
        except Exception as e:
            self.logger.error(f"TrOCR extraction failed: {e}")
            return []
