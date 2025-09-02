from PIL import Image
import torch
from typing import List, Dict, Any
import logging
import numpy as np
import cv2
import os

class TrOCREngine:
    """TrOCR engine for handwritten text recognition with improved processing"""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.use_gpu = torch.cuda.is_available()
        self.device = "cuda" if self.use_gpu else "cpu"
        
        self.processor = None
        self.model = None
        self._initialize()
    
    def _initialize(self):
        """Initialize TrOCR model"""
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            
            # Use the handwritten model as it's the most versatile for your case
            model_name = 'microsoft/trocr-base-handwritten'
            
            self.processor = TrOCRProcessor.from_pretrained(model_name)
            self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
            
            if self.use_gpu:
                self.model = self.model.to(self.device)
                
            self.logger.info(f"TrOCR initialized successfully on {self.device}.")
            
        except ImportError:
            self.logger.error("Transformers or PyTorch not installed. Run: pip install transformers torch")
            self.processor = None
            self.model = None
        except Exception as e:
            self.logger.error(f"TrOCR initialization failed: {e}")
            self.processor = None
            self.model = None
    
    def extract_text(self, image_path: str) -> List[Dict[str, Any]]:
        """Extracts text from an image path using TrOCR."""
        
        if self.processor is None or self.model is None:
            self.logger.warning("TrOCR not initialized. Skipping extraction.")
            return []
        
        try:
            # Open the image using PIL from the provided path
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image not found at path: {image_path}")
                
            pil_image = Image.open(image_path).convert('RGB')
            
            # TrOCR doesn't return bounding boxes, so we'll have to return an empty list for 'bbox'
            # We'll treat the entire image as a single text region for simplicity
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
                # TrOCR doesn't provide confidence scores per word, so we'll estimate a high confidence
                # since it's a dedicated model for handwritten text.
                confidence = 85.0
                self.logger.info(f"TrOCR extracted: '{generated_text[:50]}...'")
                
                return [{
                    'text': generated_text.strip(),
                    'confidence': confidence,
                    'bbox': [], # TrOCR doesn't provide bounding boxes
                    'engine': 'TROCR'
                }]
            else:
                return []
            
        except Exception as e:
            self.logger.error(f"TrOCR extraction failed: {e}")
            return []