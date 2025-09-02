import easyocr
import logging
import cv2
import numpy as np
from typing import List, Dict, Any
import torch

class EasyOCREngine:
    """EasyOCR engine wrapper with improved text extraction."""
    
    def __init__(self, use_gpu: bool = True, config: Dict = None):
        self.use_gpu = use_gpu
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self.reader = None
        self._initialize()
    
    def _initialize(self):
        """Initialize EasyOCR"""
        try:
            languages = self.config.get('languages', ['en'])
            
            self.reader = easyocr.Reader(
                languages, 
                gpu=self.use_gpu,
                verbose=False
            )
            self.logger.info(f"EasyOCR initialized successfully. GPU enabled: {self.reader.device == 'cuda'}")
        except ImportError:
            raise ImportError("EasyOCR not installed. Run: pip install easyocr")
        except Exception as e:
            self.logger.error(f"EasyOCR initialization failed: {e}")
            raise
    
    def extract_text(self, image_path: str) -> List[Dict[str, Any]]:
        """Extract text from an image path using EasyOCR."""
        
        if self.reader is None:
            self.logger.warning("EasyOCR not initialized. Skipping extraction.")
            return []
        
        try:
            # Read the image from the path using OpenCV
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError(f"Image not found at path: {image_path}")
                
            # EasyOCR can handle both grayscale and color images directly
            results = self.reader.readtext(image, detail=1)
            extracted_texts = []
            
            for result in results:
                if len(result) >= 3:
                    bbox = result[0]
                    text = result[1]
                    confidence = result[2]
                    
                    if text.strip():
                        extracted_texts.append({
                            'text': text.strip(),
                            'confidence': float(confidence) * 100,
                            'bbox': bbox,
                            'engine': 'EASYOCR'
                        })
            
            self.logger.info(f"EasyOCR found {len(extracted_texts)} text regions.")
            return extracted_texts
            
        except Exception as e:
            self.logger.error(f"EasyOCR extraction failed: {e}")
            return []