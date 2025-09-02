import cv2
import numpy as np
from typing import List, Dict, Any
import logging

class EasyOCREngine:
    """EasyOCR engine wrapper with improved text extraction"""
    
    def __init__(self, use_gpu: bool = True, config: Dict = None):
        self.use_gpu = use_gpu
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.reader = None
        self._initialize()
    
    def _initialize(self):
        """Initialize EasyOCR"""
        try:
            import easyocr
            
            languages = self.config.get('languages', ['en'])
            
            self.reader = easyocr.Reader(
                languages, 
                gpu=self.use_gpu,
                verbose=False  # Reduce verbose output
            )
            
        except ImportError:
            raise ImportError("EasyOCR not installed. Run: pip install easyocr")
        except Exception as e:
            self.logger.error(f"EasyOCR initialization failed: {e}")
            raise
    
    def extract_text(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Extract text using EasyOCR with better processing"""
        
        if self.reader is None:
            return []
        
        try:
            # EasyOCR can handle both grayscale and color images
            results = self.reader.readtext(image, detail=1)
            extracted_texts = []
            
            for result in results:
                if len(result) >= 3:
                    bbox = result[0]
                    text = result[1]
                    confidence = result[2]
                    
                    if text.strip():  # Only add non-empty text
                        extracted_texts.append({
                            'text': text.strip(),
                            'confidence': float(confidence) * 100,  # Convert to percentage
                            'bbox': bbox,
                            'engine': 'EASYOCR'
                        })
            
            self.logger.info(f"EasyOCR found {len(extracted_texts)} text regions")
            return extracted_texts
            
        except Exception as e:
            self.logger.error(f"EasyOCR extraction failed: {e}")
            return []
