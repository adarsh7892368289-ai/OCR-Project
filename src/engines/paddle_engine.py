import cv2
import numpy as np
from typing import List, Dict, Any
import logging

class PaddleEngine:
    """PaddleOCR engine wrapper with improved text extraction"""
    
    def __init__(self, use_gpu: bool = True, config: Dict = None):
        self.use_gpu = use_gpu
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.ocr = None
        self._initialize()
    
    def _initialize(self):
        """Initialize PaddleOCR"""
        try:
            from paddleocr import PaddleOCR
            
            self.ocr = PaddleOCR(
                use_angle_cls=self.config.get('use_angle_cls', True),
                lang=self.config.get('language', 'en'),
                use_gpu=self.use_gpu,
                show_log=False  # Reduce verbose output
            )
            
        except ImportError:
            raise ImportError("PaddleOCR not installed. Run: pip install paddleocr")
        except Exception as e:
            self.logger.error(f"PaddleOCR initialization failed: {e}")
            raise
    
    def extract_text(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Extract text using PaddleOCR with better error handling"""
        
        if self.ocr is None:
            return []
        
        try:
            # Ensure image is in the right format
            if len(image.shape) == 2:
                # Convert grayscale to BGR
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif len(image.shape) == 3 and image.shape[2] == 3:
                # Ensure it's BGR format (PaddleOCR expects BGR)
                pass
            
            results = self.ocr.ocr(image, cls=True)
            extracted_texts = []
            
            if results and results[0]:
                for line in results[0]:
                    if line and len(line) >= 2:
                        bbox = line[0]
                        text_info = line[1]
                        text = text_info[0] if text_info else ""
                        confidence = text_info[1] if len(text_info) > 1 else 0.0
                        
                        if text.strip():  # Only add non-empty text
                            extracted_texts.append({
                                'text': text.strip(),
                                'confidence': float(confidence) * 100,  # Convert to percentage
                                'bbox': bbox,
                                'engine': 'PADDLE'
                            })
            
            self.logger.info(f"PaddleOCR found {len(extracted_texts)} text regions")
            return extracted_texts
            
        except Exception as e:
            self.logger.error(f"PaddleOCR extraction failed: {e}")
            return []