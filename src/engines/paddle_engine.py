import os
import cv2
import logging
from typing import List, Dict, Any
import warnings

# Suppress the PaddlePaddle warnings
warnings.filterwarnings('ignore', category=UserWarning, module='paddle')

class PaddleOCREngine:
    """A wrapper class for the PaddleOCR engine."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.ocr = None
        self._initialize_paddleocr()
    
    def _initialize_paddleocr(self):
        """Initializes the PaddleOCR engine and handles common errors."""
        try:
            # Critical fix for OMP: Error #15
            os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
            
            from paddleocr import PaddleOCR
            
            # This will determine if a GPU is available and use it
            use_gpu = len(os.getenv('CUDA_VISIBLE_DEVICES', '')) > 0
            
            # Suppress specific logging from PaddleOCR
            self.ocr = PaddleOCR(
                use_angle_cls=True,
                lang='en',
                use_gpu=use_gpu,
                show_log=False
            )
            self.logger.info(f"PaddleOCR initialized successfully. GPU enabled: {use_gpu}")
            
        except ImportError:
            self.logger.error("PaddleOCR or its dependencies are not installed. Run: pip install paddleocr paddlepaddle")
            self.ocr = None
        except Exception as e:
            self.logger.error(f"PaddleOCR initialization failed: {e}")
            self.ocr = None

    def extract_text(self, image_path: str) -> List[Dict[str, Any]]:
        """
        Extracts text from an image path using PaddleOCR.
        Args:
            image_path (str): Path to the image file.
        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each containing
                                   'text', 'confidence', and 'bbox'.
        """
        if not self.ocr:
            self.logger.warning("PaddleOCR not initialized. Skipping extraction.")
            return []
            
        try:
            # Read the image from the path using OpenCV
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError(f"Image not found at path: {image_path}")
            
            # The PaddleOCR .ocr() method expects a BGR image.
            # Your original code was correct in this regard.
            # No need for manual conversion here, as .ocr() handles it.
            
            # Use cls=True for angle classification, which is essential for rotated text
            results = self.ocr.ocr(image, cls=True)
            extracted_texts = []
            
            # Check if results is not None and has at least one element
            if results and results[0]:
                for line in results[0]:
                    if line and len(line) >= 2:
                        bbox = line[0]
                        text_info = line[1]
                        text = text_info[0] if text_info else ""
                        confidence = text_info[1] if len(text_info) > 1 else 0.0
                        
                        if text.strip():
                            extracted_texts.append({
                                'text': text.strip(),
                                'confidence': float(confidence) * 100,
                                'bbox': bbox,
                                'engine': 'PADDLE'
                            })
            
            self.logger.info(f"PaddleOCR found {len(extracted_texts)} text regions.")
            return extracted_texts
            
        except Exception as e:
            self.logger.error(f"PaddleOCR extraction failed: {e}")
            return []