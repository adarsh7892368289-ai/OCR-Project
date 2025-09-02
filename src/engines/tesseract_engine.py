import cv2
import numpy as np
from typing import List, Dict, Any
import logging

class TesseractEngine:
    """Tesseract OCR engine wrapper"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self._check_tesseract()
    
    def _check_tesseract(self):
        """Check if Tesseract is available"""
        try:
            import pytesseract
            # Try to get version to verify installation
            pytesseract.get_tesseract_version()
        except ImportError:
            raise ImportError("PyTesseract not installed. Run: pip install pytesseract")
        except Exception as e:
            raise RuntimeError(f"Tesseract not properly configured: {e}")
    
    def extract_text(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Extract text using Tesseract"""
        
        try:
            import pytesseract
            
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Get detailed data
            data = pytesseract.image_to_data(
                gray, 
                output_type=pytesseract.Output.DICT,
                config=self.config.get('config', '--psm 6')
            )
            
            extracted_texts = []
            
            # Process results
            for i in range(len(data['text'])):
                if int(data['conf'][i]) > 0:  # Only confident results
                    text = data['text'][i].strip()
                    if text:
                        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                        bbox = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
                        
                        extracted_texts.append({
                            'text': text,
                            'confidence': float(data['conf'][i]),
                            'bbox': bbox,
                            'engine': 'TESSERACT'
                        })
            
            self.logger.info(f"Tesseract found {len(extracted_texts)} text regions")
            return extracted_texts
            
        except ImportError:
            self.logger.error("PyTesseract not available")
            return []
        except Exception as e:
            self.logger.error(f"Tesseract extraction failed: {e}")
            return []