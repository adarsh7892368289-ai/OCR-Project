import os
import cv2
import numpy as np
import pytesseract
import logging
from typing import List, Dict, Any

class TesseractEngine:
    """A wrapper class for the Tesseract OCR engine."""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self._set_tesseract_path()
        self._check_tesseract_installation()

    def _set_tesseract_path(self):
        """
        Sets the Tesseract executable path.
        This handles common Windows installation paths automatically.
        """
        try:
            # Check for Windows operating system
            if os.name == 'nt':
                # Define potential paths to check, prioritizing the common one
                potential_paths = [
                    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                    r'C:\Users\adbm\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
                ]
                
                path_found = False
                for path in potential_paths:
                    if os.path.exists(path):
                        pytesseract.pytesseract.tesseract_cmd = path
                        self.logger.info(f"Tesseract path set to: {path}")
                        path_found = True
                        break
                
                if not path_found:
                    self.logger.warning("Tesseract path could not be found. Please ensure it's installed or add its path to your system's PATH variable.")
            else:
                # For non-Windows systems, assume Tesseract is in the system's PATH
                pytesseract.pytesseract.tesseract_cmd = 'tesseract'
                self.logger.info("Tesseract path automatically set for non-Windows systems.")

        except Exception as e:
            self.logger.error(f"Failed to configure Tesseract path: {e}")

    def _check_tesseract_installation(self):
        """Check if Tesseract is available"""
        try:
            pytesseract.get_tesseract_version()
        except pytesseract.TesseractNotFoundError:
            raise RuntimeError("Tesseract is not installed or not in your PATH. Please install Tesseract or update `_set_tesseract_path`.")
        except Exception as e:
            raise RuntimeError(f"Tesseract not properly configured: {e}")

    def extract_text(self, image_path: str, config: Dict = None) -> List[Dict[str, Any]]:
        """Extracts text from an image path using Tesseract."""
        try:
            # Read the image from the path using OpenCV
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError(f"Image not found at path: {image_path}")
            
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Use the provided config or default
            pytesseract_config = config.get('config', '--psm 6') if config else '--psm 6'
            
            # Get detailed data
            data = pytesseract.image_to_data(
                gray, 
                output_type=pytesseract.Output.DICT,
                config=pytesseract_config
            )
            
            extracted_texts = []
            
            # Process results
            for i in range(len(data['text'])):
                conf = int(data['conf'][i])
                text = data['text'][i].strip()
                if conf > 0 and text:  # Only confident and non-empty results
                    x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                    bbox = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
                    
                    extracted_texts.append({
                        'text': text,
                        'confidence': float(conf),
                        'bbox': bbox,
                        'engine': 'TESSERACT'
                    })
            
            self.logger.info(f"Tesseract found {len(extracted_texts)} text regions.")
            return extracted_texts
        
        except pytesseract.TesseractNotFoundError as e:
            self.logger.error(e)
            return []
        except Exception as e:
            self.logger.error(f"Tesseract extraction failed: {e}")
            return []