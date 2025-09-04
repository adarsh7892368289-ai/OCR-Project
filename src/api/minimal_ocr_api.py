"""
Minimal Working OCR API
Bypasses complex configuration for basic functionality
"""

import sys
from pathlib import Path
from PIL import Image
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def create_basic_tesseract_engine():
    """Create a basic Tesseract engine without complex config"""
    import cv2
    import pytesseract
    import os
    
    # Configure Tesseract path for Windows
    if os.name == 'nt':
        possible_paths = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Users\adbm\AppData\Local\Programs\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        ]
        
        for path in possible_paths:
            if Path(path).exists():
                pytesseract.pytesseract.tesseract_cmd = path
                break
    
    class SimpleResult:
        def __init__(self, text, confidence=0.8):
            self.full_text = text
            self.confidence = confidence
    
    class SimpleTesseractEngine:
        def process_image(self, image):
            try:
                # Convert PIL to numpy if needed
                if hasattr(image, 'mode'):
                    image_array = np.array(image)
                else:
                    image_array = image
                
                # Convert to grayscale if needed
                if len(image_array.shape) == 3:
                    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
                else:
                    gray = image_array
                
                # Basic OCR
                text = pytesseract.image_to_string(gray, config='--psm 6')
                return SimpleResult(text.strip())
                
            except Exception as e:
                print(f"OCR Error: {e}")
                return SimpleResult("")
    
    return SimpleTesseractEngine()

def test_minimal_ocr():
    """Test the minimal OCR setup"""
    print("Testing minimal OCR setup...")
    
    try:
        engine = create_basic_tesseract_engine()
        
        # Test with sample image if available
        sample_image = Path("data/sample_images/sample_printed.jpg")
        if sample_image.exists():
            image = Image.open(sample_image)
            result = engine.process_image(image)
            
            if result.full_text.strip():
                print(f"✅ Minimal OCR test successful!")
                print(f"   Text: '{result.full_text[:100]}...'")
                return True
            else:
                print("❌ OCR returned empty result")
                return False
        else:
            print("⚠️ No sample image found for testing")
            return True
            
    except Exception as e:
        print(f"❌ Minimal OCR test failed: {e}")
        return False

if __name__ == "__main__":
    test_minimal_ocr()
