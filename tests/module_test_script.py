#!/usr/bin/env python3
"""
Advanced OCR System - Module Installation Test
Tests all required modules and their compatibility
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import traceback
from typing import Dict, List, Tuple   
def test_module_import(module_name: str, test_name: str = None) -> Tuple[bool, str]:
    """Test importing a module and return success status and message"""
    try:
        if module_name == "cv2":
            import cv2
            version = cv2.__version__
            message = f"OpenCV {version}"
        elif module_name == "torch":
            import torch
            version = torch.__version__
            cuda_available = torch.cuda.is_available()
            message = f"PyTorch {version} (CUDA: {cuda_available})"
        elif module_name == "transformers":
            import transformers
            version = transformers.__version__
            message = f"Transformers {version}"
        elif module_name == "paddleocr":
            from paddleocr import PaddleOCR
            message = "PaddleOCR imported successfully"
        elif module_name == "easyocr":
            import easyocr
            message = "EasyOCR imported successfully"
        elif module_name == "trocr":
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            message = "TrOCR (Transformers-based) imported successfully"
        elif module_name == "pytesseract":
            import pytesseract
            message = "PyTesseract imported successfully"
        elif module_name == "spacy":
            import spacy
            version = spacy.__version__
            message = f"spaCy {version}"
            # Test loading English model
            try:
                nlp = spacy.load("en_core_web_sm")
                message += " (en_core_web_sm loaded)"
            except:
                message += " (en_core_web_sm NOT available)"
        elif module_name == "fastapi":
            import fastapi
            version = fastapi.__version__
            message = f"FastAPI {version}"
        elif module_name == "numpy":
            import numpy as np
            version = np.__version__
            message = f"NumPy {version}"
        elif module_name == "PIL":
            from PIL import Image
            message = "Pillow imported successfully"
        elif module_name == "skimage":
            import skimage
            version = skimage.__version__
            message = f"scikit-image {version}"
        elif module_name == "matplotlib":
            import matplotlib
            version = matplotlib.__version__
            message = f"Matplotlib {version}"
        elif module_name == "pandas":
            import pandas as pd
            version = pd.__version__
            message = f"Pandas {version}"
        elif module_name == "nltk":
            import nltk
            version = nltk.__version__
            message = f"NLTK {version}"
            # Test NLTK data
            try:
                nltk.data.find('tokenizers/punkt')
                nltk.data.find('corpora/stopwords')
                message += " (punkt, stopwords available)"
            except:
                message += " (punkt/stopwords NOT available)"
        else:
            # Generic import
            __import__(module_name)
            message = f"{module_name} imported successfully"
        
        return True, message
    except ImportError as e:
        return False, f"ImportError: {str(e)}"
    except Exception as e:
        return False, f"Error: {str(e)}"

def run_comprehensive_tests():
    """Run comprehensive tests for all modules"""
    print("=" * 70)
    print("ADVANCED OCR SYSTEM - MODULE INSTALLATION TEST")
    print("=" * 70)
    print()
    
    # Define test modules by category
    test_modules = {
        "Core Dependencies": [
            ("numpy", "NumPy"),
            ("cv2", "OpenCV"),
            ("PIL", "Pillow"),
        ],
        "OCR Engines": [
            ("pytesseract", "PyTesseract"),
            ("easyocr", "EasyOCR"),
            ("paddleocr", "PaddleOCR"),
        ],
        "Deep Learning": [
            ("torch", "PyTorch"),
            ("transformers", "Transformers"),
            ("timm", "TIMM"),
        ],
        "Image Processing": [
            ("skimage", "scikit-image"),
            ("imageio", "ImageIO"),
            ("matplotlib", "Matplotlib"),
        ],
        "NLP": [
            ("spacy", "spaCy"),
            ("nltk", "NLTK"),
            ("textdistance", "TextDistance"),
        ],
        "API Framework": [
            ("fastapi", "FastAPI"),
            ("uvicorn", "Uvicorn"),
        ],
        "Data Processing": [
            ("pandas", "Pandas"),
            ("openpyxl", "OpenPyXL"),
        ],
        "Utilities": [
            ("yaml", "PyYAML"),
            ("loguru", "Loguru"),
            ("tqdm", "TQDM"),
            ("click", "Click"),
        ]
    }
    
    results = {}
    total_tests = 0
    passed_tests = 0
    
    for category, modules in test_modules.items():
        print(f"\n📁 {category}")
        print("-" * (len(category) + 3))
        
        category_results = []
        for module_name, display_name in modules:
            total_tests += 1
            success, message = test_module_import(module_name, display_name)
            
            if success:
                print(f"  ✅ {display_name:<20} - {message}")
                passed_tests += 1
                status = "PASS"
            else:
                print(f"  ❌ {display_name:<20} - {message}")
                status = "FAIL"
            
            category_results.append((display_name, status, message))
        
        results[category] = category_results
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
    
    if passed_tests == total_tests:
        print("\n🎉 ALL TESTS PASSED! Your OCR system is ready to use.")
    else:
        print(f"\n⚠️  {total_tests - passed_tests} tests failed. Please install missing dependencies.")
        print("\nFailed modules:")
        for category, category_results in results.items():
            for module, status, message in category_results:
                if status == "FAIL":
                    print(f"  - {module}: {message}")
    
    return passed_tests == total_tests

def test_ocr_functionality():
    """Test basic OCR functionality with available engines"""
    print("\n" + "=" * 70)
    print("OCR FUNCTIONALITY TEST")
    print("=" * 70)
    
    try:
        import numpy as np
        import cv2
        from PIL import Image, ImageDraw, ImageFont
        
        # Create a simple test image
        print("Creating test image...")
        img = Image.new('RGB', (400, 100), color='white')
        draw = ImageDraw.Draw(img)
        
        # Draw text (use default font)
        draw.text((10, 30), "Hello OCR Test 123", fill='black')
        
        # Convert to numpy array for OpenCV
        img_array = np.array(img)
        
        print("✅ Test image created successfully")
        
        # Test available OCR engines
        ocr_results = {}
        
        # Test PyTesseract
        try:
            import pytesseract
            text = pytesseract.image_to_string(img)
            ocr_results['PyTesseract'] = text.strip()
            print(f"✅ PyTesseract result: '{ocr_results['PyTesseract']}'")
        except Exception as e:
            print(f"❌ PyTesseract failed: {e}")
        
        # Test EasyOCR
        try:
            import easyocr
            reader = easyocr.Reader(['en'])
            results = reader.readtext(img_array)
            if results:
                ocr_results['EasyOCR'] = ' '.join([result[1] for result in results])
                print(f"✅ EasyOCR result: '{ocr_results['EasyOCR']}'")
            else:
                print("❌ EasyOCR: No text detected")
        except Exception as e:
            print(f"❌ EasyOCR failed: {e}")
        
        # Test PaddleOCR
        try:
            from paddleocr import PaddleOCR
            ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
            results = ocr.ocr(img_array, cls=True)
            if results and results[0]:
                ocr_results['PaddleOCR'] = ' '.join([item[1][0] for item in results[0]])
                print(f"✅ PaddleOCR result: '{ocr_results['PaddleOCR']}'")
            else:
                print("❌ PaddleOCR: No text detected")
        except Exception as e:
            print(f"❌ PaddleOCR failed: {e}")
        
        if ocr_results:
            print(f"\n🎉 OCR engines working! {len(ocr_results)} engines successfully processed text.")
        else:
            print("\n⚠️  No OCR engines are working properly.")
            
        return len(ocr_results) > 0
        
    except Exception as e:
        print(f"❌ OCR functionality test failed: {e}")
        return False

def main():
    """Main test function"""
    print("Starting Advanced OCR System installation test...")
    print(f"Python version: {sys.version}")
    print()
    
    # Run module tests
    modules_ok = run_comprehensive_tests()
    
    # Run OCR functionality test if modules are OK
    if modules_ok:
        ocr_ok = test_ocr_functionality()
        
        if ocr_ok:
            print("\n🚀 SYSTEM READY: All components installed and working!")
        else:
            print("\n⚠️  PARTIAL SUCCESS: Modules installed but OCR functionality needs attention.")
    else:
        print("\n❌ INSTALLATION INCOMPLETE: Please fix failed dependencies first.")
        
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()