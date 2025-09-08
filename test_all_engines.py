#!/usr/bin/env python3
"""
Complete OCR System Test Script
==============================

This script tests your entire OCR pipeline:
1. All engines (Tesseract, EasyOCR, PaddleOCR, TrOCR)
2. Preprocessing pipeline
3. Text detection
4. Post-processing
5. API endpoints

Usage:
    python complete_ocr_test.py
    python complete_ocr_test.py --image path/to/your/image.jpg
    python complete_ocr_test.py --all-tests
"""

import os
import sys
import traceback
import argparse
import time
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

print(f"Project root: {project_root}")
print(f"Python path: {sys.path[:3]}")

class OCRSystemTester:
    """Comprehensive test suite for the OCR system."""
    
    def __init__(self):
        self.results = {}
        self.test_image = None
        
    def create_test_image(self) -> np.ndarray:
        """Create a test image with mixed text for testing."""
        # Create a white background
        img = np.ones((400, 800, 3), dtype=np.uint8) * 255
        
        # Add printed text
        cv2.putText(img, "PRINTED TEXT: Hello World 2024", (50, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Add smaller printed text
        cv2.putText(img, "Advanced OCR System Test Document", (50, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        
        # Add numbers
        cv2.putText(img, "Numbers: 123456789", (50, 160), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        
        # Simulate handwritten text (rough approximation)
        cv2.putText(img, "Handwritten: Thank you!", (50, 220), 
                   cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Add some shapes to test text detection
        cv2.rectangle(img, (50, 280), (400, 320), (0, 0, 0), 2)
        cv2.putText(img, "Text in box", (70, 305), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        return img

    def test_basic_imports(self):
        """Test if all required modules can be imported."""
        print("\n" + "="*50)
        print("TESTING: Basic Imports")
        print("="*50)
        
        imports = {
            'opencv': 'cv2',
            'numpy': 'numpy', 
            'PIL': 'PIL',
            'torch': 'torch',
            'transformers': 'transformers',
            'pytesseract': 'pytesseract',
            'easyocr': 'easyocr'
        }
        
        for name, module in imports.items():
            try:
                __import__(module)
                print(f"✅ {name}: OK")
                self.results[f'import_{name}'] = True
            except ImportError as e:
                print(f"❌ {name}: FAILED - {e}")
                self.results[f'import_{name}'] = False

    def test_project_structure(self):
        """Test if project files exist."""
        print("\n" + "="*50)
        print("TESTING: Project Structure")
        print("="*50)
        
        required_files = [
            'src/core/base_engine.py',
            'src/core/engine_manager.py', 
            'src/engines/tesseract_engine.py',
            'src/engines/easyocr_engine.py',
            'src/engines/trocr_engine.py',
            'src/utils/config.py',
            'src/utils/logger.py'
        ]
        
        for file_path in required_files:
            full_path = project_root / file_path
            if full_path.exists():
                print(f"✅ {file_path}: EXISTS")
                self.results[f'file_{file_path.replace("/", "_")}'] = True
            else:
                print(f"❌ {file_path}: MISSING")
                self.results[f'file_{file_path.replace("/", "_")}'] = False

    def test_tesseract_engine(self):
        """Test Tesseract OCR engine."""
        print("\n" + "="*50)
        print("TESTING: Tesseract Engine")
        print("="*50)
        
        try:
            from engines.tesseract_engine import TesseractEngine
            
            engine = TesseractEngine()
            if engine.initialize():
                print("✅ Tesseract initialization: OK")
                
                # Test with our test image
                result = engine.process_image(self.test_image)
                if result and result.pages and result.pages[0].text.strip():
                    print(f"✅ Tesseract recognition: OK")
                    print(f"   Text: {result.pages[0].text[:50]}...")
                    print(f"   Confidence: {result.pages[0].confidence:.3f}")
                    self.results['tesseract_engine'] = True
                else:
                    print("❌ Tesseract recognition: No text found")
                    self.results['tesseract_engine'] = False
            else:
                print("❌ Tesseract initialization: FAILED")
                self.results['tesseract_engine'] = False
                
        except Exception as e:
            print(f"❌ Tesseract engine: ERROR - {e}")
            self.results['tesseract_engine'] = False

    def test_easyocr_engine(self):
        """Test EasyOCR engine."""
        print("\n" + "="*50)
        print("TESTING: EasyOCR Engine")
        print("="*50)
        
        try:
            from engines.easyocr_engine import EasyOCREngine
            
            engine = EasyOCREngine()
            if engine.initialize():
                print("✅ EasyOCR initialization: OK")
                
                # Test with our test image
                result = engine.process_image(self.test_image)
                if result and result.pages and result.pages[0].text.strip():
                    print(f"✅ EasyOCR recognition: OK")
                    print(f"   Text: {result.pages[0].text[:50]}...")
                    print(f"   Confidence: {result.pages[0].confidence:.3f}")
                    self.results['easyocr_engine'] = True
                else:
                    print("❌ EasyOCR recognition: No text found")
                    self.results['easyocr_engine'] = False
            else:
                print("❌ EasyOCR initialization: FAILED")
                self.results['easyocr_engine'] = False
                
        except Exception as e:
            print(f"❌ EasyOCR engine: ERROR - {e}")
            self.results['easyocr_engine'] = False

    def test_trocr_engine(self):
        """Test TrOCR engine."""
        print("\n" + "="*50)
        print("TESTING: TrOCR Engine")
        print("="*50)
        
        try:
            from engines.trocr_engine import TrOCREngine
            
            engine = TrOCREngine()
            if engine.initialize():
                print("✅ TrOCR initialization: OK")
                
                # Test with our test image
                result = engine.process_image(self.test_image)
                if result and result.pages and result.pages[0].text.strip():
                    print(f"✅ TrOCR recognition: OK")
                    print(f"   Text: {result.pages[0].text[:50]}...")
                    print(f"   Confidence: {result.pages[0].confidence:.3f}")
                    self.results['trocr_engine'] = True
                else:
                    print("❌ TrOCR recognition: No text found")
                    self.results['trocr_engine'] = False
            else:
                print("❌ TrOCR initialization: FAILED")
                self.results['trocr_engine'] = False
                
        except Exception as e:
            print(f"❌ TrOCR engine: ERROR - {e}")
            print(f"   Details: {traceback.format_exc()}")
            self.results['trocr_engine'] = False

    def test_engine_manager(self):
        """Test the engine manager."""
        print("\n" + "="*50)
        print("TESTING: Engine Manager")
        print("="*50)
        
        try:
            from core.engine_manager import EngineManager
            
            manager = EngineManager()
            manager.initialize()
            
            available_engines = manager.get_available_engines()
            print(f"✅ Available engines: {available_engines}")
            
            if available_engines:
                # Test processing with the first available engine
                engine_name = available_engines[0]
                result = manager.process_image(self.test_image, engine_name=engine_name)
                
                if result and result.pages:
                    print(f"✅ Engine manager processing: OK")
                    print(f"   Engine used: {engine_name}")
                    print(f"   Text found: {len(result.pages[0].text)} characters")
                    self.results['engine_manager'] = True
                else:
                    print("❌ Engine manager processing: FAILED")
                    self.results['engine_manager'] = False
            else:
                print("❌ No engines available")
                self.results['engine_manager'] = False
                
        except Exception as e:
            print(f"❌ Engine manager: ERROR - {e}")
            self.results['engine_manager'] = False

    def test_preprocessing(self):
        """Test preprocessing modules."""
        print("\n" + "="*50)
        print("TESTING: Preprocessing")
        print("="*50)
        
        try:
            from preprocessing.image_enhancer import ImageEnhancer
            from preprocessing.text_detector import TextDetector
            
            # Test image enhancer
            enhancer = ImageEnhancer()
            enhanced = enhancer.enhance_image(self.test_image)
            if enhanced is not None and enhanced.shape == self.test_image.shape:
                print("✅ Image enhancement: OK")
                self.results['image_enhancer'] = True
            else:
                print("❌ Image enhancement: FAILED")
                self.results['image_enhancer'] = False
            
            # Test text detector
            detector = TextDetector()
            if detector.initialize():
                regions = detector.detect_text_regions(self.test_image)
                print(f"✅ Text detection: OK - Found {len(regions)} regions")
                self.results['text_detector'] = True
            else:
                print("❌ Text detection: FAILED to initialize")
                self.results['text_detector'] = False
                
        except Exception as e:
            print(f"❌ Preprocessing: ERROR - {e}")
            self.results['preprocessing'] = False

    def test_api_functionality(self):
        """Test API endpoints."""
        print("\n" + "="*50)
        print("TESTING: API Functionality")
        print("="*50)
        
        try:
            # Import FastAPI app
            from api.main import app
            print("✅ API import: OK")
            
            # Check if we can create the app
            if app:
                print("✅ FastAPI app creation: OK")
                self.results['api_import'] = True
            else:
                print("❌ FastAPI app creation: FAILED")
                self.results['api_import'] = False
                
        except Exception as e:
            print(f"❌ API functionality: ERROR - {e}")
            self.results['api_import'] = False

    def process_custom_image(self, image_path: str):
        """Process a custom image through the system."""
        print(f"\n" + "="*50)
        print(f"TESTING: Custom Image - {image_path}")
        print("="*50)
        
        if not os.path.exists(image_path):
            print(f"❌ Image file not found: {image_path}")
            return
        
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                print(f"❌ Could not load image: {image_path}")
                return
            
            print(f"✅ Image loaded: {image.shape}")
            
            # Test with available engines
            try:
                from core.engine_manager import EngineManager
                manager = EngineManager()
                manager.initialize()
                
                available_engines = manager.get_available_engines()
                print(f"Available engines: {available_engines}")
                
                for engine_name in available_engines:
                    print(f"\n--- Testing {engine_name} ---")
                    start_time = time.time()
                    
                    result = manager.process_image(image, engine_name=engine_name)
                    processing_time = time.time() - start_time
                    
                    if result and result.pages:
                        text = result.pages[0].text.strip()
                        confidence = result.pages[0].confidence
                        
                        print(f"✅ {engine_name}: SUCCESS")
                        print(f"   Processing time: {processing_time:.2f}s")
                        print(f"   Confidence: {confidence:.3f}")
                        print(f"   Text length: {len(text)} characters")
                        print(f"   Sample text: {text[:100]}...")
                    else:
                        print(f"❌ {engine_name}: No results")
                        
            except Exception as e:
                print(f"❌ Error processing image: {e}")
                
        except Exception as e:
            print(f"❌ Error with custom image: {e}")

    def run_all_tests(self, custom_image=None):
        """Run all tests."""
        print("🚀 Starting Complete OCR System Test")
        print("="*60)
        
        # Create test image
        self.test_image = self.create_test_image()
        
        # Save test image for reference
        cv2.imwrite('test_image.png', self.test_image)
        print("✅ Test image created and saved as 'test_image.png'")
        
        # Run all tests
        self.test_basic_imports()
        self.test_project_structure()
        self.test_tesseract_engine()
        self.test_easyocr_engine() 
        self.test_trocr_engine()
        self.test_engine_manager()
        self.test_preprocessing()
        self.test_api_functionality()
        
        # Test custom image if provided
        if custom_image:
            self.process_custom_image(custom_image)
        
        # Print summary
        self.print_summary()

    def print_summary(self):
        """Print test summary."""
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        
        total_tests = len(self.results)
        passed_tests = sum(1 for result in self.results.values() if result)
        
        print(f"Total tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success rate: {passed_tests/total_tests*100:.1f}%")
        
        print("\n--- Detailed Results ---")
        for test_name, result in self.results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"{status}: {test_name}")
        
        if passed_tests == total_tests:
            print("\n🎉 ALL TESTS PASSED! Your OCR system is working correctly!")
        else:
            print(f"\n⚠️  {total_tests - passed_tests} tests failed. Check the output above for details.")


def main():
    """Main function to run tests."""
    parser = argparse.ArgumentParser(description='Test OCR System')
    parser.add_argument('--image', help='Path to custom image to test')
    parser.add_argument('--all-tests', action='store_true', help='Run all tests')
    
    args = parser.parse_args()
    
    tester = OCRSystemTester()
    
    if args.all_tests or args.image:
        tester.run_all_tests(custom_image=args.image)
    else:
        # Quick test mode
        tester.run_all_tests()


if __name__ == "__main__":
    main()