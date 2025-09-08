#!/usr/bin/env python3
"""
Direct Tesseract Test Script
Tests Tesseract OCR directly without the full pipeline
"""

import sys
import os
import cv2
import numpy as np
from PIL import Image

# Add src to path
sys.path.insert(0, 'src')

def test_tesseract_direct():
    """Test Tesseract directly"""
    print("🔍 DIRECT TESSERACT TEST")
    print("="*50)

    try:
        # Import Tesseract engine
        from src.engines.tesseract_engine import TesseractEngine
        print("✅ TesseractEngine imported successfully")

        # Create engine instance
        engine = TesseractEngine()
        print("✅ TesseractEngine instance created")

        # Initialize engine
        success = engine.initialize()
        if success:
            print("✅ Tesseract engine initialized successfully")
        else:
            print("❌ Tesseract engine initialization failed")
            return False

        # Create a simple test image
        test_image = np.ones((100, 300, 3), dtype=np.uint8) * 255
        cv2.putText(test_image, "HELLO TESSERACT", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        print("✅ Test image created with text: 'HELLO TESSERACT'")

        # Process the image
        print("🔄 Processing image with Tesseract...")
        result = engine.process_image(test_image)

        # Check results
        if result and result.pages:
            page = result.pages[0]
            extracted_text = page.text.strip()
            confidence = page.confidence

            print("✅ OCR completed successfully!")
            print(f"📝 Extracted text: '{extracted_text}'")
            print(".2f")
            print(f"⏱️  Processing time: {result.processing_time:.2f} seconds")

            # Check if text was extracted
            if extracted_text and len(extracted_text) > 0:
                print("✅ Tesseract is working correctly!")
                return True
            else:
                print("⚠️  Tesseract processed image but extracted no text")
                return False
        else:
            print("❌ No OCR result returned")
            return False

    except Exception as e:
        print(f"❌ Error during Tesseract test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_tesseract_with_sample_image():
    """Test Tesseract with a real sample image"""
    print("\n🔍 TESTING WITH SAMPLE IMAGE")
    print("="*50)

    try:
        from src.engines.tesseract_engine import TesseractEngine

        engine = TesseractEngine()
        if not engine.initialize():
            print("❌ Engine initialization failed")
            return False

        # Use a sample image from the data directory
        sample_images = [
            "data/sample_images/ocr_test.jpg",
            "data/sample_images/sample_printed.jpg",
            "data/sample_images/printed_text.png"
        ]

        for image_path in sample_images:
            if os.path.exists(image_path):
                print(f"📸 Testing with: {image_path}")

                # Load image
                image = cv2.imread(image_path)
                if image is None:
                    print(f"❌ Could not load image: {image_path}")
                    continue

                # Process with Tesseract
                result = engine.process_image(image)

                if result and result.pages:
                    page = result.pages[0]
                    extracted_text = page.text.strip()
                    confidence = page.confidence

                    print("✅ OCR completed!")
                    print(f"📝 Extracted text length: {len(extracted_text)} characters")
                    print(".2f")
                    print(f"📄 First 100 chars: {extracted_text[:100]}...")

                    if extracted_text and len(extracted_text) > 5:
                        print("✅ Tesseract successfully extracted text from sample image!")
                        return True

        print("⚠️  No suitable sample images found or processed successfully")
        return False

    except Exception as e:
        print(f"❌ Error testing with sample image: {e}")
        return False

if __name__ == "__main__":
    print("Testing Tesseract OCR functionality...\n")

    # Test 1: Direct Tesseract test
    success1 = test_tesseract_direct()

    # Test 2: Test with sample image
    success2 = test_tesseract_with_sample_image()

    print("\n" + "="*60)
    print("📊 TESSERACT TEST RESULTS")
    print("="*60)

    if success1 or success2:
        print("✅ CONCLUSION: Tesseract is working correctly!")
        print("   - Tesseract executable found and accessible")
        print("   - pytesseract can communicate with Tesseract")
        print("   - OCR processing completes successfully")
        if success1:
            print("   - Successfully processed synthetic test image")
        if success2:
            print("   - Successfully processed real sample image")
    else:
        print("❌ CONCLUSION: Tesseract is not working correctly")
        print("   - Check Tesseract installation")
        print("   - Verify pytesseract version compatibility")
        print("   - Ensure Tesseract is in PATH or pytesseract.tesseract_cmd is set")

    print("\n🔧 If issues persist:")
    print("   1. Install Tesseract: https://github.com/UB-Mannheim/tesseract/wiki")
    print("   2. Install pytesseract: pip install pytesseract")
    print("   3. Set path: pytesseract.pytesseract.tesseract_cmd = r'C:\\path\\to\\tesseract.exe'")
