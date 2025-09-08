#!/usr/bin/env python3
"""
Test script to check if PaddleOCR is working correctly
"""

import sys
import os
import cv2
import numpy as np

# Add src to path
sys.path.insert(0, 'src')

def test_paddle_import():
    """Test if PaddleOCR can be imported"""
    print("Testing PaddleOCR import...")
    try:
        from paddleocr import PaddleOCR
        print("✅ PaddleOCR import successful")
        return True
    except ImportError as e:
        print(f"❌ PaddleOCR import failed: {e}")
        print("Please install PaddleOCR: pip install paddlepaddle paddleocr")
        return False

def test_paddle_initialization():
    """Test PaddleOCR initialization"""
    print("\nTesting PaddleOCR initialization...")
    try:
        from paddleocr import PaddleOCR

        # Initialize with basic settings (updated for current version)
        ocr = PaddleOCR(
            use_textline_orientation=True,  # Updated parameter
            lang='en'
        )
        print("✅ PaddleOCR initialization successful")
        return ocr
    except Exception as e:
        print(f"❌ PaddleOCR initialization failed: {e}")
        return None

def test_paddle_ocr(ocr, image_path):
    """Test OCR on an image"""
    print(f"\nTesting OCR on image: {image_path}")

    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return False

    try:
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print("❌ Failed to read image")
            return False

        print(f"Image shape: {image.shape}")

        # Perform OCR
        results = ocr.ocr(image, cls=True)

        if results and results[0]:
            print(f"✅ OCR successful! Found {len(results[0])} text regions")

            # Print first few results
            for i, line in enumerate(results[0][:3]):
                if line and len(line) >= 2:
                    bbox = line[0]
                    text_info = line[1]
                    text = text_info[0] if text_info else ""
                    confidence = text_info[1] if len(text_info) > 1 else 0.0
                    print(f"  Region {i+1}: '{text}' (confidence: {confidence:.2f})")

            return True
        else:
            print("❌ OCR found no text")
            return False

    except Exception as e:
        print(f"❌ OCR processing failed: {e}")
        return False

def test_paddle_engine():
    """Test the PaddleOCREngine class"""
    print("\nTesting PaddleOCREngine class...")
    try:
        from src.engines.paddle_engine import PaddleOCREngine

        engine = PaddleOCREngine()
        success = engine.initialize()

        if success:
            print("✅ PaddleOCREngine initialization successful")
            return engine
        else:
            print("❌ PaddleOCREngine initialization failed")
            return None

    except Exception as e:
        print(f"❌ PaddleOCREngine test failed: {e}")
        return None

def main():
    """Main test function"""
    print("🔍 PADDLEOCR DIAGNOSTIC TEST")
    print("="*50)

    # Test 1: Import
    if not test_paddle_import():
        print("\n❌ Cannot proceed without PaddleOCR installed")
        return

    # Test 2: Initialization
    ocr = test_paddle_initialization()
    if not ocr:
        print("\n❌ Cannot proceed without PaddleOCR initialization")
        return

    # Test 3: OCR on sample image
    sample_images = [
        'data/sample_images/ocr_test.jpg',
        'data/sample_images/sample_printed.jpg',
        'data/sample_images/printed_text.png'
    ]

    for image_path in sample_images:
        if os.path.exists(image_path):
            test_paddle_ocr(ocr, image_path)
            break
    else:
        print("❌ No sample images found for testing")

    # Test 4: Engine class
    engine = test_paddle_engine()
    if engine:
        # Test with image
        for image_path in sample_images:
            if os.path.exists(image_path):
                print(f"\nTesting PaddleOCREngine with {image_path}")
                try:
                    result = engine.extract_text(image_path)
                    if result:
                        print(f"✅ Engine extracted {len(result)} text regions")
                        for i, item in enumerate(result[:2]):
                            print(f"  {i+1}: '{item['text']}' (conf: {item['confidence']:.1f})")
                    else:
                        print("❌ Engine returned no results")
                except Exception as e:
                    print(f"❌ Engine test failed: {e}")
                break

    print("\n" + "="*50)
    print("🏁 PADDLEOCR TEST COMPLETE")

if __name__ == "__main__":
    main()
