#!/usr/bin/env python3
"""
Test Script: Tesseract, EasyOCR, and PaddleOCR on img3 Sample Image
Purpose: Test three OCR engines on the img3 sample image from data/sample_images
Author: OCR Testing Framework
Date: 2025
"""

import sys
import os
import time
import cv2
import numpy as np
import json
from typing import List, Dict, Any
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def load_img3_sample():
    """Load img3 sample image from data/sample_images"""
    sample_dir = project_root / "data" / "sample_images"

    # Try img3.png first, then img3.jpg
    img3_png = sample_dir / "img3.png"
    img3_jpg = sample_dir / "img3.jpg"

    image_path = None
    if img3_png.exists():
        image_path = img3_png
    elif img3_jpg.exists():
        image_path = img3_jpg
    else:
        raise FileNotFoundError("img3.png or img3.jpg not found in data/sample_images")

    print(f"Loading image: {image_path}")
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    print(f"Image loaded: {image.shape}")
    return image

def test_tesseract_on_img3():
    """Test Tesseract engine on img3"""
    print("=" * 80)
    print("TESTING TESSERACT ENGINE ON IMG3")
    print("=" * 80)

    try:
        from src.engines.tesseract_engine import TesseractEngine, find_tesseract

        # Check if Tesseract is installed
        tesseract_path = find_tesseract()
        if not tesseract_path:
            print("❌ Tesseract not found - Please install Tesseract OCR")
            return None

        # Initialize engine
        engine = TesseractEngine()
        if not engine.initialize():
            print("❌ Tesseract engine initialization failed")
            return None

        print("✅ Tesseract engine initialized")

        # Load img3
        image = load_img3_sample()

        # Run OCR
        start_time = time.time()
        results = engine.process_image(image)
        processing_time = time.time() - start_time

        print(f"⏱️  Processing time: {processing_time:.3f}s")
        print(f"📄 Results count: {len(results)}")

        # Print results
        if results:
            print("\n🔤 TESSERACT OCR RESULTS:")
            for i, result in enumerate(results[:10]):  # Show first 10 results
                print(f"   [{i+1}] '{result.text}' (conf: {result.confidence:.3f})")
            if len(results) > 10:
                print(f"   ... and {len(results) - 10} more results")

            # Calculate average confidence
            avg_conf = sum(r.confidence for r in results) / len(results)
            print(f"🎯 Average confidence: {avg_conf:.3f}")
        else:
            print("❌ No OCR results from Tesseract")

        return results

    except Exception as e:
        print(f"❌ Tesseract test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_easyocr_on_img3():
    """Test EasyOCR engine on img3"""
    print("\n" + "=" * 80)
    print("TESTING EASYOCR ENGINE ON IMG3")
    print("=" * 80)

    try:
        from src.engines.easyocr_engine import EasyOCREngine

        # Initialize engine with English only for faster startup
        config = {
            "languages": ["en"],
            "gpu": False,  # Use CPU for consistent testing
            "model_dir": None
        }

        engine = EasyOCREngine(config)
        print("🔄 Initializing EasyOCR (may download models on first run)...")

        if not engine.initialize():
            print("❌ EasyOCR engine initialization failed")
            return None

        print("✅ EasyOCR engine initialized")

        # Load img3
        image = load_img3_sample()

        # Run OCR
        start_time = time.time()
        results = engine.process_image(image)
        processing_time = time.time() - start_time

        print(f"⏱️  Processing time: {processing_time:.3f}s")
        print(f"📄 Results count: {len(results)}")

        # Print results
        if results:
            print("\n🔤 EASYOCR OCR RESULTS:")
            for i, result in enumerate(results[:10]):  # Show first 10 results
                print(f"   [{i+1}] '{result.text}' (conf: {result.confidence:.3f})")
            if len(results) > 10:
                print(f"   ... and {len(results) - 10} more results")

            # Calculate average confidence
            avg_conf = sum(r.confidence for r in results) / len(results)
            print(f"🎯 Average confidence: {avg_conf:.3f}")
        else:
            print("❌ No OCR results from EasyOCR")

        return results

    except Exception as e:
        print(f"❌ EasyOCR test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_paddleocr_on_img3():
    """Test PaddleOCR engine on img3"""
    print("\n" + "=" * 80)
    print("TESTING PADDLEOCR ENGINE ON IMG3")
    print("=" * 80)

    try:
        from src.engines.paddleocr_engine import PaddleOCREngine

        # Initialize engine
        config = {
            "use_gpu": False,  # Use CPU for consistent testing
            "lang": "en"
        }

        engine = PaddleOCREngine(config)
        print("🔄 Initializing PaddleOCR...")

        if not engine.initialize():
            print("❌ PaddleOCR engine initialization failed")
            return None

        print("✅ PaddleOCR engine initialized")

        # Load img3
        image = load_img3_sample()

        # Run OCR
        start_time = time.time()
        results = engine.process_image(image)
        processing_time = time.time() - start_time

        print(f"⏱️  Processing time: {processing_time:.3f}s")
        print(f"📄 Results count: {len(results)}")

        # Print results
        if results:
            print("\n🔤 PADDLEOCR OCR RESULTS:")
            for i, result in enumerate(results[:10]):  # Show first 10 results
                print(f"   [{i+1}] '{result.text}' (conf: {result.confidence:.3f})")
            if len(results) > 10:
                print(f"   ... and {len(results) - 10} more results")

            # Calculate average confidence
            avg_conf = sum(r.confidence for r in results) / len(results)
            print(f"🎯 Average confidence: {avg_conf:.3f}")
        else:
            print("❌ No OCR results from PaddleOCR")

        return results

    except Exception as e:
        print(f"❌ PaddleOCR test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def save_results_to_file(tesseract_results, easyocr_results, paddleocr_results):
    """Save raw OCR results to a JSON file"""
    def serialize_result(result):
        """Convert result object to JSON-serializable dict"""
        return {
            "text": str(result.text),
            "confidence": float(result.confidence),
            "bbox": {
                "x": int(result.bbox.x) if result.bbox else None,
                "y": int(result.bbox.y) if result.bbox else None,
                "width": int(result.bbox.width) if result.bbox else None,
                "height": int(result.bbox.height) if result.bbox else None
            } if result.bbox else None,
            "metadata": result.metadata
        }

    results_data = {
        "test_info": {
            "image": "img3.png",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "engines_tested": ["tesseract", "easyocr", "paddleocr"]
        },
        "tesseract": {
            "results_count": len(tesseract_results) if tesseract_results else 0,
            "average_confidence": float(sum(r.confidence for r in tesseract_results) / len(tesseract_results)) if tesseract_results else 0.0,
            "raw_results": [serialize_result(r) for r in tesseract_results] if tesseract_results else []
        },
        "easyocr": {
            "results_count": len(easyocr_results) if easyocr_results else 0,
            "average_confidence": float(sum(r.confidence for r in easyocr_results) / len(easyocr_results)) if easyocr_results else 0.0,
            "raw_results": [serialize_result(r) for r in easyocr_results] if easyocr_results else []
        },
        "paddleocr": {
            "results_count": len(paddleocr_results) if paddleocr_results else 0,
            "average_confidence": float(sum(r.confidence for r in paddleocr_results) / len(paddleocr_results)) if paddleocr_results else 0.0,
            "raw_results": [serialize_result(r) for r in paddleocr_results] if paddleocr_results else []
        }
    }

    output_file = project_root / "ocr_test_results_img3.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Raw results saved to: {output_file}")

def compare_results(tesseract_results, easyocr_results, paddleocr_results):
    """Compare results from all three engines"""
    print("\n" + "=" * 80)
    print("COMPARISON: TESSERACT VS EASYOCR VS PADDLEOCR ON IMG3")
    print("=" * 80)

    engines = {
        "Tesseract": tesseract_results,
        "EasyOCR": easyocr_results,
        "PaddleOCR": paddleocr_results
    }

    for name, results in engines.items():
        if results:
            count = len(results)
            avg_conf = sum(r.confidence for r in results) / len(results)
            print(f"📊 {name}: {count} results, avg conf: {avg_conf:.3f}")
        else:
            print(f"📊 {name}: Failed or no results")

    # Find the best performing engine
    valid_engines = [(name, results) for name, results in engines.items() if results]
    if len(valid_engines) >= 2:
        best_engine = max(valid_engines, key=lambda x: sum(r.confidence for r in x[1]) / len(x[1]))
        print(f"🏆 {best_engine[0]} has the highest average confidence")

def main():
    """Main test function"""
    print("🧪 TESTING TESSERACT, EASYOCR, AND PADDLEOCR ON IMG3 SAMPLE IMAGE")
    print("📁 Image location: data/sample_images/img3.png or img3.jpg")
    print("-" * 80)

    # Test Tesseract
    tesseract_results = test_tesseract_on_img3()

    # Test EasyOCR
    easyocr_results = test_easyocr_on_img3()

    # Test PaddleOCR
    paddleocr_results = test_paddleocr_on_img3()

    # Compare results
    compare_results(tesseract_results, easyocr_results, paddleocr_results)

    # Save results to file
    save_results_to_file(tesseract_results, easyocr_results, paddleocr_results)

    print("\n" + "=" * 80)
    print("TEST COMPLETED")
    print("=" * 80)

if __name__ == "__main__":
    main()
