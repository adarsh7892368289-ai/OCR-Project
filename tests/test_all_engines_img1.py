#!/usr/bin/env python3
"""
Test All Four Engines with img1.jpg - Raw Data Extraction
Purpose: Extract and store raw OCR data from Tesseract, EasyOCR, PaddleOCR, and TrOCR engines
Author: OCR Testing Framework
Date: 2025
"""

import sys
import os
import time
import numpy as np
import cv2
import json
from typing import List, Dict, Any
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def load_test_image():
    """Load img1.jpg from data/sample_images/"""
    image_path = project_root / "data" / "sample_images" / "img3_enhanced.jpg"

    if not image_path.exists():
        print(f"❌ Test image not found: {image_path}")
        return None

    try:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"❌ Failed to load image: {image_path}")
            return None

        print(f"✅ Loaded test image: {image_path}")
        print(f"📊 Image shape: {image.shape}")

        return image
    except Exception as e:
        print(f"❌ Error loading image: {e}")
        return None

def save_raw_results(engine_name, results, output_dir):
    """Save raw OCR text to plain text file with preserved line structure"""
    if not results:
        print(f"⚠️  No results to save for {engine_name}")
        return

    try:
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)

        # Debug: Print bounding box info for first few results
        if results and len(results) > 0:
            print(f"🔍 Debug {engine_name}: First result bbox type: {type(results[0].bbox)}")
            if hasattr(results[0].bbox, 'y'):
                print(f"🔍 Debug {engine_name}: bbox.y = {results[0].bbox.y}")
            if hasattr(results[0].bbox, 'x'):
                print(f"🔍 Debug {engine_name}: bbox.x = {results[0].bbox.x}")
            if hasattr(results[0].bbox, 'width'):
                print(f"🔍 Debug {engine_name}: bbox.width = {results[0].bbox.width}")
            if hasattr(results[0].bbox, 'height'):
                print(f"🔍 Debug {engine_name}: bbox.height = {results[0].bbox.height}")

        # Sort results by vertical position (y-coordinate) to maintain line order
        sorted_results = sorted(results, key=lambda r: r.bbox.y if r.bbox else 0)

        # Group text by lines based on vertical position
        lines = []
        current_line = []
        current_y = None
        line_tolerance = 30  # pixels tolerance for same line (increased for better grouping)

        for result in sorted_results:
            if not result.text or not result.text.strip():
                continue

            # Get the y-coordinate of the bounding box
            y_pos = result.bbox.y if result.bbox else 0

            # If this is the first result or significantly different y-position, start new line
            if current_y is None or abs(y_pos - current_y) > line_tolerance:
                if current_line:
                    # Join words in current line with spaces
                    line_text = " ".join(current_line)
                    lines.append(line_text)
                current_line = [result.text]
                current_y = y_pos
            else:
                # Add to current line
                current_line.append(result.text)

        # Add the last line
        if current_line:
            line_text = " ".join(current_line)
            lines.append(line_text)

        # Join all lines with newlines to preserve receipt structure
        raw_text = "\n".join(lines)

        # Save to plain text file
        output_file = output_dir / f"{engine_name.lower()}_raw_text.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(raw_text)

        print(f"💾 Saved raw text ({len(raw_text)} characters, {len(lines)} lines) to {output_file}")

    except Exception as e:
        print(f"❌ Error saving results for {engine_name}: {e}")
        import traceback
        traceback.print_exc()

def test_engine_raw(engine_class, engine_name, config, image, output_dir):
    """Test a single OCR engine and save raw results"""
    print(f"\n{'='*60}")
    print(f"EXTRACTING RAW DATA FROM {engine_name.upper()}")
    print(f"{'='*60}")

    start_time = time.time()
    success = False
    results = None

    try:
        # Initialize engine
        print(f"🔄 Initializing {engine_name}...")
        engine = engine_class(config)

        if engine.initialize():
            print(f"✅ {engine_name} initialized")

            # Process image
            print(f"🔍 Processing image with {engine_name}...")
            process_start = time.time()
            results = engine.process_image(image)
            process_time = time.time() - process_start

            if isinstance(results, list) and len(results) > 0:
                print(f"✅ Extracted {len(results)} raw OCR results")
                print(f"⏱️  Processing time: {process_time:.3f}s")

                # Save raw results
                save_raw_results(engine_name, results, output_dir)

                success = True
            else:
                print("❌ No OCR results extracted")
        else:
            print(f"❌ {engine_name} initialization failed")

    except Exception as e:
        print(f"❌ Error testing {engine_name}: {e}")
        import traceback
        traceback.print_exc()

    total_time = time.time() - start_time
    print(f"⏱️  Total time: {total_time:.3f}s")

    return success, results, total_time

def main():
    """Main function to extract raw OCR data from all four engines"""
    print("=" * 80)
    print("RAW OCR DATA EXTRACTION - ALL FOUR ENGINES WITH IMG1.JPG")
    print("=" * 80)
    print("Purpose: Extract and store raw OCR data from all engines")
    print("Image: data/sample_images/img1.jpg")
    print("-" * 80)

    # Create output directory for raw results
    output_dir = project_root / "ocr_raw_results_img3"
    print(f"📁 Output directory: {output_dir}")

    # Load test image
    image = load_test_image()
    if image is None:
        print("❌ Cannot proceed without test image")
        return False

    # Define engine configurations
    engine_configs = {
        'Tesseract': {
            'class': None,
            'config': {},
            'name': 'Tesseract'
        },
        'EasyOCR': {
            'class': None,
            'config': {
                "languages": ["en"],
                "gpu": False
            },
            'name': 'EasyOCR'
        },
        'PaddleOCR': {
            'class': None,
            'config': {
                "languages": ["en"],
                "gpu": False
            },
            'name': 'PaddleOCR'
        },
        'TrOCR': {
            'class': None,
            'config': {
                "device": "cpu",
                "model_name": "microsoft/trocr-base-printed"
            },
            'name': 'TrOCR'
        }
    }

    # Import engine classes
    print("\n🔍 Importing engine classes...")

    try:
        from src.engines.tesseract_engine import TesseractEngine
        engine_configs['Tesseract']['class'] = TesseractEngine
        print("✅ TesseractEngine imported")

        from src.engines.easyocr_engine import EasyOCREngine
        engine_configs['EasyOCR']['class'] = EasyOCREngine
        print("✅ EasyOCREngine imported")

        from src.engines.paddleocr_engine import PaddleOCREngine
        engine_configs['PaddleOCR']['class'] = PaddleOCREngine
        print("✅ PaddleOCREngine imported")

        from src.engines.trocr_engine import TrOCREngine
        engine_configs['TrOCR']['class'] = TrOCREngine
        print("✅ TrOCREngine imported")

    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("💡 Make sure all required packages are installed:")
        print("   pip install pytesseract easyocr paddlepaddle transformers torch")
        return False

    # Test each engine and extract raw data
    overall_start_time = time.time()
    test_results = {}

    for engine_key, engine_info in engine_configs.items():
        if engine_info['class'] is None:
            print(f"⚠️  Skipping {engine_key} - class not imported")
            test_results[engine_key] = {
                'success': False,
                'results': None,
                'time': 0.0
            }
            continue

        success, results, test_time = test_engine_raw(
            engine_info['class'],
            engine_info['name'],
            engine_info['config'],
            image,
            output_dir
        )

        test_results[engine_key] = {
            'success': success,
            'results': results,
            'time': test_time
        }

    # Summary
    total_time = time.time() - overall_start_time
    successful_engines = sum(1 for r in test_results.values() if r['success'])
    total_engines = len(test_results)

    print(f"\n{'='*80}")
    print("RAW DATA EXTRACTION SUMMARY")
    print(f"{'='*80}")
    print(f"📊 Engines processed: {total_engines}")
    print(f"✅ Successful extractions: {successful_engines}")
    print(f"❌ Failed extractions: {total_engines - successful_engines}")
    print(f"⏱️  Total time: {total_time:.3f}s")
    print(f"📁 Raw results saved to: {output_dir}")

    print(f"\n📋 EXTRACTION RESULTS:")
    for engine_name, result in test_results.items():
        status = "✅" if result['success'] else "❌"
        time_str = f"{result['time']:.3f}s"
        if result['success'] and result['results']:
            result_count = len(result['results'])
            print(f"   {status} {engine_name}: {result_count} raw results extracted ({time_str})")
        else:
            print(f"   {status} {engine_name}: Failed ({time_str})")

    # List saved files
    if output_dir.exists():
        txt_files = list(output_dir.glob("*.txt"))
        if txt_files:
            print(f"\n📄 Saved raw text files:")
            for txt_file in txt_files:
                file_size = txt_file.stat().st_size
                print(f"   📄 {txt_file.name} ({file_size} bytes)")

    if successful_engines == total_engines:
        print(f"\n🎉 ALL ENGINES PROCESSED SUCCESSFULLY!")
        print("✅ Raw OCR data extracted from all four engines")
        print(f"📊 Check {output_dir} for JSON files containing raw results")
    elif successful_engines > 0:
        print(f"\n⚠️  PARTIAL SUCCESS: {successful_engines}/{total_engines} engines extracted data")
        print("🔧 Check failed engines for installation issues")
    else:
        print(f"\n❌ ALL ENGINES FAILED!")
        print("🔧 Check dependencies and configurations")

    return successful_engines > 0

if __name__ == "__main__":
    success = main()

    if success:
        print(f"\n🎉 Raw data extraction completed successfully")
        print("📊 Raw OCR results saved to JSON files")
    else:
        print(f"\n💥 Raw data extraction failed")
        print("🔧 Check error messages above")

    sys.exit(0 if success else 1)
