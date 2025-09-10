#!/usr/bin/env python3
"""
Test Preprocessing Pipeline to PaddleOCR
Purpose: Load image, apply preprocessing (quality analysis + enhancement), run PaddleOCR, store results
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

def load_test_image(image_path=None):
    """Load test image from specified path or default"""
    if image_path is None:
        # Default to img3.jpg like in preprocessing test
        image_path = project_root / "data" / "sample_images" / "img5.jpg"

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

def save_raw_results(results, output_dir, filename="paddleocr_raw_text.txt"):
    """Save raw OCR text to plain text file with preserved line structure"""
    if not results:
        print("⚠️  No results to save")
        return

    try:
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)

        # Sort results by vertical position (y-coordinate) to maintain line order
        sorted_results = sorted(results, key=lambda r: r.bbox.y if r.bbox else 0)

        # Group text by lines based on vertical position
        lines = []
        current_line = []
        current_y = None
        line_tolerance = 30  # pixels tolerance for same line

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

        # Join all lines with newlines to preserve structure
        raw_text = "\n".join(lines)

        # Save to plain text file
        output_file = output_dir / filename
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(raw_text)

        print(f"💾 Saved raw text ({len(raw_text)} characters, {len(lines)} lines) to {output_file}")

    except Exception as e:
        print(f"❌ Error saving results: {e}")
        import traceback
        traceback.print_exc()

def test_preprocessing_to_paddleocr(image_path=None):
    """Test complete pipeline: preprocessing + PaddleOCR"""
    print("=" * 80)
    print("PREPROCESSING TO PADDLEOCR TEST")
    print("=" * 80)
    print("Purpose: Quality analysis → Enhancement → PaddleOCR → Store results")
    print("-" * 80)

    # Load test image
    image = load_test_image(image_path)
    if image is None:
        print("❌ Cannot proceed without test image")
        return False

    # Create output directory
    output_dir = project_root / "preprocessed_and_paddleocr"
    print(f"📁 Output directory: {output_dir}")

    try:
        # Import required components
        from src.preprocessing.quality_analyzer import QualityAnalyzer
        from src.preprocessing.image_enhancer import AIImageEnhancer
        from src.engines.paddleocr_engine import PaddleOCREngine

        print("\n🔄 Initializing components...")

        # Initialize quality analyzer
        quality_analyzer = QualityAnalyzer()
        print("✅ QualityAnalyzer initialized")

        # Initialize image enhancer
        image_enhancer = AIImageEnhancer()
        print("✅ AIImageEnhancer initialized")

        # Initialize PaddleOCR engine
        paddle_config = {
            "languages": ["en"],
            "gpu": False
        }
        paddle_engine = PaddleOCREngine(paddle_config)
        if not paddle_engine.initialize():
            print("❌ PaddleOCR initialization failed")
            return False
        print("✅ PaddleOCREngine initialized")

        # STAGE 1: Quality Analysis
        print("\n" + "=" * 60)
        print("STAGE 1: QUALITY ANALYSIS")
        print("=" * 60)

        quality_start = time.time()
        quality_metrics = quality_analyzer.analyze_image(image)
        quality_time = time.time() - quality_start

        print(f"  - Overall score: {quality_metrics.overall_score:.3f}")
        print(f"  - Sharpness score: {quality_metrics.sharpness_score:.3f}")
        print(f"  - Contrast score: {quality_metrics.contrast_score:.3f}")
        print(f"  - Brightness score: {quality_metrics.brightness_score:.3f}")
        print(f"  - Noise level: {quality_metrics.noise_level:.3f}")
        print(f"  - Image type: {quality_metrics.image_type}")
        print(f"  - Quality level: {quality_metrics.quality_level}")

        # STAGE 2: Conditional Enhancement
        print("\n" + "=" * 60)
        print("STAGE 2: CONDITIONAL ENHANCEMENT")
        print("=" * 60)

        enhancement_start = time.time()

        # Use smart enhancement (conditional based on quality)
        enhancement_result = image_enhancer.smart_enhance_image(image, quality_metrics)
        enhanced_image = enhancement_result.enhanced_image
        enhancement_skipped = enhancement_result.enhancement_applied == "skipped"

        enhancement_time = time.time() - enhancement_start

        if enhancement_skipped:
            print(f"Enhancement SKIPPED")
            print(f"  - Reason: {enhancement_result.parameters_used.get('skip_reason', 'N/A')}")
        else:
            print(f"Enhancement APPLIED")
            print(f"  - Strategy: {enhancement_result.enhancement_applied}")
            print(f"  - Operations: {enhancement_result.operations_performed}")
        print(f"  - Enhanced shape: {enhanced_image.shape}")

        # Save enhanced image for reference
        enhanced_path = output_dir / "enhanced_image.jpg"
        output_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(enhanced_path), enhanced_image)
        print(f"💾 Enhanced image saved to: {enhanced_path}")

        # STAGE 3: PaddleOCR Processing
        print("\n" + "=" * 60)
        print("STAGE 3: PADDLEOCR PROCESSING")
        print("=" * 60)

        ocr_start = time.time()
        ocr_results = paddle_engine.process_image(enhanced_image)
        ocr_time = time.time() - ocr_start

        print(f"  - Processing time: {ocr_time:.3f}s")
        print(f"  - Results extracted: {len(ocr_results)}")

        if ocr_results:
            # Show sample results
            print("  - Sample results:")
            for i, result in enumerate(ocr_results[:5]):  # Show first 5
                bbox = result.bbox
                print(f"    {i+1}. '{result.text}' at ({bbox.x:.0f}, {bbox.y:.0f})")

        # STAGE 4: Save Results
        print("\n" + "=" * 60)
        print("STAGE 4: SAVE RESULTS")
        print("=" * 60)

        save_raw_results(ocr_results, output_dir)

        # Save additional metadata
        metadata = {
            "image_path": str(image_path) if image_path else "default",
            "original_shape": image.shape,
            "enhanced_shape": enhanced_image.shape,
            "quality_metrics": {
                "overall_score": quality_metrics.overall_score,
                "sharpness_score": quality_metrics.sharpness_score,
                "contrast_score": quality_metrics.contrast_score,
                "brightness_score": quality_metrics.brightness_score,
                "noise_level": quality_metrics.noise_level,
                "image_type": str(quality_metrics.image_type),
                "quality_level": str(quality_metrics.quality_level)
            },
            "enhancement": {
                "applied": enhancement_result.enhancement_applied,
                "skipped": enhancement_skipped,
                "operations": enhancement_result.operations_performed,
                "quality_improvement": enhancement_result.quality_improvement,
                "processing_time": enhancement_result.processing_time
            },
            "paddleocr": {
                "results_count": len(ocr_results),
                "processing_time": ocr_time
            },
            "total_processing_time": quality_time + enhancement_time + ocr_time,
            "timestamp": time.time()
        }

        metadata_file = output_dir / "processing_metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, default=str)
        print(f"💾 Metadata saved to: {metadata_file}")

        # Performance Summary
        total_time = quality_time + enhancement_time + ocr_time

        print("\n" + "=" * 60)
        print("PERFORMANCE SUMMARY")
        print("=" * 60)
        print(f"  - Quality analysis: {quality_time:.3f}s")
        print(f"  - Enhancement: {enhancement_time:.3f}s")
        print(f"  - PaddleOCR: {ocr_time:.3f}s")
        print(f"  - Total time: {total_time:.3f}s")

        # List saved files
        saved_files = list(output_dir.glob("*"))
        if saved_files:
            print("\n📄 Saved files:")
            for file in saved_files:
                file_size = file.stat().st_size
                print(f"   📄 {file.name} ({file_size} bytes)")

        return len(ocr_results) > 0

    except Exception as e:
        print(f"❌ Error in preprocessing to PaddleOCR test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description='Test preprocessing pipeline to PaddleOCR')
    parser.add_argument('--image', type=str, help='Path to input image (optional, uses default if not provided)')

    args = parser.parse_args()

    image_path = None
    if args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"❌ Specified image not found: {image_path}")
            return False

    print("🧪 Testing preprocessing pipeline to PaddleOCR...")

    try:
        success = test_preprocessing_to_paddleocr(image_path)

        if success:
            print("\n✅ Test completed successfully!")
            print("📊 Preprocessing → Enhancement → PaddleOCR pipeline executed")
            print("💾 Results saved to preprocessed_and_paddleocr/ folder")
        else:
            print("\n⚠️  Test completed but no OCR results extracted.")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
