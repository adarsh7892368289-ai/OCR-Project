#!/usr/bin/env python3
"""
Basic OCR usage examples demonstrating the simplified OCR library.

This file shows:
1. Simple single-engine OCR (PaddleOCR only)
2. Multi-engine OCR (all 4 engines in parallel)
3. Basic preprocessing options
4. Result inspection
"""

import sys
from pathlib import Path
from datetime import datetime
import warnings

# Suppress common warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Configuration
IMAGE_PATH = "tests/images/img6.jpg"
OUTPUT_DIR = "results"

# Add src to path for development
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))


def save_result(result, filename):
    """Save OCR result to a text file"""
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(exist_ok=True)
    
    filepath = output_path / filename
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"OCR Result - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        
        # Basic info
        f.write(f"Success: {result.success}\n")
        f.write(f"Engine(s): {result.engine_used}\n")
        f.write(f"Confidence: {result.confidence:.1%}\n")
        f.write(f"Processing Time: {result.processing_time:.2f}s\n\n")
        
        # Text statistics
        f.write(f"Text Length: {len(result.text)} characters\n")
        f.write(f"Word Count: {result.word_count}\n")
        f.write(f"Line Count: {result.line_count}\n\n")
        
        # Quality metrics if available
        if result.quality_metrics:
            f.write(f"Image Quality Score: {result.quality_metrics.overall_score:.3f}\n")
            f.write(f"Quality Level: {result.quality_metrics.quality_level.value}\n\n")
        
        # Extracted text
        f.write("EXTRACTED TEXT:\n")
        f.write("-" * 60 + "\n")
        f.write(result.text)
        f.write("\n")
    
    return filepath


def example_1_simplest_usage():
    """
    Example 1: Simplest possible OCR
    - Uses default settings
    - Uses PaddleOCR only (single-engine strategy)
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Simplest OCR Usage")
    print("=" * 60)
    
    from advanced_ocr import OCRLibrary
    
    # Create OCR instance with defaults
    ocr = OCRLibrary()
    
    # Process image - uses PaddleOCR only by default
    result = ocr.process_image(IMAGE_PATH)
    
    # Display results
    print(f"\n✓ OCR completed successfully!")
    print(f"  Engine used: {result.engine_used}")
    print(f"  Confidence: {result.confidence:.1%}")
    print(f"  Processing time: {result.processing_time:.2f}s")
    print(f"  Text length: {len(result.text)} characters")
    print(f"  Word count: {result.word_count}")
    
    # Save result
    saved_path = save_result(result, "01_simple_ocr.txt")
    print(f"  Result saved to: {saved_path}")
    
    # Show first 200 characters of extracted text
    print(f"\nExtracted text preview:")
    print(f"  {result.text[:200]}...")


def example_2_multi_engine_ocr():
    """
    Example 2: Multi-Engine OCR
    - Uses all 4 engines in parallel
    - Combines results for best accuracy
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Multi-Engine OCR (All 4 Engines)")
    print("=" * 60)
    
    from advanced_ocr import OCRLibrary, ProcessingOptions, ProcessingStrategy
    
    ocr = OCRLibrary()
    
    # Specify MULTI_ENGINE strategy
    options = ProcessingOptions(
        strategy=ProcessingStrategy.MULTI_ENGINE
    )
    
    print("\nRunning all 4 engines in parallel...")
    result = ocr.process_image(IMAGE_PATH, options)
    
    print(f"\n✓ Multi-engine OCR completed!")
    print(f"  Engines used: {result.engine_used}")
    print(f"  Combined confidence: {result.confidence:.1%}")
    print(f"  Total processing time: {result.processing_time:.2f}s")
    print(f"  Text length: {len(result.text)} characters")
    
    # Show combination details if available
    if "combination_method" in result.metadata:
        print(f"  Combination method: {result.metadata['combination_method']}")
    if "consensus_engines" in result.metadata:
        print(f"  Consensus engines: {result.metadata['consensus_engines']}")
    if "individual_confidences" in result.metadata:
        print(f"\n  Individual engine confidences:")
        for engine, conf in result.metadata['individual_confidences'].items():
            print(f"    {engine}: {conf:.1%}")
    
    saved_path = save_result(result, "02_multi_engine_ocr.txt")
    print(f"\n  Result saved to: {saved_path}")


def example_3_check_available_engines():
    """
    Example 4: Check Available Engines
    - Lists all initialized engines
    - Shows engine information
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Available Engines")
    print("=" * 60)
    
    from advanced_ocr import OCRLibrary
    
    ocr = OCRLibrary()
    
    # Get available engines
    engines = ocr.get_available_engines()
    
    print(f"\nAvailable engines: {len(engines)}")
    for i, engine in enumerate(engines, 1):
        print(f"  {i}. {engine}")
    
    # Get detailed engine info
    engine_info = ocr.get_engine_info()
    
    print("\nEngine Details:")
    for name, info in engine_info.items():
        print(f"\n  {name.upper()}:")
        print(f"    Available: {info['available']}")
        if info['supported_languages']:
            langs = ', '.join(list(info['supported_languages'])[:5])
            print(f"    Languages: {langs}...")


def example_4_result_inspection():
    """
    Example 5: Detailed Result Inspection
    - Shows all available result properties
    - Demonstrates metadata access
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Detailed Result Inspection")
    print("=" * 60)
    
    from advanced_ocr import OCRLibrary, ProcessingOptions
    
    ocr = OCRLibrary()
    
    options = ProcessingOptions(
        enhance_image=True,
        include_regions=True  # Request region information
    )
    
    result = ocr.process_image(IMAGE_PATH, options)
    
    print("\nResult Properties:")
    print(f"  success: {result.success}")
    print(f"  text: '{result.text[:50]}...' ({len(result.text)} chars)")
    print(f"  confidence: {result.confidence:.3f}")
    print(f"  processing_time: {result.processing_time:.2f}s")
    print(f"  engine_used: {result.engine_used}")
    print(f"  word_count: {result.word_count}")
    print(f"  line_count: {result.line_count}")
    print(f"  language: {result.language}")
    print(f"  has_regions: {result.has_regions}")
    
    if result.strategy_used:
        print(f"  strategy_used: {result.strategy_used.value}")
    
    print("\nMetadata:")
    for key, value in result.metadata.items():
        if isinstance(value, (str, int, float, bool)):
            print(f"  {key}: {value}")
    
    print("\nQuality Metrics:")
    if result.quality_metrics:
        qm = result.quality_metrics
        print(f"  overall_score: {qm.overall_score:.3f}")
        print(f"  quality_level: {qm.quality_level.value}")
        print(f"  sharpness_score: {qm.sharpness_score:.3f}")
        print(f"  contrast_score: {qm.contrast_score:.3f}")
        print(f"  brightness_score: {qm.brightness_score:.3f}")
        print(f"  noise_level: {qm.noise_level:.3f}")
        print(f"  needs_enhancement: {qm.needs_enhancement}")
    
    saved_path = save_result(result, "05_detailed_result.txt")
    print(f"\nResult saved to: {saved_path}")


def main():
    """Run all basic usage examples"""
    print("\n" + "=" * 60)
    print("ADVANCED OCR LIBRARY - BASIC USAGE EXAMPLES")
    print("=" * 60)
    print(f"\nImage: {IMAGE_PATH}")
    print(f"Output: {OUTPUT_DIR}/")
    
    # Check if image exists
    if not Path(IMAGE_PATH).exists():
        print(f"\n❌ ERROR: Image not found: {IMAGE_PATH}")
        print("Please update IMAGE_PATH at the top of this file")
        return
    
    try:
        # Run examples
        example_1_simplest_usage()
        example_2_multi_engine_ocr()
        # example_3_check_available_engines()
        # example_4_result_inspection()
        
        # Summary
        print("\n" + "=" * 60)
        print("✓ ALL BASIC EXAMPLES COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nNext Steps:")
        print("  • Check the 'results/' folder for extracted text")
        print("  • Run 'advanced_usage.py' for advanced features")
        print("  • Run 'batch_processing.py' for multiple images")
        print("  • Experiment with different ProcessingOptions")
        print("\nKey Takeaways:")
        print("  • Default: Uses PaddleOCR only (fast, accurate)")
        print("  • MULTI_ENGINE: Uses all 4 engines (best accuracy)")
        print("  • Preprocessing: Improves results for poor quality images")
        print("=" * 60 + "\n")
        
    except ImportError as e:
        print(f"\n❌ ERROR: Could not import OCR library: {e}")
        print("Make sure you're running from the project root directory")
        print("Install required packages: pip install -r requirements.txt")
        
    except Exception as e:
        print(f"\n❌ ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()