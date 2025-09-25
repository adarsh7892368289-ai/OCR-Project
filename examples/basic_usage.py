#!/usr/bin/env python3
"""
Simple OCR Usage Examples
Edit the paths at the top and run this script
"""

import sys
from pathlib import Path

# === EDIT THESE PATHS ===
IMAGE_PATH = "tests/images/img2.jpg"
OUTPUT_DIR = "results"

# Add your OCR library to Python path (adjust if needed)
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

def example_1_basic_ocr():
    """Example 1: Basic text extraction"""
    print("Example 1: Basic OCR")
    print("-" * 30)
    
    # Import your library
    from advanced_ocr import OCRLibrary
    
    # Initialize and process
    ocr = OCRLibrary()
    result = ocr.process_image(IMAGE_PATH)
    
    print(f"Success: {result.success}")
    print(f"Text length: {len(result.text)} characters")
    print(f"Confidence: {result.confidence:.1%}")
    print(f"Engine: {result.engine_used}")
    print(f"Time: {result.processing_time:.2f}s")
    
    print("\nExtracted text:")
    print("-" * 20)
    print(result.text)
    
    # Save to file
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(exist_ok=True)
    
    with open(output_path / "basic_result.txt", "w", encoding="utf-8") as f:
        f.write(result.text)
    
    print(f"\nSaved to: {output_path}/basic_result.txt")
    return result

def example_2_receipt_processing():
    """Example 2: Receipt/structured document processing"""
    print("\nExample 2: Receipt Processing")
    print("-" * 30)
    
    from advanced_ocr import OCRLibrary, ProcessingOptions
    
    ocr = OCRLibrary()
    
    # Configure for receipts/invoices
    options = ProcessingOptions(
        engines=['paddleocr'],      # Good for structured text
        preserve_formatting=True,   # Keep structure
        include_regions=True,      # Get regions
        enhance_image=True
    )
    
    result = ocr.process_image(IMAGE_PATH, options)
    
    print(f"Regions found: {len(result.regions)}")
    print(f"Confidence: {result.confidence:.1%}")
    
    # Show structured lines
    lines = result.text.strip().split('\n')
    print(f"\nFound {len(lines)} lines:")
    for i, line in enumerate(lines[:10]):  # First 10 lines
        if line.strip():
            print(f"{i+1:2d}: {line}")
    
    return result

def example_3_multiple_engines():
    """Example 3: Try different engines"""
    print("\nExample 3: Multiple Engines")
    print("-" * 30)
    
    from advanced_ocr import OCRLibrary, ProcessingOptions
    
    ocr = OCRLibrary()
    available_engines = ocr.get_available_engines()
    
    print(f"Available engines: {available_engines}")
    
    # Test each engine
    for engine in available_engines[:2]:  # Test first 2 engines
        print(f"\nTesting {engine}:")
        
        options = ProcessingOptions(engines=[engine])
        
        try:
            result = ocr.process_image(IMAGE_PATH, options)
            print(f"  Confidence: {result.confidence:.3f}")
            print(f"  Time: {result.processing_time:.2f}s")
            print(f"  Text length: {len(result.text)}")
            print(f"  Preview: {result.text[:50]}...")
        except Exception as e:
            print(f"  Failed: {e}")

def example_4_high_accuracy():
    """Example 4: High accuracy settings"""
    print("\nExample 4: High Accuracy Processing")
    print("-" * 30)
    
    from advanced_ocr import OCRLibrary, ProcessingOptions, ProcessingStrategy
    
    ocr = OCRLibrary()
    
    # High accuracy options
    options = ProcessingOptions(
        engines=['paddleocr', 'easyocr'],  # Multiple engines
        strategy=ProcessingStrategy.ENHANCED,
        enhance_image=True,
        detect_orientation=True,
        correct_rotation=True,
        min_confidence=0.8
    )
    
    result = ocr.process_image(IMAGE_PATH, options)
    
    print(f"Strategy used: {result.strategy_used.value}")
    print(f"Final confidence: {result.confidence:.3f}")
    print(f"Quality score: {result.quality_metrics.overall_score:.3f}")
    print(f"Enhancement applied: {result.quality_metrics.needs_enhancement}")
    
    return result

def main():
    print("OCR LIBRARY USAGE EXAMPLES")
    print("=" * 40)
    print(f"Processing image: {IMAGE_PATH}")
    
    try:
        # Run examples
        example_1_basic_ocr()
        example_2_receipt_processing() 
        example_3_multiple_engines()
        example_4_high_accuracy()
        
        print(f"\nAll examples completed!")
        print(f"Check the '{OUTPUT_DIR}' folder for saved results.")
        
    except FileNotFoundError:
        print(f"ERROR: Image file not found: {IMAGE_PATH}")
        print("Please update IMAGE_PATH in the configuration section")
        
    except ImportError as e:
        print(f"ERROR: Could not import OCR library: {e}")
        print("Make sure you're running from the correct directory")
        
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    main()