#!/usr/bin/env python3
"""
Basic OCR Usage Examples
Demonstrates simple, everyday usage of the advanced-ocr library
"""

import sys
from pathlib import Path
from datetime import datetime

# === EDIT THESE PATHS ===
IMAGE_PATH = "tests/images/img3.jpg"
OUTPUT_DIR = "results"

# Add your OCR library to Python path (adjust if needed)
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

def save_result(result, filename):
    """Save OCR result to file"""
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(exist_ok=True)
    
    filepath = output_path / filename
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"OCR Result - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Success: {result.success}\n")
        f.write(f"Engine: {result.engine_used}\n")
        f.write(f"Confidence: {result.confidence:.1%}\n")
        f.write(f"Time: {result.processing_time:.2f}s\n")
        f.write(f"Text Length: {len(result.text)} characters\n")
        f.write(f"Word Count: {result.word_count}\n")
        f.write(f"Line Count: {result.line_count}\n\n")
        f.write("EXTRACTED TEXT:\n")
        f.write("-" * 20 + "\n")
        f.write(result.text)
    
    return filepath

def example_1_simple_ocr():
    """Simplest possible OCR usage"""
    print("Example 1: Simple OCR")
    print("-" * 25)
    
    from advanced_ocr import OCRLibrary
    
    # Just create and use - uses default settings
    ocr = OCRLibrary()
    result = ocr.process_image(IMAGE_PATH)
    
    print(f"Text extracted: {len(result.text)} characters")
    print(f"Confidence: {result.confidence:.1%}")
    print(f"Engine used: {result.engine_used}")
    print(f"Processing time: {result.processing_time:.2f}s")
    
    # Save result
    saved_path = save_result(result, "simple_ocr_result.txt")
    print(f"Saved to: {saved_path}")

def example_2_check_available_engines():
    """See what OCR engines are available"""
    print("\nExample 2: Available Engines")
    print("-" * 30)
    
    from advanced_ocr import OCRLibrary
    
    ocr = OCRLibrary()
    engines = ocr.get_available_engines()
    
    print(f"Available engines: {engines}")
    print(f"Total engines: {len(engines)}")
    
    # Test each engine individually
    for engine in engines[:2]:  
        print(f"\nTesting {engine}:")
        try:
            from advanced_ocr import ProcessingOptions
            options = ProcessingOptions(engines=[engine])
            result = ocr.process_image(IMAGE_PATH, options)
            print(f"  Success: {result.success}")
            print(f"  Confidence: {result.confidence:.1%}")
            print(f"  Time: {result.processing_time:.2f}s")
            saved_path = save_result(result, f"{engine}_result.txt")
        except Exception as e:
            print(f"  Failed: {e}")

def example_3_basic_options():
    """Basic processing options"""
    print("\nExample 3: Basic Processing Options")
    print("-" * 35)
    
    from advanced_ocr import OCRLibrary, ProcessingOptions
    
    ocr = OCRLibrary()
    
    # Basic enhancement options
    options = ProcessingOptions(
        enhance_image=True,
        detect_orientation=True,
        correct_rotation=True,
        min_confidence=0.3  # Lower threshold for better results
    )
    
    result = ocr.process_image(IMAGE_PATH, options)
    
    print(f"Enhancement enabled: {options.enhance_image}")
    print(f"Orientation detection: {options.detect_orientation}")
    print(f"Rotation correction: {options.correct_rotation}")
    print(f"Min confidence: {options.min_confidence}")
    print(f"Result confidence: {result.confidence:.1%}")
    
    saved_path = save_result(result, "basic_options_result.txt")
    print(f"Saved to: {saved_path}")

def example_4_language_options():
    """Language specification options"""
    print("\nExample 4: Language Options")
    print("-" * 27)
    
    from advanced_ocr import OCRLibrary, ProcessingOptions
    
    ocr = OCRLibrary()
    
    # Specify languages
    options = ProcessingOptions(
        languages=['en'],  # English
        enhance_image=True
    )
    
    result = ocr.process_image(IMAGE_PATH, options)
    
    print(f"Languages specified: {options.languages}")
    print(f"Detected language: {result.language}")
    print(f"All detected languages: {result.detected_languages}")
    print(f"Confidence: {result.confidence:.1%}")
    
    saved_path = save_result(result, "language_options_result.txt")
    print(f"Saved to: {saved_path}")

def example_5_result_details():
    """Examine detailed result properties"""
    print("\nExample 5: Result Details")
    print("-" * 25)
    
    from advanced_ocr import OCRLibrary, ProcessingOptions
    
    ocr = OCRLibrary()
    options = ProcessingOptions(include_regions=True)
    result = ocr.process_image(IMAGE_PATH, options)
    
    print("Result Properties:")
    print(f"  Success: {result.success}")
    print(f"  Text length: {len(result.text)} characters")
    print(f"  Word count: {result.word_count}")
    print(f"  Line count: {result.line_count}")
    print(f"  Confidence: {result.confidence:.3f}")
    print(f"  Engine: {result.engine_used}")
    print(f"  Language: {result.language}")
    print(f"  Has regions: {result.has_regions}")
    print(f"  Processing time: {result.processing_time:.2f}s")
    
    # Show strategy if available
    if hasattr(result, 'strategy_used'):
        print(f"  Strategy used: {result.strategy_used.value}")
    
    # Show quality metrics if available
    if result.quality_metrics:
        print(f"  Quality score: {result.quality_metrics.overall_score:.3f}")
        print(f"  Quality level: {result.quality_metrics.quality_level.value}")
        print(f"  Enhancement needed: {result.quality_metrics.needs_enhancement}")
    
    saved_path = save_result(result, "detailed_result.txt")
    print(f"Saved to: {saved_path}")

def main():
    print("BASIC OCR USAGE EXAMPLES")
    print("=" * 30)
    print(f"Processing image: {IMAGE_PATH}")
    print(f"Results saved to: {OUTPUT_DIR}/")
    
    try:
        example_1_simple_ocr()
        example_2_check_available_engines()
        example_3_basic_options()
        example_4_language_options()
        example_5_result_details()
        
        print(f"\n" + "=" * 40)
        print("BASIC EXAMPLES COMPLETED!")
        print("=" * 40)
        print("Next steps:")
        print("- Run advanced_usage.py for processing strategies")
        print("- Run batch_processing.py for multiple images")
        
    except FileNotFoundError:
        print(f"ERROR: Image file not found: {IMAGE_PATH}")
        print("Please update IMAGE_PATH at the top of this file")
        
    except ImportError as e:
        print(f"ERROR: Could not import OCR library: {e}")
        print("Make sure you're in the project root directory")
        
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    main()