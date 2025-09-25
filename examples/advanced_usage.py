#!/usr/bin/env python3
"""
Advanced OCR Usage Examples
Demonstrates processing strategies, quality analysis, and advanced options
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

def save_advanced_result(result, filename, options=None):
    """Save OCR result with advanced metadata"""
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(exist_ok=True)
    
    filepath = output_path / filename
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"Advanced OCR Result - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")
        
        # Basic result info
        f.write("RESULT SUMMARY:\n")
        f.write("-" * 20 + "\n")
        f.write(f"Success: {result.success}\n")
        f.write(f"Engine: {result.engine_used}\n")
        f.write(f"Confidence: {result.confidence:.3f} ({result.confidence:.1%})\n")
        f.write(f"Processing Time: {result.processing_time:.2f}s\n")
        f.write(f"Text Length: {len(result.text)} characters\n")
        f.write(f"Word Count: {result.word_count}\n")
        f.write(f"Line Count: {result.line_count}\n")
        
        # Strategy info
        if hasattr(result, 'strategy_used'):
            f.write(f"Strategy Used: {result.strategy_used.value}\n")
        
        # Quality metrics
        if result.quality_metrics:
            f.write(f"\nQUALITY METRICS:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Overall Score: {result.quality_metrics.overall_score:.3f}\n")
            f.write(f"Quality Level: {result.quality_metrics.quality_level.value}\n")
            f.write(f"Sharpness: {result.quality_metrics.sharpness_score:.3f}\n")
            f.write(f"Contrast: {result.quality_metrics.contrast_score:.3f}\n")
            f.write(f"Brightness: {result.quality_metrics.brightness_score:.3f}\n")
            f.write(f"Noise Level: {result.quality_metrics.noise_level:.3f}\n")
            f.write(f"Enhancement Needed: {result.quality_metrics.needs_enhancement}\n")
            f.write(f"Recommended Strategy: {result.quality_metrics.recommended_strategy.value}\n")
            f.write(f"Image Type: {result.quality_metrics.image_type.value}\n")
        
        # Processing options used
        if options:
            f.write(f"\nPROCESSING OPTIONS:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Engines: {options.engines}\n")
            f.write(f"Strategy: {options.strategy.value if options.strategy else 'Auto-detect'}\n")
            f.write(f"Enhance Image: {options.enhance_image}\n")
            f.write(f"Detect Orientation: {options.detect_orientation}\n")
            f.write(f"Correct Rotation: {options.correct_rotation}\n")
            f.write(f"Min Confidence: {options.min_confidence}\n")
            f.write(f"Languages: {options.languages}\n")
            f.write(f"Include Regions: {options.include_regions}\n")
            f.write(f"Parallel Processing: {options.use_parallel_processing}\n")
        
        f.write(f"\n{'='*60}\n\n")
        f.write("EXTRACTED TEXT:\n")
        f.write("-" * 20 + "\n")
        f.write(result.text)
    
    return filepath

def example_1_processing_strategies():
    """Compare different processing strategies"""
    print("Example 1: Processing Strategies")
    print("-" * 32)
    
    from advanced_ocr import OCRLibrary, ProcessingOptions, ProcessingStrategy
    
    ocr = OCRLibrary()
    
    # Test all three strategies
    strategies = [
        ProcessingStrategy.MINIMAL,
        ProcessingStrategy.BALANCED,
        ProcessingStrategy.ENHANCED
    ]
    
    results = {}
    
    for strategy in strategies:
        print(f"\nTesting {strategy.value.upper()} strategy:")
        
        options = ProcessingOptions(strategy=strategy)
        result = ocr.process_image(IMAGE_PATH, options)
        
        print(f"  Confidence: {result.confidence:.3f}")
        print(f"  Time: {result.processing_time:.2f}s")
        print(f"  Text length: {len(result.text)}")
        
        # Save individual strategy result
        filename = f"strategy_{strategy.value}_result.txt"
        saved_path = save_advanced_result(result, filename, options)
        print(f"  Saved to: {saved_path.name}")
        
        results[strategy] = result
    
    # Find best strategy
    best_strategy = max(results.items(), key=lambda x: x[1].confidence)
    print(f"\nBest strategy: {best_strategy[0].value} (confidence: {best_strategy[1].confidence:.3f})")
    
    return results

def example_2_quality_analysis():
    """Detailed image quality analysis"""
    print("\nExample 2: Quality Analysis")
    print("-" * 27)
    
    from advanced_ocr import OCRLibrary, ProcessingOptions, ProcessingStrategy
    
    ocr = OCRLibrary()
    
    # Enhanced processing to get detailed quality metrics
    options = ProcessingOptions(
        strategy=ProcessingStrategy.ENHANCED,
        enhance_image=True,
        include_regions=True
    )
    
    result = ocr.process_image(IMAGE_PATH, options)
    
    if result.quality_metrics:
        metrics = result.quality_metrics
        print(f"Quality Analysis:")
        print(f"  Overall Score: {metrics.overall_score:.3f}/1.0")
        print(f"  Quality Level: {metrics.quality_level.value}")
        print(f"  Image Type: {metrics.image_type.value}")
        print(f"  Enhancement Needed: {metrics.needs_enhancement}")
        print(f"  Recommended Strategy: {metrics.recommended_strategy.value}")
        print(f"  Is Good Quality: {metrics.is_good_quality}")
        
        print(f"\nDetailed Metrics:")
        print(f"  Sharpness: {metrics.sharpness_score:.3f}")
        print(f"  Contrast: {metrics.contrast_score:.3f}")
        print(f"  Brightness: {metrics.brightness_score:.3f}")
        print(f"  Noise Level: {metrics.noise_level:.3f}")
        print(f"  Text Regions: {metrics.text_region_count}")
        print(f"  Estimated DPI: {metrics.estimated_dpi}")
        
        if metrics.enhancement_suggestions:
            print(f"  Suggestions: {', '.join(metrics.enhancement_suggestions)}")
    else:
        print("No quality metrics available")
    
    saved_path = save_advanced_result(result, "quality_analysis_result.txt", options)
    print(f"Saved to: {saved_path.name}")

def example_3_multiple_engines():
    """Use multiple engines for better accuracy"""
    print("\nExample 3: Multiple Engines")
    print("-" * 27)
    
    from advanced_ocr import OCRLibrary, ProcessingOptions
    
    ocr = OCRLibrary()
    available_engines = ocr.get_available_engines()
    
    # Use multiple engines
    options = ProcessingOptions(
        engines=available_engines[:2],  # Use first 2 available engines
        enhance_image=True,
        min_confidence=0.3  # Lower threshold to allow more engines to succeed
    )
    
    print(f"Using engines: {options.engines}")
    
    result = ocr.process_image(IMAGE_PATH, options)
    
    print(f"Selected engine: {result.engine_used}")
    print(f"Final confidence: {result.confidence:.3f}")
    print(f"Processing time: {result.processing_time:.2f}s")
    
    saved_path = save_advanced_result(result, "multiple_engines_result.txt", options)
    print(f"Saved to: {saved_path.name}")

def example_4_performance_tuning():
    """Performance and timeout settings"""
    print("\nExample 4: Performance Tuning")
    print("-" * 30)
    
    from advanced_ocr import OCRLibrary, ProcessingOptions, ProcessingStrategy
    
    ocr = OCRLibrary()
    
    # Performance-optimized settings
    options = ProcessingOptions(
        strategy=ProcessingStrategy.BALANCED,
        max_processing_time=60,  # 60 second timeout
        use_parallel_processing=True,
        early_termination=True,
        early_termination_threshold=0.95,  # Stop if confidence > 95%
        enhance_image=False  # Disable for speed
    )
    
    print("Performance settings:")
    print(f"  Max processing time: {options.max_processing_time}s")
    print(f"  Parallel processing: {options.use_parallel_processing}")
    print(f"  Early termination: {options.early_termination}")
    print(f"  Early termination threshold: {options.early_termination_threshold}")
    
    result = ocr.process_image(IMAGE_PATH, options)
    
    print(f"\nResults:")
    print(f"  Actual processing time: {result.processing_time:.2f}s")
    print(f"  Confidence: {result.confidence:.3f}")
    print(f"  Early termination used: {result.confidence >= options.early_termination_threshold}")
    
    saved_path = save_advanced_result(result, "performance_tuned_result.txt", options)
    print(f"Saved to: {saved_path.name}")

def example_5_detailed_output():
    """Get detailed region and word information"""
    print("\nExample 5: Detailed Output")
    print("-" * 26)
    
    from advanced_ocr import OCRLibrary, ProcessingOptions
    
    ocr = OCRLibrary()
    
    # Request detailed output
    options = ProcessingOptions(
        include_regions=True,
        include_word_boxes=True,
        preserve_formatting=True,
        enhance_image=True
    )
    
    result = ocr.process_image(IMAGE_PATH, options)
    
    print(f"Detailed output:")
    print(f"  Has regions: {result.has_regions}")
    print(f"  Number of regions: {len(result.regions)}")
    print(f"  Formatting preserved: {options.preserve_formatting}")
    
    if result.regions:
        print(f"\nFirst few regions:")
        for i, region in enumerate(result.regions[:5], 1):
            print(f"  Region {i}:")
            print(f"    Text: '{region.text[:50]}...' ")
            print(f"    Confidence: {region.confidence:.3f}")
            print(f"    Type: {region.text_type.value}")
            print(f"    Valid: {region.is_valid}")
            if region.bbox:
                print(f"    Box: ({region.bbox.x}, {region.bbox.y}, {region.bbox.width}, {region.bbox.height})")
    
    saved_path = save_advanced_result(result, "detailed_output_result.txt", options)
    print(f"Saved to: {saved_path.name}")

def example_6_all_options_showcase():
    """Showcase all available ProcessingOptions"""
    print("\nExample 6: All Processing Options")
    print("-" * 34)
    
    from advanced_ocr import OCRLibrary, ProcessingOptions, ProcessingStrategy
    
    ocr = OCRLibrary()
    
    # Show all available options
    options = ProcessingOptions(
        # Engine selection
        engines=['paddleocr'],  # Specify engines
        strategy=ProcessingStrategy.ENHANCED,  # Processing strategy
        
        # Processing settings
        enhance_image=True,
        detect_orientation=True,
        correct_rotation=True,
        
        # Quality thresholds
        min_confidence=0.3,
        early_termination=True,
        early_termination_threshold=0.90,
        
        # Performance settings
        max_processing_time=90,
        use_parallel_processing=True,
        batch_size=1,
        
        # Language settings
        languages=['en'],
        
        # Output settings
        include_regions=True,
        include_word_boxes=True,
        preserve_formatting=True
    )
    
    print("All options configured:")
    print(f"  Engines: {options.engines}")
    print(f"  Strategy: {options.strategy.value}")
    print(f"  Enhancement: {options.enhance_image}")
    print(f"  Orientation detection: {options.detect_orientation}")
    print(f"  Rotation correction: {options.correct_rotation}")
    print(f"  Min confidence: {options.min_confidence}")
    print(f"  Max time: {options.max_processing_time}s")
    print(f"  Languages: {options.languages}")
    print(f"  Include regions: {options.include_regions}")
    print(f"  Include word boxes: {options.include_word_boxes}")
    print(f"  Preserve formatting: {options.preserve_formatting}")
    
    result = ocr.process_image(IMAGE_PATH, options)
    
    print(f"\nResult with all options:")
    print(f"  Success: {result.success}")
    print(f"  Confidence: {result.confidence:.3f}")
    print(f"  Time: {result.processing_time:.2f}s")
    print(f"  Strategy used: {result.strategy_used.value}")
    print(f"  Regions found: {len(result.regions)}")
    
    saved_path = save_advanced_result(result, "all_options_result.txt", options)
    print(f"Saved to: {saved_path.name}")

def main():
    print("ADVANCED OCR USAGE EXAMPLES")
    print("=" * 35)
    print(f"Processing image: {IMAGE_PATH}")
    print(f"Results saved to: {OUTPUT_DIR}/")
    
    try:
        example_1_processing_strategies()
        example_2_quality_analysis()
        example_3_multiple_engines()
        example_4_performance_tuning()
        example_5_detailed_output()
        example_6_all_options_showcase()
        
        print(f"\n" + "=" * 45)
        print("ADVANCED EXAMPLES COMPLETED!")
        print("=" * 45)
        print("Key takeaways:")
        print("- ProcessingStrategy.ENHANCED for poor quality images")
        print("- Multiple engines improve accuracy")
        print("- Quality metrics help understand image issues")
        print("- Performance tuning prevents long waits")
        print("- Detailed output provides region information")
        
    except FileNotFoundError:
        print(f"ERROR: Image file not found: {IMAGE_PATH}")
        print("Please update IMAGE_PATH at the top of this file")
        
    except ImportError as e:
        print(f"ERROR: Could not import OCR library: {e}")
        
    except Exception as e:
        print(f"ERROR: {e}")

if __name__ == "__main__":
    main()