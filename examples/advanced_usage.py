#!/usr/bin/env python3
"""
Advanced OCR Usage Examples
Comprehensive examples showing all features of your OCR library
Edit the paths below and run: python advanced_examples.py
"""

import sys
from pathlib import Path
import time
import json

# === CONFIGURATION - EDIT THESE PATHS ===
SAMPLE_IMAGES = {
    "receipt": "tests/images/img1.jpg",          # Receipt/invoice image
    "document": "tests/images/img1.jpg",         # Clean document image  
    "poor_quality": "tests/images/img1.jpg",     # Blurry/low quality image
    "handwritten": "tests/images/img1.jpg",      # Handwritten text (if available)
    "multi_lang": "tests/images/img1.jpg",       # Multi-language document
}

OUTPUT_DIR = "advanced_results"
RUN_ALL_EXAMPLES = True  # Set to False to choose which examples to run

# Add OCR library to path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

def setup_output():
    """Create output directory and return path"""
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(exist_ok=True)
    return output_path

def save_result(result, filename, output_dir, extra_info=None):
    """Save OCR result to file with metadata"""
    output_file = output_dir / f"{filename}.txt"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"OCR RESULT: {filename}\n")
        f.write("=" * 50 + "\n\n")
        
        # Metadata
        f.write(f"Engine: {result.engine_used}\n")
        f.write(f"Confidence: {result.confidence:.3f}\n")
        f.write(f"Processing Time: {result.processing_time:.2f}s\n")
        f.write(f"Text Length: {len(result.text)} characters\n")
        
        if hasattr(result, 'strategy_used'):
            f.write(f"Strategy: {result.strategy_used.value}\n")
        
        if hasattr(result, 'quality_metrics') and result.quality_metrics:
            f.write(f"Quality Score: {result.quality_metrics.overall_score:.3f}\n")
            f.write(f"Enhancement Needed: {result.quality_metrics.needs_enhancement}\n")
        
        if extra_info:
            f.write(f"\nAdditional Info:\n")
            for key, value in extra_info.items():
                f.write(f"{key}: {value}\n")
        
        f.write(f"\nExtracted Text:\n")
        f.write("-" * 20 + "\n")
        f.write(result.text)
    
    print(f"Saved: {output_file}")
    return output_file

def example_1_basic_processing(output_dir):
    """Example 1: Basic OCR with automatic settings"""
    print("\n1. BASIC PROCESSING")
    print("-" * 40)
    
    from advanced_ocr import OCRLibrary
    
    ocr = OCRLibrary()
    
    # Process with default settings
    image_path = SAMPLE_IMAGES["document"]
    print(f"Processing: {image_path}")
    
    start_time = time.time()
    result = ocr.process_image(image_path)
    processing_time = time.time() - start_time
    
    print(f"Success: {result.success}")
    print(f"Confidence: {result.confidence:.3f}")
    print(f"Engine: {result.engine_used}")
    print(f"Time: {processing_time:.2f}s")
    print(f"Characters extracted: {len(result.text)}")
    
    save_result(result, "01_basic_processing", output_dir)
    return result

def example_2_receipt_processing(output_dir):
    """Example 2: Structured document processing (receipts, invoices)"""
    print("\n2. RECEIPT/STRUCTURED DOCUMENT PROCESSING")
    print("-" * 40)
    
    from advanced_ocr import OCRLibrary, ProcessingOptions
    
    ocr = OCRLibrary()
    
    # Configure for structured documents
    options = ProcessingOptions(
        engines=['paddleocr'],              # Best for structured text
        preserve_formatting=True,           # Keep layout structure
        include_regions=True,              # Get text regions
        detect_orientation=True,           # Handle rotated images
        correct_rotation=True,
        enhance_image=True
    )
    
    image_path = SAMPLE_IMAGES["receipt"]
    print(f"Processing receipt: {image_path}")
    
    result = ocr.process_image(image_path, options)
    
    print(f"Regions detected: {len(result.regions)}")
    print(f"Confidence: {result.confidence:.3f}")
    
    # Analyze structure
    lines = result.text.split('\n')
    non_empty_lines = [line for line in lines if line.strip()]
    
    print(f"Total lines: {len(lines)}")
    print(f"Non-empty lines: {len(non_empty_lines)}")
    
    # Show first few lines
    print("\nFirst 5 lines:")
    for i, line in enumerate(non_empty_lines[:5]):
        print(f"  {i+1}: {line}")
    
    extra_info = {
        "Total Lines": len(lines),
        "Non-empty Lines": len(non_empty_lines),
        "Regions Detected": len(result.regions)
    }
    
    save_result(result, "02_receipt_processing", output_dir, extra_info)
    return result

def example_3_engine_comparison(output_dir):
    """Example 3: Compare different OCR engines"""
    print("\n3. ENGINE COMPARISON")
    print("-" * 40)
    
    from advanced_ocr import OCRLibrary, ProcessingOptions
    
    ocr = OCRLibrary()
    available_engines = ocr.get_available_engines()
    
    print(f"Available engines: {available_engines}")
    
    image_path = SAMPLE_IMAGES["document"]
    results = {}
    
    # Test each engine
    for engine in available_engines:
        print(f"\nTesting {engine}...")
        
        options = ProcessingOptions(
            engines=[engine],
            min_confidence=0.3  # Lower threshold for comparison
        )
        
        try:
            start_time = time.time()
            result = ocr.process_image(image_path, options)
            processing_time = time.time() - start_time
            
            results[engine] = {
                'result': result,
                'success': True,
                'error': None
            }
            
            print(f"  Confidence: {result.confidence:.3f}")
            print(f"  Time: {processing_time:.2f}s")
            print(f"  Characters: {len(result.text)}")
            
        except Exception as e:
            results[engine] = {
                'result': None,
                'success': False,
                'error': str(e)
            }
            print(f"  FAILED: {e}")
    
    # Save comparison results
    comparison_file = output_dir / "03_engine_comparison.txt"
    with open(comparison_file, 'w', encoding='utf-8') as f:
        f.write("ENGINE COMPARISON RESULTS\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Test Image: {image_path}\n\n")
        
        for engine, data in results.items():
            f.write(f"{engine.upper()}:\n")
            f.write("-" * len(engine) + "\n")
            
            if data['success']:
                result = data['result']
                f.write(f"Confidence: {result.confidence:.3f}\n")
                f.write(f"Processing Time: {result.processing_time:.2f}s\n")
                f.write(f"Text Length: {len(result.text)}\n")
                f.write(f"Preview: {result.text[:100]}...\n\n")
            else:
                f.write(f"FAILED: {data['error']}\n\n")
    
    print(f"\nComparison saved to: {comparison_file}")
    return results

def example_4_quality_analysis(output_dir):
    """Example 4: Image quality analysis and adaptive processing"""
    print("\n4. QUALITY ANALYSIS & ADAPTIVE PROCESSING")
    print("-" * 40)
    
    from advanced_ocr import OCRLibrary, ProcessingOptions
    
    ocr = OCRLibrary()
    
    # Test with different quality images
    test_images = {
        "good_quality": SAMPLE_IMAGES["document"],
        "poor_quality": SAMPLE_IMAGES["poor_quality"]
    }
    
    for img_type, image_path in test_images.items():
        print(f"\nAnalyzing {img_type}: {image_path}")
        
        # Let system auto-determine strategy
        options = ProcessingOptions(
            strategy=None,                      # Auto-determine
            enhance_image=True,                 # Allow enhancement
            early_termination=True,             # Stop early if confident
            early_termination_threshold=0.95
        )
        
        result = ocr.process_image(image_path, options)
        
        print(f"  Strategy used: {result.strategy_used.value}")
        print(f"  Quality score: {result.quality_metrics.overall_score:.3f}")
        print(f"  Sharpness: {result.quality_metrics.sharpness_score:.3f}")
        print(f"  Contrast: {result.quality_metrics.contrast_score:.3f}")
        print(f"  Noise level: {result.quality_metrics.noise_level:.3f}")
        print(f"  Enhancement needed: {result.quality_metrics.needs_enhancement}")
        print(f"  Final confidence: {result.confidence:.3f}")
        
        extra_info = {
            "Image Type": img_type,
            "Strategy": result.strategy_used.value,
            "Quality Score": f"{result.quality_metrics.overall_score:.3f}",
            "Sharpness": f"{result.quality_metrics.sharpness_score:.3f}",
            "Contrast": f"{result.quality_metrics.contrast_score:.3f}",
            "Noise Level": f"{result.quality_metrics.noise_level:.3f}",
            "Enhancement Needed": result.quality_metrics.needs_enhancement
        }
        
        save_result(result, f"04_quality_{img_type}", output_dir, extra_info)

def example_5_high_accuracy_processing(output_dir):
    """Example 5: Maximum accuracy processing for critical documents"""
    print("\n5. HIGH-ACCURACY PROCESSING")
    print("-" * 40)
    
    from advanced_ocr import OCRLibrary, ProcessingOptions, ProcessingStrategy
    
    ocr = OCRLibrary()
    
    # Maximum accuracy settings
    options = ProcessingOptions(
        engines=['paddleocr', 'easyocr'],       # Multiple engines
        strategy=ProcessingStrategy.ENHANCED,   # Best quality
        enhance_image=True,                     # Apply enhancements
        detect_orientation=True,                # Fix rotation
        correct_rotation=True,
        min_confidence=0.8,                     # High threshold
        use_parallel_processing=True            # Use parallel processing
    )
    
    image_path = SAMPLE_IMAGES["document"]
    print(f"High-accuracy processing: {image_path}")
    
    start_time = time.time()
    result = ocr.process_image(image_path, options)
    total_time = time.time() - start_time
    
    print(f"Final confidence: {result.confidence:.3f}")
    print(f"Strategy used: {result.strategy_used.value}")
    print(f"Total processing time: {total_time:.2f}s")
    print(f"Characters extracted: {len(result.text)}")
    
    extra_info = {
        "Processing Mode": "High Accuracy",
        "Multiple Engines Used": "Yes",
        "Total Processing Time": f"{total_time:.2f}s"
    }
    
    save_result(result, "05_high_accuracy", output_dir, extra_info)
    return result

def example_6_batch_processing(output_dir):
    """Example 6: Batch processing multiple images"""
    print("\n6. BATCH PROCESSING")
    print("-" * 40)
    
    from advanced_ocr import OCRLibrary, ProcessingOptions
    
    ocr = OCRLibrary()
    
    # Collect available images
    image_paths = []
    for img_type, path in SAMPLE_IMAGES.items():
        if Path(path).exists():
            image_paths.append(path)
    
    if len(image_paths) < 2:
        print("Not enough images for batch processing demo")
        return
    
    print(f"Batch processing {len(image_paths)} images...")
    
    # Configure for batch processing
    options = ProcessingOptions(
        engines=['easyocr'],            # Fast engine for batch
        min_confidence=0.5,             # Lower threshold
        use_parallel_processing=True    # Speed up processing
    )
    
    start_time = time.time()
    batch_result = ocr.process_batch(image_paths, options)
    total_time = time.time() - start_time
    
    print(f"Batch completed in {total_time:.2f}s")
    print(f"Success rate: {batch_result.success_rate:.1f}%")
    print(f"Average confidence: {batch_result.average_confidence:.3f}")
    
    # Save batch results
    batch_file = output_dir / "06_batch_results.txt"
    with open(batch_file, 'w', encoding='utf-8') as f:
        f.write("BATCH PROCESSING RESULTS\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Total images: {len(image_paths)}\n")
        f.write(f"Successful: {batch_result.successful_count}\n")
        f.write(f"Failed: {batch_result.failed_count}\n")
        f.write(f"Success rate: {batch_result.success_rate:.1f}%\n")
        f.write(f"Average confidence: {batch_result.average_confidence:.3f}\n")
        f.write(f"Total time: {total_time:.2f}s\n\n")
        
        f.write("INDIVIDUAL RESULTS:\n")
        f.write("-" * 20 + "\n")
        
        for i, result in enumerate(batch_result.results):
            f.write(f"\nImage {i+1}: {image_paths[i]}\n")
            if result.success:
                f.write(f"  Confidence: {result.confidence:.3f}\n")
                f.write(f"  Time: {result.processing_time:.2f}s\n")
                f.write(f"  Characters: {len(result.text)}\n")
                f.write(f"  Preview: {result.text[:50]}...\n")
            else:
                f.write(f"  FAILED: {result.metadata.get('error', 'Unknown error')}\n")
    
    print(f"Batch results saved to: {batch_file}")
    return batch_result

def example_7_custom_workflows(output_dir):
    """Example 7: Custom workflows for specific use cases"""
    print("\n7. CUSTOM WORKFLOWS")
    print("-" * 40)
    
    from advanced_ocr import OCRLibrary, ProcessingOptions, ProcessingStrategy
    
    ocr = OCRLibrary()
    
    # Workflow 1: Fast processing for digital screenshots
    print("\nWorkflow 1: Fast Digital Screenshot Processing")
    fast_options = ProcessingOptions(
        engines=['easyocr'],                    # Fast engine
        strategy=ProcessingStrategy.MINIMAL,    # Minimal processing
        enhance_image=False,                    # Skip enhancement
        detect_orientation=False,               # Skip rotation detection
        min_confidence=0.7
    )
    
    result1 = ocr.process_image(SAMPLE_IMAGES["document"], fast_options)
    print(f"  Time: {result1.processing_time:.2f}s, Confidence: {result1.confidence:.3f}")
    
    # Workflow 2: Thorough processing for damaged documents
    print("\nWorkflow 2: Damaged Document Recovery")
    thorough_options = ProcessingOptions(
        engines=['paddleocr', 'easyocr', 'tesseract'],  # Multiple engines
        strategy=ProcessingStrategy.ENHANCED,            # Maximum enhancement
        enhance_image=True,
        detect_orientation=True,
        correct_rotation=True,
        min_confidence=0.4                              # Lower threshold
    )
    
    result2 = ocr.process_image(SAMPLE_IMAGES["poor_quality"], thorough_options)
    print(f"  Time: {result2.processing_time:.2f}s, Confidence: {result2.confidence:.3f}")
    
    # Save workflow results
    for i, (result, workflow) in enumerate([(result1, "fast"), (result2, "thorough")], 1):
        extra_info = {"Workflow Type": workflow.title()}
        save_result(result, f"07_workflow_{workflow}", output_dir, extra_info)

def example_8_error_handling(output_dir):
    """Example 8: Error handling and edge cases"""
    print("\n8. ERROR HANDLING & EDGE CASES")
    print("-" * 40)
    
    from advanced_ocr import OCRLibrary, ProcessingOptions
    from advanced_ocr.exceptions import OCRLibraryError, EngineNotAvailableError
    
    ocr = OCRLibrary()
    
    # Test cases
    test_cases = [
        ("valid_image", SAMPLE_IMAGES["document"]),
        ("nonexistent_image", "nonexistent_file.jpg"),
        ("invalid_engine", SAMPLE_IMAGES["document"])  # Will use invalid engine
    ]
    
    results = []
    
    for test_name, image_path in test_cases:
        print(f"\nTesting: {test_name}")
        
        try:
            if test_name == "invalid_engine":
                # Try with non-existent engine
                options = ProcessingOptions(engines=['nonexistent_engine'])
                result = ocr.process_image(image_path, options)
            else:
                result = ocr.process_image(image_path)
            
            print(f"  SUCCESS: {result.confidence:.3f} confidence")
            results.append((test_name, "SUCCESS", result.confidence, None))
            
        except EngineNotAvailableError as e:
            print(f"  ENGINE ERROR: {e}")
            results.append((test_name, "ENGINE_ERROR", 0.0, str(e)))
            
        except OCRLibraryError as e:
            print(f"  OCR ERROR: {e}")
            results.append((test_name, "OCR_ERROR", 0.0, str(e)))
            
        except Exception as e:
            print(f"  UNEXPECTED ERROR: {e}")
            results.append((test_name, "UNEXPECTED_ERROR", 0.0, str(e)))
    
    # Save error handling results
    error_file = output_dir / "08_error_handling.txt"
    with open(error_file, 'w', encoding='utf-8') as f:
        f.write("ERROR HANDLING TEST RESULTS\n")
        f.write("=" * 40 + "\n\n")
        
        for test_name, status, confidence, error in results:
            f.write(f"Test: {test_name}\n")
            f.write(f"Status: {status}\n")
            f.write(f"Confidence: {confidence:.3f}\n")
            if error:
                f.write(f"Error: {error}\n")
            f.write("\n")
    
    print(f"Error handling results saved to: {error_file}")

def main():
    """Run all advanced examples"""
    print("ADVANCED OCR USAGE EXAMPLES")
    print("=" * 50)
    
    # Setup
    output_dir = setup_output()
    print(f"Output directory: {output_dir}")
    
    # Check if main image exists
    main_image = SAMPLE_IMAGES["document"]
    if not Path(main_image).exists():
        print(f"ERROR: Main test image not found: {main_image}")
        print("Please update the SAMPLE_IMAGES paths in the configuration section")
        return
    
    try:
        start_time = time.time()
        
        # Run examples
        example_1_basic_processing(output_dir)
        example_2_receipt_processing(output_dir)
        example_3_engine_comparison(output_dir)
        example_4_quality_analysis(output_dir)
        example_5_high_accuracy_processing(output_dir)
        example_6_batch_processing(output_dir)
        example_7_custom_workflows(output_dir)
        example_8_error_handling(output_dir)
        
        total_time = time.time() - start_time
        
        print(f"\n{'='*50}")
        print(f"ALL ADVANCED EXAMPLES COMPLETED!")
        print(f"Total execution time: {total_time:.2f}s")
        print(f"Results saved to: {output_dir}/")
        print(f"{'='*50}")
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure your OCR library is properly installed and accessible")
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()