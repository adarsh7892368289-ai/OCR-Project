#!/usr/bin/env python3
"""
Batch Processing Example - Process multiple images at once
Configure your folder paths below and run this script
"""

import sys
from pathlib import Path

# === CONFIGURATION - EDIT THESE PATHS ===
INPUT_FOLDER = "tests/images"              # Folder containing images to process
OUTPUT_FOLDER = "batch_results"            # Folder to save results
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']  # File types to process

# Processing options
USE_FAST_ENGINE = True                     # True = fast (easyocr), False = accurate (paddleocr)
MIN_CONFIDENCE = 0.6                       # Minimum confidence threshold
MAX_IMAGES = 10                           # Maximum number of images to process (None for all)

# Add OCR library to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def collect_images(folder_path):
    """Find all image files in the specified folder"""
    folder = Path(folder_path)
    if not folder.exists():
        print(f"Error: Folder '{folder_path}' does not exist")
        return []
    
    image_files = []
    for ext in IMAGE_EXTENSIONS:
        image_files.extend(folder.glob(f'*{ext.lower()}'))
        image_files.extend(folder.glob(f'*{ext.upper()}'))
    
    # Limit number of images if specified
    if MAX_IMAGES:
        image_files = image_files[:MAX_IMAGES]
    
    return sorted(image_files)

def setup_output_folder():
    """Create output folder if it doesn't exist"""
    output_path = Path(OUTPUT_FOLDER)
    output_path.mkdir(exist_ok=True)
    return output_path

def process_single_image(ocr, image_path, options):
    """Process a single image and return result"""
    try:
        result = ocr.process_image(str(image_path), options)
        return {
            'success': True,
            'text': result.text,
            'confidence': result.confidence,
            'engine': result.engine_used,
            'time': result.processing_time,
            'error': None
        }
    except Exception as e:
        return {
            'success': False,
            'text': '',
            'confidence': 0.0,
            'engine': 'none',
            'time': 0.0,
            'error': str(e)
        }

def save_individual_results(results, output_folder):
    """Save each image result to a separate text file"""
    for i, (image_file, result) in enumerate(results.items()):
        # Create output filename
        output_name = f"{image_file.stem}_result.txt"
        output_path = output_folder / output_name
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"OCR RESULT FOR: {image_file.name}\n")
            f.write("=" * 50 + "\n\n")
            
            if result['success']:
                f.write(f"Success: Yes\n")
                f.write(f"Confidence: {result['confidence']:.1%}\n")
                f.write(f"Engine: {result['engine']}\n")
                f.write(f"Processing time: {result['time']:.2f}s\n")
                f.write(f"Text length: {len(result['text'])} characters\n\n")
                f.write("EXTRACTED TEXT:\n")
                f.write("-" * 20 + "\n")
                f.write(result['text'])
            else:
                f.write(f"Success: No\n")
                f.write(f"Error: {result['error']}\n")

def save_summary_report(results, output_folder, total_time):
    """Save a summary report of all processed images"""
    summary_path = output_folder / "batch_summary.txt"
    
    successful = [r for r in results.values() if r['success']]
    failed = [r for r in results.values() if not r['success']]
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("BATCH PROCESSING SUMMARY\n")
        f.write("=" * 40 + "\n\n")
        
        f.write(f"Total images: {len(results)}\n")
        f.write(f"Successful: {len(successful)}\n")
        f.write(f"Failed: {len(failed)}\n")
        f.write(f"Success rate: {len(successful)/len(results)*100:.1f}%\n")
        f.write(f"Total processing time: {total_time:.2f}s\n\n")
        
        if successful:
            avg_confidence = sum(r['confidence'] for r in successful) / len(successful)
            avg_time = sum(r['time'] for r in successful) / len(successful)
            total_chars = sum(len(r['text']) for r in successful)
            
            f.write("SUCCESSFUL PROCESSING STATS:\n")
            f.write("-" * 30 + "\n")
            f.write(f"Average confidence: {avg_confidence:.1%}\n")
            f.write(f"Average processing time: {avg_time:.2f}s per image\n")
            f.write(f"Total characters extracted: {total_chars:,}\n\n")
        
        f.write("DETAILED RESULTS:\n")
        f.write("-" * 20 + "\n")
        
        for image_file, result in results.items():
            f.write(f"{image_file.name}:\n")
            if result['success']:
                f.write(f"  Confidence: {result['confidence']:.1%}\n")
                f.write(f"  Time: {result['time']:.2f}s\n")
                f.write(f"  Characters: {len(result['text'])}\n")
                f.write(f"  Preview: {result['text'][:50]}...\n")
            else:
                f.write(f"  FAILED: {result['error']}\n")
            f.write("\n")
    
    return summary_path

def main():
    """Main batch processing function"""
    print("BATCH IMAGE PROCESSING")
    print("=" * 30)
    
    # Setup
    image_files = collect_images(INPUT_FOLDER)
    if not image_files:
        print(f"No image files found in '{INPUT_FOLDER}'")
        print(f"Looking for extensions: {IMAGE_EXTENSIONS}")
        return
    
    output_folder = setup_output_folder()
    
    print(f"Found {len(image_files)} images to process")
    print(f"Output folder: {output_folder}")
    print(f"Processing settings:")
    print(f"  Engine: {'EasyOCR (fast)' if USE_FAST_ENGINE else 'PaddleOCR (accurate)'}")
    print(f"  Min confidence: {MIN_CONFIDENCE:.1%}")
    
    # Initialize OCR
    try:
        from advanced_ocr import OCRLibrary, ProcessingOptions
        
        ocr = OCRLibrary()
        
        # Configure processing options
        options = ProcessingOptions(
            engines=['easyocr'] if USE_FAST_ENGINE else ['paddleocr'],
            min_confidence=MIN_CONFIDENCE,
            enhance_image=True,
            detect_orientation=True
        )
        
    except Exception as e:
        print(f"Failed to initialize OCR library: {e}")
        return
    
    # Process images
    results = {}
    start_time = time.time()
    
    print(f"\nProcessing images:")
    print("-" * 20)
    
    for i, image_file in enumerate(image_files, 1):
        print(f"{i:2d}/{len(image_files)}: {image_file.name}...", end=' ')
        
        result = process_single_image(ocr, image_file, options)
        results[image_file] = result
        
        if result['success']:
            print(f"OK ({result['confidence']:.1%}, {result['time']:.1f}s)")
        else:
            print(f"FAILED ({result['error']})")
    
    total_time = time.time() - start_time
    
    # Save results
    print(f"\nSaving results...")
    save_individual_results(results, output_folder)
    summary_path = save_summary_report(results, output_folder, total_time)
    
    # Print summary
    successful = sum(1 for r in results.values() if r['success'])
    print(f"\nBATCH PROCESSING COMPLETE")
    print("-" * 30)
    print(f"Total images: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {len(results) - successful}")
    print(f"Success rate: {successful/len(results)*100:.1f}%")
    print(f"Total time: {total_time:.2f}s")
    print(f"Average time: {total_time/len(results):.2f}s per image")
    print(f"\nResults saved to: {output_folder}")
    print(f"Summary report: {summary_path}")

if __name__ == "__main__":
    import time
    main()