#!/usr/bin/env python3
"""
Enhanced OCR System - Main Entry Point
COMPLETELY FIXED VERSION - No duplicates, proper formatting
"""

import sys
import os
import logging
import time
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import the OCR processor
try:
    from src.ocr_processor import OCRProcessor
except ImportError as e:
    print(f"Error: Could not import OCRProcessor: {e}")
    print("Make sure ocr_processor.py is in the same directory as main.py")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)

def format_results(results):
    """Format results for clean display - FIXED VERSION"""
    if not results:
        return "No results to display"
    
    if "error" in results:
        return f"Error processing image: {results['error']}"
    
    output = []
    output.append("="*80)
    output.append("OCR PROCESSING RESULTS")
    output.append("="*80)
    output.append("")
    
    # Display best result
    best_result = results.get('best_result', {})
    if best_result and best_result.get('text'):
        source_name = {
            'tesseract': 'Tesseract OCR',
            'easyocr': 'EasyOCR',
            'paddleocr': 'PaddleOCR', 
            'trocr_handwriting': 'TrOCR Handwriting'
        }.get(best_result.get('source', ''), 'Unknown')
        
        output.append(f"🏆 BEST RESULT ({source_name}):")
        output.append(f"Confidence: {best_result.get('confidence', 0)}%")
        output.append(f"Processing Time: {best_result.get('processing_time', 0)}ms")
        
        # Format text properly with line breaks
        text = best_result.get('text', 'No text extracted')
        output.append(f"Text: {text}")
        output.append("")
        output.append("-" * 80)
        output.append("")
    
    # Display all results - NO DUPLICATES
    all_results = results.get('all_results', {})
    if all_results:
        output.append("ALL OCR ENGINE RESULTS:")
        output.append("-" * 80)
        output.append("")
        
        engine_names = {
            'tesseract': 'Tesseract OCR',
            'easyocr': 'EasyOCR', 
            'paddleocr': 'PaddleOCR',
            'trocr_handwriting': 'TrOCR Handwriting'
        }
        
        # Process each engine EXACTLY ONCE
        engine_count = 1
        for engine_key in ['tesseract', 'easyocr', 'paddleocr', 'trocr_handwriting']:
            if engine_key in all_results:
                result = all_results[engine_key]
                display_name = engine_names[engine_key]
                status_icon = "✅" if result.get('status') == 'SUCCESS' else "❌"
                
                output.append(f"{engine_count}. {status_icon} {display_name}")
                output.append(f"   Status: {result.get('status', 'UNKNOWN')}")
                
                if result.get('status') == 'SUCCESS':
                    output.append(f"   Confidence: {result.get('confidence', 0)}%")
                    output.append(f"   Processing Time: {result.get('processing_time', 0)}ms")
                    output.append(f"   Text Length: {result.get('text_length', 0)} characters")
                    
                    # Show text with proper formatting
                    text = result.get('text', '')
                    if text:
                        # Limit display to first 500 characters for readability
                        if len(text) > 500:
                            display_text = text[:500] + "..."
                        else:
                            display_text = text
                        output.append(f"   Text: {display_text}")
                    else:
                        output.append(f"   Text: No text extracted")
                else:
                    error_msg = result.get('error', 'Unknown error')
                    output.append(f"   Error: {error_msg}")
                
                output.append("")
                engine_count += 1
    
    output.append("="*80)
    return "\n".join(output)

def main():
    """Main function"""
    if len(sys.argv) != 2:
        print("Usage: python main.py <image_path>")
        print("Example: python main.py data/input/sample_images/img3.jpg")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        sys.exit(1)
    
    # Validate file extension
    valid_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp'}
    file_ext = Path(image_path).suffix.lower()
    if file_ext not in valid_extensions:
        print(f"Error: Unsupported file format '{file_ext}'. Supported formats: {', '.join(valid_extensions)}")
        sys.exit(1)
    
    print(f"Processing image: {image_path}")
    print("Please wait, this may take a few moments...")
    print()
    
    try:
        # Initialize OCR processor
        processor = OCRProcessor()
        
        # Process the image
        start_time = time.time()
        results = processor.process_image(image_path)
        total_time = time.time() - start_time
        
        # Display results with proper formatting
        formatted_output = format_results(results)
        print(formatted_output)
        
        # Summary
        print(f"\nTotal processing time: {total_time:.2f} seconds")
        
        # Optionally save results to file
        output_file = Path(image_path).stem + "_ocr_results.txt"
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(formatted_output)
                f.write(f"\n\nTotal processing time: {total_time:.2f} seconds")
            print(f"Results saved to: {output_file}")
        except Exception as e:
            logger.warning(f"Could not save results to file: {e}")
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        print(f"An unexpected error occurred: {e}")
        import traceback
        print("Full error details:")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()