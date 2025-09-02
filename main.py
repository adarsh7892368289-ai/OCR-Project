#!/usr/bin/env python3
"""
Advanced OCR Processing System
Handles both handwritten and printed text with multiple OCR engines
"""

import os
import sys
import argparse
from pathlib import Path

# Add src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from ocr_processor import AdvancedOCRProcessor

def main():
    parser = argparse.ArgumentParser(description='Advanced OCR Processing System')
    parser.add_argument('image_path', help='Path to the image file')
    parser.add_argument('--output', '-o', help='Output file for extracted text')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    # Validate image path
    if not os.path.exists(args.image_path):
        print(f"Error: Image file '{args.image_path}' not found!")
        sys.exit(1)
    
    # Initialize OCR processor
    log_level = logging.DEBUG if args.verbose else logging.INFO
    ocr_processor = AdvancedOCRProcessor(log_level=log_level)
    
    print(f"Processing image: {args.image_path}")
    print("Please wait, this may take a few moments...\n")
    
    try:
        # Process with all engines
        results = ocr_processor.process_image_all_engines(args.image_path)
        
        # Print results to console
        ocr_processor.print_results(results)
        
        # Save to file if specified
        if args.output:
            best_result = ocr_processor.get_best_result(results)
            if best_result:
                with open(args.output, 'w', encoding='utf-8') as f:
                    f.write(f"Best OCR Result ({best_result.engine}):\n")
                    f.write(f"Confidence: {best_result.confidence:.1f}%\n")
                    f.write(f"Processing Time: {best_result.processing_time:.0f}ms\n\n")
                    f.write("Extracted Text:\n")
                    f.write(best_result.text)
                    
                    f.write("\n\n" + "="*50 + "\n")
                    f.write("ALL RESULTS:\n")
                    f.write("="*50 + "\n\n")
                    
                    for result in results:
                        f.write(f"{result.engine}: {result.confidence:.1f}% ")
                        f.write(f"({result.processing_time:.0f}ms)\n")
                        f.write(f"Text: {result.text}\n\n")
                
                print(f"\nResults saved to: {args.output}")
            else:
                print("No successful OCR results to save.")
        
    except Exception as e:
        print(f"Error processing image: {e}")
        sys.exit(1)

if __name__ == "__main__":
    import logging
    main()
