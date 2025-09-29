#!/usr/bin/env python3
"""
Basic example for OCR text extraction from an image.
Edit IMAGE_PATH to specify the input image.
"""

import sys
from pathlib import Path

# Configuration: set input image and optional output file
IMAGE_PATH = "tests/images/img1.jpg"
OUTPUT_FILE = "extracted_text.txt"

# Add OCR library to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def main():
    """Run OCR on the specified image and print results."""
    try:
        from advanced_ocr import OCRLibrary
        
        print("Initializing OCR Library...")
        ocr = OCRLibrary()
        
        print(f"Processing image: {IMAGE_PATH}")
        result = ocr.process_image(IMAGE_PATH)
        
        print("\n" + "="*50)
        print("EXTRACTED TEXT:")
        print("="*50)
        print(result.text)
        print("="*50)
        
        print(f"\nConfidence: {result.confidence:.1%}")
        print(f"Engine used: {result.engine_used}")
        print(f"Processing time: {result.processing_time:.2f}s")
        print(f"Text length: {len(result.text)} characters")
        
        if OUTPUT_FILE:
            with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                f.write(result.text)
            print(f"\nText saved to: {OUTPUT_FILE}")
        
        return result
        
    except Exception as e:
        print(f"Error: {e}")
        return None

if __name__ == "__main__":
    main()
