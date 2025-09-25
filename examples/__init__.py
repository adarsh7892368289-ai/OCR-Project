#!/usr/bin/env python3
"""
Simple OCR Library Usage - Basic text extraction from images
Configure your image path below and run this script
"""

import sys
from pathlib import Path

# === CONFIGURATION - EDIT THESE PATHS ===
IMAGE_PATH = "tests/images/img1.jpg"  # Change this to your image path
OUTPUT_FILE = "extracted_text.txt"    # Optional: save result to file (set to None to skip)

# Add OCR library to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def main():
    """Simple OCR text extraction"""
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
        
        # Save to file if specified
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