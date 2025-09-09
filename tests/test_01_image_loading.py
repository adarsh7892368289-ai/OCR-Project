#!/usr/bin/env python3
"""
Test 1: Basic Image Loading
Tests if img1.jpg can be loaded and basic properties are correct
"""

import sys
import os
sys.path.append('.')

import cv2
import numpy as np
from pathlib import Path

def test_image_loading():
    """Test basic image loading and validation"""
    print("=" * 60)
    print("TEST 1: BASIC IMAGE LOADING")
    print("=" * 60)
    
    # Check if image file exists
    image_path = "data/sample_images/img1.jpg"
    print(f"Looking for image at: {image_path}")
    
    if not Path(image_path).exists():
        print(f"ERROR: Test image not found at {image_path}")
        print("Please ensure the image exists at this path")
        return False
    
    print(f"✓ Image file found")
    
    # Load the image
    try:
        image = cv2.imread(image_path)
        if image is None:
            print("ERROR: Failed to load image - file might be corrupted")
            return False
        
        print(f"✓ Image loaded successfully")
        
    except Exception as e:
        print(f"ERROR: Exception while loading image: {e}")
        return False
    
    # Check image properties
    height, width = image.shape[:2]
    channels = image.shape[2] if len(image.shape) > 2 else 1
    
    print(f"✓ Image dimensions: {width} x {height}")
    print(f"✓ Image channels: {channels}")
    print(f"✓ Image data type: {image.dtype}")
    
    # Validate dimensions
    if height < 32 or width < 32:
        print(f"WARNING: Image is very small ({width}x{height}) - OCR might not work well")
    elif height > 4096 or width > 4096:
        print(f"WARNING: Image is very large ({width}x{height}) - processing might be slow")
    else:
        print(f"✓ Image size is good for OCR processing")
    
    # Check if image has content (not all black/white)
    mean_intensity = np.mean(image)
    std_intensity = np.std(image)
    
    print(f"✓ Mean pixel intensity: {mean_intensity:.1f}")
    print(f"✓ Pixel intensity std dev: {std_intensity:.1f}")
    
    if std_intensity < 10:
        print("WARNING: Image has very low contrast - might affect OCR quality")
    else:
        print("✓ Image appears to have good contrast")
    
    # Save a copy for debugging
    debug_dir = Path("debug")
    debug_dir.mkdir(exist_ok=True)
    
    debug_path = debug_dir / "loaded_img1.jpg"
    cv2.imwrite(str(debug_path), image)
    print(f"✓ Debug copy saved to: {debug_path}")
    
    print("\n" + "=" * 60)
    print("TEST 1: PASSED - Image loading works correctly")
    print("=" * 60)
    
    return True, image

def main():
    """Run the test"""
    try:
        success, image = test_image_loading()
        if success:
            print(f"\nSUCCESS: Image is ready for OCR processing")
            print(f"Next step: Run Test 2 (Configuration Loading)")
        else:
            print(f"\nFAILED: Please fix image loading issues before proceeding")
        return success
        
    except Exception as e:
        print(f"\nUNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)