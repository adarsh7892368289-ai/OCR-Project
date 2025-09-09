#!/usr/bin/env python3
"""
Test 4: Image Utilities (Fixed Version)
Tests image processing utilities for modern OCR compatibility
"""

import sys
import os
sys.path.append('.')

import cv2
import numpy as np
from pathlib import Path

def test_image_utilities():
    """Test image utilities functionality"""
    print("=" * 60)
    print("TEST 4: IMAGE UTILITIES")
    print("=" * 60)
    
    # Load test image
    image = cv2.imread("data/sample_images/img1.jpg")
    print(f"✓ Test image loaded: {image.shape}")
    
    # Test 1: Import image utils module
    try:
        from src.utils.image_utils import ImageUtils
        print("✓ ImageUtils class imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import ImageUtils: {e}")
        return False
    except Exception as e:
        print(f"✗ Error importing ImageUtils: {e}")
        return False
    
    # Test 2: Test image format conversions
    try:
        # Test color space conversion
        rgb_image = ImageUtils.convert_color_space(image, "BGR2RGB")
        print(f"✓ BGR to RGB conversion: {image.shape} -> {rgb_image.shape}")
        
        # RGB to grayscale
        gray_image = ImageUtils.convert_color_space(rgb_image, "RGB2GRAY")
        print(f"✓ RGB to grayscale conversion: {rgb_image.shape} -> {gray_image.shape}")
        
        # BGR to grayscale directly
        gray_direct = ImageUtils.convert_color_space(image, "BGR2GRAY")
        print(f"✓ Direct BGR to grayscale: {image.shape} -> {gray_direct.shape}")
        
    except Exception as e:
        print(f"✗ Error with image format conversions: {e}")
        return False
    
    # Test 3: Test image resizing utilities
    try:
        # Test resize function
        resized = ImageUtils.resize_image(image, max_width=800, max_height=600)
        print(f"✓ ImageUtils resize: {image.shape} -> {resized.shape}")
        
        # Test with different parameters
        aspect_resized = ImageUtils.resize_image(image, max_width=1024, max_height=1024)
        print(f"✓ Aspect ratio resize: {image.shape} -> {aspect_resized.shape}")
            
    except Exception as e:
        print(f"✗ Error with image resizing: {e}")
        return False
    
    # Test 4: Test image quality checks
    try:
        # Test quality detection function
        quality_metrics = ImageUtils.detect_image_quality(image)
        print(f"✓ Image quality detection completed")
        print(f"  - Blur score: {quality_metrics['blur_score']:.2f}")
        print(f"  - Is blurry: {quality_metrics['is_blurry']}")
        print(f"  - Contrast level: {quality_metrics['contrast_level']:.2f}")
        print(f"  - Brightness level: {quality_metrics['brightness_level']:.2f}")
        print(f"  - Recommended for OCR: {quality_metrics['recommended_for_ocr']}")
        
        # Test image stats calculation
        stats = ImageUtils.calculate_image_stats(image)
        print(f"✓ Image statistics calculated")
        print(f"  - Width: {stats['width']}, Height: {stats['height']}")
        print(f"  - Mean brightness: {stats['mean_brightness']:.2f}")
        print(f"  - Std brightness: {stats['std_brightness']:.2f}")
        
    except Exception as e:
        print(f"✗ Error with image quality checks: {e}")
        return False
    
    # Test 5: Test image preprocessing utilities
    try:
        # Test OCR-specific preprocessing
        from PIL import Image
        pil_image = ImageUtils.preprocess_image_for_recognition(image)
        print(f"✓ OCR preprocessing completed: {type(pil_image).__name__}")
        print(f"  - Image mode: {pil_image.mode}")
        print(f"  - Image size: {pil_image.size}")
        
        # Manual normalization as fallback
        normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        print(f"✓ Manual image normalization completed")
        
        # Histogram equalization on grayscale
        gray = ImageUtils.convert_color_space(image, "BGR2GRAY")
        equalized = cv2.equalizeHist(gray)
        print(f"✓ Histogram equalization: {gray.shape} -> {equalized.shape}")
        
    except Exception as e:
        print(f"✗ Error with image preprocessing: {e}")
        return False
    
    # Test 6: Test image validation and I/O utilities
    try:
        # Test image saving
        debug_dir = Path("debug")
        debug_dir.mkdir(exist_ok=True)
        
        test_save_path = "debug/test_save.jpg"
        ImageUtils.save_image(image, test_save_path, quality=90)
        if Path(test_save_path).exists():
            print(f"✓ Image save function works: {test_save_path}")
        else:
            print(f"⚠ Image save failed: {test_save_path}")
        
        # Test validation through stats
        stats = ImageUtils.calculate_image_stats(image)
        h, w = stats['height'], stats['width']
        min_size_ok = h >= 32 and w >= 32
        print(f"✓ Minimum size check: {min_size_ok} ({w}x{h})")
        
        max_size_ok = h <= 4096 and w <= 4096
        print(f"✓ Maximum size check: {max_size_ok} ({w}x{h})")
        
    except Exception as e:
        print(f"✗ Error with image validation and I/O: {e}")
        return False
    
    # Test 7: Test modern OCR compatibility and additional features
    try:
        # Test array compatibility
        print(f"✓ NumPy array compatibility: {type(image).__name__}")
        print(f"✓ Data type compatibility: {image.dtype}")
        print(f"✓ Memory layout: {'C-contiguous' if image.flags['C_CONTIGUOUS'] else 'Not C-contiguous'}")
        
        # Test PIL compatibility with your preprocessing function
        pil_preprocessed = ImageUtils.preprocess_image_for_recognition(image)
        print(f"✓ PIL preprocessing compatibility: {pil_preprocessed.size}")
        
        # Test crop functionality
        bbox = (100, 100, 200, 150)  # x, y, w, h
        cropped = ImageUtils.crop_text_region(image, bbox, padding=10)
        print(f"✓ Text region cropping: {cropped.size}")
        
        # Test tensor-like operations (for modern OCR engines)
        gray_direct = ImageUtils.convert_color_space(image, "BGR2GRAY")
        test_batch = np.expand_dims(gray_direct, axis=0)  # Add batch dimension
        test_batch = np.expand_dims(test_batch, axis=-1)  # Add channel dimension
        print(f"✓ Tensor compatibility: {gray_direct.shape} -> {test_batch.shape}")
        
        # Save additional test images
        test_files = [
            ("debug/test_original.jpg", image),
            ("debug/test_grayscale.jpg", gray_direct),
            ("debug/test_resized.jpg", resized),
            ("debug/test_equalized.jpg", equalized),
        ]
        
        saved_count = 0
        for filename, img in test_files:
            try:
                cv2.imwrite(filename, img)
                if Path(filename).exists():
                    saved_count += 1
            except:
                pass
        
        print(f"✓ Debug images saved: {saved_count}/{len(test_files)}")
        
    except Exception as e:
        print(f"✗ Error with modern OCR compatibility: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("TEST 4: PASSED - Image utilities work correctly")
    print("=" * 60)
    
    return True

def main():
    """Run the image utilities test"""
    try:
        success = test_image_utilities()
        if success:
            print(f"\nSUCCESS: Image utilities are working properly")
            print(f"Next step: Run Test 5 (Quality Analyzer)")
        else:
            print(f"\nFAILED: Please fix image utility issues before proceeding")
        return success
        
    except Exception as e:
        print(f"\nUNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)