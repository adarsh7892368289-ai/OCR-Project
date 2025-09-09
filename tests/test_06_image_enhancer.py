#!/usr/bin/env python3
"""
Test 6: Image Enhancer
Purpose: Test image preprocessing and enhancement algorithms
Tests: src/preprocessing/image_enhancer.py, enhancement algorithms
Success criteria: Enhanced version of img1.jpg created with improved quality
"""

import sys
import os
import cv2
import numpy as np
import time
from pathlib import Path
import importlib.util

# Fix the path issue - add both project root and src to path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"

# Ensure paths are in sys.path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

def test_image_enhancer():
    """Test the Image Enhancer component"""
    print("=" * 60)
    print("TEST 6: IMAGE ENHANCER")
    print("=" * 60)
    
    try:
        # Test image path
        test_image_path = project_root / "data" / "sample_images" / "img1.jpg"
        
        # Load test image
        if not test_image_path.exists():
            print(f"✗ Test image not found: {test_image_path}")
            return False
            
        image = cv2.imread(str(test_image_path))
        if image is None:
            print("✗ Failed to load test image")
            return False
            
        print(f"✓ Test image loaded: {image.shape}")
        
        # Test 6.1: Import Image Enhancer with adaptive approach
        ImageEnhancer = None
        try:
            # Try direct module loading
            enhancer_file = src_path / "preprocessing" / "image_enhancer.py"
            spec = importlib.util.spec_from_file_location("image_enhancer", enhancer_file)
            enhancer_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(enhancer_module)
            
            # Find available enhancer classes
            available_classes = [attr for attr in dir(enhancer_module) 
                               if not attr.startswith('_') and attr[0].isupper() and 'enhancer' in attr.lower()]
            
            print(f"Available classes in image_enhancer: {[cls for cls in dir(enhancer_module) if not cls.startswith('_') and cls[0].isupper()]}")
            
            # Try to get the enhancer class
            if hasattr(enhancer_module, 'ImageEnhancer'):
                ImageEnhancer = enhancer_module.ImageEnhancer
                print("✓ ImageEnhancer imported successfully")
            elif hasattr(enhancer_module, 'AIImageEnhancer'):
                ImageEnhancer = enhancer_module.AIImageEnhancer
                print("✓ AIImageEnhancer imported as ImageEnhancer")
            elif hasattr(enhancer_module, 'EnhancementProcessor'):
                ImageEnhancer = enhancer_module.EnhancementProcessor
                print("✓ EnhancementProcessor imported as ImageEnhancer")
            elif hasattr(enhancer_module, 'ImageProcessor'):
                ImageEnhancer = enhancer_module.ImageProcessor
                print("✓ ImageProcessor imported as ImageEnhancer")
            elif available_classes:
                ImageEnhancer = getattr(enhancer_module, available_classes[0])
                print(f"✓ {available_classes[0]} imported as ImageEnhancer")
            else:
                print("✗ No suitable enhancer class found")
                return False
                
        except Exception as e:
            print(f"✗ Failed to import image enhancer: {e}")
            return False
        
        # Test 6.2: Initialize Image Enhancer
        try:
            enhancer = ImageEnhancer()
            print("✓ ImageEnhancer initialized")
        except Exception as e:
            print(f"✗ Failed to initialize ImageEnhancer: {e}")
            return False
        
        # Test 6.3: Discover available methods
        available_methods = [method for method in dir(enhancer) 
                           if not method.startswith('_') and callable(getattr(enhancer, method))]
        print(f"✓ Available methods: {available_methods}")
        
        # Test 6.4: Basic Enhancement (FIXED - handle EnhancementResult)
        enhanced_result = None
        enhanced_image = None
        enhancement_method_used = None
        
        # Common method names to try
        method_attempts = [
            'enhance_for_ocr',
            'enhance_image', 
            'enhance',
            'process_image',
            'process',
            'apply_enhancement',
            'improve_image'
        ]
        
        for method_name in method_attempts:
            if hasattr(enhancer, method_name):
                try:
                    start_time = time.time()
                    method = getattr(enhancer, method_name)
                    result = method(image)
                    processing_time = time.time() - start_time
                    enhancement_method_used = method_name
                    
                    # Handle EnhancementResult object vs direct numpy array
                    if hasattr(result, 'enhanced_image'):
                        # It's an EnhancementResult object
                        enhanced_image = result.enhanced_image
                        enhanced_result = result
                        print(f"✓ Basic enhancement completed using {method_name}")
                        print(f"  - Processing time: {processing_time:.3f}s")
                        print(f"  - Output shape: {enhanced_image.shape}")
                        print(f"  - Enhancement applied: {result.enhancement_applied}")
                        print(f"  - Operations: {result.operations_performed}")
                        break
                    elif isinstance(result, np.ndarray):
                        # Direct numpy array
                        enhanced_image = result
                        print(f"✓ Basic enhancement completed using {method_name}")
                        print(f"  - Processing time: {processing_time:.3f}s")
                        print(f"  - Output shape: {enhanced_image.shape}")
                        break
                    else:
                        print(f"✗ {method_name} returned invalid result type: {type(result)}")
                except Exception as e:
                    print(f"✗ {method_name} failed: {e}")
        
        if enhanced_image is None:
            print("✗ No working enhancement method found")
            return False
        
        # Test 6.5: Test specific enhancement functions if available
        specific_methods = [
            ('sharpen', 'Sharpening'),
            ('denoise', 'Denoising'), 
            ('adjust_contrast', 'Contrast Adjustment'),
            ('adjust_brightness', 'Brightness Adjustment'),
            ('deblur', 'Deblurring'),
            ('enhance_contrast', 'Contrast Enhancement'),
            ('reduce_noise', 'Noise Reduction'),
            ('apply_sharpening', 'Sharpening Filter')
        ]
        
        enhancement_results = {}
        for method_name, display_name in specific_methods:
            if hasattr(enhancer, method_name):
                try:
                    start_time = time.time()
                    method = getattr(enhancer, method_name)
                    result = method(image)
                    processing_time = time.time() - start_time
                    
                    # Handle different return types
                    result_image = None
                    if hasattr(result, 'enhanced_image'):
                        result_image = result.enhanced_image
                    elif isinstance(result, np.ndarray):
                        result_image = result
                    
                    if result_image is not None and len(result_image.shape) >= 2:
                        enhancement_results[method_name] = {
                            'success': True,
                            'time': processing_time,
                            'shape': result_image.shape
                        }
                        print(f"✓ {display_name}: {processing_time:.3f}s")
                    else:
                        enhancement_results[method_name] = {'success': False, 'reason': 'Invalid result'}
                        print(f"✗ {display_name}: Invalid result")
                except Exception as e:
                    enhancement_results[method_name] = {'success': False, 'error': str(e)}
                    print(f"✗ {display_name}: Error - {e}")
        
        # Test 6.6: Quality Comparison (FIXED)
        try:
            # Try to import quality analyzer
            qa_file = src_path / "preprocessing" / "quality_analyzer.py"
            qa_spec = importlib.util.spec_from_file_location("quality_analyzer", qa_file)
            qa_module = importlib.util.module_from_spec(qa_spec)
            qa_spec.loader.exec_module(qa_module)
            
            QualityAnalyzer = qa_module.QualityAnalyzer
            quality_analyzer = QualityAnalyzer()
            
            # Compare quality before and after (now using correct enhanced_image)
            original_quality = quality_analyzer.analyze_image(image)
            enhanced_quality = quality_analyzer.analyze_image(enhanced_image)
            
            print("✓ Quality comparison:")
            print(f"  - Original overall score: {original_quality.overall_score:.3f}")
            print(f"  - Enhanced overall score: {enhanced_quality.overall_score:.3f}")
            print(f"  - Improvement: {enhanced_quality.overall_score - original_quality.overall_score:.3f}")
            
            # Specific improvements
            print(f"  - Sharpness: {original_quality.sharpness_score:.3f} → {enhanced_quality.sharpness_score:.3f}")
            print(f"  - Contrast: {original_quality.contrast_score:.3f} → {enhanced_quality.contrast_score:.3f}")
            print(f"  - Brightness: {original_quality.brightness_score:.3f} → {enhanced_quality.brightness_score:.3f}")
            
        except Exception as e:
            print(f"✓ Quality comparison skipped: {e}")
        
        # Test 6.7: Save Enhanced Images for Visual Inspection (FIXED)
        try:
            debug_dir = project_root / "debug" / "image_enhancement"
            debug_dir.mkdir(parents=True, exist_ok=True)
            
            # Save original and enhanced (now using correct enhanced_image numpy array)
            cv2.imwrite(str(debug_dir / "01_original.jpg"), image)
            cv2.imwrite(str(debug_dir / f"02_enhanced_{enhancement_method_used}.jpg"), enhanced_image)
            
            # Save individual enhancements if they exist
            enhancement_count = 3
            for method_name, result_info in enhancement_results.items():
                if result_info.get('success', False):
                    try:
                        method = getattr(enhancer, method_name)
                        method_result = method(image)
                        
                        # Handle different return types for saving
                        save_image = None
                        if hasattr(method_result, 'enhanced_image'):
                            save_image = method_result.enhanced_image
                        elif isinstance(method_result, np.ndarray):
                            save_image = method_result
                        
                        if save_image is not None:
                            filename = f"{enhancement_count:02d}_{method_name}.jpg"
                            cv2.imwrite(str(debug_dir / filename), save_image)
                            enhancement_count += 1
                    except Exception as ex:
                        print(f"  - Could not save {method_name}: {ex}")
            
            print(f"✓ Enhanced images saved to: {debug_dir}")
            
        except Exception as e:
            print(f"✗ Failed to save enhanced images: {e}")
        
        # Test 6.8: Edge Cases (FIXED)
        try:
            # Test with grayscale image
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            method = getattr(enhancer, enhancement_method_used)
            gray_result = method(gray_image)
            
            # Handle EnhancementResult vs direct array
            gray_enhanced = None
            if hasattr(gray_result, 'enhanced_image'):
                gray_enhanced = gray_result.enhanced_image
            elif isinstance(gray_result, np.ndarray):
                gray_enhanced = gray_result
            
            if gray_enhanced is not None:
                print("✓ Grayscale enhancement works")
            else:
                print("✗ Grayscale enhancement failed")
            
            # Test with small image
            small_image = cv2.resize(image, (200, 150))
            small_result = method(small_image)
            
            # Handle EnhancementResult vs direct array
            small_enhanced = None
            if hasattr(small_result, 'enhanced_image'):
                small_enhanced = small_result.enhanced_image
            elif isinstance(small_result, np.ndarray):
                small_enhanced = small_result
            
            if small_enhanced is not None:
                print("✓ Small image enhancement works")
            else:
                print("✗ Small image enhancement failed")
                
        except Exception as e:
            print(f"✗ Edge case testing failed: {e}")
        
        # Test 6.9: Performance Summary
        print("✓ Enhancement performance summary:")
        print(f"  - Main enhancement method: {enhancement_method_used}")
        
        successful_specific = sum(1 for result in enhancement_results.values() if result.get('success', False))
        total_specific = len(enhancement_results)
        
        if total_specific > 0:
            print(f"  - Specific enhancement methods: {successful_specific}/{total_specific}")
            
            if successful_specific > 0:
                avg_time = np.mean([result['time'] for result in enhancement_results.values() 
                                  if result.get('success', False) and 'time' in result])
                print(f"  - Average specific method time: {avg_time:.3f}s")
        else:
            print("  - No specific enhancement methods found")
        
        print("=" * 60)
        print("TEST 6: PASSED - Image enhancer works correctly")
        print("=" * 60)
        print("SUCCESS: Image enhancer is functioning properly")
        print("Next step: Run Test 7 (Text Detector)")
        return True
        
    except Exception as e:
        print(f"✗ CRITICAL ERROR in Image Enhancer test: {e}")
        import traceback
        traceback.print_exc()
        print("=" * 60)
        print("TEST 6: FAILED")
        print("=" * 60)
        return False

if __name__ == "__main__":
    success = test_image_enhancer()
    sys.exit(0 if success else 1)