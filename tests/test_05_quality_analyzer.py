#!/usr/bin/env python3
"""
Test 5: Quality Analyzer
Tests image quality assessment functionality
"""

import sys
import os
import cv2
import numpy as np
import time
from pathlib import Path

# Fix the path issue - add both project root and src to path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"

# Ensure paths are in sys.path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

def test_quality_analyzer():
    """Test quality analyzer functionality"""
    print("=" * 60)
    print("TEST 5: QUALITY ANALYZER")
    print("=" * 60)
    
    try:
        # Test image loading
        test_image_path = project_root / "data" / "sample_images" / "img1.jpg"
        if not test_image_path.exists():
            print(f"✗ Test image not found: {test_image_path}")
            return False
            
        image = cv2.imread(str(test_image_path))
        print(f"✓ Test image loaded: {image.shape}")
        
        # Import quality analyzer with absolute import
        try:
            # Try absolute import first
            sys.path.insert(0, str(src_path))
            from src.preprocessing.quality_analyzer import QualityAnalyzer
            print("✓ QualityAnalyzer imported successfully")
        except ImportError as e1:
            try:
                # Try direct module import
                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    "quality_analyzer", 
                    src_path / "preprocessing" / "quality_analyzer.py"
                )
                qa_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(qa_module)
                QualityAnalyzer = qa_module.QualityAnalyzer
                print("✓ QualityAnalyzer imported via direct module loading")
            except Exception as e2:
                print(f"✗ Failed to import QualityAnalyzer: {e1}")
                print(f"✗ Direct import also failed: {e2}")
                return False
        
        # Initialize analyzer
        try:
            analyzer = QualityAnalyzer()
            print("✓ QualityAnalyzer initialized")
        except Exception as e:
            print(f"✗ Failed to initialize QualityAnalyzer: {e}")
            return False
        
        # Test comprehensive quality analysis
        try:
            start_time = time.time()
            quality_metrics = analyzer.analyze_image(image)
            processing_time = time.time() - start_time
            
            print("✓ Comprehensive quality analysis completed")
            print(f"  - Overall score: {quality_metrics.overall_score:.2f}")
            print(f"  - Quality level: {quality_metrics.quality_level.value}")
            print(f"  - Image type: {quality_metrics.image_type.value}")
            print(f"  - Processing time: {processing_time:.3f}s")
        except Exception as e:
            print(f"✗ Comprehensive analysis failed: {e}")
            print(f"  Error details: {type(e).__name__}: {e}")
            return False
        
        # Test individual metrics
        try:
            print("✓ Individual quality metrics:")
            print(f"  - Sharpness: {quality_metrics.sharpness_score:.2f}")
            print(f"  - Noise level: {quality_metrics.noise_level:.2f}")
            print(f"  - Contrast: {quality_metrics.contrast_score:.2f}")
            print(f"  - Brightness: {quality_metrics.brightness_score:.2f}")
            print(f"  - Skew angle: {quality_metrics.skew_angle:.2f}°")
            print(f"  - Text density: {quality_metrics.text_density:.2f}")
        except Exception as e:
            print(f"✗ Individual metrics access failed: {e}")
            return False
        
        # Test warnings system
        try:
            if quality_metrics.warnings:
                print(f"✓ Quality warnings generated: {len(quality_metrics.warnings)}")
                for i, warning in enumerate(quality_metrics.warnings[:3], 1):
                    print(f"  {i}. {warning}")
            else:
                print("✓ No quality warnings (image appears good)")
        except Exception as e:
            print(f"✗ Warnings system failed: {e}")
            return False
        
        # Test caching functionality
        try:
            cached_metrics = analyzer.analyze_image(image, cache_key="test_image")
            print("✓ Analysis caching works")
        except Exception as e:
            print(f"✗ Caching functionality failed: {e}")
            return False
        
        # Test edge cases
        try:
            # Test with grayscale image
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray_metrics = analyzer.analyze_image(gray_image)
            print(f"✓ Grayscale analysis: score={gray_metrics.overall_score:.2f}")
            
            # Test with small image
            small_image = cv2.resize(image, (200, 150))
            small_metrics = analyzer.analyze_image(small_image)
            print(f"✓ Small image analysis: score={small_metrics.overall_score:.2f}")
            
        except Exception as e:
            print(f"✗ Edge case testing failed: {e}")
            return False
        
        print("=" * 60)
        print("TEST 5: PASSED - Quality analyzer works correctly")
        print("=" * 60)
        print("SUCCESS: Quality analyzer is functioning properly")
        print("Next step: Run Test 6 (Image Enhancer)")
        return True
        
    except Exception as e:
        print(f"✗ CRITICAL ERROR in Test 5: {e}")
        import traceback
        traceback.print_exc()
        print("=" * 60)
        print("TEST 5: FAILED")
        print("=" * 60)
        return False

if __name__ == "__main__":
    success = test_quality_analyzer()
    sys.exit(0 if success else 1)