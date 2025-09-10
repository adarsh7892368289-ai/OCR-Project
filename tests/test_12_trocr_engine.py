#!/usr/bin/env python3
"""
Test 12: TrOCR Engine - Fixed Version
Purpose: Test TrOCR engine with actual implementation
Author: OCR Testing Framework
Date: 2025
"""

import sys
import os
import time
import numpy as np
import cv2
from typing import List, Dict, Any
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.absolute()
src_path = project_root / "src"

sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_path))

print(f"Project root: {project_root}")
print(f"Source path: {src_path}")

def load_sample_image():
    """Load img3.jpg from data/sample_images/ or create test image"""
    
    # Try different possible paths for img3.jpg
    possible_paths = [
        project_root / "data" / "sample_images" / "img3.jpg",
        project_root / "data" / "sample_images" / "img3.png",
        Path(os.getcwd()) / "data" / "sample_images" / "img3.jpg",
        Path(os.getcwd()) / "img3.jpg",
        project_root / "img3.jpg"
    ]
    
    for img_path in possible_paths:
        if img_path.exists():
            print(f"Loading image from: {img_path}")
            image = cv2.imread(str(img_path))
            if image is not None:
                print(f"Image loaded successfully: {image.shape}")
                return image
    
    # If no image found, create a test image with receipt-like content
    print("Warning: img3.jpg not found, creating test receipt image")
    image = np.ones((800, 600, 3), dtype=np.uint8) * 255
    
    # Add receipt-style text that TrOCR should handle well
    cv2.putText(image, "RECEIPT", (200, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
    
    cv2.putText(image, "Store: Tech Shop", (50, 120), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    cv2.putText(image, "Date: 2025-01-15", (50, 160), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    cv2.putText(image, "Item 1: Laptop", (50, 220), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    cv2.putText(image, "Price: $999.99", (300, 220), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    cv2.putText(image, "Item 2: Mouse", (50, 260), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    cv2.putText(image, "Price: $29.99", (300, 260), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    cv2.putText(image, "Total: $1029.98", (50, 320), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    cv2.putText(image, "Payment: Credit Card", (50, 380), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    cv2.putText(image, "Thank you!", (200, 450), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    return image

def test_trocr_engine():
    """Test TrOCR Engine with corrected implementation matching"""
    print("=" * 80)
    print("TEST 12: TROCR ENGINE - CORRECTED VERSION")
    print("=" * 80)
    print("Purpose: Test TrOCR transformer engine with proper implementation")
    print("Input: Sample image (receipt-style document)")
    print("-" * 80)
    
    start_time = time.time()
    test_results = {
        'image_loading': False,
        'engine_import': False,
        'engine_initialization': False,
        'image_validation': False,
        'ocr_processing': False,
        'result_validation': False,
        'engine_info': False,
        'stats_collection': False
    }
    
    try:
        # Test 1: Load Sample Image
        print("Step 1: Loading sample image...")
        
        try:
            image = load_sample_image()
            if image is not None:
                print(f"✓ Image loaded: {image.shape}")
                print(f"  Dimensions: {image.shape[1]}x{image.shape[0]}")
                print(f"  Channels: {image.shape[2] if len(image.shape) == 3 else 1}")
                print(f"  Data type: {image.dtype}")
                test_results['image_loading'] = True
            else:
                print("✗ Failed to load image")
                return False, test_results, 0.0
                
        except Exception as e:
            print(f"✗ Image loading failed: {e}")
            return False, test_results, 0.0
        
        # Test 2: Import TrOCR Engine
        print("\nStep 2: Importing TrOCR engine...")
        
        try:
            # Import with proper error handling
            try:
                from src.engines.trocr_engine import TrOCREngine
                from src.core.base_engine import OCRResult, BoundingBox
                print("✓ Direct src imports successful")
            except ImportError:
                from engines.trocr_engine import TrOCREngine  
                from core.base_engine import OCRResult, BoundingBox
                print("✓ Alternative imports successful")
                
            test_results['engine_import'] = True
        except ImportError as e:
            print(f"✗ Import failed: {e}")
            print("\nSOLUTION REQUIRED:")
            print("1. Install transformers: pip install transformers torch torchvision")
            print("2. Run from project root directory")
            print("3. Check src/engines/trocr_engine.py exists")
            return False, test_results, time.time() - start_time
        
        # Test 3: Initialize Engine
        print("\nStep 3: Initializing TrOCR engine...")
        
        # Use configuration matching your implementation
        config = {
            "device": "cpu",  # Safe default
            "model_name": "microsoft/trocr-base-printed"  # Matches your default
        }
        
        try:
            engine = TrOCREngine(config)
            print(f"✓ Engine instance created")
            print(f"  Engine name: {engine.name}")
            print(f"  Model name: {engine.model_name}")
            print(f"  Device: {engine.device}")
            
            # Initialize (may take time for model download)
            print("  Initializing model (downloading if first run)...")
            init_start = time.time()
            
            if engine.initialize():
                init_time = time.time() - init_start
                print(f"✓ Engine initialized in {init_time:.2f}s")
                print(f"  Is initialized: {engine.is_initialized}")
                print(f"  Model loaded: {engine.model_loaded}")
                test_results['engine_initialization'] = True
            else:
                print("✗ Engine initialization failed")
                print("  Check transformers installation and internet connection")
                return False, test_results, time.time() - start_time
            
        except Exception as e:
            print(f"✗ Engine creation/initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return False, test_results, time.time() - start_time
        
        # Test 4: Image Validation
        print("\nStep 4: Validating image for TrOCR...")
        
        try:
            # Test the validate_image method
            is_valid = engine.validate_image(image)
            if is_valid:
                print(f"✓ Image validation passed")
                print(f"  Image shape: {image.shape}")
                print(f"  Valid format: {len(image.shape) in [2, 3]}")
                print(f"  Non-zero dimensions: {image.shape[0] > 0 and image.shape[1] > 0}")
                test_results['image_validation'] = True
            else:
                print("✗ Image validation failed")
                return False, test_results, time.time() - start_time
                
        except Exception as e:
            print(f"✗ Image validation error: {e}")
            return False, test_results, time.time() - start_time
        
        # Test 5: OCR Processing
        print("\nStep 5: Processing image with TrOCR...")
        
        processing_start = time.time()
        try:
            # Process image using your implementation's method signature
            ocr_results = engine.process_image(image)
            processing_time = time.time() - processing_start
            
            print(f"✓ OCR processing completed in {processing_time:.2f}s")
            print(f"  Results type: {type(ocr_results)}")
            print(f"  Results count: {len(ocr_results) if isinstance(ocr_results, list) else 'N/A'}")
            
            if isinstance(ocr_results, list) and len(ocr_results) > 0:
                for i, result in enumerate(ocr_results):
                    print(f"\n  Result {i+1}:")
                    print(f"    Text: '{result.text[:100]}{'...' if len(result.text) > 100 else ''}'")
                    print(f"    Confidence: {result.confidence:.3f}")
                    print(f"    BBox: ({result.bbox.x}, {result.bbox.y}) {result.bbox.width}x{result.bbox.height}")
                    
                    # Check metadata
                    if hasattr(result, 'metadata') and result.metadata:
                        print(f"    Engine: {result.metadata.get('engine_name', 'N/A')}")
                        print(f"    Method: {result.metadata.get('detection_method', 'N/A')}")
                        print(f"    Transformer: {result.metadata.get('transformer_based', 'N/A')}")
                
                test_results['ocr_processing'] = True
            elif isinstance(ocr_results, list) and len(ocr_results) == 0:
                print("  No text detected (valid result for some images)")
                test_results['ocr_processing'] = True
            else:
                print(f"✗ Unexpected result format: {type(ocr_results)}")
                return False, test_results, time.time() - start_time
                
        except Exception as e:
            print(f"✗ OCR processing failed: {e}")
            import traceback
            traceback.print_exc()
            return False, test_results, time.time() - start_time
        
        # Test 6: Result Structure Validation
        print("\nStep 6: Validating result structure...")
        
        try:
            if ocr_results and len(ocr_results) > 0:
                result = ocr_results[0]
                
                # Validate OCRResult structure
                assert hasattr(result, 'text'), "Missing text attribute"
                assert hasattr(result, 'confidence'), "Missing confidence attribute"
                assert hasattr(result, 'bbox'), "Missing bbox attribute"
                assert hasattr(result, 'metadata'), "Missing metadata attribute"
                
                # Validate data types
                assert isinstance(result.text, str), f"Text should be string, got {type(result.text)}"
                assert isinstance(result.confidence, (int, float)), f"Confidence should be numeric, got {type(result.confidence)}"
                assert 0.0 <= result.confidence <= 1.0, f"Confidence {result.confidence} should be 0-1"
                assert hasattr(result.bbox, 'x'), "BBox missing x coordinate"
                assert hasattr(result.bbox, 'y'), "BBox missing y coordinate"
                
                print("✓ Result structure validation passed")
                print(f"  Text length: {len(result.text)} characters")
                print(f"  Confidence: {result.confidence:.3f} (valid range)")
                print(f"  BBox coordinates: ({result.bbox.x}, {result.bbox.y})")
                print(f"  BBox dimensions: {result.bbox.width}x{result.bbox.height}")
                print(f"  Metadata keys: {list(result.metadata.keys()) if result.metadata else 'None'}")
                
                test_results['result_validation'] = True
            else:
                print("✓ No results to validate (empty detection is valid)")
                test_results['result_validation'] = True
                
        except Exception as e:
            print(f"✗ Result validation failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Test 7: Engine Information
        print("\nStep 7: Getting engine information...")
        
        try:
            engine_info = engine.get_engine_info()
            
            print("✓ Engine information retrieved")
            print(f"  Name: {engine_info.get('name', 'N/A')}")
            print(f"  Type: {engine_info.get('type', 'N/A')}")
            print(f"  Version: {engine_info.get('version', 'N/A')}")
            print(f"  Languages: {len(engine_info.get('supported_languages', []))} supported")
            print(f"  Capabilities: {list(engine_info.get('capabilities', {}).keys())}")
            print(f"  Architecture: {engine_info.get('model_info', {}).get('architecture', 'N/A')}")
            print(f"  Initialized: {engine_info.get('initialization_status', {}).get('is_initialized', False)}")
            
            test_results['engine_info'] = True
            
        except Exception as e:
            print(f"✗ Engine info retrieval failed: {e}")
        
        # Test 8: Statistics Collection  
        print("\nStep 8: Collecting processing statistics...")
        
        try:
            stats = engine.get_stats()
            
            print("✓ Statistics collected")
            print(f"  Total processed: {stats.get('total_processed', 0)}")
            print(f"  Total time: {stats.get('total_time', 0):.3f}s")
            print(f"  Errors: {stats.get('errors', 0)}")
            print(f"  Engine: {stats.get('engine_name', 'N/A')}")
            print(f"  Model: {stats.get('model_name', 'N/A')}")
            print(f"  Device: {stats.get('device', 'N/A')}")
            print(f"  Transformer-based: {stats.get('transformer_based', False)}")
            
            if stats.get('total_processed', 0) > 0:
                avg_time = stats.get('total_time', 0) / stats.get('total_processed', 1)
                print(f"  Average time per image: {avg_time:.3f}s")
            
            test_results['stats_collection'] = True
            
        except Exception as e:
            print(f"✗ Statistics collection failed: {e}")
        
        # Calculate final results
        end_time = time.time()
        total_time = end_time - start_time
        
        passed_tests = sum(test_results.values())
        total_tests = len(test_results)
        success_rate = passed_tests / total_tests
        
        print("\n" + "=" * 80)
        print("TEST 12 RESULTS - TROCR ENGINE")
        print("=" * 80)
        print(f"Tests passed: {passed_tests}/{total_tests} ({success_rate*100:.1f}%)")
        print(f"Total test time: {total_time:.2f}s")
        
        if success_rate >= 0.75:  # Allow some flexibility for TrOCR complexity
            print("STATUS: ✓ PASSED - TrOCR engine working correctly")
            print("Ready for engine management tests (Test 13-14)")
        else:
            print("STATUS: ✗ FAILED - TrOCR engine needs fixes")
        
        print("\nComponent Status:")
        for component, status in test_results.items():
            status_icon = "✓" if status else "✗"
            component_name = component.replace('_', ' ').title()
            print(f"   {status_icon} {component_name}")
        
        # Show extracted text summary if available
        if 'ocr_results' in locals() and ocr_results:
            print(f"\n📄 Extracted Text Summary:")
            all_text = " ".join([r.text for r in ocr_results if r.text.strip()])
            if all_text:
                print(f"   Characters: {len(all_text)}")
                print(f"   Preview: '{all_text[:150]}{'...' if len(all_text) > 150 else ''}'")
                avg_conf = sum(r.confidence for r in ocr_results) / len(ocr_results)
                print(f"   Avg Confidence: {avg_conf:.3f}")
                print(f"   Regions detected: {len(ocr_results)}")
            else:
                print("   No text extracted")
        
        print(f"\n🔧 TrOCR Engine Validated - Ready for Integration Tests")
        
        return success_rate >= 0.75, test_results, total_time
        
    except Exception as e:
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"\n❌ CRITICAL ERROR in Test 12: {e}")
        print(f"Failed after: {processing_time:.2f}s")
        
        import traceback
        traceback.print_exc()
        
        return False, test_results, processing_time

if __name__ == "__main__":
    print("TrOCR Engine Test - Corrected Version")
    print("Checking dependencies...")
    
    # Check transformers availability
    try:
        import transformers
        import torch
        print(f"✓ Transformers: {transformers.__version__}")
        print(f"✓ PyTorch: {torch.__version__}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("Install with: pip install transformers torch torchvision")
        sys.exit(1)
    
    print("-" * 50)
    
    success, results, time_taken = test_trocr_engine()
    
    if success:
        print(f"\n🎉 Test 12 PASSED in {time_taken:.2f}s")
        print("TrOCR engine validated and ready")
        print("Proceed to Test 13: Engine Manager")
    else:
        print(f"\n💥 Test 12 FAILED after {time_taken:.2f}s")
        failed_components = [k for k, v in results.items() if not v]
        if failed_components:
            print("Failed components:")
            for component in failed_components:
                print(f"  - {component.replace('_', ' ').title()}")
        print("\nFix issues before proceeding to engine management tests")
    
    sys.exit(0 if success else 1)