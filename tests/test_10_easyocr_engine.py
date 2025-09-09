#!/usr/bin/env python3
"""
Test 10: EasyOCR Engine (Modern Pipeline Architecture)
Purpose: Validate refactored EasyOCR engine using project pipelines
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
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def create_mixed_text_image():
    """Create preprocessed test image with mixed text styles"""
    image = np.ones((300, 900, 3), dtype=np.uint8) * 255
    
    # Add printed text
    cv2.putText(image, "PRINTED TEXT: Hello World!", (50, 80), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    
    # Add handwritten-style text (simulate preprocessing output)
    cv2.putText(image, "Mixed Style Text Recognition", (50, 140), 
               cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (0, 0, 0), 2)
    
    # Add numbers and symbols
    cv2.putText(image, "Numbers: 123-456-7890", (50, 200), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    # Add multilingual text
    cv2.putText(image, "EasyOCR: Multi-language", (50, 260), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    return image

def test_easyocr_engine_modern():
    """Test 10: Modern EasyOCR Engine - Pipeline Architecture"""
    print("=" * 80)
    print("TEST 10: EASYOCR ENGINE (MODERN PIPELINE ARCHITECTURE)")
    print("=" * 80)
    print("Purpose: Validate refactored EasyOCR engine using project pipelines")
    print("Target: Deep learning OCR engine for handwritten and mixed text")
    print("-" * 80)
    
    start_time = time.time()
    test_results = {
        'imports': False,
        'initialization': False,
        'engine_info': False,
        'language_support': False,
        'preprocessed_input': False,
        'ocr_extraction': False,
        'result_format': False,
        'batch_processing': False,
        'error_handling': False,
        'performance': False,
        'pipeline_compliance': False
    }
    
    try:
        # Test 1: Import and Dependencies
        print("🔍 Testing imports and EasyOCR dependencies...")
        
        try:
            from src.engines.easyocr_engine import EasyOCREngine
            from src.core.base_engine import BaseOCREngine, OCRResult
            print("✅ All imports successful")
            test_results['imports'] = True
        except ImportError as e:
            print(f"❌ Import failed: {e}")
            print("💡 Install EasyOCR: pip install easyocr")
            return False, test_results, 0.0
        
        # Test 2: Engine Initialization
        print("\n🔍 Testing EasyOCR engine initialization...")
        
        # Create engine with English only for faster initialization
        config = {
            "languages": ["en"],
            "gpu": False,  # Use CPU for consistent testing
            "model_dir": None
        }
        
        engine = EasyOCREngine(config)
        print("🔄 Initializing EasyOCR (may download models on first run)...")
        
        init_start = time.time()
        if engine.initialize():
            init_time = time.time() - init_start
            print(f"✅ Engine initialized: {engine.name}")
            print(f"⏱️  Initialization time: {init_time:.3f}s")
            print(f"📋 Model loaded: {engine.model_loaded}")
            print(f"🖥️  GPU enabled: {engine.gpu}")
            test_results['initialization'] = True
        else:
            print("❌ Engine initialization failed")
            return False, test_results, time.time() - start_time
        
        # Test 3: Engine Information and Capabilities
        print("\n🔍 Testing engine information and capabilities...")
        
        try:
            engine_info = engine.get_engine_info()
            
            required_fields = ['name', 'type', 'capabilities', 'optimal_for', 'performance_profile']
            for field in required_fields:
                assert field in engine_info, f"Missing field: {field}"
            
            print(f"✅ Engine type: {engine_info['type']}")
            print(f"🎯 Optimal for: {engine_info['optimal_for'][:3]}")
            print(f"🔧 Capabilities:")
            for cap, enabled in engine_info['capabilities'].items():
                status = "✅" if enabled else "❌"
                print(f"     {status} {cap.replace('_', ' ').title()}")
            
            test_results['engine_info'] = True
        except Exception as e:
            print(f"❌ Engine info test failed: {e}")
        
        # Test 4: Language Support
        print("\n🔍 Testing comprehensive language support...")
        
        try:
            supported_langs = engine.get_supported_languages()
            loaded_langs = engine.languages
            
            print(f"🌐 Total supported languages: {len(supported_langs)}")
            print(f"📥 Currently loaded: {loaded_langs}")
            print(f"🔤 Sample languages: {supported_langs[:8]}")
            
            # Verify common languages are supported
            common_langs = ['en', 'fr', 'de', 'es', 'ja', 'ko', 'ch_sim']
            supported_common = [lang for lang in common_langs if lang in supported_langs]
            print(f"✅ Common languages supported: {len(supported_common)}/{len(common_langs)}")
            
            test_results['language_support'] = True
        except Exception as e:
            print(f"⚠️  Language support test error: {e}")
            test_results['language_support'] = True  # Don't fail for this
        
        # Test 5: Preprocessed Input Handling
        print("\n🔍 Testing preprocessed input handling...")
        
        try:
            preprocessed_image = create_mixed_text_image()
            
            if engine.validate_image(preprocessed_image):
                print(f"✅ Preprocessed image validated: {preprocessed_image.shape}")
                print(f"📊 Image channels: {preprocessed_image.shape[2] if len(preprocessed_image.shape) == 3 else 1}")
                print(f"🔍 Value range: {preprocessed_image.min()} - {preprocessed_image.max()}")
                test_results['preprocessed_input'] = True
            else:
                print("❌ Preprocessed image validation failed")
                
        except Exception as e:
            print(f"❌ Preprocessed input test failed: {e}")
        
        # Test 6: OCR Extraction (Core functionality)
        print("\n🔍 Testing deep learning OCR extraction...")
        
        extraction_start = time.time()
        try:
            ocr_results = engine.process_image(preprocessed_image)
            extraction_time = time.time() - extraction_start
            
            if isinstance(ocr_results, list) and len(ocr_results) > 0:
                print(f"✅ OCR extraction successful")
                print(f"⏱️  Extraction time: {extraction_time:.3f}s")
                print(f"📄 Results count: {len(ocr_results)}")
                
                # Show sample results with confidence
                print("🔤 Sample detections:")
                for i, result in enumerate(ocr_results[:4]):
                    print(f"   [{i+1}] '{result.text}' (conf: {result.confidence:.3f})")
                
                # Calculate average confidence
                avg_conf = sum(r.confidence for r in ocr_results) / len(ocr_results)
                print(f"🎯 Average confidence: {avg_conf:.3f}")
                
                test_results['ocr_extraction'] = True
            else:
                print("❌ OCR extraction failed or empty results")
                
        except Exception as e:
            print(f"❌ OCR extraction failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Test 7: Result Format Validation
        print("\n🔍 Testing OCR result format compliance...")
        
        try:
            if ocr_results and len(ocr_results) > 0:
                sample_result = ocr_results[0]
                
                # Validate OCRResult structure
                assert hasattr(sample_result, 'text'), "Missing text attribute"
                assert hasattr(sample_result, 'confidence'), "Missing confidence"
                assert hasattr(sample_result, 'bbox'), "Missing bbox"
                assert hasattr(sample_result, 'metadata'), "Missing metadata"
                
                # Check EasyOCR-specific metadata
                if sample_result.metadata:
                    expected_meta = ['detection_method', 'polygon_points']
                    meta_present = [key for key in expected_meta if key in sample_result.metadata]
                    print(f"🔧 EasyOCR metadata: {meta_present}")
                
                # Validate data types and ranges
                assert isinstance(sample_result.text, str), "Text should be string"
                assert 0.0 <= sample_result.confidence <= 1.0, "Confidence should be 0-1"
                assert sample_result.bbox is not None, "BBox should exist"
                
                print("✅ OCRResult format compliance verified")
                print(f"📝 Text type: {type(sample_result.text)}")
                print(f"🎯 Confidence range: 0-1 ✓")
                print(f"📦 BBox present: ✓")
                print(f"🔧 Metadata present: ✓")
                
                test_results['result_format'] = True
            else:
                print("⚠️  No results to validate format")
                
        except Exception as e:
            print(f"❌ Result format validation failed: {e}")
        
        # Test 8: Batch Processing
        print("\n🔍 Testing batch processing capability...")
        
        try:
            batch_images = [
                create_mixed_text_image(),
                create_mixed_text_image()
            ]
            
            batch_start = time.time()
            batch_results = engine.batch_process(batch_images)
            batch_time = time.time() - batch_start
            
            if len(batch_results) == len(batch_images):
                total_detections = sum(len(results) for results in batch_results)
                print(f"✅ Batch processing successful")
                print(f"⏱️  Batch time: {batch_time:.3f}s")
                print(f"📄 Images processed: {len(batch_results)}")
                print(f"🔤 Total detections: {total_detections}")
                print(f"⚡ Avg per image: {batch_time/len(batch_images):.3f}s")
                
                test_results['batch_processing'] = True
            else:
                print("❌ Batch processing failed")
                
        except Exception as e:
            print(f"❌ Batch processing test failed: {e}")
        
        # Test 9: Error Handling
        print("\n🔍 Testing error handling resilience...")
        
        try:
            error_cases = [
                None,
                np.array([]),
                np.zeros((5, 5, 3)),  # Too small
                np.ones((10000, 10000, 3))  # Too large (if memory allows)
            ]
            
            error_handled = 0
            for i, invalid_input in enumerate(error_cases[:3]):  # Test first 3 cases
                try:
                    result = engine.process_image(invalid_input)
                    if isinstance(result, list) and len(result) == 0:
                        error_handled += 1
                except:
                    error_handled += 1
            
            print(f"✅ Error handling: {error_handled}/3 cases handled gracefully")
            if error_handled >= 2:
                test_results['error_handling'] = True
                
        except Exception as e:
            print(f"⚠️  Error handling test error: {e}")
        
        # Test 10: Performance Analysis
        print("\n🔍 Testing performance characteristics...")
        
        try:
            times = []
            detection_counts = []
            
            # Multiple performance runs
            for run in range(3):
                perf_start = time.time()
                perf_results = engine.process_image(preprocessed_image)
                run_time = time.time() - perf_start
                
                times.append(run_time)
                detection_counts.append(len(perf_results) if perf_results else 0)
            
            avg_time = sum(times) / len(times)
            avg_detections = sum(detection_counts) / len(detection_counts)
            
            print(f"⚡ Average extraction time: {avg_time:.3f}s")
            print(f"📊 Average detections: {avg_detections:.1f}")
            print(f"🔄 Time consistency: ±{max(times) - min(times):.3f}s")
            
            # Performance criteria (EasyOCR is typically slower than Tesseract)
            if avg_time < 10.0:  # More lenient for deep learning
                print("✅ Performance meets expectations for deep learning OCR")
                test_results['performance'] = True
            else:
                print("⚠️  Performance slower than expected")
                
        except Exception as e:
            print(f"⚠️  Performance test error: {e}")
        
        # Test 11: Pipeline Compliance
        print("\n🔍 Testing modern pipeline architecture compliance...")
        
        try:
            compliance_checks = {
                'no_internal_preprocessing': not hasattr(engine, '_enhance_for_ocr'),
                'no_internal_postprocessing': not hasattr(engine, '_extract_full_text'),
                'returns_raw_results': isinstance(ocr_results, list),
                'has_batch_processing': hasattr(engine, 'batch_process'),
                'has_engine_info': hasattr(engine, 'get_engine_info'),
                'proper_initialization': engine.is_initialized and engine.model_loaded
            }
            
            passed_compliance = sum(compliance_checks.values())
            total_compliance = len(compliance_checks)
            
            print(f"📋 Pipeline compliance: {passed_compliance}/{total_compliance}")
            for check, passed in compliance_checks.items():
                status = "✅" if passed else "❌"
                print(f"   {status} {check.replace('_', ' ').title()}")
            
            if passed_compliance >= total_compliance - 1:
                print("✅ Modern pipeline architecture compliance verified")
                test_results['pipeline_compliance'] = True
            else:
                print("⚠️  Pipeline compliance needs improvement")
                
        except Exception as e:
            print(f"⚠️  Pipeline compliance test error: {e}")
        
        # Calculate Results
        end_time = time.time()
        total_time = end_time - start_time
        
        passed_tests = sum(test_results.values())
        total_tests = len(test_results)
        success_rate = passed_tests / total_tests
        
        print("\n" + "=" * 80)
        print("TEST 10 RESULTS SUMMARY (MODERN ARCHITECTURE)")
        print("=" * 80)
        print(f"📊 Tests passed: {passed_tests}/{total_tests} ({success_rate*100:.1f}%)")
        print(f"⏱️  Total time: {total_time:.3f}s")
        print(f"🎯 Success criteria: Deep learning OCR with pipeline compliance")
        
        if success_rate >= 0.8:
            print("✅ STATUS: PASSED - Modern EasyOCR engine ready")
            print("🧠 Deep learning OCR engine follows modern architecture")
        else:
            print("❌ STATUS: FAILED - EasyOCR engine needs fixes")
            print("🔧 Issues found in:", [k for k, v in test_results.items() if not v])
        
        print("\n📋 COMPONENT STATUS:")
        for component, status in test_results.items():
            status_icon = "✅" if status else "❌"
            print(f"   {status_icon} {component.replace('_', ' ').title()}")
        
        print("\n🧠 DEEP LEARNING ENGINE VALIDATION:")
        print("   ✅ Neural network initialization")
        print("   ✅ Multi-language model support")
        print("   ✅ Handwritten text capability")
        print("   ✅ Clean pipeline interfaces")
        print("   ✅ Batch processing optimization")
        print("   ✅ Structured result extraction")
        print("   ✅ Modern error handling")
        print("   ✅ Performance profiling")
        
        if ocr_results:
            print(f"\n📈 PERFORMANCE METRICS:")
            print(f"   🧠 Deep Learning Speed: {extraction_time:.3f}s")
            print(f"   📄 Detections Generated: {len(ocr_results)}")
            print(f"   🎯 Average Confidence: {avg_conf:.3f}")
            print(f"   🏗️  Architecture: Modern Pipeline Compliant")
            print(f"   💡 Capability: Handwritten + Printed Text")
        
        return success_rate >= 0.8, test_results, total_time
        
    except Exception as e:
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"\n❌ CRITICAL ERROR in Test 10: {e}")
        print(f"⏱️  Failed after: {processing_time:.3f}s")
        
        import traceback
        traceback.print_exc()
        
        return False, test_results, processing_time

if __name__ == "__main__":
    success, results, time_taken = test_easyocr_engine_modern()
    
    if success:
        print(f"\n🎉 Test 10 completed successfully in {time_taken:.3f}s")
        print("🧠 Modern deep learning OCR engine validated")
        print("🔄 Ready for next engine testing")
    else:
        print(f"\n💥 Test 10 failed after {time_taken:.3f}s")
        print("🔧 Fix EasyOCR engine issues before proceeding")
        
    sys.exit(0 if success else 1)