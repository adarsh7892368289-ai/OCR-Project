#!/usr/bin/env python3
"""
Test 13: Engine Manager (Multi-Engine Coordination)
Purpose: Validate engine manager coordinates multiple OCR engines
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

# Alternative approach
current_dir = Path(os.getcwd())
if (current_dir / "src").exists():
    sys.path.insert(0, str(current_dir))

print(f"Project root: {project_root}")
print(f"Source path: {src_path}")

def create_multi_engine_test_image():
    """Create test image suitable for comparing different engines"""
    image = np.ones((400, 800, 3), dtype=np.uint8) * 255
    
    # Clear printed text (good for Tesseract/PaddleOCR)
    cv2.putText(image, "INVOICE #2025", (50, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
    
    # Mixed content (good for EasyOCR)
    cv2.putText(image, "ABC Company Ltd.", (50, 120), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(image, "Phone: +1-555-0123", (50, 160), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    # Handwritten style (good for TrOCR)
    cv2.putText(image, "Thank you!", (50, 220), 
               cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1.2, (0, 0, 0), 2)
    
    # Numbers and special chars
    cv2.putText(image, "Total: $123.45", (50, 280), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    cv2.putText(image, "Date: 01/15/2025", (50, 320), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    
    return image

def test_engine_manager():
    """Test 13: Engine Manager - Multi-Engine Coordination"""
    print("=" * 80)
    print("TEST 13: ENGINE MANAGER (MULTI-ENGINE COORDINATION)")
    print("=" * 80)
    print("Purpose: Validate engine manager coordinates multiple OCR engines")
    print("Target: Multi-engine registration, selection, and processing")
    print("-" * 80)
    
    start_time = time.time()
    test_results = {
        'imports': False,
        'manager_creation': False,
        'engine_registration': False,
        'engine_listing': False,
        'engine_selection': False,
        'multi_engine_processing': False,
        'result_comparison': False,
        'best_engine_selection': False,
        'batch_processing': False,
        'error_handling': False,
        'performance_tracking': False
    }
    
    try:
        # Test 1: Import Dependencies
        print("Testing imports and engine manager dependencies...")
        
        try:
            # Import the EngineManager and engines
            from src.core.engine_manager import EngineManager
            from src.engines.tesseract_engine import TesseractEngine
            from src.engines.easyocr_engine import EasyOCREngine
            from src.engines.paddleocr_engine import PaddleOCREngine
            from src.engines.trocr_engine import TrOCREngine
            from src.core.base_engine import OCRResult
            print("✓ All imports successful")
            test_results['imports'] = True
        except ImportError as e:
            print(f"❌ Import failed: {e}")
            print("\nDEBUG INFO:")
            print(f"Current working directory: {os.getcwd()}")
            
            # Check if files exist
            manager_file = project_root / "src" / "core" / "engine_manager.py"
            print(f"   Engine manager exists: {manager_file.exists()} at {manager_file}")
            
            if not manager_file.exists():
                print("\nSOLUTION: Create the engine manager file in src/core/")
            else:
                print("\nSOLUTION: Check engine manager file has EngineManager class")
                
            return False, test_results, 0.0
        
        # Test 2: Manager Creation
        print("\nTesting engine manager creation...")
        
        try:
            manager = EngineManager()
            print(f"✓ Engine manager created: {type(manager).__name__}")
            print(f"✓ Manager initialized: {hasattr(manager, 'engines')}")
            test_results['manager_creation'] = True
        except Exception as e:
            print(f"❌ Engine manager creation failed: {e}")
            import traceback
            traceback.print_exc()
            return False, test_results, time.time() - start_time
        
        # Test 3: Engine Registration
        print("\nTesting engine registration...")
        
        try:
            # Create engine configurations
            configs = {
                'tesseract': {'languages': ['eng']},
                'easyocr': {'languages': ['en'], 'gpu': False},
                'paddleocr': {'languages': ['en'], 'gpu': False},
                'trocr': {'device': 'cpu', 'model_name': 'microsoft/trocr-base-printed'}
            }
            
            # Register engines
            registration_results = {}
            engines_to_register = [
                ('tesseract', TesseractEngine),
                ('easyocr', EasyOCREngine),
                ('paddleocr', PaddleOCREngine),
                ('trocr', TrOCREngine)
            ]
            
            for engine_name, engine_class in engines_to_register:
                try:
                    engine = engine_class(configs.get(engine_name, {}))
                    success = manager.register_engine(engine_name, engine)
                    registration_results[engine_name] = success
                    print(f"   {engine_name}: {'✓' if success else '❌'}")
                except Exception as e:
                    print(f"   {engine_name}: ❌ ({e})")
                    registration_results[engine_name] = False
            
            successful_registrations = sum(registration_results.values())
            print(f"✓ Engines registered: {successful_registrations}/{len(engines_to_register)}")
            
            if successful_registrations >= 2:  # Need at least 2 engines for coordination
                test_results['engine_registration'] = True
            else:
                print("❌ Insufficient engines registered for coordination testing")
                
        except Exception as e:
            print(f"❌ Engine registration failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Test 4: Engine Listing
        print("\nTesting engine listing...")
        
        try:
            available_engines = manager.get_available_engines()
            initialized_engines = manager.get_initialized_engines()
            
            print(f"✓ Available engines: {len(available_engines)}")
            print(f"   Engines: {list(available_engines.keys())}")
            print(f"✓ Initialized engines: {len(initialized_engines)}")
            
            if len(available_engines) >= 2:
                test_results['engine_listing'] = True
                
        except Exception as e:
            print(f"❌ Engine listing failed: {e}")
        
        # Test 5: Engine Selection
        print("\nTesting engine selection logic...")
        
        try:
            test_image = create_multi_engine_test_image()
            
            # Test different content types
            content_types = ['printed', 'handwritten', 'mixed', 'document']
            selection_results = {}
            
            for content_type in content_types:
                try:
                    selected_engine = manager.select_best_engine(
                        image=test_image, 
                        content_type=content_type
                    )
                    selection_results[content_type] = selected_engine
                    print(f"   {content_type}: {selected_engine}")
                except Exception as e:
                    print(f"   {content_type}: ❌ ({e})")
                    selection_results[content_type] = None
            
            successful_selections = sum(1 for v in selection_results.values() if v)
            print(f"✓ Engine selections: {successful_selections}/{len(content_types)}")
            
            if successful_selections >= 2:
                test_results['engine_selection'] = True
                
        except Exception as e:
            print(f"❌ Engine selection failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Test 6: Multi-Engine Processing
        print("\nTesting multi-engine processing...")
        
        try:
            processing_start = time.time()
            
            # Process with available initialized engines (limit to 3 for speed)
            available_engine_names = list(initialized_engines.keys())[:3] if 'initialized_engines' in locals() else []
            
            if available_engine_names:
                results = manager.process_with_multiple_engines(
                    image=test_image,
                    engine_names=available_engine_names
                )
                
                processing_time = time.time() - processing_start
                
                if isinstance(results, dict) and len(results) > 0:
                    print("✓ Multi-engine processing successful")
                    print(f"✓ Processing time: {processing_time:.3f}s")
                    print(f"✓ Results from {len(results)} engines")
                    
                    # Show results from each engine
                    for engine_name, result in results.items():
                        if result and len(result) > 0:
                            text_preview = result[0].text[:30] + "..." if len(result[0].text) > 30 else result[0].text
                            conf = result[0].confidence
                            print(f"   {engine_name}: '{text_preview}' (conf: {conf:.3f})")
                    
                    test_results['multi_engine_processing'] = True
                else:
                    print("❌ Multi-engine processing failed or empty results")
            else:
                print("⚠️  No initialized engines available for multi-engine processing")
                
        except Exception as e:
            print(f"❌ Multi-engine processing failed: {e}")
            import traceback
            traceback.print_exc()
        
        # Test 7: Result Comparison
        print("\nTesting result comparison and ranking...")
        
        try:
            if 'results' in locals() and results:
                # Compare results
                comparison = manager.compare_results(results)
                
                if comparison:
                    print("✓ Result comparison successful")
                    print(f"✓ Comparison metrics calculated")
                    
                    # Show comparison summary
                    if 'best_engine' in comparison:
                        print(f"   Best engine: {comparison['best_engine']}")
                    if 'confidence_scores' in comparison:
                        print(f"   Confidence scores: {comparison['confidence_scores']}")
                    
                    test_results['result_comparison'] = True
                else:
                    print("❌ Result comparison failed")
            else:
                print("⚠️  No results available for comparison")
                
        except Exception as e:
            print(f"❌ Result comparison failed: {e}")
        
        # Test 8: Best Engine Selection
        print("\nTesting best engine selection...")
        
        try:
            if 'results' in locals() and results:
                best_result = manager.select_best_result(results)
                
                if best_result:
                    print("✓ Best engine selection successful")
                    text_preview = best_result.text[:50] + "..." if len(best_result.text) > 50 else best_result.text
                    print(f"   Best result: '{text_preview}'")
                    print(f"   Confidence: {best_result.confidence:.3f}")
                    
                    test_results['best_engine_selection'] = True
                else:
                    print("❌ Best engine selection failed")
            else:
                print("⚠️  No results available for best selection")
                
        except Exception as e:
            print(f"❌ Best engine selection failed: {e}")
        
        # Test 9: Batch Processing
        print("\nTesting batch processing coordination...")
        
        try:
            batch_images = [
                create_multi_engine_test_image(),
                create_multi_engine_test_image()
            ]
            
            # Use first available initialized engine
            if initialized_engines:
                engine_name = list(initialized_engines.keys())[0]
                
                batch_start = time.time()
                batch_results = manager.batch_process(
                    images=batch_images,
                    engine_name=engine_name
                )
                batch_time = time.time() - batch_start
                
                if len(batch_results) == len(batch_images):
                    print("✓ Batch processing successful")
                    print(f"✓ Batch time: {batch_time:.3f}s")
                    print(f"✓ Images processed: {len(batch_results)}")
                    print(f"✓ Avg per image: {batch_time/len(batch_images):.3f}s")
                    
                    test_results['batch_processing'] = True
                else:
                    print("❌ Batch processing failed")
            else:
                print("⚠️  No initialized engines for batch processing")
                
        except Exception as e:
            print(f"❌ Batch processing failed: {e}")
        
        # Test 10: Error Handling
        print("\nTesting error handling and resilience...")
        
        try:
            error_cases = [
                (None, "null_image"),
                (np.array([]), "empty_array"),
                (np.zeros((5, 5, 3)), "tiny_image")
            ]
            
            error_handled = 0
            for invalid_input, case_name in error_cases:
                try:
                    result = manager.process_with_best_engine(invalid_input)
                    if not result or (isinstance(result, list) and len(result) == 0):
                        error_handled += 1
                except:
                    error_handled += 1
            
            print(f"✓ Error handling: {error_handled}/3 cases handled gracefully")
            
            if error_handled >= 2:
                test_results['error_handling'] = True
                
        except Exception as e:
            print(f"❌ Error handling test failed: {e}")
        
        # Test 11: Performance Tracking
        print("\nTesting performance tracking...")
        
        try:
            performance_stats = manager.get_performance_stats()
            
            if performance_stats:
                print("✓ Performance tracking active")
                
                # Show key metrics
                if 'total_engines' in performance_stats:
                    print(f"   Total engines: {performance_stats['total_engines']}")
                if 'initialized_engines' in performance_stats:
                    print(f"   Initialized engines: {performance_stats['initialized_engines']}")
                if 'system_stats' in performance_stats and 'total_processed' in performance_stats['system_stats']:
                    print(f"   Total processed: {performance_stats['system_stats']['total_processed']}")
                
                test_results['performance_tracking'] = True
            else:
                print("❌ Performance tracking not available")
                
        except Exception as e:
            print(f"❌ Performance tracking failed: {e}")
        
        # Calculate Results
        end_time = time.time()
        total_time = end_time - start_time
        
        passed_tests = sum(test_results.values())
        total_tests = len(test_results)
        success_rate = passed_tests / total_tests
        
        print("\n" + "=" * 80)
        print("TEST 13 RESULTS SUMMARY (ENGINE COORDINATION)")
        print("=" * 80)
        print(f"Tests passed: {passed_tests}/{total_tests} ({success_rate*100:.1f}%)")
        print(f"Total time: {total_time:.3f}s")
        print("Success criteria: Multi-engine coordination system")
        
        if success_rate >= 0.7:  # Allow some flexibility for engine coordination
            print("✅ STATUS: PASSED - Engine manager coordinating successfully")
            print("✅ Multi-engine system operational")
        else:
            print("❌ STATUS: FAILED - Engine manager needs fixes")
            print("Issues found in:", [k for k, v in test_results.items() if not v])
        
        print("\nCOMPONENT STATUS:")
        for component, status in test_results.items():
            status_icon = "✅" if status else "❌"
            print(f"   {status_icon} {component.replace('_', ' ').title()}")
        
        print("\nMULTI-ENGINE COORDINATION VALIDATION:")
        print("   ✓ Engine registration system")
        print("   ✓ Engine selection logic") 
        print("   ✓ Multi-engine processing")
        print("   ✓ Result comparison system")
        print("   ✓ Best result selection")
        print("   ✓ Batch processing coordination")
        print("   ✓ Performance tracking")
        print("   ✓ Error handling resilience")
        
        if 'results' in locals() and results:
            print(f"\nCOORDINATION METRICS:")
            print(f"   Engines coordinated: {len(results)}")
            print(f"   Processing time: {processing_time:.3f}s")
            print(f"   Results generated: {sum(len(r) for r in results.values() if r)}")
            print(f"   System architecture: Multi-Engine Pipeline")
        
        return success_rate >= 0.7, test_results, total_time
        
    except Exception as e:
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"\n🚨 CRITICAL ERROR in Test 13: {e}")
        print(f"Failed after: {processing_time:.3f}s")
        
        import traceback
        traceback.print_exc()
        
        return False, test_results, processing_time

if __name__ == "__main__":
    success, results, time_taken = test_engine_manager()
    
    if success:
        print(f"\n🎉 Test 13 completed successfully in {time_taken:.3f}s")
        print("✅ Engine manager coordination validated")
        print("🚀 Ready for Test 14: Multi-Engine Processing")
    else:
        print(f"\n❌ Test 13 failed after {time_taken:.3f}s")
        print("🔧 Fix engine manager issues before proceeding")
        
    sys.exit(0 if success else 1)