#!/usr/bin/env python3
# scripts/complete_step4_test.py - Complete Step 4 Testing Script

"""
Complete Step 4 Testing Script
==============================

This script provides a comprehensive testing suite for the Step 4 adaptive preprocessing pipeline.
It handles dependency issues, runs various test scenarios, and provides detailed reporting.

Usage:
    python scripts/complete_step4_test.py [options]

Options:
    --create-deps    Create minimal dependency stubs if missing
    --mock-test      Use mock dependencies for testing
    --benchmark      Run performance benchmarks
    --integration    Run integration tests
    --all            Run all tests
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
import traceback

# Ensure we can import modules
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

def setup_environment():
    """Setup testing environment"""
    print("Setting up testing environment...")
    
    # Create necessary directories
    directories = [
        project_root / "test_output",
        project_root / "logs",
        project_root / "debug",
        project_root / "benchmark"
    ]
    
    for dir_path in directories:
        dir_path.mkdir(exist_ok=True)
    
    print("Environment setup complete")

def check_dependencies():
    """Check if all required dependencies are available"""
    print("Checking dependencies...")
    
    required_modules = [
        "numpy",
        "opencv-python",
        "Pillow"
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            if module == "opencv-python":
                import cv2
            elif module == "Pillow":
                import PIL
            else:
                __import__(module)
            print(f"✓ {module}")
        except ImportError:
            missing_modules.append(module)
            print(f"✗ {module}")
    
    if missing_modules:
        print(f"\nMissing modules: {', '.join(missing_modules)}")
        print("Install with: pip install " + " ".join(missing_modules))
        return False
    
    return True

def create_test_images():
    """Create test images for various scenarios"""
    import numpy as np
    import cv2
    
    print("Creating test images...")
    
    images = {}
    
    # 1. Clean document
    clean = np.ones((400, 600, 3), dtype=np.uint8) * 255
    cv2.putText(clean, "SAMPLE DOCUMENT", (100, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    cv2.putText(clean, "This is clean text for testing", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(clean, "OCR preprocessing pipeline", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    images["clean_document"] = clean
    
    # 2. Noisy document
    noisy = clean.copy()
    noise = np.random.normal(0, 30, noisy.shape).astype(np.int16)
    noisy = np.clip(noisy.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    images["noisy_document"] = noisy
    
    # 3. Blurry document
    blurry = cv2.GaussianBlur(clean, (9, 9), 0)
    images["blurry_document"] = blurry
    
    # 4. Skewed document
    center = (clean.shape[1] // 2, clean.shape[0] // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, 3.0, 1.0)
    skewed = cv2.warpAffine(clean, rotation_matrix, (clean.shape[1], clean.shape[0]))
    images["skewed_document"] = skewed
    
    # 5. Low contrast
    low_contrast = cv2.addWeighted(clean, 0.6, np.ones_like(clean) * 128, 0.4, 0)
    images["low_contrast_document"] = low_contrast
    
    # 6. Small image
    small = cv2.resize(clean, (200, 150))
    images["small_document"] = small
    
    # 7. Large image
    large = cv2.resize(clean, (1200, 800))
    images["large_document"] = large
    
    print(f"Created {len(images)} test images")
    return images

def run_basic_tests(preprocessor, test_images):
    """Run basic functionality tests"""
    print("\n" + "="*50)
    print("BASIC FUNCTIONALITY TESTS")
    print("="*50)
    
    results = {
        "total_tests": 0,
        "passed_tests": 0,
        "failed_tests": 0,
        "test_results": {}
    }
    
    # Test each image type
    for image_name, image in test_images.items():
        print(f"\nTesting with {image_name}...")
        results["total_tests"] += 1
        
        try:
            start_time = time.time()
            result = preprocessor.process_image(image)
            processing_time = time.time() - start_time
            
            # Validate result
            test_passed = (
                result is not None and
                hasattr(result, 'success') and
                hasattr(result, 'processed_image') and
                result.processed_image is not None
            )
            
            if test_passed:
                results["passed_tests"] += 1
                status = "PASS"
                print(f"  ✓ {status} - {processing_time:.2f}s")
            else:
                results["failed_tests"] += 1
                status = "FAIL"
                print(f"  ✗ {status} - Invalid result")
            
            results["test_results"][image_name] = {
                "status": status,
                "processing_time": processing_time,
                "success": getattr(result, 'success', False) if result else False,
                "steps_count": len(getattr(result, 'processing_steps', [])) if result else 0
            }
            
        except Exception as e:
            results["failed_tests"] += 1
            results["test_results"][image_name] = {
                "status": "ERROR",
                "error": str(e),
                "processing_time": 0
            }
            print(f"  ✗ ERROR - {str(e)}")
    
    return results

def run_processing_level_tests(preprocessor, test_image):
    """Test different processing levels"""
    print("\n" + "="*50)
    print("PROCESSING LEVEL TESTS")
    print("="*50)
    
    try:
        from preprocessing.adaptive_processor import ProcessingOptions, ProcessingLevel
        
        levels = [
            ProcessingLevel.MINIMAL,
            ProcessingLevel.LIGHT,
            ProcessingLevel.BALANCED,
            ProcessingLevel.INTENSIVE
        ]
        
        results = {}
        
        for level in levels:
            print(f"\nTesting {level.value} level...")
            try:
                options = ProcessingOptions(processing_level=level)
                start_time = time.time()
                result = preprocessor.process_image(test_image, options)
                processing_time = time.time() - start_time
                
                results[level.value] = {
                    "success": getattr(result, 'success', False) if result else False,
                    "processing_time": processing_time,
                    "steps_count": len(getattr(result, 'processing_steps', [])) if result else 0
                }
                
                print(f"  ✓ {processing_time:.2f}s - {results[level.value]['steps_count']} steps")
                
            except Exception as e:
                results[level.value] = {"error": str(e)}
                print(f"  ✗ ERROR - {str(e)}")
        
        return results
        
    except ImportError as e:
        print(f"Could not import processing levels: {e}")
        return {"error": "Import failed"}

def run_pipeline_tests(preprocessor, test_image):
    """Test different pipeline strategies"""
    print("\n" + "="*50)
    print("PIPELINE STRATEGY TESTS")  
    print("="*50)
    
    try:
        from preprocessing.adaptive_processor import ProcessingOptions, PipelineStrategy
        
        strategies = [
            PipelineStrategy.SPEED_OPTIMIZED,
            PipelineStrategy.QUALITY_OPTIMIZED,
            PipelineStrategy.CONTENT_AWARE
        ]
        
        results = {}
        
        for strategy in strategies:
            print(f"\nTesting {strategy.value} strategy...")
            try:
                options = ProcessingOptions(strategy=strategy)
                start_time = time.time()
                result = preprocessor.process_image(test_image, options)
                processing_time = time.time() - start_time
                
                pipeline_used = "unknown"
                if result and hasattr(result, 'metadata'):
                    pipeline_used = result.metadata.get('pipeline_used', 'unknown')
                
                results[strategy.value] = {
                    "success": getattr(result, 'success', False) if result else False,
                    "processing_time": processing_time,
                    "pipeline_used": pipeline_used
                }
                
                print(f"  ✓ {processing_time:.2f}s - Pipeline: {pipeline_used}")
                
            except Exception as e:
                results[strategy.value] = {"error": str(e)}
                print(f"  ✗ ERROR - {str(e)}")
        
        return results
        
    except ImportError as e:
        print(f"Could not import pipeline strategies: {e}")
        return {"error": "Import failed"}

def run_batch_tests(preprocessor, test_images):
    """Test batch processing"""
    print("\n" + "="*50)
    print("BATCH PROCESSING TESTS")
    print("="*50)
    
    try:
        # Use first 4 images for batch test
        image_list = list(test_images.values())[:4]
        
        print(f"Processing batch of {len(image_list)} images...")
        
        def progress_callback(completed, total):
            print(f"  Progress: {completed}/{total}")
        
        start_time = time.time()
        results = preprocessor.process_batch(image_list, progress_callback=progress_callback)
        total_time = time.time() - start_time
        
        success_count = sum(1 for r in results if getattr(r, 'success', False))
        
        batch_results = {
            "total_images": len(image_list),
            "successful_images": success_count,
            "total_time": total_time,
            "average_time_per_image": total_time / len(image_list),
            "success_rate": success_count / len(results) if results else 0
        }
        
        print(f"\nBatch Results:")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Average per image: {batch_results['average_time_per_image']:.2f}s")
        print(f"  Success rate: {batch_results['success_rate']:.1%}")
        
        return batch_results
        
    except Exception as e:
        print(f"Batch processing failed: {e}")
        return {"error": str(e)}

def run_error_handling_tests(preprocessor):
    """Test error handling with invalid inputs"""
    print("\n" + "="*50)
    print("ERROR HANDLING TESTS")
    print("="*50)
    
    import numpy as np
    
    test_cases = [
        ("None input", None),
        ("Empty array", np.array([])),
        ("1D array", np.array([1, 2, 3])),
        ("Wrong shape", np.random.rand(10, 10, 10, 3)),
        ("Zero dimensions", np.zeros((0, 0, 3))),
    ]
    
    results = {}
    
    for test_name, test_input in test_cases:
        print(f"\nTesting {test_name}...")
        
        try:
            result = preprocessor.process_image(test_input)
            
            # Should handle gracefully
            handled_gracefully = (
                result is not None and
                hasattr(result, 'success') and
                not getattr(result, 'success', True)  # Should fail gracefully
            )
            
            if handled_gracefully:
                print(f"  ✓ Handled gracefully")
                results[test_name] = {"status": "PASS", "handled_gracefully": True}
            else:
                print(f"  ? Unexpected result")
                results[test_name] = {"status": "UNCERTAIN", "result_type": type(result).__name__}
                
        except Exception as e:
            print(f"  ✗ Exception raised: {str(e)}")
            results[test_name] = {"status": "EXCEPTION", "error": str(e)}
    
    return results

def run_performance_benchmark(preprocessor, test_images):
    """Run performance benchmarks"""
    print("\n" + "="*50)
    print("PERFORMANCE BENCHMARK")
    print("="*50)
    
    # Test with different image sizes
    size_tests = [
        ("Small", test_images.get("small_document")),
        ("Medium", test_images.get("clean_document")),
        ("Large", test_images.get("large_document"))
    ]
    
    benchmark_results = {}
    
    for size_name, test_image in size_tests:
        if test_image is None:
            continue
            
        print(f"\nBenchmarking {size_name} image {test_image.shape}...")
        
        # Run multiple times for average
        times = []
        successes = 0
        
        for i in range(3):  # 3 runs for average
            try:
                start_time = time.time()
                result = preprocessor.process_image(test_image)
                end_time = time.time()
                
                processing_time = end_time - start_time
                times.append(processing_time)
                
                if getattr(result, 'success', False):
                    successes += 1
                    
            except Exception as e:
                print(f"    Run {i+1} failed: {str(e)}")
        
        if times:
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            
            benchmark_results[size_name.lower()] = {
                "average_time": avg_time,
                "min_time": min_time,
                "max_time": max_time,
                "success_rate": successes / len(times),
                "image_size": test_image.shape
            }
            
            print(f"  Average: {avg_time:.2f}s")
            print(f"  Range: {min_time:.2f}s - {max_time:.2f}s")
            print(f"  Success rate: {successes}/{len(times)}")
    
    return benchmark_results

def run_mock_tests():
    """Run tests with mock dependencies"""
    print("\n" + "="*50)
    print("MOCK DEPENDENCY TESTS")
    print("="*50)
    
    try:
        # This would use the mock testing runner
        print("Mock tests would run here...")
        # Import and run the mock test runner
        return {"status": "MOCK_TESTS_PLACEHOLDER"}
        
    except Exception as e:
        print(f"Mock tests failed: {e}")
        return {"error": str(e)}

def generate_report(all_results, output_dir):
    """Generate comprehensive test report"""
    print("\n" + "="*50)
    print("GENERATING REPORT")
    print("="*50)
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Create summary
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "test_results": all_results,
        "overall_status": "UNKNOWN"
    }
    
    # Calculate overall status
    total_tests = 0
    passed_tests = 0
    
    for category, results in all_results.items():
        if isinstance(results, dict):
            if "total_tests" in results and "passed_tests" in results:
                total_tests += results["total_tests"]
                passed_tests += results["passed_tests"]
    
    if total_tests > 0:
        success_rate = passed_tests / total_tests
        if success_rate >= 0.9:
            summary["overall_status"] = "EXCELLENT"
        elif success_rate >= 0.7:
            summary["overall_status"] = "GOOD" 
        elif success_rate >= 0.5:
            summary["overall_status"] = "FAIR"
        else:
            summary["overall_status"] = "POOR"
    else:
        summary["overall_status"] = "NO_TESTS"
    
    summary["success_rate"] = passed_tests / total_tests if total_tests > 0 else 0
    summary["total_tests"] = total_tests
    summary["passed_tests"] = passed_tests
    
    # Save detailed results
    results_file = output_path / "test_results.json"
    with open(results_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    # Create readable report
    report_file = output_path / "test_report.txt"
    with open(report_file, 'w') as f:
        f.write("Step 4 Adaptive Preprocessing Pipeline - Test Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Test Date: {summary['timestamp']}\n")
        f.write(f"Overall Status: {summary['overall_status']}\n")
        f.write(f"Success Rate: {summary['success_rate']:.1%}\n")
        f.write(f"Tests Passed: {summary['passed_tests']}/{summary['total_tests']}\n\n")
        
        # Write detailed results
        for category, results in all_results.items():
            f.write(f"{category.upper().replace('_', ' ')}\n")
            f.write("-" * 40 + "\n")
            if isinstance(results, dict):
                for key, value in results.items():
                    f.write(f"  {key}: {value}\n")
            else:
                f.write(f"  Result: {results}\n")
            f.write("\n")
    
    print(f"Report saved to: {output_path}")
    print(f"Overall Status: {summary['overall_status']}")
    print(f"Success Rate: {summary['success_rate']:.1%}")
    
    return summary

def main():
    """Main test runner"""
    parser = argparse.ArgumentParser(description='Complete Step 4 Testing Script')
    parser.add_argument('--create-deps', action='store_true', help='Create minimal dependency stubs')
    parser.add_argument('--mock-test', action='store_true', help='Use mock dependencies')
    parser.add_argument('--benchmark', action='store_true', help='Run performance benchmarks')
    parser.add_argument('--integration', action='store_true', help='Run integration tests')
    parser.add_argument('--all', action='store_true', help='Run all tests')
    
    args = parser.parse_args()
    
    if not any(vars(args).values()):
        args.all = True  # Default to all tests
    
    print("Complete Step 4 Testing Script")
    print("=" * 60)
    print(f"Project root: {project_root}")
    
    # Setup environment
    setup_environment()
    
    # Check dependencies
    if not check_dependencies():
        print("\nMissing required Python packages!")
        return 1
    
    all_results = {}
    
    try:
        # Handle dependency creation
        if args.create_deps or args.all:
            print("\nChecking and creating dependencies...")
            try:
                # Run dependency checker
                exec(open(project_root / "scripts" / "check_step4_dependencies.py").read())
            except FileNotFoundError:
                print("Dependency checker not found, proceeding with available components...")
        
        # Try to import and initialize preprocessor
        preprocessor = None
        try:
            from preprocessing.adaptive_processor import AdaptivePreprocessor
            preprocessor = AdaptivePreprocessor()
            print("✓ Successfully initialized AdaptivePreprocessor")
        except ImportError as e:
            print(f"✗ Could not import AdaptivePreprocessor: {e}")
            if args.mock_test or args.all:
                print("Falling back to mock tests...")
                all_results["mock_tests"] = run_mock_tests()
            else:
                print("Run with --mock-test to use mock dependencies")
                return 1
        except Exception as e:
            print(f"✗ Error initializing preprocessor: {e}")
            traceback.print_exc()
            return 1
        
        if preprocessor:
            # Create test images
            test_images = create_test_images()
            
            # Run basic tests
            if args.all or not any([args.benchmark, args.integration]):
                all_results["basic_tests"] = run_basic_tests(preprocessor, test_images)
                all_results["processing_levels"] = run_processing_level_tests(
                    preprocessor, test_images["clean_document"]
                )
                all_results["pipeline_strategies"] = run_pipeline_tests(
                    preprocessor, test_images["noisy_document"]
                )
                all_results["batch_processing"] = run_batch_tests(preprocessor, test_images)
                all_results["error_handling"] = run_error_handling_tests(preprocessor)
            
            # Run benchmarks
            if args.benchmark or args.all:
                all_results["performance_benchmark"] = run_performance_benchmark(
                    preprocessor, test_images
                )
            
            # Run integration tests
            if args.integration or args.all:
                try:
                    from core.enhanced_engine_manager import EnhancedEngineManager
                    print("✓ Enhanced Engine Manager available for integration tests")
                    # Integration tests would go here
                    all_results["integration_tests"] = {"status": "AVAILABLE"}
                except ImportError:
                    print("✗ Enhanced Engine Manager not available")
                    all_results["integration_tests"] = {"status": "UNAVAILABLE", "reason": "Import failed"}
            
            # Cleanup
            preprocessor.shutdown()
        
        # Generate report
        if all_results:
            report_summary = generate_report(all_results, project_root / "test_output")
            
            # Print final summary
            print("\n" + "=" * 60)
            print("FINAL SUMMARY")
            print("=" * 60)
            print(f"Overall Status: {report_summary['overall_status']}")
            print(f"Success Rate: {report_summary['success_rate']:.1%}")
            print(f"Tests Passed: {report_summary['passed_tests']}/{report_summary['total_tests']}")
            
            if report_summary['overall_status'] in ['EXCELLENT', 'GOOD']:
                print("\n✓ Step 4 is ready for production use!")
                print("Next steps:")
                print("  1. Integrate with actual OCR engines")
                print("  2. Test with real-world documents")
                print("  3. Move to Step 5: Advanced Post-processing")
            elif report_summary['overall_status'] == 'FAIR':
                print("\n⚠ Step 4 has some issues but basic functionality works")
                print("Recommended actions:")
                print("  1. Review failed tests in the detailed report")
                print("  2. Fix critical issues before proceeding")
                print("  3. Consider using mock testing for development")
            else:
                print("\n✗ Step 4 has significant issues")
                print("Required actions:")
                print("  1. Review error logs and fix critical issues")
                print("  2. Ensure all dependencies are properly installed")
                print("  3. Use dependency checker to create missing components")
            
            return 0 if report_summary['success_rate'] > 0.5 else 1
        else:
            print("\nNo tests were executed!")
            return 1
            
    except KeyboardInterrupt:
        print("\n\nTest execution interrupted by user")
        return 1
    except Exception as e:
        print(f"\nTest execution failed with error: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())