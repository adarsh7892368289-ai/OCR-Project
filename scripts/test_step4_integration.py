
# scripts/test_step4_integration.py - Integration Testing Script

import sys
import os
import cv2
import numpy as np
import time
import json
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from preprocessing.adaptive_processor import (
    AdaptivePreprocessor, ProcessingOptions, ProcessingLevel, PipelineStrategy
)
from preprocessing.quality_analyzer import ImageType

def create_test_images():
    """Create various test images for comprehensive testing"""
    images = {}
    
    # 1. Clean document image
    clean_doc = np.ones((400, 600, 3), dtype=np.uint8) * 255
    cv2.putText(clean_doc, "CLEAN DOCUMENT TEXT", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(clean_doc, "This is a test document", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(clean_doc, "with good quality text", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    images["clean_document"] = clean_doc
    
    # 2. Noisy document
    noisy_doc = clean_doc.copy()
    noise = np.random.normal(0, 30, noisy_doc.shape).astype(np.uint8)
    noisy_doc = cv2.add(noisy_doc, noise)
    images["noisy_document"] = noisy_doc
    
    # 3. Skewed document
    center = (clean_doc.shape[1] // 2, clean_doc.shape[0] // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, 5.0, 1.0)
    skewed_doc = cv2.warpAffine(clean_doc, rotation_matrix, (clean_doc.shape[1], clean_doc.shape[0]))
    images["skewed_document"] = skewed_doc
    
    # 4. Blurry document
    blurry_doc = cv2.GaussianBlur(clean_doc, (15, 15), 0)
    images["blurry_document"] = blurry_doc
    
    # 5. Low contrast document
    low_contrast = clean_doc.copy()
    low_contrast = cv2.addWeighted(low_contrast, 0.5, np.ones_like(low_contrast) * 128, 0.5, 0)
    images["low_contrast_document"] = low_contrast
    
    # 6. Table-like structure
    table_doc = np.ones((400, 600, 3), dtype=np.uint8) * 255
    # Draw table structure
    for i in range(5):
        y = 50 + i * 60
        cv2.line(table_doc, (50, y), (550, y), (0, 0, 0), 2)
    for j in range(4):
        x = 50 + j * 125
        cv2.line(table_doc, (x, 50), (x, 290), (0, 0, 0), 2)
    # Add some text in cells
    cv2.putText(table_doc, "Header 1", (60, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(table_doc, "Data 1", (60, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    images["table_document"] = table_doc
    
    # 7. Handwriting simulation
    handwriting = np.ones((400, 600, 3), dtype=np.uint8) * 255
    # Simulate handwriting with irregular lines
    points = [(100, 100), (120, 95), (140, 105), (160, 100), (180, 110)]
    for i in range(len(points) - 1):
        cv2.line(handwriting, points[i], points[i+1], (0, 0, 0), 3)
    images["handwriting_sample"] = handwriting
    
    return images

def test_basic_functionality(preprocessor, images):
    """Test basic preprocessing functionality"""
    print("Testing basic functionality...")
    results = {}
    
    for name, image in images.items():
        print(f"  Processing {name}...")
        start_time = time.time()
        
        result = preprocessor.process_image(image)
        
        processing_time = time.time() - start_time
        
        results[name] = {
            "success": result.success,
            "processing_time": processing_time,
            "steps": result.processing_steps,
            "warnings": result.warnings,
            "quality_improvement": result.metadata.get("quality_improvement", 0),
            "pipeline_used": result.metadata.get("pipeline_used", "unknown")
        }
        
        print(f"    - Success: {result.success}")
        print(f"    - Time: {processing_time:.2f}s")
        print(f"    - Pipeline: {result.metadata.get('pipeline_used', 'unknown')}")
        print(f"    - Quality improvement: {result.metadata.get('quality_improvement', 0):.3f}")
    
    return results

def test_processing_levels(preprocessor, test_image):
    """Test different processing levels"""
    print("\nTesting different processing levels...")
    results = {}
    
    levels = [ProcessingLevel.MINIMAL, ProcessingLevel.LIGHT, 
             ProcessingLevel.BALANCED, ProcessingLevel.INTENSIVE]
    
    for level in levels:
        print(f"  Testing {level.value} level...")
        options = ProcessingOptions(processing_level=level)
        
        start_time = time.time()
        result = preprocessor.process_image(test_image, options)
        processing_time = time.time() - start_time
        
        results[level.value] = {
            "success": result.success,
            "processing_time": processing_time,
            "steps_count": len(result.processing_steps),
            "quality_improvement": result.metadata.get("quality_improvement", 0)
        }
        
        print(f"    - Time: {processing_time:.2f}s")
        print(f"    - Steps: {len(result.processing_steps)}")
        print(f"    - Quality improvement: {result.metadata.get('quality_improvement', 0):.3f}")
    
    return results

def test_pipeline_strategies(preprocessor, test_image):
    """Test different pipeline strategies"""
    print("\nTesting pipeline strategies...")
    results = {}
    
    strategies = [PipelineStrategy.SPEED_OPTIMIZED, PipelineStrategy.QUALITY_OPTIMIZED,
                 PipelineStrategy.CONTENT_AWARE]
    
    for strategy in strategies:
        print(f"  Testing {strategy.value} strategy...")
        options = ProcessingOptions(strategy=strategy)
        
        start_time = time.time()
        result = preprocessor.process_image(test_image, options)
        processing_time = time.time() - start_time
        
        results[strategy.value] = {
            "success": result.success,
            "processing_time": processing_time,
            "pipeline_used": result.metadata.get("pipeline_used", "unknown"),
            "quality_improvement": result.metadata.get("quality_improvement", 0)
        }
        
        print(f"    - Time: {processing_time:.2f}s")
        print(f"    - Pipeline: {result.metadata.get('pipeline_used', 'unknown')}")
        print(f"    - Quality improvement: {result.metadata.get('quality_improvement', 0):.3f}")
    
    return results

def test_batch_processing(preprocessor, images):
    """Test batch processing capabilities"""
    print("\nTesting batch processing...")
    
    image_list = list(images.values())[:4]  # Use first 4 images
    
    def progress_callback(completed, total):
        print(f"    Progress: {completed}/{total} ({completed/total*100:.1f}%)")
    
    start_time = time.time()
    results = preprocessor.process_batch(image_list, progress_callback=progress_callback)
    total_time = time.time() - start_time
    
    print(f"  Batch processing completed in {total_time:.2f}s")
    print(f"  Average time per image: {total_time/len(image_list):.2f}s")
    
    success_count = sum(1 for r in results if r.success)
    print(f"  Success rate: {success_count}/{len(results)} ({success_count/len(results)*100:.1f}%)")
    
    return {
        "total_time": total_time,
        "average_time_per_image": total_time / len(image_list),
        "success_rate": success_count / len(results),
        "total_images": len(results)
    }

def test_custom_pipeline(preprocessor):
    """Test custom pipeline functionality"""
    print("\nTesting custom pipeline...")
    
    # Define custom pipeline
    custom_pipeline = {
        "name": "test_custom_pipeline",
        "description": "Custom pipeline for testing",
        "steps": [
            {
                "name": "skew_correction",
                "parameters": {"quality": "high_quality"},
                "conditions": {"requires_skew_correction": 0.5}
            },
            {
                "name": "enhancement",
                "parameters": {"strategy": "conservative"},
                "conditions": {}
            },
            {
                "name": "contrast_enhancement",
                "parameters": {"method": "clahe"},
                "conditions": {}
            }
        ]
    }
    
    # Validate pipeline
    errors = preprocessor.validate_pipeline(custom_pipeline)
    print(f"  Pipeline validation errors: {len(errors)}")
    
    if errors:
        for error in errors:
            print(f"    - {error}")
        return {"success": False, "errors": errors}
    
    # Add custom pipeline
    preprocessor.add_custom_pipeline("test_custom", custom_pipeline)
    
    # Check if added
    available_pipelines = preprocessor.get_available_pipelines()
    custom_added = "test_custom" in available_pipelines
    print(f"  Custom pipeline added: {custom_added}")
    
    # Get pipeline info
    pipeline_info = preprocessor.get_pipeline_info("test_custom")
    print(f"  Pipeline info retrieved: {pipeline_info is not None}")
    
    return {
        "success": True,
        "validation_errors": len(errors),
        "pipeline_added": custom_added,
        "info_retrieved": pipeline_info is not None
    }

def test_performance_monitoring(preprocessor, test_image):
    """Test performance monitoring and statistics"""
    print("\nTesting performance monitoring...")
    
    # Get initial statistics
    initial_stats = preprocessor.get_processing_statistics()
    initial_count = initial_stats["total_processed"]
    
    # Process several images
    for i in range(5):
        result = preprocessor.process_image(test_image)
        print(f"  Processed image {i+1}/5")
    
    # Get updated statistics
    updated_stats = preprocessor.get_processing_statistics()
    
    print(f"  Images processed: {updated_stats['total_processed'] - initial_count}")
    print(f"  Success rate: {updated_stats.get('success_rate', 0):.2f}")
    print(f"  Average processing time: {updated_stats['average_processing_time']:.2f}s")
    print(f"  Quality improvement rate: {updated_stats.get('quality_improvement_rate', 0):.2f}")
    
    # Test statistics reset
    preprocessor.reset_statistics()
    reset_stats = preprocessor.get_processing_statistics()
    print(f"  Statistics reset: {reset_stats['total_processed'] == 0}")
    
    return {
        "images_processed": updated_stats['total_processed'] - initial_count,
        "success_rate": updated_stats.get('success_rate', 0),
        "average_processing_time": updated_stats['average_processing_time'],
        "statistics_reset": reset_stats['total_processed'] == 0
    }

def test_error_handling(preprocessor):
    """Test error handling capabilities"""
    print("\nTesting error handling...")
    results = {}
    
    # Test with None image
    print("  Testing None image...")
    result = preprocessor.process_image(None)
    results["none_image"] = {
        "success": result.success,
        "has_warnings": len(result.warnings) > 0
    }
    
    # Test with empty array
    print("  Testing empty array...")
    empty_array = np.array([])
    result = preprocessor.process_image(empty_array)
    results["empty_array"] = {
        "success": result.success,
        "has_warnings": len(result.warnings) > 0
    }
    
    # Test with invalid shape
    print("  Testing invalid shape...")
    invalid_shape = np.random.randint(0, 255, (10,), dtype=np.uint8)
    result = preprocessor.process_image(invalid_shape)
    results["invalid_shape"] = {
        "success": result.success,
        "has_warnings": len(result.warnings) > 0
    }
    
    # Test with very short timeout
    print("  Testing short timeout...")
    test_image = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
    options = ProcessingOptions(processing_timeout=0.001)
    result = preprocessor.process_image(test_image, options)
    results["short_timeout"] = {
        "success": result.success,
        "has_warnings": len(result.warnings) > 0
    }
    
    return results

def save_results(results, output_path):
    """Save test results to JSON file"""
    output_file = Path(output_path) / "step4_integration_results.json"
    
    # Convert numpy types to Python types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    # Clean results for JSON
    clean_results = json.loads(json.dumps(results, default=convert_numpy))
    
    with open(output_file, 'w') as f:
        json.dump(clean_results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")

def main():
    """Main integration test function"""
    print("Step 4 Integration Test - Adaptive Preprocessing Pipeline")
    print("=" * 60)
    
    # Create output directory
    output_path = Path("./test_results")
    output_path.mkdir(exist_ok=True)
    
    # Initialize preprocessor
    print("Initializing preprocessor...")
    try:
        preprocessor = AdaptivePreprocessor()
        print("  Preprocessor initialized successfully")
    except Exception as e:
        print(f"  Failed to initialize preprocessor: {e}")
        return
    
    # Create test images
    print("\nCreating test images...")
    images = create_test_images()
    print(f"  Created {len(images)} test images")
    
    # Run tests
    test_results = {}
    
    try:
        # Test basic functionality
        test_results["basic_functionality"] = test_basic_functionality(preprocessor, images)
        
        # Test processing levels
        test_results["processing_levels"] = test_processing_levels(preprocessor, images["noisy_document"])
        
        # Test pipeline strategies
        test_results["pipeline_strategies"] = test_pipeline_strategies(preprocessor, images["skewed_document"])
        
        # Test batch processing
        test_results["batch_processing"] = test_batch_processing(preprocessor, images)
        
        # Test custom pipeline
        test_results["custom_pipeline"] = test_custom_pipeline(preprocessor)
        
        # Test performance monitoring
        test_results["performance_monitoring"] = test_performance_monitoring(preprocessor, images["clean_document"])
        
        # Test error handling
        test_results["error_handling"] = test_error_handling(preprocessor)
        
        # Save results
        save_results(test_results, output_path)
        
        # Print summary
        print("\n" + "=" * 60)
        print("INTEGRATION TEST SUMMARY")
        print("=" * 60)
        
        total_tests = 0
        passed_tests = 0
        
        for category, results in test_results.items():
            print(f"\n{category.replace('_', ' ').title()}:")
            
            if isinstance(results, dict):
                for test_name, test_result in results.items():
                    total_tests += 1
                    if isinstance(test_result, dict) and test_result.get("success", True):
                        passed_tests += 1
                        status = "PASS"
                    else:
                        status = "FAIL"
                    print(f"  {test_name}: {status}")
        
        print(f"\nOverall Results:")
        print(f"  Total Tests: {total_tests}")
        print(f"  Passed: {passed_tests}")
        print(f"  Failed: {total_tests - passed_tests}")
        print(f"  Success Rate: {passed_tests/total_tests*100:.1f}%")
        
    except Exception as e:
        print(f"\nIntegration test failed with error: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup
        print("\nCleaning up...")
        preprocessor.shutdown()
        print("Integration test completed.")

if __name__ == "__main__":
    main()