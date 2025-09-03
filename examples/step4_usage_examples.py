# examples/step4_usage_examples.py - Usage Examples for Step 4

"""
Step 4: Adaptive Preprocessing Pipeline - Usage Examples

This file demonstrates various ways to use the adaptive preprocessing pipeline
for different OCR scenarios and requirements.
"""

import sys
import os
import cv2
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from preprocessing.adaptive_processor import (
    AdaptivePreprocessor, ProcessingOptions, ProcessingLevel, 
    PipelineStrategy, quick_process_image, batch_process_images
)

def example_basic_usage():
    """Example 1: Basic preprocessing with default settings"""
    print("Example 1: Basic Usage")
    print("-" * 40)
    
    # Load an image (replace with your image path)
    image_path = "data/sample_images/document.jpg"
    if os.path.exists(image_path):
        image = cv2.imread(image_path)
    else:
        # Create synthetic image for demo
        image = np.ones((400, 600, 3), dtype=np.uint8) * 255
        cv2.putText(image, "Sample Document Text", (50, 200), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
    
    # Initialize preprocessor
    preprocessor = AdaptivePreprocessor()
    
    # Process image with default settings
    result = preprocessor.process_image(image)
    
    # Display results
    print(f"Processing successful: {result.success}")
    print(f"Processing time: {result.processing_time:.2f}s")
    print(f"Steps executed: {', '.join(result.processing_steps)}")
    print(f"Quality improvement: {result.metadata.get('quality_improvement', 0):.3f}")
    
    # Save processed image
    output_path = "output/processed_basic.jpg"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, result.processed_image)
    print(f"Processed image saved to: {output_path}")
    
    preprocessor.shutdown()

def example_speed_optimized():
    """Example 2: Speed-optimized processing for real-time applications"""
    print("\nExample 2: Speed-Optimized Processing")
    print("-" * 40)
    
    # Create test image
    image = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
    
    # Configure for speed
    options = ProcessingOptions(
        processing_level=ProcessingLevel.LIGHT,
        strategy=PipelineStrategy.SPEED_OPTIMIZED,
        enable_quality_validation=False,
        max_processing_iterations=1
    )
    
    preprocessor = AdaptivePreprocessor()
    
    # Measure processing time
    import time
    start_time = time.time()
    result = preprocessor.process_image(image, options)
    processing_time = time.time() - start_time
    
    print(f"Speed-optimized processing time: {processing_time:.3f}s")
    print(f"Pipeline used: {result.metadata.get('pipeline_used')}")
    
    preprocessor.shutdown()

def example_quality_optimized():
    """Example 3: Quality-optimized processing for archival scanning"""
    print("\nExample 3: Quality-Optimized Processing")
    print("-" * 40)
    
    # Create a low-quality test image
    image = np.ones((400, 600, 3), dtype=np.uint8) * 255
    cv2.putText(image, "Low Quality Document", (50, 200), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    # Add noise and blur
    noise = np.random.normal(0, 25, image.shape).astype(np.uint8)
    image = cv2.add(image, noise)
    image = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Configure for maximum quality
    options = ProcessingOptions(
        processing_level=ProcessingLevel.INTENSIVE,
        strategy=PipelineStrategy.QUALITY_OPTIMIZED,
        enable_quality_validation=True,
        max_processing_iterations=3,
        quality_improvement_threshold=0.01
    )
    
    preprocessor = AdaptivePreprocessor()
    result = preprocessor.process_image(image, options)
    
    print(f"Quality-optimized processing completed")
    print(f"Processing time: {result.processing_time:.2f}s")
    print(f"Quality improvement: {result.metadata.get('quality_improvement', 0):.3f}")
    print(f"Steps executed: {len(result.processing_steps)}")
    
    preprocessor.shutdown()

def example_custom_pipeline():
    """Example 4: Creating and using custom pipelines"""
    print("\nExample 4: Custom Pipeline")
    print("-" * 40)
    
    preprocessor = AdaptivePreprocessor()
    
    # Define custom pipeline for handwritten documents
    handwriting_pipeline = {
        "name": "handwriting_special",
        "description": "Special pipeline for handwritten documents",
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
                "name": "noise_reduction",
                "parameters": {"method": "bilateral"},
                "conditions": {"requires_noise_reduction": 0.1}
            }
        ]
    }
    
    # Validate and add pipeline
    errors = preprocessor.validate_pipeline(handwriting_pipeline)
    if errors:
        print(f"Pipeline validation errors: {errors}")
        return
    
    preprocessor.add_custom_pipeline("handwriting_special", handwriting_pipeline)
    print("Custom pipeline added successfully")
    
    # List available pipelines
    available = preprocessor.get_available_pipelines()
    print(f"Available pipelines: {', '.join(available)}")
    
    # Use custom pipeline (would need to implement custom strategy selection)
    test_image = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
    result = preprocessor.process_image(test_image)
    
    print(f"Pipeline used: {result.metadata.get('pipeline_used')}")
    
    preprocessor.shutdown()

def example_batch_processing():
    """Example 5: Batch processing multiple images"""
    print("\nExample 5: Batch Processing")
    print("-" * 40)
    
    # Create multiple test images
    images = []
    for i in range(5):
        img = np.ones((300, 400, 3), dtype=np.uint8) * 255
        cv2.putText(img, f"Document {i+1}", (50, 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        images.append(img)
    
    print(f"Processing {len(images)} images...")
    
    # Progress callback
    def show_progress(completed, total):
        print(f"Progress: {completed}/{total} ({completed/total*100:.1f}%)")
    
    preprocessor = AdaptivePreprocessor()
    
    # Process batch
    import time
    start_time = time.time()
    results = preprocessor.process_batch(images, progress_callback=show_progress)
    total_time = time.time() - start_time
    
    # Show results
    success_count = sum(1 for r in results if r.success)
    print(f"\nBatch processing completed:")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Average per image: {total_time/len(images):.2f}s")
    print(f"  Success rate: {success_count}/{len(results)}")
    
    preprocessor.shutdown()

def example_utility_functions():
    """Example 6: Using utility functions for quick processing"""
    print("\nExample 6: Utility Functions")
    print("-" * 40)
    
    # Create test images
    test_image = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
    test_images = [test_image.copy() for _ in range(3)]
    
    # Quick single image processing
    print("Quick single image processing...")
    processed_single = quick_process_image(test_image, ProcessingLevel.BALANCED)
    print(f"Processed image shape: {processed_single.shape}")
    
    # Quick batch processing
    print("Quick batch processing...")
    processed_batch = batch_process_images(test_images, ProcessingLevel.LIGHT)
    print(f"Processed {len(processed_batch)} images")

def example_configuration_management():
    """Example 7: Configuration management"""
    print("\nExample 7: Configuration Management")
    print("-" * 40)
    
    preprocessor = AdaptivePreprocessor()
    
    # Export current configuration
    config_path = "config_export.yaml"
    preprocessor.export_config(config_path)
    print(f"Configuration exported to: {config_path}")
    
    # Show statistics
    stats = preprocessor.get_processing_statistics()
    print(f"Current statistics: {stats}")
    
    # Configure individual components
    enhancer_config = {
        "enhancement_level": "aggressive",
        "enable_ai_guidance": True
    }
    preprocessor.configure_component("image_enhancer", enhancer_config)
    print("Image enhancer reconfigured")
    
    # Load configuration from file (if exists)
    if os.path.exists(config_path):
        new_preprocessor = AdaptivePreprocessor()
        new_preprocessor.load_config_from_file(config_path)
        print("Configuration loaded from file")
        new_preprocessor.shutdown()
    
    preprocessor.shutdown()
    
    # Clean up
    if os.path.exists(config_path):
        os.remove(config_path)

def example_integration_with_ocr_engines():
    """Example 8: Integration with OCR engines"""
    print("\nExample 8: OCR Engine Integration")
    print("-" * 40)
    
    # This example shows how to integrate preprocessing with OCR engines
    
    def preprocess_for_engine(image, engine_name):
        """Preprocess image optimized for specific OCR engine"""
        
        preprocessor = AdaptivePreprocessor()
        
        # Configure based on engine requirements
        if engine_name == "tesseract":
            options = ProcessingOptions(
                processing_level=ProcessingLevel.BALANCED,
                strategy=PipelineStrategy.CONTENT_AWARE
            )
        elif engine_name == "easyocr":
            options = ProcessingOptions(
                processing_level=ProcessingLevel.LIGHT,
                strategy=PipelineStrategy.SPEED_OPTIMIZED
            )
        elif engine_name == "trocr":
            options = ProcessingOptions(
                processing_level=ProcessingLevel.INTENSIVE,
                strategy=PipelineStrategy.QUALITY_OPTIMIZED
            )
        else:
            options = ProcessingOptions()
        
        result = preprocessor.process_image(image, options)
        preprocessor.shutdown()
        
        return result.processed_image, result.metadata
    
    # Test image
    test_image = np.ones((300, 400, 3), dtype=np.uint8) * 255
    cv2.putText(test_image, "OCR Integration Test", (50, 150), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    # Preprocess for different engines
    engines = ["tesseract", "easyocr", "trocr"]
    
    for engine in engines:
        processed_img, metadata = preprocess_for_engine(test_image, engine)
        print(f"{engine.upper()}: Pipeline used - {metadata.get('pipeline_used')}")

def main():
    """Run all examples"""
    print("Step 4: Adaptive Preprocessing Pipeline - Usage Examples")
    print("=" * 60)
    
    # Create output directory
    os.makedirs("output", exist_ok=True)
    
    # Run examples
    try:
        example_basic_usage()
        example_speed_optimized()
        example_quality_optimized()
        example_custom_pipeline()
        example_batch_processing()
        example_utility_functions()
        example_configuration_management()
        example_integration_with_ocr_engines()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()