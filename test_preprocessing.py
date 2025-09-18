"""
Test script for Advanced OCR Preprocessing Pipeline
Tests the complete preprocessing pipeline with sample images.
"""

import sys
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from typing import Optional

# Add the src directory to the path to import our modules
current_dir = Path(__file__).parent
src_dir = current_dir / 'src'
sys.path.insert(0, str(src_dir))

try:
    from advanced_ocr.config import OCRConfig
    from advanced_ocr.preprocessing.image_processor import ImageProcessor, PreprocessingResult
    from advanced_ocr.preprocessing.quality_analyzer import QualityAnalyzer, QualityMetrics
    from advanced_ocr.utils.image_utils import ImageLoader
    from advanced_ocr.results import ProcessingMetrics
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure the advanced_ocr package is properly structured in the src/ directory")
    sys.exit(1)


def setup_logging():
    """Setup logging for the test."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('preprocessing_test.log')
        ]
    )
    return logging.getLogger(__name__)


def load_test_image(image_path: str) -> Optional[np.ndarray]:
    """
    Load test image with error handling.
    
    Args:
        image_path: Path to the test image
        
    Returns:
        Loaded image as numpy array or None if failed
    """
    try:
        if not os.path.exists(image_path):
            print(f"Image file not found: {image_path}")
            return None
        
        # Use OpenCV to load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            return None
        
        print(f"Successfully loaded image: {image.shape}")
        return image
        
    except Exception as e:
        print(f"Error loading image: {e}")
        return None


def create_test_config() -> OCRConfig:
    """Create CONSERVATIVE test configuration for preprocessing."""
    config = OCRConfig()
    
    # CONSERVATIVE preprocessing parameters - OCR-friendly
    config.max_image_dimension = 4096      # Higher for OCR (was 2048)
    config.min_image_dimension = 800       # Higher minimum (was 300)
    config.target_dpi = 300
    config.enable_adaptive_enhancement = True
    config.enhancement_strength = 0.3      # Much lower (was 0.7)
    config.preserve_aspect_ratio = True
    
    # CONSERVATIVE quality thresholds - less sensitive
    config.quality_thresholds = {
        'blur_threshold': 50.0,            # Lower (was 100.0)
        'noise_threshold': 0.25,           # Higher tolerance (was 0.15)
        'contrast_threshold': 0.2,         # Lower requirement (was 0.3)
        'resolution_threshold': 100,       # Lower requirement (was 150)
        'skew_threshold': 3.0,             # Higher tolerance (was 2.0)
        'lighting_variance': 0.1           # Much lower sensitivity (was 0.2)
    }
    
    return config


def display_quality_metrics(metrics: QualityMetrics, title: str = "Quality Metrics"):
    """Display quality metrics in a readable format."""
    print(f"\n{'='*50}")
    print(f"{title}")
    print(f"{'='*50}")
    print(f"Overall Score:     {metrics.overall_score:.3f}")
    print(f"OCR Readiness:     {metrics.ocr_readiness:.3f}")
    print(f"Blur Score:        {metrics.blur_score:.3f}")
    print(f"Noise Score:       {metrics.noise_score:.3f}")
    print(f"Contrast Score:    {metrics.contrast_score:.3f}")
    print(f"Resolution Score:  {metrics.resolution_score:.3f}")
    print(f"Lighting Score:    {metrics.lighting_score:.3f}")
    print(f"Skew Angle:        {metrics.skew_angle:.2f}°")
    print(f"Text Orientation:  {metrics.text_orientation:.1f}°")
    
    if metrics.issues:
        print(f"\nDetected Issues:")
        for issue in metrics.issues:
            print(f"  - {issue.value}")
    
    if metrics.recommendations:
        print(f"\nRecommendations:")
        for rec in metrics.recommendations:
            print(f"  - {rec}")
    
    if metrics.enhancement_params:
        print(f"\nEnhancement Parameters:")
        for param, value in metrics.enhancement_params.items():
            print(f"  - {param}: {value}")


def display_processing_metrics(metrics: ProcessingMetrics):
    """Display processing performance metrics."""
    print(f"\n{'='*50}")
    print(f"Processing Performance")
    print(f"{'='*50}")
    print(f"Total Time:           {metrics.duration:.3f}s")
    print(f"Preprocessing Time:   {metrics.duration:.3f}s")
    print(f"Quality Improvement:  {metrics.metadata.get('quality_improvement', 0):.3f}")
    print(f"Transformations:      {metrics.metadata.get('transformations_count', 0)}")
    
    # Check for minimal processing flag
    if metrics.metadata.get('minimal_processing', False):
        print(f"Processing Mode:      Minimal (image already good quality)")
    else:
        print(f"Processing Mode:      Full enhancement")
    
    if hasattr(metrics, 'error_message') and metrics.error_message:
        print(f"Error Message:        {metrics.error_message}")


def display_transformations(transformations: dict):
    """Display applied transformations."""
    print(f"\n{'='*50}")
    print(f"Applied Transformations")
    print(f"{'='*50}")
    
    if not transformations:
        print("No transformations applied")
        return
    
    for key, value in transformations.items():
        print(f"{key}: {value}")


def create_comparison_plot(original: np.ndarray, processed: np.ndarray, 
                          original_metrics: QualityMetrics, final_metrics: QualityMetrics,
                          save_path: str = None):
    """
    Create a comparison plot showing original vs processed image with metrics.
    
    Args:
        original: Original image
        processed: Processed image
        original_metrics: Quality metrics of original image
        final_metrics: Quality metrics of processed image
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(16, 12))
    
    # Convert BGR to RGB for matplotlib display
    if len(original.shape) == 3:
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        processed_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
    else:
        original_rgb = original
        processed_rgb = processed
    
    # Original image
    plt.subplot(2, 2, 1)
    plt.imshow(original_rgb, cmap='gray' if len(original_rgb.shape) == 2 else None)
    plt.title(f'Original Image\nOverall Quality: {original_metrics.overall_score:.3f}', fontsize=12)
    plt.axis('off')
    
    # Processed image
    plt.subplot(2, 2, 2)
    plt.imshow(processed_rgb, cmap='gray' if len(processed_rgb.shape) == 2 else None)
    plt.title(f'Processed Image\nOverall Quality: {final_metrics.overall_score:.3f}', fontsize=12)
    plt.axis('off')
    
    # Quality comparison chart
    plt.subplot(2, 2, 3)
    categories = ['Blur', 'Noise', 'Contrast', 'Resolution', 'Lighting', 'Overall']
    original_scores = [
        original_metrics.blur_score,
        original_metrics.noise_score,
        original_metrics.contrast_score,
        original_metrics.resolution_score,
        original_metrics.lighting_score,
        original_metrics.overall_score
    ]
    processed_scores = [
        final_metrics.blur_score,
        final_metrics.noise_score,
        final_metrics.contrast_score,
        final_metrics.resolution_score,
        final_metrics.lighting_score,
        final_metrics.overall_score
    ]
    
    x = np.arange(len(categories))
    width = 0.35
    
    plt.bar(x - width/2, original_scores, width, label='Original', alpha=0.8)
    plt.bar(x + width/2, processed_scores, width, label='Processed', alpha=0.8)
    
    plt.xlabel('Quality Metrics')
    plt.ylabel('Score')
    plt.title('Quality Improvement Comparison')
    plt.xticks(x, categories, rotation=45)
    plt.legend()
    plt.ylim(0, 1.1)
    plt.grid(axis='y', alpha=0.3)
    
    # Metrics text summary
    plt.subplot(2, 2, 4)
    plt.axis('off')
    
    metrics_text = f"""
PROCESSING SUMMARY

Image Dimensions:
  Original: {original.shape[1]}x{original.shape[0]}
  Processed: {processed.shape[1]}x{processed.shape[0]}

Quality Improvement:
  Overall: {original_metrics.overall_score:.3f} → {final_metrics.overall_score:.3f}
  OCR Readiness: {original_metrics.ocr_readiness:.3f} → {final_metrics.ocr_readiness:.3f}
  
Key Improvements:
  Blur: {original_metrics.blur_score:.3f} → {final_metrics.blur_score:.3f}
  Contrast: {original_metrics.contrast_score:.3f} → {final_metrics.contrast_score:.3f}
  Resolution: {original_metrics.resolution_score:.3f} → {final_metrics.resolution_score:.3f}

Issues Detected:
  Original: {len(original_metrics.issues)}
  Processed: {len(final_metrics.issues)}
    """
    
    plt.text(0.1, 0.5, metrics_text, fontsize=10, verticalalignment='center',
             fontfamily='monospace', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")
    
    plt.show()


def test_individual_components(image: np.ndarray, config: OCRConfig):
    """Test individual components of the preprocessing pipeline."""
    print(f"\n{'='*60}")
    print("TESTING INDIVIDUAL COMPONENTS")
    print(f"{'='*60}")
    
    # Test Quality Analyzer
    print("\n1. Testing Quality Analyzer...")
    quality_analyzer = QualityAnalyzer(config)
    quality_metrics = quality_analyzer.analyze_quality(image)
    display_quality_metrics(quality_metrics, "Original Image Quality Analysis")
    
    # Test Image Processor components
    print("\n2. Testing Image Processor Components...")
    image_processor = ImageProcessor(config)
    
    # Test format normalization
    normalized_image, format_info = image_processor._normalize_image_format(image.copy())
    print(f"Format Normalization: {format_info}")
    
    # Test skew correction if needed (with conservative threshold)
    if abs(quality_metrics.skew_angle) > 2.0:  # Conservative threshold
        corrected_image, rotation_info = image_processor._correct_skew(
            normalized_image.copy(), quality_metrics.skew_angle
        )
        print(f"Skew Correction: {rotation_info}")
    
    # Test conservative resolution optimization
    optimized_image, resize_info = image_processor._optimize_resolution_conservative(
        normalized_image.copy(), quality_metrics
    )
    print(f"Resolution Optimization: {resize_info}")
    
    print("Individual component testing completed.")


def main():
    """Main test function."""
    logger = setup_logging()
    
    print("="*60)
    print("ADVANCED OCR PREPROCESSING PIPELINE TEST")
    print("="*60)
    
    # Define test image path
    image_path = "data/sample_images/img1.jpg"
    
    # Create alternative paths if the main path doesn't exist
    alternative_paths = [
        "input_data/sample_images/img1.jpg",
        "sample_images/img1.jpg",
        "data/sample_images/img1.jpg",
        "test_data/img1.jpg",
        "img1.jpg"
    ]
    
    # Try to find the image
    test_image = None
    actual_path = None
    
    for path in alternative_paths:
        if os.path.exists(path):
            test_image = load_test_image(path)
            actual_path = path
            break
    
    # If no image found, create a synthetic test image
    if test_image is None:
        print("No test image found. Creating synthetic test image...")
        test_image = create_synthetic_test_image()
        actual_path = "synthetic_test_image"
    
    if test_image is None:
        print("Failed to load or create test image. Exiting.")
        return
    
    print(f"Using test image: {actual_path}")
    print(f"Image shape: {test_image.shape}")
    print(f"Image dtype: {test_image.dtype}")
    
    # Create CONSERVATIVE configuration
    config = create_test_config()
    print("\nConfiguration created successfully")
    
    # Test individual components first
    test_individual_components(test_image, config)
    
    # Test complete preprocessing pipeline
    print(f"\n{'='*60}")
    print("TESTING COMPLETE PREPROCESSING PIPELINE")
    print(f"{'='*60}")
    
    try:
        # Initialize image processor
        image_processor = ImageProcessor(config)
        print("Image processor initialized successfully")
        
        # Analyze original image quality
        quality_analyzer = QualityAnalyzer(config)
        original_quality = quality_analyzer.analyze_quality(test_image)
        display_quality_metrics(original_quality, "ORIGINAL IMAGE QUALITY")
        
        # Process the image
        print("\nProcessing image...")
        preprocessing_result = image_processor.process_image(test_image)
        
        # Display results
        display_quality_metrics(preprocessing_result.quality_metrics, "PROCESSED IMAGE QUALITY")
        display_processing_metrics(preprocessing_result.processing_metrics)
        display_transformations(preprocessing_result.transformations_applied)
        
        # Create and display comparison plot
        print("\nCreating comparison visualization...")
        create_comparison_plot(
            preprocessing_result.original_image,
            preprocessing_result.enhanced_image,
            original_quality,
            preprocessing_result.quality_metrics,
            save_path="preprocessing_comparison.png"
        )
        
        # Save processed image
        output_path = "processed_image.jpg"
        cv2.imwrite(output_path, preprocessing_result.enhanced_image)
        print(f"Processed image saved to: {output_path}")
        
        # Test quick preprocessing mode
        print("\nTesting quick preprocessing mode...")
        quick_start = cv2.getTickCount()
        quick_processed = image_processor.quick_preprocess(test_image)
        quick_time = (cv2.getTickCount() - quick_start) / cv2.getTickFrequency()
        
        cv2.imwrite("quick_processed_image.jpg", quick_processed)
        print(f"Quick preprocessing completed in {quick_time:.3f}s")
        print(f"Quick processed image saved to: quick_processed_image.jpg")
        
        # Summary
        print(f"\n{'='*60}")
        print("TEST SUMMARY")
        print(f"{'='*60}")
        print(f"✓ Original image loaded successfully")
        print(f"✓ Quality analysis completed")
        print(f"✓ Image preprocessing completed")
        print(f"✓ Quality improvement: {preprocessing_result.processing_metrics.metadata.get('quality_improvement', 0):.3f}")
        print(f"✓ Processing time: {preprocessing_result.processing_metrics.duration:.3f}s")
        print(f"✓ Transformations applied: {preprocessing_result.processing_metrics.metadata.get('transformations_count', 0)}")
        print(f"✓ Output images saved")
        print(f"✓ Comparison plot generated")
        
        return preprocessing_result
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_synthetic_test_image() -> np.ndarray:
    """Create a synthetic test image for testing when no sample image is available."""
    print("Creating synthetic test image with various quality issues...")
    
    # Create base image
    image = np.ones((600, 800, 3), dtype=np.uint8) * 255
    
    # Add text-like rectangles to simulate document
    cv2.rectangle(image, (50, 100), (750, 130), (0, 0, 0), -1)  # Title
    cv2.rectangle(image, (50, 180), (700, 200), (0, 0, 0), -1)  # Line 1
    cv2.rectangle(image, (50, 220), (720, 240), (0, 0, 0), -1)  # Line 2
    cv2.rectangle(image, (50, 260), (650, 280), (0, 0, 0), -1)  # Line 3
    cv2.rectangle(image, (50, 320), (680, 340), (0, 0, 0), -1)  # Line 4
    cv2.rectangle(image, (50, 360), (710, 380), (0, 0, 0), -1)  # Line 5
    
    # Add quality issues
    # 1. Add noise
    noise = np.random.normal(0, 15, image.shape).astype(np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # 2. Reduce contrast
    image = cv2.convertScaleAbs(image, alpha=0.7, beta=30)
    
    # 3. Add blur
    image = cv2.GaussianBlur(image, (3, 3), 0.8)
    
    # 4. Add slight rotation (skew)
    rows, cols = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((cols/2, rows/2), -2.5, 1)
    image = cv2.warpAffine(image, rotation_matrix, (cols, rows), borderValue=(255, 255, 255))
    
    print("Synthetic test image created with blur, noise, low contrast, and skew")
    return image


if __name__ == "__main__":
    result = main()
    if result:
        print("\n🎉 Preprocessing pipeline test completed successfully!")
        print("Check the generated images and comparison plot.")
    else:
        print("\n❌ Test failed. Check the error messages above.")