"""
Test file for preprocessing pipeline visualization.

This test loads an image from the data folder and runs the preprocessing pipeline
step by step, displaying the image after each preprocessing step using matplotlib.
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from advanced_ocr.config import OCRConfig
from advanced_ocr.utils.model_utils import ModelLoader
from advanced_ocr.preprocessing.image_processor import ImageProcessor, ImageEnhancer
from advanced_ocr.preprocessing.quality_analyzer import QualityAnalyzer
from advanced_ocr.preprocessing.text_detector import TextDetector


def display_image(image: np.ndarray, title: str, step: int):
    """
    Save image with title and step number to file.

    Args:
        image (np.ndarray): Image to save
        title (str): Title for the image
        step (int): Step number
    """
    # Create output directory if it doesn't exist
    output_dir = Path("test_output")
    output_dir.mkdir(exist_ok=True)

    # Save image
    filename = f"step_{step}_{title.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png"
    filepath = output_dir / filename

    # Convert BGR to RGB if needed for saving
    if len(image.shape) == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(str(filepath), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    else:
        cv2.imwrite(str(filepath), image)

    print(f"Saved image: {filepath}")


def draw_bounding_boxes(image: np.ndarray, text_regions):
    """
    Draw bounding boxes on the image for visualization.

    Args:
        image (np.ndarray): Image to draw on
        text_regions: List of text regions with bounding boxes

    Returns:
        np.ndarray: Image with bounding boxes drawn
    """
    image_with_boxes = image.copy()
    for region in text_regions:
        bbox = region.bbox
        # Draw rectangle
        cv2.rectangle(image_with_boxes,
                     (bbox.x_min, bbox.y_min),
                     (bbox.x_max, bbox.y_max),
                     (0, 255, 0), 2)
        # Add confidence text
        cv2.putText(image_with_boxes,
                   ".2f",
                   (bbox.x_min, bbox.y_min - 5),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   0.5, (0, 255, 0), 1)
    return image_with_boxes


def test_preprocessing_pipeline_visualization():
    """
    Test the preprocessing pipeline by running each step manually
    and displaying the image after each preprocessing step.
    """
    # Path to sample image
    image_path = Path("data/sample_images/img1.jpg")

    if not image_path.exists():
        raise FileNotFoundError(f"Sample image not found: {image_path}")

    # Load original image
    original_image = cv2.imread(str(image_path))
    if original_image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    print(f"Loaded image with shape: {original_image.shape}")

    # Display original image
    display_image(original_image, "Original Image", 1)

    # Create configuration
    config = OCRConfig()
    config.set_profile("balanced")  # Use balanced profile for testing

    # Create model loader
    model_loader = ModelLoader(config, "models")

    # Create preprocessing components
    quality_analyzer = QualityAnalyzer(config)
    image_enhancer = ImageEnhancer(config)
    text_detector = TextDetector(model_loader, config)

    # Step 2: Resize image if needed
    image_processor = ImageProcessor(model_loader, config)
    resized_image, scale_factor = image_processor._resize_image_if_needed(original_image)

    if scale_factor != 1.0:
        print(".3f")
        display_image(resized_image, f"Resized Image (scale: {scale_factor:.3f})", 2)
    else:
        print("Image does not need resizing")
        resized_image = original_image.copy()

    # Step 3: Quality analysis
    print("Performing quality analysis...")
    quality_metrics = quality_analyzer.analyze_image_quality(resized_image)
    print(f"Quality metrics: overall_score={quality_metrics.overall_score:.3f}, "
          f"level={quality_metrics.overall_level.value}")
    print(f"Blur score: {quality_metrics.blur_score:.3f}, "
          f"Noise score: {quality_metrics.noise_score:.3f}")
    print(f"Contrast score: {quality_metrics.contrast_score:.3f}, "
          f"Brightness score: {quality_metrics.brightness_score:.3f}")

    # Quality analysis doesn't change the image, so no new display

    # Step 4: Image enhancement
    print("Applying image enhancement...")
    enhancement_strategy = image_enhancer.determine_enhancement_strategy(quality_metrics)
    enhanced_image, applied_enhancements = image_enhancer.enhance_image(
        resized_image, enhancement_strategy, quality_metrics
    )

    print(f"Enhancement strategy: {enhancement_strategy.value}")
    print(f"Enhancements applied: {applied_enhancements}")

    display_image(enhanced_image, f"Enhanced Image ({enhancement_strategy.value})", 3)

    # Step 5: Text region detection
    print("Detecting text regions...")
    text_regions = text_detector.detect_text_regions(enhanced_image)

    print(f"Detected {len(text_regions)} text regions")

    # Draw bounding boxes on the enhanced image
    image_with_boxes = draw_bounding_boxes(enhanced_image, text_regions)

    display_image(image_with_boxes, f"Text Detection Results ({len(text_regions)} regions)", 4)

    # Print summary of detected regions
    print("\nDetected text regions:")
    for i, region in enumerate(text_regions[:5]):  # Show first 5
        bbox = region.bbox
        print(f"Region {i+1}: bbox=({bbox.x_min}, {bbox.y_min}, {bbox.x_max}, {bbox.y_max}), "
              ".3f")

    if len(text_regions) > 5:
        print(f"... and {len(text_regions) - 5} more regions")

    print("\nPreprocessing pipeline test completed successfully!")


if __name__ == "__main__":
    # Run the test
    test_preprocessing_pipeline_visualization()
