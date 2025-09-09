#!/usr/bin/env python3
"""
Test script for the adaptive preprocessing pipeline
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import modules directly to avoid relative import issues
from src.preprocessing.adaptive_processor import (
    AdaptivePreprocessor,
    ProcessingOptions,
    ProcessingLevel,
    PipelineStrategy
)

def load_image(image_path: str) -> np.ndarray:
    """Load image from file"""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    return image

def display_results(original: np.ndarray, processed: np.ndarray,
                   result_info: dict, save_path: str = None):
    """Display original and processed images with results"""

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Original image
    axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Processed image
    axes[1].imshow(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Processed Image')
    axes[1].axis('off')

    # Add text information
    info_text = f"Pipeline: {result_info.get('pipeline_used', 'N/A')}\n"
    info_text += f"Processing Time: {result_info.get('processing_time', 0):.2f}s\n"
    info_text += f"Steps: {', '.join(result_info.get('processing_steps', []))}"

    fig.suptitle('Adaptive Preprocessing Pipeline Results', fontsize=14)
    plt.figtext(0.02, 0.02, info_text, fontsize=10, verticalalignment='bottom')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Results saved to: {save_path}")

    plt.tight_layout()
    plt.show()

def main():
    """Main test function"""

    # Image path - using skewed_text.png as it should benefit from preprocessing
    image_path = "data/sample_images/skewed_text.png"

    print("=== Adaptive Preprocessing Pipeline Test ===")
    print(f"Loading image: {image_path}")

    try:
        # Load image
        original_image = load_image(image_path)
        print(f"Image loaded successfully. Shape: {original_image.shape}")

        # Initialize preprocessor
        print("Initializing adaptive preprocessor...")
        preprocessor = AdaptivePreprocessor()

        # Configure processing options
        options = ProcessingOptions(
            processing_level=ProcessingLevel.BALANCED,
            strategy=PipelineStrategy.CONTENT_AWARE,
            enable_quality_validation=True,
            max_processing_iterations=2
        )

        print("Processing options:")
        print(f"  Level: {options.processing_level.value}")
        print(f"  Strategy: {options.strategy.value}")
        print(f"  Quality validation: {options.enable_quality_validation}")

        # Process image
        print("\nProcessing image through adaptive pipeline...")
        result = preprocessor.process_image(original_image, options)

        # Display results
        print("\n=== Processing Results ===")
        print(f"Success: {result.success}")
        print(f"Quality Improvement: {result.metadata.get('quality_improvement', 0):.3f}")
        print(f"Pipeline Used: {result.metadata.get('pipeline_used', 'N/A')}")
        print(f"Processing Steps: {len(result.processing_steps)}")
        for i, step in enumerate(result.processing_steps, 1):
            print(f"  {i}. {step}")

        if result.warnings:
            print(f"Warnings: {len(result.warnings)}")
            for warning in result.warnings:
                print(f"  - {warning}")

        # Quality metrics
        print("\n=== Quality Metrics ===")
        orig_metrics = result.quality_metrics.get('original', {})
        final_metrics = result.quality_metrics.get('final', {})

        print(f"Original Overall Score: {orig_metrics.get('overall_score', 0):.3f}")
        print(f"Final Overall Score: {final_metrics.get('overall_score', 0):.3f}")

        # Performance stats
        perf_stats = result.performance_stats
        print("\n=== Performance Statistics ===")
        print(f"Processing Time: {perf_stats.get('processing_time', 0):.2f}s")
        print(f"Steps Executed: {perf_stats.get('steps_executed', 0)}")
        print(f"Quality Change: {perf_stats.get('quality_change', 0):.3f}")
        print(f"Memory Usage (MB): Original {perf_stats.get('memory_usage', {}).get('original_image_mb', 0):.2f}, Processed {perf_stats.get('memory_usage', {}).get('processed_image_mb', 0):.2f}")

        # Display images
        print("\nDisplaying results...")
        display_results(
            original_image,
            result.processed_image,
            {
                'pipeline_used': result.metadata.get('pipeline_used', 'N/A'),
                'processing_time': result.processing_time,
                'processing_steps': result.processing_steps
            },
            save_path="pipeline_test_results.png"
        )

        # Cleanup
        preprocessor.shutdown()
        print("\nTest completed successfully!")

    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
