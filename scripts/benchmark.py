# scripts/benchmark.py

import time
import cv2
import numpy as np
from pathlib import Path
import argparse
from src.api.main import ModernOCRSystem
import statistics

def benchmark_ocr_system(image_paths: list, runs: int = 3):
    """Benchmark OCR system performance"""
    
    print("Initializing OCR System...")
    ocr_system = ModernOCRSystem()
    
    results = []
    
    for image_path in image_paths:
        print(f"\nProcessing: {image_path}")
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Could not load image: {image_path}")
            continue
            
        image_results = []
        
        # Run multiple times
        for run in range(runs):
            print(f"  Run {run + 1}/{runs}...")
            
            start_time = time.time()
            result = ocr_system.process_image(image)
            end_time = time.time()
            
            processing_time = end_time - start_time
            
            image_results.append({
                "processing_time": processing_time,
                "success": result["success"],
                "confidence": result.get("confidence", 0),
                "word_count": result.get("word_count", 0),
                "text_length": len(result.get("text", ""))
            })
            
        # Calculate statistics
        times = [r["processing_time"] for r in image_results]
        confidences = [r["confidence"] for r in image_results if r["success"]]
        
        results.append({
            "image": image_path.name,
            "avg_time": statistics.mean(times),
            "min_time": min(times),
            "max_time": max(times),
            "std_time": statistics.stdev(times) if len(times) > 1 else 0,
            "success_rate": sum(1 for r in image_results if r["success"]) / len(image_results),
            "avg_confidence": statistics.mean(confidences) if confidences else 0,
            "word_count": image_results[0]["word_count"] if image_results else 0
        })
        
    return results

def print_benchmark_results(results: list):
    """Print benchmark results in a formatted table"""
    
    print("\n" + "="*100)
    print("BENCHMARK RESULTS")
    print("="*100)
    
    print(f"{'Image':<30} {'Avg Time (s)':<12} {'Min Time (s)':<12} {'Max Time (s)':<12} {'Success Rate':<12} {'Confidence':<12} {'Words':<8}")
    print("-"*100)
    
    total_time = 0
    total_success = 0
    total_confidence = 0
    
    for result in results:
        print(f"{result['image']:<30} "
              f"{result['avg_time']:<12.3f} "
              f"{result['min_time']:<12.3f} "
              f"{result['max_time']:<12.3f} "
              f"{result['success_rate']:<12.1%} "
              f"{result['avg_confidence']:<12.3f} "
              f"{result['word_count']:<8}")
        
        total_time += result['avg_time']
        total_success += result['success_rate']
        total_confidence += result['avg_confidence']
        
    print("-"*100)
    print(f"{'AVERAGE':<30} "
          f"{total_time/len(results):<12.3f} "
          f"{'N/A':<12} "
          f"{'N/A':<12} "
          f"{total_success/len(results):<12.1%} "
          f"{total_confidence/len(results):<12.3f} "
          f"{'N/A':<8}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark OCR system performance")
    parser.add_argument("--images", "-i", required=True, help="Directory containing test images")
    parser.add_argument("--runs", "-r", type=int, default=3, help="Number of runs per image")
    parser.add_argument("--extensions", "-e", nargs="+", default=[".jpg", ".jpeg", ".png", ".tiff", ".bmp"], help="Image extensions to process")
    
    args = parser.parse_args()
    
    image_dir = Path(args.images)
    if not image_dir.exists():
        print(f"Error: Directory {image_dir} does not exist")
        exit(1)
        
    # Find all images
    image_paths = []
    for ext in args.extensions:
        image_paths.extend(image_dir.glob(f"*{ext}"))
        image_paths.extend(image_dir.glob(f"*{ext.upper()}"))
        
    if not image_paths:
        print(f"No images found in {image_dir}")
        exit(1)
        
    print(f"Found {len(image_paths)} images")
    
    # Run benchmark
    results = benchmark_ocr_system(image_paths, args.runs)
    print_benchmark_results(results)