# Create this new file: src/preprocessing/optimized_preprocessing.py

import cv2
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

# Import your existing components
try:
    from .quality_analyzer import QualityAnalyzer, QualityMetrics, ImageType
    from .image_enhancer import AIImageEnhancer, EnhancementResult
    from .text_detector import AdvancedTextDetector, TextRegion
except ImportError:
    from src.preprocessing.quality_analyzer import QualityAnalyzer, QualityMetrics, ImageType
    from src.preprocessing.image_enhancer import AIImageEnhancer, EnhancementResult
    from src.preprocessing.text_detector import AdvancedTextDetector, TextRegion

logger = logging.getLogger(__name__)

@dataclass
class OptimizedProcessingResult:
    """Result of optimized preprocessing pipeline"""
    original_image: np.ndarray
    enhanced_image: Optional[np.ndarray] = None
    quality_metrics: Optional[QualityMetrics] = None
    enhancement_result: Optional[EnhancementResult] = None
    detected_regions: List[TextRegion] = field(default_factory=list)
    
    # Performance metrics
    total_processing_time: float = 0.0
    quality_analysis_time: float = 0.0
    enhancement_time: float = 0.0
    detection_time: float = 0.0
    
    # Processing decisions
    enhancement_skipped: bool = False
    skip_reason: str = ""
    parallel_processing_used: bool = False
    
    @property
    def regions_count(self) -> int:
        return len(self.detected_regions)
    
    @property
    def average_confidence(self) -> float:
        if not self.detected_regions:
            return 0.0
        return sum(r.confidence for r in self.detected_regions) / len(self.detected_regions)
    
    @property
    def high_confidence_regions(self) -> List[TextRegion]:
        return [r for r in self.detected_regions if r.confidence > 0.8]

class OptimizedPreprocessingPipeline:
    """
    High-performance preprocessing pipeline with conditional enhancement
    and parallel processing
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize optimized preprocessing pipeline"""
        self.config = config or {}
        
        # Initialize components
        self.quality_analyzer = QualityAnalyzer(self.config.get('quality_analyzer', {}))
        self.image_enhancer = AIImageEnhancer(self.config.get('image_enhancer', {}))
        self.text_detector = AdvancedTextDetector(self.config.get('text_detector', {}))
        
        # Optimization settings
        self.enable_conditional_enhancement = self.config.get('enable_conditional_enhancement', True)
        self.enable_parallel_detection = self.config.get('enable_parallel_detection', True)
        self.enable_reading_order = self.config.get('enable_reading_order', True)
        
        # Performance thresholds
        self.enhancement_skip_threshold = self.config.get('enhancement_skip_threshold', 0.65)
        self.parallel_processing_threshold = self.config.get('parallel_processing_threshold', 2000000)
        
        logger.info("Optimized preprocessing pipeline initialized")
    
    def process_image(self, image: np.ndarray, 
                     force_enhancement: bool = False,
                     debug_output_dir: Optional[str] = None) -> OptimizedProcessingResult:
        """
        MAIN OPTIMIZED PROCESSING METHOD
        
        Args:
            image: Input image to process
            force_enhancement: Force enhancement even if conditional logic suggests skipping
            debug_output_dir: Directory to save debug images
            
        Returns:
            OptimizedProcessingResult with all processing results and metrics
        """
        total_start_time = time.time()
        
        if image is None or image.size == 0:
            return OptimizedProcessingResult(
                original_image=np.zeros((100, 100, 3), dtype=np.uint8),
                total_processing_time=0.0
            )
        
        logger.info(f"Starting optimized processing: {image.shape}")
        
        # Stage 1: Quality Analysis
        quality_start = time.time()
        quality_metrics = self.quality_analyzer.analyze_image(image)
        quality_time = time.time() - quality_start
        
        logger.info(f"Quality analysis: {quality_time:.3f}s, Score: {quality_metrics.overall_score:.3f}")
        
        # Stage 2: Conditional Enhancement
        enhancement_start = time.time()
        
        if self.enable_conditional_enhancement:
            # Use smart enhancement with conditional logic
            enhancement_result = self.image_enhancer.smart_enhance_image(
                image, quality_metrics, force_enhancement
            )
            enhanced_image = enhancement_result.enhanced_image
            enhancement_skipped = enhancement_result.enhancement_applied == "skipped"
            skip_reason = enhancement_result.metadata.get("skip_reason", "")
        else:
            # Use standard enhancement
            enhancement_result = self.image_enhancer.enhance_image(image, None, quality_metrics)
            enhanced_image = enhancement_result.enhanced_image
            enhancement_skipped = False
            skip_reason = ""
        
        enhancement_time = time.time() - enhancement_start
        
        if enhancement_skipped:
            logger.info(f"Enhancement SKIPPED in {enhancement_time:.3f}s: {skip_reason}")
        else:
            logger.info(f"Enhancement completed in {enhancement_time:.3f}s")
        
        # Stage 3: Text Detection (with parallel processing if enabled)
        detection_start = time.time()
        
        # Decide which image to use for detection
        detection_image = enhanced_image if not enhancement_skipped else image
        
        if self.enable_parallel_detection:
            detected_regions = self.text_detector.detect_text_regions_parallel(detection_image)
            parallel_used = True
        else:
            detected_regions = self.text_detector.detect_text_regions(detection_image)
            parallel_used = False
        
        detection_time = time.time() - detection_start
        total_time = time.time() - total_start_time
        
        logger.info(f"Detection completed in {detection_time:.3f}s, {len(detected_regions)} regions")
        logger.info(f"TOTAL PROCESSING TIME: {total_time:.3f}s")
        
        # Create result
        result = OptimizedProcessingResult(
            original_image=image,
            enhanced_image=enhanced_image,
            quality_metrics=quality_metrics,
            enhancement_result=enhancement_result,
            detected_regions=detected_regions,
            total_processing_time=total_time,
            quality_analysis_time=quality_time,
            enhancement_time=enhancement_time,
            detection_time=detection_time,
            enhancement_skipped=enhancement_skipped,
            skip_reason=skip_reason,
            parallel_processing_used=parallel_used
        )
        
        # Save debug images if requested
        if debug_output_dir:
            self._save_debug_images(result, debug_output_dir)
        
        return result
    
    def _save_debug_images(self, result: OptimizedProcessingResult, 
                          debug_output_dir: str):
        """Save debug images showing processing results"""
        debug_dir = Path(debug_output_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save original image
            cv2.imwrite(str(debug_dir / "01_original.jpg"), result.original_image)
            
            # Save enhanced image (if different)
            if not result.enhancement_skipped and result.enhanced_image is not None:
                cv2.imwrite(str(debug_dir / "02_enhanced.jpg"), result.enhanced_image)
            
            # Save detection visualization
            if result.detected_regions:
                vis_image = self.text_detector.visualize_detections(
                    result.enhanced_image if not result.enhancement_skipped else result.original_image,
                    result.detected_regions
                )
                cv2.imwrite(str(debug_dir / "03_detections.jpg"), vis_image)
            
            # Save processing summary
            summary = {
                "total_processing_time": result.total_processing_time,
                "quality_analysis_time": result.quality_analysis_time,
                "enhancement_time": result.enhancement_time,
                "detection_time": result.detection_time,
                "enhancement_skipped": result.enhancement_skipped,
                "skip_reason": result.skip_reason,
                "parallel_processing_used": result.parallel_processing_used,
                "regions_count": result.regions_count,
                "average_confidence": result.average_confidence,
                "high_confidence_regions": len(result.high_confidence_regions),
                "quality_score": result.quality_metrics.overall_score if result.quality_metrics else None
            }
            
            import json
            with open(debug_dir / "processing_summary.json", 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"Debug images saved to {debug_dir}")
            
        except Exception as e:
            logger.warning(f"Failed to save debug images: {e}")
    
    def benchmark_processing(self, test_images: List[np.ndarray]) -> Dict[str, Any]:
        """
        Benchmark the optimized pipeline against multiple images
        """
        if not test_images:
            return {}
        
        logger.info(f"Benchmarking pipeline with {len(test_images)} images")
        
        results = []
        total_start = time.time()
        
        for i, image in enumerate(test_images):
            result = self.process_image(image)
            results.append(result)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(test_images)} images")
        
        total_benchmark_time = time.time() - total_start
        
        # Calculate statistics
        total_times = [r.total_processing_time for r in results]
        enhancement_times = [r.enhancement_time for r in results]
        detection_times = [r.detection_time for r in results]
        quality_times = [r.quality_analysis_time for r in results]
        
        enhancement_skipped_count = sum(1 for r in results if r.enhancement_skipped)
        parallel_used_count = sum(1 for r in results if r.parallel_processing_used)
        
        total_regions = sum(r.regions_count for r in results)
        avg_confidence = sum(r.average_confidence for r in results) / len(results)
        
        benchmark_stats = {
            "benchmark_summary": {
                "total_images": len(test_images),
                "total_benchmark_time": total_benchmark_time,
                "avg_processing_time_per_image": sum(total_times) / len(total_times),
                "images_per_second": len(test_images) / total_benchmark_time
            },
            "timing_breakdown": {
                "avg_quality_analysis": sum(quality_times) / len(quality_times),
                "avg_enhancement_time": sum(enhancement_times) / len(enhancement_times),
                "avg_detection_time": sum(detection_times) / len(detection_times),
                "min_total_time": min(total_times),
                "max_total_time": max(total_times)
            },
            "optimization_impact": {
                "enhancement_skipped_percentage": (enhancement_skipped_count / len(results)) * 100,
                "parallel_processing_percentage": (parallel_used_count / len(results)) * 100,
                "total_regions_detected": total_regions,
                "average_regions_per_image": total_regions / len(results),
                "average_confidence_score": avg_confidence
            },
            "individual_results": [
                {
                    "image_index": i,
                    "total_time": r.total_processing_time,
                    "regions_count": r.regions_count,
                    "average_confidence": r.average_confidence,
                    "enhancement_skipped": r.enhancement_skipped,
                    "parallel_used": r.parallel_processing_used
                }
                for i, r in enumerate(results)
            ]
        }
        
        logger.info(f"Benchmark completed: {len(test_images)} images in {total_benchmark_time:.3f}s")
        logger.info(f"Average processing time: {benchmark_stats['benchmark_summary']['avg_processing_time_per_image']:.3f}s")
        logger.info(f"Enhancement skipped: {benchmark_stats['optimization_impact']['enhancement_skipped_percentage']:.1f}%")
        
        return benchmark_stats
    
    def process_batch_images(self, images: List[np.ndarray], 
                           progress_callback: Optional[callable] = None) -> List[OptimizedProcessingResult]:
        """Process multiple images efficiently"""
        results = []
        
        for i, image in enumerate(images):
            result = self.process_image(image)
            results.append(result)
            
            if progress_callback:
                progress_callback(i + 1, len(images), result)
        
        return results
    
    def get_optimization_settings(self) -> Dict[str, Any]:
        """Get current optimization settings"""
        return {
            "conditional_enhancement": self.enable_conditional_enhancement,
            "parallel_detection": self.enable_parallel_detection,
            "reading_order": self.enable_reading_order,
            "enhancement_skip_threshold": self.enhancement_skip_threshold,
            "parallel_threshold": self.parallel_processing_threshold,
            "max_workers": self.text_detector.max_workers if hasattr(self.text_detector, 'max_workers') else 'N/A'
        }

# Convenience function for easy integration
def create_optimized_pipeline(config_path: Optional[str] = None) -> OptimizedPreprocessingPipeline:
    """
    Factory function to create optimized preprocessing pipeline
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Configured OptimizedPreprocessingPipeline instance
    """
    config = {}
    
    if config_path:
        try:
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Could not load config from {config_path}: {e}")
    
    # Set optimized defaults if not specified
    optimized_defaults = {
        'enable_conditional_enhancement': True,
        'enable_parallel_detection': True,
        'enable_reading_order': True,
        'enhancement_skip_threshold': 0.65,
        'parallel_processing_threshold': 2000000,
        'text_detector': {
            'enable_parallel': True,
            'max_workers': 4,
            'enable_reading_order': True,
            'confidence_threshold': 0.6,
            'nms_threshold': 0.3,
            'min_region_area': 150
        },
        'image_enhancer': {
            'enable_ai_guidance': True,
            'measure_improvement': True,
            'enhancement_level': 'medium'
        }
    }
    
    # Merge defaults with provided config
    for key, value in optimized_defaults.items():
        if key not in config:
            config[key] = value
        elif isinstance(value, dict) and isinstance(config[key], dict):
            for subkey, subvalue in value.items():
                if subkey not in config[key]:
                    config[key][subkey] = subvalue
    
    return OptimizedPreprocessingPipeline(config)