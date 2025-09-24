from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Union, Optional, Callable
from pathlib import Path
import time

from .ocr_pipeline import OCRLibrary
from .types import OCRResult, ProcessingOptions
from .utils.logger import setup_logger

class BatchProcessor:
    """Efficient batch processing with parallel execution"""
    
    def __init__(self, max_workers: int = 4, ocr_config: Optional[str] = None):
        self.max_workers = max_workers
        self.logger = setup_logger(self.__class__.__name__)
        self.ocr_library = OCRLibrary(ocr_config)
    
    def process_directory(self, 
                         directory: Union[str, Path],
                         options: Optional[ProcessingOptions] = None,
                         pattern: str = "*",
                         recursive: bool = False,
                         progress_callback: Optional[Callable] = None) -> List[OCRResult]:
        """
        Process all images in a directory
        
        Args:
            directory: Directory containing images
            options: Processing options
            pattern: File pattern to match (e.g., "*.jpg", "*")  
            recursive: Search subdirectories
            progress_callback: Called with (current, total) progress
            
        Returns:
            List of OCRResult objects
        """
        directory = Path(directory)
        if not directory.exists():
            raise ValueError(f"Directory does not exist: {directory}")
        
        # Find image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        if recursive:
            image_files = [f for f in directory.rglob(pattern) 
                          if f.suffix.lower() in image_extensions]
        else:
            image_files = [f for f in directory.glob(pattern)
                          if f.suffix.lower() in image_extensions]
        
        if not image_files:
            self.logger.warning(f"No image files found in {directory}")
            return []
        
        return self.process_files(image_files, options, progress_callback)
    
    def process_files(self,
                     file_paths: List[Union[str, Path]],
                     options: Optional[ProcessingOptions] = None,
                     progress_callback: Optional[Callable] = None) -> List[OCRResult]:
        """
        Process a list of image files in parallel
        
        Args:
            file_paths: List of image file paths
            options: Processing options
            progress_callback: Called with (current, total) progress
            
        Returns:
            List of OCRResult objects in same order as input
        """
        total_files = len(file_paths)
        results = [None] * total_files  # Preserve order
        completed = 0
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all jobs with their indices
            future_to_index = {
                executor.submit(self.ocr_library.extract_text, file_path, options): i
                for i, file_path in enumerate(file_paths)
            }
            
            # Process completed futures
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    result = future.result(timeout=300)  # 5-minute timeout
                    results[index] = result
                except Exception as e:
                    # Create error result
                    error_result = self._create_error_result(file_paths[index], str(e))
                    results[index] = error_result
                    self.logger.error(f"Failed to process {file_paths[index]}: {e}")
                
                completed += 1
                
                # Call progress callback if provided
                if progress_callback:
                    progress_callback(completed, total_files)
        
        processing_time = time.time() - start_time
        success_count = sum(1 for r in results if r.success)
        
        self.logger.info(f"Batch processing completed: {success_count}/{total_files} successful "
                        f"in {processing_time:.2f}s")
        
        return results
    
    def _create_error_result(self, file_path: Union[str, Path], error: str) -> OCRResult:
        """Create an error result for failed processing"""
        from .types import QualityMetrics, ProcessingStrategy
        from .preprocessing.quality_analyzer import ImageQuality

        return OCRResult(
            text="",
            confidence=0.0,
            processing_time=0.0,
            engine_used="none",
            quality_metrics=QualityMetrics(
                overall_score=0.0,
                sharpness_score=0.0,
                contrast_score=0.0,
                brightness_score=0.0,
                noise_level=0.0,
                blur_score=0.0,
                quality_level="error",
                needs_enhancement=False,
                image_quality=ImageQuality.POOR,
                enhancement_recommendations=[],
                processing_time=0.0
            ),
            strategy_used=ProcessingStrategy.MINIMAL,
            metadata={'error': error, 'file_path': str(file_path)}
        )
