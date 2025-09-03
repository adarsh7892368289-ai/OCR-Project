"""
Simple OCR API - Complete Integration
Easy way to process images through your OCR system

File Location: src/api/simple_ocr_api.py

Usage:
    python -m src.api.simple_ocr_api --image path/to/image.jpg
    
Or use programmatically:
    from src.api.simple_ocr_api import SimpleOCR
    ocr = SimpleOCR()
    result = ocr.process_image("path/to/image.jpg")
"""

import argparse
import json
from pathlib import Path
import time
import sys
from typing import Optional, Dict, Any, Union, List
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.core.engine_manager import EngineManager
from src.engines.trocr_engine import TrOCREngine
from src.engines.tesseract_engine import TesseractEngine
from src.engines.easyocr_engine import EasyOCREngine
from src.preprocessing.adaptive_processor import AdaptivePreprocessor
from src.postprocessing.postprocessing_pipeline import PostProcessingPipeline
from src.utils.logger import get_logger


class SimpleOCR:
    """
    Simple OCR interface that combines all your components
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the complete OCR system
        
        Args:
            config_path: Path to configuration file
        """
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        
        try:
            # Initialize components
            self.logger.info("Initializing OCR system...")
            
            # 1. Initialize engines
            self.engine_manager = EngineManager(config_path)
            self._register_engines()
            
            # 2. Initialize preprocessing
            self.preprocessor = AdaptivePreprocessor(config_path)
            
            # 3. Initialize post-processing pipeline
            self.postprocessor = PostProcessingPipeline(config_path)
            
            self.logger.info("OCR system initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize OCR system: {e}")
            raise
    
    def _register_engines(self):
        """Register available OCR engines"""
        try:
            # Register TrOCR (best for mixed text)
            trocr_engine = TrOCREngine()
            self.engine_manager.register_engine("trocr", trocr_engine)
            
            # Register Tesseract (good for printed text)
            tesseract_engine = TesseractEngine()
            self.engine_manager.register_engine("tesseract", tesseract_engine)
            
            # Register EasyOCR (good backup)
            easyocr_engine = EasyOCREngine()
            self.engine_manager.register_engine("easyocr", easyocr_engine)
            
            # Set TrOCR as primary for mixed text capability
            self.engine_manager.set_primary_engine("trocr")
            
            self.logger.info("Engines registered: TrOCR (primary), Tesseract, EasyOCR")
            
        except Exception as e:
            self.logger.error(f"Error registering engines: {e}")
            # Fallback to available engines
            available_engines = self.engine_manager.get_available_engines()
            if available_engines:
                self.engine_manager.set_primary_engine(available_engines[0])
                self.logger.info(f"Using fallback engine: {available_engines[0]}")
    
    def process_image(
        self,
        image_path: Union[str, Path],
        output_format: str = "json",
        save_results: bool = False,
        output_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a single image through the complete OCR pipeline
        
        Args:
            image_path: Path to the image file
            output_format: Output format (json, txt, html, etc.)
            save_results: Whether to save results to file
            output_dir: Directory to save results
            
        Returns:
            Dictionary containing OCR results and processing info
        """
        start_time = time.time()
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        self.logger.info(f"Processing image: {image_path.name}")
        
        try:
            # Step 1: Preprocessing
            self.logger.info("Step 1: Preprocessing image...")
            preprocessed_result = self.preprocessor.process(str(image_path))
            
            # Step 2: OCR Recognition
            self.logger.info("Step 2: Running OCR recognition...")
            ocr_result = self.engine_manager.process_image(
                preprocessed_result.processed_image_path,
                prefer_engine="trocr"  # Best for mixed text
            )
            
            # Step 3: Post-processing
            self.logger.info("Step 3: Post-processing results...")
            pipeline_result = self.postprocessor.process(
                ocr_result=ocr_result,
                output_formats=[output_format]
            )
            
            # Step 4: Prepare final result
            total_time = time.time() - start_time
            
            final_result = {
                "success": True,
                "image_path": str(image_path),
                "extracted_text": ocr_result.text,
                "confidence": ocr_result.confidence,
                "total_processing_time": total_time,
                "processing_stages": {
                    "preprocessing": {
                        "time": preprocessed_result.processing_time,
                        "strategy_used": preprocessed_result.strategy_used,
                        "enhancements_applied": preprocessed_result.enhancements_applied
                    },
                    "ocr_recognition": {
                        "time": ocr_result.processing_time,
                        "engine_used": ocr_result.engine_name,
                        "regions_detected": len(ocr_result.regions)
                    },
                    "postprocessing": {
                        "time": pipeline_result.total_processing_time,
                        "corrections_applied": pipeline_result.success,
                        "output_format": output_format
                    }
                },
                "detailed_results": {
                    "regions": [
                        {
                            "text": region.text,
                            "confidence": region.confidence,
                            "bbox": {
                                "x": region.bbox.x,
                                "y": region.bbox.y,
                                "width": region.bbox.width,
                                "height": region.bbox.height
                            }
                        }
                        for region in ocr_result.regions
                    ],
                    "formatted_output": pipeline_result.formatted_results.get(output_format)
                }
            }
            
            # Save results if requested
            if save_results:
                self._save_results(final_result, image_path, output_dir)
            
            self.logger.info(f"Processing completed successfully in {total_time:.2f}s")
            return final_result
            
        except Exception as e:
            self.logger.error(f"Error processing image: {e}")
            return {
                "success": False,
                "image_path": str(image_path),
                "error": str(e),
                "total_processing_time": time.time() - start_time
            }
    
    def process_batch(
        self,
        image_paths: List[Union[str, Path]],
        output_format: str = "json",
        save_results: bool = False,
        output_dir: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Process multiple images
        
        Args:
            image_paths: List of image file paths
            output_format: Output format for all images
            save_results: Whether to save results
            output_dir: Directory to save results
            
        Returns:
            List of processing results
        """
        results = []
        
        self.logger.info(f"Starting batch processing of {len(image_paths)} images")
        
        for i, image_path in enumerate(image_paths, 1):
            self.logger.info(f"Processing image {i}/{len(image_paths)}: {Path(image_path).name}")
            
            result = self.process_image(
                image_path=image_path,
                output_format=output_format,
                save_results=save_results,
                output_dir=output_dir
            )
            
            results.append(result)
        
        # Summary statistics
        successful = sum(1 for r in results if r["success"])
        total_time = sum(r["total_processing_time"] for r in results)
        
        self.logger.info(f"Batch processing completed: {successful}/{len(image_paths)} successful, total time: {total_time:.2f}s")
        
        return results
    
    def _save_results(self, result: Dict[str, Any], image_path: Path, output_dir: Optional[str]):
        """Save processing results to file"""
        if output_dir:
            output_path = Path(output_dir)
        else:
            output_path = image_path.parent / "ocr_results"
        
        output_path.mkdir(exist_ok=True)
        
        # Save JSON result
        result_file = output_path / f"{image_path.stem}_result.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        # Save plain text
        text_file = output_path / f"{image_path.stem}_text.txt"
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(result["extracted_text"])
        
        self.logger.info(f"Results saved to {output_path}")
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get information about the OCR system configuration"""
        return {
            "available_engines": self.engine_manager.get_available_engines(),
            "primary_engine": self.engine_manager.primary_engine,
            "preprocessing_strategies": self.preprocessor.get_available_strategies(),
            "supported_formats": ["json", "txt", "html", "xml", "csv", "markdown", "pdf"],
            "system_stats": {
                "engine_manager": self.engine_manager.get_statistics(),
                "preprocessor": self.preprocessor.get_statistics(),
                "postprocessor": self.postprocessor.get_statistics()
            }
        }


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description="Simple OCR System")
    parser.add_argument("--image", "-i", required=True, help="Path to image file")
    parser.add_argument("--output-format", "-f", default="json", 
                       choices=["json", "txt", "html", "xml", "csv", "markdown"],
                       help="Output format")
    parser.add_argument("--save", "-s", action="store_true", help="Save results to file")
    parser.add_argument("--output-dir", "-o", help="Output directory for results")
    parser.add_argument("--config", "-c", help="Configuration file path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    args = parser.parse_args()
    
    # Set up logging
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    try:
        # Initialize OCR system
        print("Initializing OCR system...")
        ocr = SimpleOCR(config_path=args.config)
        
        # Process image
        print(f"Processing image: {args.image}")
        result = ocr.process_image(
            image_path=args.image,
            output_format=args.output_format,
            save_results=args.save,
            output_dir=args.output_dir
        )
        
        if result["success"]:
            print("\n" + "="*50)
            print("OCR RESULTS")
            print("="*50)
            print(f"Extracted Text:")
            print("-" * 20)
            print(result["extracted_text"])
            print("-" * 20)
            print(f"Confidence: {result['confidence']:.2%}")
            print(f"Processing Time: {result['total_processing_time']:.2f}s")
            print(f"Regions Detected: {len(result['detailed_results']['regions'])}")
            print(f"Engine Used: {result['processing_stages']['ocr_recognition']['engine_used']}")
            
            if args.output_format == "json":
                print("\nFull JSON Result:")
                print(json.dumps(result, indent=2))
        else:
            print(f"Error: {result['error']}")
            return 1
            
    except Exception as e:
        print(f"Fatal error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())