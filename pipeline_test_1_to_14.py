#!/usr/bin/env python3
"""
OCR Pipeline Test Script - Tests up to Test 14 (Multi-Engine Processing)
No postprocessing - Raw OCR output only
"""

import sys
import os
import cv2
import numpy as np
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

try:
    from src.utils.config import load_config
    from src.utils.logger import setup_logger
    from src.utils.image_utils import ImageUtils
    from src.preprocessing.quality_analyzer import IntelligentQualityAnalyzer
    from src.preprocessing.image_enhancer import AIImageEnhancer
    from src.preprocessing.text_detector import AdvancedTextDetector
    from src.core.base_engine import OCRResult, TextRegion, BoundingBox, TextType
    from src.engines.tesseract_engine import TesseractEngine
    from src.engines.paddleocr_engine import PaddleOCREngine
    from src.engines.easyocr_engine import EasyOCREngine
    from src.engines.trocr_engine import TrOCREngine
    from src.core.engine_manager import EngineManager
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all required modules are available")
    sys.exit(1)

# Setup logger
logger = setup_logger(__name__)

class PipelineTestRunner:
    """Test runner for OCR Pipeline - Tests 1-14 only"""
    
    def __init__(self):
        self.test_results = {}
        self.config = self._load_config()
        self.test_image_path = None
        
    def _load_config(self):
        """Load configuration files"""
        try:
            # Try to load main OCR config
            config_path = Path("data/configs/ocr_config.yaml")
            if config_path.exists():
                config = load_config(str(config_path))
                logger.info("Loaded OCR configuration")
                return config
            else:
                logger.warning("OCR config not found, using defaults")
                return self._get_default_config()
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return self._get_default_config()
    
    def _get_default_config(self):
        """Get default configuration"""
        return {
            "engines": {
                "tesseract": {"enabled": True, "priority": 3},
                "paddleocr": {"enabled": True, "priority": 1},
                "easyocr": {"enabled": True, "priority": 2},
                "trocr": {"enabled": True, "priority": 4}
            },
            "preprocessing": {
                "quality_analysis": True,
                "enhancement": True,
                "text_detection": True
            }
        }
    
    def test_1_image_loading(self):
        """Test 1: Image Loading"""
        logger.info("Running Test 1: Image Loading")
        
        # Look for img1.jpg first, then any available image
        sample_dir = Path("data/sample_images")
        test_images = ["img3.jpg"]
        
        image_path = None
        
        if sample_dir.exists():
            # First try to find img1.jpg
            for img_name in test_images:
                potential_path = sample_dir / img_name
                if potential_path.exists():
                    image_path = str(potential_path)
                    break
            
            # If not found, take any jpg/png file
            if not image_path:
                for ext in ["*.jpg", "*.jpeg", "*.png"]:
                    images = list(sample_dir.glob(ext))
                    if images:
                        image_path = str(images[0])
                        break
        
        if not image_path:
            return {"status": "FAILED", "error": "No test images found in data/sample_images/"}
        
        try:
            # Test OpenCV loading
            image = cv2.imread(image_path)
            if image is None:
                return {"status": "FAILED", "error": f"OpenCV failed to load {image_path}"}
            
            # Test with ImageUtils
            image_utils = ImageUtils.load_image(image_path)
            if image_utils is None:
                return {"status": "FAILED", "error": f"ImageUtils failed to load {image_path}"}
            
            # Check dimensions and quality
            height, width = image.shape[:2]
            if height < 50 or width < 50:
                return {"status": "FAILED", "error": f"Image too small: {width}x{height}"}
            
            self.test_image_path = image_path
            
            return {
                "status": "PASSED",
                "image_path": image_path,
                "dimensions": f"{width}x{height}",
                "channels": image.shape[2] if len(image.shape) == 3 else 1,
                "size_mb": round(os.path.getsize(image_path) / (1024*1024), 2)
            }
            
        except Exception as e:
            return {"status": "FAILED", "error": str(e)}
    
    def test_2_configuration_system(self):
        """Test 2: Configuration System"""
        logger.info("Running Test 2: Configuration System")
        
        try:
            # Test config loading
            config_files_tested = []
            
            # Test main OCR config
            ocr_config_path = Path("data/configs/ocr_config.yaml")
            if ocr_config_path.exists():
                config = load_config(str(ocr_config_path))
                config_files_tested.append("ocr_config.yaml")
            
            # Test other config files
            config_dir = Path("data/configs")
            if config_dir.exists():
                for config_file in config_dir.glob("*.yaml"):
                    try:
                        load_config(str(config_file))
                        config_files_tested.append(config_file.name)
                    except:
                        pass
            
            return {
                "status": "PASSED",
                "config_files_loaded": config_files_tested,
                "config_available": self.config is not None
            }
            
        except Exception as e:
            return {"status": "FAILED", "error": str(e)}
    
    def test_3_logger_system(self):
        """Test 3: Logger System"""
        logger.info("Running Test 3: Logger System")
        
        try:
            # Test logger creation
            test_logger = setup_logger("test_logger")
            
            # Test different log levels
            test_logger.debug("Debug message")
            test_logger.info("Info message")
            test_logger.warning("Warning message")
            test_logger.error("Error message")
            
            return {
                "status": "PASSED",
                "logger_created": True,
                "log_levels_tested": ["debug", "info", "warning", "error"]
            }
            
        except Exception as e:
            return {"status": "FAILED", "error": str(e)}
    
    def test_4_image_utilities(self):
        """Test 4: Image Utilities"""
        logger.info("Running Test 4: Image Utilities")
        
        if not self.test_image_path:
            return {"status": "FAILED", "error": "No test image available"}
        
        try:
            # Test ImageUtils functions
            image = ImageUtils.load_image(self.test_image_path)
            if image is None:
                return {"status": "FAILED", "error": "Failed to load image"}
            
            # Test basic operations
            operations_tested = []
            
            # Test format conversion if available
            if hasattr(ImageUtils, 'convert_to_grayscale'):
                gray = ImageUtils.convert_to_grayscale(image)
                operations_tested.append("grayscale_conversion")
            
            # Test resize if available
            if hasattr(ImageUtils, 'resize_image'):
                resized = ImageUtils.resize_image(image, (200, 200))
                operations_tested.append("resize")
            
            # Test basic OpenCV operations
            gray_cv = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            operations_tested.append("cv2_grayscale")
            
            return {
                "status": "PASSED",
                "image_loaded": True,
                "operations_tested": operations_tested,
                "original_shape": image.shape
            }
            
        except Exception as e:
            return {"status": "FAILED", "error": str(e)}
    
    def test_5_quality_analyzer(self):
        """Test 5: Quality Analyzer"""
        logger.info("Running Test 5: Quality Analyzer")
        
        if not self.test_image_path:
            return {"status": "FAILED", "error": "No test image available"}
        
        try:
            image = ImageUtils.load_image(self.test_image_path)
            analyzer = IntelligentQualityAnalyzer(self.config.get("quality_analyzer", {}))
            
            quality_metrics = analyzer.analyze_image(image)
            
            return {
                "status": "PASSED",
                "quality_metrics": {
                    "overall_score": quality_metrics.overall_score,
                    "sharpness_score": quality_metrics.sharpness_score,
                    "contrast_score": quality_metrics.contrast_score,
                    "brightness_score": quality_metrics.brightness_score,
                    "quality_level": quality_metrics.quality_level
                }
            }
            
        except Exception as e:
            return {"status": "FAILED", "error": str(e)}
    
    def test_6_image_enhancer(self):
        """Test 6: Image Enhancer"""
        logger.info("Running Test 6: Image Enhancer")
        
        if not self.test_image_path:
            return {"status": "FAILED", "error": "No test image available"}
        
        try:
            image = ImageUtils.load_image(self.test_image_path)
            enhancer = AIImageEnhancer(self.config.get("image_enhancer", {}))
            
            # Get quality metrics first
            analyzer = IntelligentQualityAnalyzer()
            quality_metrics = analyzer.analyze_image(image)
            
            enhancement_result = enhancer.enhance_image(image, quality_metrics=quality_metrics)
            
            return {
                "status": "PASSED",
                "enhancement_applied": enhancement_result.enhancement_applied,
                "quality_improvement": enhancement_result.quality_improvement,
                "enhanced_image_shape": enhancement_result.enhanced_image.shape
            }
            
        except Exception as e:
            return {"status": "FAILED", "error": str(e)}
    
    def test_7_text_detector(self):
        """Test 7: Text Detector"""
        logger.info("Running Test 7: Text Detector")
        
        if not self.test_image_path:
            return {"status": "FAILED", "error": "No test image available"}
        
        try:
            image = ImageUtils.load_image(self.test_image_path)
            detector = AdvancedTextDetector(self.config.get("text_detector", {}))
            
            text_regions = detector.detect_text_regions(image)
            
            # Extract bounding box info
            bounding_boxes = []
            for region in text_regions:
                if hasattr(region, 'bbox'):
                    bbox = region.bbox
                    bounding_boxes.append({
                        "x": bbox.x,
                        "y": bbox.y,
                        "width": bbox.width,
                        "height": bbox.height
                    })
            
            return {
                "status": "PASSED",
                "text_regions_detected": len(text_regions),
                "bounding_boxes": bounding_boxes
            }
            
        except Exception as e:
            return {"status": "FAILED", "error": str(e)}
    
    def test_8_base_engine_classes(self):
        """Test 8: Base Engine Classes"""
        logger.info("Running Test 8: Base Engine Classes")
        
        try:
            # Test OCRResult instantiation
            bbox = BoundingBox(x=10, y=20, width=100, height=30)
            text_region = TextRegion(
                text="Test text",
                bbox=bbox,
                confidence=0.95,
                text_type=TextType.PARAGRAPH
            )
            
            ocr_result = OCRResult(
                text="Complete test text",
                confidence=0.90,
                processing_time=1.5,
                regions=[text_region],
                engine_name="test_engine"
            )
            
            return {
                "status": "PASSED",
                "ocr_result_created": True,
                "bbox_created": True,
                "text_region_created": True,
                "test_text": ocr_result.text,
                "test_confidence": ocr_result.confidence
            }
            
        except Exception as e:
            return {"status": "FAILED", "error": str(e)}
    
    def test_9_tesseract_engine(self):
        """Test 9: Tesseract Engine"""
        logger.info("Running Test 9: Tesseract Engine")
        
        if not self.test_image_path:
            return {"status": "FAILED", "error": "No test image available"}
        
        try:
            image = ImageUtils.load_image(self.test_image_path)
            engine = TesseractEngine()
            
            if not engine.is_available():
                return {"status": "FAILED", "error": "Tesseract not available"}
            
            result = engine.extract_text(image)
            
            # Save engine debug info
            self._save_engine_debug_info("tesseract", result)
            
            return {
                "status": "PASSED",
                "engine_available": True,
                "text_extracted": len(result.text) > 0,
                "text_length": len(result.text),
                "confidence": result.confidence,
                "processing_time": result.processing_time,
                "extracted_text": result.text[:200] + "..." if len(result.text) > 200 else result.text
            }
            
        except Exception as e:
            return {"status": "FAILED", "error": str(e)}
    
    def test_10_easyocr_engine(self):
        """Test 10: EasyOCR Engine"""
        logger.info("Running Test 10: EasyOCR Engine")
        
        if not self.test_image_path:
            return {"status": "FAILED", "error": "No test image available"}
        
        try:
            image = ImageUtils.load_image(self.test_image_path)
            engine = EasyOCREngine()
            
            if not engine.is_available():
                return {"status": "FAILED", "error": "EasyOCR not available"}
            
            result = engine.extract_text(image)
            
            # Save engine debug info
            self._save_engine_debug_info("easyocr", result)
            
            return {
                "status": "PASSED",
                "engine_available": True,
                "text_extracted": len(result.text) > 0,
                "text_length": len(result.text),
                "confidence": result.confidence,
                "processing_time": result.processing_time,
                "extracted_text": result.text[:200] + "..." if len(result.text) > 200 else result.text
            }
            
        except Exception as e:
            return {"status": "FAILED", "error": str(e)}
    
    def test_11_paddleocr_engine(self):
        """Test 11: PaddleOCR Engine"""
        logger.info("Running Test 11: PaddleOCR Engine")
        
        if not self.test_image_path:
            return {"status": "FAILED", "error": "No test image available"}
        
        try:
            image = ImageUtils.load_image(self.test_image_path)
            engine = PaddleOCREngine()
            
            if not engine.is_available():
                return {"status": "FAILED", "error": "PaddleOCR not available"}
            
            result = engine.extract_text(image)
            
            # Save engine debug info
            self._save_engine_debug_info("paddleocr", result)
            
            return {
                "status": "PASSED",
                "engine_available": True,
                "text_extracted": len(result.text) > 0,
                "text_length": len(result.text),
                "confidence": result.confidence,
                "processing_time": result.processing_time,
                "extracted_text": result.text[:200] + "..." if len(result.text) > 200 else result.text
            }
            
        except Exception as e:
            return {"status": "FAILED", "error": str(e)}
    
    def test_12_trocr_engine(self):
        """Test 12: TrOCR Engine"""
        logger.info("Running Test 12: TrOCR Engine")
        
        if not self.test_image_path:
            return {"status": "FAILED", "error": "No test image available"}
        
        try:
            image = ImageUtils.load_image(self.test_image_path)
            engine = TrOCREngine()
            
            if not engine.is_available():
                return {"status": "SKIP", "reason": "TrOCR not available (optional)"}
            
            result = engine.extract_text(image)
            
            # Save engine debug info
            self._save_engine_debug_info("trocr", result)
            
            return {
                "status": "PASSED",
                "engine_available": True,
                "text_extracted": len(result.text) > 0,
                "text_length": len(result.text),
                "confidence": result.confidence,
                "processing_time": result.processing_time,
                "extracted_text": result.text[:200] + "..." if len(result.text) > 200 else result.text
            }
            
        except Exception as e:
            return {"status": "FAILED", "error": str(e)}
    
    def test_13_engine_manager(self):
        """Test 13: Engine Manager"""
        logger.info("Running Test 13: Engine Manager")
        
        try:
            manager = EngineManager(self.config.get("engine_manager", {}))
            
            # Register engines
            engines_registered = []
            
            engines_to_test = [
                ("tesseract", TesseractEngine),
                ("paddleocr", PaddleOCREngine),
                ("easyocr", EasyOCREngine),
                ("trocr", TrOCREngine)
            ]
            
            for name, engine_class in engines_to_test:
                try:
                    engine = engine_class()
                    if manager.register_engine(name, engine):
                        engines_registered.append(name)
                except Exception as e:
                    logger.warning(f"Failed to register {name}: {e}")
            
            available_engines = list(manager.get_available_engines().keys())
            initialized_engines = list(manager.get_initialized_engines().keys())
            
            return {
                "status": "PASSED",
                "engines_registered": engines_registered,
                "available_engines": available_engines,
                "initialized_engines": initialized_engines,
                "manager_created": True
            }
            
        except Exception as e:
            return {"status": "FAILED", "error": str(e)}
    
    def test_14_multi_engine_processing(self):
        """Test 14: Multi-Engine Processing"""
        logger.info("Running Test 14: Multi-Engine Processing")
        
        if not self.test_image_path:
            return {"status": "FAILED", "error": "No test image available"}
        
        try:
            image = ImageUtils.load_image(self.test_image_path)
            manager = EngineManager(self.config.get("engine_manager", {}))
            
            # Register available engines
            engines_tested = {}
            engines_to_test = [
                ("tesseract", TesseractEngine),
                ("paddleocr", PaddleOCREngine),
                ("easyocr", EasyOCREngine),
                ("trocr", TrOCREngine)
            ]
            
            for name, engine_class in engines_to_test:
                try:
                    engine = engine_class()
                    if engine.is_available() and manager.register_engine(name, engine):
                        # Test individual engine
                        result = engine.extract_text(image)
                        engines_tested[name] = {
                            "available": True,
                            "text_length": len(result.text),
                            "confidence": result.confidence,
                            "processing_time": result.processing_time,
                            "extracted_text": result.text,
                            "regions": len(result.regions) if hasattr(result, 'regions') and result.regions else 0
                        }
                except Exception as e:
                    engines_tested[name] = {
                        "available": False,
                        "error": str(e)
                    }
            
            # Test best engine selection
            try:
                best_engine = manager.select_best_engine(image, 'default')
                best_results_list = manager.process_with_best_engine(image, 'default')  # This returns List[OCRResult]
                
                # Extract the best single result from the list
                best_result = None
                if best_results_list:
                    # Use the manager's method to select the single best result
                    if hasattr(manager, 'select_best_result'):
                        # Create a dict format that select_best_result expects
                        results_dict = {best_engine: best_results_list}
                        best_result = manager.select_best_result(results_dict)
                    else:
                        # Fallback: just take the first result or the one with highest confidence
                        best_result = max(best_results_list, key=lambda x: x.confidence) if best_results_list else None
                
                # Save all raw OCR results with proper formatting
                self._save_raw_ocr_results(engines_tested, best_result, best_engine)
                
                return {
                    "status": "PASSED",
                    "engines_tested": list(engines_tested.keys()),
                    "engines_available": [name for name, data in engines_tested.items() if data.get("available", False)],
                    "best_engine_selected": best_engine,
                    "best_result_confidence": best_result.confidence if best_result else 0.0,
                    "best_result_text_length": len(best_result.text) if best_result else 0,
                    "best_results_count": len(best_results_list) if best_results_list else 0,
                    "results_saved": True
                }
                
            except Exception as e:
                return {
                    "status": "PARTIAL",
                    "engines_tested": list(engines_tested.keys()),
                    "engines_available": [name for name, data in engines_tested.items() if data.get("available", False)],
                    "manager_error": str(e),
                    "individual_engine_results": engines_tested
                }
            
        except Exception as e:
            return {"status": "FAILED", "error": str(e)}
    
    def _save_engine_debug_info(self, engine_name, result):
        """Save individual engine debug information"""
        try:
            debug_dir = Path("debug")
            debug_dir.mkdir(exist_ok=True)
            
            debug_file = debug_dir / f"{engine_name}_debug.txt"
            with open(debug_file, 'w', encoding='utf-8') as f:
                f.write(f"{engine_name.upper()} ENGINE DEBUG INFO\n")
                f.write("="*50 + "\n")
                f.write(f"Image: {self.test_image_path}\n")
                f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Text Length: {len(result.text)} characters\n")
                f.write(f"Confidence: {result.confidence:.3f}\n")
                f.write(f"Processing Time: {result.processing_time:.3f}s\n")
                
                if hasattr(result, 'regions') and result.regions:
                    f.write(f"Text Regions: {len(result.regions)}\n")
                    f.write("\nRegion Details:\n")
                    for i, region in enumerate(result.regions):
                        f.write(f"  Region {i+1}:\n")
                        f.write(f"    Text: {region.text}\n")
                        f.write(f"    Confidence: {region.confidence:.3f}\n")
                        if hasattr(region, 'bbox'):
                            bbox = region.bbox
                            f.write(f"    BBox: ({bbox.x}, {bbox.y}, {bbox.width}, {bbox.height})\n")
                
                f.write("\nExtracted Text:\n")
                f.write("-"*30 + "\n")
                f.write(result.text)
                f.write("\n" + "-"*30 + "\n")
                
        except Exception as e:
            logger.warning(f"Could not save debug info for {engine_name}: {e}")
    
    def _save_raw_ocr_results(self, engines_tested, best_result, best_engine):
        """Save raw OCR results with proper formatting and word order preservation"""
        try:
            output_file = "debug/raw_ocr_results.txt"
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("="*80 + "\n")
                f.write("RAW OCR RESULTS - NO POSTPROCESSING\n")
                f.write("Text extracted as-is from engines, maintaining original order\n")
                f.write("="*80 + "\n")
                f.write(f"Test Image: {self.test_image_path}\n")
                f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Best Engine Selected: {best_engine}\n")
                f.write("="*80 + "\n\n")
                
                # Save results from each engine with detailed formatting
                for engine_name, data in engines_tested.items():
                    f.write("█" * 60 + "\n")
                    f.write(f"  {engine_name.upper()} ENGINE RESULTS\n")
                    f.write("█" * 60 + "\n")
                    
                    if data.get("available", False):
                        f.write(f"Status: ✅ AVAILABLE\n")
                        f.write(f"Text Length: {data['text_length']} characters\n")
                        f.write(f"Overall Confidence: {data['confidence']:.3f}\n")
                        f.write(f"Processing Time: {data['processing_time']:.3f}s\n")
                        f.write(f"Text Regions Detected: {data.get('regions', 0)}\n")
                        f.write("\n" + "─" * 50 + "\n")
                        f.write("EXTRACTED TEXT (Raw Output - Maintaining Original Order):\n")
                        f.write("─" * 50 + "\n")
                        
                        # Format text to show structure while maintaining order
                        raw_text = data['extracted_text']
                        if raw_text.strip():
                            # Split into lines but preserve the original structure
                            lines = raw_text.split('\n')
                            for line_num, line in enumerate(lines, 1):
                                if line.strip():  # Only show non-empty lines
                                    f.write(f"{line_num:2d}| {line}\n")
                                else:
                                    f.write(f"{line_num:2d}| [EMPTY LINE]\n")
                        else:
                            f.write("[NO TEXT DETECTED]\n")
                            
                        # f.write("─" * 50 + "\n")
                        # f.write("CONTINUOUS TEXT (As single block):\n")
                        # f.write("─" * 20 + "\n")
                        # f.write(raw_text if raw_text.strip() else "[NO TEXT DETECTED]")
                        # f.write("\n" + "─" * 20 + "\n\n")
                        
                    else:
                        f.write(f"Status: ❌ NOT AVAILABLE\n")
                        f.write(f"Error: {data.get('error', 'Unknown error')}\n\n")
                
                # Save best result section with proper type checking
                f.write("█" * 60 + "\n")
                f.write("  BEST RESULT (FINAL RAW OUTPUT)\n")
                f.write("█" * 60 + "\n")
                
                # FIX: Handle case where best_result might be a list or None
                if best_result:
                    # If best_result is a list, take the first element
                    if isinstance(best_result, list):
                        if len(best_result) > 0:
                            best_result = best_result[0]
                        else:
                            best_result = None
                    
                    # If we still have a valid result
                    if best_result and hasattr(best_result, 'confidence'):
                        f.write(f"Selected Engine: {best_engine}\n")
                        f.write(f"Final Confidence: {best_result.confidence:.3f}\n")
                        f.write(f"Processing Time: {best_result.processing_time:.3f}s\n")
                        f.write(f"Total Characters: {len(best_result.text)}\n")
                        
                        if hasattr(best_result, 'regions') and best_result.regions:
                            f.write(f"Text Regions: {len(best_result.regions)}\n")
                        
                        f.write("\n" + "─" * 50 + "\n")
                        f.write("FINAL TEXT (As extracted, preserving word order):\n")
                        f.write("─" * 50 + "\n")
                        
                        # Show the final text with line numbers for reference
                        if best_result.text.strip():
                            lines = best_result.text.split('\n')
                            for line_num, line in enumerate(lines, 1):
                                if line.strip():
                                    f.write(f"{line_num:2d}| {line}\n")
                                else:
                                    f.write(f"{line_num:2d}| [EMPTY LINE]\n")
                        else:
                            f.write("[NO TEXT DETECTED]\n")
                        
                        f.write("─" * 50 + "\n")
                        f.write("FINAL CONTINUOUS TEXT:\n")
                        f.write("─" * 25 + "\n")
                        f.write(best_result.text if best_result.text.strip() else "[NO TEXT DETECTED]")
                        f.write("\n" + "─" * 25 + "\n")
                        
                        # Additional analysis
                        f.write("\n" + "─" * 30 + "\n")
                        f.write("TEXT ANALYSIS:\n")
                        f.write("─" * 30 + "\n")
                        words = best_result.text.split()
                        f.write(f"Word Count: {len(words)}\n")
                        f.write(f"Line Count: {len(best_result.text.split(chr(10)))}\n")
                        f.write(f"Character Count (with spaces): {len(best_result.text)}\n")
                        f.write(f"Character Count (without spaces): {len(best_result.text.replace(' ', ''))}\n")
                        
                    else:
                        f.write("❌ BEST RESULT DATA INVALID OR MISSING\n")
                        f.write(f"Result Type: {type(best_result)}\n")
                        if hasattr(best_result, '__dict__'):
                            f.write(f"Available attributes: {list(best_result.__dict__.keys())}\n")
                else:
                    f.write("❌ NO VALID RESULT FROM ANY ENGINE\n")
                
                f.write("\n" + "="*80 + "\n")
                f.write("END OF RAW OCR RESULTS\n")
                f.write("="*80 + "\n")
            
            logger.info(f"Raw OCR results saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Error saving raw OCR results: {e}")
            # Log additional debug info
            logger.error(f"best_result type: {type(best_result)}")
            if best_result:
                logger.error(f"best_result contents: {str(best_result)[:200]}...")
    
    def run_all_tests(self):
        """Run all tests (1-14)"""
        tests = [
            ("Test 1: Image Loading", self.test_1_image_loading),
            ("Test 2: Configuration System", self.test_2_configuration_system),
            ("Test 3: Logger System", self.test_3_logger_system),
            ("Test 4: Image Utilities", self.test_4_image_utilities),
            ("Test 5: Quality Analyzer", self.test_5_quality_analyzer),
            ("Test 6: Image Enhancer", self.test_6_image_enhancer),
            ("Test 7: Text Detector", self.test_7_text_detector),
            ("Test 8: Base Engine Classes", self.test_8_base_engine_classes),
            ("Test 9: Tesseract Engine", self.test_9_tesseract_engine),
            ("Test 10: EasyOCR Engine", self.test_10_easyocr_engine),
            ("Test 11: PaddleOCR Engine", self.test_11_paddleocr_engine),
            ("Test 12: TrOCR Engine", self.test_12_trocr_engine),
            ("Test 13: Engine Manager", self.test_13_engine_manager),
            ("Test 14: Multi-Engine Processing", self.test_14_multi_engine_processing),
        ]
        
        print("\n" + "="*80)
        print("OCR PIPELINE TESTING - TESTS 1-14 (NO POSTPROCESSING)")
        print("="*80)
        
        total_tests = len(tests)
        passed = 0
        failed = 0
        skipped = 0
        
        for test_name, test_func in tests:
            print(f"\n{test_name}")
            print("-" * len(test_name))
            
            try:
                result = test_func()
                self.test_results[test_name] = result
                
                status = result["status"]
                if status == "PASSED":
                    print("✅ PASSED")
                    passed += 1
                    
                    # Print key details
                    for key, value in result.items():
                        if key != "status" and not key.startswith("extracted_text"):
                            if isinstance(value, (dict, list)):
                                if len(str(value)) > 100:
                                    print(f"   {key}: [Complex data - see detailed results]")
                                else:
                                    print(f"   {key}: {json.dumps(value, indent=6)}")
                            else:
                                print(f"   {key}: {value}")
                                
                elif status == "FAILED":
                    print("❌ FAILED")
                    print(f"   Error: {result.get('error', 'Unknown error')}")
                    failed += 1
                    
                elif status == "SKIP":
                    print("⏭️  SKIPPED")
                    print(f"   Reason: {result.get('reason', 'Unknown reason')}")
                    skipped += 1
                    
                elif status == "PARTIAL":
                    print("⚠️  PARTIAL SUCCESS")
                    print(f"   Some components working, check details")
                    passed += 0.5  # Count as half success
                    
            except Exception as e:
                print(f"❌ FAILED - Exception: {e}")
                self.test_results[test_name] = {"status": "FAILED", "error": str(e)}
                failed += 1
        
        # Print summary
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        print(f"Total Tests: {total_tests}")
        print(f"✅ Passed: {int(passed)}")
        print(f"⚠️  Partial: {int((passed % 1) * 2)}")
        print(f"❌ Failed: {failed}")
        print(f"⏭️  Skipped: {skipped}")
        success_rate = (passed/(total_tests-skipped)*100) if (total_tests-skipped) > 0 else 0
        print(f"Success Rate: {success_rate:.1f}%")
        
        # Print pipeline status
        print("\n" + "─"*50)
        print("PIPELINE STATUS")
        print("─"*50)
        
        if self.test_image_path:
            print(f"📄 Test Image: {self.test_image_path}")
        
        # Check critical components
        critical_tests = ["Test 1: Image Loading", "Test 9: Tesseract Engine", 
                         "Test 10: EasyOCR Engine", "Test 11: PaddleOCR Engine"]
        critical_passed = sum(1 for test in critical_tests 
                            if test in self.test_results and 
                            self.test_results[test]["status"] == "PASSED")
        
        print(f"🔧 Critical Components: {critical_passed}/{len(critical_tests)} working")
        
        # Engine availability
        engines_available = []
        for engine in ["tesseract", "easyocr", "paddleocr", "trocr"]:
            test_key = f"Test {9 + ['tesseract', 'easyocr', 'paddleocr', 'trocr'].index(engine)}: {engine.title()}{'OCR' if engine != 'tesseract' else ''} Engine"
            if test_key in self.test_results:
                if self.test_results[test_key]["status"] == "PASSED":
                    engines_available.append(engine)
        
        print(f"🚀 OCR Engines Available: {engines_available}")
        
        # Save detailed results
        self._save_test_results()
        
        # Final recommendations
        print("\n" + "─"*50)
        print("RECOMMENDATIONS")
        print("─"*50)
        
        if failed == 0:
            print("✅ All tests passed! Pipeline is ready for integration testing.")
        elif critical_passed >= 2:
            print("⚠️  Core functionality working. Address failed components before production.")
        else:
            print("❌ Critical issues found. Fix fundamental components first.")
        
        return self.test_results
    
    def _save_test_results(self):
        """Save detailed test results to file"""
        try:
            output_file = "debug/pipeline_test_results_1_to_14.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
                    "test_scope": "Tests 1-14 (No Postprocessing)",
                    "test_image": self.test_image_path,
                    "config_loaded": self.config is not None,
                    "results": self.test_results,
                    "summary": {
                        "total_tests": len(self.test_results),
                        "passed": sum(1 for r in self.test_results.values() if r["status"] == "PASSED"),
                        "failed": sum(1 for r in self.test_results.values() if r["status"] == "FAILED"),
                        "skipped": sum(1 for r in self.test_results.values() if r["status"] == "SKIP"),
                        "partial": sum(1 for r in self.test_results.values() if r["status"] == "PARTIAL")
                    }
                }, f, indent=2, default=str)
            
            logger.info(f"Detailed test results saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Error saving test results: {e}")

def main():
    """Main test runner"""
    print("Starting OCR Pipeline Test Suite")
    print("Testing components 1-14 without postprocessing")
    print("This will test your OCR engines and preprocessing pipeline")
    
    tester = PipelineTestRunner()
    results = tester.run_all_tests()
    
    print(f"\n{'='*60}")
    print("TEST COMPLETED")
    print("="*60)
    print(f"📊 Check 'pipeline_test_results_1_to_14.json' for detailed results")
    print(f"📄 Check 'raw_ocr_results.txt' for raw OCR text output")
    print(f"🔍 Check 'debug/' folder for individual engine debug files")
    print("="*60)

if __name__ == "__main__":
    main()