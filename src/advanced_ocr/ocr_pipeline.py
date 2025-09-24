import cv2
import numpy as np
import time
from pathlib import Path
from typing import Union, Optional
from typing import List, Dict
from .types import OCRResult, ProcessingOptions, QualityMetrics, ProcessingStrategy
from .exceptions import OCRLibraryError, EngineNotAvailableError
from .core.engine_manager import EngineManager
from .preprocessing.quality_analyzer import IntelligentQualityAnalyzer
from .preprocessing.image_enhancer import AIImageEnhancer
from .utils.config import load_config
from .utils.logger import setup_logger
from .utils.image_utils import ImageUtils

class OCRLibrary:
    """Main OCR Library class - primary interface for users"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize OCR Library with engines and preprocessing components"""
        self.logger = setup_logger(self.__class__.__name__)
        self.config = self._load_config(config_path)
        
        # Initialize core components
        self.quality_analyzer = IntelligentQualityAnalyzer(
            self.config.get("quality_analyzer", {})
        )
        self.image_enhancer = AIImageEnhancer(
            self.config.get("image_enhancer", {})
        )
        self.engine_manager = EngineManager(
            self.config.get("engine_manager", {})
        )
        
        # Initialize and register engines
        self._initialize_engines()
        
        self.logger.info("OCR Library initialized successfully")
    
    def extract_text(self, 
                    image_input: Union[str, Path, np.ndarray], 
                    options: Optional[ProcessingOptions] = None) -> OCRResult:
        """Extract text from a single image"""
        start_time = time.time()
        
        try:
            # Set default options
            if options is None:
                options = ProcessingOptions()
            
            # Load and validate image
            image = self._load_image(image_input)
            
            # Analyze image quality
            quality_metrics = self.quality_analyzer.analyze_image(image)
            
            # Determine processing strategy
            strategy = self._determine_strategy(quality_metrics, options)
            
            # Apply preprocessing based on strategy
            processed_image = self._preprocess_image(image, quality_metrics, strategy, options)
            
            # Extract text using optimal engine(s) - this now returns OCRResult
            ocr_result = self._extract_with_engines(processed_image, options, strategy)
            
            # Create final result - ocr_result is already an OCRResult from engine
            total_time = time.time() - start_time
            
            # Update the existing result with pipeline metadata
            ocr_result.processing_time = total_time
            ocr_result.quality_metrics = quality_metrics
            ocr_result.strategy_used = strategy
            ocr_result.metadata.update({
                'image_shape': image.shape,
                'preprocessing_applied': strategy.value,
                'total_pipeline_time': total_time
            })
            
            return ocr_result
            
        except Exception as e:
            self.logger.error(f"OCR processing failed: {e}")
            raise OCRLibraryError(f"Failed to extract text: {str(e)}") from e

    def extract_text_batch(self, 
                        image_paths: List[Union[str, Path]], 
                        options: Optional[ProcessingOptions] = None) -> List[OCRResult]:
        """Extract text from multiple images sequentially"""
        results = []
        for image_path in image_paths:
            try:
                result = self.extract_text(image_path, options)
                results.append(result)
            except Exception as e:
                # Create error result - use proper QualityMetrics constructor
                error_metrics = QualityMetrics(
                    overall_score=0.0,
                    sharpness_score=0.0,
                    noise_level=1.0,
                    contrast_score=0.0,
                    brightness_score=0.0,
                    needs_enhancement=False
                )
                
                error_result = OCRResult(
                    text="",
                    confidence=0.0,
                    processing_time=0.0,
                    engine_used="none",
                    quality_metrics=error_metrics,
                    strategy_used=ProcessingStrategy.MINIMAL,
                    metadata={'error': str(e), 'image_path': str(image_path)}
                )
                results.append(error_result)
                self.logger.error(f"Failed to process {image_path}: {e}")
        
        return results
        
    def get_available_engines(self) -> List[str]:
        """Get list of available OCR engines"""
        return list(self.engine_manager.get_available_engines().keys())
    
    def get_engine_info(self) -> Dict[str, Dict]:
        """Get detailed information about available engines"""
        engines = {}
        for name, engine in self.engine_manager.get_available_engines().items():
            engines[name] = {
                'available': engine.is_available(),
                'name': name,
                'initialized': name in self.engine_manager.get_initialized_engines()
            }
        return engines
    
    # Private methods
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load configuration from file or use defaults"""
        if config_path and Path(config_path).exists():
            return load_config(config_path)
        
        # Default configuration based on test results
        return {
            "engines": {
                "paddleocr": {"enabled": True, "priority": 1, "confidence_threshold": 0.7},
                "easyocr": {"enabled": True, "priority": 2, "confidence_threshold": 0.6},
                "tesseract": {"enabled": True, "priority": 3, "confidence_threshold": 0.5},
                "trocr": {"enabled": True, "priority": 4, "confidence_threshold": 0.8}
            },
            "preprocessing": {
                "quality_analysis": True,
                "enhancement": True
            }
        }
    
    def _initialize_engines(self):
        """Initialize and register OCR engines"""
        from .engines.paddleocr import PaddleOCR
        from .engines.easyocr import EasyOCR
        from .engines.tesseract import Tesseract
        from .engines.trocr import TrOCR
        
        engine_classes = {
            'paddleocr': PaddleOCR,
            'easyocr': EasyOCR,
            'tesseract': Tesseract,
            'trocr': TrOCR
        }
        
        for name, engine_class in engine_classes.items():
            try:
                if self.config["engines"].get(name, {}).get("enabled", False):
                    engine = engine_class()
                    if engine.is_available():
                        self.engine_manager.register_engine(name, engine)
                        self.logger.info(f"Registered {name} engine")
                    else:
                        self.logger.warning(f"{name} engine not available")
            except Exception as e:
                self.logger.error(f"Failed to initialize {name}: {e}")
    
    def _load_image(self, image_input: Union[str, Path, np.ndarray]) -> np.ndarray:
        """Load image from file path or validate numpy array"""
        if isinstance(image_input, (str, Path)):
            image = ImageUtils.load_image(str(image_input))
            if image is None:
                raise OCRLibraryError(f"Could not load image: {image_input}")
            return image
        elif isinstance(image_input, np.ndarray):
            return image_input
        else:
            raise OCRLibraryError(f"Invalid image input type: {type(image_input)}")
    
    def _determine_strategy(self, quality_metrics: QualityMetrics, 
                          options: ProcessingOptions) -> ProcessingStrategy:
        """Determine processing strategy based on quality and options"""
        if options.strategy:
            return options.strategy
        
        # Auto-determine based on quality
        if quality_metrics.overall_score > 0.8:
            return ProcessingStrategy.MINIMAL
        elif quality_metrics.overall_score > 0.5:
            return ProcessingStrategy.BALANCED  
        else:
            return ProcessingStrategy.ENHANCED
    
    def _preprocess_image(self, image: np.ndarray, quality_metrics: QualityMetrics,
                        strategy: ProcessingStrategy, options: ProcessingOptions) -> np.ndarray:
        """Apply preprocessing based on strategy and options"""
        processed_image = image.copy()
        
        if options.enhance_image and quality_metrics.needs_enhancement:
            if strategy in [ProcessingStrategy.BALANCED, ProcessingStrategy.ENHANCED]:
                # FIX: Correct parameter order - image, strategy, quality_metrics
                enhancement_result = self.image_enhancer.enhance_image(
                    processed_image, 
                    None,  # Let enhancer determine strategy
                    quality_metrics
                )
                processed_image = enhancement_result.enhanced_image
                self.logger.debug(f"Applied {enhancement_result.enhancement_applied} enhancement")
        
        return processed_image

    
    def _extract_with_engines(self, image: np.ndarray, options: ProcessingOptions,
                        strategy: ProcessingStrategy) -> OCRResult:
        """Extract text using the best available engine(s) - simplified logic"""
        
        # Get available engines
        available_engines = self.engine_manager.get_initialized_engines()
        
        if not available_engines:
            raise EngineNotAvailableError("No engines are available")
        
        # Determine which engines to use
        target_engines = options.engines or list(available_engines.keys())
        engines_to_use = [eng for eng in target_engines if eng in available_engines]
        
        if not engines_to_use:
            raise EngineNotAvailableError("No requested engines are available")
        
        # Single engine processing (most common case)
        if len(engines_to_use) == 1:
            engine_name = engines_to_use[0]
            result = self.engine_manager.extract_with_engine(image, engine_name)
            
            if result.confidence >= options.min_confidence:
                return result
            else:
                raise OCRLibraryError(f"Result confidence ({result.confidence:.3f}) below threshold ({options.min_confidence})")
        
        # Multi-engine processing
        else:
            # Try engines in priority order first
            best_engine = self.engine_manager.select_best_engine()
            if best_engine in engines_to_use:
                result = self.engine_manager.extract_with_engine(image, best_engine)
                
                # Early termination for high confidence results
                if (options.early_termination and 
                    result.confidence >= options.early_termination_threshold):
                    return result
                
                # If result is good enough, return it
                if result.confidence >= options.min_confidence:
                    return result
            
            # If primary engine didn't work well, try multiple engines
            results = self.engine_manager.extract_with_multiple_engines(
                image, engines_to_use, use_parallel=True
            )
            
            # Select best result
            best_result = self.engine_manager.select_best_result(results)
            
            if best_result is None:
                raise OCRLibraryError("No engine produced acceptable results")
            
            if best_result.confidence >= options.min_confidence:
                return best_result
            else:
                raise OCRLibraryError(f"Best result confidence ({best_result.confidence:.3f}) below threshold ({options.min_confidence})")

    def get_available_engines(self) -> List[str]:
        """Get list of available OCR engines"""
        return list(self.engine_manager.get_initialized_engines().keys())

    def get_engine_info(self) -> Dict[str, Dict]:
        """Get detailed information about available engines"""
        return self.engine_manager.get_engine_info()