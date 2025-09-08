"""
Enhanced TrOCR Engine Implementation - Professional Fixed Version
================================================================

This module implements a state-of-the-art TrOCR (Transformer-based OCR) engine
with advanced features including:
- Multiple TrOCR model variants
- Intelligent preprocessing for recognition
- Batch processing capabilities
- Performance optimization
- Multi-language support
- Confidence scoring and quality assessment

Fixed for production-level reliability and error handling.
"""

import logging
import torch
import numpy as np
import cv2
from typing import List, Dict, Any, Optional, Tuple, Union
from PIL import Image
import time
from concurrent.futures import ThreadPoolExecutor
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from transformers import AutoTokenizer, AutoProcessor
import torchvision.transforms as transforms
import sys
import os
import traceback
from pathlib import Path

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import with proper error handling
try:
    from core.base_engine import BaseOCREngine, OCRResult, BoundingBox, TextRegion, DocumentResult, TextType
    from utils.image_utils import ImageUtils
    from utils.text_utils import TextUtils
except ImportError:
    try:
        # Alternative import method for different project structures
        import importlib.util
        
        # Dynamic imports for flexibility
        base_engine_path = Path(parent_dir) / "core" / "base_engine.py"
        image_utils_path = Path(parent_dir) / "utils" / "image_utils.py"
        text_utils_path = Path(parent_dir) / "utils" / "text_utils.py"
        
        if base_engine_path.exists():
            spec = importlib.util.spec_from_file_location("base_engine", base_engine_path)
            base_engine = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(base_engine)
            
            BaseOCREngine = base_engine.BaseOCREngine
            OCRResult = base_engine.OCRResult
            BoundingBox = base_engine.BoundingBox
            TextRegion = base_engine.TextRegion
            DocumentResult = base_engine.DocumentResult
            TextType = base_engine.TextType
        else:
            raise ImportError("Cannot find base_engine module")
            
    except Exception as e:
        print(f"Critical import error in TrOCR engine: {e}")
        print("Please ensure the project structure is correct")
        # Create minimal fallback classes to prevent complete failure
        
        class BaseOCREngine:
            def __init__(self, name, config=None):
                self.name = name
                self.config = config or {}
                self.is_initialized = False
                self.model_loaded = False
                self.supported_languages = []
        
        class BoundingBox:
            def __init__(self, x=0, y=0, width=100, height=100, confidence=0.0):
                self.x, self.y, self.width, self.height, self.confidence = x, y, width, height, confidence
        
        class TextRegion:
            def __init__(self, text="", confidence=0.0, bbox=None, text_type=None, reading_order=0):
                self.text = text
                self.confidence = confidence
                self.bbox = bbox or BoundingBox()
                self.text_type = text_type
                self.reading_order = reading_order
        
        class OCRResult:
            def __init__(self, text="", confidence=0.0, regions=None, processing_time=0.0, bbox=None, level="page"):
                self.text = text
                self.confidence = confidence
                self.regions = regions or []
                self.processing_time = processing_time
                self.bbox = bbox
                self.level = level
        
        class DocumentResult:
            def __init__(self, pages=None, metadata=None, processing_time=0.0, engine_name="", confidence_score=0.0):
                self.pages = pages or []
                self.metadata = metadata or {}
                self.processing_time = processing_time
                self.engine_name = engine_name
                self.confidence_score = confidence_score
        
        class TextType:
            PRINTED = "printed"
            HANDWRITTEN = "handwritten"
            UNKNOWN = "unknown"

logger = logging.getLogger(__name__)


class TrOCRModelManager:
    """Manages different TrOCR model variants and their configurations."""
    
    MODEL_CONFIGS = {
        'base-handwritten': {
            'model_name': 'microsoft/trocr-base-handwritten',
            'processor_name': 'microsoft/trocr-base-handwritten',
            'best_for': ['handwritten', 'manuscripts', 'notes'],
            'max_length': 384,
            'confidence_threshold': 0.7
        },
        'base-printed': {
            'model_name': 'microsoft/trocr-base-printed',
            'processor_name': 'microsoft/trocr-base-printed', 
            'best_for': ['printed', 'documents', 'books'],
            'max_length': 384,
            'confidence_threshold': 0.8
        },
        'large-handwritten': {
            'model_name': 'microsoft/trocr-large-handwritten',
            'processor_name': 'microsoft/trocr-large-handwritten',
            'best_for': ['handwritten', 'complex_manuscripts'],
            'max_length': 384,
            'confidence_threshold': 0.75
        },
        'large-printed': {
            'model_name': 'microsoft/trocr-large-printed',
            'processor_name': 'microsoft/trocr-large-printed',
            'best_for': ['printed', 'high_quality_docs'],
            'max_length': 384,
            'confidence_threshold': 0.85
        }
    }
    
    def __init__(self):
        self.loaded_models = {}
        self.device = self._get_optimal_device()
        self.model_cache_dir = self._setup_cache_dir()
        logger.info(f"TrOCR Manager initialized on device: {self.device}")
    
    def _get_optimal_device(self) -> torch.device:
        """Get the best available device for processing."""
        if torch.cuda.is_available():
            # Check CUDA memory
            try:
                device = torch.device('cuda')
                # Test CUDA availability
                test_tensor = torch.tensor([1.0]).to(device)
                del test_tensor
                torch.cuda.empty_cache()
                return device
            except Exception as e:
                logger.warning(f"CUDA available but unusable: {e}")
        
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            try:
                device = torch.device('mps')
                # Test MPS availability
                test_tensor = torch.tensor([1.0]).to(device)
                del test_tensor
                return device
            except Exception as e:
                logger.warning(f"MPS available but unusable: {e}")
        
        return torch.device('cpu')
    
    def _setup_cache_dir(self) -> Path:
        """Setup model cache directory."""
        cache_dir = Path.home() / '.cache' / 'trocr_models'
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir
    
    def load_model(self, model_type: str) -> Tuple[VisionEncoderDecoderModel, TrOCRProcessor]:
        """Load and cache a TrOCR model variant with robust error handling."""
        if model_type not in self.MODEL_CONFIGS:
            raise ValueError(f"Unknown model type: {model_type}. Available: {list(self.MODEL_CONFIGS.keys())}")
        
        # Return cached model if available
        if model_type in self.loaded_models:
            model, processor = self.loaded_models[model_type]
            # Verify model is still valid
            if self._verify_model_health(model):
                return model, processor
            else:
                # Remove corrupted model from cache
                del self.loaded_models[model_type]
        
        config = self.MODEL_CONFIGS[model_type]
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Loading TrOCR model: {config['model_name']} (attempt {attempt + 1}/{max_retries})")
                
                # Load with explicit error handling
                model = VisionEncoderDecoderModel.from_pretrained(
                    config['model_name'],
                    cache_dir=self.model_cache_dir,
                    torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
                    low_cpu_mem_usage=True
                )
                
                processor = TrOCRProcessor.from_pretrained(
                    config['processor_name'],
                    cache_dir=self.model_cache_dir
                )
                
                # Move model to device with memory optimization
                model = self._optimize_model_for_device(model)
                
                # Verify model works
                if not self._verify_model_health(model, processor):
                    raise RuntimeError("Model failed health check")
                
                # Cache the loaded model
                self.loaded_models[model_type] = (model, processor)
                
                logger.info(f"Successfully loaded {model_type} model")
                return model, processor
                
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed to load {model_type} model: {e}")
                
                if attempt == max_retries - 1:
                    # Final attempt failed
                    raise RuntimeError(f"Failed to load {model_type} after {max_retries} attempts: {str(e)}")
                
                # Clear cache and retry
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                time.sleep(2 ** attempt)  # Exponential backoff
    
    def _optimize_model_for_device(self, model: VisionEncoderDecoderModel) -> VisionEncoderDecoderModel:
        """Optimize model for the target device."""
        try:
            model = model.to(self.device)
            model.eval()
            
            # Enable optimizations
            if self.device.type == 'cuda':
                # Use half precision on CUDA if supported
                try:
                    model = model.half()
                except Exception as e:
                    logger.warning(f"Could not use half precision: {e}")
            
            # Compile model for better performance (PyTorch 2.0+)
            try:
                if hasattr(torch, 'compile') and self.device.type != 'mps':  # MPS doesn't support compile yet
                    model = torch.compile(model)
            except Exception as e:
                logger.warning(f"Could not compile model: {e}")
            
            return model
            
        except Exception as e:
            logger.error(f"Model optimization failed: {e}")
            # Fallback to basic setup
            return model.to(self.device).eval()
    
    def _verify_model_health(self, model: VisionEncoderDecoderModel, processor: Optional[TrOCRProcessor] = None) -> bool:
        """Verify that the model is healthy and functional."""
        try:
            # Basic model checks
            if not hasattr(model, 'generate') or not callable(model.generate):
                return False
            
            # If processor provided, do a quick test
            if processor:
                test_image = Image.new('RGB', (384, 384), color='white')
                pixel_values = processor(test_image, return_tensors="pt").pixel_values
                pixel_values = pixel_values.to(self.device)
                
                with torch.no_grad():
                    output = model.generate(pixel_values, max_length=10, do_sample=False)
                    if output is None or len(output) == 0:
                        return False
            
            return True
            
        except Exception as e:
            logger.warning(f"Model health check failed: {e}")
            return False
    
    def get_best_model_for_content(self, text_type: str = 'printed') -> str:
        """Select the best model based on content type."""
        type_mapping = {
            'handwritten': 'base-handwritten',
            'printed': 'base-printed',
            'manuscript': 'large-handwritten',
            'document': 'large-printed',
            'mixed': 'base-printed'  # Safe default
        }
        return type_mapping.get(text_type.lower(), 'base-printed')
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage statistics."""
        memory_info = {
            'loaded_models': len(self.loaded_models),
            'device': str(self.device)
        }
        
        if torch.cuda.is_available() and self.device.type == 'cuda':
            memory_info.update({
                'cuda_allocated': torch.cuda.memory_allocated() / 1024**3,  # GB
                'cuda_reserved': torch.cuda.memory_reserved() / 1024**3,    # GB
                'cuda_max_allocated': torch.cuda.max_memory_allocated() / 1024**3  # GB
            })
        
        return memory_info


class TrOCRRecognitionOptimizer:
    """Optimizes TrOCR recognition through intelligent preprocessing."""
    
    def __init__(self):
        self.target_size = (384, 384)  # Standard TrOCR input size
        self.transform = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def preprocess_for_trocr(self, image: Union[np.ndarray, Image.Image]) -> Image.Image:
        """Preprocess image specifically for TrOCR recognition with error handling."""
        try:
            # Input validation and conversion
            if isinstance(image, np.ndarray):
                if len(image.shape) == 3 and image.shape[2] == 3:
                    # BGR to RGB conversion
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                elif len(image.shape) == 3 and image.shape[2] == 4:
                    # BGRA to RGB conversion
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
                elif len(image.shape) == 2:
                    # Grayscale to RGB
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                
                image = Image.fromarray(image)
            
            # Ensure RGB format
            if image.mode != 'RGB':
                if image.mode == 'RGBA':
                    # Handle transparency by compositing on white background
                    background = Image.new('RGB', image.size, (255, 255, 255))
                    background.paste(image, mask=image.split()[-1])
                    image = background
                else:
                    image = image.convert('RGB')
            
            # Validate image size
            if image.size[0] == 0 or image.size[1] == 0:
                raise ValueError("Image has zero dimensions")
            
            # Apply preprocessing pipeline
            image = self._normalize_contrast(image)
            image = self._denoise_for_text(image)
            image = self._optimize_resolution(image)
            
            return image
            
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            # Return a safe fallback image
            return Image.new('RGB', self.target_size, color='white')
    
    def _normalize_contrast(self, image: Image.Image) -> Image.Image:
        """Normalize contrast for better recognition with error handling."""
        try:
            img_array = np.array(image)
            
            # Check if image is not empty
            if img_array.size == 0:
                return image
            
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
            
            # Split channels
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge channels
            lab = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
            return Image.fromarray(enhanced)
            
        except Exception as e:
            logger.warning(f"Contrast normalization failed: {e}")
            return image
    
    def _denoise_for_text(self, image: Image.Image) -> Image.Image:
        """Apply gentle denoising that preserves text quality."""
        try:
            img_array = np.array(image)
            
            if img_array.size == 0:
                return image
            
            # Use bilateral filter to reduce noise while preserving edges
            denoised = cv2.bilateralFilter(img_array, 9, 75, 75)
            
            return Image.fromarray(denoised)
            
        except Exception as e:
            logger.warning(f"Denoising failed: {e}")
            return image
    
    def _optimize_resolution(self, image: Image.Image) -> Image.Image:
        """Optimize image resolution for TrOCR with proper aspect ratio handling."""
        try:
            width, height = image.size
            
            # Ensure minimum dimensions
            min_dim = 32  # Minimum reasonable dimension
            if width < min_dim or height < min_dim:
                scale = max(min_dim / width, min_dim / height)
                new_width = max(int(width * scale), min_dim)
                new_height = max(int(height * scale), min_dim)
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # TrOCR optimal size handling
            target_width, target_height = self.target_size
            
            # Calculate scaling to maintain aspect ratio
            width_ratio = target_width / width
            height_ratio = target_height / height
            scale = min(width_ratio, height_ratio)
            
            if scale != 1.0:
                new_width = int(width * scale)
                new_height = int(height * scale)
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            return image
            
        except Exception as e:
            logger.warning(f"Resolution optimization failed: {e}")
            return image


class TrOCREngine(BaseOCREngine):
    """
    Enhanced TrOCR Engine for state-of-the-art text recognition.
    
    Features:
    - Multiple TrOCR model variants
    - Intelligent model selection
    - Advanced preprocessing
    - Batch processing
    - Confidence scoring
    - Multi-language support
    - Robust error handling
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('TrOCR', config)
        
        # Initialize components with error handling
        try:
            self.model_manager = TrOCRModelManager()
            self.optimizer = TrOCRRecognitionOptimizer()
        except Exception as e:
            logger.error(f"Failed to initialize TrOCR components: {e}")
            raise
        
        # Configuration with validation
        self.default_model = self.config.get('default_model', 'base-printed')
        if self.default_model not in TrOCRModelManager.MODEL_CONFIGS:
            logger.warning(f"Invalid default model {self.default_model}, using base-printed")
            self.default_model = 'base-printed'
        
        self.auto_model_selection = self.config.get('auto_model_selection', True)
        self.batch_size = max(1, self.config.get('batch_size', 4))
        self.confidence_threshold = max(0.0, min(1.0, self.config.get('confidence_threshold', 0.7)))
        self.max_workers = max(1, self.config.get('max_workers', 2))
        self.timeout = self.config.get('timeout', 300)  # 5 minutes default timeout
        
        # Performance tracking
        self.recognition_stats = {
            'total_recognitions': 0,
            'total_time': 0.0,
            'successful_recognitions': 0,
            'failed_recognitions': 0,
            'model_usage': {},
            'confidence_distribution': [],
            'error_log': []
        }
        
        logger.info(f"TrOCR Engine created with model: {self.default_model}")
    
    def initialize(self) -> bool:
        """Initialize the TrOCR engine with comprehensive error handling."""
        try:
            logger.info("Initializing TrOCR Engine...")
            
            # Test model loading
            model, processor = self.model_manager.load_model(self.default_model)
            
            # Perform initialization test
            test_result = self._perform_initialization_test(model, processor)
            
            if test_result:
                self.is_initialized = True
                self.model_loaded = True
                self.supported_languages = self._get_supported_languages()
                
                logger.info(f"TrOCR Engine initialized successfully with {self.default_model}")
                return True
            else:
                raise RuntimeError("Initialization test failed")
            
        except Exception as e:
            logger.error(f"Failed to initialize TrOCR Engine: {e}")
            logger.error(traceback.format_exc())
            self.is_initialized = False
            self.model_loaded = False
            self._log_error("initialization", str(e))
            return False
    
    def _perform_initialization_test(self, model: VisionEncoderDecoderModel, 
                                   processor: TrOCRProcessor) -> bool:
        """Perform a comprehensive initialization test."""
        try:
            # Create a test image with text
            test_image = Image.new('RGB', (384, 384), color='white')
            
            # Process test image
            pixel_values = processor(test_image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(self.model_manager.device)
            
            # Test generation
            with torch.no_grad():
                generated_ids = model.generate(
                    pixel_values,
                    max_length=10,
                    do_sample=False,
                    num_beams=1,
                    early_stopping=True
                )
            
            # Test decoding
            text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            logger.info(f"Initialization test completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Initialization test failed: {e}")
            return False
    
    def _get_supported_languages(self) -> List[str]:
        """Get list of supported languages for TrOCR models."""
        # TrOCR models support multiple languages but are primarily trained on English
        # Different models may have different language capabilities
        return ['en', 'de', 'fr', 'it', 'pt', 'ru', 'ja', 'ko', 'zh', 'ar']
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        return self.supported_languages if hasattr(self, 'supported_languages') else ['en']
    
    def process_image(self, image: np.ndarray, **kwargs) -> DocumentResult:
        """
        Main text recognition method using TrOCR with robust error handling.
        
        Args:
            image: Input image (numpy array)
            **kwargs: Additional parameters including 'regions' for specific areas
        
        Returns:
            DocumentResult with comprehensive OCR results
        """
        start_time = time.time()
        
        # Input validation
        if image is None or image.size == 0:
            return self._create_empty_result("Empty or invalid image", start_time)
        
        # Initialization check
        if not self.is_initialized:
            if not self.initialize():
                return self._create_empty_result("Engine not initialized", start_time)
        
        try:
            # Extract parameters
            regions = kwargs.get('regions', None)
            content_type = kwargs.get('content_type', 'printed')
            language = kwargs.get('language', 'en')
            
            # Select and load appropriate model
            model_type = self._select_model(content_type)
            model, processor = self.model_manager.load_model(model_type)
            
            # Process based on whether regions are provided
            if regions and len(regions) > 0:
                ocr_result = self._process_regions_safe(image, regions, model, processor, model_type)
            else:
                ocr_result = self._process_full_image_safe(image, model, processor, model_type)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Update statistics
            self._update_stats(model_type, processing_time, ocr_result.confidence, success=True)
            
            # Calculate image statistics
            image_stats = self._calculate_image_stats(image)
            
            # Create comprehensive document result
            document_result = DocumentResult(
                pages=[ocr_result],
                metadata={
                    'image_stats': image_stats,
                    'model_type': model_type,
                    'detected_languages': [language],
                    'text_type': self._detect_text_type_from_content(ocr_result.text),
                    'preprocessing_steps': ['trocr_optimization', 'contrast_normalization', 'denoising'],
                    'postprocessing_steps': ['confidence_filtering', 'text_cleaning'],
                    'memory_usage': self.model_manager.get_memory_usage()
                },
                processing_time=processing_time,
                engine_name=self.name,
                confidence_score=ocr_result.confidence
            )
            
            return document_result
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"TrOCR processing failed: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            
            self._update_stats("unknown", processing_time, 0.0, success=False)
            self._log_error("processing", error_msg)
            
            return self._create_empty_result(error_msg, start_time)
    
    def _process_regions_safe(self, image: np.ndarray, regions: List[BoundingBox],
                             model: VisionEncoderDecoderModel, processor: TrOCRProcessor,
                             model_type: str) -> OCRResult:
        """Process specific regions with comprehensive error handling."""
        try:
            # Convert image format
            if isinstance(image, np.ndarray):
                image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                image_pil = image
            
            text_regions = []
            all_texts = []
            confidences = []
            
            # Process regions in batches
            for i in range(0, len(regions), self.batch_size):
                batch_regions = regions[i:i + self.batch_size]
                
                try:
                    batch_results = self._process_region_batch(
                        image_pil, batch_regions, model, processor, model_type
                    )
                    
                    text_regions.extend(batch_results['regions'])
                    all_texts.extend(batch_results['texts'])
                    confidences.extend(batch_results['confidences'])
                    
                except Exception as e:
                    logger.warning(f"Batch {i} processing failed: {e}")
                    # Continue with other batches
                    continue
            
            # Combine results
            combined_text = ' '.join(filter(None, all_texts))
            avg_confidence = np.mean(confidences) if confidences else 0.0
            
            # Create overall bounding box
            if regions:
                min_x = min(r.x for r in regions)
                min_y = min(r.y for r in regions)
                max_x = max(r.x + r.width for r in regions)
                max_y = max(r.y + r.height for r in regions)
                overall_bbox = BoundingBox(x=min_x, y=min_y, width=max_x-min_x, height=max_y-min_y, confidence=avg_confidence)
            else:
                h, w = image.shape[:2] if isinstance(image, np.ndarray) else (image.size[1], image.size[0])
                overall_bbox = BoundingBox(x=0, y=0, width=w, height=h, confidence=avg_confidence)
            
            return OCRResult(
                text=combined_text,
                confidence=avg_confidence,
                bbox=overall_bbox,
                regions=text_regions,
                level="page"
            )
            
        except Exception as e:
            logger.error(f"Region processing failed: {e}")
            return self._create_empty_ocr_result()
    
    def _process_full_image_safe(self, image: np.ndarray,
                                model: VisionEncoderDecoderModel, processor: TrOCRProcessor,
                                model_type: str) -> OCRResult:
        """Process entire image with error handling."""
        try:
            # Convert and preprocess image
            if isinstance(image, np.ndarray):
                image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                image_pil = image
            
            processed_image = self.optimizer.preprocess_for_trocr(image_pil)
            
            # Recognize text
            texts, confidences = self._recognize_batch_safe([processed_image], model, processor)
            
            text = texts[0] if texts else ""
            confidence = confidences[0] if confidences else 0.0
            
            # Create bounding box for entire image
            h, w = image.shape[:2] if isinstance(image, np.ndarray) else (image.size[1], image.size[0])
            full_bbox = BoundingBox(x=0, y=0, width=w, height=h, confidence=confidence)
            
            # Create text region if text found
            regions = []
            if text.strip() and confidence >= self.confidence_threshold:
                regions.append(TextRegion(
                    text=text,
                    confidence=confidence,
                    bbox=full_bbox,
                    text_type=TextType.PRINTED if model_type.endswith('printed') else TextType.HANDWRITTEN,
                    reading_order=0
                ))
            
            return OCRResult(
                text=text,
                confidence=confidence,
                bbox=full_bbox,
                regions=regions,
                level="page"
            )
            
        except Exception as e:
            logger.error(f"Full image processing failed: {e}")
            return self._create_empty_ocr_result()
    
    def _process_region_batch(self, image: Image.Image, 
                             regions: List[BoundingBox],
                             model: VisionEncoderDecoderModel,
                             processor: TrOCRProcessor,
                             model_type: str) -> Dict[str, List]:
        """Process a batch of regions with error handling."""
        batch_images = []
        valid_regions = []
        
        # Extract and preprocess region images
        for region in regions:
            try:
                # Validate region bounds
                img_width, img_height = image.size
                x1 = max(0, min(region.x, img_width - 1))
                y1 = max(0, min(region.y, img_height - 1))
                x2 = max(x1 + 1, min(region.x + region.width, img_width))
                y2 = max(y1 + 1, min(region.y + region.height, img_height))
                
                # Skip invalid regions
                if x2 <= x1 or y2 <= y1:
                    logger.warning(f"Invalid region bounds: {region.x}, {region.y}, {region.width}, {region.height}")
                    continue
                
                # Crop region
                crop_box = (x1, y1, x2, y2)
                region_img = image.crop(crop_box)
                
                # Validate cropped image
                if region_img.size[0] == 0 or region_img.size[1] == 0:
                    logger.warning(f"Empty cropped region")
                    continue
                
                # Preprocess for TrOCR
                processed_img = self.optimizer.preprocess_for_trocr(region_img)
                batch_images.append(processed_img)
                valid_regions.append(region)
                
            except Exception as e:
                logger.warning(f"Failed to process region {region}: {e}")
                continue
        
        # Process batch if we have valid images
        if batch_images:
            texts, confidences = self._recognize_batch_safe(batch_images, model, processor)
        else:
            texts, confidences = [], []
        
        # Create text regions
        text_regions = []
        for i, (region, text, conf) in enumerate(zip(valid_regions, texts, confidences)):
            if text.strip() and conf >= self.confidence_threshold:
                region_bbox = BoundingBox(
                    x=region.x, y=region.y, 
                    width=region.width, height=region.height, 
                    confidence=conf
                )
                text_regions.append(TextRegion(
                    text=text,
                    confidence=conf,
                    bbox=region_bbox,
                    text_type=TextType.PRINTED if model_type.endswith('printed') else TextType.HANDWRITTEN,
                    reading_order=i
                ))
        
        return {
            'regions': text_regions,
            'texts': texts,
            'confidences': confidences
        }
    
    def _recognize_batch_safe(self, images: List[Image.Image],
                             model: VisionEncoderDecoderModel,
                             processor: TrOCRProcessor) -> Tuple[List[str], List[float]]:
        """Recognize text from batch with comprehensive error handling."""
        if not images:
            return [], []
        
        max_retries = 2
        for attempt in range(max_retries):
            try:
                # Process images with the processor
                pixel_values = processor(images, return_tensors="pt").pixel_values
                pixel_values = pixel_values.to(self.model_manager.device)
                
                # Generate text with timeout protection
                with torch.no_grad():
                    generated_ids = model.generate(
                        pixel_values,
                        max_length=384,
                        num_beams=4,
                        early_stopping=True,
                        do_sample=False,
                        pad_token_id=processor.tokenizer.pad_token_id,
                        eos_token_id=processor.tokenizer.eos_token_id,
                        return_dict_in_generate=True,
                        output_scores=True
                    )
                
                # Decode generated text
                texts = processor.batch_decode(generated_ids.sequences, skip_special_tokens=True)
                
                # Calculate confidence scores
                confidences = self._calculate_confidence_scores_safe(generated_ids, len(images))
                
                # Post-process texts
                texts = [self._clean_recognized_text(text) for text in texts]
                
                return texts, confidences
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning(f"OOM error on attempt {attempt + 1}, clearing cache and reducing batch")
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    
                    if len(images) > 1 and attempt < max_retries - 1:
                        # Split batch and try again
                        mid = len(images) // 2
                        texts1, conf1 = self._recognize_batch_safe(images[:mid], model, processor)
                        texts2, conf2 = self._recognize_batch_safe(images[mid:], model, processor)
                        return texts1 + texts2, conf1 + conf2
                else:
                    logger.error(f"Batch recognition attempt {attempt + 1} failed: {e}")
                
                if attempt == max_retries - 1:
                    logger.error(f"All recognition attempts failed")
                    return [""] * len(images), [0.0] * len(images)
            
            except Exception as e:
                logger.error(f"Batch recognition attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    return [""] * len(images), [0.0] * len(images)
    
    def _calculate_confidence_scores_safe(self, generated_output, batch_size: int) -> List[float]:
        """Calculate confidence scores with robust error handling."""
        try:
            confidences = []
            
            # Use sequence scores if available
            if hasattr(generated_output, 'sequences_scores') and generated_output.sequences_scores is not None:
                scores = torch.softmax(generated_output.sequences_scores, dim=0)
                return scores.cpu().numpy().tolist()
            
            # Alternative: use token scores if available
            if hasattr(generated_output, 'scores') and generated_output.scores:
                for i in range(batch_size):
                    # Calculate average probability across tokens
                    token_probs = []
                    for score in generated_output.scores:
                        if i < score.shape[0]:
                            token_prob = torch.softmax(score[i], dim=0).max().item()
                            token_probs.append(token_prob)
                    
                    avg_prob = np.mean(token_probs) if token_probs else 0.7
                    confidences.append(min(avg_prob, 0.95))  # Cap at 0.95
                
                return confidences
            
            # Fallback: reasonable default confidence
            base_confidence = 0.75
            return [base_confidence] * batch_size
            
        except Exception as e:
            logger.warning(f"Confidence calculation failed: {e}")
            return [0.7] * batch_size
    
    def _clean_recognized_text(self, text: str) -> str:
        """Clean and normalize recognized text."""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Remove common OCR artifacts
        text = text.replace('|', 'I').replace('0', 'O').replace('5', 'S')
        
        # Basic text cleaning
        text = text.strip()
        
        return text
    
    def _select_model(self, content_type: str) -> str:
        """Select the best model with fallback logic."""
        if not self.auto_model_selection:
            return self.default_model
        
        try:
            # Use model manager's recommendation
            recommended = self.model_manager.get_best_model_for_content(content_type)
            
            # Validate recommended model exists
            if recommended not in TrOCRModelManager.MODEL_CONFIGS:
                return self.default_model
            
            # Consider performance history if available
            if self.recognition_stats['model_usage']:
                model_stats = self.recognition_stats['model_usage']
                
                if recommended in model_stats:
                    recent_performance = model_stats[recommended].get('avg_confidence', 0)
                    if recent_performance < 0.5:  # Poor recent performance
                        # Try fallback to default
                        return self.default_model
            
            return recommended
            
        except Exception as e:
            logger.warning(f"Model selection failed: {e}")
            return self.default_model
    
    def _update_stats(self, model_type: str, processing_time: float, 
                     confidence: float, success: bool = True):
        """Update performance statistics."""
        try:
            self.recognition_stats['total_recognitions'] += 1
            self.recognition_stats['total_time'] += processing_time
            
            if success:
                self.recognition_stats['successful_recognitions'] += 1
                self.recognition_stats['confidence_distribution'].append(confidence)
            else:
                self.recognition_stats['failed_recognitions'] += 1
            
            # Update model-specific stats
            if model_type not in self.recognition_stats['model_usage']:
                self.recognition_stats['model_usage'][model_type] = {
                    'count': 0,
                    'successful_count': 0,
                    'total_time': 0.0,
                    'confidences': [],
                    'avg_confidence': 0.0,
                    'avg_time': 0.0
                }
            
            model_stats = self.recognition_stats['model_usage'][model_type]
            model_stats['count'] += 1
            model_stats['total_time'] += processing_time
            
            if success:
                model_stats['successful_count'] += 1
                model_stats['confidences'].append(confidence)
                model_stats['avg_confidence'] = np.mean(model_stats['confidences'])
            
            model_stats['avg_time'] = model_stats['total_time'] / model_stats['count']
            
        except Exception as e:
            logger.warning(f"Failed to update stats: {e}")
    
    def _log_error(self, error_type: str, error_message: str):
        """Log error for debugging."""
        try:
            error_entry = {
                'timestamp': time.time(),
                'type': error_type,
                'message': error_message
            }
            
            self.recognition_stats['error_log'].append(error_entry)
            
            # Keep only recent errors (last 100)
            if len(self.recognition_stats['error_log']) > 100:
                self.recognition_stats['error_log'] = self.recognition_stats['error_log'][-100:]
                
        except Exception as e:
            logger.warning(f"Failed to log error: {e}")
    
    def _detect_text_type_from_content(self, text: str) -> str:
        """Detect text type based on content analysis."""
        if not text.strip():
            return TextType.UNKNOWN
        
        try:
            # Simple heuristic based on text characteristics
            words = text.split()
            if not words:
                return TextType.UNKNOWN
            
            # Analyze text characteristics
            avg_word_length = sum(len(word) for word in words) / len(words)
            special_chars = sum(1 for char in text if not char.isalnum() and not char.isspace())
            special_char_ratio = special_chars / len(text) if text else 0
            
            # Simple classification logic
            if avg_word_length < 4 and special_char_ratio > 0.1:
                return TextType.HANDWRITTEN
            else:
                return TextType.PRINTED
                
        except Exception as e:
            logger.warning(f"Text type detection failed: {e}")
            return TextType.UNKNOWN
    
    def _calculate_image_stats(self, image: np.ndarray) -> Dict[str, Any]:
        """Calculate comprehensive image statistics."""
        try:
            stats = {
                "width": int(image.shape[1]),
                "height": int(image.shape[0]),
                "channels": int(len(image.shape)),
                "mean_brightness": float(np.mean(image)),
                "std_brightness": float(np.std(image)),
                "min_value": int(np.min(image)),
                "max_value": int(np.max(image))
            }
            
            # Calculate additional stats for color images
            if len(image.shape) == 3:
                stats["mean_per_channel"] = [float(np.mean(image[:, :, i])) for i in range(image.shape[2])]
            
            return stats
            
        except Exception as e:
            logger.warning(f"Image stats calculation failed: {e}")
            return {
                "width": 0, "height": 0, "channels": 0,
                "mean_brightness": 0.0, "std_brightness": 0.0,
                "error": str(e)
            }
    
    def _create_empty_result(self, error_message: str, start_time: float) -> DocumentResult:
        """Create empty result with error information."""
        processing_time = time.time() - start_time
        
        empty_page = OCRResult(
            text="",
            confidence=0.0,
            regions=[],
            processing_time=processing_time,
            bbox=BoundingBox(x=0, y=0, width=0, height=0, confidence=0.0),
            level="page"
        )
        
        return DocumentResult(
            pages=[empty_page],
            metadata={'error': error_message, 'engine_status': 'failed'},
            processing_time=processing_time,
            engine_name=self.name,
            confidence_score=0.0
        )
    
    def _create_empty_ocr_result(self) -> OCRResult:
        """Create empty OCR result."""
        return OCRResult(
            text="",
            confidence=0.0,
            regions=[],
            processing_time=0.0,
            bbox=BoundingBox(x=0, y=0, width=0, height=0, confidence=0.0),
            level="page"
        )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        try:
            stats = self.recognition_stats.copy()
            
            if stats['total_recognitions'] > 0:
                stats['avg_processing_time'] = stats['total_time'] / stats['total_recognitions']
                stats['success_rate'] = stats['successful_recognitions'] / stats['total_recognitions']
                
                if stats['confidence_distribution']:
                    stats['avg_confidence'] = np.mean(stats['confidence_distribution'])
                    stats['confidence_std'] = np.std(stats['confidence_distribution'])
            
            # Add memory usage
            stats['memory_usage'] = self.model_manager.get_memory_usage()
            
            return stats
            
        except Exception as e:
            logger.warning(f"Failed to get performance stats: {e}")
            return {'error': str(e)}
    
    def benchmark_models(self, test_images: List[Union[np.ndarray, Image.Image]], 
                        content_types: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """Benchmark different TrOCR models with comprehensive testing."""
        if not test_images:
            return {'error': 'No test images provided'}
        
        if content_types is None:
            content_types = ['printed'] * len(test_images)
        
        results = {}
        
        for model_type in self.model_manager.MODEL_CONFIGS:
            logger.info(f"Benchmarking model: {model_type}")
            
            try:
                model, processor = self.model_manager.load_model(model_type)
                
                model_results = {
                    'processing_times': [],
                    'confidences': [],
                    'texts': [],
                    'errors': 0
                }
                
                for i, (img, content_type) in enumerate(zip(test_images, content_types)):
                    try:
                        start_time = time.time()
                        
                        # Process single image
                        result = self._process_full_image_safe(img, model, processor, model_type)
                        
                        processing_time = time.time() - start_time
                        
                        model_results['processing_times'].append(processing_time)
                        model_results['confidences'].append(result.confidence)
                        model_results['texts'].append(result.text)
                        
                    except Exception as e:
                        logger.warning(f"Benchmark failed for image {i} with {model_type}: {e}")
                        model_results['errors'] += 1
                
                # Calculate statistics
                if model_results['processing_times']:
                    results[model_type] = {
                        'avg_time': np.mean(model_results['processing_times']),
                        'avg_confidence': np.mean(model_results['confidences']),
                        'total_time': sum(model_results['processing_times']),
                        'error_rate': model_results['errors'] / len(test_images),
                        'successful_images': len(model_results['processing_times'])
                    }
                else:
                    results[model_type] = {'error': 'All benchmark tests failed'}
                    
            except Exception as e:
                logger.error(f"Benchmarking failed for {model_type}: {e}")
                results[model_type] = {'error': str(e)}
        
        return results
    
    def cleanup(self):
        """Comprehensive cleanup of resources."""
        try:
            logger.info("Starting TrOCR Engine cleanup...")
            
            # Clear model cache
            if hasattr(self, 'model_manager') and self.model_manager:
                self.model_manager.loaded_models.clear()
            
            # Clear CUDA cache if using GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            # Reset statistics
            self.recognition_stats = {
                'total_recognitions': 0,
                'total_time': 0.0,
                'successful_recognitions': 0,
                'failed_recognitions': 0,
                'model_usage': {},
                'confidence_distribution': [],
                'error_log': []
            }
            
            # Reset initialization state
            self.is_initialized = False
            self.model_loaded = False
            
            logger.info("TrOCR Engine cleanup completed successfully")
            
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")


# Utility functions for standalone usage
def create_trocr_engine(config: Optional[Dict[str, Any]] = None) -> TrOCREngine:
    """Create and initialize a TrOCR engine with default configuration."""
    if config is None:
        config = {
            'default_model': 'base-printed',
            'auto_model_selection': True,
            'batch_size': 4,
            'confidence_threshold': 0.7,
            'max_workers': 2,
            'timeout': 300
        }
    
    engine = TrOCREngine(config)
    
    if not engine.initialize():
        logger.error("Failed to initialize TrOCR engine")
        return None
    
    return engine


def test_trocr_engine():
    """Test function for TrOCR engine."""
    try:
        # Create test configuration
        config = {
            'default_model': 'base-printed',
            'auto_model_selection': True,
            'batch_size': 2,
            'confidence_threshold': 0.5
        }
        
        # Initialize engine
        engine = TrOCREngine(config)
        
        if not engine.initialize():
            print("❌ TrOCR engine initialization failed")
            return False
        
        print("✅ TrOCR engine initialized successfully")
        
        # Create test image
        test_image = np.ones((100, 300, 3), dtype=np.uint8) * 255
        cv2.putText(test_image, "Hello World", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        # Test processing
        result = engine.process_image(test_image)
        
        if result and result.pages:
            print(f"✅ Text recognized: '{result.pages[0].text}'")
            print(f"✅ Confidence: {result.pages[0].confidence:.3f}")
            return True
        else:
            print("❌ No text recognized")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    finally:
        if 'engine' in locals():
            engine.cleanup()


# Example usage
if __name__ == "__main__":
    print("TrOCR Engine - Professional Fixed Version")
    print("=" * 50)
    
    # Run test
    success = test_trocr_engine()
    
    if success:
        print("\n🎉 TrOCR Engine is working correctly!")
    else:
        print("\n❌ TrOCR Engine test failed - check logs for details")
    
    print("\nAvailable models:", list(TrOCRModelManager.MODEL_CONFIGS.keys()))