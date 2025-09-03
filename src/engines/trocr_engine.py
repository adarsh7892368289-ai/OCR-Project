# src/engines/trocr_engine.py

import cv2
import numpy as np
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, AutoTokenizer
from typing import List, Dict, Any, Tuple, Optional, Union
import time
import warnings
import logging
from pathlib import Path
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import gc
warnings.filterwarnings("ignore")

from ..core.base_engine import BaseOCREngine, OCRResult, DocumentResult, TextRegion, BoundingBox, TextType
from ..preprocessing.text_detector import AdvancedTextDetector, TextRegion as DetectedRegion

@dataclass
class TrOCRModelConfig:
    """Configuration for TrOCR model variants"""
    model_name: str
    processor_name: Optional[str] = None
    specialization: str = "general"  # printed, handwritten, scene, general
    max_length: int = 256
    confidence_threshold: float = 0.3
    batch_size: int = 8
    
class EnhancedTrOCREngine(BaseOCREngine):
    """Enhanced TrOCR Engine with multi-model support and advanced text processing"""
    
    # Pre-defined model configurations
    MODEL_CONFIGS = {
        "printed": TrOCRModelConfig(
            model_name="microsoft/trocr-base-printed",
            specialization="printed",
            confidence_threshold=0.4,
            batch_size=16
        ),
        "handwritten": TrOCRModelConfig(
            model_name="microsoft/trocr-base-handwritten", 
            specialization="handwritten",
            confidence_threshold=0.3,
            batch_size=8
        ),
        "scene": TrOCRModelConfig(
            model_name="microsoft/trocr-base-str",
            specialization="scene",
            confidence_threshold=0.35,
            batch_size=12
        ),
        "large_printed": TrOCRModelConfig(
            model_name="microsoft/trocr-large-printed",
            specialization="printed",
            confidence_threshold=0.5,
            batch_size=4
        ),
        "large_handwritten": TrOCRModelConfig(
            model_name="microsoft/trocr-large-handwritten",
            specialization="handwritten", 
            confidence_threshold=0.4,
            batch_size=4
        )
    }
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("Enhanced_TrOCR", config)
        
        self.models = {}  # Dictionary to store multiple models
        self.processors = {}  # Dictionary to store processors
        self.tokenizers = {}  # Dictionary to store tokenizers
        
        # Configuration
        self.device = self._setup_device(config.get("device", "auto"))
        self.enable_multi_model = config.get("enable_multi_model", True)
        self.primary_model = config.get("primary_model", "printed")
        self.fallback_models = config.get("fallback_models", ["handwritten", "scene"])
        self.batch_processing = config.get("batch_processing", True)
        self.max_workers = config.get("max_workers", 4)
        
        # Advanced features
        self.enable_confidence_calibration = config.get("enable_confidence_calibration", True)
        self.enable_text_type_detection = config.get("enable_text_type_detection", True)
        self.enable_adaptive_preprocessing = config.get("enable_adaptive_preprocessing", True)
        self.use_integrated_detection = config.get("use_integrated_detection", True)
        
        # Performance optimization
        self.fp16_inference = config.get("fp16_inference", True)
        self.compile_models = config.get("compile_models", False)
        self.cache_embeddings = config.get("cache_embeddings", False)
        
        # Initialize text detector if integrated detection is enabled
        self.text_detector = None
        if self.use_integrated_detection:
            detection_config = config.get("detection_config", {})
            self.text_detector = AdvancedTextDetector(detection_config)
        
        # Performance tracking
        self.model_performance = {}
        self.processing_stats = {
            'total_regions_processed': 0,
            'successful_recognitions': 0,
            'average_confidence': 0.0,
            'model_usage_count': {},
            'processing_times': []
        }
        
        # Set capabilities
        self.supports_handwriting = True
        self.supports_multiple_languages = True  
        self.supports_orientation_detection = True
        self.supports_structure_analysis = False
        self.max_image_size = (4096, 4096)
        self.min_image_size = (16, 16)
        
        self.logger.info(f"Enhanced TrOCR engine initialized with device: {self.device}")
    
    def _setup_device(self, device_config: str) -> torch.device:
        """Setup computation device"""
        if device_config == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
                self.logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = torch.device("mps")
                self.logger.info("Using Apple Silicon GPU (MPS)")
            else:
                device = torch.device("cpu")
                self.logger.info("Using CPU")
        else:
            device = torch.device(device_config)
            
        return device
    
    def initialize(self) -> bool:
        """Initialize TrOCR models and processors"""
        try:
            self.logger.info("Initializing Enhanced TrOCR models...")
            
            # Determine which models to load
            models_to_load = [self.primary_model]
            if self.enable_multi_model:
                models_to_load.extend(self.fallback_models)
            
            # Remove duplicates while preserving order
            models_to_load = list(dict.fromkeys(models_to_load))
            
            # Load each model
            for model_key in models_to_load:
                if model_key not in self.MODEL_CONFIGS:
                    self.logger.error(f"Unknown model configuration: {model_key}")
                    continue
                    
                success = self._load_model(model_key)
                if not success:
                    self.logger.warning(f"Failed to load model: {model_key}")
            
            if not self.models:
                self.logger.error("No models loaded successfully")
                return False
            
            # Initialize performance tracking for loaded models
            for model_key in self.models.keys():
                self.model_performance[model_key] = {
                    'usage_count': 0,
                    'success_rate': 0.0,
                    'avg_confidence': 0.0,
                    'avg_processing_time': 0.0
                }
            
            # Set supported languages (TrOCR primarily supports English and some multilingual models)
            self.supported_languages = ["en"]
            if any("multilingual" in config.model_name.lower() 
                  for config in self.MODEL_CONFIGS.values() 
                  if any(key in self.models for key in self.MODEL_CONFIGS.keys())):
                self.supported_languages.extend(["de", "fr", "es", "it", "pt", "ja", "ko", "zh"])
            
            self.is_initialized = True
            self.model_loaded = True
            
            self.logger.info(f"Enhanced TrOCR initialized with {len(self.models)} models")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Enhanced TrOCR: {e}")
            return False
    
    def _load_model(self, model_key: str) -> bool:
        """Load a specific TrOCR model"""
        config = self.MODEL_CONFIGS[model_key]
        
        try:
            self.logger.info(f"Loading {model_key} model: {config.model_name}")
            
            # Load processor
            processor_name = config.processor_name or config.model_name
            processor = TrOCRProcessor.from_pretrained(processor_name)
            
            # Load model
            model = VisionEncoderDecoderModel.from_pretrained(config.model_name)
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(config.model_name)
            
            # Move to device and optimize
            model.to(self.device)
            model.eval()
            
            # Enable FP16 inference if requested and supported
            if self.fp16_inference and self.device.type == "cuda":
                model = model.half()
            
            # Compile models for faster inference (PyTorch 2.0+)
            if self.compile_models and hasattr(torch, 'compile'):
                try:
                    model = torch.compile(model)
                    self.logger.info(f"Model {model_key} compiled for optimization")
                except Exception as e:
                    self.logger.warning(f"Failed to compile model {model_key}: {e}")
            
            # Store model components
            self.models[model_key] = model
            self.processors[model_key] = processor
            self.tokenizers[model_key] = tokenizer
            
            self.logger.info(f"Successfully loaded {model_key} model")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load {model_key} model: {e}")
            return False
    
    def get_supported_languages(self) -> List[str]:
        """Get supported languages"""
        return self.supported_languages
    
    def process_image(self, image: np.ndarray, **kwargs) -> DocumentResult:
        """Process image with enhanced TrOCR recognition"""
        if not self.is_initialized:
            raise RuntimeError("Enhanced TrOCR engine not initialized")
        
        start_time = time.time()
        
        try:
            # Validate image
            if not self.validate_image(image):
                raise ValueError("Invalid image for processing")
            
            # Detect text regions using integrated or external detection
            if self.use_integrated_detection and self.text_detector:
                text_regions = self.text_detector.detect_text_regions(image)
            else:
                # Fallback to simple detection or use provided regions
                text_regions = kwargs.get('text_regions', self._fallback_text_detection(image))
            
            if not text_regions:
                self.logger.warning("No text regions detected")
                return self._create_empty_result(time.time() - start_time)
            
            # Convert text regions to our format if needed
            if text_regions and isinstance(text_regions[0], tuple):
                # Legacy format conversion
                text_regions = [DetectedRegion(bbox=bbox, confidence=0.8, method="legacy") 
                              for bbox in text_regions]
            
            # Process text regions
            if self.batch_processing and len(text_regions) > 1:
                results = self._batch_process_regions(image, text_regions, **kwargs)
            else:
                results = self._sequential_process_regions(image, text_regions, **kwargs)
            
            # Post-process results
            results = self._post_process_results(results, image.shape[:2])
            
            # Extract full text with proper formatting
            full_text = self._extract_structured_text(results, text_regions)
            
            # Calculate final confidence score
            processing_time = time.time() - start_time
            confidence_score = self.calculate_confidence(results)
            image_stats = self._calculate_enhanced_image_stats(image)
            
            # Update performance statistics
            self._update_processing_stats(results, processing_time)
            
            # Convert detected regions to TextRegion format
            processed_text_regions = self._convert_to_text_regions(results, text_regions)
            
            return DocumentResult(
                full_text=full_text,
                results=results,
                text_regions=processed_text_regions,
                document_structure=self._analyze_document_structure(processed_text_regions),
                processing_time=processing_time,
                engine_name=self.name,
                image_stats=image_stats,
                confidence_score=confidence_score,
                detected_languages=self._detect_languages(full_text),
                text_type=self._determine_dominant_text_type(results),
                preprocessing_steps=["text_detection", "region_extraction", "preprocessing"],
                postprocessing_steps=["confidence_calibration", "text_correction", "structure_analysis"]
            )
            
        except Exception as e:
            self.logger.error(f"Enhanced TrOCR processing error: {e}")
            return self._create_empty_result(time.time() - start_time, str(e))
    
    def _fallback_text_detection(self, image: np.ndarray) -> List[DetectedRegion]:
        """Fallback text detection when integrated detection is not available"""
        h, w = image.shape[:2]
        
        # Simple grid-based detection as fallback
        regions = []
        
        # Try to detect text using simple methods
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Use morphological operations to find text-like regions
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
        morphed = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            x, y, width, height = cv2.boundingRect(contour)
            if width > 20 and height > 10 and width < w * 0.8 and height < h * 0.8:
                regions.append(DetectedRegion(
                    bbox=(x, y, width, height),
                    confidence=0.6,
                    method="fallback"
                ))
        
        # If no regions found, use the entire image
        if not regions:
            regions.append(DetectedRegion(
                bbox=(0, 0, w, h),
                confidence=0.5,
                method="full_image"
            ))
        
        return regions
    
    def _batch_process_regions(self, image: np.ndarray, text_regions: List[DetectedRegion], 
                             **kwargs) -> List[OCRResult]:
        """Process multiple text regions in batches for efficiency"""
        
        results = []
        batch_size = self.MODEL_CONFIGS[self.primary_model].batch_size
        
        # Process regions in batches
        for i in range(0, len(text_regions), batch_size):
            batch_regions = text_regions[i:i + batch_size]
            batch_results = self._process_region_batch(image, batch_regions, **kwargs)
            results.extend(batch_results)
        
        return results
    
    def _process_region_batch(self, image: np.ndarray, regions: List[DetectedRegion],
                            **kwargs) -> List[OCRResult]:
        """Process a batch of regions simultaneously"""
        
        # Extract region images
        region_images = []
        valid_regions = []
        
        for region in regions:
            region_image = self._extract_region_image(image, region)
            if region_image is not None:
                region_images.append(region_image)
                valid_regions.append(region)
        
        if not region_images:
            return []
        
        # Select best model for this batch
        model_key = self._select_model_for_regions(valid_regions)
        model = self.models[model_key]
        processor = self.processors[model_key]
        
        try:
            # Preprocess all images
            processed_images = []
            for region_image in region_images:
                processed_image = self._preprocess_for_trocr(region_image)
                processed_images.append(processed_image)
            
            # Stack images into batch
            batch_tensor = torch.stack([
                processor(processed_image, return_tensors="pt").pixel_values[0] 
                for processed_image in processed_images
            ]).to(self.device)
            
            # Perform batch inference
            with torch.no_grad():
                if self.fp16_inference and self.device.type == "cuda":
                    batch_tensor = batch_tensor.half()
                
                generated_ids = model.generate(
                    batch_tensor,
                    max_new_tokens=self.MODEL_CONFIGS[model_key].max_length,
                    do_sample=False,
                    num_beams=4,
                    early_stopping=True,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id
                )
            
            # Decode results
            generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
            
            # Create OCR results
            results = []
            for i, (text, region) in enumerate(zip(generated_texts, valid_regions)):
                if text.strip():
                    # Calculate confidence
                    confidence = self._calculate_region_confidence(
                        text, region_images[i], model_key
                    )
                    
                    # Create bounding box
                    bbox = BoundingBox(
                        x=region.bbox[0], y=region.bbox[1],
                        width=region.bbox[2], height=region.bbox[3],
                        confidence=confidence
                    )
                    
                    result = OCRResult(
                        text=text.strip(),
                        confidence=confidence,
                        bbox=bbox,
                        level="line",
                        text_type=self._classify_text_type(text, region_images[i]),
                        processing_metadata={
                            'model_used': model_key,
                            'batch_processed': True,
                            'region_confidence': region.confidence,
                            'detection_method': region.method
                        }
                    )
                    
                    if self.validate_result(result):
                        results.append(result)
            
            # Update model performance
            self._update_model_performance(model_key, results, len(valid_regions))
            
            return results
            
        except Exception as e:
            self.logger.error(f"Batch processing failed with {model_key}: {e}")
            # Fallback to sequential processing
            return self._sequential_process_regions(image, valid_regions, **kwargs)
    
    def _sequential_process_regions(self, image: np.ndarray, text_regions: List[DetectedRegion],
                                  **kwargs) -> List[OCRResult]:
        """Process text regions sequentially with model selection"""
        
        results = []
        
        for region in text_regions:
            try:
                region_results = self._process_single_region(image, region, **kwargs)
                results.extend(region_results)
            except Exception as e:
                self.logger.warning(f"Failed to process region {region.bbox}: {e}")
                continue
        
        return results
    
    def _process_single_region(self, image: np.ndarray, region: DetectedRegion,
                             **kwargs) -> List[OCRResult]:
        """Process a single text region with intelligent model selection"""
        
        # Extract region image
        region_image = self._extract_region_image(image, region)
        if region_image is None:
            return []
        
        # Select best model for this region
        model_key = self._select_model_for_region(region, region_image)
        
        # Try primary model first
        result = self._recognize_with_model(region_image, region, model_key)
        
        # If primary model fails and multi-model is enabled, try fallback models
        if (not result or result.confidence < self.MODEL_CONFIGS[model_key].confidence_threshold) and self.enable_multi_model:
            for fallback_model in self.fallback_models:
                if fallback_model != model_key and fallback_model in self.models:
                    fallback_result = self._recognize_with_model(region_image, region, fallback_model)
                    if fallback_result and fallback_result.confidence > (result.confidence if result else 0):
                        result = fallback_result
                        model_key = fallback_model
                        break
        
        if result:
            # Update model performance
            self._update_model_performance(model_key, [result], 1)
            return [result]
        
        return []
    
    def _recognize_with_model(self, region_image: np.ndarray, region: DetectedRegion, 
                            model_key: str) -> Optional[OCRResult]:
        """Recognize text in region using specified model"""
        
        if model_key not in self.models:
            return None
        
        model = self.models[model_key]
        processor = self.processors[model_key]
        config = self.MODEL_CONFIGS[model_key]
        
        try:
            # Preprocess image for TrOCR
            processed_image = self._preprocess_for_trocr(region_image)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(processed_image)
            
            # Process with TrOCR
            pixel_values = processor(pil_image, return_tensors="pt").pixel_values.to(self.device)
            
            if self.fp16_inference and self.device.type == "cuda":
                pixel_values = pixel_values.half()
            
            with torch.no_grad():
                generated_ids = model.generate(
                    pixel_values,
                    max_new_tokens=config.max_length,
                    do_sample=False,
                    num_beams=4,
                    early_stopping=True,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id
                )
            
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            if not generated_text.strip():
                return None
            
            # Calculate confidence
            confidence = self._calculate_region_confidence(generated_text, region_image, model_key)
            
            # Apply confidence calibration if enabled
            if self.enable_confidence_calibration:
                confidence = self._calibrate_confidence(confidence, model_key, generated_text, region_image)
            
            # Create bounding box
            bbox = BoundingBox(
                x=region.bbox[0], y=region.bbox[1],
                width=region.bbox[2], height=region.bbox[3],
                confidence=confidence
            )
            
            # Determine text type
            text_type = self._classify_text_type(generated_text, region_image)
            
            result = OCRResult(
                text=generated_text.strip(),
                confidence=confidence,
                bbox=bbox,
                level="line",
                text_type=text_type,
                processing_metadata={
                    'model_used': model_key,
                    'region_confidence': region.confidence,
                    'detection_method': region.method,
                    'preprocessing_applied': self.enable_adaptive_preprocessing
                }
            )
            
            return result if self.validate_result(result) else None
            
        except Exception as e:
            self.logger.error(f"Recognition failed with model {model_key}: {e}")
            return None
    
    def _select_model_for_regions(self, regions: List[DetectedRegion]) -> str:
        """Select best model for a batch of regions"""
        
        # Analyze regions to determine best model
        text_type_votes = {"printed": 0, "handwritten": 0, "scene": 0}
        
        for region in regions:
            # Simple heuristic based on region properties
            aspect_ratio = region.bbox[2] / region.bbox[3] if region.bbox[3] > 0 else 1
            
            if aspect_ratio > 5:  # Very wide regions might be printed
                text_type_votes["printed"] += 1
            elif region.bbox[2] * region.bbox[3] < 1000:  # Small regions might be scene text
                text_type_votes["scene"] += 1
            else:
                text_type_votes["handwritten"] += 1
        
        # Select model based on votes and performance history
        best_model = max(text_type_votes.items(), key=lambda x: x[1])[0]
        
        # Map text type to model
        type_to_model = {
            "printed": "printed",
            "handwritten": "handwritten", 
            "scene": "scene"
        }
        
        selected_model = type_to_model.get(best_model, self.primary_model)
        
        # Ensure the model is available
        if selected_model not in self.models:
            selected_model = self.primary_model
        
        return selected_model
    
    def _select_model_for_region(self, region: DetectedRegion, region_image: np.ndarray) -> str:
        """Select the best model for a single region based on characteristics"""
        
        if not self.enable_multi_model:
            return self.primary_model
        
        # Analyze region characteristics
        if self.enable_text_type_detection:
            text_type = self._analyze_region_characteristics(region, region_image)
            
            # Map characteristics to model selection
            if text_type == TextType.HANDWRITTEN:
                preferred_model = "handwritten"
            elif text_type == TextType.PRINTED:
                preferred_model = "printed"  
            else:
                preferred_model = self.primary_model
            
            # Check if preferred model is available
            if preferred_model in self.models:
                return preferred_model
        
        # Consider model performance history
        best_model = self.primary_model
        best_score = 0.0
        
        for model_key, performance in self.model_performance.items():
            if model_key in self.models:
                # Score based on success rate and confidence
                score = performance['success_rate'] * 0.7 + performance['avg_confidence'] * 0.3
                if score > best_score:
                    best_score = score
                    best_model = model_key
        
        return best_model