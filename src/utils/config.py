# src/utils/config.py

import json
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, List, Union
import os
from dataclasses import dataclass, asdict, field
from enum import Enum

class ProcessingLevel(Enum):
    """Image processing intensity levels"""
    MINIMAL = "minimal"
    LIGHT = "light" 
    MEDIUM = "medium"
    AGGRESSIVE = "aggressive"
    MAXIMUM = "maximum"

class EngineStrategy(Enum):
    """Engine selection strategies"""
    ADAPTIVE = "adaptive"
    PERFORMANCE = "performance"
    ROUND_ROBIN = "round_robin"
    PRIORITY = "priority"

@dataclass
class DetectionConfig:
    """Text detection configuration"""
    method: str = "deep_learning"  # traditional, deep_learning, hybrid
    model_name: str = "craft"
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.4
    text_threshold: float = 0.7
    link_threshold: float = 0.4
    low_text: float = 0.4
    cuda: bool = True
    canvas_size: int = 1280
    mag_ratio: float = 1.5
    slope_ths: float = 0.1
    ycenter_ths: float = 0.5
    height_ths: float = 0.7
    width_ths: float = 0.5
    add_margin: float = 0.1

@dataclass
class PreprocessingConfig:
    """Enhanced preprocessing configuration"""
    # Image enhancement
    enhancement_level: ProcessingLevel = ProcessingLevel.MEDIUM
    preserve_aspect_ratio: bool = True
    target_dpi: Optional[int] = 300
    min_image_size: tuple = (32, 32)
    max_image_size: tuple = (4096, 4096)
    
    # Skew correction
    enable_skew_correction: bool = True
    angle_range: float = 45.0
    angle_step: float = 0.5
    skew_threshold: float = 0.5
    
    # Noise reduction
    enable_denoising: bool = True
    denoise_method: str = "bilateral"  # gaussian, bilateral, morphological
    kernel_size: int = 3
    
    # Contrast and brightness
    auto_contrast: bool = True
    contrast_limit: float = 2.0
    brightness_adjustment: float = 0.0
    gamma_correction: float = 1.0
    
    # Binarization
    enable_binarization: bool = True
    binarization_method: str = "adaptive"  # otsu, adaptive, sauvola
    block_size: int = 15
    c_constant: float = 2.0
    
    # Morphological operations
    enable_morphological: bool = False
    morph_operation: str = "opening"  # opening, closing, gradient
    morph_kernel_size: tuple = (3, 3)
    morph_iterations: int = 1

@dataclass
class PostprocessingConfig:
    """Enhanced postprocessing configuration"""
    # Confidence filtering
    min_confidence: float = 0.3
    confidence_weighted_average: bool = True
    
    # Text filtering
    min_word_length: int = 1
    min_text_area: int = 100
    filter_special_chars: bool = False
    allowed_chars: Optional[str] = None
    
    # Language and spell checking
    language: str = "en"
    enable_spell_check: bool = True
    spell_check_confidence: float = 0.8
    custom_dictionary: List[str] = field(default_factory=list)
    domain_vocabulary: List[str] = field(default_factory=list)
    
    # Layout analysis
    enable_layout_analysis: bool = True
    line_height_threshold: float = 1.5
    paragraph_gap_threshold: float = 2.0
    reading_order_method: str = "top_to_bottom"  # top_to_bottom, left_to_right, natural
    
    # Text correction
    enable_context_correction: bool = True
    correction_model: str = "transformer"  # rule_based, transformer
    max_correction_distance: int = 2
    
    # Structure detection
    detect_headers: bool = True
    detect_lists: bool = True
    detect_tables: bool = True
    header_size_ratio: float = 1.2
    list_indent_threshold: int = 20

@dataclass
class EngineConfig:
    """Individual engine configuration"""
    enabled: bool = True
    priority: int = 1
    config: Dict[str, Any] = field(default_factory=dict)
    timeout: float = 30.0
    retry_count: int = 2
    fallback_engines: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Base post-init method for all engine configs"""
        pass
    
    def __post_init__(self):
        """Base post-init method for all engine configs"""
        pass

@dataclass
class TesseractConfig(EngineConfig):
    """Tesseract-specific configuration"""
    def __post_init__(self):
        super().__post_init__()
        default_config = {
            "psm": 6,
            "oem": 3,
            "lang": "eng",
            "whitelist": None,
            "blacklist": None,
            "dpi": 300,
            "timeout": 30
        }
        self.config = {**default_config, **self.config}

@dataclass
class EasyOCRConfig(EngineConfig):
    """EasyOCR-specific configuration"""
    def __post_init__(self):
        super().__post_init__()
        default_config = {
            "languages": ["en"],
            "gpu": True,
            "model_storage_directory": None,
            "user_network_directory": None,
            "recog_network": "standard",
            "download_enabled": True,
            "detector": True,
            "recognizer": True,
            "width_ths": 0.7,
            "height_ths": 0.7,
            "paragraph": False,
            "min_size": 10,
            "text_threshold": 0.7,
            "low_text": 0.4,
            "link_threshold": 0.4,
            "canvas_size": 2560,
            "mag_ratio": 1.0,
            "slope_ths": 0.1,
            "ycenter_ths": 0.5,
            "add_margin": 0.1
        }
        self.config = {**default_config, **self.config}

@dataclass
class PaddleOCRConfig(EngineConfig):
    """PaddleOCR-specific configuration"""
    def __post_init__(self):
        super().__post_init__()
        default_config = {
            "ocr_version": "PP-OCRv4",
            "lang": "en",
            "det_model_dir": None,
            "rec_model_dir": None,
            "cls_model_dir": None,
            "use_angle_cls": True,
            "use_space_char": True,
            "use_gpu": True,
            "gpu_mem": 500,
            "cpu_threads": 10,
            "enable_mkldnn": False,
            "det_algorithm": "DB",
            "det_limit_side_len": 960,
            "det_limit_type": "max",
            "rec_algorithm": "SVTR_LCNet",
            "rec_image_shape": "3, 48, 320",
            "rec_batch_num": 6,
            "max_text_length": 25,
            "drop_score": 0.5,
            "use_dilation": False,
            "score_mode": "fast"
        }
        self.config = {**default_config, **self.config}

@dataclass
class TrOCRConfig(EngineConfig):
    """TrOCR-specific configuration"""
    def __post_init__(self):
        super().__post_init__()
        default_config = {
            "model_name": "microsoft/trocr-base-printed",
            "processor_name": None,  # Use same as model_name if None
            "device": "auto",
            "max_new_tokens": 256,
            "num_beams": 4,
            "early_stopping": True,
            "do_sample": False,
            "batch_size": 8,
            "max_length": 256,
            "temperature": 1.0,
            "top_k": 50,
            "top_p": 1.0,
            "repetition_penalty": 1.0,
            "length_penalty": 1.0,
            "no_repeat_ngram_size": 0,
            "fp16": True,
            "dataloader_num_workers": 0
        }
        self.config = {**default_config, **self.config}

@dataclass
class SystemConfig:
    """System-level configuration"""
    # Processing
    parallel_processing: bool = True
    max_workers: int = 4
    processing_timeout: float = 300.0
    memory_limit: int = 4096  # MB
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None
    log_rotation: bool = True
    max_log_size: int = 100  # MB
    
    # Caching
    enable_caching: bool = True
    cache_dir: str = "./cache"
    max_cache_size: int = 1024  # MB
    cache_ttl: int = 3600  # seconds
    
    # Performance monitoring
    enable_profiling: bool = False
    profile_output_dir: str = "./profiles"
    performance_tracking: bool = True
    
    # Security
    max_image_size: int = 50 * 1024 * 1024  # 50MB
    allowed_formats: List[str] = field(default_factory=lambda: [
        "jpg", "jpeg", "png", "bmp", "tiff", "tif", "webp"
    ])
    sanitize_output: bool = True

@dataclass
class OutputConfig:
    """Output formatting configuration"""
    # Text formatting
    preserve_formatting: bool = True
    include_confidence: bool = False
    include_bounding_boxes: bool = False
    include_metadata: bool = False
    normalize_whitespace: bool = True
    
    # Structure preservation
    preserve_reading_order: bool = True
    include_structure_info: bool = False
    markdown_headers: bool = False
    preserve_tables: bool = True
    
    # Export options
    export_formats: List[str] = field(default_factory=lambda: ["text"])
    output_encoding: str = "utf-8"
    include_original_image: bool = False
    
    # Quality metrics
    include_quality_metrics: bool = False
    confidence_threshold_warning: float = 0.7

@dataclass
class OCRConfig:
    """Complete OCR System Configuration"""
    
    # Core configurations
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    postprocessing: PostprocessingConfig = field(default_factory=PostprocessingConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    
    # Engine configurations
    engines: Dict[str, EngineConfig] = field(default_factory=dict)
    
    # Engine management
    engine_selection_strategy: EngineStrategy = EngineStrategy.ADAPTIVE
    enable_multi_engine: bool = False
    engine_consensus_threshold: float = 0.8
    default_engine: str = "tesseract"  # Changed from paddleocr to tesseract
    
    def __post_init__(self):
        if not self.engines:
            self.engines = {
                "tesseract": TesseractConfig(),
                "easyocr": EasyOCRConfig(),
                "paddleocr": PaddleOCRConfig(),
                "trocr": TrOCRConfig()
            }

class Config:
    """Enhanced Configuration manager for OCR system"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_data = {}
        self.config_file_path = config_path
        self.default_config = OCRConfig()
        
        # Environment variable overrides
        self._env_prefix = "OCR_"
        
        if config_path:
            self.load_from_file(config_path)
        else:
            # Load default configuration
            self._load_defaults()
            
        # Apply environment variable overrides
        self._apply_env_overrides()
    
    def _load_defaults(self):
        """Load default configuration"""
        self.config_data = asdict(self.default_config)
    
    def load_from_file(self, config_path: str):
        """Load configuration from file with enhanced error handling"""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    loaded_config = yaml.safe_load(f) or {}
                elif config_path.suffix.lower() == '.json':
                    loaded_config = json.load(f)
                else:
                    raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
            
            # Validate configuration structure
            self._validate_config(loaded_config)
            
            # Merge with defaults
            default_dict = asdict(self.default_config)
            self.config_data = self._deep_merge(default_dict, loaded_config)
            
            self.config_file_path = str(config_path)
            
        except Exception as e:
            raise RuntimeError(f"Error loading configuration from {config_path}: {e}")
    
    def _validate_config(self, config: Dict[str, Any]):
        """Validate configuration structure and values"""
        # Basic structure validation
        if not isinstance(config, dict):
            raise ValueError("Configuration must be a dictionary")
        
        # Validate engine selection strategy
        if "engine_selection_strategy" in config:
            strategy = config["engine_selection_strategy"]
            if strategy not in [s.value for s in EngineStrategy]:
                raise ValueError(f"Invalid engine selection strategy: {strategy}")
        
        # Validate processing levels
        if "preprocessing" in config and "enhancement_level" in config["preprocessing"]:
            level = config["preprocessing"]["enhancement_level"]
            if level not in [l.value for l in ProcessingLevel]:
                raise ValueError(f"Invalid preprocessing level: {level}")
        
        # Add more validation as needed
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides"""
        for key, value in os.environ.items():
            if key.startswith(self._env_prefix):
                config_key = key[len(self._env_prefix):].lower().replace('_', '.')
                
                # Convert string values to appropriate types
                converted_value = self._convert_env_value(value)
                self.set(config_key, converted_value)
    
    def _convert_env_value(self, value: str) -> Any:
        """Convert environment variable string to appropriate type"""
        # Try boolean conversion
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Try integer conversion
        try:
            return int(value)
        except ValueError:
            pass
        
        # Try float conversion
        try:
            return float(value)
        except ValueError:
            pass
        
        # Try JSON conversion for complex types
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            pass
        
        # Return as string
        return value
    
    def _deep_merge(self, default: dict, override: dict) -> dict:
        """Deep merge two dictionaries with type preservation"""
        result = default.copy()
        
        for key, value in override.items():
            if key in result:
                if isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = self._deep_merge(result[key], value)
                elif isinstance(result[key], list) and isinstance(value, list):
                    # For lists, replace completely or merge based on context
                    result[key] = value
                else:
                    result[key] = value
            else:
                result[key] = value
        
        return result
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation with type safety"""
        keys = key.split('.')
        value = self.config_data
        
        try:
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any):
        """Set configuration value using dot notation"""
        keys = key.split('.')
        config = self.config_data
        
        # Navigate to the parent dictionary
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            elif not isinstance(config[k], dict):
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def get_engine_config(self, engine_name: str) -> Dict[str, Any]:
        """Get configuration for a specific engine"""
        engine_config = self.get(f"engines.{engine_name}", {})
        
        # Merge with engine-specific defaults
        if engine_name == "tesseract":
            default = asdict(TesseractConfig())
        elif engine_name == "easyocr":
            default = asdict(EasyOCRConfig())
        elif engine_name == "paddleocr":
            default = asdict(PaddleOCRConfig())
        elif engine_name == "trocr":
            default = asdict(TrOCRConfig())
        else:
            default = asdict(EngineConfig())
        
        return self._deep_merge(default, engine_config)
    
    def save_to_file(self, config_path: Optional[str] = None):
        """Save current configuration to file"""
        if config_path is None:
            if self.config_file_path is None:
                raise ValueError("No config file path specified")
            config_path = self.config_file_path
        
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Remove None values and empty dicts for cleaner output
        clean_config = self._clean_config_for_output(self.config_data)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                yaml.dump(clean_config, f, default_flow_style=False, indent=2, sort_keys=False)
            elif config_path.suffix.lower() == '.json':
                json.dump(clean_config, f, indent=2, sort_keys=False, ensure_ascii=False)
            else:
                raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
    
    def _clean_config_for_output(self, config: Any) -> Any:
        """Clean configuration for output by removing None values and empty containers"""
        if isinstance(config, dict):
            cleaned = {}
            for k, v in config.items():
                cleaned_value = self._clean_config_for_output(v)
                if cleaned_value is not None and cleaned_value != {} and cleaned_value != []:
                    cleaned[k] = cleaned_value
            return cleaned
        elif isinstance(config, list):
            return [self._clean_config_for_output(item) for item in config if item is not None]
        else:
            return config
    
    def create_engine_config_template(self, output_path: str):
        """Create a template configuration file with all options"""
        template_config = asdict(OCRConfig())
        
        # Add comments/descriptions (if using YAML)
        if output_path.endswith(('.yaml', '.yml')):
            # Add detailed comments to template
            template_config['_comments'] = {
                'description': 'OCR System Configuration Template',
                'engines': 'Configure individual OCR engines',
                'preprocessing': 'Image preprocessing options',
                'postprocessing': 'Text postprocessing options',
                'system': 'System-level settings'
            }
        
        with open(output_path, 'w') as f:
            if output_path.endswith(('.yaml', '.yml')):
                yaml.dump(template_config, f, default_flow_style=False, indent=2)
            else:
                json.dump(template_config, f, indent=2)
    
    def validate_current_config(self) -> List[str]:
        """Validate current configuration and return list of issues"""
        issues = []
        
        # Check required fields
        required_fields = ['engines', 'preprocessing', 'postprocessing']
        for field in required_fields:
            if not self.get(field):
                issues.append(f"Missing required configuration section: {field}")
        
        # Validate engine configurations
        engines = self.get('engines', {})
        if not engines:
            issues.append("No engines configured")
        
        for engine_name, engine_config in engines.items():
            if not engine_config.get('enabled', True):
                continue
            
            # Engine-specific validation
            if engine_name == "tesseract":
                lang = engine_config.get('config', {}).get('lang')
                if not lang:
                    issues.append(f"Tesseract engine missing language configuration")
        
        # Validate numeric ranges
        confidence = self.get('postprocessing.min_confidence', 0.0)
        if not 0.0 <= confidence <= 1.0:
            issues.append(f"Invalid confidence threshold: {confidence} (must be 0.0-1.0)")
        
        return issues
    
    def __str__(self) -> str:
        return f"OCRConfig(engines={list(self.get('engines', {}).keys())}, strategy={self.get('engine_selection_strategy')})"
    
    def __repr__(self) -> str:
        return self.__str__()

class ConfigManager:
    """Simplified ConfigManager for backward compatibility"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = Config(config_path)
    
    def get_config(self) -> Dict[str, Any]:
        """Get the complete configuration dictionary"""
        return self.config.config_data
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation"""
        return self.config.get(key, default)
    
    def get_engine_config(self, engine_name: str) -> Dict[str, Any]:
        """Get configuration for a specific engine"""
        return self.config.get_engine_config(engine_name)
    
    def load_config(self, config_path: str):
        """Load configuration from file"""
        self.config.load_from_file(config_path)
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return issues"""
        return self.config.validate_current_config()

# Don't instantiate globally to avoid import issues during startup
# # config_manager = ConfigManager()  # Disabled to avoid import issues