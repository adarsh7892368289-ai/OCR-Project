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

@dataclass
class TesseractConfig(EngineConfig):
    """Tesseract-specific configuration"""
    def __post_init__(self):
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
    default_engine: str = "tesseract"
    
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
        self.config_data: Dict[str, Any] = {}
        self.config_file_path = config_path
        self.default_config = OCRConfig()
        
        # Environment variable overrides
        self._env_prefix = "OCR_"
        
        try:
            if config_path and Path(config_path).exists():
                self.load_from_file(config_path)
            else:
                # Load default configuration
                self._load_defaults()
        except Exception as e:
            # Fallback to defaults if config loading fails
            print(f"Warning: Failed to load config from {config_path}: {e}")
            self._load_defaults()
            
        # Apply environment variable overrides
        try:
            self._apply_env_overrides()
        except Exception as e:
            print(f"Warning: Failed to apply environment overrides: {e}")
    
    def _load_defaults(self):
        """Load default configuration"""
        try:
            self.config_data = asdict(self.default_config)
        except Exception as e:
            # Ultra-safe fallback
            self.config_data = {
                "engines": {
                    "tesseract": {"enabled": True, "priority": 1, "config": {}, "timeout": 30.0, "retry_count": 2, "fallback_engines": []},
                    "easyocr": {"enabled": True, "priority": 2, "config": {}, "timeout": 30.0, "retry_count": 2, "fallback_engines": []},
                    "paddleocr": {"enabled": True, "priority": 3, "config": {}, "timeout": 30.0, "retry_count": 2, "fallback_engines": []},
                    "trocr": {"enabled": True, "priority": 4, "config": {}, "timeout": 30.0, "retry_count": 2, "fallback_engines": []}
                },
                "system": {"log_level": "INFO", "parallel_processing": True, "max_workers": 4},
                "preprocessing": {"enhancement_level": "medium"},
                "postprocessing": {"min_confidence": 0.3},
                "detection": {"method": "deep_learning", "model_name": "craft"},
                "output": {"preserve_formatting": True},
                "engine_selection_strategy": "adaptive",
                "default_engine": "tesseract"
            }
    
    def load_from_file(self, config_path: str):
        """Load configuration from file with enhanced error handling"""
        # FIXED: Convert string to Path object properly
        config_path_obj = Path(config_path)
        
        if not config_path_obj.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path_obj, 'r', encoding='utf-8') as f:
                # FIXED: Use config_path_obj.suffix instead of config_path.suffix
                if config_path_obj.suffix.lower() in ['.yaml', '.yml']:
                    try:
                        import yaml
                        loaded_config = yaml.safe_load(f) or {}
                    except ImportError:
                        raise RuntimeError("PyYAML not installed but YAML config file provided")
                elif config_path_obj.suffix.lower() == '.json':
                    loaded_config = json.load(f)
                else:
                    raise ValueError(f"Unsupported configuration file format: {config_path_obj.suffix}")
            
            # Validate configuration structure
            if loaded_config:
                self._validate_config(loaded_config)
                
                # Merge with defaults
                default_dict = asdict(self.default_config)
                self.config_data = self._deep_merge(default_dict, loaded_config)
            else:
                self._load_defaults()
            
            # FIXED: Store the original string path, not Path object
            self.config_file_path = config_path
            
        except Exception as e:
            raise RuntimeError(f"Error loading configuration from {config_path}: {e}")
    
    def _validate_config(self, config: Dict[str, Any]):
        """Validate configuration structure and values"""
        if not isinstance(config, dict):
            raise ValueError("Configuration must be a dictionary")
        
        # Validate engine selection strategy
        if "engine_selection_strategy" in config:
            strategy = config["engine_selection_strategy"]
            valid_strategies = [s.value for s in EngineStrategy]
            if strategy not in valid_strategies:
                print(f"Warning: Invalid engine selection strategy '{strategy}', using 'adaptive'")
                config["engine_selection_strategy"] = "adaptive"
        
        # Validate processing levels
        if "preprocessing" in config and isinstance(config["preprocessing"], dict):
            if "enhancement_level" in config["preprocessing"]:
                level = config["preprocessing"]["enhancement_level"]
                valid_levels = [l.value for l in ProcessingLevel]
                if level not in valid_levels:
                    print(f"Warning: Invalid preprocessing level '{level}', using 'medium'")
                    config["preprocessing"]["enhancement_level"] = "medium"
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides"""
        for key, value in os.environ.items():
            if key.startswith(self._env_prefix):
                try:
                    config_key = key[len(self._env_prefix):].lower().replace('_', '.')
                    converted_value = self._convert_env_value(value)
                    self.set(config_key, converted_value)
                except Exception as e:
                    print(f"Warning: Failed to apply env override {key}: {e}")
    
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
                    result[key] = value
                else:
                    result[key] = value
            else:
                result[key] = value
        
        return result
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation with bulletproof error handling"""
        if not key:
            return default
            
        try:
            keys = key.split('.')
            value = self.config_data
            
            for k in keys:
                if isinstance(value, dict) and k in value:
                    value = value[k]
                else:
                    return default
                    
            return value if value is not None else default
            
        except (KeyError, TypeError, AttributeError, ValueError):
            return default
    
    def set(self, key: str, value: Any):
        """Set configuration value using dot notation"""
        if not key:
            return
            
        try:
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
        except Exception as e:
            print(f"Warning: Failed to set config {key}={value}: {e}")
    
    def get_engine_config(self, engine_name: str) -> Dict[str, Any]:
        """Get configuration for a specific engine with safe fallbacks"""
        try:
            engine_config = self.get(f"engines.{engine_name}", {})
            
            # Provide safe defaults for any engine
            default_engine_config = {
                "enabled": True,
                "priority": 1,
                "config": {},
                "timeout": 30.0,
                "retry_count": 2,
                "fallback_engines": []
            }
            
            # Merge with engine-specific defaults
            if engine_name == "tesseract":
                try:
                    specific_default = asdict(TesseractConfig())
                except:
                    specific_default = default_engine_config.copy()
                    specific_default["config"] = {"psm": 6, "oem": 3, "lang": "eng"}
            elif engine_name == "easyocr":
                try:
                    specific_default = asdict(EasyOCRConfig())
                except:
                    specific_default = default_engine_config.copy()
                    specific_default["config"] = {"languages": ["en"], "gpu": True}
            elif engine_name == "paddleocr":
                try:
                    specific_default = asdict(PaddleOCRConfig())
                except:
                    specific_default = default_engine_config.copy()
                    specific_default["config"] = {"lang": "en", "use_gpu": True}
            elif engine_name == "trocr":
                try:
                    specific_default = asdict(TrOCRConfig())
                except:
                    specific_default = default_engine_config.copy()
                    specific_default["config"] = {"model_name": "microsoft/trocr-base-printed"}
            else:
                specific_default = default_engine_config
            
            return self._deep_merge(specific_default, engine_config)
            
        except Exception as e:
            print(f"Warning: Failed to get engine config for {engine_name}: {e}")
            return {
                "enabled": True,
                "priority": 1,
                "config": {},
                "timeout": 30.0,
                "retry_count": 2,
                "fallback_engines": []
            }
    
    def save_to_file(self, config_path: Optional[str] = None):
        """Save current configuration to file"""
        try:
            import os
            
            target_path: str
            if config_path is not None:
                target_path = config_path
            elif self.config_file_path is not None:
                target_path = self.config_file_path
            else:
                raise ValueError("No config file path specified")
            
            # Create directory if it doesn't exist
            target_dir = os.path.dirname(target_path)
            if target_dir and not os.path.exists(target_dir):
                os.makedirs(target_dir, exist_ok=True)
            
            # Remove None values and empty dicts for cleaner output
            clean_config = self._clean_config_for_output(self.config_data)
            
            # FIXED: Use string splitting instead of suffix attribute on string
            file_ext = target_path.lower().split('.')[-1] if '.' in target_path else ''
            
            with open(target_path, 'w', encoding='utf-8') as f:
                if file_ext in ['yaml', 'yml']:
                    try:
                        import yaml
                        yaml.dump(clean_config, f, default_flow_style=False, indent=2, sort_keys=False)
                    except ImportError:
                        # Fallback to JSON if PyYAML not available
                        json.dump(clean_config, f, indent=2, sort_keys=False, ensure_ascii=False)
                elif file_ext == 'json':
                    json.dump(clean_config, f, indent=2, sort_keys=False, ensure_ascii=False)
                else:
                    raise ValueError(f"Unsupported configuration file format: .{file_ext}")
        except Exception as e:
            print(f"Warning: Failed to save config to {config_path}: {e}")
    
    def _clean_config_for_output(self, config: Any) -> Any:
        """Clean configuration for output by removing None values and empty containers"""
        if isinstance(config, dict):
            cleaned = {}
            for k, v in config.items():
                try:
                    cleaned_value = self._clean_config_for_output(v)
                    if cleaned_value is not None and cleaned_value != {} and cleaned_value != []:
                        cleaned[k] = cleaned_value
                except:
                    if v is not None:
                        cleaned[k] = v
            return cleaned
        elif isinstance(config, list):
            return [self._clean_config_for_output(item) for item in config if item is not None]
        else:
            return config
    
    def validate_current_config(self) -> List[str]:
        """Validate current configuration and return list of issues"""
        issues = []
        
        try:
            # Check required fields
            required_fields = ['engines']
            for field in required_fields:
                if not self.get(field):
                    issues.append(f"Missing required configuration section: {field}")
            
            # Validate engine configurations
            engines = self.get('engines', {})
            if not engines:
                issues.append("No engines configured")
            
            # Validate numeric ranges
            confidence = self.get('postprocessing.min_confidence', 0.3)
            try:
                confidence = float(confidence)
                if not 0.0 <= confidence <= 1.0:
                    issues.append(f"Invalid confidence threshold: {confidence} (must be 0.0-1.0)")
            except (ValueError, TypeError):
                issues.append(f"Invalid confidence threshold type: {type(confidence)}")
                
        except Exception as e:
            issues.append(f"Configuration validation error: {e}")
        
        return issues
    
    def __str__(self) -> str:
        try:
            engines = list(self.get('engines', {}).keys())
            strategy = self.get('engine_selection_strategy', 'unknown')
            return f"OCRConfig(engines={engines}, strategy={strategy})"
        except:
            return "OCRConfig(error accessing config)"
    
    def __repr__(self) -> str:
        return self.__str__()

class ConfigManager:
    """Simplified ConfigManager for backward compatibility"""
    
    def __init__(self, config_path: Optional[str] = None):
        try:
            self.config = Config(config_path)
        except Exception as e:
            print(f"Warning: ConfigManager initialization failed, using minimal config: {e}")
            # Create minimal config as fallback
            self.config = Config()
    
    def get_config(self) -> Dict[str, Any]:
        """Get the complete configuration dictionary"""
        try:
            return self.config.config_data
        except:
            return {}
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation"""
        try:
            return self.config.get(key, default)
        except:
            return default
    
    def get_section(self, section_name: str, default: Any = None) -> Any:
        """Get a configuration section using dot notation (backward compatibility method)"""
        try:
            # First try dot notation
            result = self.get(section_name, default)
            if result != default:
                return result
            
            # If dot notation fails, try getting from full config dict
            full_config = self.get_config()
            return full_config.get(section_name, default)
            
        except Exception:
            return default
    
    def get_engine_config(self, engine_name: str) -> Dict[str, Any]:
        """Get configuration for a specific engine"""
        try:
            return self.config.get_engine_config(engine_name)
        except:
            return {
                "enabled": True,
                "priority": 1,
                "config": {},
                "timeout": 30.0,
                "retry_count": 2,
                "fallback_engines": []
            }
    
    def load_config(self, config_path: str):
        """Load configuration from file"""
        try:
            self.config.load_from_file(config_path)
        except Exception as e:
            print(f"Warning: Failed to load config from {config_path}: {e}")
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return issues"""
        try:
            return self.config.validate_current_config()
        except:
            return ["Configuration validation failed"]

# Safe module-level functions
def create_default_config() -> Config:
    """Create a default configuration instance"""
    try:
        return Config()
    except Exception as e:
        print(f"Error creating default config: {e}")
        # Return minimal working config
        config = Config.__new__(Config)
        config.config_data = {
            "engines": {"tesseract": {"enabled": True}},
            "system": {"log_level": "INFO"},
            "default_engine": "tesseract"
        }
        return config

def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration with fallback to defaults"""
    try:
        return Config(config_path)
    except Exception as e:
        print(f"Warning: Failed to load config, using defaults: {e}")
        return create_default_config()