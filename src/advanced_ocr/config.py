"""
Smart configuration management with intelligent defaults and validation.
Provides flexible configuration for all OCR pipeline components.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Tuple
from enum import Enum
import yaml
import os
from pathlib import Path


class EngineStrategy(Enum):
    """OCR engine selection strategies."""
    SINGLE = "single"      # Use best single engine
    DUAL = "dual"          # Use two complementary engines
    ADAPTIVE = "adaptive"  # Smart selection based on content
    ALL = "all"           # Use all available engines


class QualityThreshold(Enum):
    """Quality threshold levels for processing decisions."""
    LOW = "low"           # 0.4 - Accept lower quality results
    MEDIUM = "medium"     # 0.6 - Balanced quality/speed
    HIGH = "high"         # 0.8 - High quality requirements
    STRICT = "strict"     # 0.9 - Maximum quality


class ImageFormat(Enum):
    """Supported image formats."""
    JPEG = "jpeg"
    PNG = "png"
    TIFF = "tiff"
    BMP = "bmp"
    WEBP = "webp"


@dataclass
class EngineConfig:
    """Configuration for individual OCR engines."""
    
    # Engine identification
    name: str
    enabled: bool = True
    priority: int = 1  # Higher number = higher priority
    
    # Performance settings
    gpu_enabled: bool = True
    batch_size: int = 1
    max_workers: int = 1
    timeout: float = 30.0  # seconds
    
    # Quality thresholds
    min_confidence: float = 0.5
    min_word_confidence: float = 0.3
    min_line_confidence: float = 0.4
    
    # Engine-specific parameters
    model_path: Optional[str] = None
    language_codes: List[str] = field(default_factory=lambda: ["en"])
    custom_parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Resource limits
    max_memory_mb: int = 2048
    max_image_size: Tuple[int, int] = (4096, 4096)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not 0 <= self.min_confidence <= 1:
            raise ValueError("min_confidence must be between 0 and 1")
        if not 0 <= self.min_word_confidence <= 1:
            raise ValueError("min_word_confidence must be between 0 and 1")
        if not 0 <= self.min_line_confidence <= 1:
            raise ValueError("min_line_confidence must be between 0 and 1")
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")
        if self.max_memory_mb <= 0:
            raise ValueError("max_memory_mb must be positive")
    
    @classmethod
    def create_tesseract_config(cls) -> 'EngineConfig':
        """Create optimized Tesseract configuration."""
        return cls(
            name="tesseract",
            priority=2,
            gpu_enabled=False,
            timeout=20.0,
            min_confidence=0.4,
            custom_parameters={
                'psm': 6,  # Uniform block of text
                'oem': 3,  # Default OCR Engine Mode
                'dpi': 300,
                'preserve_interword_spaces': 1
            }
        )
    
    @classmethod
    def create_paddleocr_config(cls) -> 'EngineConfig':
        """Create optimized PaddleOCR configuration."""
        return cls(
            name="paddleocr",
            priority=5,
            gpu_enabled=True,
            timeout=25.0,
            min_confidence=0.6,
            custom_parameters={
                'use_angle_cls': True,
                'use_gpu': True,
                'det_model_dir': None,
                'rec_model_dir': None,
                'cls_model_dir': None,
                'show_log': False
            }
        )
    
    @classmethod
    def create_easyocr_config(cls) -> 'EngineConfig':
        """Create optimized EasyOCR configuration."""
        return cls(
            name="easyocr",
            priority=4,
            gpu_enabled=True,
            timeout=30.0,
            min_confidence=0.5,
            custom_parameters={
                'decoder': 'beamsearch',
                'beamWidth': 5,
                'batch_size': 1,
                'workers': 1,
                'allowlist': None,
                'blocklist': None
            }
        )
    
    @classmethod
    def create_trocr_config(cls) -> 'EngineConfig':
        """Create optimized TrOCR configuration."""
        return cls(
            name="trocr",
            priority=3,
            gpu_enabled=True,
            timeout=35.0,
            min_confidence=0.7,
            max_memory_mb=3072,
            custom_parameters={
                'model_name': 'microsoft/trocr-base-printed',
                'processor_name': 'microsoft/trocr-base-printed',
                'device': 'auto',
                'torch_dtype': 'float16'
            }
        )


@dataclass
class PreprocessingConfig:
    """Configuration for image preprocessing operations."""
    
    # Enable/disable preprocessing steps
    enabled: bool = True
    auto_enhance: bool = True
    normalize_orientation: bool = True
    resize_images: bool = True
    
    # Quality-based enhancement
    blur_detection_threshold: float = 100.0  # Laplacian variance threshold
    noise_reduction_threshold: float = 0.1   # Noise level threshold
    contrast_enhancement_threshold: float = 0.3  # Contrast threshold
    
    # Image transformations
    target_dpi: int = 300
    max_image_dimension: int = 3000
    min_image_dimension: int = 100
    preserve_aspect_ratio: bool = True
    
    # Enhancement parameters
    gaussian_blur_kernel: int = 3
    sharpen_strength: float = 0.5
    contrast_enhancement_factor: float = 1.2
    brightness_adjustment: float = 0.0
    
    # Denoising settings
    denoise_strength: float = 0.3
    denoise_template_window: int = 7
    denoise_search_window: int = 21
    
    # Binarization (for specific cases)
    adaptive_threshold: bool = False
    threshold_block_size: int = 11
    threshold_c: float = 2.0
    
    def __post_init__(self):
        """Validate preprocessing configuration."""
        if self.target_dpi <= 0:
            raise ValueError("target_dpi must be positive")
        if self.max_image_dimension <= self.min_image_dimension:
            raise ValueError("max_image_dimension must be greater than min_image_dimension")
        if not 0 <= self.sharpen_strength <= 2:
            raise ValueError("sharpen_strength must be between 0 and 2")
        if not 0.5 <= self.contrast_enhancement_factor <= 3:
            raise ValueError("contrast_enhancement_factor must be between 0.5 and 3")


@dataclass
class PostprocessingConfig:
    """Configuration for text postprocessing operations."""
    
    # Enable/disable postprocessing steps
    enabled: bool = True
    text_cleaning: bool = True
    result_fusion: bool = True
    layout_reconstruction: bool = True
    confidence_enhancement: bool = True
    
    # Text cleaning parameters
    remove_empty_lines: bool = True
    fix_spacing: bool = True
    normalize_unicode: bool = True
    remove_artifacts: bool = True
    
    # Fusion settings
    fusion_strategy: str = "weighted_voting"  # weighted_voting, best_confidence, consensus
    min_engines_for_fusion: int = 2
    consensus_threshold: float = 0.6
    similarity_threshold: float = 0.8
    
    # Layout reconstruction
    line_spacing_threshold: float = 1.5  # Multiple of line height
    paragraph_spacing_threshold: float = 2.0  # Multiple of line height
    column_detection: bool = True
    reading_order_optimization: bool = True
    
    # Confidence enhancement
    text_quality_weight: float = 0.3
    spatial_quality_weight: float = 0.2
    consensus_confidence_boost: float = 0.1
    single_engine_penalty: float = 0.05
    
    # Language processing
    auto_language_detection: bool = True
    spell_check: bool = False
    grammar_check: bool = False
    supported_languages: List[str] = field(default_factory=lambda: ["en", "es", "fr", "de"])
    
    def __post_init__(self):
        """Validate postprocessing configuration."""
        if not 0 <= self.consensus_threshold <= 1:
            raise ValueError("consensus_threshold must be between 0 and 1")
        if not 0 <= self.similarity_threshold <= 1:
            raise ValueError("similarity_threshold must be between 0 and 1")
        if self.fusion_strategy not in ["weighted_voting", "best_confidence", "consensus"]:
            raise ValueError("Invalid fusion_strategy")


@dataclass
class PerformanceConfig:
    """Configuration for performance optimization."""
    
    # Parallel processing
    max_workers: int = 4
    enable_parallel_engines: bool = True
    enable_batch_processing: bool = True
    
    # Memory management
    max_memory_usage_mb: int = 4096
    model_cache_size: int = 3
    image_cache_size: int = 10
    enable_memory_monitoring: bool = True
    
    # GPU settings
    gpu_memory_fraction: float = 0.7
    allow_memory_growth: bool = True
    gpu_device_ids: List[int] = field(default_factory=lambda: [0])
    
    # Optimization flags
    use_mixed_precision: bool = True
    enable_model_compilation: bool = True
    optimize_for_inference: bool = True
    
    # Timeouts and retries
    default_timeout: float = 60.0
    max_retries: int = 2
    retry_delay: float = 1.0
    
    def __post_init__(self):
        """Validate performance configuration."""
        if self.max_workers <= 0:
            raise ValueError("max_workers must be positive")
        if not 0.1 <= self.gpu_memory_fraction <= 1.0:
            raise ValueError("gpu_memory_fraction must be between 0.1 and 1.0")
        if self.default_timeout <= 0:
            raise ValueError("default_timeout must be positive")


@dataclass
class OCRConfig:
    """Main configuration container for the OCR system."""
    
    # Strategy settings
    engine_strategy: EngineStrategy = EngineStrategy.ADAPTIVE
    quality_threshold: QualityThreshold = QualityThreshold.MEDIUM
    
    # Component configurations
    engines: Dict[str, EngineConfig] = field(default_factory=dict)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    postprocessing: PostprocessingConfig = field(default_factory=PostprocessingConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    
    # Input/Output settings
    supported_formats: List[ImageFormat] = field(default_factory=lambda: [
        ImageFormat.JPEG, ImageFormat.PNG, ImageFormat.TIFF, ImageFormat.BMP
    ])
    output_format: str = "json"  # json, yaml, text
    include_debug_info: bool = False
    
    # Logging and monitoring
    log_level: str = "INFO"
    enable_metrics: bool = True
    metrics_export_path: Optional[str] = None
    
    # Model and cache paths
    models_cache_dir: Optional[str] = None
    temp_dir: Optional[str] = None
    
    def __post_init__(self):
        """Initialize default engine configurations if not provided."""
        if not self.engines:
            self._initialize_default_engines()
        
        # Set up default paths
        if self.models_cache_dir is None:
            self.models_cache_dir = os.path.expanduser("~/.advanced_ocr/models")
        if self.temp_dir is None:
            self.temp_dir = os.path.expanduser("~/.advanced_ocr/temp")
        
        # Create directories if they don't exist
        Path(self.models_cache_dir).mkdir(parents=True, exist_ok=True)
        Path(self.temp_dir).mkdir(parents=True, exist_ok=True)
    
    def _initialize_default_engines(self):
        """Initialize default engine configurations."""
        self.engines = {
            "paddleocr": EngineConfig.create_paddleocr_config(),
            "easyocr": EngineConfig.create_easyocr_config(),
            "trocr": EngineConfig.create_trocr_config(),
            "tesseract": EngineConfig.create_tesseract_config()
        }
    
    @property
    def enabled_engines(self) -> List[str]:
        """Get list of enabled engine names."""
        return [name for name, config in self.engines.items() if config.enabled]
    
    @property
    def primary_engine(self) -> Optional[str]:
        """Get primary engine based on priority."""
        enabled = [(name, config) for name, config in self.engines.items() if config.enabled]
        if not enabled:
            return None
        return max(enabled, key=lambda x: x[1].priority)[0]
    
    def get_engine_config(self, engine_name: str) -> Optional[EngineConfig]:
        """Get configuration for specific engine."""
        return self.engines.get(engine_name)
    
    def enable_engine(self, engine_name: str):
        """Enable specific engine."""
        if engine_name in self.engines:
            self.engines[engine_name].enabled = True
    
    def disable_engine(self, engine_name: str):
        """Disable specific engine."""
        if engine_name in self.engines:
            self.engines[engine_name].enabled = False
    
    def set_engine_priority(self, engine_name: str, priority: int):
        """Set priority for specific engine."""
        if engine_name in self.engines:
            self.engines[engine_name].priority = priority
    
    def get_quality_threshold_value(self) -> float:
        """Get numeric value for quality threshold."""
        thresholds = {
            QualityThreshold.LOW: 0.4,
            QualityThreshold.MEDIUM: 0.6,
            QualityThreshold.HIGH: 0.8,
            QualityThreshold.STRICT: 0.9
        }
        return thresholds[self.quality_threshold]
    
    def optimize_for_speed(self):
        """Optimize configuration for speed over accuracy."""
        self.engine_strategy = EngineStrategy.SINGLE
        self.quality_threshold = QualityThreshold.LOW
        self.preprocessing.auto_enhance = False
        self.postprocessing.result_fusion = False
        self.postprocessing.layout_reconstruction = False
        
        # Reduce timeouts
        for engine_config in self.engines.values():
            engine_config.timeout = min(engine_config.timeout, 15.0)
    
    def optimize_for_accuracy(self):
        """Optimize configuration for accuracy over speed."""
        self.engine_strategy = EngineStrategy.ALL
        self.quality_threshold = QualityThreshold.HIGH
        self.preprocessing.auto_enhance = True
        self.postprocessing.result_fusion = True
        self.postprocessing.layout_reconstruction = True
        self.postprocessing.confidence_enhancement = True
    
    def optimize_for_handwriting(self):
        """Optimize configuration for handwritten text."""
        self.engine_strategy = EngineStrategy.DUAL
        self.quality_threshold = QualityThreshold.MEDIUM
        
        # Prioritize TrOCR for handwriting
        if "trocr" in self.engines:
            self.engines["trocr"].priority = 10
            self.engines["trocr"].enabled = True
        
        # Enable preprocessing for handwriting
        self.preprocessing.auto_enhance = True
        self.preprocessing.normalize_orientation = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'engine_strategy': self.engine_strategy.value,
            'quality_threshold': self.quality_threshold.value,
            'engines': {name: {
                'name': config.name,
                'enabled': config.enabled,
                'priority': config.priority,
                'gpu_enabled': config.gpu_enabled,
                'timeout': config.timeout,
                'min_confidence': config.min_confidence,
                'language_codes': config.language_codes,
                'custom_parameters': config.custom_parameters
            } for name, config in self.engines.items()},
            'preprocessing': {
                'enabled': self.preprocessing.enabled,
                'auto_enhance': self.preprocessing.auto_enhance,
                'target_dpi': self.preprocessing.target_dpi,
                'max_image_dimension': self.preprocessing.max_image_dimension
            },
            'postprocessing': {
                'enabled': self.postprocessing.enabled,
                'text_cleaning': self.postprocessing.text_cleaning,
                'result_fusion': self.postprocessing.result_fusion,
                'layout_reconstruction': self.postprocessing.layout_reconstruction,
                'fusion_strategy': self.postprocessing.fusion_strategy
            },
            'performance': {
                'max_workers': self.performance.max_workers,
                'max_memory_usage_mb': self.performance.max_memory_usage_mb,
                'gpu_memory_fraction': self.performance.gpu_memory_fraction,
                'default_timeout': self.performance.default_timeout
            },
            'paths': {
                'models_cache_dir': self.models_cache_dir,
                'temp_dir': self.temp_dir
            },
            'output': {
                'format': self.output_format,
                'include_debug_info': self.include_debug_info,
                'log_level': self.log_level
            }
        }
    
    def to_yaml(self) -> str:
        """Convert configuration to YAML string."""
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)
    
    def save_to_file(self, file_path: Union[str, Path]):
        """Save configuration to YAML file."""
        file_path = Path(file_path)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(self.to_yaml())
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'OCRConfig':
        """Create configuration from dictionary."""
        # Extract main settings
        engine_strategy = EngineStrategy(config_dict.get('engine_strategy', 'adaptive'))
        quality_threshold = QualityThreshold(config_dict.get('quality_threshold', 'medium'))
        
        # Create base config
        config = cls(
            engine_strategy=engine_strategy,
            quality_threshold=quality_threshold
        )
        
        # Update engines if provided
        if 'engines' in config_dict:
            engines = {}
            for name, engine_dict in config_dict['engines'].items():
                engines[name] = EngineConfig(
                    name=engine_dict['name'],
                    enabled=engine_dict.get('enabled', True),
                    priority=engine_dict.get('priority', 1),
                    gpu_enabled=engine_dict.get('gpu_enabled', True),
                    timeout=engine_dict.get('timeout', 30.0),
                    min_confidence=engine_dict.get('min_confidence', 0.5),
                    language_codes=engine_dict.get('language_codes', ['en']),
                    custom_parameters=engine_dict.get('custom_parameters', {})
                )
            config.engines = engines
        
        # Update other settings
        if 'output' in config_dict:
            output = config_dict['output']
            config.output_format = output.get('format', 'json')
            config.include_debug_info = output.get('include_debug_info', False)
            config.log_level = output.get('log_level', 'INFO')
        
        if 'paths' in config_dict:
            paths = config_dict['paths']
            config.models_cache_dir = paths.get('models_cache_dir')
            config.temp_dir = paths.get('temp_dir')
        
        return config
    
    @classmethod
    def from_yaml(cls, yaml_content: str) -> 'OCRConfig':
        """Create configuration from YAML string."""
        config_dict = yaml.safe_load(yaml_content)
        return cls.from_dict(config_dict)
    
    @classmethod
    def load_from_file(cls, file_path: Union[str, Path]) -> 'OCRConfig':
        """Load configuration from YAML file."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            return cls.from_yaml(f.read())
    
    @classmethod
    def create_default(cls) -> 'OCRConfig':
        """Create default configuration."""
        return cls()
    
    @classmethod
    def create_fast(cls) -> 'OCRConfig':
        """Create configuration optimized for speed."""
        config = cls()
        config.optimize_for_speed()
        return config
    
    @classmethod
    def create_accurate(cls) -> 'OCRConfig':
        """Create configuration optimized for accuracy."""
        config = cls()
        config.optimize_for_accuracy()
        return config
    
    @classmethod
    def create_handwriting(cls) -> 'OCRConfig':
        """Create configuration optimized for handwriting."""
        config = cls()
        config.optimize_for_handwriting()
        return config