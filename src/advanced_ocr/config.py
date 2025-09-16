# src/advanced_ocr/config.py
"""
Advanced OCR Configuration Management

This module provides comprehensive configuration management for the advanced OCR
system, supporting multiple processing profiles, engine-specific settings, and
dynamic parameter optimization based on content analysis.

The configuration system supports:
- Multiple processing profiles (fast, balanced, accurate)
- Engine-specific parameter tuning
- Content-type aware processing settings
- Performance optimization parameters
- Resource management and limits

Classes:
    OCRConfig: Main configuration container with validation
    EngineConfig: Engine-specific configuration parameters  
    PreprocessingConfig: Image preprocessing settings
    PostprocessingConfig: Text processing and fusion parameters
    PerformanceConfig: Speed and resource optimization settings
    
Example:
    >>> config = OCRConfig.load_from_file("config.yaml")
    >>> config.set_profile("accurate")
    >>> print(f"Text detection threshold: {config.preprocessing.text_detection_threshold}")
    
    >>> fast_config = OCRConfig.create_profile("fast")
    >>> engine_cfg = fast_config.get_engine_config("tesseract")
"""

import os
import yaml
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from enum import Enum


class ProcessingProfile(Enum):
    """Predefined processing profiles balancing speed vs accuracy."""
    FAST = "fast"
    BALANCED = "balanced" 
    ACCURATE = "accurate"
    CUSTOM = "custom"


class EngineType(Enum):
    """Supported OCR engine types."""
    TESSERACT = "tesseract"
    PADDLEOCR = "paddleocr"
    EASYOCR = "easyocr"
    TROCR = "trocr"


class ContentTypePreset(Enum):
    """Content-specific processing presets."""
    PRINTED_DOCUMENT = "printed_document"
    HANDWRITTEN_TEXT = "handwritten_text"
    MIXED_CONTENT = "mixed_content"
    FORM_PROCESSING = "form_processing"
    TABLE_EXTRACTION = "table_extraction"


@dataclass
class EngineConfig:
    """
    Engine-specific configuration parameters.
    
    Contains optimization settings, model paths, and processing parameters
    for individual OCR engines. Each engine type has specific parameters
    that affect accuracy, speed, and resource usage.
    
    Attributes:
        enabled (bool): Whether this engine is available for use
        priority (int): Selection priority (1=highest, 10=lowest)
        confidence_threshold (float): Minimum confidence for result acceptance
        timeout (float): Maximum processing time in seconds
        model_path (Optional[str]): Custom model file path
        gpu_enabled (bool): Enable GPU acceleration if available
        batch_size (int): Batch processing size for efficiency
        custom_params (Dict[str, Any]): Engine-specific parameters
    """
    enabled: bool = True
    priority: int = 5
    confidence_threshold: float = 0.6
    timeout: float = 30.0
    model_path: Optional[str] = None
    gpu_enabled: bool = False
    batch_size: int = 1
    custom_params: Dict[str, Any] = field(default_factory=dict)
    
    def validate(self) -> bool:
        """
        Validate configuration parameters.
        
        Returns:
            bool: True if configuration is valid
            
        Raises:
            ValueError: If critical parameters are invalid
        """
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError(f"confidence_threshold must be 0.0-1.0, got {self.confidence_threshold}")
        
        if not 1 <= self.priority <= 10:
            raise ValueError(f"priority must be 1-10, got {self.priority}")
        
        if self.timeout <= 0:
            raise ValueError(f"timeout must be positive, got {self.timeout}")
        
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        
        return True


@dataclass  
class PreprocessingConfig:
    """
    Image preprocessing configuration parameters.
    
    Controls the image enhancement, text detection, and content analysis
    phases that occur before OCR processing. Critical parameters affect
    the performance bottlenecks identified in the project requirements.
    
    Attributes:
        text_detection_enabled (bool): Enable text region detection
        text_detection_threshold (float): Confidence threshold for text detection
        max_regions (int): Maximum text regions to process (performance critical)
        image_enhancement_enabled (bool): Enable adaptive image enhancement
        max_image_size (int): Maximum image dimension for processing
        quality_analysis_enabled (bool): Enable image quality assessment
        content_classification_enabled (bool): Enable content type detection
        preprocessing_timeout (float): Maximum preprocessing time
    """
    text_detection_enabled: bool = True
    text_detection_threshold: float = 0.7  # Critical: was 0.1 causing 2660 regions
    max_regions: int = 80  # Critical: limit regions to 20-80 range
    image_enhancement_enabled: bool = True
    max_image_size: int = 2048
    quality_analysis_enabled: bool = True
    content_classification_enabled: bool = True
    preprocessing_timeout: float = 10.0
    
    # Enhancement parameters
    blur_correction_enabled: bool = True
    noise_reduction_enabled: bool = True  
    contrast_enhancement_enabled: bool = True
    rotation_correction_enabled: bool = True
    
    # Text detection fine-tuning (critical performance parameters)
    nms_threshold: float = 0.3  # Non-maximum suppression for region filtering
    min_region_area: int = 100  # Minimum pixel area for text regions
    merge_nearby_regions: bool = True  # Merge overlapping/nearby regions
    region_padding: int = 5  # Padding around detected text regions
    
    def validate(self) -> bool:
        """Validate preprocessing configuration."""
        if not 0.0 <= self.text_detection_threshold <= 1.0:
            raise ValueError(f"text_detection_threshold must be 0.0-1.0")
        
        if self.max_regions <= 0 or self.max_regions > 500:
            raise ValueError(f"max_regions must be 1-500 for performance")
        
        if self.max_image_size < 256:
            raise ValueError(f"max_image_size must be >= 256")
        
        if self.preprocessing_timeout <= 0:
            raise ValueError(f"preprocessing_timeout must be positive")
        
        return True


@dataclass
class PostprocessingConfig:
    """
    Text postprocessing and result fusion configuration.
    
    Controls text cleaning, multi-engine result fusion, confidence analysis,
    and layout reconstruction. Parameters affect the quality of final results
    and processing speed.
    
    Attributes:
        text_cleaning_enabled (bool): Enable text cleaning and normalization
        spell_correction_enabled (bool): Enable context-aware spell correction
        result_fusion_enabled (bool): Enable multi-engine result combination
        layout_reconstruction_enabled (bool): Enable document layout analysis
        confidence_analysis_enabled (bool): Enable advanced confidence scoring
        min_word_confidence (float): Minimum confidence for word acceptance
        fusion_strategy (str): Strategy for combining multiple engine results
    """
    text_cleaning_enabled: bool = True
    spell_correction_enabled: bool = False  # Computationally expensive
    result_fusion_enabled: bool = True
    layout_reconstruction_enabled: bool = True
    confidence_analysis_enabled: bool = True
    min_word_confidence: float = 0.3
    fusion_strategy: str = "confidence_weighted"  # Options: confidence_weighted, majority_vote, best_engine
    
    # Text cleaning parameters
    remove_extra_whitespace: bool = True
    normalize_unicode: bool = True
    remove_ocr_artifacts: bool = True
    preserve_formatting: bool = True
    
    # Layout reconstruction parameters
    preserve_line_breaks: bool = True
    preserve_paragraphs: bool = True
    reading_order_analysis: bool = True
    
    def validate(self) -> bool:
        """Validate postprocessing configuration."""
        if not 0.0 <= self.min_word_confidence <= 1.0:
            raise ValueError(f"min_word_confidence must be 0.0-1.0")
        
        valid_strategies = ["confidence_weighted", "majority_vote", "best_engine"]
        if self.fusion_strategy not in valid_strategies:
            raise ValueError(f"fusion_strategy must be one of {valid_strategies}")
        
        return True


@dataclass
class PerformanceConfig:
    """
    Performance optimization and resource management configuration.
    
    Critical settings that affect the processing speed requirements
    (<3 seconds total processing time) and resource usage optimization.
    
    Attributes:
        max_processing_time (float): Maximum total processing time in seconds
        memory_limit_mb (int): Maximum memory usage in MB
        cpu_threads (int): Number of CPU threads for parallel processing
        gpu_memory_limit_mb (int): Maximum GPU memory usage in MB
        enable_model_caching (bool): Cache models in memory for reuse
        enable_parallel_engines (bool): Run multiple engines in parallel
        processing_priority (str): System processing priority
    """
    max_processing_time: float = 3.0  # Critical: <3 second requirement
    memory_limit_mb: int = 2048
    cpu_threads: int = 4
    gpu_memory_limit_mb: int = 1024
    enable_model_caching: bool = True
    enable_parallel_engines: bool = True
    processing_priority: str = "normal"  # Options: low, normal, high
    
    # Engine coordination performance
    engine_selection_timeout: float = 0.1
    engine_initialization_timeout: float = 2.0
    result_fusion_timeout: float = 0.5
    
    # Memory management
    cleanup_after_processing: bool = True
    force_garbage_collection: bool = False
    
    def validate(self) -> bool:
        """Validate performance configuration."""
        if self.max_processing_time <= 0:
            raise ValueError(f"max_processing_time must be positive")
        
        if self.memory_limit_mb < 256:
            raise ValueError(f"memory_limit_mb must be >= 256")
        
        if self.cpu_threads < 1:
            raise ValueError(f"cpu_threads must be >= 1")
        
        valid_priorities = ["low", "normal", "high"]
        if self.processing_priority not in valid_priorities:
            raise ValueError(f"processing_priority must be one of {valid_priorities}")
        
        return True


@dataclass
class OCRConfig:
    """
    Main OCR system configuration container.
    
    Provides comprehensive configuration management with support for multiple
    processing profiles, validation, serialization, and dynamic updates.
    This is the primary interface for configuring the advanced OCR system.
    
    Attributes:
        profile (ProcessingProfile): Active processing profile
        engines (Dict[str, EngineConfig]): Engine-specific configurations
        preprocessing (PreprocessingConfig): Image preprocessing settings
        postprocessing (PostprocessingConfig): Text postprocessing settings
        performance (PerformanceConfig): Performance optimization settings
        metadata (Dict[str, Any]): Additional configuration metadata
    
    Example:
        >>> config = OCRConfig()
        >>> config.set_profile("accurate")
        >>> config.engines["tesseract"].confidence_threshold = 0.8
        >>> config.save_to_file("my_config.yaml")
    """
    profile: ProcessingProfile = ProcessingProfile.BALANCED
    engines: Dict[str, EngineConfig] = field(default_factory=dict)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    postprocessing: PostprocessingConfig = field(default_factory=PostprocessingConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize default engine configurations."""
        if not self.engines:
            self._initialize_default_engines()
    
    def _initialize_default_engines(self):
        """Create default configurations for all supported engines."""
        # Tesseract - reliable fallback engine
        self.engines["tesseract"] = EngineConfig(
            enabled=True,
            priority=3,
            confidence_threshold=0.6,
            timeout=15.0,
            custom_params={
                "psm": 6,  # Uniform block of text
                "oem": 3,  # Default OCR Engine Mode
            }
        )
        
        # PaddleOCR - optimized for printed documents
        self.engines["paddleocr"] = EngineConfig(
            enabled=True,
            priority=2,
            confidence_threshold=0.7,
            timeout=20.0,
            gpu_enabled=True,
            custom_params={
                "use_angle_cls": True,
                "use_space_char": True,
            }
        )
        
        # EasyOCR - multilingual support
        self.engines["easyocr"] = EngineConfig(
            enabled=True,
            priority=4,
            confidence_threshold=0.6,
            timeout=25.0,
            gpu_enabled=True,
            custom_params={
                "width_tha": 0.7,
                "height_tha": 0.7,
            }
        )
        
        # TrOCR - optimized for handwritten text (critical fix target)
        self.engines["trocr"] = EngineConfig(
            enabled=True,
            priority=1,  # Highest priority for handwritten content
            confidence_threshold=0.5,
            timeout=30.0,
            gpu_enabled=True,
            batch_size=4,
            custom_params={
                "max_image_size": 800,  # Critical: prevent OOM errors
                "use_cache": True,
            }
        )
    
    def set_profile(self, profile: Union[str, ProcessingProfile]):
        """
        Set processing profile and update configurations accordingly.
        
        Args:
            profile: Processing profile name or enum value
            
        Raises:
            ValueError: If profile name is invalid
        """
        if isinstance(profile, str):
            try:
                profile = ProcessingProfile(profile.lower())
            except ValueError:
                raise ValueError(f"Invalid profile: {profile}")
        
        self.profile = profile
        
        if profile == ProcessingProfile.FAST:
            self._apply_fast_profile()
        elif profile == ProcessingProfile.BALANCED:
            self._apply_balanced_profile()
        elif profile == ProcessingProfile.ACCURATE:
            self._apply_accurate_profile()
    
    def _apply_fast_profile(self):
        """Apply fast processing profile optimizations."""
        # Optimize for speed over accuracy
        self.preprocessing.text_detection_threshold = 0.6
        self.preprocessing.max_regions = 50
        self.preprocessing.image_enhancement_enabled = False
        self.preprocessing.quality_analysis_enabled = False
        
        self.postprocessing.spell_correction_enabled = False
        self.postprocessing.layout_reconstruction_enabled = False
        self.postprocessing.result_fusion_enabled = False
        
        self.performance.max_processing_time = 1.5
        self.performance.enable_parallel_engines = False
        
        # Use only fastest engines
        for engine_name in self.engines:
            if engine_name not in ["tesseract", "paddleocr"]:
                self.engines[engine_name].enabled = False
    
    def _apply_balanced_profile(self):
        """Apply balanced processing profile settings."""
        # Balance speed and accuracy
        self.preprocessing.text_detection_threshold = 0.7
        self.preprocessing.max_regions = 80
        self.preprocessing.image_enhancement_enabled = True
        self.preprocessing.quality_analysis_enabled = True
        
        self.postprocessing.spell_correction_enabled = False
        self.postprocessing.layout_reconstruction_enabled = True
        self.postprocessing.result_fusion_enabled = True
        
        self.performance.max_processing_time = 3.0
        self.performance.enable_parallel_engines = True
        
        # Enable primary engines
        for engine_name in ["tesseract", "paddleocr", "trocr"]:
            if engine_name in self.engines:
                self.engines[engine_name].enabled = True
    
    def _apply_accurate_profile(self):
        """Apply accurate processing profile settings."""
        # Optimize for accuracy over speed
        self.preprocessing.text_detection_threshold = 0.8
        self.preprocessing.max_regions = 100
        self.preprocessing.image_enhancement_enabled = True
        self.preprocessing.quality_analysis_enabled = True
        
        self.postprocessing.spell_correction_enabled = True
        self.postprocessing.layout_reconstruction_enabled = True
        self.postprocessing.result_fusion_enabled = True
        
        self.performance.max_processing_time = 5.0
        self.performance.enable_parallel_engines = True
        
        # Enable all engines
        for engine_name in self.engines:
            self.engines[engine_name].enabled = True
    
    def get_engine_config(self, engine_name: str) -> Optional[EngineConfig]:
        """
        Get configuration for specific engine.
        
        Args:
            engine_name: Name of the OCR engine
            
        Returns:
            EngineConfig object or None if engine not configured
        """
        return self.engines.get(engine_name)
    
    def update_engine_config(self, engine_name: str, **kwargs):
        """
        Update configuration parameters for specific engine.
        
        Args:
            engine_name: Name of the OCR engine
            **kwargs: Configuration parameters to update
        """
        if engine_name not in self.engines:
            self.engines[engine_name] = EngineConfig()
        
        for key, value in kwargs.items():
            if hasattr(self.engines[engine_name], key):
                setattr(self.engines[engine_name], key, value)
            else:
                # Add to custom_params if not a standard attribute
                self.engines[engine_name].custom_params[key] = value
    
    def validate(self) -> bool:
        """
        Validate all configuration parameters.
        
        Returns:
            bool: True if all configurations are valid
            
        Raises:
            ValueError: If any configuration is invalid
        """
        # Validate engine configurations
        for engine_name, engine_config in self.engines.items():
            try:
                engine_config.validate()
            except ValueError as e:
                raise ValueError(f"Invalid {engine_name} config: {e}")
        
        # Validate component configurations
        self.preprocessing.validate()
        self.postprocessing.validate()
        self.performance.validate()
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary format.
        
        Returns:
            Dictionary representation of the configuration
        """
        return {
            "profile": self.profile.value,
            "engines": {name: asdict(config) for name, config in self.engines.items()},
            "preprocessing": asdict(self.preprocessing),
            "postprocessing": asdict(self.postprocessing),
            "performance": asdict(self.performance),
            "metadata": self.metadata
        }
    
    def save_to_file(self, filepath: Union[str, Path]):
        """
        Save configuration to YAML file.
        
        Args:
            filepath: Path to save configuration file
            
        Raises:
            IOError: If file cannot be written
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
        except Exception as e:
            raise IOError(f"Failed to save configuration: {e}")
    
    @classmethod
    def load_from_file(cls, filepath: Union[str, Path]) -> 'OCRConfig':
        """
        Load configuration from YAML file.
        
        Args:
            filepath: Path to configuration file
            
        Returns:
            OCRConfig instance loaded from file
            
        Raises:
            FileNotFoundError: If configuration file doesn't exist
            ValueError: If configuration file is invalid
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
        except Exception as e:
            raise ValueError(f"Failed to load configuration: {e}")
        
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OCRConfig':
        """
        Create configuration from dictionary.
        
        Args:
            data: Dictionary containing configuration data
            
        Returns:
            OCRConfig instance created from dictionary
        """
        config = cls()
        
        # Set profile
        if "profile" in data:
            config.set_profile(data["profile"])
        
        # Load engine configurations
        if "engines" in data:
            config.engines = {}
            for name, engine_data in data["engines"].items():
                config.engines[name] = EngineConfig(**engine_data)
        
        # Load component configurations
        if "preprocessing" in data:
            config.preprocessing = PreprocessingConfig(**data["preprocessing"])
        
        if "postprocessing" in data:
            config.postprocessing = PostprocessingConfig(**data["postprocessing"])
        
        if "performance" in data:
            config.performance = PerformanceConfig(**data["performance"])
        
        if "metadata" in data:
            config.metadata = data["metadata"]
        
        return config
    
    @classmethod
    def create_profile(cls, profile: Union[str, ProcessingProfile]) -> 'OCRConfig':
        """
        Create configuration with specific profile applied.
        
        Args:
            profile: Processing profile to apply
            
        Returns:
            OCRConfig instance with profile applied
        """
        config = cls()
        config.set_profile(profile)
        return config


# Default configuration factory functions
def create_fast_config() -> OCRConfig:
    """Create configuration optimized for speed."""
    return OCRConfig.create_profile(ProcessingProfile.FAST)


def create_balanced_config() -> OCRConfig:
    """Create configuration balancing speed and accuracy."""
    return OCRConfig.create_profile(ProcessingProfile.BALANCED)


def create_accurate_config() -> OCRConfig:
    """Create configuration optimized for accuracy."""
    return OCRConfig.create_profile(ProcessingProfile.ACCURATE)


# Environment-based configuration loading
def load_config_from_env() -> OCRConfig:
    """
    Load configuration from environment variables and default files.
    
    Checks for configuration in the following order:
    1. ADVANCED_OCR_CONFIG environment variable (file path)
    2. ./ocr_config.yaml in current directory
    3. ~/.advanced_ocr/config.yaml in user home
    4. Default balanced configuration
    
    Returns:
        OCRConfig instance loaded from environment or defaults
    """
    # Check environment variable
    config_path = os.getenv("ADVANCED_OCR_CONFIG")
    if config_path and Path(config_path).exists():
        return OCRConfig.load_from_file(config_path)
    
    # Check current directory
    local_config = Path("ocr_config.yaml")
    if local_config.exists():
        return OCRConfig.load_from_file(local_config)
    
    # Check user home directory
    home_config = Path.home() / ".advanced_ocr" / "config.yaml"
    if home_config.exists():
        return OCRConfig.load_from_file(home_config)
    
    # Return default configuration
    return create_balanced_config()


__all__ = [
    'OCRConfig',
    'EngineConfig',
    'PreprocessingConfig',
    'PostprocessingConfig',
    'PerformanceConfig',
    'ProcessingProfile',
    'EngineType',
    'ContentTypePreset',
    'create_fast_config',
    'create_balanced_config', 
    'create_accurate_config',
    'load_config_from_env'
]