"""
Configuration management utilities for Advanced OCR System.

This module handles loading, merging, and validating configuration files
for the OCR library, including default configurations and user customizations.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
from copy import deepcopy
import logging

# Get logger for this module
logger = logging.getLogger(__name__)

class ConfigurationError(Exception):
    """Raised when there are configuration-related errors"""
    pass

def get_resource_path(filename: str) -> Path:
    """
    Get the full path to a resource file in the resources directory.
    
    Args:
        filename: Name of the resource file
        
    Returns:
        Path object pointing to the resource file
    """
    # Get the directory containing this config.py file
    current_dir = Path(__file__).parent
    # Navigate to the resources directory: utils -> advanced_ocr -> resources
    resources_dir = current_dir.parent / "resources"
    return resources_dir / filename

def load_yaml_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a YAML file safely with proper error handling.
    
    Args:
        file_path: Path to the YAML file
        
    Returns:
        Dictionary containing the loaded YAML data
        
    Raises:
        ConfigurationError: If the file cannot be loaded or parsed
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise ConfigurationError(f"Configuration file not found: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = yaml.safe_load(file)
            
        if content is None:
            logger.warning(f"Empty configuration file: {file_path}")
            return {}
            
        if not isinstance(content, dict):
            raise ConfigurationError(f"Configuration file must contain a dictionary, got {type(content)}")
            
        return content
        
    except yaml.YAMLError as e:
        raise ConfigurationError(f"Failed to parse YAML file {file_path}: {e}")
    except Exception as e:
        raise ConfigurationError(f"Failed to load configuration file {file_path}: {e}")

def load_default_config() -> Dict[str, Any]:
    """
    Load the default configuration from resources/default.yaml.
    
    Returns:
        Dictionary containing the default configuration
        
    Raises:
        ConfigurationError: If the default config cannot be loaded
    """
    try:
        default_config_path = get_resource_path("default.yaml")
        config = load_yaml_file(default_config_path)
        logger.debug("Loaded default configuration successfully")
        return config
        
    except Exception as e:
        logger.error(f"Failed to load default configuration: {e}")
        # Return minimal fallback config
        return get_minimal_fallback_config()

def load_engine_config() -> Dict[str, Any]:
    """
    Load the engine configuration from resources/engines.yaml.
    
    Returns:
        Dictionary containing the engine configuration
        
    Raises:
        ConfigurationError: If the engine config cannot be loaded
    """
    try:
        engine_config_path = get_resource_path("engines.yaml")
        config = load_yaml_file(engine_config_path)
        logger.debug("Loaded engine configuration successfully")
        return config
        
    except Exception as e:
        logger.error(f"Failed to load engine configuration: {e}")
        # Return minimal fallback engine config
        return get_minimal_engine_config()

def load_custom_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a custom user configuration file.
    
    Args:
        config_path: Path to the custom configuration file
        
    Returns:
        Dictionary containing the custom configuration
        
    Raises:
        ConfigurationError: If the config cannot be loaded
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise ConfigurationError(f"Custom configuration file not found: {config_path}")
    
    try:
        config = load_yaml_file(config_path)
        logger.info(f"Loaded custom configuration from: {config_path}")
        return config
        
    except Exception as e:
        raise ConfigurationError(f"Failed to load custom configuration: {e}")

def merge_configs(base_config: Dict[str, Any], 
                 override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two configuration dictionaries.
    
    The override_config values will take precedence over base_config values.
    Nested dictionaries are merged recursively.
    
    Args:
        base_config: Base configuration dictionary
        override_config: Configuration dictionary to merge on top
        
    Returns:
        New dictionary with merged configuration
        
    Example:
        >>> base = {"a": 1, "b": {"x": 1, "y": 2}}
        >>> override = {"b": {"x": 10}, "c": 3}
        >>> result = merge_configs(base, override)
        >>> # Result: {"a": 1, "b": {"x": 10, "y": 2}, "c": 3}
    """
    # Create deep copy to avoid modifying original
    merged = deepcopy(base_config)
    
    def _deep_merge(base_dict: Dict[str, Any], override_dict: Dict[str, Any]) -> None:
        """Recursively merge dictionaries in place"""
        for key, value in override_dict.items():
            if (key in base_dict 
                and isinstance(base_dict[key], dict) 
                and isinstance(value, dict)):
                # Recursively merge nested dictionaries
                _deep_merge(base_dict[key], value)
            else:
                # Override the value
                base_dict[key] = deepcopy(value)
    
    _deep_merge(merged, override_config)
    return merged

def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate the configuration dictionary has required fields.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        True if configuration is valid
        
    Raises:
        ConfigurationError: If configuration is invalid
    """
    required_sections = ['processing', 'performance', 'engines']
    
    for section in required_sections:
        if section not in config:
            raise ConfigurationError(f"Missing required configuration section: {section}")
    
    # Validate processing section
    processing = config.get('processing', {})
    if 'default_engine' not in processing:
        raise ConfigurationError("Missing 'default_engine' in processing configuration")
    
    # Validate engines section
    engines = config.get('engines', {})
    if not engines:
        raise ConfigurationError("No engines configured")
    
    # Check that at least one engine is enabled
    enabled_engines = [name for name, settings in engines.items() 
                      if settings.get('enabled', False)]
    
    if not enabled_engines:
        logger.warning("No engines are enabled in configuration")
    
    return True

def get_config_value(config: Dict[str, Any], 
                    key_path: str, 
                    default: Any = None) -> Any:
    """
    Get a nested configuration value using dot notation.
    
    Args:
        config: Configuration dictionary
        key_path: Dot-separated path to the config value (e.g., "processing.strategy")
        default: Default value if key is not found
        
    Returns:
        Configuration value or default
        
    Example:
        >>> config = {"processing": {"strategy": "balanced"}}
        >>> value = get_config_value(config, "processing.strategy", "fast")
        >>> # Returns: "balanced"
    """
    keys = key_path.split('.')
    current = config
    
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    
    return current

def set_config_value(config: Dict[str, Any], 
                    key_path: str, 
                    value: Any) -> None:
    """
    Set a nested configuration value using dot notation.
    
    Args:
        config: Configuration dictionary to modify
        key_path: Dot-separated path to set (e.g., "processing.strategy")
        value: Value to set
        
    Example:
        >>> config = {"processing": {"strategy": "balanced"}}
        >>> set_config_value(config, "processing.min_confidence", 0.8)
        >>> # config now contains: {"processing": {"strategy": "balanced", "min_confidence": 0.8}}
    """
    keys = key_path.split('.')
    current = config
    
    # Navigate to the parent dictionary
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        elif not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    
    # Set the final value
    current[keys[-1]] = value

def save_config(config: Dict[str, Any], 
               file_path: Union[str, Path]) -> None:
    """
    Save configuration to a YAML file.
    
    Args:
        config: Configuration dictionary to save
        file_path: Path where to save the configuration
        
    Raises:
        ConfigurationError: If the file cannot be saved
    """
    file_path = Path(file_path)
    
    try:
        # Create directory if it doesn't exist
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as file:
            yaml.safe_dump(config, file, default_flow_style=False, indent=2, sort_keys=True)
            
        logger.info(f"Configuration saved to: {file_path}")
        
    except Exception as e:
        raise ConfigurationError(f"Failed to save configuration to {file_path}: {e}")

def get_minimal_fallback_config() -> Dict[str, Any]:
    """
    Get a minimal fallback configuration when default config cannot be loaded.
    
    Returns:
        Dictionary with minimal configuration
    """
    return {
        "library": {
            "name": "Advanced OCR System",
            "version": "1.0.0"
        },
        "processing": {
            "default_engine": "paddleocr",
            "strategy": "balanced",
            "min_confidence": 0.6,
            "preprocessing": {
                "enabled": True,
                "auto_enhance": True
            },
            "postprocessing": {
                "enabled": True,
                "noise_removal": True
            }
        },
        "performance": {
            "max_workers": 4,
            "batch_size": 8,
            "processing_timeout": 30,
            "cache_enabled": True
        },
        "quality": {
            "enabled": True,
            "enhancement_threshold": 0.6,
            "sharpness_threshold": 100.0,
            "contrast_threshold": 0.3
        },
        "enhancement": {
            "denoise": True,
            "sharpen": True,
            "contrast_enhance": True
        },
        "output": {
            "format": "text",
            "include_confidence": True
        },
        "logging": {
            "level": "INFO",
            "console": True
        },
        "engines": {
            "paddleocr": {"enabled": True, "priority": 1},
            "easyocr": {"enabled": True, "priority": 2},
            "tesseract": {"enabled": True, "priority": 3},
            "trocr": {"enabled": False, "priority": 4}
        }
    }

def get_minimal_engine_config() -> Dict[str, Any]:
    """
    Get a minimal fallback engine configuration.
    
    Returns:
        Dictionary with minimal engine configuration
    """
    return {
        "engines": {
            "paddleocr": {
                "enabled": True,
                "priority": 1,
                "description": "PaddleOCR - High accuracy OCR",
                "features": ["text_detection", "text_recognition"],
                "config": {
                    "use_angle_cls": True,
                    "lang": "en",
                    "use_gpu": False
                }
            },
            "easyocr": {
                "enabled": True,
                "priority": 2,
                "description": "EasyOCR - Easy to use OCR",
                "features": ["text_detection", "text_recognition"],
                "config": {
                    "gpu": False
                }
            },
            "tesseract": {
                "enabled": True,
                "priority": 3,
                "description": "Tesseract - Traditional OCR",
                "features": ["text_recognition"],
                "config": {
                    "psm": 6,
                    "oem": 3,
                    "lang": "eng"
                }
            },
            "trocr": {
                "enabled": False,
                "priority": 4,
                "description": "TrOCR - Transformer-based OCR",
                "features": ["handwriting_recognition"],
                "config": {
                    "model_name": "microsoft/trocr-base-printed",
                    "device": "cpu"
                }
            }
        }
    }

def create_user_config_template(file_path: Union[str, Path]) -> None:
    """
    Create a user configuration template file.
    
    Args:
        file_path: Path where to create the template
    """
    template = {
        "# Advanced OCR System - User Configuration": None,
        "# Customize these settings as needed": None,
        "": None,
        "processing": {
            "default_engine": "paddleocr",
            "strategy": "balanced",  # Options: fast, balanced, accurate
            "min_confidence": 0.6,
            "preprocessing": {
                "enabled": True,
                "auto_enhance": True
            }
        },
        "performance": {
            "max_workers": 4,
            "batch_size": 8
        },
        "engines": {
            "paddleocr": {"enabled": True},
            "easyocr": {"enabled": True}, 
            "tesseract": {"enabled": True},
            "trocr": {"enabled": False}  # Heavy dependencies
        }
    }
    
    # Clean up None values (they were just for comments)
    clean_template = {k: v for k, v in template.items() if v is not None and not k.startswith("#")}
    
    save_config(clean_template, file_path)
    logger.info(f"Created configuration template at: {file_path}")

# Environment variable support
def get_config_from_env() -> Dict[str, Any]:
    """
    Get configuration overrides from environment variables.
    
    Environment variables should be prefixed with OCR_ and use underscores.
    Example: OCR_PROCESSING_STRATEGY=accurate
    
    Returns:
        Dictionary with configuration overrides from environment
    """
    config_overrides = {}
    
    # Mapping of environment variables to config paths
    env_mappings = {
        'OCR_DEFAULT_ENGINE': 'processing.default_engine',
        'OCR_STRATEGY': 'processing.strategy', 
        'OCR_MIN_CONFIDENCE': 'processing.min_confidence',
        'OCR_MAX_WORKERS': 'performance.max_workers',
        'OCR_BATCH_SIZE': 'performance.batch_size',
        'OCR_LOG_LEVEL': 'logging.level',
        'OCR_CACHE_ENABLED': 'performance.cache_enabled'
    }
    
    for env_var, config_path in env_mappings.items():
        value = os.getenv(env_var)
        if value is not None:
            # Try to convert to appropriate type
            if value.lower() in ('true', 'false'):
                value = value.lower() == 'true'
            elif value.isdigit():
                value = int(value)
            elif value.replace('.', '').isdigit():
                value = float(value)
                
            set_config_value(config_overrides, config_path, value)
            logger.debug(f"Applied environment override: {env_var}={value}")
    
    return config_overrides

# Main configuration loading function
def load_config(config_path: Optional[Union[str, Path]] = None, 
               include_env_overrides: bool = True) -> Dict[str, Any]:
    """
    Load complete configuration with all sources merged.
    
    Loading order (later sources override earlier ones):
    1. Default configuration (resources/default.yaml)
    2. Engine configuration (resources/engines.yaml) 
    3. Custom configuration file (if provided)
    4. Environment variable overrides (if enabled)
    
    Args:
        config_path: Optional path to custom configuration file
        include_env_overrides: Whether to include environment variable overrides
        
    Returns:
        Complete merged configuration dictionary
    """
    try:
        # Start with default configuration
        config = load_default_config()
        
        # Merge engine configuration
        engine_config = load_engine_config()
        config = merge_configs(config, engine_config)
        
        # Merge custom configuration if provided
        if config_path:
            custom_config = load_custom_config(config_path)
            config = merge_configs(config, custom_config)
        
        # Merge environment overrides
        if include_env_overrides:
            env_config = get_config_from_env()
            if env_config:
                config = merge_configs(config, env_config)
                logger.debug("Applied environment variable overrides")
        
        # Validate the final configuration
        validate_config(config)
        
        logger.info("Configuration loaded and validated successfully")
        return config
        
    except Exception as e:
        logger.error(f"Configuration loading failed: {e}")
        logger.warning("Using minimal fallback configuration")
        return get_minimal_fallback_config()