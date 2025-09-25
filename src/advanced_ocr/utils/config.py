# src/advanced_ocr/utils/config.py
"""
Configuration management utilities for Advanced OCR System.
Handles loading, merging, and validating configuration files.
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
from copy import deepcopy
import logging

# Import from our exceptions module instead of defining here
from ..exceptions import ConfigurationError

# Get logger for this module
logger = logging.getLogger(__name__)


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


def merge_configs(base_config: Dict[str, Any], 
                 override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two configuration dictionaries.
    
    Args:
        base_config: Base configuration dictionary
        override_config: Configuration dictionary to merge on top
        
    Returns:
        New dictionary with merged configuration
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


def get_default_config() -> Dict[str, Any]:
    """
    Get the minimal fallback configuration.
    Note: Detailed engine configs come from engines.yaml
    
    Returns:
        Dictionary containing the fallback configuration
    """
    return {
        "engines": {
            # Minimal fallback - engines.yaml provides detailed configs
            "paddleocr": {"enabled": True, "priority": 1},
            "easyocr": {"enabled": True, "priority": 2},
            "tesseract": {"enabled": True, "priority": 3},
            "trocr": {"enabled": True, "priority": 4}
        },
        "preprocessing": {
            "quality_analysis": True,
            "enhancement": True
        },
        "quality_analyzer": {
            "sharpness_threshold": 100.0,
            "noise_threshold": 0.1,
            "contrast_threshold": 0.3,
            "brightness_min": 50,
            "brightness_max": 200
        },
        "image_enhancer": {
            "denoise_strength": 3,
            "sharpen_strength": 0.5,
            "contrast_factor": 1.2,
            "brightness_adjustment": 0
        },
        "engine_manager": {
            "parallel_processing": True,
            "max_workers": 3,
            "timeout": 300
        }
    }


def load_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Load configuration with fallbacks.
    
    Loading order:
    1. Start with minimal defaults
    2. Load default.yaml (general settings)
    3. Load engines.yaml (detailed engine configs) 
    4. Load custom config if provided
    
    Args:
        config_path: Optional path to custom configuration file
        
    Returns:
        Complete configuration dictionary
    """
    try:
        # Start with defaults
        config = get_default_config()
        logger.debug("Started with default configuration")
        
        # Load default.yaml from resources (general settings)
        try:
            default_resource = get_resource_path("default.yaml")
            if default_resource.exists():
                resource_config = load_yaml_file(default_resource)
                config = merge_configs(config, resource_config)
                logger.debug("Loaded default.yaml from resources")
            else:
                logger.debug("default.yaml not found, using fallback defaults")
        except Exception as e:
            logger.debug(f"Could not load default.yaml: {e}")
        
        # Load engines.yaml from resources (detailed engine configurations)
        try:
            engines_resource = get_resource_path("engines.yaml")
            if engines_resource.exists():
                engines_config = load_yaml_file(engines_resource)
                config = merge_configs(config, engines_config)
                logger.debug("Loaded engines.yaml from resources")
                logger.info(f"Loaded engine configurations for: {list(engines_config.get('engines', {}).keys())}")
            else:
                logger.warning("engines.yaml not found, using minimal engine defaults")
        except Exception as e:
            logger.debug(f"Could not load engines.yaml: {e}")
        
        # Load custom config if provided (user overrides)
        if config_path:
            config_path = Path(config_path)
            if config_path.exists():
                try:
                    custom_config = load_yaml_file(config_path)
                    config = merge_configs(config, custom_config)
                    logger.info(f"Loaded custom configuration from: {config_path}")
                except Exception as e:
                    logger.error(f"Failed to load custom config {config_path}: {e}")
            else:
                logger.warning(f"Custom config file not found: {config_path}")
        
        # Log final configuration summary
        enabled_engines = [
            name for name, settings in config.get("engines", {}).items()
            if settings.get("enabled", False)
        ]
        logger.info(f"Configuration loaded successfully. Enabled engines: {enabled_engines}")
        
        return config
        
    except Exception as e:
        logger.error(f"Configuration loading failed: {e}")
        logger.warning("Using minimal fallback configuration")
        return get_default_config()


def get_config_value(config: Dict[str, Any], 
                    key_path: str, 
                    default: Any = None) -> Any:
    """
    Get a nested configuration value using dot notation.
    
    Args:
        config: Configuration dictionary
        key_path: Dot-separated path to the config value
        default: Default value if key is not found
        
    Returns:
        Configuration value or default
    """
    keys = key_path.split('.')
    current = config
    
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    
    return current


def validate_engine_config(config: Dict[str, Any]) -> bool:
    """
    Validate that engine configuration is valid.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        True if valid
        
    Raises:
        ConfigurationError: If configuration is invalid
    """
    if "engines" not in config:
        raise ConfigurationError("Missing 'engines' section in configuration")
    
    engines = config["engines"]
    if not isinstance(engines, dict):
        raise ConfigurationError("'engines' section must be a dictionary")
    
    if not engines:
        raise ConfigurationError("No engines configured")
    
    # Check that at least one engine is enabled
    enabled_engines = [name for name, settings in engines.items() 
                      if settings.get('enabled', False)]
    
    if not enabled_engines:
        logger.warning("No engines are enabled in configuration")
        return False
    
    logger.info(f"Configuration validation passed. {len(enabled_engines)} engines enabled.")
    return True


# Export main functions
__all__ = [
    'load_config',
    'get_config_value', 
    'merge_configs',
    'validate_engine_config',
    'get_default_config',
    'get_resource_path',
    'load_yaml_file'
]