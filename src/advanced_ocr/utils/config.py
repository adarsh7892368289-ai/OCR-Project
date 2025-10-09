"""Configuration management for the OCR library.

Handles loading, merging, and validating configuration from YAML files.
Configuration priority: defaults → default.yaml → engines.yaml → custom config.

Examples
--------
    from advanced_ocr.utils.config import load_config, get_config_value
    
    config = load_config()
    config = load_config("custom_config.yaml")
    
    threshold = get_config_value(config, 'quality_analyzer.sharpness_threshold', 0.1)
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
from copy import deepcopy
import logging

from ..exceptions import ConfigurationError

logger = logging.getLogger(__name__)


def get_resource_path(filename: str) -> Path:
    """Get path to a resource file in the resources directory."""
    current_dir = Path(__file__).parent
    resources_dir = current_dir.parent / "resources"
    return resources_dir / filename


def load_yaml_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Load and parse a YAML configuration file."""
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
            raise ConfigurationError(
                f"Configuration file must contain a dictionary, got {type(content)}"
            )
            
        return content
        
    except yaml.YAMLError as e:
        raise ConfigurationError(f"Failed to parse YAML file {file_path}: {e}")
    except Exception as e:
        raise ConfigurationError(f"Failed to load configuration file {file_path}: {e}")


def merge_configs(base_config: Dict[str, Any], 
                 override_config: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two configuration dictionaries, override takes precedence."""
    merged = deepcopy(base_config)
    
    def _deep_merge(base_dict: Dict[str, Any], override_dict: Dict[str, Any]) -> None:
        """Recursively merge dictionaries in place."""
        for key, value in override_dict.items():
            if (key in base_dict 
                and isinstance(base_dict[key], dict) 
                and isinstance(value, dict)):
                _deep_merge(base_dict[key], value)
            else:
                base_dict[key] = deepcopy(value)
    
    _deep_merge(merged, override_config)
    return merged


def get_default_config() -> Dict[str, Any]:
    """Get hardcoded default configuration as fallback."""
    return {
        "engines": {
            "paddleocr": {"enabled": True},
            "easyocr": {"enabled": True},
            "tesseract": {"enabled": True},
            "trocr": {"enabled": True}
        },
        "quality_analyzer": {
            "sharpness_threshold": 0.1,
            "noise_threshold": 0.3,
            "contrast_threshold": 0.4,
            "brightness_range": [0.2, 0.8],
            "enhancement_threshold": 0.6
        },
        "image_enhancer": {
            "denoise_strength": 3,
            "sharpen_strength": 0.5,
            "contrast_factor": 1.2,
            "brightness_adjustment": 0,
            "noise_reduction": {"enabled": True, "method": "gaussian"},
            "contrast_enhancement": {"enabled": True, "method": "clahe"},
            "sharpening": {"enabled": True, "method": "unsharp_mask"}
        },
        "engine_manager": {
            "parallel_processing": True,
            "max_workers": 4,
            "timeout_per_engine": 30
        }
    }


def get_config_value(config: Dict[str, Any], 
                    key_path: str, 
                    default: Any = None) -> Any:
    """Get nested configuration value using dot notation (e.g., 'engines.paddleocr.enabled')."""
    keys = key_path.split('.')
    current = config
    
    for key in keys:
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    
    return current


def load_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """Load and merge configuration from multiple sources."""
    try:
        config = get_default_config()
        logger.debug("Started with default configuration")
        
        # Load default.yaml from resources
        try:
            default_resource = get_resource_path("default.yaml")
            if default_resource.exists():
                resource_config = load_yaml_file(default_resource)
                config = merge_configs(config, resource_config)
                logger.debug("Loaded default.yaml from resources")
        except Exception as e:
            logger.debug(f"Could not load default.yaml: {e}")
        
        # Load engines.yaml from resources
        try:
            engines_resource = get_resource_path("engines.yaml")
            if engines_resource.exists():
                engines_config = load_yaml_file(engines_resource)
                config = merge_configs(config, engines_config)
                logger.debug("Loaded engines.yaml from resources")
        except Exception as e:
            logger.debug(f"Could not load engines.yaml: {e}")
        
        # Load custom config if provided
        if config_path:
            config_path = Path(config_path)
            if config_path.exists():
                try:
                    custom_config = load_yaml_file(config_path)
                    config = merge_configs(config, custom_config)
                    logger.info(f"Loaded custom configuration from: {config_path}")
                except Exception as e:
                    logger.error(f"Failed to load custom config {config_path}: {e}")
        
        # Validate configuration
        validate_engine_config(config)
        
        enabled_engines = [
            name for name, settings in config.get("engines", {}).items()
            if settings.get("enabled", False)
        ]
        logger.info(f"Configuration loaded successfully. Enabled engines: {enabled_engines}")
        
        return config
        
    except ConfigurationError as e:
        logger.error(f"Configuration validation failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Configuration loading failed: {e}")
        logger.warning("Using minimal fallback configuration")
        return get_default_config()


def validate_engine_config(config: Dict[str, Any]) -> bool:
    """Validate engine configuration structure and settings."""
    if "engines" not in config:
        raise ConfigurationError("Missing 'engines' section in configuration")
    
    engines = config["engines"]
    if not isinstance(engines, dict):
        raise ConfigurationError("'engines' section must be a dictionary")
    
    if not engines:
        raise ConfigurationError("No engines configured")
    
    # Check that at least one engine is enabled
    enabled_engines = [
        name for name, settings in engines.items() 
        if settings.get('enabled', False)
    ]
    
    if not enabled_engines:
        logger.warning("No engines are enabled in configuration")
        return False
    
    logger.info(f"Configuration validation passed. {len(enabled_engines)} engines enabled.")
    return True


__all__ = [
    'load_config',
    'merge_configs',
    'validate_engine_config',
    'get_default_config',
    'get_resource_path',
    'load_yaml_file',
    'get_config_value'
]