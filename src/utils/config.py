# src/utils/config.py

import json
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
import os
from dataclasses import dataclass, asdict

@dataclass
class OCRConfig:
    """OCR System Configuration"""
    
    # Engine configurations
    engines: Dict[str, Dict[str, Any]] = None
    
    # Processing options
    preprocessing: Dict[str, Any] = None
    postprocessing: Dict[str, Any] = None
    
    # System settings
    parallel_processing: bool = True
    max_workers: int = 3
    log_level: str = "INFO"
    
    # Output settings
    output: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.engines is None:
            self.engines = {
                "tesseract": {
                    "psm": 6,
                    "lang": "eng",
                    "whitelist": None,
                    "blacklist": None
                },
                "easyocr": {
                    "languages": ["en"],
                    "gpu": True,
                    "model_dir": None
                },
                "trocr": {
                    "model_name": "microsoft/trocr-base-handwritten",
                    "device": "auto",
                    "max_new_tokens": 128,
                    "batch_size": 4
                }
            }
            
        if self.preprocessing is None:
            self.preprocessing = {
                "enhancement_level": "medium",
                "preserve_aspect_ratio": True,
                "angle_range": 45,
                "angle_step": 0.5,
                "min_text_size": 10,
                "max_text_size": 300
            }
            
        if self.postprocessing is None:
            self.postprocessing = {
                "min_confidence": 0.5,
                "min_word_length": 2,
                "language": "en",
                "domain_vocabulary": [],
                "line_height_threshold": 1.5,
                "paragraph_gap_threshold": 2.0
            }
            
        if self.output is None:
            self.output = {
                "preserve_formatting": True,
                "include_confidence": False
            }

class Config:
    """Configuration manager for OCR system"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_data = {}
        self.default_config = OCRConfig()
        
        if config_path:
            self.load_from_file(config_path)
        else:
            # Load default configuration
            self.config_data = asdict(self.default_config)
            
    def load_from_file(self, config_path: str):
        """Load configuration from file"""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        try:
            with open(config_path, 'r') as f:
                if config_path.suffix.lower() in ['.yaml', '.yml']:
                    self.config_data = yaml.safe_load(f)
                elif config_path.suffix.lower() == '.json':
                    self.config_data = json.load(f)
                else:
                    raise ValueError("Unsupported configuration file format")
                    
            # Merge with defaults
            default_dict = asdict(self.default_config)
            self.config_data = self._deep_merge(default_dict, self.config_data)
            
        except Exception as e:
            raise RuntimeError(f"Error loading configuration: {e}")
            
    def _deep_merge(self, default: dict, override: dict) -> dict:
        """Deep merge two dictionaries"""
        result = default.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
                
        return result
        
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation"""
        keys = key.split('.')
        value = self.config_data
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value
        
    def set(self, key: str, value: Any):
        """Set configuration value using dot notation"""
        keys = key.split('.')
        config = self.config_data
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
            
        config[keys[-1]] = value
        
    def save_to_file(self, config_path: str):
        """Save configuration to file"""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                yaml.dump(self.config_data, f, default_flow_style=False)
            elif config_path.suffix.lower() == '.json':
                json.dump(self.config_data, f, indent=2)
            else:
                raise ValueError("Unsupported configuration file format")
