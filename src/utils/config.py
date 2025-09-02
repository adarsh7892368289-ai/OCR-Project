import yaml
import os
from typing import Dict, Any
from pathlib import Path

def load_config(config_path: str = "config/development.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file"""
    
    # Default configuration
    default_config = {
        'engines': {
            'paddle_ocr': {
                'enabled': True,
                'use_gpu': True,
                'language': 'en',
                'use_angle_cls': True
            },
            'trocr': {
                'enabled': True,
                'model_name': 'microsoft/trocr-base-handwritten',
                'use_gpu': True
            },
            'easyocr': {
                'enabled': True,
                'languages': ['en'],
                'use_gpu': True
            },
            'tesseract': {
                'enabled': False,
                'config': '--psm 6'
            }
        },
        'preprocessing': {
            'enhance_contrast': True,
            'denoise': True,
            'max_dimension': 2048,
            'dpi_threshold': 300
        },
        'output': {
            'save_json': True,
            'save_annotated_images': False,
            'confidence_threshold': 0.5
        },
        'performance': {
            'batch_size': 1,
            'max_workers': 4,
            'gpu_memory_limit': 0.8
        }
    }
    
    # Try to load config file
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                file_config = yaml.safe_load(f) or {}
            
            # Merge with default config
            merged_config = deep_merge(default_config, file_config)
            return merged_config
            
        except Exception as e:
            print(f"Warning: Could not load config file {config_path}: {e}")
            print("Using default configuration...")
            return default_config
    else:
        print(f"Config file {config_path} not found. Using default configuration...")
        return default_config

def deep_merge(base_dict: Dict, update_dict: Dict) -> Dict:
    """Deep merge two dictionaries"""
    result = base_dict.copy()
    
    for key, value in update_dict.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result

def create_default_config(config_path: str = "config/development.yaml"):
    """Create a default configuration file"""
    
    config_content = """# Advanced OCR System Configuration

# OCR Engine Configuration
engines:
  paddle_ocr:
    enabled: true
    use_gpu: true
    language: 'en'
    use_angle_cls: true
    
  trocr:
    enabled: true
    model_name: 'microsoft/trocr-base-handwritten'
    use_gpu: true
    
  easyocr:
    enabled: true
    languages: ['en']
    use_gpu: true
    
  tesseract:
    enabled: false
    config: '--psm 6'

# Image Preprocessing Configuration
preprocessing:
  enhance_contrast: true
  denoise: true
  max_dimension: 2048
  dpi_threshold: 300

# Output Configuration
output:
  save_json: true
  save_annotated_images: false
  confidence_threshold: 0.5
  
# Performance Configuration
performance:
  batch_size: 1
  max_workers: 4
  gpu_memory_limit: 0.8
"""
    
    # Ensure directory exists
    Path(config_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Write config file
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    print(f"Created default configuration file: {config_path}")
