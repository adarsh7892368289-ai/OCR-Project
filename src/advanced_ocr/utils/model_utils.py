# src/advanced_ocr/utils/model_utils.py
"""
Advanced OCR System - Model Management Utilities

This module provides ML model loading, caching, and management utilities for the OCR system.
Handles model downloads, version validation, memory management, and caching without
performing any model inference or selection logic.

Classes:
    ModelCache: In-memory model caching with LRU eviction
    ModelDownloader: Model file downloading with validation
    ModelVersionManager: Model version compatibility checking
    ModelLoader: Main model loading orchestrator
    cached_model_load: Decorator for automatic model caching

Example:
    >>> loader = ModelLoader(config, "models")
    >>> model = loader.load_model("tesseract_model", "pytorch")
    >>> print(f"Model loaded successfully: {model is not None}")

    >>> cache_info = loader.get_cache_info()
    >>> print(f"Cached models: {cache_info['total_models']}")
"""

import os
import hashlib
import pickle
import json
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, Any, Optional, Union, Tuple, List, Callable
import threading
import time
import gc
from functools import wraps

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    import transformers
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    transformers = None

try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    tf = None

try:
    import sklearn
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    sklearn = None
    
try:
    import gdown
    GDOWN_AVAILABLE = True
except ImportError:
    GDOWN_AVAILABLE = False

try:
    from craft_text_detector import CRAFT
    CRAFT_AVAILABLE = True
except ImportError:
    CRAFT_AVAILABLE = False    


from ..config import OCRConfig
from ..utils.logger import OCRLogger


class ModelCache:
    """
    Manages in-memory model caching with LRU eviction and memory monitoring.
    """
    
    def __init__(self, max_memory_mb: int = 2048, max_models: int = 5):
        """
        Initialize model cache with memory and count limits.
        
        Args:
            max_memory_mb (int): Maximum memory usage in MB
            max_models (int): Maximum number of cached models
        """
        self.max_memory_mb = max_memory_mb
        self.max_models = max_models
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._access_times: Dict[str, float] = {}
        self._lock = threading.RLock()
        self.logger = OCRLogger("ModelCache")
    
    def get(self, key: str) -> Optional[Any]:
        """
        Retrieve model from cache.
        
        Args:
            key (str): Model cache key
            
        Returns:
            Optional[Any]: Cached model or None if not found
        """
        with self._lock:
            if key in self._cache:
                self._access_times[key] = time.time()
                self.logger.debug(f"Model cache hit: {key}")
                return self._cache[key]['model']
            
            self.logger.debug(f"Model cache miss: {key}")
            return None
    
    def put(self, key: str, model: Any, size_mb: Optional[float] = None) -> bool:
        """
        Store model in cache with automatic eviction if needed.
        
        Args:
            key (str): Model cache key
            model (Any): Model object to cache
            size_mb (Optional[float]): Model size in MB (estimated if None)
            
        Returns:
            bool: True if successfully cached, False otherwise
        """
        with self._lock:
            # Estimate model size if not provided
            if size_mb is None:
                size_mb = self._estimate_model_size(model)
            
            # Check if we need to evict models
            current_memory = self._get_current_memory_usage()
            
            if (current_memory + size_mb > self.max_memory_mb or 
                len(self._cache) >= self.max_models):
                
                if not self._evict_lru_models(size_mb):
                    self.logger.warning(f"Failed to evict enough memory for model: {key}")
                    return False
            
            # Store the model
            self._cache[key] = {
                'model': model,
                'size_mb': size_mb,
                'created_at': time.time()
            }
            self._access_times[key] = time.time()
            
            self.logger.info(f"Model cached: {key} ({size_mb:.1f} MB)")
            return True
    
    def remove(self, key: str) -> bool:
        """
        Remove model from cache.
        
        Args:
            key (str): Model cache key
            
        Returns:
            bool: True if model was removed, False if not found
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                del self._access_times[key]
                self.logger.debug(f"Model removed from cache: {key}")
                return True
            return False
    
    def clear(self) -> None:
        """Clear all cached models."""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
            gc.collect()
            self.logger.info("Model cache cleared")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dict[str, Any]: Cache statistics
        """
        with self._lock:
            return {
                'total_models': len(self._cache),
                'memory_usage_mb': self._get_current_memory_usage(),
                'max_memory_mb': self.max_memory_mb,
                'max_models': self.max_models,
                'models': list(self._cache.keys())
            }
    
    def _estimate_model_size(self, model: Any) -> float:
        """Estimate model size in MB."""
        try:
            if TORCH_AVAILABLE and hasattr(model, 'parameters'):
                # PyTorch model
                param_size = sum(p.numel() * p.element_size() for p in model.parameters())
                buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
                return (param_size + buffer_size) / (1024 * 1024)
            
            elif hasattr(model, 'count_params'):
                # TensorFlow/Keras model
                return model.count_params() * 4 / (1024 * 1024)  # Assume float32
            
            else:
                # Generic estimation using pickle
                import sys
                return sys.getsizeof(pickle.dumps(model)) / (1024 * 1024)
        
        except Exception:
            return 100.0  # Default estimate
    
    def _get_current_memory_usage(self) -> float:
        """Get current memory usage of cached models in MB."""
        return sum(info['size_mb'] for info in self._cache.values())
    
    def _evict_lru_models(self, required_mb: float) -> bool:
        """
        Evict least recently used models to free memory.
        
        Args:
            required_mb (float): Required memory in MB
            
        Returns:
            bool: True if enough memory was freed
        """
        current_memory = self._get_current_memory_usage()
        target_memory = self.max_memory_mb - required_mb
        
        if current_memory <= target_memory:
            return True
        
        # Sort by access time (oldest first)
        sorted_models = sorted(self._access_times.items(), key=lambda x: x[1])
        
        for key, _ in sorted_models:
            if current_memory <= target_memory:
                break
            
            model_size = self._cache[key]['size_mb']
            del self._cache[key]
            del self._access_times[key]
            current_memory -= model_size
            
            self.logger.debug(f"Evicted model from cache: {key} ({model_size:.1f} MB)")
        
        # Force garbage collection
        gc.collect()
        
        return current_memory <= target_memory


class ModelDownloader:
    """
    Handles model downloads with progress tracking and validation.
    """
    
    def __init__(self, models_dir: str = "models"):
        """
        Initialize model downloader.
        
        Args:
            models_dir (str): Directory to store downloaded models
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.logger = OCRLogger("ModelDownloader")
    
    def download_model(self, url: str, filename: str, 
                      expected_hash: Optional[str] = None,
                      chunk_size: int = 8192) -> Path:
        """
        Download model file with validation.
        
        Args:
            url (str): Download URL
            filename (str): Local filename
            expected_hash (Optional[str]): Expected SHA256 hash for validation
            chunk_size (int): Download chunk size in bytes
            
        Returns:
            Path: Path to downloaded file
            
        Raises:
            ValueError: If hash validation fails
            urllib.error.URLError: If download fails
        """
        filepath = self.models_dir / filename
        
        # Check if file already exists and is valid
        if filepath.exists() and expected_hash:
            if self._validate_file_hash(filepath, expected_hash):
                self.logger.info(f"Model already exists and is valid: {filename}")
                return filepath
        
        self.logger.info(f"Downloading model: {filename} from {url}")
        
        try:
            # Download with progress tracking
            with urllib.request.urlopen(url) as response:
                total_size = int(response.headers.get('content-length', 0))
                
                with open(filepath, 'wb') as f:
                    downloaded = 0
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            if downloaded % (chunk_size * 100) == 0:  # Log every ~800KB
                                self.logger.debug(f"Download progress: {progress:.1f}%")
            
            # Validate hash if provided
            if expected_hash:
                if not self._validate_file_hash(filepath, expected_hash):
                    filepath.unlink()  # Remove invalid file
                    raise ValueError(f"Hash validation failed for {filename}")
            
            self.logger.info(f"Model downloaded successfully: {filename}")
            return filepath
        
        except Exception as e:
            if filepath.exists():
                filepath.unlink()  # Clean up partial download
            raise e
    
    def _validate_file_hash(self, filepath: Path, expected_hash: str) -> bool:
        """
        Validate file SHA256 hash.
        
        Args:
            filepath (Path): Path to file
            expected_hash (str): Expected SHA256 hash
            
        Returns:
            bool: True if hash matches
        """
        try:
            sha256_hash = hashlib.sha256()
            with open(filepath, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(chunk)
            
            actual_hash = sha256_hash.hexdigest()
            return actual_hash.lower() == expected_hash.lower()
        
        except Exception:
            return False


class ModelVersionManager:
    """
    Manages model version compatibility and requirements.
    """
    
    def __init__(self, config: OCRConfig):
        """
        Initialize version manager.
        
        Args:
            config (OCRConfig): OCR configuration
        """
        self.config = config
        self.logger = OCRLogger("ModelVersionManager")
    
    def check_model_compatibility(self, model_name: str, 
                                model_version: str) -> Tuple[bool, str]:
        """
        Check if model version is compatible with current system.
        
        Args:
            model_name (str): Name of the model
            model_version (str): Version string
            
        Returns:
            Tuple[bool, str]: (is_compatible, reason)
        """
        # Get model requirements from config
        model_config = self.config.get(f"models.{model_name}", {})
        required_version = model_config.get("version", "any")
        
        if required_version == "any":
            return True, "No version constraint"
        
        # Simple version comparison (can be enhanced for semantic versioning)
        if model_version == required_version:
            return True, f"Version matches requirement: {required_version}"
        
        return False, f"Version mismatch: got {model_version}, required {required_version}"
    
    def check_framework_compatibility(self, framework: str, 
                                    min_version: Optional[str] = None) -> Tuple[bool, str]:
        """
        Check if ML framework is available and compatible.
        
        Args:
            framework (str): Framework name ('pytorch', 'tensorflow', 'transformers', 'sklearn')
            min_version (Optional[str]): Minimum required version
            
        Returns:
            Tuple[bool, str]: (is_compatible, reason)
        """
        # Handle framework name variations
        framework_lower = framework.lower()
        
        if framework_lower in ["torch", "pytorch"]:
            if not TORCH_AVAILABLE:
                return False, "PyTorch not available"
            if min_version and hasattr(torch, '__version__'):
                return True, f"PyTorch {torch.__version__} available"
            return True, "PyTorch available"
        
        elif framework_lower in ["tensorflow", "tf"]:
            if not TF_AVAILABLE:
                return False, "TensorFlow not available"
            if min_version and hasattr(tf, '__version__'):
                return True, f"TensorFlow {tf.__version__} available"
            return True, "TensorFlow available"
        
        elif framework_lower == "transformers":
            if not TRANSFORMERS_AVAILABLE:
                return False, "Transformers library not available"
            if min_version and hasattr(transformers, '__version__'):
                return True, f"Transformers {transformers.__version__} available"
            return True, "Transformers library available"
        
        elif framework_lower == "sklearn":
            if not SKLEARN_AVAILABLE:
                return False, "Scikit-learn not available"
            if min_version and SKLEARN_AVAILABLE:
                try:
                    import sklearn
                    return True, f"Scikit-learn {sklearn.__version__} available"
                except AttributeError:
                    return True, "Scikit-learn available"
            return True, "Scikit-learn available"
        
        return False, f"Unknown framework: {framework}"

class ModelLoader:
    """
    Main model loading orchestrator with caching and validation.
    """
    
    def __init__(self, config: OCRConfig, models_dir: str = "models"):
        """
        Initialize model loader.
        
        Args:
            config (OCRConfig): OCR configuration
            models_dir (str): Directory containing model files
        """
        self.config = config
        self.models_dir = Path(models_dir)
        
        # Get memory limit from config with fallback
        memory_limit = getattr(config, 'performance', None)
        if memory_limit and hasattr(memory_limit, 'memory_limit_mb'):
            max_memory = memory_limit.memory_limit_mb
        else:
            max_memory = config.get("performance.memory_limit_mb", 2048)
        
        self.cache = ModelCache(max_memory_mb=max_memory, max_models=5)
        self.downloader = ModelDownloader(models_dir)
        self.version_manager = ModelVersionManager(config)
        self.logger = OCRLogger("ModelLoader")
        
        # Model loading functions registry
        self._loaders: Dict[str, Callable] = {}
        self._register_default_loaders()
    
    def _download_craft_model(self, model_path: Path) -> None:
        """Download CRAFT model using gdown."""
        if not GDOWN_AVAILABLE:
            raise ImportError("gdown not available. Install with: pip install gdown")
        
        import gdown
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Downloading CRAFT model (17MB)...")
        
        # Official CRAFT model from Google Drive
        file_id = "1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ"
        gdown.download(f"https://drive.google.com/uc?id={file_id}", str(model_path), quiet=False)
        
        self.logger.info(f"CRAFT model downloaded: {model_path}")

    def register_loader(self, model_type: str, loader_func: Callable) -> None:
        """
        Register custom model loader function.
        
        Args:
            model_type (str): Model type identifier
            loader_func (Callable): Function to load model
        """
        self._loaders[model_type] = loader_func
        self.logger.debug(f"Registered loader for model type: {model_type}")
    
    def load_model(self, model_name: str, model_type: str = None, 
                   framework: str = None, force_reload: bool = False, **kwargs) -> Any:
        """
        Load model with caching and validation.
        
        Args:
            model_name (str): Name of the model
            model_type (str): Type of model (e.g., 'pytorch', 'tensorflow', 'pickle')
            framework (str): Framework name (alternative to model_type for compatibility)
            force_reload (bool): Force reload even if cached
            **kwargs: Additional arguments for model loading
            
        Returns:
            Any: Loaded model object
            
        Raises:
            ValueError: If model type not supported or loading fails
            FileNotFoundError: If model file not found
        """
        # Handle framework parameter for text_detector.py compatibility
        if framework and not model_type:
            model_type = framework
        elif not model_type and not framework:
            raise ValueError("Either model_type or framework parameter must be provided")
        
        cache_key = f"{model_name}_{model_type}"
        
        # Check cache first
        if not force_reload:
            cached_model = self.cache.get(cache_key)
            if cached_model is not None:
                self.logger.debug(f"Using cached model: {cache_key}")
                return cached_model
        
        # Check if loader is registered
        if model_type not in self._loaders:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Get model configuration
        model_config = self.config.get(f"models.{model_name}", {})
        
        # Check framework compatibility
        framework_name = model_config.get("framework", model_type)
        if framework_name:
            is_compatible, reason = self.version_manager.check_framework_compatibility(framework_name)
            if not is_compatible:
                raise ValueError(f"Framework compatibility issue: {reason}")
        
        # Load model
        try:
            self.logger.info(f"Loading model: {model_name} ({model_type})")
            start_time = time.time()
            
            model = self._loaders[model_type](model_name, model_config, **kwargs)
            
            load_time = time.time() - start_time
            self.logger.info(f"Model loaded in {load_time:.2f}s: {model_name}")
            
            # Cache the model
            self.cache.put(cache_key, model)
            
            return model
        
        except Exception as e:
            self.logger.error(f"Failed to load model {model_name}: {str(e)}")
            raise e
    
    def unload_model(self, model_name: str, model_type: str) -> bool:
        """
        Unload model from cache.
        
        Args:
            model_name (str): Name of the model
            model_type (str): Type of model
            
        Returns:
            bool: True if model was unloaded
        """
        cache_key = f"{model_name}_{model_type}"
        return self.cache.remove(cache_key)
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get model cache information."""
        return self.cache.get_cache_info()
    
    def clear_cache(self) -> None:
        """Clear all cached models."""
        self.cache.clear()
    
    # COMPLETE FIX: Replace the _register_default_loaders method in your model_utils.py

    def _register_default_loaders(self) -> None:
        """Register default model loaders."""
        
        def load_pytorch_model(model_name: str, model_config: Dict[str, Any], **kwargs) -> Any:
            """Load PyTorch model with CRAFT auto-download."""
            if not TORCH_AVAILABLE:
                raise ImportError("PyTorch not available")
            
            model_path = self.models_dir / f"{model_name}.pth"
            
            # Special handling for CRAFT model
            if model_name == "craft_mlt_25k" and not model_path.exists():
                self.logger.info("CRAFT model not found locally. Downloading...")
                try:
                    self._download_craft_model(model_path)
                except Exception as e:
                    self.logger.error(f"Failed to download CRAFT model: {e}")
                    raise FileNotFoundError(f"CRAFT model download failed: {e}")
            
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            device = kwargs.get('device', 'cpu')
            return torch.load(model_path, map_location=device)
        
        def load_tensorflow_model(model_name: str, model_config: Dict[str, Any], **kwargs) -> Any:
            """Load TensorFlow model."""
            if not TF_AVAILABLE:
                raise ImportError("TensorFlow not available")
            
            model_path = self.models_dir / model_name
            if not model_path.exists():
                raise FileNotFoundError(f"Model directory not found: {model_path}")
            
            return tf.keras.models.load_model(str(model_path))
        
        def load_transformers_model(model_name: str, model_config: Dict[str, Any], **kwargs) -> Any:
            """Load Transformers model with proper TrOCR support."""
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError("Transformers library not available")
            
            if "trocr" in model_name.lower():
                model_class = "VisionEncoderDecoderModel"
            else:
                model_class = model_config.get("model_class", "AutoModel")
            
            # FIX: Map generic model names to actual HuggingFace model identifiers
            model_name_mapping = {
                "trocr": "microsoft/trocr-base-printed",  # Default TrOCR model
                "trocr_handwritten": "microsoft/trocr-base-handwritten", 
                "trocr_printed": "microsoft/trocr-base-printed",
                "trocr_large": "microsoft/trocr-large-printed",
            }
            
            # Use mapped name if available
            actual_model_name = model_name_mapping.get(model_name, model_name)
            
            model_class_obj = getattr(transformers, model_class)
            
            # Check if local model exists first
            local_path = self.models_dir / model_name
            if local_path.exists():
                return model_class_obj.from_pretrained(str(local_path))
            else:
                # Download from HuggingFace Hub using the correct identifier
                self.logger.info(f"Loading model from HuggingFace: {actual_model_name}")
                return model_class_obj.from_pretrained(actual_model_name)
        
        def load_pickle_model(model_name: str, model_config: Dict[str, Any], **kwargs) -> Any:
            """Load pickled model."""
            model_path = self.models_dir / f"{model_name}.pkl"
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        
        def load_sklearn_model(model_name: str, model_config: Dict[str, Any], **kwargs) -> Any:
            """Load scikit-learn model."""
            if not SKLEARN_AVAILABLE:
                raise ImportError("Scikit-learn not available")
            
            model_path = self.models_dir / f"{model_name}.pkl"
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        
        # Register default loaders
        self._loaders.update({
            'pytorch': load_pytorch_model,
            'torch': load_pytorch_model,  # Alias
            'tensorflow': load_tensorflow_model,
            'tf': load_tensorflow_model,  # Alias
            'transformers': load_transformers_model,
            'pickle': load_pickle_model,
            'sklearn': load_sklearn_model,
        })
        
# Decorator for automatic model caching
def cached_model_load(model_name: str, model_type: str):
    """
    Decorator for automatic model loading and caching.
    
    Args:
        model_name (str): Name of the model
        model_type (str): Type of model
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Get model loader from config (assume it's passed as kwarg or available globally)
            model_loader = kwargs.get('model_loader')
            if not model_loader:
                # Try to create a default loader
                config = kwargs.get('config') or OCRConfig()
                model_loader = ModelLoader(config)
            
            # Load model
            model = model_loader.load_model(model_name, model_type)
            
            # Add model to function kwargs
            kwargs['model'] = model
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


# Utility functions for external use
def create_model_loader(config: Optional[OCRConfig] = None, 
                       models_dir: str = "models") -> ModelLoader:
    """
    Create a model loader instance.
    
    Args:
        config (Optional[OCRConfig]): OCR configuration
        models_dir (str): Models directory
        
    Returns:
        ModelLoader: Configured model loader
    """
    if config is None:
        config = OCRConfig()
    
    return ModelLoader(config, models_dir)


def get_available_frameworks() -> Dict[str, bool]:
    """
    Check which ML frameworks are available.
    
    Returns:
        Dict[str, bool]: Framework availability status
    """
    return {
        'pytorch': TORCH_AVAILABLE,
        'tensorflow': TF_AVAILABLE,
        'transformers': TRANSFORMERS_AVAILABLE,
        'sklearn': SKLEARN_AVAILABLE,
    }


def estimate_model_memory_usage(model: Any) -> float:
    """
    Estimate model memory usage in MB.
    
    Args:
        model (Any): Model object
        
    Returns:
        float: Estimated memory usage in MB
    """
    cache = ModelCache()
    return cache._estimate_model_size(model)


__all__ = [
    'ModelCache', 'ModelDownloader', 'ModelVersionManager', 'ModelLoader',
    'cached_model_load', 'create_model_loader', 'get_available_frameworks',
    'estimate_model_memory_usage'
]