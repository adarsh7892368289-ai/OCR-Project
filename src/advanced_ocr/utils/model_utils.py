"""
Advanced OCR System - Model Management Utilities
Smart ML model loading, caching, and memory management for OCR engines.
"""

import os
import sys
import time
import hashlib
import pickle
import threading
import psutil
from pathlib import Path
from typing import Dict, Any, Optional, Callable, Tuple, List
from collections import OrderedDict
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, Future
import logging
import weakref
import gc

from ..config import OCRConfig


@dataclass
class ModelInfo:
    """Information about a loaded model."""
    
    name: str
    version: str = "latest"
    size_mb: float = 0.0
    load_time: float = 0.0
    last_used: float = field(default_factory=time.time)
    use_count: int = 0
    memory_usage: float = 0.0
    model_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CacheStats:
    """Cache performance statistics."""
    
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_memory_mb: float = 0.0
    models_loaded: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


class ModelCache:
    """
    LRU cache for ML models with memory management.
    Thread-safe implementation with automatic cleanup.
    """
    
    def __init__(self, max_memory_mb: float = 2048, max_models: int = 10):
        """
        Initialize model cache.
        
        Args:
            max_memory_mb: Maximum memory usage in MB
            max_models: Maximum number of models to cache
        """
        self.max_memory_mb = max_memory_mb
        self.max_models = max_models
        self.logger = logging.getLogger(__name__)
        
        # Thread-safe cache storage
        self._cache: OrderedDict[str, Tuple[Any, ModelInfo]] = OrderedDict()
        self._lock = threading.RLock()
        self._stats = CacheStats()
        
        # Weak references to track model usage
        self._model_refs: Dict[str, weakref.ReferenceType] = {}
        
        # Background cleanup thread
        self._cleanup_thread = None
        self._stop_cleanup = threading.Event()
        self._start_cleanup_thread()
    
    def get(self, key: str) -> Optional[Tuple[Any, ModelInfo]]:
        """
        Get model from cache.
        
        Args:
            key: Model cache key
            
        Returns:
            Tuple of (model, model_info) or None if not found
        """
        with self._lock:
            if key in self._cache:
                # Move to end (most recently used)
                model, info = self._cache.pop(key)
                self._cache[key] = (model, info)
                
                # Update usage stats
                info.last_used = time.time()
                info.use_count += 1
                self._stats.hits += 1
                
                self.logger.debug(f"Cache hit for model: {key}")
                return model, info
            else:
                self._stats.misses += 1
                self.logger.debug(f"Cache miss for model: {key}")
                return None
    
    def put(self, key: str, model: Any, info: ModelInfo) -> bool:
        """
        Store model in cache.
        
        Args:
            key: Model cache key
            model: Model object to cache
            info: Model information
            
        Returns:
            True if stored successfully, False otherwise
        """
        with self._lock:
            try:
                # Check if we need to evict models
                self._ensure_space(info.memory_usage)
                
                # Store model
                self._cache[key] = (model, info)
                self._stats.total_memory_mb += info.memory_usage
                self._stats.models_loaded += 1
                
                # Create weak reference for tracking
                self._model_refs[key] = weakref.ref(model, self._model_cleanup_callback(key))
                
                self.logger.info(
                    f"Cached model {key}: {info.memory_usage:.1f}MB "
                    f"(Total: {self._stats.total_memory_mb:.1f}MB)"
                )
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to cache model {key}: {e}")
                return False
    
    def _ensure_space(self, required_mb: float) -> None:
        """Ensure sufficient cache space by evicting models if needed."""
        # Check memory limit
        while (self._stats.total_memory_mb + required_mb > self.max_memory_mb or 
               len(self._cache) >= self.max_models):
            
            if not self._cache:
                break
                
            # Evict least recently used model
            evicted_key, (evicted_model, evicted_info) = self._cache.popitem(last=False)
            self._stats.total_memory_mb -= evicted_info.memory_usage
            self._stats.evictions += 1
            
            # Clean up weak reference
            if evicted_key in self._model_refs:
                del self._model_refs[evicted_key]
            
            self.logger.info(f"Evicted model {evicted_key} ({evicted_info.memory_usage:.1f}MB)")
            
            # Force garbage collection
            del evicted_model
            gc.collect()
    
    def _model_cleanup_callback(self, key: str) -> Callable:
        """Create cleanup callback for weak references."""
        def cleanup(ref):
            with self._lock:
                if key in self._model_refs and self._model_refs[key] is ref:
                    del self._model_refs[key]
                    self.logger.debug(f"Cleaned up weak reference for model: {key}")
        return cleanup
    
    def clear(self) -> None:
        """Clear all cached models."""
        with self._lock:
            self._cache.clear()
            self._model_refs.clear()
            self._stats = CacheStats()
            gc.collect()
            self.logger.info("Cleared model cache")
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        with self._lock:
            return CacheStats(
                hits=self._stats.hits,
                misses=self._stats.misses,
                evictions=self._stats.evictions,
                total_memory_mb=self._stats.total_memory_mb,
                models_loaded=len(self._cache)
            )
    
    def _start_cleanup_thread(self) -> None:
        """Start background cleanup thread."""
        def cleanup_worker():
            while not self._stop_cleanup.wait(60):  # Check every minute
                try:
                    self._periodic_cleanup()
                except Exception as e:
                    self.logger.error(f"Cleanup thread error: {e}")
        
        self._cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        self._cleanup_thread.start()
    
    def _periodic_cleanup(self) -> None:
        """Periodic cleanup of unused models."""
        current_time = time.time()
        cleanup_threshold = 1800  # 30 minutes
        
        with self._lock:
            to_remove = []
            for key, (model, info) in self._cache.items():
                if current_time - info.last_used > cleanup_threshold:
                    to_remove.append(key)
            
            for key in to_remove:
                model, info = self._cache.pop(key)
                self._stats.total_memory_mb -= info.memory_usage
                if key in self._model_refs:
                    del self._model_refs[key]
                self.logger.info(f"Cleanup: removed unused model {key}")
                del model
        
        if to_remove:
            gc.collect()
    
    def __del__(self):
        """Cleanup on destruction."""
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._stop_cleanup.set()


class ModelDownloader:
    """
    Automatic model download and management.
    Handles downloading models with progress tracking and verification.
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize model downloader.
        
        Args:
            cache_dir: Directory to cache downloaded models
        """
        self.cache_dir = Path(cache_dir or self._get_default_cache_dir())
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Model registry with download information
        self.model_registry = {
            'paddleocr': {
                'det_model': {
                    'url': 'https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar',
                    'filename': 'en_PP-OCRv3_det_infer.tar',
                    'extract': True,
                    'size_mb': 2.3
                },
                'rec_model': {
                    'url': 'https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_rec_infer.tar',
                    'filename': 'en_PP-OCRv3_rec_infer.tar',
                    'extract': True,
                    'size_mb': 8.5
                }
            },
            'easyocr': {
                'detector': {
                    'url': 'https://github.com/JaidedAI/EasyOCR/releases/download/v1.3.2/craft_mlt_25k.pth',
                    'filename': 'craft_mlt_25k.pth',
                    'extract': False,
                    'size_mb': 87.8
                },
                'recognizer': {
                    'url': 'https://github.com/JaidedAI/EasyOCR/releases/download/v1.3.2/english_g2.pth',
                    'filename': 'english_g2.pth',
                    'extract': False,
                    'size_mb': 45.2
                }
            }
        }
    
    def _get_default_cache_dir(self) -> str:
        """Get default cache directory based on platform."""
        if sys.platform == "win32":
            cache_base = os.environ.get('LOCALAPPDATA', os.path.expanduser('~'))
            return os.path.join(cache_base, 'AdvancedOCR', 'models')
        else:
            cache_base = os.environ.get('XDG_CACHE_HOME', os.path.expanduser('~/.cache'))
            return os.path.join(cache_base, 'advanced-ocr', 'models')
    
    def download_model(self, engine: str, model_name: str, 
                      progress_callback: Optional[Callable[[float], None]] = None) -> Optional[str]:
        """
        Download model if not already cached.
        
        Args:
            engine: Engine name (e.g., 'paddleocr', 'easyocr')
            model_name: Model name within engine
            progress_callback: Optional progress callback function
            
        Returns:
            Path to downloaded/cached model or None if failed
        """
        if engine not in self.model_registry:
            self.logger.error(f"Unknown engine: {engine}")
            return None
        
        if model_name not in self.model_registry[engine]:
            self.logger.error(f"Unknown model {model_name} for engine {engine}")
            return None
        
        model_info = self.model_registry[engine][model_name]
        model_path = self.cache_dir / engine / model_info['filename']
        
        # Check if already downloaded
        if model_path.exists() and self._verify_model(model_path, model_info):
            self.logger.debug(f"Model already cached: {model_path}")
            return str(model_path)
        
        try:
            return self._download_file(model_info, model_path, progress_callback)
        except Exception as e:
            self.logger.error(f"Failed to download model {engine}/{model_name}: {e}")
            return None
    
    def _download_file(self, model_info: Dict[str, Any], target_path: Path,
                      progress_callback: Optional[Callable[[float], None]] = None) -> str:
        """Download file with progress tracking."""
        import requests
        from tqdm import tqdm
        
        target_path.parent.mkdir(parents=True, exist_ok=True)
        url = model_info['url']
        
        self.logger.info(f"Downloading model from {url}")
        
        with requests.get(url, stream=True) as response:
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            
            with open(target_path, 'wb') as f:
                if total_size == 0:
                    f.write(response.content)
                else:
                    downloaded = 0
                    chunk_size = 8192
                    
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            
                            if progress_callback:
                                progress = downloaded / total_size
                                progress_callback(progress)
        
        # Extract if needed
        if model_info.get('extract', False):
            return self._extract_model(target_path)
        
        return str(target_path)
    
    def _extract_model(self, archive_path: Path) -> str:
        """Extract model archive."""
        import tarfile
        import zipfile
        
        extract_dir = archive_path.parent / archive_path.stem
        extract_dir.mkdir(exist_ok=True)
        
        if archive_path.suffix == '.tar':
            with tarfile.open(archive_path, 'r') as tar:
                tar.extractall(extract_dir)
        elif archive_path.suffix in ['.zip', '.tar.gz']:
            if archive_path.suffix == '.zip':
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
            else:
                with tarfile.open(archive_path, 'r:gz') as tar:
                    tar.extractall(extract_dir)
        
        return str(extract_dir)
    
    def _verify_model(self, model_path: Path, model_info: Dict[str, Any]) -> bool:
        """Verify downloaded model integrity."""
        if not model_path.exists():
            return False
        
        # Check file size
        expected_size_mb = model_info.get('size_mb', 0)
        if expected_size_mb > 0:
            actual_size_mb = model_path.stat().st_size / (1024 * 1024)
            size_diff_pct = abs(actual_size_mb - expected_size_mb) / expected_size_mb
            
            if size_diff_pct > 0.1:  # 10% tolerance
                self.logger.warning(
                    f"Model size mismatch: expected {expected_size_mb:.1f}MB, "
                    f"got {actual_size_mb:.1f}MB"
                )
                return False
        
        return True


class ModelLoader:
    """
    Main model loading orchestrator with caching and error handling.
    Coordinates model downloading, caching, and loading operations.
    """
    
    def __init__(self, config: Optional[OCRConfig] = None):
        """
        Initialize model loader.
        
        Args:
            config: OCR configuration
        """
        self.config = config or OCRConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        cache_memory_mb = getattr(self.config, 'model_cache_memory_mb', 2048)
        max_models = getattr(self.config, 'max_cached_models', 10)
        
        self.cache = ModelCache(max_memory_mb=cache_memory_mb, max_models=max_models)
        self.downloader = ModelDownloader()
        
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix='ModelLoader')
        
        # Model loading functions registry
        self.loaders: Dict[str, Callable] = {}
        self._register_default_loaders()
    
    def _register_default_loaders(self) -> None:
        """Register default model loaders for common engines."""
        self.loaders.update({
            'paddleocr': self._load_paddleocr_models,
            'easyocr': self._load_easyocr_models,
            'trocr': self._load_trocr_models,
        })
    
    def load_model(self, engine: str, force_reload: bool = False) -> Optional[Any]:
        """
        Load model for specified engine.
        
        Args:
            engine: Engine name
            force_reload: Force reload even if cached
            
        Returns:
            Loaded model object or None if failed
        """
        cache_key = f"{engine}_model"
        
        # Check cache first (unless force reload)
        if not force_reload:
            cached = self.cache.get(cache_key)
            if cached:
                model, info = cached
                self.logger.debug(f"Using cached model for {engine}")
                return model
        
        # Load model
        try:
            start_time = time.time()
            self.logger.info(f"Loading model for engine: {engine}")
            
            if engine not in self.loaders:
                self.logger.error(f"No loader registered for engine: {engine}")
                return None
            
            # Load using registered loader
            model = self.loaders[engine]()
            if model is None:
                return None
            
            # Calculate memory usage
            memory_usage = self._estimate_model_memory(model)
            load_time = time.time() - start_time
            
            # Create model info
            model_info = ModelInfo(
                name=engine,
                size_mb=memory_usage,
                load_time=load_time,
                memory_usage=memory_usage,
                metadata={'engine': engine, 'loaded_at': time.time()}
            )
            
            # Cache the model
            self.cache.put(cache_key, model, model_info)
            
            self.logger.info(
                f"Loaded {engine} model: {memory_usage:.1f}MB in {load_time:.2f}s"
            )
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to load model for {engine}: {e}")
            return None
    
    def load_model_async(self, engine: str, force_reload: bool = False) -> Future:
        """
        Load model asynchronously.
        
        Args:
            engine: Engine name
            force_reload: Force reload even if cached
            
        Returns:
            Future object for the loading operation
        """
        return self.executor.submit(self.load_model, engine, force_reload)
    
    def _load_paddleocr_models(self) -> Optional[Any]:
        """Load PaddleOCR models."""
        try:
            # Import PaddleOCR
            from paddleocr import PaddleOCR
            
            # Download models if needed
            det_path = self.downloader.download_model('paddleocr', 'det_model')
            rec_path = self.downloader.download_model('paddleocr', 'rec_model')
            
            # Initialize PaddleOCR
            ocr = PaddleOCR(
                use_angle_cls=True,
                lang='en',
                use_gpu=self._has_gpu(),
                show_log=False
            )
            
            return ocr
            
        except ImportError:
            self.logger.error("PaddleOCR not installed. Install with: pip install paddleocr")
            return None
        except Exception as e:
            self.logger.error(f"Failed to load PaddleOCR: {e}")
            return None
    
    def _load_easyocr_models(self) -> Optional[Any]:
        """Load EasyOCR models."""
        try:
            # Import EasyOCR
            import easyocr
            
            # Download models if needed
            det_path = self.downloader.download_model('easyocr', 'detector')
            rec_path = self.downloader.download_model('easyocr', 'recognizer')
            
            # Initialize EasyOCR
            reader = easyocr.Reader(
                ['en'],
                gpu=self._has_gpu(),
                verbose=False
            )
            
            return reader
            
        except ImportError:
            self.logger.error("EasyOCR not installed. Install with: pip install easyocr")
            return None
        except Exception as e:
            self.logger.error(f"Failed to load EasyOCR: {e}")
            return None
    
    def _load_trocr_models(self) -> Optional[Any]:
        """Load TrOCR models."""
        try:
            # Import transformers
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            
            # Load processor and model
            processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
            model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
            
            return {'processor': processor, 'model': model}
            
        except ImportError:
            self.logger.error("Transformers not installed. Install with: pip install transformers")
            return None
        except Exception as e:
            self.logger.error(f"Failed to load TrOCR: {e}")
            return None
    
    def _has_gpu(self) -> bool:
        """Check if GPU is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            try:
                import tensorflow as tf
                return len(tf.config.list_physical_devices('GPU')) > 0
            except ImportError:
                return False
    
    def _estimate_model_memory(self, model: Any) -> float:
        """Estimate model memory usage in MB."""
        try:
            # Try to get actual memory usage
            process = psutil.Process()
            memory_info = process.memory_info()
            return memory_info.rss / (1024 * 1024)  # Convert to MB
        except:
            # Fallback to rough estimation
            return 100.0  # Default estimate
    
    def get_cache_stats(self) -> CacheStats:
        """Get model cache statistics."""
        return self.cache.get_stats()
    
    def clear_cache(self) -> None:
        """Clear model cache."""
        self.cache.clear()
    
    def __del__(self):
        """Cleanup on destruction."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=False)


# Convenience functions for common operations
def get_model_loader(config: Optional[OCRConfig] = None) -> ModelLoader:
    """Get singleton model loader instance."""
    if not hasattr(get_model_loader, '_instance'):
        get_model_loader._instance = ModelLoader(config)
    return get_model_loader._instance


def load_engine_model(engine: str, config: Optional[OCRConfig] = None) -> Optional[Any]:
    """
    Convenience function to load model for specific engine.
    
    Args:
        engine: Engine name
        config: Optional OCR configuration
        
    Returns:
        Loaded model or None if failed
    """
    loader = get_model_loader(config)
    return loader.load_model(engine)


def clear_model_cache() -> None:
    """Clear global model cache."""
    if hasattr(get_model_loader, '_instance'):
        get_model_loader._instance.clear_cache()