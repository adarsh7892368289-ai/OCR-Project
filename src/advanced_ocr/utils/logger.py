"""
Advanced logging system with structured logging, performance monitoring, and error context.
Provides comprehensive logging and metrics collection for the OCR pipeline.
"""

import logging
import json
import time
import threading
import psutil
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import traceback
from datetime import datetime


class JSONFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging."""
    
    def format(self, record):
        """Format log record as JSON."""
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception information if present
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__,
                'message': str(record.exc_info[1]),
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # Add custom fields from extra
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'lineno', 'funcName', 'created', 'msecs', 
                          'relativeCreated', 'thread', 'threadName', 'processName', 
                          'process', 'message', 'exc_info', 'exc_text', 'stack_info']:
                log_entry[key] = value
        
        return json.dumps(log_entry, ensure_ascii=False, default=str)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used_mb: float = 0.0
    gpu_memory_mb: float = 0.0
    disk_io_read_mb: float = 0.0
    disk_io_write_mb: float = 0.0
    network_sent_mb: float = 0.0
    network_recv_mb: float = 0.0
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def capture_current(cls) -> 'PerformanceMetrics':
        """Capture current system performance metrics."""
        try:
            process = psutil.Process()
            
            # CPU and memory
            cpu_percent = process.cpu_percent()
            memory_info = process.memory_info()
            memory_percent = process.memory_percent()
            
            # Disk I/O
            io_counters = process.io_counters() if hasattr(process, 'io_counters') else None
            disk_read = io_counters.read_bytes / (1024 * 1024) if io_counters else 0.0
            disk_write = io_counters.write_bytes / (1024 * 1024) if io_counters else 0.0
            
            # Network I/O (system-wide)
            net_io = psutil.net_io_counters()
            net_sent = net_io.bytes_sent / (1024 * 1024) if net_io else 0.0
            net_recv = net_io.bytes_recv / (1024 * 1024) if net_io else 0.0
            
            # GPU memory (if available)
            gpu_memory = 0.0
            try:
                import nvidia_ml_py3 as nvml
                nvml.nvmlInit()
                handle = nvml.nvmlDeviceGetHandleByIndex(0)
                info = nvml.nvmlDeviceGetMemoryInfo(handle)
                gpu_memory = info.used / (1024 * 1024)
            except (ImportError, Exception):
                pass
            
            return cls(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_used_mb=memory_info.rss / (1024 * 1024),
                gpu_memory_mb=gpu_memory,
                disk_io_read_mb=disk_read,
                disk_io_write_mb=disk_write,
                network_sent_mb=net_sent,
                network_recv_mb=net_recv
            )
            
        except Exception as e:
            # Return empty metrics if capture fails
            return cls()


@dataclass
class ProcessingStageMetrics:
    """Metrics for a specific processing stage."""
    
    stage_name: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    input_size: Optional[int] = None
    output_size: Optional[int] = None
    error_count: int = 0
    warning_count: int = 0
    custom_metrics: Dict[str, Any] = field(default_factory=dict)
    performance_start: Optional[PerformanceMetrics] = None
    performance_end: Optional[PerformanceMetrics] = None
    
    def finish(self):
        """Mark stage as finished and calculate duration."""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.performance_end = PerformanceMetrics.capture_current()
    
    def add_custom_metric(self, key: str, value: Any):
        """Add custom metric to stage."""
        self.custom_metrics[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        if self.performance_start:
            result['performance_start'] = self.performance_start.to_dict()
        if self.performance_end:
            result['performance_end'] = self.performance_end.to_dict()
        return result


class MetricsCollector:
    """Real-time performance tracking and memory usage monitoring."""
    
    def __init__(self):
        self.metrics_history: List[PerformanceMetrics] = []
        self.stage_metrics: Dict[str, ProcessingStageMetrics] = {}
        self.lock = threading.Lock()
        self._monitoring = False
        self._monitor_thread = None
        self._monitor_interval = 1.0  # seconds
    
    def start_monitoring(self, interval: float = 1.0):
        """Start background performance monitoring."""
        with self.lock:
            if self._monitoring:
                return
            
            self._monitoring = True
            self._monitor_interval = interval
            self._monitor_thread = threading.Thread(
                target=self._monitor_performance,
                daemon=True
            )
            self._monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop background performance monitoring."""
        with self.lock:
            self._monitoring = False
            if self._monitor_thread:
                self._monitor_thread.join(timeout=2.0)
                self._monitor_thread = None
    
    def _monitor_performance(self):
        """Background thread for performance monitoring."""
        while self._monitoring:
            try:
                metrics = PerformanceMetrics.capture_current()
                with self.lock:
                    self.metrics_history.append(metrics)
                    # Keep only last 1000 entries
                    if len(self.metrics_history) > 1000:
                        self.metrics_history = self.metrics_history[-1000:]
                
                time.sleep(self._monitor_interval)
            except Exception:
                # Continue monitoring even if metrics capture fails
                time.sleep(self._monitor_interval)
    
    def start_stage(self, stage_name: str, input_size: Optional[int] = None) -> ProcessingStageMetrics:
        """Start tracking a processing stage."""
        metrics = ProcessingStageMetrics(
            stage_name=stage_name,
            start_time=time.time(),
            input_size=input_size,
            performance_start=PerformanceMetrics.capture_current()
        )
        
        with self.lock:
            self.stage_metrics[stage_name] = metrics
        
        return metrics
    
    def finish_stage(self, stage_name: str, output_size: Optional[int] = None):
        """Finish tracking a processing stage."""
        with self.lock:
            if stage_name in self.stage_metrics:
                stage = self.stage_metrics[stage_name]
                stage.output_size = output_size
                stage.finish()
    
    def add_stage_metric(self, stage_name: str, key: str, value: Any):
        """Add custom metric to a stage."""
        with self.lock:
            if stage_name in self.stage_metrics:
                self.stage_metrics[stage_name].add_custom_metric(key, value)
    
    def get_current_metrics(self) -> Optional[PerformanceMetrics]:
        """Get most recent performance metrics."""
        with self.lock:
            return self.metrics_history[-1] if self.metrics_history else None
    
    def get_average_metrics(self, last_n: int = 60) -> Optional[PerformanceMetrics]:
        """Get average metrics over last N measurements."""
        with self.lock:
            if not self.metrics_history:
                return None
            
            recent = self.metrics_history[-last_n:] if len(self.metrics_history) >= last_n else self.metrics_history
            
            if not recent:
                return None
            
            return PerformanceMetrics(
                cpu_percent=sum(m.cpu_percent for m in recent) / len(recent),
                memory_percent=sum(m.memory_percent for m in recent) / len(recent),
                memory_used_mb=sum(m.memory_used_mb for m in recent) / len(recent),
                gpu_memory_mb=sum(m.gpu_memory_mb for m in recent) / len(recent),
                disk_io_read_mb=sum(m.disk_io_read_mb for m in recent) / len(recent),
                disk_io_write_mb=sum(m.disk_io_write_mb for m in recent) / len(recent),
                network_sent_mb=sum(m.network_sent_mb for m in recent) / len(recent),
                network_recv_mb=sum(m.network_recv_mb for m in recent) / len(recent),
                timestamp=recent[-1].timestamp
            )
    
    def get_stage_summary(self) -> Dict[str, Any]:
        """Get summary of all stage metrics."""
        with self.lock:
            return {
                stage_name: metrics.to_dict() 
                for stage_name, metrics in self.stage_metrics.items()
            }
    
    def clear_history(self):
        """Clear metrics history."""
        with self.lock:
            self.metrics_history.clear()
            self.stage_metrics.clear()


class ProcessingStageTimer:
    """Context manager for automatic timing measurement."""
    
    def __init__(self, stage_name: str, logger: 'OCRLogger', 
                 input_size: Optional[int] = None):
        self.stage_name = stage_name
        self.logger = logger
        self.input_size = input_size
        self.stage_metrics = None
    
    def __enter__(self):
        """Start timing the stage."""
        self.stage_metrics = self.logger.metrics_collector.start_stage(
            self.stage_name, self.input_size
        )
        self.logger.info(f"Started processing stage: {self.stage_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Finish timing the stage."""
        if exc_type is not None:
            # Log error and increment error count
            self.logger.error(
                f"Error in processing stage: {self.stage_name}",
                exc_info=(exc_type, exc_val, exc_tb),
                stage=self.stage_name
            )
            if self.stage_metrics:
                self.stage_metrics.error_count += 1
        
        self.logger.metrics_collector.finish_stage(self.stage_name)
        
        # Log completion
        if self.stage_metrics and self.stage_metrics.duration:
            self.logger.info(
                f"Completed processing stage: {self.stage_name} "
                f"in {self.stage_metrics.duration:.3f}s",
                stage=self.stage_name,
                duration=self.stage_metrics.duration
            )
    
    def add_metric(self, key: str, value: Any):
        """Add custom metric to this stage."""
        self.logger.metrics_collector.add_stage_metric(self.stage_name, key, value)


class OCRLogger:
    """Main logger class with structured logging and performance monitoring."""
    
    def __init__(self, name: str = "advanced_ocr", level: str = "INFO", 
                 log_file: Optional[Union[str, Path]] = None,
                 enable_json: bool = True, enable_metrics: bool = True):
        """
        Initialize OCR logger.
        
        Args:
            name: Logger name
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Optional log file path
            enable_json: Use JSON formatting
            enable_metrics: Enable performance metrics collection
        """
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Create formatters
        if enable_json:
            formatter = JSONFormatter()
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler if specified
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_path, encoding='utf-8')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        # Initialize metrics collector
        self.metrics_collector = MetricsCollector() if enable_metrics else None
        if self.metrics_collector and enable_metrics:
            self.metrics_collector.start_monitoring()
    
    def debug(self, message: str, **kwargs):
        """Log debug message with optional context."""
        self.logger.debug(message, extra=kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message with optional context."""
        self.logger.info(message, extra=kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with optional context."""
        self.logger.warning(message, extra=kwargs)
    
    def error(self, message: str, exc_info: Optional[Any] = None, **kwargs):
        """Log error message with optional exception info and context."""
        self.logger.error(message, exc_info=exc_info, extra=kwargs)
    
    def critical(self, message: str, exc_info: Optional[Any] = None, **kwargs):
        """Log critical message with optional exception info and context."""
        self.logger.critical(message, exc_info=exc_info, extra=kwargs)
    
    def log_processing_stage(self, stage_name: str, input_size: Optional[int] = None) -> ProcessingStageTimer:
        """Create context manager for timing a processing stage."""
        return ProcessingStageTimer(stage_name, self, input_size)
    
    def log_engine_performance(self, engine_name: str, processing_time: float, 
                              confidence: float, word_count: int):
        """Log OCR engine performance metrics."""
        self.info(
            f"OCR engine performance: {engine_name}",
            engine=engine_name,
            processing_time=processing_time,
            confidence=confidence,
            word_count=word_count,
            words_per_second=word_count / processing_time if processing_time > 0 else 0
        )
    
    def log_image_processing(self, operation: str, input_size: tuple, 
                           output_size: tuple, processing_time: float):
        """Log image processing operation."""
        self.info(
            f"Image processing: {operation}",
            operation=operation,
            input_size=input_size,
            output_size=output_size,
            processing_time=processing_time,
            size_change_ratio=output_size[0] * output_size[1] / (input_size[0] * input_size[1])
        )
    
    def log_quality_metrics(self, stage: str, quality_scores: Dict[str, float]):
        """Log quality assessment metrics."""
        self.info(
            f"Quality assessment: {stage}",
            stage=stage,
            **quality_scores
        )
    
    def log_memory_usage(self, stage: str, operation: str = ""):
        """Log current memory usage."""
        if self.metrics_collector:
            metrics = self.metrics_collector.get_current_metrics()
            if metrics:
                self.info(
                    f"Memory usage - {stage}: {operation}",
                    stage=stage,
                    operation=operation,
                    memory_mb=metrics.memory_used_mb,
                    memory_percent=metrics.memory_percent,
                    gpu_memory_mb=metrics.gpu_memory_mb
                )
    
    def log_batch_progress(self, current: int, total: int, batch_id: str = ""):
        """Log batch processing progress."""
        progress = (current / total) * 100 if total > 0 else 0
        self.info(
            f"Batch processing progress: {current}/{total} ({progress:.1f}%)",
            batch_id=batch_id,
            current=current,
            total=total,
            progress_percent=progress
        )
    
    def log_error_with_context(self, error: Exception, stage: str, 
                              context: Dict[str, Any] = None):
        """Log error with full processing context."""
        context = context or {}
        
        # Add system context
        current_metrics = None
        if self.metrics_collector:
            current_metrics = self.metrics_collector.get_current_metrics()
        
        error_context = {
            'stage': stage,
            'error_type': type(error).__name__,
            'error_message': str(error),
            **context
        }
        
        if current_metrics:
            error_context.update({
                'memory_mb': current_metrics.memory_used_mb,
                'cpu_percent': current_metrics.cpu_percent,
                'gpu_memory_mb': current_metrics.gpu_memory_mb
            })
        
        self.error(
            f"Processing error in {stage}: {str(error)}",
            exc_info=True,
            **error_context
        )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        if not self.metrics_collector:
            return {}
        
        current = self.metrics_collector.get_current_metrics()
        average = self.metrics_collector.get_average_metrics()
        stages = self.metrics_collector.get_stage_summary()
        
        return {
            'current_performance': current.to_dict() if current else None,
            'average_performance': average.to_dict() if average else None,
            'stage_metrics': stages,
            'total_stages': len(stages),
            'monitoring_active': self.metrics_collector._monitoring
        }
    
    def export_metrics(self, file_path: Union[str, Path]):
        """Export collected metrics to JSON file."""
        if not self.metrics_collector:
            self.warning("No metrics collector available for export")
            return
        
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        summary = self.get_performance_summary()
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        
        self.info(f"Metrics exported to {file_path}")
    
    def close(self):
        """Clean up logger resources."""
        if self.metrics_collector:
            self.metrics_collector.stop_monitoring()
        
        # Close all handlers
        for handler in self.logger.handlers:
            handler.close()
        
        self.logger.handlers.clear()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Global logger instance for convenience
_global_logger: Optional[OCRLogger] = None


def get_logger(name: str = "advanced_ocr", level: str = "INFO", 
               log_file: Optional[Union[str, Path]] = None,
               enable_json: bool = True, enable_metrics: bool = True) -> OCRLogger:
    """Get or create global logger instance."""
    global _global_logger
    
    if _global_logger is None:
        _global_logger = OCRLogger(
            name=name,
            level=level,
            log_file=log_file,
            enable_json=enable_json,
            enable_metrics=enable_metrics
        )
    
    return _global_logger


def setup_logging(config_dict: Dict[str, Any]) -> OCRLogger:
    """Set up logging from configuration dictionary."""
    return get_logger(
        level=config_dict.get('log_level', 'INFO'),
        log_file=config_dict.get('log_file'),
        enable_json=config_dict.get('enable_json_logging', True),
        enable_metrics=config_dict.get('enable_metrics', True)
    )


@contextmanager
def log_processing_stage(stage_name: str, logger: Optional[OCRLogger] = None):
    """Convenient context manager for logging processing stages."""
    if logger is None:
        logger = get_logger()
    
    with logger.log_processing_stage(stage_name) as timer:
        yield timer