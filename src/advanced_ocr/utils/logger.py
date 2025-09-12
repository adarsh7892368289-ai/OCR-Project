"""
Advanced OCR Logging System.

This module provides comprehensive logging capabilities for OCR operations,
including performance monitoring, structured logging, and error tracking.
It supports multiple output formats and integrates seamlessly with OCR engines.

Features:
    - Structured JSON logging for machine processing
    - Performance metrics collection and analysis
    - OCR-specific context logging (engine, confidence, etc.)
    - Thread-safe operations
    - Rotating file handlers with size limits
    - Console and file output support
"""

import logging
import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any, Union
from datetime import datetime
import json
import traceback
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import threading
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import time

@dataclass
class PerformanceMetric:
    """
    Data structure for storing performance metrics.

    This dataclass holds information about the execution of an operation,
    including timing, memory usage, success status, and additional metadata.

    Attributes:
        operation_name (str): Name of the operation being measured.
        start_time (float): Timestamp when the operation started.
        end_time (float): Timestamp when the operation ended.
        duration (float): Total duration of the operation in seconds.
        memory_usage_mb (Optional[float]): Memory usage in MB, if available.
        success (bool): Whether the operation completed successfully.
        error_message (Optional[str]): Error message if the operation failed.
        metadata (Dict[str, Any]): Additional metadata associated with the operation.
    """
    operation_name: str
    start_time: float
    end_time: float
    duration: float
    memory_usage_mb: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the performance metric to a dictionary for JSON serialization.

        Returns:
            Dict[str, Any]: Dictionary representation of the metric.
        """
        return asdict(self)

class PerformanceMonitor:
    """
    Thread-safe performance monitoring and metrics collection.

    This class provides a thread-safe way to record, store, and retrieve
    performance metrics for various operations. It supports statistical
    analysis and can handle concurrent access from multiple threads.
    """

    def __init__(self):
        """
        Initialize the performance monitor.

        Creates a thread lock and initializes the metrics storage.
        """
        self._lock = threading.Lock()
        self._metrics: Dict[str, list] = {}

    def record_metric(self, metric: PerformanceMetric):
        """
        Record a performance metric.

        Args:
            metric (PerformanceMetric): The performance metric to record.
        """
        with self._lock:
            if metric.operation_name not in self._metrics:
                self._metrics[metric.operation_name] = []
            self._metrics[metric.operation_name].append(metric)

    def get_metrics(self, operation_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get performance metrics.

        Args:
            operation_name (Optional[str]): Specific operation name to filter by.
                If None, returns all metrics.

        Returns:
            Dict[str, Any]: Dictionary containing the requested metrics.
                If operation_name is specified, returns metrics for that operation.
                Otherwise, returns all metrics grouped by operation name.
        """
        with self._lock:
            if operation_name:
                return {
                    'operation': operation_name,
                    'metrics': [m.to_dict() for m in self._metrics.get(operation_name, [])]
                }
            else:
                return {
                    op_name: [m.to_dict() for m in metrics]
                    for op_name, metrics in self._metrics.items()
                }

    def get_statistics(self, operation_name: str) -> Dict[str, float]:
        """
        Get statistical summary for an operation.

        Args:
            operation_name (str): Name of the operation to get statistics for.

        Returns:
            Dict[str, float]: Dictionary containing statistical metrics including
                count, success_count, success_rate, avg_duration, min_duration,
                max_duration, and total_duration. Returns empty dict if no metrics exist.
        """
        with self._lock:
            metrics = self._metrics.get(operation_name, [])
            if not metrics:
                return {}

            durations = [m.duration for m in metrics if m.success]
            if not durations:
                return {'success_rate': 0.0}

            return {
                'count': len(metrics),
                'success_count': len(durations),
                'success_rate': len(durations) / len(metrics),
                'avg_duration': sum(durations) / len(durations),
                'min_duration': min(durations),
                'max_duration': max(durations),
                'total_duration': sum(durations)
            }

    def clear_metrics(self, operation_name: Optional[str] = None):
        """
        Clear performance metrics.

        Args:
            operation_name (Optional[str]): Specific operation name to clear.
                If None, clears all metrics.
        """
        with self._lock:
            if operation_name:
                self._metrics.pop(operation_name, None)
            else:
                self._metrics.clear()

class OCRFormatter(logging.Formatter):
    """
    Custom formatter for OCR logging with structured output.

    This formatter supports both human-readable and structured JSON output,
    including OCR-specific context and performance metrics.
    """

    def __init__(self, include_performance: bool = False):
        """
        Initialize the OCRFormatter.

        Args:
            include_performance (bool): Whether to include performance metrics in the output.
        """
        super().__init__()
        self.include_performance = include_performance

    def format(self, record: logging.LogRecord) -> str:
        """
        Format a log record with OCR-specific information.

        Args:
            record (logging.LogRecord): The log record to format.

        Returns:
            str: The formatted log message, either as JSON or human-readable string.
        """
        # Base timestamp and level
        timestamp = datetime.fromtimestamp(record.created).isoformat()

        # Build structured log entry
        log_entry = {
            'timestamp': timestamp,
            'level': record.levelname,
            'component': record.name,
            'message': record.getMessage(),
        }

        # Add thread information for debugging
        if hasattr(record, 'thread') and record.thread:
            log_entry['thread_id'] = record.thread

        # Add OCR-specific context if available
        ocr_context = {}
        if hasattr(record, 'engine_name'):
            ocr_context['engine'] = record.engine_name
        if hasattr(record, 'processing_time'):
            ocr_context['processing_time_ms'] = round(record.processing_time * 1000, 2)
        if hasattr(record, 'image_size'):
            ocr_context['image_size'] = record.image_size
        if hasattr(record, 'confidence'):
            ocr_context['confidence'] = record.confidence
        if hasattr(record, 'text_length'):
            ocr_context['text_length'] = record.text_length

        if ocr_context:
            log_entry['ocr_context'] = ocr_context

        # Add exception information if present
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__ if record.exc_info[0] else None,
                'message': str(record.exc_info[1]) if record.exc_info[1] else None,
                'traceback': traceback.format_exception(*record.exc_info)
            }

        # Performance metrics if enabled
        if self.include_performance and hasattr(record, 'performance_metric'):
            log_entry['performance'] = record.performance_metric.to_dict()

        # Format as JSON for structured logging or human-readable for console
        if getattr(record, 'structured', False):
            return json.dumps(log_entry, indent=2)
        else:
            # Human-readable format
            base_msg = f"{timestamp} | {record.levelname:8} | {record.name:20} | {record.getMessage()}"

            if ocr_context:
                context_str = " | ".join([f"{k}={v}" for k, v in ocr_context.items()])
                base_msg += f" | {context_str}"

            if record.exc_info:
                base_msg += f"\n{self.formatException(record.exc_info)}"

            return base_msg

class OCRLogger:
    """
    Advanced OCR Logger with performance monitoring and structured output
    """
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(name)
        self.performance_monitor = PerformanceMonitor()
        self._setup_complete = False
    
    def setup_logging(self, 
                     level: str = "INFO",
                     log_file: Optional[str] = None,
                     max_file_size_mb: int = 100,
                     backup_count: int = 5,
                     enable_console: bool = True,
                     enable_performance_logging: bool = True,
                     log_directory: Optional[str] = None):
        """
        Setup comprehensive logging configuration
        
        Args:
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_file: Log file name (optional)
            max_file_size_mb: Maximum log file size in MB
            backup_count: Number of backup log files to keep
            enable_console: Enable console output
            enable_performance_logging: Enable performance metric logging
            log_directory: Directory for log files
        """
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Set logging level
        numeric_level = getattr(logging, level.upper(), logging.INFO)
        self.logger.setLevel(numeric_level)
        
        # Console handler
        if enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_formatter = OCRFormatter(include_performance=False)
            console_handler.setFormatter(console_formatter)
            console_handler.setLevel(numeric_level)
            self.logger.addHandler(console_handler)
        
        # File handler
        if log_file:
            # Ensure log directory exists
            if log_directory:
                os.makedirs(log_directory, exist_ok=True)
                log_path = os.path.join(log_directory, log_file)
            else:
                log_path = log_file
            
            # Rotating file handler
            file_handler = RotatingFileHandler(
                log_path,
                maxBytes=max_file_size_mb * 1024 * 1024,
                backupCount=backup_count,
                encoding='utf-8'
            )
            
            file_formatter = OCRFormatter(include_performance=enable_performance_logging)
            file_handler.setFormatter(file_formatter)
            file_handler.setLevel(numeric_level)
            self.logger.addHandler(file_handler)
        
        # Performance log handler (separate file)
        if enable_performance_logging and log_directory:
            perf_log_path = os.path.join(log_directory, f"{self.name}_performance.log")
            perf_handler = TimedRotatingFileHandler(
                perf_log_path,
                when='midnight',
                interval=1,
                backupCount=7,
                encoding='utf-8'
            )
            
            perf_formatter = OCRFormatter(include_performance=True)
            perf_handler.setFormatter(perf_formatter)
            perf_handler.setLevel(logging.INFO)
            
            # Add filter to only log performance metrics
            perf_handler.addFilter(lambda record: hasattr(record, 'performance_metric'))
            self.logger.addHandler(perf_handler)
        
        self._setup_complete = True
        self.info("OCR Logging system initialized", extra={'component': 'logger_setup'})
    
    @contextmanager
    def performance_timer(self, operation_name: str, **metadata):
        """
        Context manager for timing operations with automatic logging
        
        Args:
            operation_name: Name of the operation being timed
            **metadata: Additional metadata to include
        """
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        try:
            yield
            
            # Success case
            end_time = time.time()
            end_memory = self._get_memory_usage()
            duration = end_time - start_time
            
            metric = PerformanceMetric(
                operation_name=operation_name,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                memory_usage_mb=end_memory - start_memory if start_memory and end_memory else None,
                success=True,
                metadata=metadata or {}
            )
            
            # Record metric and log
            self.performance_monitor.record_metric(metric)
            self._log_performance_metric(metric)
            
        except Exception as e:
            # Error case
            end_time = time.time()
            duration = end_time - start_time
            
            metric = PerformanceMetric(
                operation_name=operation_name,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                success=False,
                error_message=str(e),
                metadata=metadata or {}
            )
            
            # Record metric and log
            self.performance_monitor.record_metric(metric)
            self._log_performance_metric(metric)
            
            # Re-raise the exception
            raise
    
    def _get_memory_usage(self) -> Optional[float]:
        """Get current memory usage in MB"""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # Convert to MB
        except ImportError:
            return None
        except Exception:
            return None
    
    def _log_performance_metric(self, metric: PerformanceMetric):
        """Log performance metric"""
        if metric.success:
            self.info(
                f"Operation '{metric.operation_name}' completed in {metric.duration:.3f}s",
                extra={
                    'performance_metric': metric,
                    'processing_time': metric.duration,
                    'operation': metric.operation_name
                }
            )
        else:
            self.error(
                f"Operation '{metric.operation_name}' failed after {metric.duration:.3f}s: {metric.error_message}",
                extra={
                    'performance_metric': metric,
                    'processing_time': metric.duration,
                    'operation': metric.operation_name
                }
            )
    
    # Standard logging methods with OCR-specific enhancements
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self.logger.debug(message, extra=kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message"""
        self.logger.info(message, extra=kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self.logger.warning(message, extra=kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message"""
        self.logger.error(message, extra=kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message"""
        self.logger.critical(message, extra=kwargs)
    
    def exception(self, message: str, **kwargs):
        """Log exception with traceback"""
        self.logger.exception(message, extra=kwargs)
    
    # OCR-specific logging methods
    def log_engine_performance(self, engine_name: str, processing_time: float, 
                             text_length: int, confidence: float, 
                             image_size: tuple, success: bool = True):
        """Log OCR engine performance"""
        self.info(
            f"Engine {engine_name} processed image",
            extra={
                'engine_name': engine_name,
                'processing_time': processing_time,
                'text_length': text_length,
                'confidence': confidence,
                'image_size': f"{image_size[0]}x{image_size[1]}",
                'success': success
            }
        )
    
    def log_preprocessing_step(self, step_name: str, processing_time: float, 
                             input_size: tuple, output_size: tuple, 
                             enhancement_applied: bool = False):
        """Log preprocessing step"""
        self.debug(
            f"Preprocessing step '{step_name}' completed",
            extra={
                'step_name': step_name,
                'processing_time': processing_time,
                'input_size': f"{input_size[0]}x{input_size[1]}",
                'output_size': f"{output_size[0]}x{output_size[1]}",
                'enhancement_applied': enhancement_applied
            }
        )
    
    def log_postprocessing_step(self, step_name: str, processing_time: float,
                              input_text_length: int, output_text_length: int,
                              corrections_made: int = 0):
        """Log postprocessing step"""
        self.debug(
            f"Postprocessing step '{step_name}' completed",
            extra={
                'step_name': step_name,
                'processing_time': processing_time,
                'input_text_length': input_text_length,
                'output_text_length': output_text_length,
                'corrections_made': corrections_made
            }
        )
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        return self.performance_monitor.get_metrics()
    
    def reset_performance_statistics(self):
        """Reset performance statistics"""
        self.performance_monitor.clear_metrics()

# Global logger instance factory
_loggers: Dict[str, OCRLogger] = {}
_logger_lock = threading.Lock()

def get_logger(name: str) -> OCRLogger:
    """
    Get or create OCR logger instance (thread-safe singleton pattern)
    
    Args:
        name: Logger name (usually module name)
        
    Returns:
        OCRLogger instance
    """
    with _logger_lock:
        if name not in _loggers:
            _loggers[name] = OCRLogger(name)
        return _loggers[name]

def setup_logger(name: str, 
                level: str = "INFO",
                log_file: Optional[str] = None,
                log_directory: Optional[str] = None,
                enable_console: bool = True,
                enable_performance_logging: bool = True) -> OCRLogger:
    """
    Setup and configure OCR logger
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Log file name
        log_directory: Directory for log files  
        enable_console: Enable console logging
        enable_performance_logging: Enable performance logging
        
    Returns:
        Configured OCRLogger instance
    """
    logger = get_logger(name)
    
    logger.setup_logging(
        level=level,
        log_file=log_file,
        log_directory=log_directory,
        enable_console=enable_console,
        enable_performance_logging=enable_performance_logging
    )
    
    return logger

def configure_global_logging(level: str = "INFO", 
                           log_directory: str = "logs",
                           enable_performance_logging: bool = True):
    """
    Configure global logging for the entire OCR system
    
    Args:
        level: Global logging level
        log_directory: Directory for all log files
        enable_performance_logging: Enable performance monitoring
    """
    
    # Ensure log directory exists
    os.makedirs(log_directory, exist_ok=True)
    
    # Configure root OCR logger
    root_logger = setup_logger(
        "advanced_ocr",
        level=level,
        log_file="advanced_ocr.log",
        log_directory=log_directory,
        enable_performance_logging=enable_performance_logging
    )
    
    root_logger.info("Global OCR logging configured", 
                    extra={'log_directory': log_directory, 'level': level})

# Export main functionality
__all__ = [
    'OCRLogger',
    'PerformanceMetric', 
    'PerformanceMonitor',
    'get_logger',
    'setup_logger',
    'configure_global_logging'
]