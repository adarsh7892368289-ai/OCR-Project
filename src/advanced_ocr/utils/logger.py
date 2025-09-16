# src/advanced_ocr/utils/logger.py
"""
Advanced OCR Logging and Metrics System

This module provides comprehensive logging infrastructure for the advanced OCR
system with focus on performance tracking, bottleneck identification, and
real-time monitoring of critical optimization metrics.

The logging system tracks:
- Processing stage performance (preprocessing, OCR, postprocessing)
- Critical performance bottlenecks (text detection regions, TrOCR efficiency)
- Memory usage and resource consumption
- Engine performance comparison and optimization
- Real-time processing metrics for the <3 second requirement

Classes:
    OCRLogger: Main performance tracking and logging interface
    ProcessingStageTimer: Context manager for timing processing stages
    MetricsCollector: Real-time metrics collection and aggregation
    OCRDebugLogger: Development debugging with optional image snapshots
    LogConfig: Configuration for logging behavior and output formats

Example:
    >>> logger = OCRLogger("ocr_system")
    >>> with logger.stage_timer("preprocessing") as timer:
    ...     # preprocessing code here
    ...     timer.log_metric("regions_detected", 45)
    >>>
    >>> logger.log_performance_summary()
    >>> logger.check_performance_targets()
"""

import logging
import time
import json
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Union, IO
from pathlib import Path
from enum import Enum
import psutil
import os


class LogLevel(Enum):
    """Logging levels for different types of messages."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogFormat(Enum):
    """Supported log output formats."""
    JSON = "json"
    TEXT = "text"
    STRUCTURED = "structured"


@dataclass
class LogConfig:
    """
    Configuration for logging system behavior.
    
    Attributes:
        log_level (LogLevel): Minimum logging level
        log_format (LogFormat): Output format for log messages
        log_to_file (bool): Enable file logging
        log_to_console (bool): Enable console logging  
        log_file_path (Optional[str]): Path for log file output
        max_log_size_mb (int): Maximum log file size before rotation
        enable_performance_logging (bool): Enable detailed performance tracking
        enable_debug_snapshots (bool): Save debug images during development
        metrics_collection_interval (float): Metrics collection frequency in seconds
    """
    log_level: LogLevel = LogLevel.INFO
    log_format: LogFormat = LogFormat.STRUCTURED
    log_to_file: bool = True
    log_to_console: bool = True
    log_file_path: Optional[str] = None
    max_log_size_mb: int = 100
    enable_performance_logging: bool = True
    enable_debug_snapshots: bool = False
    metrics_collection_interval: float = 0.1


@dataclass
class ProcessingStageMetrics:
    """
    Metrics for individual processing stages.
    
    Tracks performance data for each stage of the OCR pipeline to identify
    bottlenecks and optimization opportunities.
    
    Attributes:
        stage_name (str): Name of the processing stage
        start_time (float): Stage start timestamp
        end_time (float): Stage end timestamp
        duration (float): Processing duration in seconds
        memory_start (float): Memory usage at stage start (MB)
        memory_peak (float): Peak memory usage during stage (MB)
        memory_end (float): Memory usage at stage end (MB)
        cpu_usage (float): Average CPU usage during stage
        custom_metrics (Dict[str, Any]): Stage-specific metrics
    """
    stage_name: str
    start_time: float
    end_time: float = 0.0
    duration: float = 0.0
    memory_start: float = 0.0
    memory_peak: float = 0.0
    memory_end: float = 0.0
    cpu_usage: float = 0.0
    custom_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize memory tracking."""
        if self.memory_start == 0.0:
            self.memory_start = self._get_memory_usage()
    
    def finalize(self):
        """Finalize stage metrics calculation."""
        if self.end_time == 0.0:
            self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.memory_end = self._get_memory_usage()
    
    @staticmethod
    def _get_memory_usage() -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0.0


@dataclass
class CriticalPerformanceMetrics:
    """
    Critical performance metrics for optimization requirements.
    
    Tracks the specific performance issues identified in the project:
    - Text detection region count (target: 20-80 regions)
    - TrOCR character extraction rate (target: >1000 chars)  
    - Total processing time (target: <3 seconds)
    
    Attributes:
        total_processing_time (float): End-to-end processing time
        text_regions_before_filtering (int): Initial region count from detection
        text_regions_after_filtering (int): Final region count after optimization
        characters_extracted (int): Total characters extracted by OCR engines
        character_extraction_rate (float): Characters per second extraction rate
        memory_peak_usage (float): Peak memory usage during processing
        performance_targets_met (bool): Whether all targets were achieved
        bottlenecks_identified (List[str]): List of identified performance bottlenecks
    """
    total_processing_time: float = 0.0
    text_regions_before_filtering: int = 0
    text_regions_after_filtering: int = 0
    characters_extracted: int = 0
    character_extraction_rate: float = 0.0
    memory_peak_usage: float = 0.0
    performance_targets_met: bool = False
    bottlenecks_identified: List[str] = field(default_factory=list)
    
    def calculate_efficiency_metrics(self):
        """Calculate derived efficiency metrics."""
        # Text detection efficiency
        if self.text_regions_before_filtering > 0:
            detection_efficiency = self.text_regions_after_filtering / self.text_regions_before_filtering
            if detection_efficiency < 0.1:
                self.bottlenecks_identified.append(f"poor_detection_efficiency_{detection_efficiency:.3f}")
        
        # Character extraction rate
        if self.total_processing_time > 0:
            self.character_extraction_rate = self.characters_extracted / self.total_processing_time
            if self.character_extraction_rate < 300:
                self.bottlenecks_identified.append(f"slow_char_extraction_{self.character_extraction_rate:.1f}")
        
        # Performance targets check
        self.performance_targets_met = (
            self.total_processing_time < 3.0 and
            self.text_regions_after_filtering <= 100 and
            self.character_extraction_rate >= 300
        )
        
        # Identify bottlenecks
        if self.total_processing_time >= 3.0:
            self.bottlenecks_identified.append(f"total_time_slow_{self.total_processing_time:.2f}s")
        
        if self.text_regions_after_filtering > 100:
            self.bottlenecks_identified.append(f"too_many_regions_{self.text_regions_after_filtering}")


class ProcessingStageTimer:
    """
    Context manager for timing processing stages with detailed metrics collection.
    
    Provides automatic timing, memory tracking, and custom metrics logging
    for individual processing stages in the OCR pipeline.
    
    Example:
        >>> with logger.stage_timer("text_detection") as timer:
        ...     regions = detect_text_regions(image)
        ...     timer.log_metric("regions_detected", len(regions))
        ...     timer.log_metric("confidence_threshold", 0.7)
    """
    
    def __init__(self, stage_name: str, logger: 'OCRLogger'):
        """
        Initialize stage timer.

        Args:
            stage_name: Name of the processing stage
            logger: Parent OCR logger instance
        """
        self.stage_name = stage_name
        self.logger = logger
        self.metrics = ProcessingStageMetrics(
            stage_name=stage_name,
            start_time=time.time()
        )
        self._memory_monitor_active = False
        self._memory_monitor_thread: Optional[threading.Thread] = None
    
    def __enter__(self) -> 'ProcessingStageTimer':
        """Enter context manager and start monitoring."""
        self._start_memory_monitoring()
        self.logger._log_stage_start(self.stage_name)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and finalize metrics."""
        self._stop_memory_monitoring()
        self.metrics.finalize()
        self.logger._log_stage_complete(self.stage_name, self.metrics)
    
    def log_metric(self, name: str, value: Any):
        """
        Log custom metric for this processing stage.
        
        Args:
            name: Metric name
            value: Metric value
        """
        self.metrics.custom_metrics[name] = value
        self.logger._log_custom_metric(self.stage_name, name, value)
    
    def update_peak_memory(self, memory_usage: float):
        """Update peak memory usage if current usage is higher."""
        if memory_usage > self.metrics.memory_peak:
            self.metrics.memory_peak = memory_usage
    
    def _start_memory_monitoring(self):
        """Start background memory monitoring thread."""
        if not self.logger.config.enable_performance_logging:
            return
        
        self._memory_monitor_active = True
        self._memory_monitor_thread = threading.Thread(
            target=self._monitor_memory,
            daemon=True
        )
        self._memory_monitor_thread.start()
    
    def _stop_memory_monitoring(self):
        """Stop background memory monitoring."""
        self._memory_monitor_active = False
        if self._memory_monitor_thread:
            self._memory_monitor_thread.join(timeout=1.0)
    
    def _monitor_memory(self):
        """Background memory monitoring loop."""
        while self._memory_monitor_active:
            try:
                current_memory = ProcessingStageMetrics._get_memory_usage()
                self.update_peak_memory(current_memory)
                time.sleep(self.logger.config.metrics_collection_interval)
            except:
                break


class MetricsCollector:
    """
    Real-time metrics collection and aggregation system.
    
    Collects and aggregates performance metrics across processing sessions
    to provide insights into system performance trends and optimization opportunities.
    """
    
    def __init__(self, logger_name: str):
        """
        Initialize metrics collector.
        
        Args:
            logger_name: Name for this metrics collector instance
        """
        self.logger_name = logger_name
        self.session_metrics: List[CriticalPerformanceMetrics] = []
        self.stage_metrics: Dict[str, List[ProcessingStageMetrics]] = {}
        self.engine_performance: Dict[str, List[Dict[str, Any]]] = {}
        self._lock = threading.Lock()
    
    def add_session_metrics(self, metrics: CriticalPerformanceMetrics):
        """
        Add metrics from a completed processing session.
        
        Args:
            metrics: Performance metrics from processing session
        """
        with self._lock:
            metrics.calculate_efficiency_metrics()
            self.session_metrics.append(metrics)
    
    def add_stage_metrics(self, metrics: ProcessingStageMetrics):
        """
        Add metrics from a completed processing stage.
        
        Args:
            metrics: Stage performance metrics
        """
        with self._lock:
            if metrics.stage_name not in self.stage_metrics:
                self.stage_metrics[metrics.stage_name] = []
            self.stage_metrics[metrics.stage_name].append(metrics)
    
    def add_engine_performance(self, engine_name: str, performance_data: Dict[str, Any]):
        """
        Add performance data for a specific OCR engine.
        
        Args:
            engine_name: Name of the OCR engine
            performance_data: Performance metrics dictionary
        """
        with self._lock:
            if engine_name not in self.engine_performance:
                self.engine_performance[engine_name] = []
            self.engine_performance[engine_name].append(performance_data)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Generate comprehensive performance summary.
        
        Returns:
            Dictionary containing aggregated performance statistics
        """
        with self._lock:
            if not self.session_metrics:
                return {"error": "No performance data available"}
            
            # Aggregate session metrics
            total_sessions = len(self.session_metrics)
            successful_sessions = sum(1 for m in self.session_metrics if m.performance_targets_met)
            avg_processing_time = sum(m.total_processing_time for m in self.session_metrics) / total_sessions
            avg_extraction_rate = sum(m.character_extraction_rate for m in self.session_metrics) / total_sessions
            
            # Identify common bottlenecks
            all_bottlenecks = []
            for metrics in self.session_metrics:
                all_bottlenecks.extend(metrics.bottlenecks_identified)
            
            bottleneck_counts = {}
            for bottleneck in all_bottlenecks:
                bottleneck_counts[bottleneck] = bottleneck_counts.get(bottleneck, 0) + 1
            
            # Stage performance summary
            stage_summary = {}
            for stage_name, stage_list in self.stage_metrics.items():
                if stage_list:
                    avg_duration = sum(s.duration for s in stage_list) / len(stage_list)
                    avg_memory = sum(s.memory_peak for s in stage_list) / len(stage_list)
                    stage_summary[stage_name] = {
                        "avg_duration": round(avg_duration, 3),
                        "avg_memory_mb": round(avg_memory, 1),
                        "executions": len(stage_list)
                    }
            
            return {
                "total_sessions": total_sessions,
                "success_rate": round(successful_sessions / total_sessions, 3),
                "avg_processing_time": round(avg_processing_time, 3),
                "avg_extraction_rate": round(avg_extraction_rate, 1),
                "performance_targets_met": successful_sessions,
                "common_bottlenecks": bottleneck_counts,
                "stage_performance": stage_summary,
                "engine_count": len(self.engine_performance)
            }
    
    def clear_metrics(self):
        """Clear all collected metrics data."""
        with self._lock:
            self.session_metrics.clear()
            self.stage_metrics.clear()
            self.engine_performance.clear()


class OCRLogger:
    """
    Main OCR logging interface for the advanced OCR system.

    Provides comprehensive logging capabilities with focus on performance tracking,
    bottleneck identification, and optimization metrics for the critical performance
    requirements (text detection regions, TrOCR efficiency, processing speed).

    Features:
    - Structured JSON logging for analysis
    - Real-time performance monitoring
    - Processing stage timing and metrics
    - Memory usage tracking
    - Critical performance bottleneck detection
    - Development debugging with image snapshots

    Example:
        >>> logger = OCRLogger("ocr_pipeline")
        >>> logger.info("Starting OCR processing", extra={"image_size": (1920, 1080)})
        >>>
        >>> with logger.stage_timer("preprocessing") as timer:
        ...     processed_image = preprocess_image(image)
        ...     timer.log_metric("enhancement_applied", "contrast_boost")
        >>>
        >>> logger.log_critical_metrics(processing_time=2.1, regions_filtered=45, chars_extracted=1200)
    """
    
    def __init__(self, name: str, config: Optional[LogConfig] = None):
        """
        Initialize performance logger.
        
        Args:
            name: Logger name identifier
            config: Optional logging configuration (uses defaults if None)
        """
        self.name = name
        self.config = config or LogConfig()
        self.metrics_collector = MetricsCollector(name)
        self.current_session_start: Optional[float] = None
        self.current_session_metrics = CriticalPerformanceMetrics()
        
        # Initialize Python logger
        self.logger = logging.getLogger(f"advanced_ocr.{name}")
        self.logger.setLevel(getattr(logging, self.config.log_level.value))
        
        # Setup handlers
        self._setup_handlers()
        
        # Thread safety
        self._lock = threading.Lock()
    
    def _setup_handlers(self):
        """Setup logging handlers based on configuration."""
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Create formatter
        if self.config.log_format == LogFormat.JSON:
            formatter = self._create_json_formatter()
        else:
            formatter = self._create_structured_formatter()
        
        # Console handler
        if self.config.log_to_console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # File handler
        if self.config.log_to_file:
            log_path = self.config.log_file_path or f"advanced_ocr_{self.name}.log"
            
            # Create log directory if needed
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Use rotating file handler
            from logging.handlers import RotatingFileHandler
            file_handler = RotatingFileHandler(
                log_path,
                maxBytes=self.config.max_log_size_mb * 1024 * 1024,
                backupCount=5
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def _create_json_formatter(self) -> logging.Formatter:
        """Create JSON formatter for structured logging."""
        class JSONFormatter(logging.Formatter):
            def format(self, record):
                log_entry = {
                    "timestamp": time.time(),
                    "level": record.levelname,
                    "logger": record.name,
                    "message": record.getMessage(),
                    "module": record.module,
                    "function": record.funcName,
                    "line": record.lineno
                }
                
                # Add extra fields
                if hasattr(record, 'extra_data'):
                    log_entry.update(record.extra_data)
                
                return json.dumps(log_entry)
        
        return JSONFormatter()
    
    def _create_structured_formatter(self) -> logging.Formatter:
        """Create structured text formatter."""
        return logging.Formatter(
            fmt='%(asctime)s | %(name)s | %(levelname)-8s | %(funcName)s:%(lineno)d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    def start_session(self):
        """Start a new processing session for metrics tracking."""
        with self._lock:
            self.current_session_start = time.time()
            self.current_session_metrics = CriticalPerformanceMetrics()
        
        self.info("Processing session started")
    
    def end_session(self):
        """End current processing session and finalize metrics."""
        if self.current_session_start is None:
            self.warning("end_session called without start_session")
            return
        
        with self._lock:
            self.current_session_metrics.total_processing_time = time.time() - self.current_session_start
            self.current_session_metrics.calculate_efficiency_metrics()
            self.metrics_collector.add_session_metrics(self.current_session_metrics)
        
        self.info(
            "Processing session completed",
            extra={"performance_metrics": asdict(self.current_session_metrics)}
        )
    
    @contextmanager
    def stage_timer(self, stage_name: str):
        """
        Create context manager for timing processing stages.
        
        Args:
            stage_name: Name of the processing stage
            
        Yields:
            ProcessingStageTimer: Timer context manager
        """
        timer = ProcessingStageTimer(stage_name, self)
        try:
            yield timer
        finally:
            if self.config.enable_performance_logging:
                self.metrics_collector.add_stage_metrics(timer.metrics)
    
    def log_critical_metrics(self, **metrics):
        """
        Log critical performance metrics for optimization tracking.
        
        Args:
            **metrics: Critical performance metrics (processing_time, regions_filtered, etc.)
        """
        with self._lock:
            for key, value in metrics.items():
                if hasattr(self.current_session_metrics, key):
                    setattr(self.current_session_metrics, key, value)
        
        self.info("Critical performance metrics logged", extra={"critical_metrics": metrics})
    
    def log_engine_performance(self, engine_name: str, **performance_data):
        """
        Log performance data for specific OCR engine.
        
        Args:
            engine_name: Name of the OCR engine
            **performance_data: Engine performance metrics
        """
        performance_data['timestamp'] = time.time()
        self.metrics_collector.add_engine_performance(engine_name, performance_data)
        
        self.debug(
            f"Engine performance logged: {engine_name}",
            extra={"engine_performance": performance_data}
        )
    
    def check_performance_targets(self) -> bool:
        """
        Check if current session meets performance targets.
        
        Returns:
            bool: True if all performance targets are met
        """
        with self._lock:
            targets_met = (
                self.current_session_metrics.total_processing_time < 3.0 and
                self.current_session_metrics.text_regions_after_filtering <= 100 and
                self.current_session_metrics.character_extraction_rate >= 300
            )
        
        if targets_met:
            self.info("Performance targets met")
        else:
            self.warning(
                "Performance targets not met",
                extra={"current_metrics": asdict(self.current_session_metrics)}
            )
        
        return targets_met
    
    def log_performance_summary(self) -> Dict[str, Any]:
        """
        Log comprehensive performance summary.
        
        Returns:
            Dictionary containing performance summary statistics
        """
        summary = self.metrics_collector.get_performance_summary()
        
        self.info(
            "Performance summary generated",
            extra={"performance_summary": summary}
        )
        
        return summary
    
    def debug(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log debug message with optional extra data."""
        self._log_with_extra(logging.DEBUG, message, extra)
    
    def info(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log info message with optional extra data."""
        self._log_with_extra(logging.INFO, message, extra)
    
    def warning(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log warning message with optional extra data."""
        self._log_with_extra(logging.WARNING, message, extra)
    
    def error(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log error message with optional extra data."""
        self._log_with_extra(logging.ERROR, message, extra)
    
    def critical(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log critical message with optional extra data."""
        self._log_with_extra(logging.CRITICAL, message, extra)
    
    def _log_with_extra(self, level: int, message: str, extra: Optional[Dict[str, Any]]):
        """Internal method to log messages with extra data."""
        if extra:
            # Create a LogRecord with extra data
            record = self.logger.makeRecord(
                self.logger.name, level, __file__, 0, message, (), None
            )
            record.extra_data = extra
            self.logger.handle(record)
        else:
            self.logger.log(level, message)
    
    def _log_stage_start(self, stage_name: str):
        """Internal method to log processing stage start."""
        self.debug(f"Processing stage started: {stage_name}")
    
    def _log_stage_complete(self, stage_name: str, metrics: ProcessingStageMetrics):
        """Internal method to log processing stage completion."""
        self.debug(
            f"Processing stage completed: {stage_name}",
            extra={
                "stage_metrics": {
                    "duration": round(metrics.duration, 3),
                    "memory_peak": round(metrics.memory_peak, 1),
                    "custom_metrics": metrics.custom_metrics
                }
            }
        )
    
    def _log_custom_metric(self, stage_name: str, metric_name: str, metric_value: Any):
        """Internal method to log custom stage metrics."""
        if self.config.log_level == LogLevel.DEBUG:
            self.debug(
                f"Stage metric: {stage_name}.{metric_name} = {metric_value}"
            )


class OCRDebugLogger(OCRLogger):
    """
    Extended logger with debugging capabilities for development.
    
    Provides additional debugging features including image snapshot saving,
    detailed processing step logging, and enhanced error diagnostics.
    """
    
    def __init__(self, name: str, config: Optional[LogConfig] = None, debug_dir: Optional[str] = None):
        """
        Initialize debug logger.
        
        Args:
            name: Logger name identifier
            config: Optional logging configuration
            debug_dir: Directory for saving debug artifacts
        """
        super().__init__(name, config)
        self.debug_dir = Path(debug_dir) if debug_dir else Path("debug_output")
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        self._image_counter = 0
    
    def save_debug_image(self, image_array, stage_name: str, description: str = ""):
        """
        Save debug image snapshot during processing.
        
        Args:
            image_array: Image array to save (numpy array)
            stage_name: Processing stage name
            description: Optional description for the image
        """
        if not self.config.enable_debug_snapshots:
            return
        
        try:
            import cv2
            import numpy as np
            
            self._image_counter += 1
            filename = f"{self._image_counter:03d}_{stage_name}_{description}.jpg".replace(" ", "_")
            filepath = self.debug_dir / filename
            
            # Convert to uint8 if needed
            if image_array.dtype != np.uint8:
                if image_array.max() <= 1.0:
                    image_array = (image_array * 255).astype(np.uint8)
                else:
                    image_array = image_array.astype(np.uint8)
            
            cv2.imwrite(str(filepath), image_array)
            
            self.debug(
                f"Debug image saved: {filename}",
                extra={"debug_image": str(filepath), "stage": stage_name}
            )
            
        except Exception as e:
            self.error(f"Failed to save debug image: {e}")


# Factory functions for common logger configurations
def create_performance_logger(name: str, log_level: str = "INFO") -> OCRLogger:
    """
    Create performance logger with standard configuration.

    Args:
        name: Logger name
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        OCRLogger configured for performance tracking
    """
    config = LogConfig(
        log_level=LogLevel(log_level.upper()),
        log_format=LogFormat.STRUCTURED,
        enable_performance_logging=True,
        enable_debug_snapshots=False
    )
    return OCRLogger(name, config)


def create_debug_logger(name: str, debug_dir: str = "debug_output") -> OCRDebugLogger:
    """
    Create debug logger for development use.
    
    Args:
        name: Logger name
        debug_dir: Directory for debug artifacts
        
    Returns:
        OCRDebugLogger configured for development debugging
    """
    config = LogConfig(
        log_level=LogLevel.DEBUG,
        log_format=LogFormat.JSON,
        enable_performance_logging=True,
        enable_debug_snapshots=True
    )
    return OCRDebugLogger(name, config, debug_dir)


__all__ = [
    'OCRLogger',
    'OCRDebugLogger',
    'ProcessingStageTimer',
    'MetricsCollector',
    'LogConfig',
    'ProcessingStageMetrics',
    'CriticalPerformanceMetrics',
    'LogLevel',
    'LogFormat',
    'create_performance_logger',
    'create_debug_logger'
]

