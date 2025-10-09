"""Logging utilities for consistent logging across the library.

Provides centralized logging setup and configuration for all library components.

Examples
--------
    from advanced_ocr.utils.logging import setup_logging, setup_logger
    
    # Setup library-wide logging
    setup_logging(level="DEBUG", log_file="ocr.log")
    
    # Get logger for a specific component
    logger = setup_logger("OCRLibrary")
    logger.info("Processing started")
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Union


def setup_logging(level: Union[str, int] = "INFO", 
                  log_file: Optional[str] = None,
                  name: str = "advanced_ocr") -> None:
    """Set up logging configuration for the entire library."""
    # Convert string level to int if needed
    if isinstance(level, str):
        numeric_level = getattr(logging, level.upper(), logging.INFO)
    else:
        numeric_level = level
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Get root logger for the library
    root_logger = logging.getLogger(name)
    root_logger.setLevel(numeric_level)
    
    # Clear existing handlers to avoid duplicates
    if root_logger.hasHandlers():
        root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        try:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(str(log_path), encoding='utf-8')
            file_handler.setLevel(numeric_level)
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)
        except Exception as e:
            root_logger.warning(f"Failed to setup file logging to {log_file}: {e}")
    
    # Prevent propagation to avoid duplicate messages
    root_logger.propagate = False


def setup_logger(name: str, level: Union[str, int] = "INFO") -> logging.Logger:
    """Set up a logger for a specific component."""
    # Convert string level to int if needed
    if isinstance(level, str):
        numeric_level = getattr(logging, level.upper(), logging.INFO)
    else:
        numeric_level = level
    
    # Create logger with full name
    logger_name = f"advanced_ocr.{name}"
    logger = logging.getLogger(logger_name)
    logger.setLevel(numeric_level)
    
    # If no handlers exist, set up basic logging
    if not logging.getLogger("advanced_ocr").hasHandlers():
        setup_logging(level=level)
    
    return logger


def get_logger(name: str = "advanced_ocr") -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)


__all__ = [
    "setup_logging",
    "setup_logger", 
    "get_logger",
]