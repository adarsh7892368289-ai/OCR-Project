# src/utils/logger.py

import logging
import sys
from pathlib import Path
from typing import Optional, Union

def setup_logger(name: str = "ocr_system", level: Union[str, int] = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Setup logger with consistent formatting"""
    
    logger = logging.getLogger(name)
    
    # Convert string level to int if needed
    if isinstance(level, str):
        numeric_level = getattr(logging, level.upper(), logging.INFO)
    else:
        numeric_level = level
        
    logger.setLevel(numeric_level)
    
    # Clear existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        try:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(str(log_path), encoding='utf-8')
            file_handler.setLevel(numeric_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            # If file logging fails, at least we have console logging
            logger.warning(f"Failed to setup file logging to {log_file}: {e}")
    
    # Prevent propagation to root logger to avoid duplicate messages
    logger.propagate = False
    
    return logger

def get_logger(name: str = "ocr_system", level: Union[str, int] = "INFO") -> logging.Logger:
    """Get or create a logger with the given name"""
    # Check if logger already exists
    existing_logger = logging.getLogger(name)
    if existing_logger.hasHandlers():
        return existing_logger
    
    # Create new logger if it doesn't exist or has no handlers
    return setup_logger(name, level)

# Default logger for the module
_default_logger = None

def get_default_logger() -> logging.Logger:
    """Get the default logger for the OCR system"""
    global _default_logger
    if _default_logger is None:
        try:
            _default_logger = setup_logger("ocr_system", "INFO", "logs/ocr_system.log")
        except Exception as e:
            # Fallback to console-only logging if file logging fails
            print(f"Warning: Failed to create default logger with file output: {e}")
            _default_logger = setup_logger("ocr_system", "INFO")
    return _default_logger

def configure_logger_from_config(config_dict: dict) -> logging.Logger:
    """Configure logger using configuration dictionary"""
    try:
        # Extract logging configuration
        log_level = config_dict.get('system', {}).get('log_level', 'INFO')
        log_file = config_dict.get('system', {}).get('log_file', None)
        
        # Create logger
        logger = setup_logger("ocr_system", log_level, log_file)
        return logger
        
    except Exception as e:
        # Fallback to basic logger
        print(f"Warning: Failed to configure logger from config: {e}")
        return setup_logger("ocr_system", "INFO")

# Module-level convenience functions
def debug(msg: str, *args, **kwargs):
    """Log debug message using default logger"""
    get_default_logger().debug(msg, *args, **kwargs)

def info(msg: str, *args, **kwargs):
    """Log info message using default logger"""
    get_default_logger().info(msg, *args, **kwargs)

def warning(msg: str, *args, **kwargs):
    """Log warning message using default logger"""
    get_default_logger().warning(msg, *args, **kwargs)

def error(msg: str, *args, **kwargs):
    """Log error message using default logger"""
    get_default_logger().error(msg, *args, **kwargs)

def critical(msg: str, *args, **kwargs):
    """Log critical message using default logger"""
    get_default_logger().critical(msg, *args, **kwargs)