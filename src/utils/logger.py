# src/utils/logger.py

import logging
import sys
from pathlib import Path
from typing import Optional

def setup_logger(name: str, level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Setup logger with consistent formatting"""
    
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Get logger - alias for setup_logger for backward compatibility"""
    return setup_logger(name, level)

# Default logger for the module
_default_logger = None

def get_default_logger() -> logging.Logger:
    """Get the default logger for the OCR system"""
    global _default_logger
    if _default_logger is None:
        _default_logger = setup_logger("ocr_system", "INFO", "logs/ocr_system.log")
    return _default_logger