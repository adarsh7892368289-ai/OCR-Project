"""
Advanced OCR System - Public API
ONLY JOB: Provide clean public interface  
DEPENDENCIES: core.py, config.py
USED BY: End users
"""

from .core import OCRCore
from .config import OCRConfig
from .results import OCRResult, BatchResult

class OCR:
    """Main OCR class - Public API interface"""
    
    def __init__(self, config=None):
        """Initialize OCR with optional configuration"""
        self.core = OCRCore(config)
    
    def extract(self, image_input, config_override=None):
        """Extract text from single image"""
        return self.core.extract_text(image_input, config_override)
    
    def batch_extract(self, image_inputs, config_override=None):
        """Extract text from multiple images"""
        return self.core.batch_extract(image_inputs, config_override)

# Public exports
__all__ = ['OCR', 'OCRConfig', 'OCRResult', 'BatchResult']