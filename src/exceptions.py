"""Custom exceptions for OCR Library"""

class OCRLibraryError(Exception):
    """Base exception for OCR Library errors"""
    pass

class EngineNotAvailableError(OCRLibraryError):
    """Raised when requested OCR engine is not available"""
    pass

class ImageLoadError(OCRLibraryError):
    """Raised when image cannot be loaded or is invalid"""
    pass

class ProcessingTimeoutError(OCRLibraryError):
    """Raised when processing exceeds time limits"""
    pass

class ConfigurationError(OCRLibraryError):
    """Raised when configuration is invalid"""
    pass