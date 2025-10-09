"""Exception classes for the Advanced OCR Library.

Defines the exception hierarchy for handling OCR-related errors.
All exceptions inherit from OCRLibraryError for easy catching.

Examples
--------
    from advanced_ocr import OCRLibraryError, EngineNotAvailableError
    
    # Catch all OCR errors
    try:
        result = ocr.process_image("document.jpg")
    except OCRLibraryError as e:
        print(f"OCR failed: {e}")
    
    # Catch specific errors
    try:
        result = ocr.process_image("document.jpg")
    except EngineNotAvailableError:
        print("Install required OCR engine")
    except ImageProcessingError:
        print("Invalid or corrupted image")
"""


class OCRLibraryError(Exception):
    """Base exception for all OCR Library errors."""
    
    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}
    
    def __str__(self) -> str:
        if self.details:
            return f"{self.message} (Details: {self.details})"
        return self.message


class EngineNotAvailableError(OCRLibraryError):
    """Raised when a requested OCR engine is not available."""
    
    def __init__(self, engine_name: str, reason: str = None):
        message = f"OCR Engine '{engine_name}' is not available"
        if reason:
            message += f": {reason}"
        
        details = {"engine_name": engine_name}
        if reason:
            details["reason"] = reason
            
        super().__init__(message, details)
        self.engine_name = engine_name
        self.reason = reason


class ImageProcessingError(OCRLibraryError):
    """Raised when image loading or preprocessing fails."""
    
    def __init__(self, message: str, image_path: str = None, operation: str = None):
        details = {}
        if image_path:
            details["image_path"] = image_path
        if operation:
            details["operation"] = operation
            
        super().__init__(message, details)
        self.image_path = image_path
        self.operation = operation


class ConfigurationError(OCRLibraryError):
    """Raised when configuration is invalid or missing."""
    
    def __init__(self, message: str, config_key: str = None, config_value = None):
        details = {}
        if config_key:
            details["config_key"] = config_key
        if config_value is not None:
            details["config_value"] = str(config_value)
            
        super().__init__(message, details)
        self.config_key = config_key
        self.config_value = config_value


class QualityAnalysisError(OCRLibraryError):
    """Raised when image quality analysis fails."""
    
    def __init__(self, message: str, metric: str = None, image_path: str = None):
        details = {}
        if metric:
            details["metric"] = metric
        if image_path:
            details["image_path"] = image_path
            
        super().__init__(message, details)
        self.metric = metric
        self.image_path = image_path


class EnhancementError(OCRLibraryError):
    """Raised when image enhancement operations fail."""
    
    def __init__(self, message: str, enhancement_type: str = None, image_path: str = None):
        details = {}
        if enhancement_type:
            details["enhancement_type"] = enhancement_type
        if image_path:
            details["image_path"] = image_path
            
        super().__init__(message, details)
        self.enhancement_type = enhancement_type
        self.image_path = image_path


class ValidationError(OCRLibraryError):
    """Raised when input validation fails."""
    
    def __init__(self, message: str, parameter: str = None, expected_type: str = None):
        details = {}
        if parameter:
            details["parameter"] = parameter
        if expected_type:
            details["expected_type"] = expected_type
            
        super().__init__(message, details)
        self.parameter = parameter
        self.expected_type = expected_type


class ProcessingTimeoutError(OCRLibraryError):
    """Raised when OCR processing exceeds time limits."""
    
    def __init__(self, message: str, timeout_seconds: int = None):
        details = {}
        if timeout_seconds:
            details["timeout_seconds"] = timeout_seconds
            
        super().__init__(message, details)
        self.timeout_seconds = timeout_seconds


class EngineInitializationError(OCRLibraryError):
    """Raised when an OCR engine fails to initialize."""
    
    def __init__(self, message: str, engine_name: str = None, underlying_error: Exception = None):
        details = {}
        if engine_name:
            details["engine_name"] = engine_name
        if underlying_error:
            details["underlying_error"] = str(underlying_error)
            details["error_type"] = type(underlying_error).__name__
            
        super().__init__(message, details)
        self.engine_name = engine_name
        self.underlying_error = underlying_error


__all__ = [
    'OCRLibraryError',
    'EngineNotAvailableError', 
    'ImageProcessingError',
    'ConfigurationError',
    'QualityAnalysisError',
    'EnhancementError',
    'ValidationError',
    'ProcessingTimeoutError',
    'EngineInitializationError',
]