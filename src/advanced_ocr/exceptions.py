# src/advanced_ocr/exceptions.py
"""
Custom exception classes for the Advanced OCR Library.
Provides specific error types for different failure scenarios.
"""


class OCRLibraryError(Exception):
    """
    Base exception class for all OCR Library errors.
    
    This is the parent class for all custom exceptions in the library.
    Catch this to handle any OCR-related error generically.
    """
    
    def __init__(self, message: str, details: dict = None):
        """
        Initialize OCR Library error.
        
        Args:
            message: Human-readable error message
            details: Optional dictionary with additional error context
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}
    
    def __str__(self) -> str:
        if self.details:
            return f"{self.message} (Details: {self.details})"
        return self.message


class EngineNotAvailableError(OCRLibraryError):
    """
    Raised when a requested OCR engine is not available.
    
    This can happen when:
    - Engine dependencies are not installed
    - Engine failed to initialize
    - Engine is not registered in the manager
    """
    
    def __init__(self, engine_name: str, reason: str = None):
        """
        Initialize engine not available error.
        
        Args:
            engine_name: Name of the unavailable engine
            reason: Optional reason why engine is not available
        """
        if reason:
            message = f"OCR Engine '{engine_name}' is not available: {reason}"
        else:
            message = f"OCR Engine '{engine_name}' is not available"
        
        details = {"engine_name": engine_name}
        if reason:
            details["reason"] = reason
            
        super().__init__(message, details)
        self.engine_name = engine_name
        self.reason = reason


class ImageProcessingError(OCRLibraryError):
    """
    Raised when image preprocessing or loading fails.
    
    This can happen when:
    - Image file cannot be loaded
    - Image format is not supported  
    - Image is corrupted or invalid
    - Preprocessing operations fail
    """
    
    def __init__(self, message: str, image_path: str = None, operation: str = None):
        """
        Initialize image processing error.
        
        Args:
            message: Error description
            image_path: Path to problematic image (if applicable)
            operation: Failed operation name (if applicable)
        """
        details = {}
        if image_path:
            details["image_path"] = image_path
        if operation:
            details["operation"] = operation
            
        super().__init__(message, details)
        self.image_path = image_path
        self.operation = operation


class ConfigurationError(OCRLibraryError):
    """
    Raised when there are configuration issues.
    
    This can happen when:
    - Invalid configuration values
    - Missing required configuration
    - Configuration file parsing fails
    """
    
    def __init__(self, message: str, config_key: str = None, config_value = None):
        """
        Initialize configuration error.
        
        Args:
            message: Error description
            config_key: Problematic configuration key
            config_value: Problematic configuration value
        """
        details = {}
        if config_key:
            details["config_key"] = config_key
        if config_value is not None:
            details["config_value"] = str(config_value)
            
        super().__init__(message, details)
        self.config_key = config_key
        self.config_value = config_value


class QualityAnalysisError(OCRLibraryError):
    """
    Raised when image quality analysis fails.
    
    This can happen when:
    - Quality metrics calculation fails
    - Image analysis operations fail
    - Quality thresholds are not met
    """
    pass


class EnhancementError(OCRLibraryError):
    """
    Raised when image enhancement fails.
    
    This can happen when:
    - Enhancement algorithms fail
    - Enhanced image is worse than original
    - Enhancement operations timeout
    """
    pass


class ValidationError(OCRLibraryError):
    """
    Raised when input validation fails.
    
    This can happen when:
    - Invalid input parameters
    - Missing required parameters
    - Parameter types are incorrect
    """
    
    def __init__(self, message: str, parameter: str = None, expected_type: str = None):
        """
        Initialize validation error.
        
        Args:
            message: Error description
            parameter: Name of invalid parameter
            expected_type: Expected parameter type
        """
        details = {}
        if parameter:
            details["parameter"] = parameter
        if expected_type:
            details["expected_type"] = expected_type
            
        super().__init__(message, details)
        self.parameter = parameter
        self.expected_type = expected_type


class ProcessingTimeoutError(OCRLibraryError):
    """
    Raised when OCR processing exceeds time limits.
    
    This can happen when:
    - Processing takes longer than max_processing_time
    - Engine becomes unresponsive
    - Large images require too much processing time
    """
    
    def __init__(self, message: str, timeout_seconds: int = None):
        """
        Initialize processing timeout error.
        
        Args:
            message: Error description
            timeout_seconds: Timeout limit that was exceeded
        """
        details = {}
        if timeout_seconds:
            details["timeout_seconds"] = timeout_seconds
            
        super().__init__(message, details)
        self.timeout_seconds = timeout_seconds


class EngineInitializationError(OCRLibraryError):
    """
    Raised when an OCR engine fails to initialize properly.
    
    This can happen when:
    - Engine dependencies are missing
    - Model files cannot be loaded
    - GPU/hardware requirements not met
    """
    
    def __init__(self, message: str, engine_name: str = None, underlying_error: Exception = None):
        """
        Initialize engine initialization error.
        
        Args:
            message: Error description
            engine_name: Name of engine that failed to initialize
            underlying_error: Original exception that caused the failure
        """
        details = {}
        if engine_name:
            details["engine_name"] = engine_name
        if underlying_error:
            details["underlying_error"] = str(underlying_error)
            details["error_type"] = type(underlying_error).__name__
            
        super().__init__(message, details)
        self.engine_name = engine_name
        self.underlying_error = underlying_error


# Export all exception classes
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