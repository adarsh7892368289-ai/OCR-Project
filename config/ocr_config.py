"""
OCR Configuration settings
"""

# Tesseract configurations
TESSERACT_CONFIGS = {
    'standard': '--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,!?-()[]{}:;"\'/@#$%^&*+=<>|\\~`',
    'handwriting': '--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 ',
    'digits_only': '--psm 8 -c tessedit_char_whitelist=0123456789',
    'single_word': '--psm 8',
    'single_line': '--psm 7'
}

# Image preprocessing settings
PREPROCESSING_SETTINGS = {
    'blur_kernel_size': 5,
    'adaptive_thresh_block_size': 11,
    'adaptive_thresh_c': 2,
    'morph_kernel_size': (2, 2),
    'clahe_clip_limit': 2.0,
    'clahe_tile_grid_size': (8, 8)
}

# Confidence thresholds
CONFIDENCE_THRESHOLDS = {
    'tesseract_min': 30,
    'easyocr_min': 0.3,
    'paddle_min': 30,
    'trocr_min': 60
}

# Supported image formats
SUPPORTED_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
