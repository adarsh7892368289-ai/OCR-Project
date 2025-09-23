# File: README_USAGE.md
# Comprehensive usage documentation

"""
# Advanced OCR System - Usage Guide

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Simple Usage
```python
from advanced_ocr import AdvancedOCR

# Initialize OCR system
with AdvancedOCR() as ocr:
    result = ocr.extract_text("path/to/your/image.jpg")
    print(f"Text: {result.text}")
    print(f"Confidence: {result.confidence}")
```

### 3. Command Line Usage
```bash
# Basic usage
python run_ocr.py path/to/your/image.jpg

# Advanced usage
python examples/command_line_usage.py --image document.jpg --mode accurate --output result.json
```

## Input Formats Supported

- **File paths**: `"image.jpg"`, `"/full/path/to/image.png"`
- **PIL Images**: `Image.open("image.jpg")`
- **NumPy arrays**: `np.array(image)`
- **Bytes**: Raw image bytes from files or web requests

## Processing Modes

- **FAST**: Single engine (Tesseract), minimal preprocessing
- **BALANCED**: Two engines with fusion, good speed/accuracy balance
- **ACCURATE**: All engines, maximum preprocessing, best quality

## Configuration Examples

### Development Configuration
```python
config = OCRConfig.create_development_config()
# Fast processing, single engine, minimal resources
```

### Production Configuration
```python
config = OCRConfig.create_production_config() 
# Multiple engines, full preprocessing, high accuracy
```

### Custom Configuration
```python
config = OCRConfig(
    mode=ProcessingMode.ACCURATE,
    engines=[EngineType.TESSERACT, EngineType.PADDLEOCR],
    confidence_threshold=0.7,
    enable_preprocessing=True,
    use_gpu=True
)
```

## Environment Variables

Create a `.env` file in your project root:

```
OCR_USE_GPU=true
OCR_MAX_WORKERS=4
OCR_CONFIDENCE_THRESHOLD=0.5
TESSERACT_LANG=eng
PADDLEOCR_LANG=en
```

## Error Handling

```python
try:
    with AdvancedOCR() as ocr:
        result = ocr.extract_text("image.jpg")
        if result.confidence < 0.5:
            print("Warning: Low confidence result")
        print(result.text)
except Exception as e:
    print(f"OCR failed: {e}")
```

## Batch Processing

```python
image_paths = ["img1.jpg", "img2.jpg", "img3.jpg"]
with AdvancedOCR() as ocr:
    results = ocr.extract_text_batch(image_paths)
    for i, result in enumerate(results):
        print(f"Image {i+1}: {result.text[:50]}...")
```

## Tips for Best Results

1. **Image Quality**: Use high-resolution, well-lit images
2. **Preprocessing**: Enable preprocessing for poor quality images
3. **Engine Selection**: Use multiple engines for better accuracy
4. **Language Settings**: Set correct language in engine configs
5. **GPU Usage**: Enable GPU for faster processing with neural engines

## Troubleshooting

### Common Issues:
- **ModuleNotFoundError**: Check if all dependencies are installed
- **Tesseract not found**: Install tesseract binary and set path
- **GPU errors**: Disable GPU with `OCR_USE_GPU=false`
- **Memory errors**: Reduce image size or disable some engines

### Performance Optimization:
- Use FAST mode for quick processing
- Enable GPU for neural engines
- Adjust max_image_size for your hardware
- Use appropriate number of workers
"""