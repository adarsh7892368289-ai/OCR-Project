# Advanced OCR System

A modern, multi-engine OCR library with intelligent preprocessing and quality analysis for Python.

## Features

- **Multiple OCR Engines**: PaddleOCR, EasyOCR, Tesseract, TrOCR
- **Intelligent Preprocessing**: Automatic quality analysis and image enhancement
- **Flexible Processing**: Minimal, balanced, or enhanced processing strategies
- **Batch Processing**: Process multiple images efficiently
- **Type Safety**: Full type hints and Pydantic validation
- **Easy Integration**: Simple API with sensible defaults

## Installation

### Basic Installation
```bash
pip install advanced-ocr-system
```

### With Specific OCR Engines
```bash
# Lightweight engines (Tesseract + EasyOCR)
pip install advanced-ocr-system[tesseract,easyocr]

# AI-powered engines (PaddleOCR + TrOCR)  
pip install advanced-ocr-system[paddleocr,trocr]

# All engines
pip install advanced-ocr-system[all-engines]
```

## Quick Start

```python
from advanced_ocr import OCRLibrary, extract_text

# Simple usage
text = extract_text("document.jpg")
print(text)

# Advanced usage
ocr = OCRLibrary()
result = ocr.extract_text("document.jpg")

print(f"Text: {result.text}")
print(f"Confidence: {result.confidence}")
print(f"Engine used: {result.engine_used}")
```

## Processing Options

```python
from advanced_ocr import OCRLibrary, ProcessingOptions, ProcessingStrategy

ocr = OCRLibrary()
options = ProcessingOptions(
    engines=['paddleocr', 'easyocr'],      # Specific engines
    strategy=ProcessingStrategy.ENHANCED,   # Processing level
    enhance_image=True,                     # Apply enhancement
    min_confidence=0.8,                     # Quality threshold
    early_termination=True                  # Stop on high confidence
)

result = ocr.extract_text("image.jpg", options)
```

## Batch Processing

```python
from pathlib import Path

# Process multiple images
image_paths = list(Path("images/").glob("*.jpg"))
results = ocr.extract_text_batch(image_paths)

for result in results:
    if result.success:
        print(f"Extracted: {result.text[:100]}...")
    else:
        print(f"Failed: {result.metadata.get('error', 'Unknown error')}")
```

## Engine Information

```python
# Check available engines
engines = ocr.get_available_engines()
print(f"Available: {engines}")

# Get detailed engine info
engine_info = ocr.get_engine_info()
for name, info in engine_info.items():
    print(f"{name}: {info['success_rate']:.1%} success rate")
```

## Configuration

Create a YAML config file:

```yaml
engines:
  paddleocr:
    enabled: true
    priority: 1
    confidence_threshold: 0.7
  easyocr:
    enabled: true  
    priority: 2
    confidence_threshold: 0.6

preprocessing:
  quality_analysis: true
  enhancement: true
```

Load with custom config:
```python
ocr = OCRLibrary(config_path="config.yaml")
```

## Requirements

- Python 3.9+
- OpenCV 4.6+
- NumPy 1.24+
- PIL/Pillow 10.0+

### Optional Engine Dependencies

- **Tesseract**: `pytesseract`
- **EasyOCR**: `easyocr` 
- **PaddleOCR**: `paddlepaddle`, `paddleocr`
- **TrOCR**: `torch`, `transformers`, `timm`

## Engine Comparison

| Engine | Speed | Accuracy | Languages | Handwriting |
|--------|-------|----------|-----------|-------------|
| Tesseract | Fast | Good | 100+ | Limited |
| EasyOCR | Medium | Good | 80+ | Fair |
| PaddleOCR | Fast | Excellent | 80+ | Good |
| TrOCR | Slow | Excellent | Limited | Excellent |

## Development

```bash
# Clone repository
git clone https://github.com/yourusername/advanced-ocr-system.git
cd advanced-ocr-system

# Install development dependencies
pip install -e ".[dev,test]"

# Run tests
pytest

# Format code
black src/ tests/
isort src/ tests/

# Type checking
mypy src/
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions welcome! Please read our contributing guidelines and submit pull requests to our GitHub repository.