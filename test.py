# Simple text extraction
from src import OCRLibrary

ocr = OCRLibrary()
result = ocr.extract_text("data\sample_images\img1.jpg")
print(f"Text: {result.text}")
print(f"Confidence: {result.confidence}")
print(f"Engine used: {result.engine_used}")