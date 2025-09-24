import os
from src.advanced_ocr import OCRLibrary

ocr = OCRLibrary()
image_path = os.path.join("tests", "fixtures", "images", "img3.jpg")
result = ocr.extract_text(image_path)
print(f"Text: {result.text}")
print(f"Confidence: {result.confidence}")
print(f"Engine used: {result.engine_name}")
