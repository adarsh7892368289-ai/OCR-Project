from src.advanced_ocr import OCRLibrary
from src.advanced_ocr.types import  ProcessingOptions 
image_path="tests/images/img5.jpg"
ocr = OCRLibrary()
options = ProcessingOptions(enhance_image=True, engines=["easyocr"])
result = ocr.process_image(image_path, options)
print(result.text)