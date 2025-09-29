from src.advanced_ocr import OCRLibrary, extract_text
image_path="tests/images/img5.jpg"
ocr = OCRLibrary()
result = ocr.process_image(image_path)
print(result.text)