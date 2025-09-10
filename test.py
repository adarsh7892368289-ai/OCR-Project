import cv2
from src.preprocessing.image_enhancer import ImageEnhancer

# Load and enhance image
image = cv2.imread('data/sample_images/img3.jpg')
print('Original image shape:', image.shape)

enhancer = ImageEnhancer()
result = enhancer.enhance_image(image)

print('Enhancement result type:', type(result))
print('Enhancement result attributes:', [attr for attr in dir(result) if not attr.startswith('_')])

if hasattr(result, 'enhanced_image'):
    print('Enhanced image shape:', result.enhanced_image.shape)
    print('Enhanced image type:', type(result.enhanced_image))
    # Try to save it manually
    cv2.imwrite('test_enhanced.jpg', result.enhanced_image)
    print('Manual save completed')
else:
    print('No enhanced_image attribute found')
    
import cv2
from src.engines.tesseract_engine import TesseractEngine

image = cv2.imread('data/sample_images/img3.jpg')
engine = TesseractEngine()
result = engine.process_image(image)

print('Tesseract result type:', type(result))
print('Tesseract result length:', len(result) if isinstance(result, list) else 'Not a list')
if isinstance(result, list) and len(result) > 0:
    print('First result item:', result[0])
    print('First result type:', type(result[0]))
    if hasattr(result[0], '__dict__'):
        print('First result attributes:', vars(result[0]))