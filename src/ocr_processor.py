import cv2
import numpy as np
import pytesseract
import easyocr
from PIL import Image, ImageEnhance, ImageFilter
import time
import logging
from typing import List, Dict, Tuple, Optional
import os

try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False
    logging.warning("PaddleOCR not available. Install with: pip install paddlepaddle paddleocr")

try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    import torch
    TROCR_AVAILABLE = True
except ImportError:
    TROCR_AVAILABLE = False
    logging.warning("TrOCR not available. Install with: pip install transformers torch")

class OCRResult:
    def __init__(self, engine: str, text: str, confidence: float, processing_time: float, 
                 status: str = 'success', error: str = None):
        self.engine = engine
        self.text = text
        self.confidence = confidence
        self.processing_time = processing_time
        self.status = status
        self.error = error

    def __str__(self):
        return f"{self.engine}: {self.confidence:.1f}% ({self.processing_time:.0f}ms) - {self.text[:50]}..."

class AdvancedOCRProcessor:
    def __init__(self, log_level=logging.INFO):
        logging.basicConfig(level=log_level)
        self.logger = logging.getLogger(__name__)
        
        # Initialize OCR engines
        self.easyocr_reader = None
        self.paddle_ocr = None
        self.trocr_processor = None
        self.trocr_model = None
        
        self._initialize_engines()

    def _initialize_engines(self):
        """Initialize all available OCR engines"""
        try:
            # EasyOCR initialization
            self.easyocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available() if 'torch' in globals() else False)
            self.logger.info("EasyOCR initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize EasyOCR: {e}")
            
        # --- Tesseract Initialization ---
        try:
            # Check for Windows OS
            if os.name == 'nt':
                # Define potential paths to check, prioritizing the common one
                potential_paths = [
                    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                    r'C:\Users\adbm\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
                ]
                
                path_found = False
                for path in potential_paths:
                    if os.path.exists(path):
                        pytesseract.pytesseract.tesseract_cmd = path
                        self.logger.info(f"Tesseract path set to: {path}")
                        path_found = True
                        break
                
                if not path_found:
                    self.logger.warning("Tesseract path could not be found. Please install Tesseract or check your path.")
            else:
                # For non-Windows systems, assume Tesseract is in the system's PATH
                pytesseract.pytesseract.tesseract_cmd = 'tesseract'
                self.logger.info("Tesseract path automatically set for non-Windows systems.")

        except Exception as e:
            self.logger.warning(f"Failed to configure Tesseract path: {e}")

        if PADDLE_AVAILABLE:
            try:
                self.paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en')
                self.logger.info("PaddleOCR initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize PaddleOCR: {e}")

        if TROCR_AVAILABLE:
            try:
                self.trocr_processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
                self.trocr_model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
                self.logger.info("TrOCR initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize TrOCR: {e}")

    def preprocess_image(self, image_path: str, method: str = 'standard') -> np.ndarray:
        """Advanced image preprocessing for better OCR accuracy"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")

        if method == 'standard':
            return self._standard_preprocess(image)
        elif method == 'handwriting':
            return self._handwriting_preprocess(image)
        elif method == 'enhanced':
            return self._enhanced_preprocess(image)
        else:
            return image

    def _standard_preprocess(self, image: np.ndarray) -> np.ndarray:
        """Standard preprocessing for printed text"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Noise removal
        denoised = cv2.medianBlur(gray, 5)
        
        # Thresholding
        _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return thresh

    def _handwriting_preprocess(self, image: np.ndarray) -> np.ndarray:
        """Specialized preprocessing for handwritten text"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Gaussian blur to smooth handwriting
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Adaptive threshold for varying lighting
        adaptive_thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Morphological operations to clean up
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_CLOSE, kernel)
        
        return cleaned

    def _enhanced_preprocess(self, image: np.ndarray) -> np.ndarray:
        """Enhanced preprocessing with multiple techniques"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Bilateral filter to reduce noise while keeping edges sharp
        filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        # Adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            filtered, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        return thresh

    def tesseract_ocr(self, image_path: str, config: str = '--psm 6') -> OCRResult:
        """Tesseract OCR processing"""
        start_time = time.time()
        
        try:
            # Preprocess image
            processed_image = self.preprocess_image(image_path, 'enhanced')
            
            # OCR with confidence scores
            data = pytesseract.image_to_data( processed_image, config="--oem 3 --psm 6 -l eng -c preserve_interword_spaces=1", output_type=pytesseract.Output.DICT )
            
            # Extract text and calculate average confidence
            texts = []
            confidences = []
            
            for i in range(len(data['text'])):
                if int(data['conf'][i]) > 30:  # Filter low confidence
                    text = data['text'][i].strip()
                    if text:
                        texts.append(text)
                        confidences.append(int(data['conf'][i]))
            
            full_text = ' '.join(texts)
            avg_confidence = np.mean(confidences) if confidences else 0
            
            processing_time = (time.time() - start_time) * 1000
            
            return OCRResult(
                engine="Tesseract OCR",
                text=full_text,
                confidence=avg_confidence,
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            return OCRResult(
                engine="Tesseract OCR",
                text="",
                confidence=0,
                processing_time=processing_time,
                status='error',
                error=str(e)
            )

    def easyocr_ocr(self, image_path: str) -> OCRResult:
        """EasyOCR processing"""
        start_time = time.time()
        
        try:
            if self.easyocr_reader is None:
                raise Exception("EasyOCR not initialized")
            
            # Preprocess image
            processed_image = self.preprocess_image(image_path, 'enhanced')
            
            results = self.easyocr_reader.readtext(processed_image)
            
            texts = []
            confidences = []
            
            for (bbox, text, confidence) in results:
                if confidence > 0.3:  # Filter low confidence
                    texts.append(text)
                    confidences.append(confidence * 100)
            
            full_text = ' '.join(texts)
            avg_confidence = np.mean(confidences) if confidences else 0
            
            processing_time = (time.time() - start_time) * 1000
            
            return OCRResult(
                engine="EasyOCR",
                text=full_text,
                confidence=avg_confidence,
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            return OCRResult(
                engine="EasyOCR",
                text="",
                confidence=0,
                processing_time=processing_time,
                status='error',
                error=str(e)
            )

    def paddle_ocr_process(self, image_path: str) -> OCRResult:
        """PaddleOCR processing"""
        start_time = time.time()
        
        try:
            if self.paddle_ocr is None:
                raise Exception("PaddleOCR not initialized")
            
            results = self.paddle_ocr.ocr(image_path, cls=True)
            
            texts = []
            confidences = []
            
            for line in results:
                if line:
                    for word_info in line:
                        text = word_info[1][0]
                        confidence = word_info[1][1] * 100
                        
                        if confidence > 30:  # Filter low confidence
                            texts.append(text)
                            confidences.append(confidence)
            
            full_text = ' '.join(texts)
            avg_confidence = np.mean(confidences) if confidences else 0
            
            processing_time = (time.time() - start_time) * 1000
            
            return OCRResult(
                engine="PaddleOCR",
                text=full_text,
                confidence=avg_confidence,
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            return OCRResult(
                engine="PaddleOCR",
                text="",
                confidence=0,
                processing_time=processing_time,
                status='error',
                error=str(e)
            )

    def trocr_handwriting(self, image_path: str) -> OCRResult:
        """TrOCR for handwritten text"""
        start_time = time.time()
        
        try:
            if self.trocr_processor is None or self.trocr_model is None:
                raise Exception("TrOCR not initialized")
            
            # Preprocess for handwriting
            processed_image = self.preprocess_image(image_path, 'handwriting') 
            pil_image = Image.fromarray(processed_image).convert("RGB")
            
            pixel_values = self.trocr_processor(images=pil_image, return_tensors="pt").pixel_values
            
            generated_ids = self.trocr_model.generate(pixel_values)
            generated_text = self.trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            processing_time = (time.time() - start_time) * 1000
            
            # TrOCR doesn't provide confidence scores, so we estimate based on text quality
            confidence = min(95, max(60, len(generated_text.strip()) * 2))
            
            return OCRResult(
                engine="TrOCR Handwriting",
                text=generated_text.strip(),
                confidence=confidence,
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            return OCRResult(
                engine="TrOCR Handwriting",
                text="",
                confidence=0,
                processing_time=processing_time,
                status='error',
                error=str(e)
            )

    def process_image_all_engines(self, image_path: str) -> List[OCRResult]:
        """Process image with all available OCR engines"""
        self.logger.info(f"Processing image: {image_path}")
        
        results = []
        
        # Tesseract - Standard mode
        results.append(self.tesseract_ocr(image_path, '--psm 6'))
        
        # Tesseract - Handwriting mode
        results.append(self.tesseract_ocr(image_path, '--psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 '))
        
        # EasyOCR
        results.append(self.easyocr_ocr(image_path))
        
        # PaddleOCR
        if PADDLE_AVAILABLE:
            results.append(self.paddle_ocr_process(image_path))
        
        # TrOCR for handwriting
        if TROCR_AVAILABLE:
            results.append(self.trocr_handwriting(image_path))
        
        return results

    def get_best_result(self, results: List[OCRResult]) -> Optional[OCRResult]:
        """Get the best OCR result based on confidence and text length"""
        successful_results = [r for r in results if r.status == 'success' and r.text.strip()]
        
        if not successful_results:
            return None
        
        # Sort by confidence score and text length
        best_result = max(successful_results, 
                         key=lambda x: (x.confidence, len(x.text.strip())))
        
        return best_result

    def print_results(self, results: List[OCRResult]):
        """Print formatted OCR results"""
        print("\n" + "="*80)
        print("OCR PROCESSING RESULTS")
        print("="*80)
        
        # Print best result first
        best_result = self.get_best_result(results)
        if best_result:
            print(f"\n🏆 BEST RESULT ({best_result.engine}):")
            print(f"Confidence: {best_result.confidence:.1f}%")
            print(f"Processing Time: {best_result.processing_time:.0f}ms")
            print(f"Text: {best_result.text}")
            print("\n" + "-"*80)
        
        # Print all results
        print("\nALL OCR ENGINE RESULTS:")
        print("-"*80)
        
        for i, result in enumerate(results, 1):
            status_icon = "✅" if result.status == 'success' else "❌"
            print(f"\n{i}. {status_icon} {result.engine}")
            print(f"   Status: {result.status.upper()}")
            
            if result.status == 'success':
                print(f"   Confidence: {result.confidence:.1f}%")
                print(f"   Processing Time: {result.processing_time:.0f}ms")
                print(f"   Text Length: {len(result.text)} characters")
                print(f"   Text: {result.text}")
            else:
                print(f"   Error: {result.error}")
        
        print("\n" + "="*80)
