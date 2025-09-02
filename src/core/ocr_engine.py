import cv2
import numpy as np
import pytesseract
from PIL import Image
import easyocr
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch
import logging
import time
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class OCRProcessor:
    def __init__(self):
        """Initialize OCR engines"""
        self.results = {}
        self.initialize_engines()
    
    def initialize_engines(self):
        """Initialize all OCR engines"""
        # EasyOCR
        try:
            self.easyocr_reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
            logger.info("EasyOCR initialized successfully")
        except Exception as e:
            logger.error(f"EasyOCR initialization failed: {e}")
            self.easyocr_reader = None
        
        # PaddleOCR - FIXED initialization
        try:
            from paddleocr import PaddleOCR
            # Fixed: Remove the problematic parameters
            self.paddle_ocr = PaddleOCR(
                use_angle_cls=True,  # This is correct
                lang='en',
                use_gpu=torch.cuda.is_available(),
                show_log=False
            )
            logger.info("PaddleOCR initialized successfully")
        except Exception as e:
            logger.error(f"PaddleOCR initialization failed: {e}")
            self.paddle_ocr = None
        
        # TrOCR - Enhanced initialization
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.trocr_processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
            self.trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten").to(device)
            self.device = device
            logger.info(f"TrOCR initialized successfully on {device}")
        except Exception as e:
            logger.error(f"TrOCR initialization failed: {e}")
            self.trocr_processor = None
            self.trocr_model = None
    
    def preprocess_image(self, image_path: str) -> Dict[str, np.ndarray]:
        """Enhanced preprocessing for different OCR engines"""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Different preprocessing for different engines
        processed_images = {
            'original': img,
            'gray': gray,
            'tesseract': self._preprocess_for_tesseract(gray.copy()),
            'easyocr': self._preprocess_for_easyocr(gray.copy()),
            'paddleocr': self._preprocess_for_paddleocr(gray.copy()),
            'trocr': self._preprocess_for_trocr(gray.copy())
        }
        
        return processed_images
    
    def _preprocess_for_tesseract(self, img):
        """Preprocessing optimized for Tesseract"""
        # Denoise
        denoised = cv2.fastNlMeansDenoising(img)
        
        # Enhance contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # Adaptive threshold
        thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        
        # Morphological operations to clean up
        kernel = np.ones((1,1), np.uint8)
        processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        return processed
    
    def _preprocess_for_easyocr(self, img):
        """Preprocessing optimized for EasyOCR"""
        # EasyOCR works well with clean, sharp images
        # Sharpen the image
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(img, -1, kernel)
        
        # Enhance contrast
        enhanced = cv2.convertScaleAbs(sharpened, alpha=1.3, beta=10)
        
        return enhanced
    
    def _preprocess_for_paddleocr(self, img):
        """Preprocessing optimized for PaddleOCR"""
        # PaddleOCR prefers clean binary images
        # OTSU thresholding
        _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Remove noise
        kernel = np.ones((2,2), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        return cleaned
    
    def _preprocess_for_trocr(self, img):
        """Preprocessing optimized for handwritten text (TrOCR)"""
        # Gentle preprocessing for handwriting
        # Slight blur to smooth edges
        blurred = cv2.GaussianBlur(img, (3, 3), 0)
        
        # Gentle contrast enhancement
        enhanced = cv2.convertScaleAbs(blurred, alpha=1.1, beta=5)
        
        # Adaptive threshold with larger neighborhood
        thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                     cv2.THRESH_BINARY, 15, 8)
        
        return thresh
    
    def tesseract_ocr(self, img_dict: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """FIXED: Single Tesseract OCR call - no duplicates"""
        try:
            start_time = time.time()
            
            # Use the optimized image for Tesseract
            img = img_dict['tesseract']
            
            # Enhanced configuration for mixed content (receipts)
            custom_config = '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz.,/%-$:()[] '
            
            # Get detailed data
            data = pytesseract.image_to_data(img, config=custom_config, output_type=pytesseract.Output.DICT)
            
            # Filter results by confidence
            texts = []
            confidences = []
            for i in range(len(data['text'])):
                conf = int(data['conf'][i])
                text = data['text'][i].strip()
                if conf > 30 and text:  # Filter low confidence and empty text
                    texts.append(text)
                    confidences.append(conf)
            
            combined_text = " ".join(texts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            processing_time = int((time.time() - start_time) * 1000)
            
            return {
                "status": "SUCCESS" if combined_text else "ERROR",
                "text": combined_text,
                "confidence": round(avg_confidence, 1),
                "processing_time": processing_time,
                "text_length": len(combined_text)
            }
        
        except Exception as e:
            return {
                "status": "ERROR",
                "error": str(e),
                "processing_time": 0,
                "text_length": 0
            }
    
    def easyocr_process(self, img_dict: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """EasyOCR processing"""
        if not self.easyocr_reader:
            return {"status": "ERROR", "error": "EasyOCR not initialized"}
        
        try:
            start_time = time.time()
            
            img = img_dict['easyocr']
            
            # EasyOCR with optimized parameters
            results = self.easyocr_reader.readtext(
                img, 
                detail=1,
                paragraph=False,
                width_ths=0.7,
                height_ths=0.7
            )
            
            texts = []
            confidences = []
            for (bbox, text, conf) in results:
                if conf > 0.4:  # Lower threshold for receipts
                    texts.append(text)
                    confidences.append(conf * 100)
            
            combined_text = " ".join(texts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            processing_time = int((time.time() - start_time) * 1000)
            
            return {
                "status": "SUCCESS",
                "text": combined_text,
                "confidence": round(avg_confidence, 1),
                "processing_time": processing_time,
                "text_length": len(combined_text)
            }
        
        except Exception as e:
            return {
                "status": "ERROR",
                "error": str(e),
                "processing_time": 0,
                "text_length": 0
            }
    
    def paddleocr_process(self, img_dict: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """FIXED: PaddleOCR processing"""
        if not self.paddle_ocr:
            return {"status": "ERROR", "error": "PaddleOCR not initialized"}
        
        try:
            start_time = time.time()
            
            img = img_dict['paddleocr']
            
            # FIXED: Correct parameter usage for PaddleOCR
            results = self.paddle_ocr.ocr(img, cls=True)  # cls=True is correct
            
            texts = []
            confidences = []
            
            if results and results[0]:
                for line in results[0]:
                    if line and len(line) >= 2:
                        text_info = line[1]
                        if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                            text, conf = text_info[0], text_info[1]
                            if conf > 0.4:  # Lower threshold
                                texts.append(text)
                                confidences.append(conf * 100)
            
            combined_text = " ".join(texts)
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            processing_time = int((time.time() - start_time) * 1000)
            
            return {
                "status": "SUCCESS",
                "text": combined_text,
                "confidence": round(avg_confidence, 1),
                "processing_time": processing_time,
                "text_length": len(combined_text)
            }
        
        except Exception as e:
            return {
                "status": "ERROR",
                "error": str(e),
                "processing_time": 0,
                "text_length": 0
            }
    
    def trocr_handwriting_process(self, img_dict: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """ENHANCED: TrOCR processing for handwritten text"""
        if not self.trocr_processor or not self.trocr_model:
            return {"status": "ERROR", "error": "TrOCR not initialized"}
        
        try:
            start_time = time.time()
            
            # Use the handwriting-optimized image
            img = img_dict['trocr']
            
            # Convert to PIL Image
            pil_img = Image.fromarray(img).convert('RGB')
            
            # ENHANCED: Split image into regions for better handwriting detection
            width, height = pil_img.size
            
            # Look for handwritten regions (areas with specific characteristics)
            handwritten_regions = self._detect_handwritten_regions(img)
            
            all_texts = []
            
            # Process detected handwritten regions
            for region_coords in handwritten_regions:
                try:
                    x, y, w, h = region_coords
                    region = pil_img.crop((x, y, x + w, y + h))
                    
                    # Process with TrOCR
                    pixel_values = self.trocr_processor(region, return_tensors="pt").pixel_values
                    pixel_values = pixel_values.to(self.device)
                    
                    # Generate with better parameters
                    generated_ids = self.trocr_model.generate(
                        pixel_values,
                        max_length=100,  # Increased for longer text
                        num_beams=5,
                        early_stopping=True,
                        do_sample=False,
                        repetition_penalty=1.2
                    )
                    
                    generated_text = self.trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    
                    # Filter meaningful results
                    if len(generated_text.strip()) > 2 and not generated_text.strip().isdigit():
                        all_texts.append(generated_text.strip())
                
                except Exception as e:
                    logger.warning(f"TrOCR region processing failed: {e}")
                    continue
            
            # If no specific regions found, process the whole image
            if not all_texts:
                try:
                    pixel_values = self.trocr_processor(pil_img, return_tensors="pt").pixel_values
                    pixel_values = pixel_values.to(self.device)
                    
                    generated_ids = self.trocr_model.generate(
                        pixel_values,
                        max_length=100,
                        num_beams=3,
                        early_stopping=True
                    )
                    
                    generated_text = self.trocr_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    if generated_text.strip():
                        all_texts.append(generated_text.strip())
                
                except Exception as e:
                    logger.warning(f"TrOCR full image processing failed: {e}")
            
            combined_text = " ".join(all_texts) if all_texts else ""
            
            # Better confidence estimation
            confidence = 70.0 if combined_text and len(combined_text) > 5 else 30.0
            
            processing_time = int((time.time() - start_time) * 1000)
            
            return {
                "status": "SUCCESS",
                "text": combined_text,
                "confidence": confidence,
                "processing_time": processing_time,
                "text_length": len(combined_text)
            }
        
        except Exception as e:
            return {
                "status": "ERROR",
                "error": str(e),
                "processing_time": 0,
                "text_length": 0
            }
    
    def _detect_handwritten_regions(self, img):
        """Detect potential handwritten regions in the image"""
        # Find contours that might contain handwritten text
        contours, _ = cv2.findContours(cv2.bitwise_not(img), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by size and aspect ratio (typical for handwritten text)
            if w > 50 and h > 20 and w < 400 and h < 100:
                # Expand region slightly
                x = max(0, x - 10)
                y = max(0, y - 10)
                w = min(img.shape[1] - x, w + 20)
                h = min(img.shape[0] - y, h + 20)
                
                regions.append((x, y, w, h))
        
        return regions[:5]  # Limit to first 5 regions
    
    def process_image(self, image_path: str) -> Dict[str, Any]:
        """Main processing function - FIXED to avoid duplicates"""
        logger.info(f"Processing image: {image_path}")
        
        try:
            # Preprocess image
            img_dict = self.preprocess_image(image_path)
            
            # Process with each engine ONCE
            results = {}
            
            # 1. Tesseract OCR (SINGLE CALL)
            results["tesseract"] = self.tesseract_ocr(img_dict)
            
            # 2. EasyOCR
            results["easyocr"] = self.easyocr_process(img_dict)
            
            # 3. PaddleOCR (FIXED)
            results["paddleocr"] = self.paddleocr_process(img_dict)
            
            # 4. TrOCR Handwriting (ENHANCED)
            results["trocr_handwriting"] = self.trocr_handwriting_process(img_dict)
            
            # Find best result
            best_result = self._find_best_result(results)
            
            return {
                "best_result": best_result,
                "all_results": results
            }
        
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return {"error": str(e)}
    
    def _find_best_result(self, results: Dict[str, Dict]) -> Dict[str, Any]:
        """Find the best result based on confidence and text length"""
        valid_results = []
        
        for engine, result in results.items():
            if result.get('status') == 'SUCCESS' and result.get('text'):
                confidence = result.get('confidence', 0)
                text_length = len(result.get('text', ''))
                
                # Score based on confidence and reasonable text length
                score = (confidence * 0.7) + (min(text_length / 10, 50) * 0.3)
                valid_results.append((score, engine, result))
        
        if not valid_results:
            return {"text": "", "confidence": 0, "source": "none"}
        
        # Sort by score and return the best
        valid_results.sort(reverse=True)
        best_score, best_engine, best_result = valid_results[0]
        
        return {
            "text": best_result['text'],
            "confidence": best_result['confidence'],
            "source": best_engine,
            "processing_time": best_result.get('processing_time', 0)
        }