import os
import cv2
import logging
from typing import List, Dict, Any, Tuple, Optional
import warnings
import numpy as np
import time

# Suppress the PaddlePaddle warnings
warnings.filterwarnings('ignore', category=UserWarning, module='paddle')

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
import sys
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

try:
    from ..core.base_engine import BaseOCREngine, OCRResult, BoundingBox, TextRegion, DocumentResult, TextType
    from ..utils.image_utils import ImageUtils
except ImportError as e:
    print(f"Import error in PaddleOCR engine: {e}")
    print("Please ensure the file structure is correct and all files are in place")
    raise

class PaddleOCREngine(BaseOCREngine):
    """A wrapper class for the PaddleOCR engine following BaseOCREngine interface."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__('PaddleOCR', config)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.ocr = None
        self.languages = self.config.get("languages", ["en"])
        self.gpu = self.config.get("gpu", True)
        self.model_storage_directory = self.config.get("model_dir", None)
        
    def initialize(self) -> bool:
        """Initialize the PaddleOCR engine"""
        try:
            self._initialize_paddleocr()
            if self.ocr is not None:
                self.is_initialized = True
                self.model_loaded = True
                self.logger.info("PaddleOCR engine initialized successfully")
                return True
            else:
                self.is_initialized = False
                self.model_loaded = False
                self.logger.error("Failed to initialize PaddleOCR")
                return False
        except Exception as e:
            self.ocr = None
            self.is_initialized = False
            self.model_loaded = False
            self.logger.error(f"PaddleOCR initialization failed: {e}")
            return False

    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages"""
        return ['en', 'ch', 'fr', 'de', 'ja', 'ko']  # PaddleOCR supports multiple languages

    def _initialize_paddleocr(self):
        """Initializes the PaddleOCR engine and handles common errors."""
        try:
            # Critical fix for OMP: Error #15
            os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

            from paddleocr import PaddleOCR

            # This will determine if a GPU is available and use it
            use_gpu = len(os.getenv('CUDA_VISIBLE_DEVICES', '')) > 0

            # Suppress specific logging from PaddleOCR
            self.ocr = PaddleOCR(
                use_angle_cls=True,
                lang='en',
                use_gpu=use_gpu,
                show_log=False
            )
            self.logger.info(f"PaddleOCR initialized successfully. GPU enabled: {use_gpu}")

        except ImportError:
            self.logger.error("PaddleOCR or its dependencies are not installed. Run: pip install paddleocr paddlepaddle")
            self.ocr = None
        except Exception as e:
            self.logger.error(f"PaddleOCR initialization failed: {e}")
            self.ocr = None

    def extract_text(self, image_path: str) -> List[Dict[str, Any]]:
        """
        Extracts text from an image path using PaddleOCR.
        Args:
            image_path (str): Path to the image file.
        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each containing
                                   'text', 'confidence', and 'bbox'.
        """
        if not self.ocr:
            self.logger.warning("PaddleOCR not initialized. Skipping extraction.")
            return []
            
        try:
            # Read the image from the path using OpenCV
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError(f"Image not found at path: {image_path}")
            
            # The PaddleOCR .ocr() method expects a BGR image.
            # Your original code was correct in this regard.
            # No need for manual conversion here, as .ocr() handles it.
            
            # Use cls=True for angle classification, which is essential for rotated text
            results = self.ocr.ocr(image, cls=True)
            extracted_texts = []
            
            # Check if results is not None and has at least one element
            if results and results[0]:
                for line in results[0]:
                    if line and len(line) >= 2:
                        bbox = line[0]
                        text_info = line[1]
                        text = text_info[0] if text_info else ""
                        confidence = text_info[1] if len(text_info) > 1 else 0.0
                        
                        if text.strip():
                            extracted_texts.append({
                                'text': text.strip(),
                                'confidence': float(confidence) * 100,
                                'bbox': bbox,
                                'engine': 'PADDLE'
                            })
            
            self.logger.info(f"PaddleOCR found {len(extracted_texts)} text regions.")
            return extracted_texts
            
        except Exception as e:
            self.logger.error(f"PaddleOCR extraction failed: {e}")
            return []

    def process_image(self, image: np.ndarray, **kwargs) -> DocumentResult:
        """Process image with PaddleOCR"""
        # CRITICAL FIX: Check both initialization and ocr existence
        if not self.is_initialized or self.ocr is None:
            # Try to reinitialize
            if not self.initialize():
                raise RuntimeError("PaddleOCR engine not initialized and cannot be initialized")

        start_time = time.time()

        try:
            # Enhanced preprocessing
            processed_image = self._preprocess_for_paddleocr(image)

            # CRITICAL FIX: Double-check ocr exists before using
            if self.ocr is None:
                raise RuntimeError("PaddleOCR instance is None")

            # PaddleOCR processing
            results = self.ocr.ocr(processed_image, cls=True)
            ocr_results = self._parse_paddleocr_results(results)

            # Get full text
            full_text = self._extract_full_text(ocr_results)

            # Create text regions from results
            text_regions = self._create_text_regions(ocr_results)

            # Calculate statistics
            processing_time = time.time() - start_time
            confidence_score = self.calculate_confidence(ocr_results)
            image_stats = self._calculate_image_stats(image)

            # Create OCR result for the page
            page_result = OCRResult(
                text=full_text,
                confidence=confidence_score,
                regions=text_regions,
                processing_time=processing_time,
                bbox=None,
                level="page"
            )

            # FIXED: Create document result with correct constructor
            return DocumentResult(
                pages=[page_result],
                metadata={'image_stats': image_stats},
                processing_time=processing_time,
                engine_name=self.name,
                confidence_score=confidence_score
            )

        except Exception as e:
            self.logger.error(f"PaddleOCR processing error: {e}")
            
            # Return properly constructed empty result
            empty_page = OCRResult(
                text="",
                confidence=0.0,
                regions=[],
                processing_time=time.time() - start_time,
                bbox=None,
                level="page"
            )
            
            return DocumentResult(
                pages=[empty_page],
                metadata={'error': str(e)},
                processing_time=time.time() - start_time,
                engine_name=self.name,
                confidence_score=0.0
            )

    def _preprocess_for_paddleocr(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image optimized for PaddleOCR"""
        # Convert to RGB if needed (PaddleOCR expects RGB)
        if len(image.shape) == 3:
            if image.shape[2] == 3:
                # Assume BGR and convert to RGB
                processed = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                processed = image
        else:
            # Convert grayscale to RGB
            processed = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # Resize if image is too large
        height, width = processed.shape[:2]
        max_dim = 1920  # Reduced for better performance

        if max(height, width) > max_dim:
            scale = max_dim / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            processed = cv2.resize(processed, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)

        return processed

    def _parse_paddleocr_results(self, ocr_results: List) -> List[OCRResult]:
        """Parse PaddleOCR results into standardized format"""
        results = []

        if not ocr_results or not ocr_results[0]:
            return results

        for detection in ocr_results[0]:
            if len(detection) >= 2:
                bbox_points, text_info = detection
                text = text_info[0] if text_info else ""
                confidence = text_info[1] if len(text_info) > 1 else 0.0

                if text.strip() and confidence > 0.1:  # Filter low confidence
                    # Convert polygon to bounding box
                    x, y, w, h = self._polygon_to_bbox(bbox_points)
                    bbox = BoundingBox(x=x, y=y, width=w, height=h, confidence=confidence)

                    result = OCRResult(
                        text=text.strip(),
                        confidence=confidence,
                        bbox=bbox,
                        level="word"
                    )

                    if self.validate_result(result):
                        results.append(result)

        return results

    def _create_text_regions(self, results: List[OCRResult]) -> List[TextRegion]:
        """Create text regions from OCR results"""
        text_regions = []
        
        for i, result in enumerate(results):
            region = TextRegion(
                text=result.text,
                confidence=result.confidence,
                bbox=result.bbox,
                text_type=TextType.PRINTED,  # PaddleOCR is optimized for printed text
                reading_order=i
            )
            text_regions.append(region)
            
        return text_regions

    def _polygon_to_bbox(self, points: List[List[float]]) -> Tuple[int, int, int, int]:
        """Convert polygon points to bounding box"""
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]

        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        return (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))

    def _extract_full_text(self, results: List[OCRResult]) -> str:
        """Extract full text preserving reading order"""
        if not results:
            return ""

        # Sort by Y coordinate first, then X coordinate with null safety
        sorted_results = sorted(results, key=lambda r: (r.bbox.y if r.bbox else 0, r.bbox.x if r.bbox else 0))

        # Group into lines
        lines = []
        current_line = []
        last_y = -1
        y_threshold = 30  # Increased threshold for better line grouping

        for result in sorted_results:
            if result.bbox:  # FIXED: Check if bbox exists
                y_pos = result.bbox.y

                if last_y == -1 or abs(y_pos - last_y) <= y_threshold:
                    current_line.append(result)
                    last_y = y_pos
                else:
                    if current_line:
                        lines.append(current_line)
                    current_line = [result]
                    last_y = y_pos

        if current_line:
            lines.append(current_line)

        # Combine lines into full text
        full_text_lines = []
        for line in lines:
            # Sort words in line by X position with null safety
            line_sorted = sorted(line, key=lambda r: r.bbox.x if r.bbox else 0)
            line_text = " ".join(result.text for result in line_sorted if result.text)
            if line_text.strip():  # Only add non-empty lines
                full_text_lines.append(line_text)

        return "\n".join(full_text_lines)

    def _calculate_image_stats(self, image: np.ndarray) -> Dict[str, Any]:
        """Calculate image statistics"""
        return {
            "width": image.shape[1],
            "height": image.shape[0],
            "channels": len(image.shape),
            "mean_brightness": np.mean(image),
            "std_brightness": np.std(image)
        }