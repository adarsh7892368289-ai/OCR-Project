import numpy as np
import pytesseract
from typing import List, Dict, Any, Optional
import time
from PIL import Image
import os
import subprocess

from ..core.base_engine import (
    BaseOCREngine, 
    OCRResult, 
    DocumentResult, 
    TextRegion,
    BoundingBox,
    TextType
)

def find_tesseract():
    """Find Tesseract installation automatically"""
    possible_paths = [
        r'C:\Program Files\Tesseract-OCR\tesseract.exe',
        r'C:\Users\adbm\AppData\Local\Programs\Tesseract-OCR\tesseract.exe',
        r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
        r'C:\Users\{}\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'.format(os.getenv('USERNAME', '')),
        'tesseract',  # Linux/Mac default
        '/usr/bin/tesseract',
        '/usr/local/bin/tesseract'
    ]
    
    for path in possible_paths:
        try:
            if os.name == 'nt':  # Windows
                if os.path.exists(path):
                    return path
            else:  # Linux/Mac
                result = subprocess.run(['which', 'tesseract'], capture_output=True, text=True)
                if result.returncode == 0:
                    return result.stdout.strip()
        except:
            continue
    return None

class TesseractEngine(BaseOCREngine):
    """
    Modern Tesseract OCR Engine - Aligned with Pipeline
    
    Clean integration with your pipeline:
    - Takes preprocessed images from YOUR preprocessing pipeline
    - Returns single OCRResult compatible with YOUR base engine
    - Works with YOUR engine manager and postprocessing
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__("Tesseract", config)
        self.tesseract_config = self._build_tesseract_config()
        
        # Layout preservation settings
        self.line_height_threshold = self.config.get("line_height_threshold", 15)
        self.word_spacing_threshold = self.config.get("word_spacing_threshold", 20)
        
        self.supports_handwriting = False
        self.supports_multiple_languages = True
        self.supports_orientation_detection = True
        self.supports_structure_analysis = True
        
    def _build_tesseract_config(self) -> str:
        """Build optimized Tesseract configuration"""
        config_parts = ["--oem 1"]  # LSTM neural net mode
        
        # Page segmentation mode
        psm = self.config.get("psm", 6)
        config_parts.append(f"--psm {psm}")
        
        # Language configuration
        lang = self.config.get("lang", "eng")
        if isinstance(lang, list):
            lang = "+".join(lang)
        config_parts.append(f"-l {lang}")
        
        return " ".join(config_parts)
        
    def initialize(self) -> bool:
        """Initialize Tesseract engine"""
        try:
            tesseract_cmd = find_tesseract()
            if not tesseract_cmd:
                self.logger.error("Tesseract not found")
                return False
                
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
            version = pytesseract.get_tesseract_version()
            self.supported_languages = self._get_available_languages()
            
            self.is_initialized = True
            self.model_loaded = True
            
            self.logger.info(f"Tesseract {version} initialized with {len(self.supported_languages)} languages")
            return True
            
        except Exception as e:
            self.logger.error(f"Tesseract initialization failed: {e}")
            return False
            
    def _get_available_languages(self) -> List[str]:
        """Get available languages"""
        try:
            langs = pytesseract.get_languages()
            return [lang for lang in langs if lang != 'osd']
        except:
            return ["eng"]
            
    def get_supported_languages(self) -> List[str]:
        """Get supported languages"""
        return getattr(self, 'supported_languages', ["eng"])
        
    def process_image(self, preprocessed_image: np.ndarray, **kwargs) -> OCRResult:
        """
        FIXED: Process preprocessed image and return single OCRResult with preserved layout
        
        Args:
            preprocessed_image: Image from YOUR preprocessing pipeline
            **kwargs: Additional parameters
            
        Returns:
            OCRResult: Single result compatible with YOUR base engine
        """
        start_time = time.time()
        
        try:
            if not self.validate_image(preprocessed_image):
                raise ValueError("Invalid preprocessed image")
            
            # Convert to PIL Image for Tesseract
            pil_image = self._numpy_to_pil(preprocessed_image)
            
            # Extract OCR data
            ocr_data = pytesseract.image_to_data(
                pil_image,
                config=self.tesseract_config,
                output_type=pytesseract.Output.DICT
            )
            
            # FIXED: Combine all Tesseract detections with layout preservation
            result = self._combine_tesseract_results_with_layout(ocr_data)
            
            # Set processing time and engine info
            processing_time = time.time() - start_time
            result.processing_time = processing_time
            result.engine_name = self.name
            
            # Update stats
            self.processing_stats['total_processed'] += 1
            self.processing_stats['total_time'] += processing_time
            if result.text.strip():
                self.processing_stats['successful_extractions'] += 1
            
            return result
            
        except Exception as e:
            self.logger.error(f"OCR extraction failed: {e}")
            processing_time = time.time() - start_time
            self.processing_stats['errors'] += 1
            
            # Return empty result instead of raising
            return OCRResult(
                text="",
                confidence=0.0,
                processing_time=processing_time,
                engine_name=self.name,
                metadata={"error": str(e)}
            )
    
    def _numpy_to_pil(self, image: np.ndarray) -> Image.Image:
        """Convert numpy array to PIL Image"""
        if len(image.shape) == 3:
            if image.shape[2] == 3:
                # Assume BGR, convert to RGB
                image_rgb = image[:, :, ::-1]  # BGR to RGB
                return Image.fromarray(image_rgb)
            else:
                return Image.fromarray(image)
        else:
            # Grayscale
            return Image.fromarray(image)
    
    def _combine_tesseract_results_with_layout(self, data: Dict) -> OCRResult:
        """
        FIXED: Combine all Tesseract detections with preserved document layout
        """
        regions = []
        detection_count = 0
        
        for i in range(len(data['text'])):
            text = data['text'][i].strip()
            conf = float(data['conf'][i])
            
            if not text or conf <= 0:
                continue
                
            bbox = BoundingBox(
                x=int(data['left'][i]),
                y=int(data['top'][i]),
                width=int(data['width'][i]),
                height=int(data['height'][i]),
                confidence=conf / 100.0,
                text_type=TextType.PRINTED
            )
            
            # Create text region with spatial information
            region = TextRegion(
                text=text,
                confidence=conf / 100.0,
                bbox=bbox,
                text_type=TextType.PRINTED,
                language="en"
            )
            
            regions.append(region)
            detection_count += 1
        
        # FIXED: Reconstruct text with proper layout based on spatial positioning
        formatted_text = self._reconstruct_document_layout(regions)
        
        # Calculate overall confidence
        overall_confidence = sum(r.confidence for r in regions) / len(regions) if regions else 0.0
        
        # Create overall bounding box from all regions
        overall_bbox = self._calculate_overall_bbox(regions) if regions else BoundingBox(0, 0, 100, 30)
        
        # Return single OCRResult with preserved layout
        return OCRResult(
            text=formatted_text,
            confidence=overall_confidence,
            regions=regions,
            bbox=overall_bbox,
            level="page",
            engine_name=self.name,
            text_type=TextType.PRINTED,
            metadata={
                'detection_method': 'tesseract',
                'detection_count': detection_count,
                'tesseract_config': self.tesseract_config,
                'individual_confidences': [r.confidence for r in regions],
                'layout_preserved': True
            }
        )
    
    def _reconstruct_document_layout(self, regions: List[TextRegion]) -> str:
        """
        CORE FIX: Reconstruct document text with proper line breaks and formatting
        """
        if not regions:
            return ""
        
        # Sort regions by vertical position (top to bottom), then horizontal (left to right)
        sorted_regions = sorted(regions, key=lambda r: (r.bbox.y, r.bbox.x))
        
        # Group regions into lines based on Y-coordinate proximity
        lines = []
        current_line = []
        current_line_y = None
        
        for region in sorted_regions:
            region_center_y = region.bbox.y + region.bbox.height // 2
            
            if current_line_y is None:
                # First region
                current_line_y = region_center_y
                current_line.append(region)
            else:
                # Check if this region is on the same line
                y_distance = abs(region_center_y - current_line_y)
                
                if y_distance <= self.line_height_threshold:
                    # Same line - add to current line
                    current_line.append(region)
                else:
                    # New line - finish current line and start new one
                    if current_line:
                        lines.append(current_line)
                    current_line = [region]
                    current_line_y = region_center_y
        
        # Add the last line
        if current_line:
            lines.append(current_line)
        
        # Build the formatted text
        formatted_lines = []
        
        for line_regions in lines:
            # Sort regions in the line by X-coordinate (left to right)
            line_regions.sort(key=lambda r: r.bbox.x)
            
            # Combine text from regions in the line with appropriate spacing
            line_text_parts = []
            prev_region = None
            
            for region in line_regions:
                if prev_region is not None:
                    # Calculate horizontal distance between regions
                    horizontal_gap = region.bbox.x - (prev_region.bbox.x + prev_region.bbox.width)
                    
                    # Add extra spaces for large gaps (likely separate words/columns)
                    if horizontal_gap > self.word_spacing_threshold:
                        # Multiple spaces for large gaps
                        spaces = " " * min(4, max(2, horizontal_gap // 10))
                        line_text_parts.append(spaces)
                    else:
                        # Single space for normal word separation
                        line_text_parts.append(" ")
                
                line_text_parts.append(region.text)
                prev_region = region
            
            # Join the line parts and add to formatted lines
            line_text = "".join(line_text_parts).strip()
            if line_text:  # Only add non-empty lines
                formatted_lines.append(line_text)
        
        # Join all lines with newlines to preserve document structure
        return "\n".join(formatted_lines)
    
    def _calculate_overall_bbox(self, regions: List[TextRegion]) -> BoundingBox:
        """Calculate overall bounding box from all text regions"""
        if not regions:
            return BoundingBox(0, 0, 100, 30)
        
        min_x = min(r.bbox.x for r in regions)
        min_y = min(r.bbox.y for r in regions)
        max_x = max(r.bbox.x + r.bbox.width for r in regions)
        max_y = max(r.bbox.y + r.bbox.height for r in regions)
        
        return BoundingBox(
            x=min_x,
            y=min_y,
            width=max_x - min_x,
            height=max_y - min_y,
            confidence=sum(r.confidence for r in regions) / len(regions)
        )
    
    def batch_process(self, images: List[np.ndarray], **kwargs) -> List[OCRResult]:
        """Process multiple images - returns single result per image"""
        results = []
        for image in images:
            result = self.process_image(image, **kwargs)
            results.append(result)
        return results
    
    def get_engine_info(self) -> Dict[str, Any]:
        """Get engine information"""
        info = {
            'name': self.name,
            'version': None,
            'type': 'traditional_ocr',
            'supported_languages': self.get_supported_languages(),
            'capabilities': {
                'handwriting': self.supports_handwriting,
                'multiple_languages': self.supports_multiple_languages,
                'orientation_detection': self.supports_orientation_detection,
                'structure_analysis': self.supports_structure_analysis,
                'layout_preservation': True  # NEW: Added layout preservation capability
            },
            'optimal_for': ['printed_text', 'documents', 'books', 'forms'],
            'performance_profile': {
                'accuracy': 'high',
                'speed': 'medium',
                'memory_usage': 'low',
                'gpu_required': False
            },
            'pipeline_integration': {
                'uses_preprocessing_pipeline': True,
                'returns_single_result': True,
                'compatible_with_base_engine': True,
                'works_with_engine_manager': True,
                'preserves_document_layout': True  # NEW: Layout preservation feature
            }
        }
        
        try:
            info['version'] = str(pytesseract.get_tesseract_version())
        except:
            info['version'] = 'unknown'
            
        return info