"""Tesseract OCR engine implementation for text extraction.

Provides OCR capabilities using Tesseract with enhanced confidence scoring and
layout-aware text reconstruction. Optimized for printed text with multiple
language support and adaptive quality enhancements.
"""

import numpy as np
import pytesseract
from typing import List, Dict, Any, Optional, Tuple
import time
import cv2
from PIL import Image, ImageEnhance
import warnings

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning)

from ..core.base_engine import BaseOCREngine
from ..types import OCRResult, TextRegion, BoundingBox, TextType


class TesseractEngine(BaseOCREngine):
    """Tesseract OCR engine for text extraction from images.
    
    Features:
    - Enhanced confidence scoring (adjusts pessimistic Tesseract scores)
    - Multiple language support
    - Automatic image quality enhancement
    - Layout-aware text reconstruction
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Tesseract engine with configuration."""
        super().__init__("Tesseract", config)
        
        # Tesseract specific settings
        self.psm = self.config.get("psm", 3)
        self.oem = self.config.get("oem", 1)
        self.lang = self.config.get("lang", "eng")
        
        # Layout reconstruction settings
        self.line_height_threshold = self.config.get("line_height_threshold", 15)
        self.word_spacing_threshold = self.config.get("word_spacing_threshold", 20)
        
        # Build Tesseract config string
        self.tesseract_config = self._build_config()
        
        # Engine capabilities
        self.supports_handwriting = False
        self.supports_multiple_languages = True
        self.supports_orientation_detection = True
        self.supports_structure_analysis = True
        
    def initialize(self) -> bool:
        """Initialize Tesseract engine."""
        try:
            # Test Tesseract availability
            version = pytesseract.get_tesseract_version()
            self.supported_languages = self._get_available_languages()
            
            # Test with dummy image
            dummy_image = Image.new('RGB', (100, 50), color='white')
            try:
                _ = pytesseract.image_to_string(dummy_image, config="--psm 6")
                self.logger.info(f"Tesseract {version} initialized successfully with {len(self.supported_languages)} languages")
            except Exception as test_error:
                self.logger.warning(f"Tesseract initialization test failed, but continuing: {test_error}")
            
            self.is_initialized = True
            return True
            
        except Exception as e:
            self.logger.error(f"Tesseract initialization failed: {e}")
            self.is_initialized = False
            return False
            
    def is_available(self) -> bool:
        """Check if Tesseract is available and initialized."""
        try:
            pytesseract.get_tesseract_version()
            return self.is_initialized
        except:
            return False
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        return getattr(self, 'supported_languages', ["eng"])
            
    def extract_text(self, image: np.ndarray) -> OCRResult:
        """Extract text from preprocessed image."""
        start_time = time.time()
        
        try:
            # Validate initialization
            if not self.is_initialized:
                if not self.initialize():
                    raise RuntimeError("Tesseract initialization failed")
                    
            if not self.validate_image(image):
                raise ValueError("Invalid image format")
            
            # Enhanced image preparation
            pil_image = self._prepare_image_enhanced(image)
            
            # Try multiple extraction approaches for better results
            regions = []
            best_confidence = 0.0
            best_regions = []
            
            # Approach 1: Standard extraction with configured PSM
            try:
                data = pytesseract.image_to_data(
                    pil_image,
                    config=self.tesseract_config,
                    output_type=pytesseract.Output.DICT
                )
                regions_1 = self._parse_results_enhanced(data)
                if regions_1:
                    avg_conf_1 = sum(r.confidence for r in regions_1) / len(regions_1)
                    if avg_conf_1 > best_confidence:
                        best_confidence = avg_conf_1
                        best_regions = regions_1
            except Exception as e:
                self.logger.warning(f"Standard extraction failed: {e}")
            
            # Approach 2: Try different PSM if first failed
            if not best_regions or best_confidence < 0.3:
                try:
                    alt_config = f"--oem {self.oem} --psm 6 -l {self.lang}"
                    data = pytesseract.image_to_data(
                        pil_image,
                        config=alt_config,
                        output_type=pytesseract.Output.DICT
                    )
                    regions_2 = self._parse_results_enhanced(data)
                    if regions_2:
                        avg_conf_2 = sum(r.confidence for r in regions_2) / len(regions_2)
                        if avg_conf_2 > best_confidence:
                            best_confidence = avg_conf_2
                            best_regions = regions_2
                except Exception as e:
                    self.logger.warning(f"Alternative PSM extraction failed: {e}")
            
            regions = best_regions
            
            if not regions:
                # Return empty result if no text found
                processing_time = time.time() - start_time
                return OCRResult(
                    text="",
                    confidence=0.0,
                    processing_time=processing_time,
                    engine_used=self.name,
                    regions=[],
                    metadata={'detection_count': 0, 'reason': 'no_text_detected'}
                )
            
            # Reconstruct document layout
            formatted_text = self._reconstruct_document_layout(regions)
            
            # Calculate overall confidence
            overall_confidence = self._calculate_overall_confidence(regions)
            
            processing_time = time.time() - start_time
            
            # Update statistics
            self.processing_stats['total_processed'] += 1
            self.processing_stats['total_time'] += processing_time
            if formatted_text.strip():
                self.processing_stats['successful_extractions'] += 1
                self.logger.info(f"SUCCESS: Tesseract extracted {len(formatted_text)} chars "
                               f"(conf: {overall_confidence:.3f}) in {processing_time:.2f}s")
            else:
                self.logger.warning(f"NO TEXT: Tesseract found no text in {processing_time:.2f}s")
            
            # Calculate overall bounding box
            overall_bbox = self._calculate_overall_bbox(regions) if regions else BoundingBox(0, 0, 100, 30)
            
            return OCRResult(
                text=formatted_text,
                confidence=overall_confidence,
                processing_time=processing_time,
                engine_used=self.name,
                regions=regions,
                bbox=overall_bbox,
                metadata={
                    'detection_method': 'tesseract',
                    'detection_count': len(regions),
                    'tesseract_config': self.tesseract_config,
                    'individual_confidences': [r.confidence for r in regions],
                    'layout_preserved': True
                }
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.processing_stats['errors'] += 1
            self.logger.error(f"Tesseract extraction failed: {e}")
            
            return OCRResult(
                text="",
                confidence=0.0,
                processing_time=processing_time,
                engine_used=self.name,
                regions=[],
                metadata={"error": str(e), "extraction_failed": True}
            )
    
    def _build_config(self) -> str:
        """Build optimized Tesseract configuration string."""
        config_parts = [
            f"--oem {self.oem}",
            f"--psm {self.psm}"
        ]
        
        # Language configuration
        if isinstance(self.lang, list):
            lang_str = "+".join(self.lang)
        else:
            lang_str = self.lang
        config_parts.append(f"-l {lang_str}")
        
        # Add performance and quality improvements
        config_parts.extend([
            "-c tessedit_char_whitelist=",
            "-c preserve_interword_spaces=1",
        ])
        
        return " ".join(config_parts)
        
    def _get_available_languages(self) -> List[str]:
        """Get available Tesseract languages."""
        try:
            langs = pytesseract.get_languages()
            return [lang for lang in langs if lang != 'osd']
        except:
            return ["eng"]
    
    def _prepare_image_enhanced(self, image: np.ndarray) -> Image.Image:
        """Convert image to PIL format with quality enhancements."""
        try:
            # Convert numpy to PIL
            if len(image.shape) == 3:
                if image.shape[2] == 3:
                    # Convert BGR to RGB
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(image_rgb)
                else:
                    pil_image = Image.fromarray(image)
            else:
                # Grayscale
                pil_image = Image.fromarray(image)
            
            # Ensure proper mode
            if pil_image.mode not in ('RGB', 'L'):
                pil_image = pil_image.convert('RGB')
            
            # Tesseract works better with larger images
            width, height = pil_image.size
            min_size = 300
            
            if width < min_size or height < min_size:
                scale_factor = max(min_size / width, min_size / height)
                new_size = (int(width * scale_factor), int(height * scale_factor))
                pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)
            
            # Apply enhancement if image appears low quality
            try:
                # Check if image needs enhancement
                img_array = np.array(pil_image.convert('L'))
                contrast_measure = np.std(img_array)
                
                if contrast_measure < 30:  # Low contrast
                    # Enhance contrast
                    enhancer = ImageEnhance.Contrast(pil_image)
                    pil_image = enhancer.enhance(1.5)
                    
                    # Enhance sharpness
                    enhancer = ImageEnhance.Sharpness(pil_image)
                    pil_image = enhancer.enhance(1.2)
                    
            except Exception as enhance_error:
                self.logger.warning(f"Image enhancement failed: {enhance_error}")
            
            return pil_image
            
        except Exception as e:
            self.logger.warning(f"Enhanced image preparation failed, using basic: {e}")
            # Fallback to basic conversion
            if len(image.shape) == 3:
                if image.shape[2] == 3:
                    image_rgb = image[:, :, ::-1]  # BGR to RGB
                    return Image.fromarray(image_rgb)
                return Image.fromarray(image)
            else:
                return Image.fromarray(image)
    
    def _parse_results_enhanced(self, data: Dict) -> List[TextRegion]:
        """Parse Tesseract results with confidence adjustment.
        
        Tesseract is pessimistic with confidence scores, so we apply
        intelligent boosting based on text characteristics.
        """
        regions = []
        
        for i in range(len(data['text'])):
            text = str(data['text'][i]).strip() if data['text'][i] else ""
            conf = float(data['conf'][i]) if data['conf'][i] != -1 else 0.0
            
            if not text:
                continue
            
            # Tesseract confidence adjustment
            original_confidence = conf / 100.0
            
            # Boost confidence for reasonable text
            adjusted_confidence = original_confidence
            if len(text) > 1:
                adjusted_confidence *= 1.3
            if text.isalnum() or any(c.isalnum() for c in text):
                adjusted_confidence *= 1.2
            if len(text.split()) > 1:
                adjusted_confidence *= 1.1
            if any(c.isupper() for c in text):
                adjusted_confidence *= 1.05
            
            # Cap at 1.0
            adjusted_confidence = min(adjusted_confidence, 1.0)
            
            # Lower threshold for Tesseract
            if adjusted_confidence < 0.1:
                continue
                
            bbox = BoundingBox(
                x=int(data['left'][i]),
                y=int(data['top'][i]),
                width=int(data['width'][i]),
                height=int(data['height'][i]),
                confidence=adjusted_confidence
            )
            
            region = TextRegion(
                text=text,
                confidence=adjusted_confidence,
                bbox=bbox,
                text_type=TextType.PRINTED,
                language=self.lang if isinstance(self.lang, str) else self.lang[0],
                reading_order=i
            )
            
            regions.append(region)
        
        self.logger.info(f"Parsed {len(regions)} valid regions from Tesseract")
        return regions
    
    def _calculate_overall_confidence(self, regions: List[TextRegion]) -> float:
        """Calculate overall confidence with generous scoring for Tesseract."""
        if not regions:
            return 0.0
        
        # Weight by text length and quality indicators
        total_weighted_conf = 0.0
        total_weight = 0.0
        
        for region in regions:
            # Weight by text characteristics
            weight = len(region.text)
            if region.text.isalnum():
                weight *= 1.2
            if len(region.text.split()) > 1:
                weight *= 1.1
                
            total_weighted_conf += region.confidence * weight
            total_weight += weight
        
        base_confidence = total_weighted_conf / total_weight if total_weight > 0 else 0.0
        
        # Apply bonuses for good extraction indicators
        if len(regions) > 10:
            base_confidence *= 1.15
        elif len(regions) > 5:
            base_confidence *= 1.1
            
        if sum(len(r.text) for r in regions) > 100:
            base_confidence *= 1.2
        elif sum(len(r.text) for r in regions) > 50:
            base_confidence *= 1.1
            
        return min(base_confidence, 1.0)
    
    def _reconstruct_document_layout(self, regions: List[TextRegion]) -> str:
        """Reconstruct document layout by grouping regions into lines."""
        if not regions:
            return ""
        
        # Sort by vertical position, then horizontal
        sorted_regions = sorted(regions, key=lambda r: (
            r.bbox.y + r.bbox.height // 2,
            r.bbox.x
        ))
        
        # Group regions into lines based on vertical proximity
        lines = []
        current_line = []
        
        for region in sorted_regions:
            region_center_y = region.bbox.y + region.bbox.height // 2
            
            if not current_line:
                current_line = [region]
            else:
                # Check if region belongs to current line
                should_group = False
                
                for line_region in current_line:
                    line_center_y = line_region.bbox.y + line_region.bbox.height // 2
                    y_distance = abs(region_center_y - line_center_y)
                    
                    # Adaptive threshold
                    threshold = min(
                        self.line_height_threshold,
                        min(region.bbox.height, line_region.bbox.height) * 0.8
                    )
                    
                    if y_distance <= threshold:
                        should_group = True
                        break
                
                if should_group:
                    current_line.append(region)
                else:
                    lines.append(current_line)
                    current_line = [region]
        
        if current_line:
            lines.append(current_line)
        
        # Assemble text with intelligent spacing
        formatted_lines = []
        
        for line_regions in lines:
            # Sort regions within line by X coordinate
            line_regions.sort(key=lambda r: r.bbox.x)
            
            line_parts = []
            
            for i, region in enumerate(line_regions):
                if i > 0:
                    # Calculate horizontal gap and add appropriate spacing
                    prev_region = line_regions[i-1]
                    gap = region.bbox.x - (prev_region.bbox.x + prev_region.bbox.width)
                    
                    if gap > self.word_spacing_threshold * 3:
                        line_parts.append("    ")
                    elif gap > self.word_spacing_threshold:
                        line_parts.append("  ")
                    elif gap > 8:
                        line_parts.append(" ")
                    else:
                        # Check if we need space
                        if not prev_region.text.endswith(' ') and not region.text.startswith(' '):
                            line_parts.append(" ")
                
                line_parts.append(region.text)
            
            if line_parts:
                line_text = "".join(line_parts).strip()
                if line_text:
                    formatted_lines.append(line_text)
        
        return "\n".join(formatted_lines)
    
    def _calculate_overall_bbox(self, regions: List[TextRegion]) -> BoundingBox:
        """Calculate overall bounding box encompassing all regions."""
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
            confidence=self._calculate_overall_confidence(regions)
        )


# Alias for compatibility
Tesseract = TesseractEngine