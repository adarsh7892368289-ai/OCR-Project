"""
Advanced OCR System - Image Preprocessing Orchestrator
Orchestrate image enhancement based on quality analysis for optimal OCR results.
"""

import cv2
import numpy as np
import logging
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass
import time

from .quality_analyzer import QualityAnalyzer, QualityMetrics, QualityIssue
from ..config import OCRConfig
from ..utils.image_utils import ImageProcessor as ImageUtils, CoordinateTransformer
from ..results import ProcessingMetrics


@dataclass
class PreprocessingResult:
    """Result of image preprocessing operation."""
    
    enhanced_image: np.ndarray
    original_image: np.ndarray
    quality_metrics: QualityMetrics
    processing_metrics: ProcessingMetrics
    transformations_applied: Dict[str, Any]
    scale_factor: float = 1.0
    coordinate_transformer: Optional[CoordinateTransformer] = None


class ImageProcessor:
    """
    Advanced image preprocessing orchestrator.
    Applies adaptive enhancement based on quality analysis.
    """
    
    def __init__(self, config: Optional[OCRConfig] = None):
        """
        Initialize image processor.
        
        Args:
            config: OCR configuration with preprocessing settings
        """
        self.config = config or OCRConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.quality_analyzer = QualityAnalyzer(config)
        self.image_utils = ImageUtils()
        
        # CONSERVATIVE Enhancement settings for OCR
        self.enhancement_settings = {
            'max_dimension': getattr(self.config, 'max_image_dimension', 4096),  # Higher for OCR
            'min_dimension': getattr(self.config, 'min_image_dimension', 800),   # Higher minimum
            'target_dpi': getattr(self.config, 'target_dpi', 300),
            'enable_adaptive_enhancement': getattr(self.config, 'enable_adaptive_enhancement', True),
            'enhancement_strength': getattr(self.config, 'enhancement_strength', 0.3),  # Much lower
            'preserve_aspect_ratio': getattr(self.config, 'preserve_aspect_ratio', True),
            'quality_improvement_threshold': -0.1,  # Allow small degradation but not major
        }
    
    def process_image(self, image: np.ndarray) -> PreprocessingResult:
        """
        Process image with adaptive enhancement based on quality analysis.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            PreprocessingResult with enhanced image and metadata
        """
        start_time = time.time()
        
        if image is None or image.size == 0:
            raise ValueError("Invalid input image")
        
        try:
            # Store original image
            original_image = image.copy()
            
            self.logger.info(f"Processing image: {image.shape}")
            
            # Step 1: Initial quality analysis
            quality_metrics = self.quality_analyzer.analyze_quality(image)
            self.logger.debug(f"Initial quality: {quality_metrics.overall_score:.3f}")
            
            # Step 2: Check if image already good for OCR (EARLY EXIT)
            if quality_metrics.overall_score > 0.7 and quality_metrics.ocr_readiness > 0.6:
                self.logger.info("Image already high quality, applying minimal processing only")
                return self._apply_minimal_processing(original_image, quality_metrics, start_time)
            
            # Step 3: Apply preprocessing transformations
            processed_image, transformations = self._apply_preprocessing_pipeline(
                image, quality_metrics
            )
            
            # Step 4: Final quality check
            final_quality = self.quality_analyzer.analyze_quality(processed_image)
            self.logger.debug(f"Final quality: {final_quality.overall_score:.3f}")
            
            # Step 5: QUALITY IMPROVEMENT VALIDATION
            quality_improvement = final_quality.overall_score - quality_metrics.overall_score
            
            if quality_improvement < self.enhancement_settings['quality_improvement_threshold']:
                self.logger.warning(f"Processing degraded quality by {-quality_improvement:.3f}, reverting to minimal processing")
                return self._apply_minimal_processing(original_image, quality_metrics, start_time)
            
            # Step 6: Create coordinate transformer
            scale_factor = transformations.get('scale_factor', 1.0)
            
            # Step 7: Create processing metrics
            processing_metrics = ProcessingMetrics(stage_name="image_preprocessing")
            processing_metrics.start_time = start_time
            processing_metrics.finish()

            # Add the custom metadata to the metadata dictionary
            processing_metrics.metadata['quality_improvement'] = quality_improvement
            processing_metrics.metadata['transformations_count'] = len(transformations)
            
            result = PreprocessingResult(
                enhanced_image=processed_image,
                original_image=original_image,
                quality_metrics=final_quality,
                processing_metrics=processing_metrics,
                transformations_applied=transformations,
                scale_factor=scale_factor,
            )
            
            self.logger.info(
                f"Image processed in {processing_metrics.duration:.2f}s: "
                f"Quality {quality_metrics.overall_score:.3f} -> {final_quality.overall_score:.3f}"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Image processing failed: {e}")
            # Return minimal result with original image
            processing_metrics = ProcessingMetrics(
                stage_name="image_preprocessing"
            )
            processing_metrics.start_time = start_time
            processing_metrics.finish()
            
            return PreprocessingResult(
                enhanced_image=original_image,
                original_image=original_image,
                quality_metrics=QualityMetrics(),
                processing_metrics=processing_metrics,
                transformations_applied={}
            )
    
    def _apply_minimal_processing(self, image: np.ndarray, quality_metrics: QualityMetrics, start_time: float) -> PreprocessingResult:
        """Apply only essential transformations for already good images."""
        processed_image = image.copy()
        transformations = {}
        
        # Only essential processing
        processed_image, format_info = self._normalize_image_format(processed_image)
        transformations.update(format_info)
        
        # Only fix severe skew
        if abs(quality_metrics.skew_angle) > 3.0:  # Higher threshold
            processed_image, rotation_info = self._correct_skew(
                processed_image, quality_metrics.skew_angle
            )
            transformations.update(rotation_info)
        
        # Gentle resize only if extremely large
        height, width = processed_image.shape[:2]
        if max(height, width) > 5000:  # Only if very large
            scale_factor = 4000 / max(height, width)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            processed_image = cv2.resize(processed_image, (new_width, new_height), cv2.INTER_AREA)
            transformations.update({
                'minimal_resize': True,
                'scale_factor': scale_factor,
                'original_size': (width, height),
                'new_size': (new_width, new_height)
            })
        else:
            transformations['scale_factor'] = 1.0
        
        # Create metrics
        processing_metrics = ProcessingMetrics(stage_name="image_preprocessing")
        processing_metrics.start_time = start_time
        processing_metrics.finish()
        processing_metrics.metadata['quality_improvement'] = 0.0  # Minimal processing
        processing_metrics.metadata['transformations_count'] = len(transformations)
        processing_metrics.metadata['minimal_processing'] = True
        
        return PreprocessingResult(
            enhanced_image=processed_image,
            original_image=image,
            quality_metrics=quality_metrics,
            processing_metrics=processing_metrics,
            transformations_applied=transformations,
            scale_factor=transformations.get('scale_factor', 1.0),
        )
    
    def _apply_preprocessing_pipeline(self, image: np.ndarray, 
                                    quality_metrics: QualityMetrics) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply preprocessing pipeline based on quality analysis.
        
        Args:
            image: Input image
            quality_metrics: Quality assessment results
            
        Returns:
            Tuple of (processed_image, transformations_applied)
        """
        processed_image = image.copy()
        transformations = {}
        
        # Step 1: Normalize image format
        processed_image, format_info = self._normalize_image_format(processed_image)
        transformations.update(format_info)
        
        # Step 2: Correct geometric distortions (higher threshold)
        if abs(quality_metrics.skew_angle) > 2.0:  # Less sensitive
            processed_image, rotation_info = self._correct_skew(
                processed_image, quality_metrics.skew_angle
            )
            transformations.update(rotation_info)
        
        # Step 3: Conservative resolution optimization
        processed_image, resize_info = self._optimize_resolution_conservative(processed_image, quality_metrics)
        transformations.update(resize_info)
        
        # Step 4: Apply adaptive enhancements (much more conservative)
        if self.enhancement_settings['enable_adaptive_enhancement']:
            processed_image, enhancement_info = self._apply_adaptive_enhancements_conservative(
                processed_image, quality_metrics
            )
            transformations.update(enhancement_info)
        
        # Step 5: Final normalization
        processed_image = self._final_normalization(processed_image)
        
        return processed_image, transformations
    
    def _optimize_resolution_conservative(self, image: np.ndarray, 
                                        quality_metrics: QualityMetrics) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Conservative resolution optimization prioritizing OCR readability.
        """
        height, width = image.shape[:2]
        original_size = (width, height)
        resize_info = {'resolution_optimized': False}
        
        max_dim = self.enhancement_settings['max_dimension']  # 4096
        min_dim = self.enhancement_settings['min_dimension']  # 800
        
        # More conservative conditions
        needs_upscaling = (min(width, height) < min_dim) and quality_metrics.resolution_score < 0.4  # Higher threshold
        needs_downscaling = max(width, height) > max_dim  # Only if very large
        
        if needs_upscaling or needs_downscaling:
            if needs_upscaling:
                # Conservative upscaling
                scale_factor = min_dim / min(width, height)
                scale_factor = min(scale_factor, 2.0)  # Max 2x upscaling
                interpolation = cv2.INTER_CUBIC
                resize_info['upscaled'] = True
            else:
                # Conservative downscaling - preserve more detail
                scale_factor = max_dim / max(width, height)
                # Ensure we don't go below good OCR resolution
                if min(width, height) * scale_factor < 1200:  # Ensure good OCR resolution
                    scale_factor = 1200 / min(width, height)
                interpolation = cv2.INTER_AREA
                resize_info['downscaled'] = True
            
            # Apply resize
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            
            resized_image = cv2.resize(image, (new_width, new_height), interpolation=interpolation)
            
            resize_info.update({
                'resolution_optimized': True,
                'scale_factor': scale_factor,
                'original_size': original_size,
                'new_size': (new_width, new_height),
                'interpolation': 'cubic' if interpolation == cv2.INTER_CUBIC else 'area'
            })
            
            self.logger.debug(
                f"Conservative resize: {original_size} -> {(new_width, new_height)} "
                f"(scale: {scale_factor:.2f}x)"
            )
            
            return resized_image, resize_info
        
        resize_info['scale_factor'] = 1.0
        return image, resize_info
    
    def _apply_adaptive_enhancements_conservative(self, image: np.ndarray, 
                                                quality_metrics: QualityMetrics) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply very conservative enhancements.
        """
        enhanced_image = image.copy()
        enhancement_info = {}
        
        # Much lower base enhancement strength
        base_strength = self.enhancement_settings['enhancement_strength']  # 0.3
        enhancement_params = quality_metrics.enhancement_params
        
        # 1. Very conservative sharpening
        if 'sharpen_strength' in enhancement_params:
            strength = enhancement_params['sharpen_strength'] * base_strength * 0.5  # Even lower
            strength = min(strength, 0.2)  # Cap at 20%
            if strength > 0.05:  # Only if meaningful
                enhanced_image = self._apply_sharpening(enhanced_image, strength)
                enhancement_info['sharpening_applied'] = strength
        
        # 2. Conservative contrast enhancement (only if really needed)
        if 'contrast_enhancement' in enhancement_params and quality_metrics.contrast_score < 0.4:
            strength = enhancement_params['contrast_enhancement'] * base_strength * 0.6
            strength = min(strength, 0.3)  # Cap at 30%
            if strength > 0.1:  # Only if meaningful
                enhanced_image = self._apply_contrast_enhancement_conservative(enhanced_image, strength)
                enhancement_info['contrast_enhancement_applied'] = strength
        
        # 3. Skip noise reduction unless severe (OCR engines handle noise well)
        if 'noise_reduction_strength' in enhancement_params and quality_metrics.noise_score < 0.3:
            strength = enhancement_params['noise_reduction_strength'] * base_strength * 0.5
            strength = min(strength, 0.3)  # Very conservative
            if strength > 0.1:
                enhanced_image = self._apply_noise_reduction_conservative(enhanced_image, strength)
                enhancement_info['noise_reduction_applied'] = strength
        
        # 4. Conservative illumination correction (only if severe)
        if enhancement_params.get('illumination_correction', False) and quality_metrics.lighting_score < 0.3:
            enhanced_image = self._apply_illumination_correction_conservative(enhanced_image)
            enhancement_info['illumination_correction_applied'] = True
        
        return enhanced_image, enhancement_info
    
    def _apply_contrast_enhancement_conservative(self, image: np.ndarray, strength: float) -> np.ndarray:
        """Conservative contrast enhancement."""
        try:
            if len(image.shape) == 3:
                lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                l_channel = lab[:, :, 0]
                
                # Very gentle CLAHE
                clip_limit = 1.0 + strength * 2.0  # Lower clip limit
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
                l_channel = clahe.apply(l_channel)
                
                lab[:, :, 0] = l_channel
                enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            else:
                clip_limit = 1.0 + strength * 2.0
                clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
                enhanced = clahe.apply(image)
            
            return enhanced
        except Exception as e:
            self.logger.warning(f"Conservative contrast enhancement failed: {e}")
            return image
    
    def _apply_noise_reduction_conservative(self, image: np.ndarray, strength: float) -> np.ndarray:
        """Very conservative noise reduction."""
        try:
            # Only gentle median filtering
            kernel_size = 3  # Small kernel only
            
            if len(image.shape) == 3:
                # Gentle bilateral filter
                diameter = 5
                sigma_color = 20 + strength * 30
                sigma_space = 20 + strength * 30
                denoised = cv2.bilateralFilter(image, diameter, sigma_color, sigma_space)
            else:
                denoised = cv2.medianBlur(image, kernel_size)
            
            return denoised
        except Exception as e:
            self.logger.warning(f"Conservative noise reduction failed: {e}")
            return image
    
    def _apply_illumination_correction_conservative(self, image: np.ndarray) -> np.ndarray:
        """Very conservative illumination correction."""
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Smaller kernel for gentler correction
            kernel_size = max(gray.shape) // 80  # Smaller kernel
            kernel_size = max(kernel_size, 15)
            if kernel_size % 2 == 0:
                kernel_size += 1
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            background = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
            background = cv2.GaussianBlur(background, (kernel_size, kernel_size), 0)
            
            # Gentler correction
            if len(image.shape) == 3:
                corrected = image.copy().astype(np.float32)
                background_3ch = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR).astype(np.float32)
                
                # More conservative normalization
                corrected = corrected / (background_3ch / 255.0 + 0.1) * 255.0  # Higher offset
                corrected = np.clip(corrected, 0, 255).astype(np.uint8)
            else:
                corrected = gray.astype(np.float32)
                corrected = corrected / (background.astype(np.float32) / 255.0 + 0.1) * 255.0
                corrected = np.clip(corrected, 0, 255).astype(np.uint8)
            
            return corrected
        except Exception as e:
            self.logger.warning(f"Conservative illumination correction failed: {e}")
            return image
    
    # Keep all other methods unchanged
    def _normalize_image_format(self, image: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Normalize image to standard format for processing.
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (normalized_image, format_info)
        """
        format_info = {'format_normalized': False}
        
        # Ensure image is in correct data type
        if image.dtype != np.uint8:
            image = cv2.convertScaleAbs(image)
            format_info['dtype_converted'] = True
        
        # Handle different channel configurations
        if len(image.shape) == 3:
            if image.shape[2] == 4:  # RGBA
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
                format_info['alpha_channel_removed'] = True
            elif image.shape[2] == 3:  # RGB/BGR
                # Keep as is for now, engines will handle color conversion
                pass
        elif len(image.shape) == 2:  # Grayscale
            # Keep as grayscale
            pass
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")
        
        format_info['format_normalized'] = True
        return image, format_info
    
    def _correct_skew(self, image: np.ndarray, skew_angle: float) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Correct image skew by rotation.
        
        Args:
            image: Input image
            skew_angle: Detected skew angle in degrees
            
        Returns:
            Tuple of (corrected_image, rotation_info)
        """
        try:
            # Correct the skew by rotating in opposite direction
            rotation_angle = -skew_angle
            
            # Get image center and rotation matrix
            height, width = image.shape[:2]
            center = (width // 2, height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
            
            # Calculate new bounding dimensions
            cos_angle = abs(rotation_matrix[0, 0])
            sin_angle = abs(rotation_matrix[0, 1])
            new_width = int((height * sin_angle) + (width * cos_angle))
            new_height = int((height * cos_angle) + (width * sin_angle))
            
            # Adjust translation
            rotation_matrix[0, 2] += (new_width - width) / 2
            rotation_matrix[1, 2] += (new_height - height) / 2
            
            # Apply rotation
            corrected_image = cv2.warpAffine(
                image, rotation_matrix, (new_width, new_height),
                flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
            )
            
            rotation_info = {
                'skew_corrected': True,
                'rotation_angle': rotation_angle,
                'original_skew': skew_angle
            }
            
            self.logger.debug(f"Corrected skew: {skew_angle:.2f}° -> {rotation_angle:.2f}°")
            
            return corrected_image, rotation_info
            
        except Exception as e:
            self.logger.warning(f"Skew correction failed: {e}")
            return image, {'skew_correction_failed': True}
    
    def _apply_sharpening(self, image: np.ndarray, strength: float) -> np.ndarray:
        """Apply adaptive sharpening."""
        try:
            # Create sharpening kernel based on strength
            center_value = 4 + strength * 4
            kernel = np.array([
                [0, -1, 0],
                [-1, center_value, -1],
                [0, -1, 0]
            ])
            kernel = kernel / kernel.sum() if kernel.sum() != 0 else kernel
            
            # Apply sharpening
            sharpened = cv2.filter2D(image, -1, kernel)
            
            # Blend with original based on strength
            alpha = 0.5 + strength * 0.5  # 0.5 to 1.0
            enhanced = cv2.addWeighted(image, 1 - alpha, sharpened, alpha, 0)
            
            return enhanced
            
        except Exception as e:
            self.logger.warning(f"Sharpening failed: {e}")
            return image
    
    def _final_normalization(self, image: np.ndarray) -> np.ndarray:
        """Apply final normalization to the processed image."""
        # Ensure proper data type
        if image.dtype != np.uint8:
            image = cv2.convertScaleAbs(image)
        
        # Optional: Apply slight contrast normalization
        if len(image.shape) == 2:  # Grayscale
            # Normalize to use full dynamic range
            min_val, max_val = image.min(), image.max()
            if max_val > min_val:
                image = ((image - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        
        return image
    
    def quick_preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Quick preprocessing without full quality analysis.
        Useful for batch processing or when speed is prioritized.
        
        Args:
            image: Input image
            
        Returns:
            Quickly processed image
        """
        try:
            # Basic normalization
            processed_image, _ = self._normalize_image_format(image)
            
            # Conservative resize if needed
            height, width = processed_image.shape[:2]
            max_dim = self.enhancement_settings['max_dimension']
            
            if max(height, width) > max_dim:
                scale_factor = max_dim / max(height, width)
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)
                processed_image = cv2.resize(
                    processed_image, (new_width, new_height), 
                    interpolation=cv2.INTER_AREA
                )
            
            return self._final_normalization(processed_image)
            
        except Exception as e:
            self.logger.warning(f"Quick preprocessing failed: {e}")
            return image