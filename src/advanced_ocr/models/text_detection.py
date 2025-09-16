# src/advanced_ocr/models/text_detection.py
"""
Advanced OCR Text Detection Models

This module provides text detection model implementations for the advanced OCR
system. It includes various text detection algorithms optimized for different
types of documents and text layouts.

The module focuses on:
- Scene text detection in natural images
- Document layout analysis and text region identification
- Multi-scale text detection capabilities
- Robust detection of text in various orientations
- Integration with CRAFT and other detection algorithms
- Region proposal generation for OCR engines

Classes:
    CRAFTTextDetector: CRAFT-based text detection implementation
    EASTTextDetector: EAST-based text detection implementation
    TextDetectionPipeline: Unified text detection pipeline

Functions:
    detect_text_regions: Main text detection function
    filter_text_regions: Region filtering and validation
    merge_overlapping_regions: Region merging utilities

Example:
    >>> detector = CRAFTTextDetector()
    >>> regions = detector.detect(image)
    >>> print(f"Detected {len(regions)} text regions")

"""

import cv2
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
import logging

from ..results import TextRegion, BoundingBox
from ..utils.image_utils import ImageProcessor
from ..utils.model_utils import ModelLoader, cached_model_load

logger = logging.getLogger(__name__)


class BaseTextDetector:
    """
    Base class for text detection models.

    Provides common functionality and interface for all text detection
    implementations in the advanced OCR system.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the text detector with configuration."""
        self.config = config or {}
        self.image_processor = ImageProcessor()
        self.model_loader = ModelLoader()

        # Default detection parameters
        self.confidence_threshold = self.config.get('confidence_threshold', 0.5)
        self.min_region_size = self.config.get('min_region_size', 10)
        self.max_region_size = self.config.get('max_region_size', 10000)

        logger.info(f"Initialized {self.__class__.__name__} with confidence threshold: {self.confidence_threshold}")

    def detect(self, image: np.ndarray) -> List[TextRegion]:
        """
        Detect text regions in the input image.

        Args:
            image: Input image as numpy array

        Returns:
            List of detected text regions
        """
        raise NotImplementedError("Subclasses must implement detect method")

    def _validate_region(self, region: TextRegion) -> bool:
        """Validate a text region based on size and other criteria."""
        bbox = region.bounding_box
        area = (bbox.width) * (bbox.height)

        # Check size constraints
        if area < self.min_region_size or area > self.max_region_size:
            return False

        # Check aspect ratio (avoid extremely thin or wide regions)
        aspect_ratio = max(bbox.width, bbox.height) / max(min(bbox.width, bbox.height), 1)
        if aspect_ratio > 20:
            return False

        return True

    def _filter_regions(self, regions: List[TextRegion]) -> List[TextRegion]:
        """Filter and validate detected regions."""
        valid_regions = []

        for region in regions:
            if self._validate_region(region):
                valid_regions.append(region)

        logger.debug(f"Filtered {len(regions)} regions to {len(valid_regions)} valid regions")
        return valid_regions


class CRAFTTextDetector(BaseTextDetector):
    """
    CRAFT (Character Region Awareness For Text detection) implementation.

    CRAFT is a scene text detection algorithm that detects individual characters
    and links them to form text regions. It provides high accuracy for various
    text sizes and orientations.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        # CRAFT-specific parameters
        self.model_path = self.config.get('model_path', 'models/text_detection/craft_mlt_25k.pth')
        self.refine_net = self.config.get('refine_net', True)
        self.cuda = self.config.get('cuda', False)

        # Model components
        self._craft_net = None
        self._refine_net = None

    def detect(self, image: np.ndarray) -> List[TextRegion]:
        """
        Detect text regions using CRAFT algorithm.

        Args:
            image: Input image as numpy array

        Returns:
            List of detected text regions
        """
        try:
            # Ensure model is loaded
            if self._craft_net is None:
                self._load_models()

            # Preprocess image for CRAFT
            processed_image = self._preprocess_for_craft(image)

            # Run CRAFT detection
            score_text, score_link = self._craft_net(processed_image)

            # Post-process results
            text_regions = self._postprocess_craft_results(score_text, score_link, image.shape)

            # Filter and validate regions
            valid_regions = self._filter_regions(text_regions)

            logger.info(f"CRAFT detected {len(valid_regions)} text regions")
            return valid_regions

        except Exception as e:
            logger.error(f"CRAFT detection failed: {e}")
            return []

    def _load_models(self):
        """Load CRAFT model and optional refinement network."""
        try:
            # Import CRAFT implementation
            from .craft_utils import CRAFTModel

            self._craft_net = CRAFTModel(self.model_path, self.cuda)

            if self.refine_net:
                # Load refinement network if available
                refine_path = self.model_path.replace('.pth', '_refiner.pth')
                try:
                    self._refine_net = CRAFTModel(refine_path, self.cuda)
                    logger.info("CRAFT refinement network loaded")
                except FileNotFoundError:
                    logger.warning("CRAFT refinement network not found, using base model only")
                    self._refine_net = None

            logger.info("CRAFT models loaded successfully")

        except ImportError as e:
            raise ImportError(f"CRAFT dependencies not available: {e}")

    def _preprocess_for_craft(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for CRAFT input."""
        # CRAFT expects RGB images with specific preprocessing
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif len(image.shape) == 3 and image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize if too large (CRAFT has memory constraints)
        max_size = 2560
        height, width = image.shape[:2]
        if max(height, width) > max_size:
            scale = max_size / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = cv2.resize(image, (new_width, new_height))

        return image

    def _postprocess_craft_results(self, score_text: np.ndarray, score_link: np.ndarray,
                                 original_shape: Tuple[int, ...]) -> List[TextRegion]:
        """Post-process CRAFT detection results into TextRegion objects."""
        # This is a simplified post-processing - actual implementation would be more complex
        text_regions = []

        # Placeholder for actual CRAFT post-processing logic
        # In a real implementation, this would involve:
        # 1. Thresholding score_text and score_link
        # 2. Finding connected components
        # 3. Grouping characters into words/lines
        # 4. Creating bounding boxes

        logger.debug("CRAFT post-processing completed")
        return text_regions


class EASTTextDetector(BaseTextDetector):
    """
    EAST (Efficient and Accurate Scene Text) detector implementation.

    EAST is a fast and accurate scene text detection algorithm that directly
    predicts text regions and their geometries in a single neural network.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)

        # EAST-specific parameters
        self.model_path = self.config.get('model_path', 'models/text_detection/east.pb')
        self.input_size = self.config.get('input_size', 320)

        # Model components
        self._east_net = None

    def detect(self, image: np.ndarray) -> List[TextRegion]:
        """
        Detect text regions using EAST algorithm.

        Args:
            image: Input image as numpy array

        Returns:
            List of detected text regions
        """
        try:
            # Ensure model is loaded
            if self._east_net is None:
                self._load_model()

            # Preprocess image for EAST
            processed_image, ratio = self._preprocess_for_east(image)

            # Run EAST detection
            geometry, scores = self._east_net.process(processed_image)

            # Post-process results
            text_regions = self._postprocess_east_results(geometry, scores, ratio, image.shape)

            # Filter and validate regions
            valid_regions = self._filter_regions(text_regions)

            logger.info(f"EAST detected {len(valid_regions)} text regions")
            return valid_regions

        except Exception as e:
            logger.error(f"EAST detection failed: {e}")
            return []

    def _load_model(self):
        """Load EAST model."""
        try:
            self._east_net = cv2.dnn.readNet(self.model_path)
            logger.info("EAST model loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load EAST model: {e}")

    def _preprocess_for_east(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Preprocess image for EAST input."""
        height, width = image.shape[:2]

        # Calculate ratio for resizing
        ratio = self.input_size / max(height, width)
        new_width = int(width * ratio)
        new_height = int(height * ratio)

        # Resize image
        resized_image = cv2.resize(image, (new_width, new_height))

        # Create blob for EAST
        blob = cv2.dnn.blobFromImage(resized_image, 1.0, (new_width, new_height),
                                   (123.68, 116.78, 103.94), swapRB=True, crop=False)

        return blob, ratio

    def _postprocess_east_results(self, geometry: np.ndarray, scores: np.ndarray,
                                ratio: float, original_shape: Tuple[int, ...]) -> List[TextRegion]:
        """Post-process EAST detection results into TextRegion objects."""
        text_regions = []

        # Placeholder for actual EAST post-processing logic
        # In a real implementation, this would involve:
        # 1. Decoding geometry predictions
        # 2. Applying confidence thresholding
        # 3. Extracting rotated rectangles
        # 4. Converting to axis-aligned bounding boxes

        logger.debug("EAST post-processing completed")
        return text_regions


class TextDetectionPipeline:
    """
    Unified text detection pipeline that combines multiple detection algorithms.

    This pipeline can use multiple detectors and combine their results for
    improved accuracy and robustness.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

        # Initialize available detectors
        self.detectors = {}
        self._initialize_detectors()

        # Pipeline configuration
        self.combine_results = self.config.get('combine_results', True)
        self.voting_threshold = self.config.get('voting_threshold', 0.5)

    def _initialize_detectors(self):
        """Initialize available text detection models."""
        detector_configs = self.config.get('detectors', {})

        # Initialize CRAFT detector
        if detector_configs.get('craft', {}).get('enabled', True):
            try:
                self.detectors['craft'] = CRAFTTextDetector(detector_configs.get('craft', {}))
                logger.info("CRAFT detector initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize CRAFT detector: {e}")

        # Initialize EAST detector
        if detector_configs.get('east', {}).get('enabled', False):
            try:
                self.detectors['east'] = EASTTextDetector(detector_configs.get('east', {}))
                logger.info("EAST detector initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize EAST detector: {e}")

    def detect(self, image: np.ndarray) -> List[TextRegion]:
        """
        Detect text regions using the configured detection pipeline.

        Args:
            image: Input image as numpy array

        Returns:
            List of detected text regions
        """
        if not self.detectors:
            logger.warning("No text detectors available")
            return []

        all_regions = []

        # Run all available detectors
        for name, detector in self.detectors.items():
            try:
                regions = detector.detect(image)
                all_regions.extend(regions)
                logger.debug(f"{name} detector found {len(regions)} regions")
            except Exception as e:
                logger.error(f"{name} detector failed: {e}")
                continue

        if not all_regions:
            return []

        # Combine results if multiple detectors were used
        if self.combine_results and len(self.detectors) > 1:
            combined_regions = self._combine_detection_results(all_regions)
            logger.info(f"Combined detection results: {len(combined_regions)} regions")
            return combined_regions
        else:
            return all_regions

    def _combine_detection_results(self, all_regions: List[TextRegion]) -> List[TextRegion]:
        """Combine results from multiple detectors using voting/consensus."""
        # Placeholder for result combination logic
        # In a real implementation, this would involve:
        # 1. Non-maximum suppression
        # 2. Confidence-based voting
        # 3. Region merging
        # 4. Duplicate removal

        logger.debug("Detection result combination completed")
        return all_regions


# Convenience functions for easy usage
def detect_text_regions(image: np.ndarray, method: str = 'craft',
                       config: Optional[Dict[str, Any]] = None) -> List[TextRegion]:
    """
    Convenience function for text detection.

    Args:
        image: Input image as numpy array
        method: Detection method ('craft', 'east', or 'pipeline')
        config: Detection configuration

    Returns:
        List of detected text regions
    """
    config = config or {}

    if method == 'craft':
        detector = CRAFTTextDetector(config)
    elif method == 'east':
        detector = EASTTextDetector(config)
    elif method == 'pipeline':
        detector = TextDetectionPipeline(config)
    else:
        raise ValueError(f"Unknown detection method: {method}")

    return detector.detect(image)


def filter_text_regions(regions: List[TextRegion], min_confidence: float = 0.5,
                       min_area: int = 100, max_area: int = 10000) -> List[TextRegion]:
    """
    Filter text regions based on various criteria.

    Args:
        regions: List of text regions to filter
        min_confidence: Minimum confidence threshold
        min_area: Minimum region area
        max_area: Maximum region area

    Returns:
        Filtered list of text regions
    """
    filtered_regions = []

    for region in regions:
        # Check confidence
        if region.confidence < min_confidence:
            continue

        # Check area
        bbox = region.bounding_box
        area = bbox.width * bbox.height
        if area < min_area or area > max_area:
            continue

        filtered_regions.append(region)

    logger.debug(f"Filtered {len(regions)} regions to {len(filtered_regions)}")
    return filtered_regions
