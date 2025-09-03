# tests/test_adaptive_processor.py - Comprehensive Test Suite for Step 4

import unittest
import numpy as np
import cv2
import tempfile
import os
import time
from unittest.mock import Mock, patch, MagicMock

# Import the components
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from preprocessing.adaptive_processor import (
    AdaptivePreprocessor, ProcessingOptions, ProcessingLevel, 
    PipelineStrategy, ProcessingResult
)
from preprocessing.quality_analyzer import ImageType, ImageQuality

class TestAdaptivePreprocessor(unittest.TestCase):
    """Test suite for the Adaptive Preprocessor"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create test images
        self.test_image_color = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        self.test_image_gray = np.random.randint(0, 255, (300, 400), dtype=np.uint8)
        
        # Create document-like test image
        self.document_image = self._create_document_image()
        
        # Initialize preprocessor
        self.preprocessor = AdaptivePreprocessor()
    
    def tearDown(self):
        """Clean up after tests"""
        self.preprocessor.shutdown()
    
    def _create_document_image(self):
        """Create a synthetic document image for testing"""
        img = np.ones((400, 600, 3), dtype=np.uint8) * 255
        
        # Add some text-like rectangles
        cv2.rectangle(img, (50, 50), (550, 80), (0, 0, 0), -1)
        cv2.rectangle(img, (50, 100), (450, 130), (0, 0, 0), -1)
        cv2.rectangle(img, (50, 150), (500, 180), (0, 0, 0), -1)
        
        # Add some noise
        noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
        img = cv2.add(img, noise)
        
        # Add slight skew
        center = (img.shape[1] // 2, img.shape[0] // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, 2.0, 1.0)
        img = cv2.warpAffine(img, rotation_matrix, (img.shape[1], img.shape[0]))
        
        return img
    
    def test_initialization(self):
        """Test preprocessor initialization"""
        self.assertIsNotNone(self.preprocessor.quality_analyzer)
        self.assertIsNotNone(self.preprocessor.image_enhancer)
        self.assertIsNotNone(self.preprocessor.skew_corrector)
        self.assertIsNotNone(self.preprocessor.pipelines)
    
    def test_basic_image_processing(self):
        """Test basic image processing"""
        result = self.preprocessor.process_image(self.test_image_color)
        
        self.assertIsInstance(result, ProcessingResult)
        self.assertTrue(result.success)
        self.assertIsNotNone(result.processed_image)
        self.assertGreater(len(result.processing_steps), 0)
        self.assertGreater(result.processing_time, 0)
    
    def test_processing_options(self):
        """Test different processing options"""
        options = ProcessingOptions(
            processing_level=ProcessingLevel.MINIMAL,
            strategy=PipelineStrategy.SPEED_OPTIMIZED,
            enable_quality_validation=False
        )
        
        result = self.preprocessor.process_image(self.test_image_color, options)
        
        self.assertTrue(result.success)
        self.assertIsNotNone(result.processed_image)
    
    def test_grayscale_processing(self):
        """Test grayscale image processing"""
        result = self.preprocessor.process_image(self.test_image_gray)
        
        self.assertTrue(result.success)
        self.assertIsNotNone(result.processed_image)
    
    def test_document_processing(self):
        """Test document image processing"""
        result = self.preprocessor.process_image(self.document_image)
        
        self.assertTrue(result.success)
        self.assertIsNotNone(result.processed_image)
        self.assertIn("quality_metrics", result.__dict__)
        self.assertIn("metadata", result.__dict__)
    
    def test_batch_processing(self):
        """Test batch processing functionality"""
        images = [self.test_image_color, self.test_image_gray, self.document_image]
        
        results = self.preprocessor.process_batch(images)
        
        self.assertEqual(len(results), len(images))
        for result in results:
            self.assertIsInstance(result, ProcessingResult)
            self.assertTrue(result.success)
    
    def test_pipeline_selection(self):
        """Test pipeline selection logic"""
        # Test speed optimized
        options_speed = ProcessingOptions(strategy=PipelineStrategy.SPEED_OPTIMIZED)
        result_speed = self.preprocessor.process_image(self.document_image, options_speed)
        
        # Test quality optimized  
        options_quality = ProcessingOptions(strategy=PipelineStrategy.QUALITY_OPTIMIZED)
        result_quality = self.preprocessor.process_image(self.document_image, options_quality)
        
        self.assertTrue(result_speed.success)
        self.assertTrue(result_quality.success)
        
        # Quality optimized should generally take longer
        self.assertGreaterEqual(result_quality.processing_time, result_speed.processing_time * 0.8)
    
    def test_custom_pipeline(self):
        """Test custom pipeline functionality"""
        custom_pipeline = {
            "name": "test_custom",
            "description": "Test custom pipeline",
            "steps": [
                {
                    "name": "enhancement",
                    "parameters": {"strategy": "conservative"},
                    "conditions": {}
                }
            ]
        }
        
        self.preprocessor.add_custom_pipeline("test_custom", custom_pipeline)
        
        # Verify pipeline was added
        available = self.preprocessor.get_available_pipelines()
        self.assertIn("test_custom", available)
        
        # Test pipeline info retrieval
        info = self.preprocessor.get_pipeline_info("test_custom")
        self.assertIsNotNone(info)
        self.assertEqual(info["name"], "test_custom")
    
    def test_pipeline_validation(self):
        """Test pipeline validation"""
        # Valid pipeline
        valid_pipeline = {
            "name": "valid_test",
            "steps": [
                {"name": "enhancement", "parameters": {}}
            ]
        }
        
        errors = self.preprocessor.validate_pipeline(valid_pipeline)
        self.assertEqual(len(errors), 0)
        
        # Invalid pipeline (missing name)
        invalid_pipeline = {
            "steps": [
                {"name": "enhancement"}
            ]
        }
        
        errors = self.preprocessor.validate_pipeline(invalid_pipeline)
        self.assertGreater(len(errors), 0)
    
    def test_error_handling(self):
        """Test error handling with invalid inputs"""
        # Test with None image
        result = self.preprocessor.process_image(None)
        self.assertFalse(result.success)
        self.assertGreater(len(result.warnings), 0)
        
        # Test with empty image
        empty_image = np.array([])
        result = self.preprocessor.process_image(empty_image)
        self.assertFalse(result.success)
    
    def test_statistics_tracking(self):
        """Test statistics tracking"""
        initial_stats = self.preprocessor.get_processing_statistics()
        initial_count = initial_stats["total_processed"]
        
        # Process some images
        self.preprocessor.process_image(self.test_image_color)
        self.preprocessor.process_image(self.test_image_gray)
        
        updated_stats = self.preprocessor.get_processing_statistics()
        self.assertEqual(updated_stats["total_processed"], initial_count + 2)
        self.assertGreater(updated_stats["average_processing_time"], 0)
    
    def test_config_export_import(self):
        """Test configuration export and import"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            config_path = f.name
        
        try:
            # Export config
            self.preprocessor.export_config(config_path)
            self.assertTrue(os.path.exists(config_path))
            
            # Create new preprocessor and load config
            new_preprocessor = AdaptivePreprocessor()
            new_preprocessor.load_config_from_file(config_path)
            
            # Test that it works
            result = new_preprocessor.process_image(self.test_image_color)
            self.assertTrue(result.success)
            
            new_preprocessor.shutdown()
            
        finally:
            if os.path.exists(config_path):
                os.unlink(config_path)
    
    def test_memory_estimation(self):
        """Test memory usage estimation"""
        result = self.preprocessor.process_image(self.test_image_color)
        
        self.assertIn("memory_usage", result.performance_stats)
        memory_stats = result.performance_stats["memory_usage"]
        
        self.assertIn("original_image_mb", memory_stats)
        self.assertIn("processed_image_mb", memory_stats)
        self.assertIn("total_mb", memory_stats)
    
    @patch('threading.Thread')
    def test_timeout_handling(self):
        """Test processing timeout handling"""
        options = ProcessingOptions(processing_timeout=0.001)  # Very short timeout
        
        result = self.preprocessor.process_image(self.test_image_color, options)
        
        # Should complete but may have warnings about timeout
        self.assertIsNotNone(result)

class TestProcessingComponents(unittest.TestCase):
    """Test individual processing components"""
    
    def setUp(self):
        self.preprocessor = AdaptivePreprocessor()
        self.test_image = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
    
    def tearDown(self):
        self.preprocessor.shutdown()
    
    def test_skew_correction_step(self):
        """Test skew correction step"""
        result, warnings = self.preprocessor._execute_skew_correction(
            self.test_image, {"quality": "balanced"}
        )
        
        self.assertIsNotNone(result)
        self.assertIsInstance(warnings, list)
    
    def test_enhancement_step(self):
        """Test enhancement step"""
        # Mock quality metrics
        mock_metrics = Mock()
        mock_metrics.image_type = ImageType.PRINTED_TEXT
        
        result, warnings = self.preprocessor._execute_enhancement(
            self.test_image, mock_metrics, {"strategy": "balanced"}
        )
        
        self.assertIsNotNone(result)
        self.assertIsInstance(warnings, list)
    
    def test_noise_reduction_step(self):
        """Test noise reduction step"""
        result, warnings = self.preprocessor._execute_noise_reduction(
            self.test_image, {"method": "bilateral"}
        )
        
        self.assertIsNotNone(result)
        self.assertIsInstance(warnings, list)
    
    def test_contrast_enhancement_step(self):
        """Test contrast enhancement step"""
        result, warnings = self.preprocessor._execute_contrast_enhancement(
            self.test_image, {"method": "clahe"}
        )
        
        self.assertIsNotNone(result)
        self.assertIsInstance(warnings, list)
    
    def test_sharpening_step(self):
        """Test sharpening step"""
        result, warnings = self.preprocessor._execute_sharpening(
            self.test_image, {"strength": 0.5}
        )
        
        self.assertIsNotNone(result)
        self.assertIsInstance(warnings, list)
    
    def test_morphological_cleaning_step(self):
        """Test morphological cleaning step"""
        result, warnings = self.preprocessor._execute_morphological_cleaning(
            self.test_image, {"kernel_size": 2}
        )
        
        self.assertIsNotNone(result)
        self.assertIsInstance(warnings, list)

class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions"""
    
    def test_quick_process_image(self):
        """Test quick processing utility function"""
        from preprocessing.adaptive_processor import quick_process_image
        
        test_image = np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8)
        result = quick_process_image(test_image, ProcessingLevel.LIGHT)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.shape, test_image.shape)
    
    def test_batch_process_images_utility(self):
        """Test batch processing utility function"""
        from preprocessing.adaptive_processor import batch_process_images
        
        images = [
            np.random.randint(0, 255, (200, 300, 3), dtype=np.uint8),
            np.random.randint(0, 255, (250, 350), dtype=np.uint8)
        ]
        
        results = batch_process_images(images, ProcessingLevel.BALANCED)
        
        self.assertEqual(len(results), len(images))
        for result in results:
            self.assertIsNotNone(result)

if __name__ == '__main__':
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(unittest.makeSuite(TestAdaptivePreprocessor))
    suite.addTest(unittest.makeSuite(TestProcessingComponents))
    suite.addTest(unittest.makeSuite(TestUtilityFunctions))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print(f"{'='*60}")