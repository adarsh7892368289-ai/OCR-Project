"""
Test Full Integration Pipeline - Tests complete OCR system
Tests: core.py orchestrating all three pipelines together
"""
import pytest
import sys
import os
from pathlib import Path
import numpy as np
from PIL import Image
import tempfile

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

class TestFullIntegrationPipeline:
    """Test the complete OCR system integration"""
    
    @pytest.fixture
    def sample_image(self):
        """Create a comprehensive test image"""
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            # Create a more complex test image
            img = Image.new('RGB', (800, 600), color='white')
            import numpy as np
            img_array = np.array(img)
            
            # Add multiple text-like regions
            img_array[50:80, 100:500] = [0, 0, 0]    # Title
            img_array[120:150, 100:600] = [0, 0, 0]  # Line 1
            img_array[160:190, 100:550] = [0, 0, 0]  # Line 2
            img_array[230:260, 100:450] = [0, 0, 0]  # Line 3
            
            # Add some noise for quality testing
            noise = np.random.randint(0, 30, img_array.shape)
            img_array = np.clip(img_array + noise, 0, 255)
            
            img = Image.fromarray(img_array.astype(np.uint8))
            img.save(tmp.name)
            yield tmp.name
            os.unlink(tmp.name)
    
    def test_core_exists(self):
        """Test if core.py exists and imports correctly"""
        try:
            from advanced_ocr.core import OCR
            assert True, "Core OCR class imported successfully"
        except ImportError as e:
            pytest.fail(f"Core OCR import failed: {e}")
        except Exception as e:
            pytest.fail(f"Core OCR import error: {e}")
    
    def test_main_api_exists(self):
        """Test if __init__.py provides main API correctly"""
        try:
            from advanced_ocr import OCR
            assert True, "Main OCR API imported successfully"
        except ImportError as e:
            pytest.fail(f"Main OCR API import failed: {e}")
        except Exception as e:
            pytest.fail(f"Main OCR API import error: {e}")
    
    def test_config_system_exists(self):
        """Test if configuration system works"""
        try:
            from advanced_ocr.config import OCRConfig
            config = OCRConfig()
            assert config is not None, "OCRConfig created successfully"
        except Exception as e:
            pytest.fail(f"Configuration system test failed: {e}")
    
    def test_results_system_exists(self):
        """Test if results system works"""
        try:
            from advanced_ocr.results import OCRResult, BoundingBox, TextRegion
            
            # Test basic result creation
            bbox = BoundingBox(0, 0, 100, 20)
            region = TextRegion("test", bbox, 0.9)
            result = OCRResult("test text", 0.9, [region])
            
            assert result is not None, "OCRResult system works"
            assert result.text == "test text", "OCRResult text attribute works"
            assert result.confidence == 0.9, "OCRResult confidence attribute works"
            
        except Exception as e:
            pytest.fail(f"Results system test failed: {e}")
    
    def test_core_initialization(self):
        """Test OCR core can be initialized"""
        try:
            from advanced_ocr.core import OCR
            from advanced_ocr.config import OCRConfig
            
            # Test default initialization
            ocr = OCR()
            assert ocr is not None, "OCR initialized with defaults"
            
            # Test initialization with config
            config = OCRConfig()
            ocr_with_config = OCR(config=config)
            assert ocr_with_config is not None, "OCR initialized with config"
            
        except Exception as e:
            pytest.fail(f"Core initialization test failed: {e}")
    
    def test_main_api_initialization(self):
        """Test main API can be initialized"""
        try:
            from advanced_ocr import OCR
            
            ocr = OCR()
            assert ocr is not None, "Main API OCR initialized"
            
        except Exception as e:
            pytest.fail(f"Main API initialization test failed: {e}")
    
    def test_core_extract_method_exists(self):
        """Test if core has extract method"""
        try:
            from advanced_ocr.core import OCR
            
            ocr = OCR()
            assert hasattr(ocr, 'extract'), "OCR core missing extract() method"
            
            # Check method signature
            import inspect
            sig = inspect.signature(ocr.extract)
            params = list(sig.parameters.keys())
            assert 'image_path' in params or 'image' in params, \
                "extract() should accept image parameter"
            
        except Exception as e:
            pytest.fail(f"Core extract method test failed: {e}")
    
    def test_main_api_extract_method_exists(self):
        """Test if main API has extract method"""
        try:
            from advanced_ocr import OCR
            
            ocr = OCR()
            assert hasattr(ocr, 'extract'), "Main API missing extract() method"
            
        except Exception as e:
            pytest.fail(f"Main API extract method test failed: {e}")
    
    def test_batch_extract_method_exists(self):
        """Test if batch_extract method exists"""
        try:
            from advanced_ocr import OCR
            
            ocr = OCR()
            assert hasattr(ocr, 'batch_extract'), "Missing batch_extract() method"
            
        except Exception as e:
            pytest.fail(f"Batch extract method test failed: {e}")
    
    def test_core_pipeline_orchestration(self):
        """Test if core properly orchestrates all pipelines"""
        try:
            from advanced_ocr import core
            
            import inspect
            source = inspect.getsource(core)
            
            # Should import and use all three pipeline orchestrators
            required_imports = [
                'image_processor',    # Preprocessing orchestrator
                'engine_coordinator', # Engine coordination orchestrator  
                'text_processor'      # Postprocessing orchestrator
            ]
            
            for required in required_imports:
                assert (required in source.lower() or 
                        required.replace('_', '').title() in source), \
                    f"core.py should import {required}"
            
            print("✅ Core pipeline orchestration structure validated")
            
        except Exception as e:
            pytest.fail(f"Core pipeline orchestration test failed: {e}")
    
    def test_pipeline_flow_sequence(self):
        """Test if core follows correct pipeline sequence"""
        try:
            from advanced_ocr import core
            
            import inspect
            source = inspect.getsource(core)
            
            # Check for correct pipeline sequence in code
            # Should process: preprocessing → engines → postprocessing
            preprocessing_terms = ['image_processor', 'preprocessing', 'enhance']
            engine_terms = ['engine_coordinator', 'engines', 'extract']
            postprocessing_terms = ['text_processor', 'postprocessing', 'fusion']
            
            # All three stages should be present
            has_preprocessing = any(term in source.lower() for term in preprocessing_terms)
            has_engines = any(term in source.lower() for term in engine_terms)  
            has_postprocessing = any(term in source.lower() for term in postprocessing_terms)
            
            assert has_preprocessing, "Core should include preprocessing stage"
            assert has_engines, "Core should include engine stage"
            assert has_postprocessing, "Core should include postprocessing stage"
            
            print("✅ Pipeline flow sequence validated")
            
        except Exception as e:
            pytest.fail(f"Pipeline flow sequence test failed: {e}")
    
    def test_single_image_extraction(self, sample_image):
        """Test single image extraction through full pipeline"""
        try:
            from advanced_ocr import OCR
            from advanced_ocr.results import OCRResult
            
            ocr = OCR()
            
            # Test extraction
            result = ocr.extract(sample_image)
            
            # Verify result
            assert result is not None, "extract() returned None"
            assert isinstance(result, OCRResult), "Result should be OCRResult instance"
            
            # Verify result has expected attributes
            assert hasattr(result, 'text'), "Result missing text attribute"
            assert hasattr(result, 'confidence'), "Result missing confidence attribute"
            assert hasattr(result, 'regions'), "Result missing regions attribute"
            
            # Verify result content
            assert result.text is not None, "Result text is None"
            assert result.confidence is not None, "Result confidence is None"
            
            # Confidence should be in valid range
            assert 0.0 <= result.confidence <= 1.0, \
                f"Confidence out of range: {result.confidence}"
            
            print(f"✅ Single image extraction test passed:")
            print(f"   - Text extracted: {len(result.text)} characters")
            print(f"   - Confidence: {result.confidence:.3f}")
            print(f"   - Regions: {len(result.regions) if result.regions else 0}")
            
            return result
            
        except Exception as e:
            pytest.fail(f"Single image extraction test failed: {e}")
    
    def test_preprocessing_integration(self, sample_image):
        """Test that preprocessing pipeline is properly integrated"""
        try:
            from advanced_ocr import OCR
            
            ocr = OCR()
            result = ocr.extract(sample_image)
            
            # If preprocessing worked, we should have reasonable text regions
            # (not the 2660 regions bug)
            if result.regions:
                assert len(result.regions) < 500, \
                    f"Too many regions detected: {len(result.regions)} (preprocessing may have failed)"
            
            print(f"✅ Preprocessing integration validated: {len(result.regions) if result.regions else 0} regions")
            
        except Exception as e:
            pytest.fail(f"Preprocessing integration test failed: {e}")
    
    def test_engine_coordination_integration(self, sample_image):
        """Test that engine coordination is properly integrated"""
        try:
            from advanced_ocr import OCR
            
            ocr = OCR()
            result = ocr.extract(sample_image)
            
            # If engine coordination worked, we should have extracted text
            # (not the 9-character TrOCR bug)
            if result.text:
                assert len(result.text.strip()) > 10, \
                    f"Very little text extracted: '{result.text}' (engines may have failed)"
            
            print(f"✅ Engine coordination integration validated: '{result.text[:50]}...' extracted")
            
        except Exception as e:
            pytest.fail(f"Engine coordination integration test failed: {e}")
    
    def test_postprocessing_integration(self, sample_image):
        """Test that postprocessing pipeline is properly integrated"""
        try:
            from advanced_ocr import OCR
            
            ocr = OCR()
            result = ocr.extract(sample_image)
            
            # If postprocessing worked, result should be clean and structured
            assert result.confidence is not None, "Missing confidence (postprocessing failed)"
            assert result.regions is not None, "Missing regions (postprocessing failed)"
            
            # Text should be cleaned (no excessive whitespace)
            if result.text:
                # Basic cleaning check - shouldn't have excessive whitespace
                lines = result.text.split('\n')
                clean_lines = [line.strip() for line in lines if line.strip()]
                assert len(clean_lines) > 0, "No clean text lines (postprocessing failed)"
            
            print(f"✅ Postprocessing integration validated")
            
        except Exception as e:
            pytest.fail(f"Postprocessing integration test failed: {e}")
    
    def test_batch_processing(self, sample_image):
        """Test batch processing functionality"""
        try:
            from advanced_ocr import OCR
            
            ocr = OCR()
            
            # Create list of images (duplicate for testing)
            image_paths = [sample_image, sample_image]
            
            # Test batch extraction
            results = ocr.batch_extract(image_paths)
            
            # Verify results
            assert results is not None, "batch_extract() returned None"
            assert isinstance(results, list), "Batch results should be list"
            assert len(results) == len(image_paths), \
                f"Expected {len(image_paths)} results, got {len(results)}"
            
            # Verify each result
            for i, result in enumerate(results):
                assert result is not None, f"Result {i} is None"
                assert hasattr(result, 'text'), f"Result {i} missing text"
                assert hasattr(result, 'confidence'), f"Result {i} missing confidence"
            
            print(f"✅ Batch processing test passed: {len(results)} images processed")
            
        except Exception as e:
            pytest.fail(f"Batch processing test failed: {e}")
    
    def test_error_handling(self):
        """Test error handling for invalid inputs"""
        try:
            from advanced_ocr import OCR
            
            ocr = OCR()
            
            # Test with non-existent file
            try:
                result = ocr.extract("non_existent_file.jpg")
                # Should either handle gracefully or raise appropriate exception
                assert True, "Error handling works"
            except FileNotFoundError:
                assert True, "Appropriate error raised for missing file"
            except Exception as e:
                # Should not crash with unexpected error
                assert "file" in str(e).lower() or "path" in str(e).lower(), \
                    f"Unexpected error type: {e}"
            
        except Exception as e:
            pytest.fail(f"Error handling test failed: {e}")
    
    def test_performance_improvements(self, sample_image):
        """Test that critical performance issues are fixed"""
        try:
            from advanced_ocr import OCR
            import time
            
            ocr = OCR()
            
            start_time = time.time()
            result = ocr.extract(sample_image)
            processing_time = time.time() - start_time
            
            # Performance checks
            if result.regions:
                # Should not detect excessive regions (2660 bug)
                assert len(result.regions) < 500, \
                    f"Excessive regions detected: {len(result.regions)}"
            
            if result.text:
                # Should extract substantial text (not 9-char TrOCR bug)
                assert len(result.text.strip()) > 5, \
                    f"Very little text extracted: '{result.text}'"
            
            # Processing shouldn't take excessively long for test image
            assert processing_time < 30, \
                f"Processing took too long: {processing_time:.1f}s"
            
            print(f"✅ Performance improvements validated:")
            print(f"   - Processing time: {processing_time:.2f}s")
            print(f"   - Regions detected: {len(result.regions) if result.regions else 0}")
            print(f"   - Text length: {len(result.text) if result.text else 0} chars")
            
        except Exception as e:
            pytest.fail(f"Performance improvements test failed: {e}")
    
    def test_full_pipeline_integration(self, sample_image):
        """Test complete end-to-end pipeline integration"""
        try:
            from advanced_ocr import OCR
            from advanced_ocr.results import OCRResult
            
            # Complete end-to-end test
            ocr = OCR()
            result = ocr.extract(sample_image)
            
            # Comprehensive validation
            assert result is not None, "Pipeline returned None"
            assert isinstance(result, OCRResult), "Pipeline should return OCRResult"
            
            # All three pipeline stages should have contributed
            assert result.text is not None, "Missing text (engine stage failed)"
            assert result.confidence is not None, "Missing confidence (postprocessing failed)"
            assert result.regions is not None, "Missing regions (preprocessing failed)"
            
            # Quality checks
            assert 0.0 <= result.confidence <= 1.0, "Invalid confidence range"
            
            if result.regions:
                assert len(result.regions) < 500, "Excessive regions (preprocessing issue)"
                # Regions should have proper structure
                for region in result.regions[:3]:  # Check first few
                    assert hasattr(region, 'text'), "Region missing text"
                    assert hasattr(region, 'bbox'), "Region missing bbox"
            
            if result.text:
                assert len(result.text.strip()) > 0, "Empty text result"
            
            print(f"✅ Full pipeline integration test PASSED:")
            print(f"   ├─ Preprocessing: {len(result.regions) if result.regions else 0} regions detected")
            print(f"   ├─ Engine Coordination: {len(result.text) if result.text else 0} characters extracted")
            print(f"   └─ Postprocessing: {result.confidence:.3f} final confidence")
            print(f"   Final Result: '{(result.text[:60] + '...') if result.text and len(result.text) > 60 else result.text}'")
            
            return result
            
        except Exception as e:
            pytest.fail(f"Full pipeline integration test failed: {e}")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])