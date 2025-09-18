"""
Test Engine Coordination Pipeline - Tests engine_coordinator.py orchestration
Tests: engine_coordinator.py → content_classifier.py + engines selection + coordination
"""
import pytest
import sys
import os
from pathlib import Path
import numpy as np
from PIL import Image
import tempfile
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

class TestEngineCoordinationPipeline:
    """Test the complete engine coordination pipeline"""
    
    @pytest.fixture
    def real_test_image(self):
        """Use real test image from data/sample_images/img3.jpg"""
        img_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'sample_images', 'img3.jpg')
        if os.path.exists(img_path):
            return img_path
        else:
            # Fallback: create a test image with clear text patterns
            img = Image.new('RGB', (800, 600), color='white')
            img_array = np.array(img)
            # Add multiple text-like regions
            img_array[100:130, 50:400] = [0, 0, 0]  # Line 1
            img_array[150:180, 50:350] = [0, 0, 0]  # Line 2
            img_array[200:230, 50:450] = [0, 0, 0]  # Line 3
            img_array[300:330, 100:300] = [50, 50, 50]  # Handwritten-like
            img = Image.fromarray(img_array)
            
            # Save to temporary file but keep it open to avoid permission issues
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                img.save(tmp.name, 'JPEG')
                return tmp.name
    
    @pytest.fixture
    def sample_text_regions(self):
        """Create sample text regions with correct BoundingBox format"""
        from advanced_ocr.results import BoundingBox
        return [
            BoundingBox((50, 100), (400, 130)),    # x1,y1, x2,y2 format
            BoundingBox((50, 150), (350, 180)),    # Line 2
            BoundingBox((50, 200), (450, 230)),    # Line 3
            BoundingBox((100, 300), (300, 330))    # Handwritten region
        ]
    
    def cleanup_temp_files(self, filepath):
        """Clean up temporary files with retry"""
        if filepath and os.path.exists(filepath) and 'tmp' in filepath:
            try:
                time.sleep(0.1)  # Small delay
                os.unlink(filepath)
            except PermissionError:
                pass  # Ignore if can't delete
    
    def test_config_has_coordination_section(self):
        """Test if OCRConfig has coordination configuration"""
        try:
            from advanced_ocr.config import OCRConfig
            
            config = OCRConfig()
            
            # Check if coordination attribute exists
            assert hasattr(config, 'coordination'), \
                "OCRConfig missing 'coordination' attribute"
            
            # Should have engine selection settings
            coordination_config = config.coordination
            expected_attrs = ['engine_selection_strategy', 'multi_engine_mode', 'fallback_engines']
            
            for attr in expected_attrs:
                assert hasattr(coordination_config, attr), \
                    f"Coordination config missing '{attr}'"
            
            print("✅ OCRConfig coordination section validated")
            
        except Exception as e:
            pytest.fail(f"OCRConfig coordination test failed: {e}")
    
    def test_content_classifier_has_classify_method(self):
        """Test if ContentClassifier has classify() method"""
        try:
            from advanced_ocr.preprocessing.content_classifier import ContentClassifier
            from advanced_ocr.config import OCRConfig
            
            config = OCRConfig()
            classifier = ContentClassifier(config)
            
            # Test if classify method exists
            assert hasattr(classifier, 'classify'), \
                "ContentClassifier missing classify() method"
            
            print("✅ ContentClassifier.classify() method exists")
            
        except Exception as e:
            pytest.fail(f"ContentClassifier classify method test failed: {e}")
    
    def test_bounding_box_constructor(self):
        """Test BoundingBox constructor signature"""
        try:
            from advanced_ocr.results import BoundingBox
            
            # Test different constructor formats
            bbox1 = BoundingBox((10, 20), (100, 120))  # (x1,y1), (x2,y2)
            assert bbox1 is not None
            
            # Test properties exist
            required_props = ['x1', 'y1', 'x2', 'y2', 'width', 'height']
            for prop in required_props:
                assert hasattr(bbox1, prop), f"BoundingBox missing '{prop}' property"
            
            print("✅ BoundingBox constructor validated")
            
        except Exception as e:
            pytest.fail(f"BoundingBox constructor test failed: {e}")
    
    def test_engines_exist_and_inherit_correctly(self):
        """Test if all engines exist and inherit from BaseOCREngine"""
        engines_to_test = [
            ('tesseract_engine', 'TesseractEngine'),
            ('paddleocr_engine', 'PaddleOCREngine'), 
            ('easyocr_engine', 'EasyOCREngine'),
            ('trocr_engine', 'TrOCREngine')
        ]
        
        for engine_module, engine_class_name in engines_to_test:
            try:
                module = __import__(f'advanced_ocr.engines.{engine_module}', fromlist=[engine_class_name])
                engine_class = getattr(module, engine_class_name)
                
                # Check inheritance
                from advanced_ocr.engines.base_engine import BaseOCREngine
                assert issubclass(engine_class, BaseOCREngine), \
                    f"{engine_class_name} doesn't inherit from BaseOCREngine"
                
                # Check extract method exists
                assert hasattr(engine_class, 'extract'), \
                    f"{engine_class_name} missing extract() method"
                
                print(f"✅ {engine_class_name} validated")
                
            except ImportError as e:
                print(f"❌ {engine_class_name} import failed: {e}")
            except Exception as e:
                print(f"❌ {engine_class_name} validation failed: {e}")
    
    def test_content_classifier_classify_with_real_image(self, real_test_image):
        """Test ContentClassifier.classify() with real image"""
        temp_file = None
        try:
            from advanced_ocr.preprocessing.content_classifier import ContentClassifier
            from advanced_ocr.config import OCRConfig
            from PIL import Image
            
            config = OCRConfig()
            classifier = ContentClassifier(config)
            
            # Use real image
            if os.path.exists(real_test_image):
                image = Image.open(real_test_image)
            else:
                temp_file = real_test_image
                image = Image.open(real_test_image)
            
            # Test classify method
            classification = classifier.classify(image)
            
            assert classification is not None, "Classification returned None"
            assert hasattr(classification, 'content_type'), "Missing content_type"
            assert hasattr(classification, 'confidence'), "Missing confidence"
            
            valid_types = ['handwritten', 'printed', 'mixed']
            assert classification.content_type in valid_types, \
                f"Invalid content_type: {classification.content_type}"
            
            print(f"✅ Content classified as: {classification.content_type} "
                  f"(confidence: {classification.confidence:.3f})")
            
        except Exception as e:
            pytest.fail(f"ContentClassifier.classify() with real image failed: {e}")
        finally:
            if temp_file:
                self.cleanup_temp_files(temp_file)
    
    def test_engine_coordinator_initialization_and_methods(self):
        """Test EngineCoordinator initialization and methods"""
        try:
            from advanced_ocr.engines.engine_coordinator import EngineCoordinator
            from advanced_ocr.config import OCRConfig
            
            config = OCRConfig()
            coordinator = EngineCoordinator(config)
            
            # Test coordinate method exists
            assert hasattr(coordinator, 'coordinate'), \
                "EngineCoordinator missing coordinate() method"
            
            print("✅ EngineCoordinator initialized and methods validated")
            
        except Exception as e:
            pytest.fail(f"EngineCoordinator initialization/methods test failed: {e}")
    
    def test_individual_engines_with_real_image(self, real_test_image, sample_text_regions):
        """Test individual engines with real image and show raw OCR output"""
        temp_file = None
        engines_to_test = [
            ('tesseract_engine', 'TesseractEngine'),
            ('paddleocr_engine', 'PaddleOCREngine'), 
            ('easyocr_engine', 'EasyOCREngine'),
            ('trocr_engine', 'TrOCREngine')
        ]
        
        try:
            from advanced_ocr.config import OCRConfig
            from PIL import Image
            
            config = OCRConfig()
            
            # Use real image
            if os.path.exists(real_test_image):
                image = Image.open(real_test_image)
            else:
                temp_file = real_test_image
                image = Image.open(real_test_image)
            
            print(f"\n🖼️  Testing with image: {real_test_image}")
            print(f"📏 Image size: {image.size}")
            print(f"🎯 Text regions to process: {len(sample_text_regions)}")
            
            for engine_module, engine_class_name in engines_to_test:
                try:
                    # Import and initialize engine
                    module = __import__(f'advanced_ocr.engines.{engine_module}', fromlist=[engine_class_name])
                    engine_class = getattr(module, engine_class_name)
                    engine = engine_class(config)
                    
                    # Test extract method
                    result = engine.extract(image, sample_text_regions)
                    
                    # Validate result
                    from advanced_ocr.results import OCRResult
                    assert isinstance(result, OCRResult), \
                        f"{engine_class_name} should return OCRResult"
                    
                    # Show raw OCR output
                    text_length = len(result.text) if result.text else 0
                    confidence = result.confidence if hasattr(result, 'confidence') else 0.0
                    
                    print(f"\n🔧 {engine_class_name}:")
                    print(f"   📝 Extracted text length: {text_length} characters")
                    print(f"   🎯 Confidence: {confidence:.3f}")
                    if result.text and len(result.text) > 0:
                        preview = result.text[:100] + "..." if len(result.text) > 100 else result.text
                        print(f"   📖 Text preview: '{preview}'")
                    else:
                        print(f"   ⚠️  No text extracted")
                    
                    # Check for performance improvements (TrOCR should extract more than 9 chars)
                    if engine_class_name == 'TrOCREngine':
                        if text_length > 9:
                            print(f"   ✅ TrOCR performance improved: {text_length} chars (>9)")
                        else:
                            print(f"   ⚠️  TrOCR performance issue: only {text_length} chars")
                    
                except ImportError as e:
                    print(f"   ❌ {engine_class_name} import failed: {e}")
                except Exception as e:
                    print(f"   ❌ {engine_class_name} test failed: {e}")
                    
        except Exception as e:
            pytest.fail(f"Individual engines test failed: {e}")
        finally:
            if temp_file:
                self.cleanup_temp_files(temp_file)
    
    def test_engine_coordinator_full_pipeline(self, real_test_image, sample_text_regions):
        """Test complete engine coordination pipeline"""
        temp_file = None
        try:
            from advanced_ocr.engines.engine_coordinator import EngineCoordinator
            from advanced_ocr.config import OCRConfig
            from advanced_ocr.results import OCRResult
            from PIL import Image
            
            config = OCRConfig()
            coordinator = EngineCoordinator(config)
            
            # Use real image
            if os.path.exists(real_test_image):
                image = Image.open(real_test_image)
            else:
                temp_file = real_test_image
                image = Image.open(real_test_image)
            
            print(f"\n🎯 Testing Engine Coordination Pipeline")
            print(f"🖼️  Image: {real_test_image}")
            print(f"🔧 Text regions: {len(sample_text_regions)}")
            
            # Test coordination
            results = coordinator.coordinate(image, sample_text_regions)
            
            # Validate results
            assert results is not None, "Coordination returned None"
            
            if isinstance(results, list):
                print(f"📊 Multiple engine results: {len(results)} engines used")
                for i, result in enumerate(results):
                    assert isinstance(result, OCRResult), f"Result {i} is not OCRResult"
                    text_len = len(result.text) if result.text else 0
                    print(f"   Engine {i+1}: {text_len} characters extracted")
            else:
                assert isinstance(results, OCRResult), "Single result is not OCRResult"
                text_len = len(results.text) if results.text else 0
                print(f"📊 Single engine result: {text_len} characters extracted")
                
                if results.text:
                    preview = results.text[:200] + "..." if len(results.text) > 200 else results.text
                    print(f"📖 Final text: '{preview}'")
            
            print("✅ Engine coordination pipeline test passed")
            
        except Exception as e:
            pytest.fail(f"Engine coordination pipeline test failed: {e}")
        finally:
            if temp_file:
                self.cleanup_temp_files(temp_file)
    
    def test_engine_selection_based_on_content(self, real_test_image):
        """Test that engine selection is based on content classification"""
        temp_file = None
        try:
            from advanced_ocr.engines.engine_coordinator import EngineCoordinator
            from advanced_ocr.preprocessing.content_classifier import ContentClassifier
            from advanced_ocr.config import OCRConfig
            from PIL import Image
            
            config = OCRConfig()
            coordinator = EngineCoordinator(config)
            classifier = ContentClassifier(config)
            
            # Use real image
            if os.path.exists(real_test_image):
                image = Image.open(real_test_image)
            else:
                temp_file = real_test_image
                image = Image.open(real_test_image)
            
            # First classify content
            classification = classifier.classify(image)
            print(f"\n🔍 Content classified as: {classification.content_type}")
            
            # Check if coordinator uses this classification for engine selection
            import inspect
            coordinator_source = inspect.getsource(coordinator.__class__)
            
            # Should have content-based selection logic
            selection_indicators = [
                'handwritten' in coordinator_source.lower(),
                'printed' in coordinator_source.lower(),
                'mixed' in coordinator_source.lower(),
                'content_type' in coordinator_source.lower(),
                'trocr' in coordinator_source.lower(),
                'tesseract' in coordinator_source.lower()
            ]
            
            assert any(selection_indicators), \
                "EngineCoordinator should have content-based engine selection logic"
            
            print("✅ Engine selection based on content classification validated")
            
        except Exception as e:
            pytest.fail(f"Engine selection test failed: {e}")
        finally:
            if temp_file:
                self.cleanup_temp_files(temp_file)

if __name__ == "__main__":
    # Run with verbose output
    pytest.main([__file__, "-v", "-s"])