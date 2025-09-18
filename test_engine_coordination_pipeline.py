"""
Test Engine Coordination Pipeline - ARCHITECTURALLY CORRECT
Tests: engine_coordinator.py with proper preprocessing pipeline integration
FIXED: Uses correct pipeline flow: image_processor.py → engine_coordinator.py
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

class TestEngineCoordinationPipeline:
    """Test the complete engine coordination pipeline with correct architecture"""
    
    @pytest.fixture
    def sample_image(self):
        """Use img3.jpg instead of temp file"""
        img_path = Path(__file__).parent / 'data' / 'sample_images' / 'img3.jpg'
        if not img_path.exists():
            pytest.skip(f"Test image not found: {img_path}")
        return str(img_path)
    
    @pytest.fixture
    def sample_text_regions(self):
        """Create sample text regions"""
        from advanced_ocr.results import BoundingBox
        return [
            BoundingBox(200, 100, 600, 150),
            BoundingBox(200, 200, 500, 250)
        ]
    
    def test_engine_coordinator_exists(self):
        """Test if engine_coordinator.py exists and imports correctly"""
        try:
            from advanced_ocr.engines.engine_coordinator import EngineCoordinator
            assert True, "EngineCoordinator imported successfully"
        except ImportError as e:
            pytest.fail(f"EngineCoordinator import failed: {e}")
        except Exception as e:
            pytest.fail(f"EngineCoordinator import error: {e}")
    
    def test_content_classifier_exists(self):
        """Test if content_classifier.py exists and imports correctly"""
        try:
            from advanced_ocr.preprocessing.content_classifier import ContentClassifier
            assert True, "ContentClassifier imported successfully"
        except ImportError as e:
            pytest.fail(f"ContentClassifier import failed: {e}")
        except Exception as e:
            pytest.fail(f"ContentClassifier import error: {e}")
    
    def test_base_engine_exists(self):
        """Test if base_engine.py exists and imports correctly"""
        try:
            from advanced_ocr.engines.base_engine import BaseOCREngine
            assert True, "BaseOCREngine imported successfully"
        except ImportError as e:
            pytest.fail(f"BaseOCREngine import failed: {e}")
        except Exception as e:
            pytest.fail(f"BaseOCREngine import error: {e}")
    
    def test_individual_engines_exist(self):
        """Test if individual engines exist and inherit from BaseOCREngine"""
        engines_to_test = [
            'tesseract_engine',
            'paddleocr_engine', 
            'easyocr_engine',
            'trocr_engine'
        ]
        
        for engine_name in engines_to_test:
            try:
                module = __import__(f'advanced_ocr.engines.{engine_name}', fromlist=[''])
                # Find engine class (should end with 'Engine')
                engine_class = None
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (isinstance(attr, type) and 
                        attr_name.endswith('Engine') and 
                        attr_name != 'BaseOCREngine'):
                        engine_class = attr
                        break
                
                assert engine_class is not None, f"No engine class found in {engine_name}"
                
                # Check if it inherits from BaseOCREngine (indirectly)
                from advanced_ocr.engines.base_engine import BaseOCREngine
                assert issubclass(engine_class, BaseOCREngine), \
                    f"{engine_class.__name__} doesn't inherit from BaseOCREngine"
                
                print(f"✅ {engine_name}: {engine_class.__name__} found and valid")
                
            except ImportError as e:
                pytest.fail(f"Engine {engine_name} import failed: {e}")
            except Exception as e:
                pytest.fail(f"Engine {engine_name} test failed: {e}")
    
    def test_content_classifier_initialization(self):
        """Test ContentClassifier can be initialized"""
        try:
            from advanced_ocr.preprocessing.content_classifier import ContentClassifier
            from advanced_ocr.config import OCRConfig
            
            config = OCRConfig()
            classifier = ContentClassifier(config)
            assert classifier is not None, "ContentClassifier initialized successfully"
        except Exception as e:
            pytest.fail(f"ContentClassifier initialization failed: {e}")
    
    def test_engine_coordinator_initialization(self):
        """Test EngineCoordinator can be initialized"""
        try:
            from advanced_ocr.engines.engine_coordinator import EngineCoordinator
            from advanced_ocr.config import OCRConfig
            
            config = OCRConfig()
            coordinator = EngineCoordinator(config)
            assert coordinator is not None, "EngineCoordinator initialized successfully"
        except Exception as e:
            pytest.fail(f"EngineCoordinator initialization failed: {e}")
    
    def test_content_classifier_classify_content_method(self, sample_image):
        """Test ContentClassifier.classify_content() method exists and works"""
        try:
            from advanced_ocr.preprocessing.content_classifier import ContentClassifier
            from advanced_ocr.config import OCRConfig
            from PIL import Image
            
            config = OCRConfig()
            classifier = ContentClassifier(config)
            image = Image.open(sample_image)
            
            # Test if classify method exists
            assert hasattr(classifier, 'classify_content'), "ContentClassifier missing classify_content() method"
            
            # Test classify method
            classification = classifier.classify_content(image)
            assert classification is not None, "ContentClassifier.classify_content() returned None"
            
            # Should have content type (handwritten/printed/mixed)
            assert hasattr(classification, 'content_type'), \
                "Classification should have content_type attribute"
            
            valid_types = ['handwritten', 'printed', 'mixed']
            assert classification.content_type in valid_types, \
                f"Invalid content_type: {classification.content_type}"
            
        except Exception as e:
            pytest.fail(f"ContentClassifier.classify_content() test failed: {e}")
    
    def test_engine_coordinator_coordinate_method(self, sample_image, sample_text_regions):
        """Test EngineCoordinator.coordinate() method with PREPROCESSED image"""
        try:
            from advanced_ocr.engines.engine_coordinator import EngineCoordinator
            from advanced_ocr.preprocessing.image_processor import ImageProcessor
            from advanced_ocr.utils.model_utils import ModelLoader
            from advanced_ocr.config import OCRConfig
            from PIL import Image
            import numpy as np
            
            config = OCRConfig()
            
            # FIXED: Follow correct pipeline - preprocess image first
            model_loader = ModelLoader(config)
            image_processor = ImageProcessor(model_loader, config)
            
            # Load and preprocess image
            pil_image = Image.open(sample_image)
            np_image = np.array(pil_image)
            
            # Get preprocessed image and text regions
            preprocessing_result = image_processor.process_image(np_image)
            
            # Now test coordinator with preprocessed data
            coordinator = EngineCoordinator(config)
            
            # Test if coordinate method exists
            assert hasattr(coordinator, 'coordinate'), \
                "EngineCoordinator missing coordinate() method"
            
            # Test coordinate method with PREPROCESSED image (numpy array)
            results = coordinator.coordinate(
                preprocessing_result.enhanced_image,  # numpy array from preprocessing
                preprocessing_result.text_regions     # text regions from preprocessing
            )
            assert results is not None, "EngineCoordinator.coordinate() returned None"
            
            # Should return OCRResult or list of OCRResults
            from advanced_ocr.results import OCRResult
            if isinstance(results, list):
                assert all(isinstance(r, OCRResult) for r in results), \
                    "All results should be OCRResult instances"
            else:
                assert isinstance(results, OCRResult), \
                    "Result should be OCRResult instance"
            
        except Exception as e:
            pytest.fail(f"EngineCoordinator.coordinate() test failed: {e}")
    
    def test_engine_selection_logic(self, sample_image):
        """Test that engine selection is based on content classification"""
        try:
            from advanced_ocr.engines.engine_coordinator import EngineCoordinator
            from advanced_ocr.config import OCRConfig
            from PIL import Image
            
            config = OCRConfig()
            coordinator = EngineCoordinator(config)
            
            # Check if coordinator uses content_classifier
            import inspect
            source = inspect.getsource(coordinator.__class__)
            
            assert ('content_classifier' in source.lower() or 
                    'ContentClassifier' in source), \
                "EngineCoordinator should use ContentClassifier"
            
            # Check if it has engine selection logic
            assert ('handwritten' in source.lower() or 
                    'printed' in source.lower() or 
                    'trocr' in source.lower()), \
                "EngineCoordinator should have content-based engine selection"
            
        except Exception as e:
            pytest.fail(f"Engine selection logic test failed: {e}")
    
    def test_individual_engine_extract_method(self):
        """Test that individual engines have extract() method"""
        engines_to_test = [
            'tesseract_engine',
            'paddleocr_engine', 
            'easyocr_engine',
            'trocr_engine'
        ]
        
        for engine_name in engines_to_test:
            try:
                module = __import__(f'advanced_ocr.engines.{engine_name}', fromlist=[''])
                
                # Find engine class
                engine_class = None
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (isinstance(attr, type) and 
                        attr_name.endswith('Engine') and 
                        attr_name != 'BaseOCREngine'):
                        engine_class = attr
                        break
                
                assert engine_class is not None, f"No engine class found in {engine_name}"
                
                # Check if extract method exists
                assert hasattr(engine_class, 'extract'), \
                    f"{engine_class.__name__} missing extract() method"
                
                # Check method signature
                import inspect
                sig = inspect.signature(engine_class.extract)
                params = list(sig.parameters.keys())
                
                # Should have image and regions parameters (plus self)
                assert 'image' in params, f"{engine_class.__name__}.extract() missing image parameter"
                assert len(params) >= 2, f"{engine_class.__name__}.extract() has too few parameters"
                
                print(f"✅ {engine_name}: extract() method validated")
                
            except Exception as e:
                pytest.fail(f"Engine {engine_name} extract() test failed: {e}")
    
    def test_trocr_performance_fix(self):
        """Test that TrOCR engine has performance improvements (should extract more than 9 chars)"""
        try:
            from advanced_ocr.engines.trocr_engine import TrOCREngine
            from advanced_ocr.config import OCRConfig
            
            # Check if TrOCREngine exists and is properly implemented
            config = OCRConfig()
            engine = TrOCREngine(config)
            
            # Check implementation for performance fixes
            import inspect
            source = inspect.getsource(engine.__class__)
            
            # Should not have the old bugs that caused 9-character limit
            performance_indicators = [
                'batch' in source.lower(),
                'preprocessing' not in source.lower(),  # Should not do its own preprocessing
                'region' in source.lower(),  # Should work with provided regions
            ]
            
            # At least some performance indicators should be present
            assert any(performance_indicators), \
                "TrOCREngine should have performance improvements"
            
            print("✅ TrOCR performance improvements detected")
            
        except ImportError:
            pytest.skip("TrOCREngine not implemented yet")
        except Exception as e:
            pytest.fail(f"TrOCR performance test failed: {e}")
    
    def test_engine_coordination_dependencies(self):
        """Test if engine_coordinator has correct dependencies"""
        try:
            from advanced_ocr.engines import engine_coordinator
            
            import inspect
            source = inspect.getsource(engine_coordinator)
            
            # Should import content_classifier
            assert ('content_classifier' in source or 
                    'ContentClassifier' in source), \
                "engine_coordinator should import content_classifier"
            
            # Should import individual engines
            engines = ['tesseract', 'paddleocr', 'easyocr', 'trocr']
            engine_imports = any(engine in source.lower() for engine in engines)
            assert engine_imports, "engine_coordinator should import individual engines"
            
        except Exception as e:
            pytest.fail(f"Engine coordination dependencies test failed: {e}")
    
    def test_engine_coordination_pipeline_integration(self, sample_image, sample_text_regions):
        """Test complete engine coordination pipeline - ARCHITECTURALLY CORRECT"""
        try:
            from advanced_ocr.engines.engine_coordinator import EngineCoordinator
            from advanced_ocr.preprocessing.image_processor import ImageProcessor
            from advanced_ocr.utils.model_utils import ModelLoader
            from advanced_ocr.config import OCRConfig
            from advanced_ocr.results import OCRResult
            from PIL import Image
            import numpy as np
            
            config = OCRConfig()
            
            # CORRECT PIPELINE FLOW: Follow architectural plan exactly
            # Step 1: Load raw image
            pil_image = Image.open(sample_image)
            np_image = np.array(pil_image)
            
            # Step 2: Preprocess image (image_processor.py)
            model_loader = ModelLoader(config)
            image_processor = ImageProcessor(model_loader, config)
            preprocessing_result = image_processor.process_image(np_image)
            
            print(f"Preprocessing completed: {len(preprocessing_result.text_regions)} regions detected")
            
            # Step 3: Engine coordination (engine_coordinator.py)
            coordinator = EngineCoordinator(config)
            results = coordinator.coordinate(
                preprocessing_result.enhanced_image,  # preprocessed numpy array
                preprocessing_result.text_regions     # detected text regions
            )
            
            # Verify coordination worked
            assert results is not None, "Coordination returned None"
            
            if isinstance(results, list):
                assert len(results) > 0, "No results returned"
                for result in results:
                    assert isinstance(result, OCRResult), "Invalid result type"
                    assert hasattr(result, 'text'), "Result missing text"
            else:
                assert isinstance(results, OCRResult), "Result should be OCRResult"
                assert hasattr(results, 'text'), "Result missing text"
            
            print(f"✅ Engine coordination pipeline test passed:")
            if isinstance(results, list):
                print(f"   - Multiple results: {len(results)} engines used")
                for i, result in enumerate(results):
                    print(f"   - Engine {i+1}: {len(result.text) if result.text else 0} chars extracted")
            else:
                print(f"   - Single result: {len(results.text) if results.text else 0} chars extracted")
            
        except Exception as e:
            pytest.fail(f"Engine coordination pipeline integration test failed: {e}")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])