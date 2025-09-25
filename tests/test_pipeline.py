#!/usr/bin/env python3
"""
OCR Pipeline Debug Test Script

This script tests each component of the OCR pipeline step by step
to identify exactly where failures occur during development.
"""

import os
import sys
import traceback
from pathlib import Path

# Add the src directory to Python path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Test image path
IMAGE_PATH = "tests/images/img1.jpg"

def check_prerequisites():
    """Check if all prerequisites are met"""
    print("🔍 CHECKING PREREQUISITES")
    print("-" * 40)
    
    # Check if image exists
    if not os.path.exists(IMAGE_PATH):
        print(f"❌ Test image not found: {IMAGE_PATH}")
        print("   Please create the test image or update IMAGE_PATH")
        return False
    
    print(f"✅ Test image found: {IMAGE_PATH}")
    
    # Check image file size
    size = os.path.getsize(IMAGE_PATH)
    print(f"📊 Image size: {size:,} bytes ({size/1024:.1f} KB)")
    
    return True


def test_imports():
    """Test all imports step by step"""
    print("\n📦 TESTING IMPORTS")
    print("-" * 40)
    
    imports_to_test = [
        # Core modules
        ("advanced_ocr.types", "ProcessingOptions, OCRResult, ProcessingStrategy"),
        ("advanced_ocr.exceptions", "OCRLibraryError, EngineNotAvailableError"),
        ("advanced_ocr.pipeline", "OCRLibrary"),
        
        # Component modules
        ("advanced_ocr.core.base_engine", "BaseOCREngine"),
        ("advanced_ocr.core.engine_manager", "EngineManager"),
        ("advanced_ocr.preprocessing.quality_analyzer", "QualityAnalyzer"),
        ("advanced_ocr.preprocessing.image_enhancer", "ImageEnhancer"),
        
        # Engine modules (these might fail if dependencies not installed)
        ("advanced_ocr.engines.paddleocr", "PaddleOCREngine"),
        ("advanced_ocr.engines.easyocr", "EasyOCREngine"),
        ("advanced_ocr.engines.tesseract", "TesseractEngine"),
        ("advanced_ocr.engines.trocr", "TrOCREngine"),
    ]
    
    successful_imports = []
    failed_imports = []
    
    for module_name, classes in imports_to_test:
        try:
            module = __import__(module_name, fromlist=classes.split(", "))
            print(f"✅ {module_name}")
            successful_imports.append(module_name)
        except Exception as e:
            print(f"❌ {module_name}: {e}")
            failed_imports.append((module_name, str(e)))
    
    print(f"\n📊 Import Summary: {len(successful_imports)} success, {len(failed_imports)} failed")
    
    return len(failed_imports) == 0, successful_imports, failed_imports


def test_basic_imports():
    """Test just the essential imports needed for basic functionality"""
    print("\n🔧 TESTING BASIC IMPORTS")
    print("-" * 40)
    
    try:
        from advanced_ocr.types import ProcessingOptions, OCRResult, ProcessingStrategy
        print("✅ Types imported successfully")
        
        from advanced_ocr.exceptions import OCRLibraryError
        print("✅ Exceptions imported successfully")
        
        from advanced_ocr.pipeline import OCRLibrary
        print("✅ Pipeline imported successfully")
        
        return True, {
            'ProcessingOptions': ProcessingOptions,
            'OCRResult': OCRResult, 
            'ProcessingStrategy': ProcessingStrategy,
            'OCRLibraryError': OCRLibraryError,
            'OCRLibrary': OCRLibrary
        }
        
    except Exception as e:
        print(f"❌ Basic import failed: {e}")
        print(f"🔍 Full traceback:")
        traceback.print_exc()
        return False, {}


def test_pipeline_initialization(classes):
    """Test OCR pipeline initialization"""
    print("\n🏗️  TESTING PIPELINE INITIALIZATION")
    print("-" * 40)
    
    try:
        OCRLibrary = classes['OCRLibrary']
        ocr = OCRLibrary()
        print("✅ OCRLibrary initialized successfully")
        return True, ocr
        
    except Exception as e:
        print(f"❌ Pipeline initialization failed: {e}")
        print(f"🔍 Full traceback:")
        traceback.print_exc()
        return False, None


def test_image_loading(ocr):
    """Test image loading functionality"""
    print("\n🖼️  TESTING IMAGE LOADING")
    print("-" * 40)
    
    try:
        # Try to access the image loading method
        # This might be in utils.images or directly in pipeline
        
        # Method 1: Check if pipeline has direct image loading
        if hasattr(ocr, '_load_image'):
            image = ocr._load_image(IMAGE_PATH)
            print(f"✅ Image loaded via pipeline method: shape {getattr(image, 'shape', 'unknown')}")
            return True, image
            
        # Method 2: Try importing image utilities
        try:
            from advanced_ocr.utils.images import load_image
            image = load_image(IMAGE_PATH)
            print(f"✅ Image loaded via utils: shape {getattr(image, 'shape', 'unknown')}")
            return True, image
        except ImportError:
            pass
            
        # Method 3: Try basic CV2/PIL loading to test image validity
        try:
            import cv2
            image = cv2.imread(IMAGE_PATH)
            if image is not None:
                print(f"✅ Image loadable with OpenCV: shape {image.shape}")
                return True, image
            else:
                print("❌ OpenCV returned None - image may be corrupted")
                return False, None
        except ImportError:
            try:
                from PIL import Image
                image = Image.open(IMAGE_PATH)
                print(f"✅ Image loadable with PIL: size {image.size}")
                return True, image
            except ImportError:
                print("❌ No image loading library available")
                return False, None
        
    except Exception as e:
        print(f"❌ Image loading failed: {e}")
        print(f"🔍 Full traceback:")
        traceback.print_exc()
        return False, None


def test_quality_analysis(ocr):
    """Test quality analysis component"""
    print("\n📊 TESTING QUALITY ANALYSIS")
    print("-" * 40)
    
    try:
        # Try to access quality analyzer
        if hasattr(ocr, 'quality_analyzer'):
            analyzer = ocr.quality_analyzer
            print("✅ Quality analyzer accessible from pipeline")
        else:
            from advanced_ocr.preprocessing.quality_analyzer import QualityAnalyzer
            analyzer = QualityAnalyzer()
            print("✅ Quality analyzer imported and initialized")
        
        return True, analyzer
        
    except Exception as e:
        print(f"❌ Quality analysis setup failed: {e}")
        print(f"🔍 Full traceback:")
        traceback.print_exc()
        return False, None


def test_engine_manager(ocr):
    """Test engine manager component"""
    print("\n⚙️  TESTING ENGINE MANAGER")
    print("-" * 40)
    
    try:
        # Try to access engine manager
        if hasattr(ocr, 'engine_manager'):
            manager = ocr.engine_manager
            print("✅ Engine manager accessible from pipeline")
        else:
            from advanced_ocr.core.engine_manager import EngineManager
            manager = EngineManager()
            print("✅ Engine manager imported and initialized")
        
        # Check available engines
        if hasattr(manager, 'available_engines'):
            engines = manager.available_engines
            print(f"📋 Available engines: {list(engines.keys()) if engines else 'None'}")
        elif hasattr(manager, 'engines'):
            engines = manager.engines
            print(f"📋 Registered engines: {list(engines.keys()) if engines else 'None'}")
        else:
            print("⚠️  Could not determine available engines")
        
        return True, manager
        
    except Exception as e:
        print(f"❌ Engine manager setup failed: {e}")
        print(f"🔍 Full traceback:")
        traceback.print_exc()
        return False, None


def test_full_pipeline(ocr, classes):
    """Test the complete pipeline end-to-end"""
    print("\n🔄 TESTING FULL PIPELINE")
    print("-" * 40)
    
    try:
        ProcessingOptions = classes['ProcessingOptions']
        
        # Test with minimal options first
        print("1️⃣ Testing with default options...")
        result = ocr.process_image(IMAGE_PATH)
        
        print(f"✅ Pipeline completed successfully!")
        print(f"📝 Text length: {len(result.text)} characters")
        print(f"🎯 Confidence: {result.confidence:.3f}")
        print(f"🔧 Engine: {result.engine_used}")
        print(f"⏱️  Time: {result.processing_time:.3f}s")
        print(f"📊 Text preview: {repr(result.text[:100])}...")
        print(f"\n📄 FULL EXTRACTED TEXT:")
        print("-" * 50)
        print(result.text)
        print("-" * 50)
        return True, result
        
    except Exception as e:
        print(f"❌ Full pipeline failed: {e}")
        print(f"🔍 Full traceback:")
        traceback.print_exc()
        return False, None


def main():
    """Main test runner - stops at first failure for debugging"""
    print("🧪 OCR PIPELINE DEBUG TEST")
    print("=" * 50)
    print(f"📁 Working directory: {os.getcwd()}")
    print(f"🐍 Python path: {sys.path[:2]}...")  # Show first few paths
    
    # Step 1: Prerequisites
    if not check_prerequisites():
        print("\n❌ Prerequisites failed. Please fix before continuing.")
        return
    
    # Step 2: Test imports
    import_success, successful_imports, failed_imports = test_imports()
    if failed_imports:
        print(f"\n⚠️  Some imports failed. This might be due to missing dependencies:")
        for module, error in failed_imports:
            print(f"   - {module}: {error}")
        print("\nContinuing with basic imports...")
    
    # Step 3: Test basic imports
    basic_success, classes = test_basic_imports()
    if not basic_success:
        print("\n❌ CRITICAL: Basic imports failed. Cannot continue.")
        print("🔧 Fix the import errors in your modules first.")
        return
    
    # Step 4: Test pipeline initialization
    init_success, ocr = test_pipeline_initialization(classes)
    if not init_success:
        print("\n❌ CRITICAL: Pipeline initialization failed. Cannot continue.")
        return
    
    # Step 5: Test image loading
    image_success, image = test_image_loading(ocr)
    if not image_success:
        print("\n❌ CRITICAL: Image loading failed. Check image file and loading utilities.")
        return
    
    # Step 6: Test quality analysis
    quality_success, analyzer = test_quality_analysis(ocr)
    if not quality_success:
        print("\n⚠️  Quality analysis failed. Pipeline may work without it.")
    
    # Step 7: Test engine manager
    engine_success, manager = test_engine_manager(ocr)
    if not engine_success:
        print("\n❌ CRITICAL: Engine manager failed. Cannot continue.")
        return
    
    # Step 8: Test full pipeline
    pipeline_success, result = test_full_pipeline(ocr, classes)
    if not pipeline_success:
        print("\n❌ CRITICAL: Full pipeline test failed.")
        return
    
    # Success!
    print("\n🎉 ALL TESTS PASSED!")
    print("=" * 50)
    print("Your OCR pipeline is working correctly!")
    print(f"✅ Successfully processed: {IMAGE_PATH}")
    print(f"📝 Extracted {len(result.text)} characters of text")
    print(f"🎯 With confidence: {result.confidence:.1%}")


if __name__ == "__main__":
    main()