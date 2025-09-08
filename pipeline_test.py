#!/usr/bin/env python3
"""
Comprehensive OCR Pipeline Test Script
Tests each component of the OCR pipeline to identify where failures occur
"""

import sys
import os
import logging
import traceback
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

def test_stage(stage_name, test_func):
    """Helper to test a stage and report results"""
    print(f"\n{'='*60}")
    print(f"TESTING: {stage_name}")
    print('='*60)

    try:
        result = test_func()
        print(f"✅ {stage_name}: SUCCESS")
        return True, result
    except Exception as e:
        print(f"❌ {stage_name}: FAILED")
        print(f"Error: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False, None

def test_imports():
    """Test basic imports"""
    print("Testing basic imports...")

    # Test core imports
    from src.utils.config import Config
    from src.utils.logger import setup_logger

    # Test engine imports
    from src.engines.tesseract_engine import TesseractEngine
    from src.engines.easyocr_engine import EasyOCREngine
    from src.engines.paddle_engine import PaddleOCREngine
    from src.engines.trocr_engine import TrOCREngine

    # Test core components
    from src.core.engine_manager import OCREngineManager
    from src.core.base_engine import BaseOCREngine

    # Test preprocessing
    from src.preprocessing.text_detector import AdvancedTextDetector

    # Test postprocessing
    from src.postprocessing.postprocessing_pipeline import PostProcessingPipeline

    print("All imports successful")
    return True

def test_config():
    """Test configuration system"""
    print("Testing configuration system...")

    from src.utils.config import Config

    config = Config('data/configs/working_config.yaml')
    print(f"Config loaded: {config.get('default_engine')}")

    # Test engine configurations
    engines = ['tesseract', 'easyocr', 'paddleocr', 'trocr']
    for engine in engines:
        enabled = config.get(f"engines.{engine}.enabled")
        print(f"  {engine}: enabled={enabled}")

    return config

def test_text_detector():
    """Test text detection component"""
    print("Testing text detector...")

    from src.preprocessing.text_detector import AdvancedTextDetector
    from src.utils.config import Config

    config = Config('data/configs/working_config.yaml')
    detector = AdvancedTextDetector(config.get("text_detection", {}))

    print(f"Text detector initialized: {type(detector)}")
    return detector

def test_engine_manager():
    """Test engine manager initialization"""
    print("Testing engine manager...")

    from src.core.engine_manager import OCREngineManager
    from src.utils.config import Config

    config = Config('data/configs/working_config.yaml')
    manager = OCREngineManager(config)

    print(f"Engine manager created: {type(manager)}")
    print(f"Registered engines: {list(manager.engines.keys())}")

    return manager

def test_engine_initialization():
    """Test engine initialization"""
    print("Testing engine initialization...")

    from src.core.engine_manager import OCREngineManager
    from src.utils.config import Config

    config = Config('data/configs/working_config.yaml')
    manager = OCREngineManager(config)

    # Try to initialize engines
    init_results = manager.initialize_engines()
    print(f"Engine initialization results: {init_results}")

    available = manager.get_available_engines()
    print(f"Available engines: {available}")

    return manager, available

def test_postprocessing():
    """Test postprocessing pipeline"""
    print("Testing postprocessing pipeline...")

    from src.postprocessing.postprocessing_pipeline import PostProcessingPipeline

    pipeline = PostProcessingPipeline(config_path='data/configs/working_config.yaml')
    print(f"Postprocessing pipeline created: {type(pipeline)}")

    return pipeline

def test_full_system():
    """Test the full OCR system"""
    print("Testing full OCR system...")

    from src.api.main import UnifiedOCRSystem

    system = UnifiedOCRSystem()
    print(f"Unified OCR system created: {type(system)}")

    stats = system.get_statistics()
    print(f"System stats: {stats}")

    return system

def test_image_processing():
    """Test actual image processing"""
    print("Testing image processing...")

    from src.api.main import UnifiedOCRSystem
    import cv2
    import numpy as np

    system = UnifiedOCRSystem()

    # Create a simple test image
    test_image = np.ones((100, 200, 3), dtype=np.uint8) * 255
    cv2.putText(test_image, "TEST", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    print("Created test image")

    # Try processing
    from src.api.main import OCRRequest
    request = OCRRequest(engines=["tesseract"], confidence_threshold=0.1)

    result = system.process_image(test_image, request)
    print(f"Processing result: success={result['success']}, text='{result['text']}'")

    return result

def main():
    """Run all pipeline tests"""
    print("🔍 COMPREHENSIVE OCR PIPELINE DIAGNOSTIC")
    print("="*60)

    # Configure logging
    logging.basicConfig(level=logging.INFO)

    results = {}

    # Test 1: Basic imports
    success, _ = test_stage("Basic Imports", test_imports)
    results['imports'] = success

    if not success:
        print("\n❌ CRITICAL: Basic imports failed. Cannot continue testing.")
        return results

    # Test 2: Configuration
    success, config = test_stage("Configuration System", test_config)
    results['config'] = success

    # Test 3: Text Detector
    success, detector = test_stage("Text Detector", test_text_detector)
    results['text_detector'] = success

    # Test 4: Engine Manager
    success, manager = test_stage("Engine Manager", test_engine_manager)
    results['engine_manager'] = success

    # Test 5: Engine Initialization
    success, (manager, available) = test_stage("Engine Initialization", test_engine_initialization)
    results['engine_init'] = success

    # Test 6: Postprocessing
    success, pipeline = test_stage("Postprocessing Pipeline", test_postprocessing)
    results['postprocessing'] = success

    # Test 7: Full System
    success, system = test_stage("Full OCR System", test_full_system)
    results['full_system'] = success

    # Test 8: Image Processing (only if system works)
    if results['full_system']:
        success, result = test_stage("Image Processing", test_image_processing)
        results['image_processing'] = success
    else:
        print("\n⏭️  Skipping image processing test (full system failed)")
        results['image_processing'] = False

    # Summary
    print(f"\n{'='*60}")
    print("📊 TEST SUMMARY")
    print('='*60)

    passed = sum(results.values())
    total = len(results)

    for test, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test:20}: {status}")

    print(f"\nOverall: {passed}/{total} tests passed")

    if passed < total:
        print("\n🔧 FAILURE ANALYSIS:")
        if not results['imports']:
            print("- Basic imports failed - check Python environment and dependencies")
        if not results['config']:
            print("- Configuration system failed - check config files")
        if not results['engine_init']:
            print("- Engine initialization failed - check OCR engine installations")
        if not results['full_system']:
            print("- Full system failed - check component integration")
        if not results['image_processing']:
            print("- Image processing failed - check pipeline execution")

    return results

if __name__ == "__main__":
    main()
