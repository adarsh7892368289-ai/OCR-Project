#!/usr/bin/env python3
"""
Test file for src/utils/config.py
Run this to identify specific issues in config.py
"""

import sys
import traceback
from pathlib import Path

def test_config_imports():
    """Test basic imports from config.py"""
    print("=" * 50)
    print("TESTING: config.py imports")
    print("=" * 50)
    
    try:
        from src.utils.config import (
            Config, ConfigManager, 
            ProcessingLevel, EngineStrategy,
            DetectionConfig, PreprocessingConfig, PostprocessingConfig,
            EngineConfig, TesseractConfig, EasyOCRConfig, PaddleOCRConfig, TrOCRConfig,
            SystemConfig, OutputConfig, OCRConfig,
            create_default_config, load_config
        )
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        traceback.print_exc()
        return False

def test_config_creation():
    """Test creating default config"""
    print("\n" + "=" * 50)
    print("TESTING: Config creation")
    print("=" * 50)
    
    try:
        from src.utils.config import Config
        
        # Test 1: Default config
        print("Test 1: Creating default config...")
        config = Config()
        print("✅ Default config created successfully")
        
        # Test 2: Config with non-existent file
        print("Test 2: Config with non-existent file...")
        config2 = Config("non_existent_file.yaml")
        print("✅ Config with non-existent file handled gracefully")
        
        return True
    except Exception as e:
        print(f"❌ Config creation failed: {e}")
        traceback.print_exc()
        return False

def test_config_basic_operations():
    """Test basic config operations"""
    print("\n" + "=" * 50)
    print("TESTING: Config basic operations")
    print("=" * 50)
    
    try:
        from src.utils.config import Config
        
        config = Config()
        
        # Test get operations
        print("Test 1: Get operations...")
        engines = config.get('engines', {})
        print(f"  Engines: {list(engines.keys()) if engines else 'None'}")
        
        default_engine = config.get('default_engine', 'unknown')
        print(f"  Default engine: {default_engine}")
        
        log_level = config.get('system.log_level', 'UNKNOWN')
        print(f"  Log level: {log_level}")
        
        # Test set operations
        print("Test 2: Set operations...")
        config.set('test.value', 'test_data')
        retrieved = config.get('test.value', 'not_found')
        print(f"  Set/Get test: {retrieved}")
        
        # Test engine config
        print("Test 3: Engine config...")
        tesseract_config = config.get_engine_config('tesseract')
        print(f"  Tesseract enabled: {tesseract_config.get('enabled', 'unknown')}")
        
        print("✅ All basic operations successful")
        return True
        
    except Exception as e:
        print(f"❌ Basic operations failed: {e}")
        traceback.print_exc()
        return False

def test_config_manager():
    """Test ConfigManager class"""
    print("\n" + "=" * 50)
    print("TESTING: ConfigManager")
    print("=" * 50)
    
    try:
        from src.utils.config import ConfigManager
        
        # Test creation
        print("Test 1: ConfigManager creation...")
        manager = ConfigManager()
        print("✅ ConfigManager created")
        
        # Test operations
        print("Test 2: ConfigManager operations...")
        config_dict = manager.get_config()
        print(f"  Config dict type: {type(config_dict)}")
        
        value = manager.get('default_engine', 'unknown')
        print(f"  Default engine via manager: {value}")
        
        engine_config = manager.get_engine_config('tesseract')
        print(f"  Tesseract config type: {type(engine_config)}")
        
        print("✅ ConfigManager operations successful")
        return True
        
    except Exception as e:
        print(f"❌ ConfigManager test failed: {e}")
        traceback.print_exc()
        return False

def test_dataclass_configs():
    """Test dataclass configurations"""
    print("\n" + "=" * 50)
    print("TESTING: Dataclass configurations")
    print("=" * 50)
    
    try:
        from src.utils.config import (
            DetectionConfig, PreprocessingConfig, PostprocessingConfig,
            TesseractConfig, EasyOCRConfig, PaddleOCRConfig, TrOCRConfig,
            SystemConfig, OutputConfig, OCRConfig
        )
        
        print("Test 1: Creating dataclass configs...")
        
        detection = DetectionConfig()
        print(f"  DetectionConfig: method={detection.method}")
        
        preprocessing = PreprocessingConfig()
        print(f"  PreprocessingConfig: level={preprocessing.enhancement_level}")
        
        postprocessing = PostprocessingConfig()
        print(f"  PostprocessingConfig: min_confidence={postprocessing.min_confidence}")
        
        tesseract = TesseractConfig()
        print(f"  TesseractConfig: enabled={tesseract.enabled}")
        
        system = SystemConfig()
        print(f"  SystemConfig: log_level={system.log_level}")
        
        output = OutputConfig()
        print(f"  OutputConfig: preserve_formatting={output.preserve_formatting}")
        
        ocr_config = OCRConfig()
        print(f"  OCRConfig: default_engine={ocr_config.default_engine}")
        
        print("✅ All dataclass configs created successfully")
        return True
        
    except Exception as e:
        print(f"❌ Dataclass config test failed: {e}")
        traceback.print_exc()
        return False

def test_enums():
    """Test enum values"""
    print("\n" + "=" * 50)
    print("TESTING: Enum values")
    print("=" * 50)
    
    try:
        from src.utils.config import ProcessingLevel, EngineStrategy
        
        print("Test 1: ProcessingLevel enum...")
        levels = [level.value for level in ProcessingLevel]
        print(f"  Processing levels: {levels}")
        
        print("Test 2: EngineStrategy enum...")
        strategies = [strategy.value for strategy in EngineStrategy]
        print(f"  Engine strategies: {strategies}")
        
        print("✅ Enum tests successful")
        return True
        
    except Exception as e:
        print(f"❌ Enum test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all config tests"""
    print("STARTING CONFIG.PY TESTS")
    print("=" * 60)
    
    tests = [
        ("Imports", test_config_imports),
        ("Creation", test_config_creation),
        ("Basic Operations", test_config_basic_operations),
        ("ConfigManager", test_config_manager),
        ("Dataclasses", test_dataclass_configs),
        ("Enums", test_enums),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<20}: {status}")
        if result:
            passed += 1
    
    print(f"\nOVERALL: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - config.py is working correctly!")
        return 0
    else:
        print("⚠️  SOME TESTS FAILED - Issues found in config.py")
        return 1

if __name__ == "__main__":
    sys.exit(main())
    
    
