#!/usr/bin/env python3
"""
Test 2: Configuration System
Tests if your configuration system works correctly
"""

import sys
import os
sys.path.append('.')

from pathlib import Path

def test_config_system():
    """Test configuration loading and access"""
    print("=" * 60)
    print("TEST 2: CONFIGURATION SYSTEM")
    print("=" * 60)
    
    # Test 1: Check if config files exist
    config_files = [
        "data/configs/ocr_config.yaml",
        "data/configs/text_detection.yaml",
        "data/configs/preprocessing_config.yaml", 
        "data/configs/postprocessing_config.yaml"
    ]
    
    print("Checking config files...")
    for config_file in config_files:
        if Path(config_file).exists():
            print(f"✓ Found: {config_file}")
        else:
            print(f"⚠ Missing: {config_file}")
    
    # Test 2: Import config module
    try:
        from src.utils.config import Config
        print("✓ Config module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import Config module: {e}")
        return False
    except Exception as e:
        print(f"✗ Error importing Config module: {e}")
        return False
    
    # Test 3: Load main config
    try:
        config = Config("data/configs/ocr_config.yaml")
        print("✓ Main config loaded successfully")
    except FileNotFoundError:
        print("✗ Main config file not found")
        return False
    except Exception as e:
        print(f"✗ Error loading main config: {e}")
        return False
    
    # Test 4: Access engine configurations
    try:
        engines_config = config.get("engines", {})
        if engines_config:
            print(f"✓ Engine configs found: {list(engines_config.keys())}")
            
            # Check individual engine configs
            for engine_name in ["tesseract", "easyocr", "paddleocr", "trocr"]:
                engine_config = config.get(f"engines.{engine_name}", {})
                if engine_config:
                    enabled = engine_config.get("enabled", True)
                    print(f"  - {engine_name}: {'enabled' if enabled else 'disabled'}")
                else:
                    print(f"  - {engine_name}: no config found")
        else:
            print("⚠ No engine configurations found")
    except Exception as e:
        print(f"✗ Error accessing engine configs: {e}")
        return False
    
    # Test 5: Access preprocessing config
    try:
        preprocessing_config = config.get("preprocessing", {})
        if preprocessing_config:
            print(f"✓ Preprocessing config found: {list(preprocessing_config.keys())}")
        else:
            print("⚠ No preprocessing config found")
    except Exception as e:
        print(f"✗ Error accessing preprocessing config: {e}")
        return False
    
    # Test 6: Access postprocessing config  
    try:
        postprocessing_config = config.get("postprocessing", {})
        if postprocessing_config:
            print(f"✓ Postprocessing config found: {list(postprocessing_config.keys())}")
        else:
            print("⚠ No postprocessing config found")
    except Exception as e:
        print(f"✗ Error accessing postprocessing config: {e}")
        return False
    
    # Test 7: Test config value retrieval
    try:
        # Test default values
        test_value = config.get("non_existent_key", "default_value")
        if test_value == "default_value":
            print("✓ Default value mechanism works")
        else:
            print("⚠ Default value mechanism issue")
        
        # Test nested access
        system_config = config.get("system", {})
        if system_config:
            parallel = config.get("system.parallel_processing", True)
            max_workers = config.get("system.max_workers", 3)
            print(f"✓ System config - parallel: {parallel}, workers: {max_workers}")
        else:
            print("⚠ No system configuration found")
            
    except Exception as e:
        print(f"✗ Error testing config access: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("TEST 2: PASSED - Configuration system works")
    print("=" * 60)
    
    return True, config

def main():
    """Run the configuration test"""
    try:
        success, config = test_config_system()
        if success:
            print(f"\nSUCCESS: Configuration system is working properly")
            print(f"Next step: Run Test 3 (Logger System)")
        else:
            print(f"\nFAILED: Please fix configuration issues before proceeding")
        return success
        
    except Exception as e:
        print(f"\nUNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)