#!/usr/bin/env python3
"""
Test 3: Logger System
Tests if your logging infrastructure works correctly
"""

import sys
import os
sys.path.append('.')

import logging
from pathlib import Path

def test_logger_system():
    """Test logger functionality"""
    print("=" * 60)
    print("TEST 3: LOGGER SYSTEM")
    print("=" * 60)
    
    # Test 1: Import logger module
    try:
        from src.utils.logger import setup_logger
        print("✓ Logger module imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import logger module: {e}")
        return False
    except Exception as e:
        print(f"✗ Error importing logger module: {e}")
        return False
    
    # Test 2: Create basic logger
    try:
        logger = setup_logger("test_logger", "INFO")
        print("✓ Basic logger created successfully")
    except Exception as e:
        print(f"✗ Failed to create logger: {e}")
        return False
    
    # Test 3: Test different log levels
    try:
        print("Testing log levels...")
        logger.debug("This is a DEBUG message")
        logger.info("This is an INFO message")
        logger.warning("This is a WARNING message")
        logger.error("This is an ERROR message")
        print("✓ All log levels working")
    except Exception as e:
        print(f"✗ Error with log levels: {e}")
        return False
    
    # Test 4: Test logger with different levels
    try:
        debug_logger = setup_logger("debug_test", "DEBUG")
        debug_logger.debug("Debug level logger test")
        print("✓ DEBUG level logger works")
        
        error_logger = setup_logger("error_test", "ERROR") 
        error_logger.error("Error level logger test")
        print("✓ ERROR level logger works")
    except Exception as e:
        print(f"✗ Error with different logger levels: {e}")
        return False
    
    # Test 5: Test logger from config
    try:
        from src.utils.config import Config
        config = Config("data/configs/ocr_config.yaml")
        
        log_level = config.get("log_level", "INFO")
        config_logger = setup_logger("config_test", log_level)
        config_logger.info(f"Logger created with config level: {log_level}")
        print(f"✓ Config-based logger works (level: {log_level})")
    except Exception as e:
        print(f"✗ Error with config-based logger: {e}")
        return False
    
    # Test 6: Check log directory
    try:
        logs_dir = Path("logs")
        if logs_dir.exists():
            print(f"✓ Logs directory exists: {logs_dir}")
            log_files = list(logs_dir.glob("*.log"))
            if log_files:
                print(f"✓ Found {len(log_files)} log files")
                for log_file in log_files[:3]:  # Show first 3
                    print(f"  - {log_file.name}")
            else:
                print("⚠ No log files found (may be normal)")
        else:
            print("⚠ Logs directory doesn't exist (may be created on first use)")
    except Exception as e:
        print(f"⚠ Error checking logs directory: {e}")
    
    # Test 7: Test OCR-specific loggers
    try:
        ocr_logger = setup_logger("OCR.Engine", "INFO")
        ocr_logger.info("OCR engine logger test")
        print("✓ OCR-specific logger works")
        
        preprocessing_logger = setup_logger("OCR.Preprocessing", "INFO")
        preprocessing_logger.info("Preprocessing logger test")
        print("✓ Preprocessing logger works")
        
        engine_manager_logger = setup_logger("OCR.EngineManager", "INFO")
        engine_manager_logger.info("Engine manager logger test")
        print("✓ Engine manager logger works")
    except Exception as e:
        print(f"✗ Error with OCR-specific loggers: {e}")
        return False
    
    # Test 8: Test logger performance
    try:
        perf_logger = setup_logger("performance_test", "INFO")
        import time
        start_time = time.time()
        
        for i in range(100):
            perf_logger.info(f"Performance test message {i}")
        
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"✓ Logger performance test: 100 messages in {elapsed:.3f}s")
        
        if elapsed < 1.0:
            print("✓ Logger performance is good")
        else:
            print("⚠ Logger might be slow")
    except Exception as e:
        print(f"⚠ Logger performance test failed: {e}")
    
    print("\n" + "=" * 60)
    print("TEST 3: PASSED - Logger system works")
    print("=" * 60)
    
    return True

def main():
    """Run the logger test"""
    try:
        success = test_logger_system()
        if success:
            print(f"\nSUCCESS: Logger system is working properly")
            print(f"Next step: Run Test 4 (Image Utilities)")
        else:
            print(f"\nFAILED: Please fix logger issues before proceeding")
        return success
        
    except Exception as e:
        print(f"\nUNEXPECTED ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)