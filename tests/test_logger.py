#!/usr/bin/env python3
"""
Test file for src/utils/logger.py
Run this to identify specific issues in logger.py
"""

import sys
import traceback
import tempfile
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def test_logger_imports():
    """Test basic imports from logger.py"""
    print("=" * 50)
    print("TESTING: logger.py imports")
    print("=" * 50)
    
    try:
        from src.utils.logger import setup_logger, get_logger
        print("✅ Basic imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        traceback.print_exc()
        return False

def test_logger_creation():
    """Test creating loggers"""
    print("\n" + "=" * 50)
    print("TESTING: Logger creation")
    print("=" * 50)
    
    try:
        from src.utils.logger import setup_logger, get_logger
        
        # Test 1: Default logger
        print("Test 1: Creating default logger...")
        logger = setup_logger()
        print(f"✅ Default logger created: {logger.name}")
        
        # Test 2: Named logger
        print("Test 2: Creating named logger...")
        named_logger = setup_logger("test_logger")
        print(f"✅ Named logger created: {named_logger.name}")
        
        # Test 3: Get existing logger
        print("Test 3: Getting existing logger...")
        retrieved_logger = get_logger("test_logger")
        print(f"✅ Retrieved logger: {retrieved_logger.name}")
        
        return True
    except Exception as e:
        print(f"❌ Logger creation failed: {e}")
        traceback.print_exc()
        return False

def test_logger_basic_operations():
    """Test basic logging operations"""
    print("\n" + "=" * 50)
    print("TESTING: Logger basic operations")
    print("=" * 50)
    
    try:
        from src.utils.logger import setup_logger
        
        logger = setup_logger("test_operations")
        
        # Test different log levels
        print("Test 1: Testing different log levels...")
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        print("✅ All log levels work")
        
        # Test with extra data
        print("Test 2: Testing structured logging...")
        logger.info("Test with extra data", extra={'key': 'value'})
        print("✅ Structured logging works")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic logging operations failed: {e}")
        traceback.print_exc()
        return False

def test_logger_file_output():
    """Test file logging"""
    print("\n" + "=" * 50)
    print("TESTING: File logging")
    print("=" * 50)
    
    try:
        from src.utils.logger import setup_logger
        
        # Create temporary log file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.log', delete=False) as tmp:
            log_file = tmp.name
        
        print(f"Test 1: Creating file logger at {log_file}...")
        logger = setup_logger("file_test", log_file=log_file)
        
        logger.info("Test file logging message")
        logger.warning("Test warning to file")
        
        # Check if file was created and has content
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                content = f.read()
            if content.strip():
                print("✅ File logging successful")
                print(f"  Log file size: {len(content)} characters")
                result = True
            else:
                print("❌ Log file is empty")
                result = False
        else:
            print("❌ Log file was not created")
            result = False
        
        # Cleanup
        try:
            os.unlink(log_file)
        except:
            pass
            
        return result
        
    except Exception as e:
        print(f"❌ File logging test failed: {e}")
        traceback.print_exc()
        return False

def test_logger_levels():
    """Test different logging levels"""
    print("\n" + "=" * 50)
    print("TESTING: Logging levels")
    print("=" * 50)
    
    try:
        from src.utils.logger import setup_logger
        import logging
        
        print("Test 1: Creating logger with DEBUG level...")
        debug_logger = setup_logger("debug_test", level=logging.DEBUG)
        debug_logger.debug("Debug level message")
        print("✅ DEBUG level logger works")
        
        print("Test 2: Creating logger with WARNING level...")
        warn_logger = setup_logger("warn_test", level=logging.WARNING)
        warn_logger.warning("Warning level message")
        warn_logger.info("This should be filtered out")
        print("✅ WARNING level logger works")
        
        print("Test 3: Creating logger with ERROR level...")
        error_logger = setup_logger("error_test", level=logging.ERROR)
        error_logger.error("Error level message")
        error_logger.warning("This should be filtered out")
        print("✅ ERROR level logger works")
        
        return True
        
    except Exception as e:
        print(f"❌ Logging levels test failed: {e}")
        traceback.print_exc()
        return False

def test_logger_config_integration():
    """Test logger integration with config system"""
    print("\n" + "=" * 50)
    print("TESTING: Logger config integration")
    print("=" * 50)
    
    try:
        # This tests if logger can work with config system
        from src.utils.config import Config
        from src.utils.logger import setup_logger
        
        print("Test 1: Using config with logger...")
        config = Config()
        log_level = config.get('system.log_level', 'INFO')
        
        logger = setup_logger("config_test")
        logger.info(f"Using log level from config: {log_level}")
        print("✅ Config integration works")
        
        return True
        
    except Exception as e:
        print(f"❌ Config integration test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all logger tests"""
    print("STARTING LOGGER.PY TESTS")
    print("=" * 60)
    
    tests = [
        ("Imports", test_logger_imports),
        ("Creation", test_logger_creation),
        ("Basic Operations", test_logger_basic_operations),
        ("File Output", test_logger_file_output),
        ("Logging Levels", test_logger_levels),
        ("Config Integration", test_logger_config_integration),
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
        print("🎉 ALL TESTS PASSED - logger.py is working correctly!")
        return 0
    else:
        print("⚠️  SOME TESTS FAILED - Issues found in logger.py")
        print("\nIf logger.py doesn't exist, please share it so I can help fix the issues.")
        return 1

if __name__ == "__main__":
    sys.exit(main())