# Advanced OCR System Test Suite

This directory contains comprehensive test suites for the Advanced OCR System, designed to validate all components and ensure system reliability.

## Test Files Overview

### Quick Test (`test_quick.py`)
A fast, simple test to verify basic OCR system functionality.

**Usage:**
```bash
python tests/test_quick.py
```

**What it tests:**
- Module imports
- Configuration system
- Data structures
- Basic pipeline execution
- Batch processing
- Error handling

**When to use:** Quick sanity check, CI/CD pipelines, development verification.

### Comprehensive Test (`test_comprehensive.py`)
Detailed unit and integration tests covering all system components.

**Usage:**
```bash
python -m pytest tests/test_comprehensive.py -v
# or
python tests/test_comprehensive.py
```

**Coverage:**
- Configuration system validation
- Data structures and results
- Preprocessing components (image processor, text detector, quality analyzer)
- OCR engines (Tesseract, EasyOCR, PaddleOCR, TrOCR)
- Postprocessing components
- Core pipeline orchestration
- Utility modules
- Integration testing
- Performance benchmarking
- Error handling

### Pipeline Test (`test_pipeline.py`)
Focused tests for the OCR pipeline workflow.

**Usage:**
```bash
python -m pytest tests/test_pipeline.py -v
```

### Test Runner (`run_all_tests.py`)
Master test runner that executes all test suites with detailed reporting.

**Usage:**
```bash
# Run all tests
python tests/run_all_tests.py

# Run specific test categories
python tests/run_all_tests.py --performance    # Performance tests only
python tests/run_all_tests.py --unit          # Unit tests only
python tests/run_all_tests.py --integration   # Integration tests only

# Verbose output
python tests/run_all_tests.py --verbose
```

## Test Categories

### 1. Unit Tests
- Individual component functionality
- Isolated module testing
- Mocked dependencies

### 2. Integration Tests
- Component interactions
- Data flow validation
- Pipeline orchestration

### 3. End-to-End Tests
- Complete OCR pipeline execution
- Real image processing workflows

### 4. Performance Tests
- Speed and efficiency validation
- Memory usage monitoring
- Benchmarking against requirements

### 5. Error Handling Tests
- Edge case validation
- Failure scenario testing
- Graceful error recovery

## Test Structure

```
tests/
├── test_quick.py          # Fast sanity checks
├── test_comprehensive.py  # Complete test suite
├── test_pipeline.py       # Pipeline-specific tests
├── run_all_tests.py       # Master test runner
├── test_preprocessing.py  # Existing preprocessing tests
├── test_engines.py        # OCR engine tests
├── test_integration.py    # Integration tests
└── README.md             # This file
```

## Running Tests

### Prerequisites
```bash
pip install -r requirements.txt
pip install pytest  # Optional, for advanced test running
```

### Quick Verification
```bash
# Fast check
python tests/test_quick.py

# If it passes, run comprehensive tests
python tests/run_all_tests.py
```

### Development Testing
```bash
# Run specific test file
python -m pytest tests/test_comprehensive.py::TestCorePipeline::test_pipeline_execution -v

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

## Test Results Interpretation

### Success Indicators
- ✅ All tests pass
- Performance meets requirements (< 3 seconds pipeline time)
- Memory usage within bounds
- Error handling works correctly

### Common Issues
- Import errors: Check Python path setup
- Model loading failures: Ensure test mocks are working
- Performance issues: Review system resources

## Adding New Tests

### Test File Structure
```python
import unittest
import sys
import os

# Setup paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

class TestNewComponent(unittest.TestCase):
    def setUp(self):
        # Test setup
        pass

    def test_feature(self):
        # Test implementation
        pass

if __name__ == "__main__":
    unittest.main()
```

### Mock Strategy
Use mocks for external dependencies:
- Model files (CRAFT, TrOCR, etc.)
- External OCR engines
- File I/O operations

### Test Naming Convention
- `test_*` for test methods
- `Test*` for test classes
- Descriptive names indicating what is tested

## Continuous Integration

For CI/CD pipelines, use:
```bash
# Quick check (fast)
python tests/test_quick.py

# Full validation (comprehensive)
python tests/run_all_tests.py --verbose
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure correct Python path
   PYTHONPATH=/path/to/project/src python tests/test_quick.py
   ```

2. **Missing Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Performance Issues**
   - Check system resources
   - Review mock implementations
   - Consider hardware limitations

### Debug Mode
```bash
# Run with debug output
python tests/run_all_tests.py --verbose
```

## Contributing

When adding new features:
1. Add corresponding unit tests
2. Update integration tests if needed
3. Ensure performance requirements are met
4. Update this README if test structure changes

## Test Coverage Goals

- **Unit Tests:** 90%+ coverage of individual modules
- **Integration Tests:** All component interactions
- **Performance Tests:** Meet <3 second pipeline requirement
- **Error Handling:** Comprehensive edge case coverage

---

For questions or issues with the test suite, refer to the main project documentation or create an issue in the repository.
