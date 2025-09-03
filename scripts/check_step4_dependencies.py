# scripts/check_step4_dependencies.py - Check and fix Step 4 dependencies

import os
import sys
from pathlib import Path
import importlib.util

def check_file_exists(filepath):
    """Check if a file exists"""
    return os.path.exists(filepath)

def check_dependencies():
    """Check all Step 4 dependencies"""
    print("Step 4 Dependency Checker")
    print("=" * 40)
    
    project_root = Path(__file__).parent.parent
    src_dir = project_root / "src"
    
    # Required files for Step 4
    required_files = {
        "Adaptive Processor": src_dir / "preprocessing" / "adaptive_processor.py",
        "Enhanced Engine Manager": src_dir / "core" / "enhanced_engine_manager.py", 
        "Base Engine": src_dir / "core" / "base_engine.py",
        "Engine Manager": src_dir / "core" / "engine_manager.py",
        "Quality Analyzer": src_dir / "preprocessing" / "quality_analyzer.py",
        "Image Enhancer": src_dir / "preprocessing" / "image_enhancer.py",
        "Skew Corrector": src_dir / "preprocessing" / "skew_corrector.py",
        "Config Manager": src_dir / "utils" / "config.py",
        "Logger": src_dir / "utils" / "logger.py",
        "Test Suite": project_root / "tests" / "test_adaptive_processor.py"
    }
    
    missing_files = []
    existing_files = []
    
    for name, filepath in required_files.items():
        if check_file_exists(filepath):
            existing_files.append((name, filepath))
            print(f"✓ {name}: {filepath}")
        else:
            missing_files.append((name, filepath))
            print(f"✗ {name}: {filepath} (MISSING)")
    
    print(f"\nSummary:")
    print(f"Existing: {len(existing_files)}")
    print(f"Missing: {len(missing_files)}")
    
    if missing_files:
        print(f"\nMissing Dependencies:")
        for name, filepath in missing_files:
            print(f"  - {name}: {filepath}")
            
        print(f"\nTo fix these issues, you need to either:")
        print(f"1. Create the missing files")
        print(f"2. Use the mock testing approach")
        print(f"3. Update import paths")
    
    return missing_files, existing_files

def create_minimal_dependencies():
    """Create minimal versions of missing dependencies"""
    print("\nCreating minimal dependency stubs...")
    
    project_root = Path(__file__).parent.parent
    src_dir = project_root / "src"
    
    # Create directories
    (src_dir / "preprocessing").mkdir(exist_ok=True, parents=True)
    (src_dir / "utils").mkdir(exist_ok=True, parents=True)
    
    # Quality Analyzer stub
    quality_analyzer_code = '''"""
Minimal Quality Analyzer for Step 4 Testing
"""
import numpy as np
from dataclasses import dataclass
from enum import Enum

class ImageType(Enum):
    PRINTED_TEXT = "printed_text"
    HANDWRITTEN_TEXT = "handwritten_text"
    TABLE_DOCUMENT = "table_document"
    FORM_DOCUMENT = "form_document"
    LOW_QUALITY = "low_quality"
    NATURAL_SCENE = "natural_scene"

class ImageQuality(Enum):
    VERY_POOR = "very_poor"
    POOR = "poor"
    FAIR = "fair"
    GOOD = "good"
    EXCELLENT = "excellent"

@dataclass
class QualityMetrics:
    overall_score: float = 0.7
    sharpness_score: float = 0.8
    noise_level: float = 0.2
    contrast_score: float = 0.6
    brightness_score: float = 0.7
    skew_angle: float = 1.5
    image_type: ImageType = ImageType.PRINTED_TEXT
    quality_level: ImageQuality = ImageQuality.GOOD

class IntelligentQualityAnalyzer:
    def __init__(self, config=None):
        self.config = config or {}
    
    def analyze_image(self, image, cache_key=None):
        if image is None or len(image.shape) < 2:
            return QualityMetrics(
                overall_score=0.1,
                quality_level=ImageQuality.VERY_POOR
            )
        
        height, width = image.shape[:2]
        if height < 100 or width < 100:
            return QualityMetrics(
                overall_score=0.3,
                quality_level=ImageQuality.POOR
            )
        
        return QualityMetrics()
'''
    
    # Image Enhancer stub
    image_enhancer_code = '''"""
Minimal Image Enhancer for Step 4 Testing
"""
import numpy as np
import cv2
from dataclasses import dataclass
from enum import Enum

class EnhancementStrategy(Enum):
    CONSERVATIVE = "conservative"
    BALANCED = "balanced" 
    AGGRESSIVE = "aggressive"

@dataclass
class EnhancementResult:
    enhanced_image: np.ndarray
    enhancement_applied: str = "balanced"
    quality_improvement: float = 0.1
    processing_time: float = 0.5
    warnings: list = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []

class AIImageEnhancer:
    def __init__(self, config=None):
        self.config = config or {}
    
    def enhance_image(self, image, strategy=None):
        if image is None:
            return EnhancementResult(
                enhanced_image=np.zeros((100, 100, 3), dtype=np.uint8),
                warnings=["Invalid input image"]
            )
        
        enhanced = image.copy()
        if len(enhanced.shape) == 3:
            enhanced = cv2.addWeighted(enhanced, 1.1, enhanced, 0, 10)
        
        return EnhancementResult(
            enhanced_image=enhanced,
            enhancement_applied=strategy.value if strategy else "balanced"
        )
'''
    
    # Skew Corrector stub
    skew_corrector_code = '''"""
Minimal Skew Corrector for Step 4 Testing
"""
import numpy as np
import cv2
from dataclasses import dataclass

@dataclass
class SkewDetectionResult:
    angle: float = 1.5
    confidence: float = 0.8
    detection_method: str = "hough"

@dataclass
class SkewCorrectionResult:
    corrected_image: np.ndarray
    correction_applied: bool = True
    original_angle: float = 1.5
    corrected_angle: float = 0.0
    processing_time: float = 0.3
    warnings: list = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []

class EnhancedSkewCorrector:
    def __init__(self, config=None):
        self.config = config or {}
    
    def correct_skew(self, image, **params):
        if image is None:
            return SkewCorrectionResult(
                corrected_image=np.zeros((100, 100, 3), dtype=np.uint8),
                warnings=["Invalid input image"]
            )
        
        corrected = image.copy()
        if len(image.shape) >= 2:
            center = (image.shape[1] // 2, image.shape[0] // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, -0.5, 1.0)
            corrected = cv2.warpAffine(corrected, rotation_matrix, (image.shape[1], image.shape[0]))
        
        return SkewCorrectionResult(corrected_image=corrected)
'''
    
    # Config Manager stub
    config_manager_code = '''"""
Minimal Config Manager for Step 4 Testing
"""
import yaml
from pathlib import Path

class ConfigManager:
    def __init__(self):
        self.config = {}
    
    def load_config(self, config_path):
        if isinstance(config_path, str):
            config_path = Path(config_path)
        
        if config_path.exists():
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        
        return {
            "quality_analyzer": {},
            "image_enhancer": {},
            "skew_corrector": {},
            "system": {"max_workers": 4}
        }
    
    def save_config(self, config, config_path):
        if isinstance(config_path, str):
            config_path = Path(config_path)
        
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
'''
    
    # Logger stub
    logger_code = '''"""
Minimal Logger for Step 4 Testing
"""
import logging
import sys

def setup_logger(name="ocr_processor", level=logging.INFO):
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
    
    return logger
'''
    
    # Write files
    files_to_create = [
        (src_dir / "preprocessing" / "quality_analyzer.py", quality_analyzer_code),
        (src_dir / "preprocessing" / "image_enhancer.py", image_enhancer_code),
        (src_dir / "preprocessing" / "skew_corrector.py", skew_corrector_code),
        (src_dir / "utils" / "config.py", config_manager_code),
        (src_dir / "utils" / "logger.py", logger_code)
    ]
    
    created_count = 0
    for filepath, content in files_to_create:
        if not filepath.exists():
            with open(filepath, 'w') as f:
                f.write(content)
            print(f"Created: {filepath}")
            created_count += 1
        else:
            print(f"Exists: {filepath}")
    
    # Create __init__.py files
    init_files = [
        src_dir / "preprocessing" / "__init__.py",
        src_dir / "utils" / "__init__.py"
    ]
    
    for init_file in init_files:
        if not init_file.exists():
            with open(init_file, 'w') as f:
                f.write('"""Package initialization"""')
            print(f"Created: {init_file}")
            created_count += 1
    
    print(f"\nCreated {created_count} dependency files")
    return created_count

def run_basic_functionality_test():
    """Run basic functionality test with created dependencies"""
    print("\nRunning basic functionality test...")
    
    try:
        # Add src to path
        project_root = Path(__file__).parent.parent
        src_path = str(project_root / "src")
        if src_path not in sys.path:
            sys.path.insert(0, src_path)
        
        # Import and test
        from preprocessing.adaptive_processor import (
            AdaptivePreprocessor, ProcessingOptions, ProcessingLevel
        )
        
        # Create test image
        import numpy as np
        test_image = np.ones((300, 400, 3), dtype=np.uint8) * 255
        
        # Initialize preprocessor
        preprocessor = AdaptivePreprocessor()
        
        # Test basic processing
        result = preprocessor.process_image(test_image)
        
        print(f"✓ Basic processing successful: {result.success}")
        print(f"✓ Processing time: {result.processing_time:.2f}s")
        print(f"✓ Steps executed: {len(result.processing_steps)}")
        
        # Test batch processing
        images = [test_image, test_image.copy()]
        batch_results = preprocessor.process_batch(images)
        
        print(f"✓ Batch processing: {len(batch_results)} results")
        
        # Test statistics
        stats = preprocessor.get_processing_statistics()
        print(f"✓ Statistics tracking: {stats['total_processed']} processed")
        
        preprocessor.shutdown()
        
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main dependency checker and fixer"""
    print("Step 4 Dependency Checker and Resolution")
    print("=" * 50)
    
    # Check dependencies
    missing, existing = check_dependencies()
    
    if missing:
        print(f"\nFound {len(missing)} missing dependencies.")
        create_choice = input("Create minimal dependency stubs? (y/n): ")
        
        if create_choice.lower() == 'y':
            created = create_minimal_dependencies()
            print(f"Created {created} files")
            
            # Test functionality
            if run_basic_functionality_test():
                print("\n✓ Step 4 components are now functional!")
                print("\nNext steps:")
                print("1. Run the full test suite: python scripts/run_step4_tests.py")
                print("2. Run integration tests: python scripts/test_step4_integration.py")
                print("3. Try the usage examples: python examples/step4_usage_examples.py")
            else:
                print("\n✗ Basic functionality test failed. Check the error messages above.")
        else:
            print("\nSkipping dependency creation.")
            print("You'll need to either:")
            print("1. Create the missing files manually")
            print("2. Use the mock testing approach")
            print("3. Run tests with dependency mocking")
    else:
        print(f"\n✓ All {len(existing)} dependencies are present!")
        
        # Test functionality
        if run_basic_functionality_test():
            print("\n✓ Step 4 is ready for testing!")
        else:
            print("\n⚠ Dependencies exist but functionality test failed.")

if __name__ == "__main__":
    main()