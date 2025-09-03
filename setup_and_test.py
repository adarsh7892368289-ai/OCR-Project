"""
Complete Setup and Testing Script
Sets up your OCR system and runs comprehensive tests

File Location: setup_and_test.py (in project root)

Usage:
    python setup_and_test.py
"""

import subprocess
import sys
import os
import time
import requests
import zipfile
from pathlib import Path
from typing import List, Dict, Any
import yaml
import json

class OCRSystemSetup:
    def __init__(self):
        self.root_dir = Path(__file__).parent
        self.setup_complete = False
        self.test_results = {}
        
    def print_header(self, text: str):
        """Print formatted header"""
        print(f"\n{'='*60}")
        print(f" {text}")
        print(f"{'='*60}")
        
    def print_step(self, step: str):
        """Print formatted step"""
        print(f"\n>>> {step}")
        
    def run_command(self, cmd: List[str], description: str) -> bool:
        """Run a command and return success status"""
        self.print_step(f"Running: {description}")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"✅ {description} - SUCCESS")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ {description} - FAILED")
            print(f"Error: {e.stderr}")
            return False
        except Exception as e:
            print(f"❌ {description} - ERROR: {str(e)}")
            return False
    
    def check_python_version(self):
        """Check Python version compatibility"""
        self.print_step("Checking Python version")
        version = sys.version_info
        if version.major == 3 and version.minor >= 8:
            print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
            return True
        else:
            print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.8+")
            return False
    
    def setup_virtual_environment(self):
        """Set up virtual environment"""
        venv_path = self.root_dir / "venv"
        
        if venv_path.exists():
            print("✅ Virtual environment already exists")
            return True
            
        self.print_step("Creating virtual environment")
        return self.run_command([sys.executable, "-m", "venv", str(venv_path)], 
                               "Virtual environment creation")
    
    def get_pip_command(self):
        """Get the appropriate pip command"""
        venv_path = self.root_dir / "venv"
        if os.name == 'nt':  # Windows
            return str(venv_path / "Scripts" / "pip.exe")
        else:  # Unix/Linux/Mac
            return str(venv_path / "bin" / "pip")
    
    def install_dependencies(self):
        """Install required dependencies"""
        self.print_step("Installing dependencies")
        
        # Core dependencies
        dependencies = [
            "torch==2.0.1",
            "torchvision==0.15.2",
            "transformers==4.33.0",
            "pillow>=9.0.0",
            "opencv-python==4.8.0.76",
            "numpy>=1.21.0",
            "scipy>=1.7.0",
            "scikit-image>=0.19.0",
            "tesseract==0.1.3",
            "pytesseract==0.3.10",
            "easyocr==1.7.0",
            "paddlepaddle==2.5.1",
            "paddleocr==2.7.0.3",
            "fastapi==0.103.0",
            "uvicorn==0.23.0",
            "python-multipart==0.0.6",
            "pydantic==2.3.0",
            "pyyaml>=6.0",
            "requests>=2.28.0",
            "tqdm>=4.64.0",
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "pandas>=1.4.0"
        ]
        
        pip_cmd = self.get_pip_command()
        
        # Install each dependency
        success_count = 0
        for dep in dependencies:
            if self.run_command([pip_cmd, "install", dep], f"Installing {dep}"):
                success_count += 1
        
        print(f"\n📊 Installed {success_count}/{len(dependencies)} dependencies")
        return success_count == len(dependencies)
    
    def create_directories(self):
        """Create necessary directories"""
        self.print_step("Creating directory structure")
        
        directories = [
            "data/sample_images",
            "data/models/craft",
            "data/models/east", 
            "data/cache",
            "logs",
            "debug/text_detection",
            "benchmark",
            "profiles",
            "exports",
            "tests"
        ]
        
        for directory in directories:
            dir_path = self.root_dir / directory
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"✅ Created: {directory}")
        
        return True
    
    def download_models(self):
        """Download required models"""
        self.print_step("Downloading CRAFT model")
        
        craft_model_dir = self.root_dir / "data/models/craft"
        craft_model_path = craft_model_dir / "craft_mlt_25k.pth"
        
        if craft_model_path.exists():
            print("✅ CRAFT model already exists")
            return True
        
        try:
            # CRAFT model URL
            craft_url = "https://drive.google.com/uc?id=1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ"
            
            print("📥 Downloading CRAFT model (this may take a few minutes)...")
            response = requests.get(craft_url, stream=True)
            response.raise_for_status()
            
            with open(craft_model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            print("✅ CRAFT model downloaded successfully")
            return True
            
        except Exception as e:
            print(f"❌ Failed to download CRAFT model: {str(e)}")
            print("ℹ️  You can manually download it later from:")
            print("   https://drive.google.com/uc?id=1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ")
            return False
    
    def create_sample_images(self):
        """Create sample test images"""
        self.print_step("Creating sample test images")
        
        try:
            from PIL import Image, ImageDraw, ImageFont
            import numpy as np
            
            sample_dir = self.root_dir / "data/sample_images"
            
            # Create a simple test image with printed text
            img = Image.new('RGB', (800, 600), color='white')
            draw = ImageDraw.Draw(img)
            
            # Try to use a default font
            try:
                font = ImageFont.load_default()
            except:
                font = None
            
            # Add some text
            text_lines = [
                "Advanced OCR System Test",
                "This is printed text for testing",
                "Mixed content: 123-456-7890",
                "Email: test@example.com",
                "Date: September 4, 2025"
            ]
            
            y_position = 50
            for line in text_lines:
                draw.text((50, y_position), line, fill='black', font=font)
                y_position += 40
            
            # Save sample images
            img.save(sample_dir / "sample_printed.jpg", "JPEG")
            
            # Create a handwritten-style image
            img2 = Image.new('RGB', (800, 400), color='white')
            draw2 = ImageDraw.Draw(img2)
            
            # Simulate handwritten text with slightly irregular positioning
            handwritten_text = [
                "Handwritten Text Sample",
                "This simulates handwritten content",
                "OCR Challenge: Mixed Styles"
            ]
            
            y_pos = 80
            for i, text in enumerate(handwritten_text):
                x_offset = 60 + (i * 10)  # Slight offset to simulate handwriting
                draw2.text((x_offset, y_pos), text, fill='black', font=font)
                y_pos += 60
            
            img2.save(sample_dir / "sample_handwritten.jpg", "JPEG")
            
            print("✅ Created sample test images")
            return True
            
        except Exception as e:
            print(f"❌ Failed to create sample images: {str(e)}")
            return False
    
    def test_imports(self):
        """Test if all required modules can be imported"""
        self.print_step("Testing module imports")
        
        test_modules = [
            ("torch", "PyTorch"),
            ("transformers", "Transformers"),
            ("cv2", "OpenCV"),
            ("PIL", "Pillow"),
            ("numpy", "NumPy"),
            ("tesseract", "Tesseract"),
            ("easyocr", "EasyOCR"),
            ("fastapi", "FastAPI")
        ]
        
        success_count = 0
        failed_imports = []
        
        for module, name in test_modules:
            try:
                __import__(module)
                print(f"✅ {name} - Import successful")
                success_count += 1
            except ImportError as e:
                print(f"❌ {name} - Import failed: {str(e)}")
                failed_imports.append(name)
        
        self.test_results['imports'] = {
            'success': success_count,
            'total': len(test_modules),
            'failed': failed_imports
        }
        
        return success_count == len(test_modules)
    
    def test_core_components(self):
        """Test core OCR system components"""
        self.print_step("Testing core components")
        
        try:
            # Test configuration loading
            sys.path.insert(0, str(self.root_dir))
            
            from src.utils.config import OCRConfig
            from src.core.base_engine import OCREngine
            from src.engines.tesseract_engine import TesseractEngine
            
            print("✅ Core components imported successfully")
            
            # Test basic OCR engine
            engine = TesseractEngine()
            print("✅ TesseractEngine instantiated successfully")
            
            self.test_results['core_components'] = True
            return True
            
        except Exception as e:
            print(f"❌ Core component test failed: {str(e)}")
            self.test_results['core_components'] = False
            return False
    
    def test_sample_ocr(self):
        """Test OCR on sample images"""
        self.print_step("Testing OCR on sample images")
        
        try:
            from src.engines.tesseract_engine import TesseractEngine
            from PIL import Image
            
            sample_dir = self.root_dir / "data/sample_images"
            sample_image = sample_dir / "sample_printed.jpg"
            
            if not sample_image.exists():
                print("❌ Sample image not found")
                return False
            
            # Test basic OCR
            engine = TesseractEngine()
            image = Image.open(sample_image)
            
            result = engine.process_image(image)
            
            if result and result.text.strip():
                print(f"✅ OCR successful! Extracted text:")
                print(f"   '{result.text[:100]}...'")
                self.test_results['sample_ocr'] = True
                return True
            else:
                print("❌ OCR returned empty result")
                self.test_results['sample_ocr'] = False
                return False
                
        except Exception as e:
            print(f"❌ Sample OCR test failed: {str(e)}")
            self.test_results['sample_ocr'] = False
            return False
    
    def run_full_pipeline_test(self):
        """Test the complete OCR pipeline"""
        self.print_step("Testing complete OCR pipeline")
        
        try:
            # This would test your complete pipeline
            # For now, we'll do a simplified test
            from src.api.simple_ocr_api import SimpleOCRAPI
            
            api = SimpleOCRAPI()
            sample_image = self.root_dir / "data/sample_images/sample_printed.jpg"
            
            if sample_image.exists():
                result = api.process_image_file(str(sample_image))
                
                if result:
                    print("✅ Full pipeline test successful!")
                    print(f"   Confidence: {result.confidence:.2f}")
                    print(f"   Text length: {len(result.text)} characters")
                    self.test_results['full_pipeline'] = True
                    return True
                    
        except Exception as e:
            print(f"❌ Full pipeline test failed: {str(e)}")
            print("ℹ️  This is expected if all components aren't fully set up yet")
            
        self.test_results['full_pipeline'] = False
        return False
    
    def generate_report(self):
        """Generate setup and test report"""
        self.print_header("SETUP AND TEST REPORT")
        
        print("\n📊 Test Results Summary:")
        print("-" * 40)
        
        total_tests = 0
        passed_tests = 0
        
        for test_name, result in self.test_results.items():
            if isinstance(result, bool):
                status = "✅ PASS" if result else "❌ FAIL"
                print(f"{test_name:<20} {status}")
                total_tests += 1
                if result:
                    passed_tests += 1
            elif isinstance(result, dict) and 'success' in result:
                success_rate = f"{result['success']}/{result['total']}"
                status = "✅ PASS" if result['success'] == result['total'] else "⚠️  PARTIAL"
                print(f"{test_name:<20} {status} ({success_rate})")
                total_tests += 1
                if result['success'] == result['total']:
                    passed_tests += 1
        
        print("-" * 40)
        print(f"Overall: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("\n🎉 ALL TESTS PASSED! Your OCR system is ready to use!")
        elif passed_tests >= total_tests * 0.7:
            print(f"\n✅ Most tests passed ({passed_tests}/{total_tests}). System is mostly functional.")
        else:
            print(f"\n⚠️  Several tests failed ({total_tests - passed_tests}/{total_tests}). Check the issues above.")
        
        # Save detailed report
        report_file = self.root_dir / "setup_report.json"
        with open(report_file, 'w') as f:
            json.dump({
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'test_results': self.test_results,
                'summary': {
                    'total_tests': total_tests,
                    'passed_tests': passed_tests,
                    'success_rate': passed_tests / total_tests if total_tests > 0 else 0
                }
            }, f, indent=2)
        
        print(f"\n📄 Detailed report saved to: {report_file}")
    
    def run_setup(self):
        """Run the complete setup process"""
        self.print_header("ADVANCED OCR SYSTEM - SETUP & TEST")
        print("Setting up your state-of-the-art OCR system...")
        print("This process will:")
        print("• Check system requirements")
        print("• Set up virtual environment")
        print("• Install dependencies") 
        print("• Download models")
        print("• Run comprehensive tests")
        
        setup_steps = [
            ("System Check", self.check_python_version),
            ("Virtual Environment", self.setup_virtual_environment),
            ("Dependencies", self.install_dependencies),
            ("Directory Structure", self.create_directories),
            ("Model Download", self.download_models),
            ("Sample Images", self.create_sample_images),
            ("Import Tests", self.test_imports),
            ("Core Components", self.test_core_components),
            ("Sample OCR", self.test_sample_ocr),
            ("Full Pipeline", self.run_full_pipeline_test)
        ]
        
        print(f"\n🚀 Starting setup process with {len(setup_steps)} steps...")
        
        for step_name, step_func in setup_steps:
            self.print_header(f"STEP: {step_name}")
            
            try:
                success = step_func()
                if success:
                    print(f"✅ {step_name} completed successfully")
                else:
                    print(f"⚠️  {step_name} completed with issues")
            except Exception as e:
                print(f"❌ {step_name} failed with error: {str(e)}")
        
        # Generate final report
        self.generate_report()
        
        print(f"\n🎯 Next Steps:")
        print("1. Review the setup report above")
        print("2. If successful, try: python examples/test_mixed_text.py")
        print("3. Start the API: python -m src.api.simple_ocr_api")
        print("4. Check the documentation in README.md")

def main():
    """Main function"""
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("Advanced OCR System Setup & Test Script")
        print("\nUsage:")
        print("  python setup_and_test.py          # Run full setup")
        print("  python setup_and_test.py --help   # Show this help")
        return
    
    setup = OCRSystemSetup()
    setup.run_setup()

if __name__ == "__main__":
    main()