# setup_project.py - Run this script to set up your OCR project

import os
import sys
from pathlib import Path
import subprocess

def create_directory_structure():
    """Create the complete directory structure"""
    print("🏗️ Creating directory structure...")
    
    directories = [
        "src/core",
        "src/engines", 
        "src/utils",
        "src/models",
        "data/input/test_images",
        "data/input/sample_images",
        "data/output/json_results", 
        "data/output/text_files",
        "data/output/annotated_images",
        "data/models",
        "tests/test_data",
        "notebooks",
        "scripts", 
        "docs",
        "config",
        "logs",
        ".vscode"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Directory structure created!")

def create_init_files():
    """Create all necessary __init__.py files"""
    print("📝 Creating __init__.py files...")
    
    init_files = [
        "src/__init__.py",
        "src/core/__init__.py", 
        "src/engines/__init__.py",
        "src/utils/__init__.py",
        "src/models/__init__.py",
        "tests/__init__.py"
    ]
    
    for init_file in init_files:
        Path(init_file).touch()
    
    print("✅ __init__.py files created!")

def create_requirements_file():
    """Create the requirements.txt file"""
    print("📋 Creating requirements.txt...")
    
    requirements = """# Core ML/AI Libraries
torch>=1.9.0
torchvision>=0.10.0  
transformers>=4.21.0
numpy>=1.21.0
opencv-python>=4.5.0
Pillow>=8.3.0

# OCR Engines
paddlepaddle>=2.4.0
paddleocr>=2.6.0
easyocr>=1.6.0
pytesseract>=0.3.10

# Data Processing & Configuration
pandas>=1.3.0
pydantic>=1.8.0
PyYAML>=6.0

# CLI & UI
rich>=10.0.0
click>=8.0.0
colorama>=0.4.4
tqdm>=4.62.0

# Development
pytest>=6.2.0
black>=21.0.0
flake8>=3.9.0
"""
    
    with open('requirements.txt', 'w') as f:
        f.write(requirements)
    
    print("✅ requirements.txt created!")

def create_config_file():
    """Create the development configuration file"""
    print("⚙️ Creating configuration file...")
    
    config_content = """# Advanced OCR System Configuration

# OCR Engine Configuration  
engines:
  paddle_ocr:
    enabled: true
    use_gpu: true
    language: 'en'
    use_angle_cls: true
    
  trocr:
    enabled: true
    model_name: 'microsoft/trocr-base-handwritten'
    use_gpu: true
    
  easyocr:
    enabled: true
    languages: ['en'] 
    use_gpu: true
    
  tesseract:
    enabled: false
    config: '--psm 6'

# Image Preprocessing Configuration
preprocessing:
  enhance_contrast: true
  denoise: true
  max_dimension: 2048
  dpi_threshold: 300

# Output Configuration
output:
  save_json: true
  save_annotated_images: false
  confidence_threshold: 0.5
  
# Performance Configuration
performance:
  batch_size: 1
  max_workers: 4
  gpu_memory_limit: 0.8
"""
    
    with open('config/development.yaml', 'w') as f:
        f.write(config_content)
    
    print("✅ Configuration file created!")

def create_vscode_settings():
    """Create VS Code settings"""
    print("🔧 Creating VS Code settings...")
    
    settings = """{
    "python.pythonPath": "./venv/Scripts/python.exe",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": false,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true,
    "python.testing.unittestEnabled": false,
    "python.testing.pytestArgs": ["tests"],
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true,
        ".pytest_cache": true,
        "*.egg-info": true
    }
}"""
    
    with open('.vscode/settings.json', 'w') as f:
        f.write(settings)
    
    launch_config = """{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: OCR Main",
            "type": "python",
            "request": "launch", 
            "program": "${workspaceFolder}/main.py",
            "args": ["--image", "data/input/sample_images/test.jpg"],
            "console": "integratedTerminal",
            "cwd": "${workspaceFolder}"
        },
        {
            "name": "Python: Current File",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal"
        }
    ]
}"""
    
    with open('.vscode/launch.json', 'w') as f:
        f.write(launch_config)
    
    print("✅ VS Code settings created!")

def create_env_file():
    """Create environment variables template"""
    print("🌍 Creating .env.example...")
    
    env_content = """# GPU Configuration
CUDA_VISIBLE_DEVICES=0
TORCH_HOME=./data/models/torch

# Model Cache Directories
TRANSFORMERS_CACHE=./data/models/transformers
HF_HOME=./data/models/huggingface

# Logging
LOG_LEVEL=INFO
LOG_FILE=./logs/ocr.log

# Performance
MAX_WORKERS=4
BATCH_SIZE=1
GPU_MEMORY_LIMIT=0.8

# API Keys (if needed for future extensions)
# GOOGLE_VISION_API_KEY=your_key_here
# AZURE_VISION_API_KEY=your_key_here
"""
    
    with open('.env.example', 'w') as f:
        f.write(env_content)
    
    print("✅ Environment template created!")

def create_gitignore():
    """Create .gitignore file"""
    print("📝 Creating .gitignore...")
    
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
env/
ENV/

# IDE
.vscode/settings.json
.idea/
*.swp
*.swo

# Data
data/input/test_images/*
data/input/sample_images/*
data/output/
*.jpg
*.jpeg
*.png
*.bmp
*.tiff
*.pdf

# Logs
logs/
*.log

# Models Cache
data/models/
.cache/

# Environment Variables
.env

# OS
.DS_Store
Thumbs.db

# Jupyter
.ipynb_checkpoints/

# Testing
.pytest_cache/
.coverage
htmlcov/

# Model files
*.bin
*.safetensors
*.onnx
"""
    
    with open('.gitignore', 'w') as f:
        f.write(gitignore_content)
    
    print("✅ .gitignore created!")

def create_readme():
    """Create README.md file"""
    print("📖 Creating README.md...")
    
    readme_content = """# 🤖 Advanced AI-Powered OCR System

A professional-grade OCR system that combines multiple AI engines for maximum accuracy on both printed and handwritten text.

## ✨ Features

- **Multi-Engine Fusion**: Combines PaddleOCR, TrOCR, EasyOCR, and Tesseract
- **Handwritten Text Specialist**: Uses Microsoft's TrOCR for handwritten content
- **Intelligent Analysis**: AI-powered confidence scoring and text classification
- **Smart Preprocessing**: Automatic image enhancement for better results
- **Professional Output**: Beautiful CLI with detailed analysis and recommendations

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Run OCR on an Image
```bash
# Basic usage
python main.py --image your_image.jpg

# With detailed analysis
python main.py --image document.png --save

# Specific engines only
python main.py --image text.jpg --engines paddle,trocr
```

### 3. Example Output
```
🎯 Extracted Text
┌─────────────────────────────────────┐
│ "Hello World! This is a test image" │  
└─────────────────────────────────────┘

📊 Text Analysis
┌─────────────────────┬─────────────┐
│ Text Type           │ General Text│
│ Word Count          │ 7           │
│ Contains Numbers    │ ✗           │
│ Engine Agreement    │ ✓ High      │
└─────────────────────┴─────────────┘
```

## 🛠️ System Requirements

- Python 3.8+
- Windows 10/11, Linux, or macOS
- Optional: NVIDIA GPU with CUDA for acceleration
- 4GB+ RAM (8GB+ recommended)

## 📁 Project Structure

```
advanced-ocr-system/
├── src/
│   ├── core/           # Main OCR engine
│   ├── engines/        # Individual OCR implementations  
│   ├── utils/          # Utilities and helpers
│   └── models/         # AI models and processors
├── data/
│   ├── input/          # Input images
│   └── output/         # Results and processed files
├── config/             # Configuration files
└── main.py            # Main application
```

## 🔧 Configuration

Edit `config/development.yaml` to customize:

- Enable/disable specific OCR engines
- Adjust preprocessing settings
- Configure GPU usage
- Set output preferences

## 🎯 Supported Features

### Image Formats
- JPEG, PNG, BMP, TIFF, WebP
- Any resolution (auto-resized for optimal processing)

### Text Types
- ✅ Printed text (documents, books, signs)
- ✅ Handwritten text (notes, forms)
- ✅ Mixed content (forms with both)
- ✅ Multi-language support
- ✅ Tables and structured data

### Output Formats
- Console display with rich formatting
- JSON files with detailed metadata
- Plain text files
- Annotated images (optional)

## 🤖 AI Engines

1. **PaddleOCR** - Excellent for printed text, fast processing
2. **TrOCR** - Microsoft's transformer model for handwritten text
3. **EasyOCR** - Good general-purpose OCR with multiple languages
4. **Tesseract** - Classic OCR engine, highly configurable

## 📈 Performance Tips

- Use GPU acceleration for 3-5x speed improvement
- Ensure good image quality (300+ DPI, good lighting)
- For handwritten text, TrOCR typically performs best
- For printed documents, PaddleOCR is usually fastest and most accurate

## 🐛 Troubleshooting

### Common Issues

**"No OCR engines could be initialized"**
- Check that all dependencies are installed
- Verify CUDA installation for GPU support

**Low confidence scores**
- Try preprocessing the image (sharpen, enhance contrast)
- Ensure text is clearly visible and well-lit
- Check image resolution (higher is usually better)

**Import errors**
- Activate your virtual environment
- Reinstall requirements: `pip install -r requirements.txt`

## 📄 License

MIT License - feel free to use in personal and commercial projects.

## 🤝 Contributing

Contributions welcome! Please feel free to submit pull requests or open issues.

---
Made with ❤️ using Python and AI
"""
    
    with open('README.md', 'w') as f:
        f.write(readme_content)
    
    print("✅ README.md created!")

def create_test_script():
    """Create a simple test script"""
    print("🧪 Creating test script...")
    
    test_content = """# test_ocr.py - Simple test script

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_imports():
    print("Testing imports...")
    
    try:
        from src.core.ocr_engine import AdvancedOCREngine
        print("✅ Core engine import successful")
    except Exception as e:
        print(f"❌ Core engine import failed: {e}")
        return False
    
    try:
        from src.utils.config import load_config
        print("✅ Config utils import successful")  
    except Exception as e:
        print(f"❌ Config utils import failed: {e}")
        return False
    
    try:
        from src.utils.logger import setup_logger
        print("✅ Logger utils import successful")
    except Exception as e:
        print(f"❌ Logger utils import failed: {e}")
        return False
    
    print("🎉 All imports successful!")
    return True

def test_config():
    print("\\nTesting configuration...")
    
    try:
        from src.utils.config import load_config
        config = load_config("config/development.yaml")
        print(f"✅ Config loaded with {len(config)} sections")
        return True
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False

def create_sample_image():
    print("\\nCreating sample test image...")
    
    try:
        import cv2
        import numpy as np
        
        # Create a simple image with text
        img = np.ones((200, 600, 3), dtype=np.uint8) * 255
        
        # Add text
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, 'Hello OCR World!', (50, 100), font, 2, (0, 0, 0), 3)
        cv2.putText(img, 'Testing 123', (50, 150), font, 1, (0, 0, 0), 2)
        
        # Save image
        output_path = "data/input/sample_images/test_sample.jpg"
        cv2.imwrite(output_path, img)
        print(f"✅ Test image created: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ Sample image creation failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Running OCR System Tests\\n")
    
    all_passed = True
    
    all_passed &= test_imports()
    all_passed &= test_config() 
    all_passed &= create_sample_image()
    
    if all_passed:
        print("\\n🎉 All tests passed! Your OCR system is ready to use.")
        print("\\nTry running: python main.py --image data/input/sample_images/test_sample.jpg")
    else:
        print("\\n❌ Some tests failed. Check the error messages above.")
"""
    
    with open('test_ocr.py', 'w') as f:
        f.write(test_content)
    
    print("✅ Test script created!")

def main():
    """Main setup function"""
    print("🚀 Setting up Advanced OCR System...")
    print("=" * 60)
    
    try:
        create_directory_structure()
        create_init_files()
        create_requirements_file()
        create_config_file()
        create_vscode_settings()
        create_env_file()
        create_gitignore()
        create_readme()
        create_test_script()
        
        print("\n" + "=" * 60)
        print("🎉 SETUP COMPLETE!")
        print("=" * 60)
        
        print("\n📋 Next Steps:")
        print("1. Create and activate virtual environment:")
        print("   python -m venv venv")
        print("   .\\venv\\Scripts\\activate")
        print("")
        print("2. Install dependencies:")
        print("   pip install -r requirements.txt")
        print("")
        print("3. Test the setup:")
        print("   python test_ocr.py")
        print("")
        print("4. Run OCR on an image:")
        print("   python main.py --image your_image.jpg")
        print("")
        print("5. Open in VS Code:")
        print("   code .")
        print("")
        print("🎯 Your advanced OCR system is ready!")
        
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main()