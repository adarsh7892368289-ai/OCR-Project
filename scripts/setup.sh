#!/bin/bash
# scripts/setup.sh - Enhanced OCR System Setup

set -e

echo "🚀 Setting up Enhanced OCR System with Deep Learning Support"
echo "============================================================"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check Python version
check_python() {
    print_status "Checking Python version..."
    
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
        PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
        PYTHON_CMD="python"
    else
        print_error "Python not found! Please install Python 3.8 or higher."
        exit 1
    fi
    
    print_success "Found Python $PYTHON_VERSION"
    
    # Check if version is >= 3.8
    if [[ $(echo "$PYTHON_VERSION" | cut -d. -f1-2 | tr -d .) -lt 38 ]]; then
        print_error "Python 3.8 or higher is required. Current version: $PYTHON_VERSION"
        exit 1
    fi
}

# Create virtual environment
create_venv() {
    print_status "Creating virtual environment..."
    
    if [ ! -d "venv" ]; then
        $PYTHON_CMD -m venv venv
        print_success "Virtual environment created"
    else
        print_warning "Virtual environment already exists"
    fi
}

# Activate virtual environment
activate_venv() {
    print_status "Activating virtual environment..."
    
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
        print_success "Virtual environment activated"
    elif [ -f "venv/Scripts/activate" ]; then
        source venv/Scripts/activate
        print_success "Virtual environment activated (Windows)"
    else
        print_error "Failed to activate virtual environment"
        exit 1
    fi
}

# Install system dependencies
install_system_deps() {
    print_status "Installing system dependencies..."
    
    # Detect OS
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Ubuntu/Debian
        if command -v apt-get &> /dev/null; then
            sudo apt-get update
            sudo apt-get install -y \
                tesseract-ocr \
                tesseract-ocr-eng \
                libtesseract-dev \
                libleptonica-dev \
                pkg-config \
                libgl1-mesa-glx \
                libglib2.0-0 \
                libsm6 \
                libxext6 \
                libxrender-dev \
                libgomp1 \
                libgcc-s1
        # CentOS/RHEL/Fedora
        elif command -v yum &> /dev/null; then
            sudo yum install -y \
                tesseract \
                tesseract-langpack-eng \
                tesseract-devel \
                leptonica-devel \
                pkgconfig \
                mesa-libGL \
                glib2 \
                libSM \
                libXext \
                libXrender \
                libgomp
        fi
        print_success "System dependencies installed (Linux)"
        
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if command -v brew &> /dev/null; then
            brew install tesseract
            print_success "System dependencies installed (macOS)"
        else
            print_warning "Homebrew not found. Please install Tesseract manually."
        fi
        
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
        print_warning "Windows detected. Please install Tesseract manually from: https://github.com/UB-Mannheim/tesseract/wiki"
    fi
}

# Install Python dependencies
install_python_deps() {
    print_status "Installing Python dependencies..."
    
    # Upgrade pip
    pip install --upgrade pip wheel setuptools
    
    # Install PyTorch with CUDA support if available
    if command -v nvidia-smi &> /dev/null; then
        print_status "CUDA detected, installing PyTorch with CUDA support..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    else
        print_status "Installing PyTorch CPU version..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
    
    # Install other requirements
    pip install -r requirements.txt
    
    print_success "Python dependencies installed"
}

# Create directory structure
create_directories() {
    print_status "Creating directory structure..."
    
    directories=(
        "data/sample_images"
        "data/models"
        "data/configs"
        "data/cache"
        "logs"
        "models/craft"
        "models/east"
        "debug/text_detection"
        "benchmark"
        "profiles"
        "exports"
    )
    
    for dir in "${directories[@]}"; do
        mkdir -p "$dir"
    done
    
    print_success "Directory structure created"
}

# Download models
download_models() {
    print_status "Downloading pre-trained models..."
    
    # Create models directory
    mkdir -p models/craft
    mkdir -p models/east
    
    # Download CRAFT model (placeholder - replace with actual download)
    if [ ! -f "models/craft/craft_mlt_25k.pth" ]; then
        print_status "Downloading CRAFT model..."
        # This would be the actual download command:
        # wget -O models/craft/craft_mlt_25k.pth https://github.com/clovaai/CRAFT-pytorch/releases/download/v1.0/craft_mlt_25k.pth
        print_warning "CRAFT model download placeholder - please download manually"
    fi
    
    # Download EAST model (placeholder)
    if [ ! -f "models/east/frozen_east_text_detection.pb" ]; then
        print_status "Downloading EAST model..."
        # This would be the actual download command:
        # wget -O models/east/frozen_east_text_detection.pb https://github.com/opencv/opencv_extra/raw/master/testdata/dnn/frozen_east_text_detection.pb
        print_warning "EAST model download placeholder - please download manually"
    fi
    
    print_success "Model download setup completed"
}

# Set up configuration files
setup_configs() {
    print_status "Setting up configuration files..."
    
    # Copy configuration templates
    if [ ! -f "data/configs/ocr_config.yaml" ]; then
        cat > data/configs/ocr_config.yaml << 'EOF'
# OCR System Configuration
engines:
  tesseract:
    enabled: true
    psm: 6
    oem: 3
    lang: "eng"
  
  paddleocr:
    enabled: true
    lang: "en"
    use_gpu: true
    
  trocr:
    enabled: true
    model_name: "microsoft/trocr-base-printed"
    device: "auto"

detection:
  method: "auto"
  confidence_threshold: 0.5
  
preprocessing:
  enhancement_level: "medium"
  enable_skew_correction: true
  
postprocessing:
  min_confidence: 0.3
  enable_spell_check: true
  
system:
  parallel_processing: true
  max_workers: 4
  log_level: "INFO"
EOF
        print_success "Configuration file created"
    fi
}

# Run tests
run_tests() {
    print_status "Running system tests..."
    
    # Basic import test
    python3 -c "
import sys
try:
    import torch
    print('✓ PyTorch imported successfully')
    print(f'  CUDA available: {torch.cuda.is_available()}')
except ImportError as e:
    print('✗ PyTorch import failed:', e)
    sys.exit(1)

try:
    import cv2
    print('✓ OpenCV imported successfully')
except ImportError as e:
    print('✗ OpenCV import failed:', e)
    sys.exit(1)

try:
    import transformers
    print('✓ Transformers imported successfully')
except ImportError as e:
    print('✗ Transformers import failed:', e)
    sys.exit(1)

print('✓ All core dependencies available')
"
    
    print_success "System tests completed"
}

# Display final instructions
show_instructions() {
    echo ""
    echo "🎉 Setup completed successfully!"
    echo "==============================="
    echo ""
    echo "Next steps:"
    echo "1. Activate the virtual environment:"
    echo "   source venv/bin/activate  # Linux/Mac"
    echo "   venv\\Scripts\\activate     # Windows"
    echo ""
    echo "2. Test the installation:"
    echo "   python -c 'from src.preprocessing.text_detector import AdvancedTextDetector; print(\"✓ Advanced text detection ready!\")'"
    echo ""
    echo "3. Download models manually if needed:"
    echo "   - CRAFT: https://github.com/clovaai/CRAFT-pytorch"
    echo "   - EAST: https://github.com/opencv/opencv_extra"
    echo ""
    echo "4. Run your first detection:"
    echo "   python scripts/test_detection.py"
    echo ""
    echo "Configuration files are in: data/configs/"
    echo "Models should be placed in: models/"
    echo "Logs will be saved to: logs/"
    echo ""
    print_success "Enhanced OCR System is ready to use!"
}

# Main setup function
main() {
    echo "Starting setup process..."
    
    check_python
    create_venv
    activate_venv
    install_system_deps
    install_python_deps
    create_directories
    download_models
    setup_configs
    run_tests
    show_instructions
}

# Run setup if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi