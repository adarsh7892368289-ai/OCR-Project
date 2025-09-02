# scripts/setup.sh

#!/bin/bash

set -e

echo "Setting up Modern OCR System..."

# Check Python version
python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "Error: Python $required_version or higher is required. Found: $python_version"
    exit 1
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install system dependencies (Ubuntu/Debian)
if command -v apt-get &> /dev/null; then
    echo "Installing system dependencies (Ubuntu/Debian)..."
    sudo apt-get update
    sudo apt-get install -y \
        tesseract-ocr \
        tesseract-ocr-eng \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1
fi

# Install Python dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating directories..."
mkdir -p data/{models,sample_images,configs}
mkdir -p logs
mkdir -p tests

# Download models
echo "Downloading OCR models..."
python -c "
try:
    import easyocr
    reader = easyocr.Reader(['en'])
    print('✓ EasyOCR models downloaded')
except Exception as e:
    print(f'✗ EasyOCR model download failed: {e}')

try:
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
    print('✓ TrOCR models downloaded')
except Exception as e:
    print(f'✗ TrOCR model download failed: {e}')
"

# Create default configuration
echo "Creating default configuration..."
cat > data/configs/default.yaml << EOF
engines:
  tesseract:
    psm: 6
    lang: "eng"
  easyocr:
    languages: ["en"]
    gpu: true
  trocr:
    model_name: "microsoft/trocr-base-handwritten"
    device: "auto"

preprocessing:
  enhancement_level: "medium"
  preserve_aspect_ratio: true

postprocessing:
  min_confidence: 0.5
  language: "en"

parallel_processing: true
max_workers: 3
log_level: "INFO"
EOF

# Create .env file
echo "Creating environment configuration..."
cat > .env << EOF
# OCR System Configuration
LOG_LEVEL=INFO
CONFIG_PATH=data/configs/default.yaml
MODEL_CACHE_DIR=data/models
CUDA_VISIBLE_DEVICES=0

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=1

# Security (change in production)
SECRET_KEY=your-secret-key-change-in-production
EOF

echo "✓ Setup completed successfully!"
echo ""
echo "To start the OCR system:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Run the API server: python -m uvicorn src.api.main:app --reload"
echo "3. Open http://localhost:8000/demo for web interface"
echo ""
echo "Or use Docker:"
echo "docker-compose -f docker/docker-compose.yml up -d"
