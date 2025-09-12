"""
Advanced OCR System - Package Setup Configuration
=================================================

Production-grade Python package setup for the Advanced OCR System.
Supports pip installation, development mode, and distribution.

Author: Production OCR Team  
Version: 2.0.0
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
requirements = []
requirements_file = this_directory / "requirements.txt"
if requirements_file.exists():
    requirements = requirements_file.read_text().strip().split('\n')
    requirements = [req.strip() for req in requirements if req.strip() and not req.startswith('#')]

setup(
    # Basic package information
    name="advanced-ocr-system",
    version="2.0.0",
    description="Production-grade OCR system with AI-powered preprocessing and multi-engine support",
    long_description=long_description,
    long_description_content_type="text/markdown",
    
    # Author information
    author="Production OCR Team",
    author_email="ocr-team@company.com",
    url="https://github.com/company/advanced-ocr-system",
    
    # Package structure
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    
    # Include additional files
    include_package_data=True,
    package_data={
        "advanced_ocr": [
            "data/configs/*.yaml",
            "data/configs/*.yml",
        ],
    },
    
    # Dependencies
    install_requires=requirements,
    
    # Optional dependencies for different use cases
    extras_require={
        "gpu": [
            "torch>=1.9.0",
            "torchvision>=0.10.0",
            "onnxruntime-gpu>=1.12.0",
        ],
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.12.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
            "pre-commit>=2.15.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0",
            "myst-parser>=0.15.0",
        ],
        "all": [
            "torch>=1.9.0",
            "torchvision>=0.10.0",
            "onnxruntime-gpu>=1.12.0",
            "pytest>=6.0.0",
            "pytest-cov>=2.12.0",
            "black>=21.0.0",
            "flake8>=3.9.0",
            "mypy>=0.910",
            "pre-commit>=2.15.0",
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=0.5.0",
            "myst-parser>=0.15.0",
        ],
    },
    
    # Python version requirement
    python_requires=">=3.8",
    
    # Classification
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Text Processing :: Linguistic",
    ],
    
    # Keywords for PyPI
    keywords=[
        "ocr", "optical character recognition", "computer vision", 
        "text extraction", "document processing", "ai", "machine learning",
        "image processing", "tesseract", "paddleocr", "easyocr", "trocr"
    ],
    
    # Entry points for command-line tools
    entry_points={
        "console_scripts": [
            "advanced-ocr=advanced_ocr.cli:main",
        ],
    },
    
    # Project URLs
    project_urls={
        "Bug Reports": "https://github.com/company/advanced-ocr-system/issues",
        "Source": "https://github.com/company/advanced-ocr-system",
        "Documentation": "https://advanced-ocr-system.readthedocs.io/",
    },
    
    # Zip safety
    zip_safe=False,
)