"""
OCR Pipeline Step-by-Step Testing Script
This script helps you test each component of your OCR pipeline individually
"""

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime

# Add your src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

class OCRPipelineTester:
    def __init__(self, debug_dir="debug_output"):
        self.debug_dir = Path(debug_dir)
        self.debug_dir.mkdir(exist_ok=True)
        self.current_step = 0
        
    def save_image(self, image, step_name, description=""):
        """Save image at each step with metadata"""
        self.current_step += 1
        filename = f"step_{self.current_step:02d}_{step_name}.jpg"
        filepath = self.debug_dir / filename
        
        # Save image
        cv2.imwrite(str(filepath), image)
        
        # Save metadata
        metadata = {
            "step": self.current_step,
            "name": step_name,
            "description": description,
            "timestamp": datetime.now().isoformat(),
            "image_shape": image.shape if hasattr(image, 'shape') else None
        }
        
        metadata_file = self.debug_dir / f"step_{self.current_step:02d}_{step_name}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        print(f"Step {self.current_step}: {step_name}")
        print(f"  Description: {description}")
        print(f"  Saved: {filepath}")
        print(f"  Image shape: {image.shape if hasattr(image, 'shape') else 'N/A'}")
        print("-" * 50)
        
        return image
    
    def display_image(self, image, title="Image", figsize=(10, 6)):
        """Display image with matplotlib"""
        plt.figure(figsize=figsize)
        if len(image.shape) == 3:
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(image, cmap='gray')
        plt.title(title)
        plt.axis('off')
        plt.show()

def test_step_1_image_loading(tester, image_path):
    """Step 1: Test basic image loading"""
    print("="*60)
    print("STEP 1: IMAGE LOADING")
    print("="*60)
    
    try:
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
            
        tester.save_image(image, "original_image", f"Original input image loaded from {image_path}")
        
        # Basic image info
        print(f"Image loaded successfully!")
        print(f"Image shape: {image.shape}")
        print(f"Image dtype: {image.dtype}")
        print(f"Image size: {image.shape[1]}x{image.shape[0]} pixels")
        
        return image
        
    except Exception as e:
        print(f"ERROR in Step 1: {e}")
        return None

def test_step_2_preprocessing_imports(tester):
    """Step 2: Test if preprocessing modules can be imported"""
    print("="*60)
    print("STEP 2: PREPROCESSING IMPORTS")
    print("="*60)
    
    modules_to_test = [
        'src.preprocessing.image_enhancer',
        'src.preprocessing.quality_analyzer', 
        'src.preprocessing.skew_corrector',
        'src.preprocessing.text_detector',
        'src.preprocessing.adaptive_processor'
    ]
    
    imported_modules = {}
    
    for module_name in modules_to_test:
        try:
            module = __import__(module_name, fromlist=[''])
            imported_modules[module_name] = module
            print(f"✓ Successfully imported: {module_name}")
        except Exception as e:
            print(f"✗ Failed to import {module_name}: {e}")
            imported_modules[module_name] = None
            
    return imported_modules

def test_step_3_quality_analysis(tester, image, modules):
    """Step 3: Test quality analysis"""
    print("="*60)
    print("STEP 3: QUALITY ANALYSIS")
    print("="*60)

    try:
        if 'src.preprocessing.quality_analyzer' not in modules or modules['src.preprocessing.quality_analyzer'] is None:
            print("Quality analyzer module not available, skipping...")
            return None

        # Try to use quality analyzer
        qa_module = modules['src.preprocessing.quality_analyzer']

        # Check if QualityAnalyzer class exists
        if hasattr(qa_module, 'QualityAnalyzer'):
            analyzer = qa_module.QualityAnalyzer()
            quality_metrics = analyzer.analyze_image(image)

            print("Quality Analysis Results:")
            for key, value in quality_metrics.items():
                print(f"  {key}: {value}")

            return quality_metrics
        else:
            print("QualityAnalyzer class not found in module")
            return None

    except Exception as e:
        print(f"ERROR in Quality Analysis: {e}")
        return None

def test_step_4_image_enhancement(tester, image, modules):
    """Step 4: Test image enhancement"""
    print("="*60)
    print("STEP 4: IMAGE ENHANCEMENT")
    print("="*60)
    
    try:
        if 'src.preprocessing.image_enhancer' not in modules or modules['src.preprocessing.image_enhancer'] is None:
            print("Image enhancer module not available, creating basic enhancement...")
            
            # Basic enhancement fallback
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            enhanced = cv2.bilateralFilter(gray, 9, 75, 75)
            enhanced = cv2.convertScaleAbs(enhanced, alpha=1.2, beta=10)
            
            tester.save_image(enhanced, "basic_enhanced", "Basic enhancement using OpenCV")
            return enhanced
            
        # Try to use image enhancer
        ie_module = modules['src.preprocessing.image_enhancer']
        
        if hasattr(ie_module, 'ImageEnhancer'):
            enhancer = ie_module.ImageEnhancer()
            enhanced_result = enhancer.enhance_image(image)
            enhanced = enhanced_result.enhanced_image

            tester.save_image(enhanced, "enhanced_image", "Enhanced using ImageEnhancer class")
            return enhanced
        else:
            print("ImageEnhancer class not found, using basic enhancement")
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            enhanced = cv2.bilateralFilter(gray, 9, 75, 75)
            tester.save_image(enhanced, "basic_enhanced", "Basic enhancement fallback")
            return enhanced
            
    except Exception as e:
        print(f"ERROR in Image Enhancement: {e}")
        # Fallback enhancement
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            enhanced = cv2.bilateralFilter(gray, 9, 75, 75)
            tester.save_image(enhanced, "fallback_enhanced", f"Fallback enhancement due to error: {e}")
            return enhanced
        except:
            return image

def test_step_5_text_detection(tester, image, modules):
    """Step 5: Test text detection"""
    print("="*60)
    print("STEP 5: TEXT DETECTION")
    print("="*60)
    
    try:
        if 'src.preprocessing.text_detector' not in modules or modules['src.preprocessing.text_detector'] is None:
            print("Text detector module not available, creating basic detection...")
            
            # Basic text detection using OpenCV
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            
            # Simple contour-based text detection
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edged = cv2.Canny(blurred, 50, 200)
            
            # Find contours
            contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Draw bounding boxes
            detection_image = image.copy()
            text_regions = []
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w > 20 and h > 10:  # Filter small regions
                    cv2.rectangle(detection_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    text_regions.append({'x': x, 'y': y, 'w': w, 'h': h})
            
            tester.save_image(detection_image, "basic_text_detection", f"Basic text detection found {len(text_regions)} regions")
            print(f"Found {len(text_regions)} potential text regions")
            
            return text_regions, detection_image
            
        # Try to use text detector
        td_module = modules['src.preprocessing.text_detector']
        
        if hasattr(td_module, 'TextDetector'):
            detector = td_module.TextDetector()
            detection_result = detector.detect_text_regions(image)
            text_regions = detection_result['regions']
            detection_image = image.copy()  # For now, just return the original image

            tester.save_image(detection_image, "detected_text_regions", f"Text detection found {len(text_regions)} regions")
            print(f"Found {len(text_regions)} text regions using TextDetector")

            return text_regions, detection_image
        else:
            print("TextDetector class not found, using basic detection")
            return test_step_5_text_detection(tester, image, {})
            
    except Exception as e:
        print(f"ERROR in Text Detection: {e}")
        return [], image

def test_step_6_engine_imports(tester):
    """Step 6: Test OCR engine imports"""
    print("="*60)
    print("STEP 6: OCR ENGINE IMPORTS")
    print("="*60)
    
    engines_to_test = [
        'src.engines.tesseract_engine',
        'src.engines.easyocr_engine',
        'src.engines.paddleocr_engine',
        'src.engines.trocr_engine'
    ]
    
    imported_engines = {}
    
    for engine_name in engines_to_test:
        try:
            engine = __import__(engine_name, fromlist=[''])
            imported_engines[engine_name] = engine
            print(f"✓ Successfully imported: {engine_name}")
        except Exception as e:
            print(f"✗ Failed to import {engine_name}: {e}")
            imported_engines[engine_name] = None
            
    return imported_engines

def main():
    """Main testing function"""
    print("OCR PIPELINE STEP-BY-STEP TESTING")
    print("="*60)
    
    # Initialize tester
    tester = OCRPipelineTester()
    
    # Use the specific image
    image_path = "data/sample_images/img1.jpg"
    print(f"Testing with image: {image_path}")
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        print("Make sure you're running this script from the project root directory")
        return
    
    # Step 1: Load image
    image = test_step_1_image_loading(tester, image_path)
    if image is None:
        print("Cannot proceed without a valid image")
        return
    
    print("Continuing to Step 2...")

    # Step 2: Test preprocessing imports
    preprocessing_modules = test_step_2_preprocessing_imports(tester)

    print("Continuing to Step 3...")

    # Step 3: Quality analysis
    quality_metrics = test_step_3_quality_analysis(tester, image, preprocessing_modules)

    print("Continuing to Step 4...")

    # Step 4: Image enhancement
    enhanced_image = test_step_4_image_enhancement(tester, image, preprocessing_modules)

    print("Continuing to Step 5...")

    # Step 5: Text detection
    text_regions, detection_image = test_step_5_text_detection(tester, enhanced_image, preprocessing_modules)

    print("Continuing to Step 6...")
    
    # Step 6: Engine imports
    engines = test_step_6_engine_imports(tester)
    
    print("\n" + "="*60)
    print("TESTING COMPLETE!")
    print("="*60)
    print(f"All debug images and metadata saved to: {tester.debug_dir}")
    print("Check the debug_output folder for detailed results of each step.")

if __name__ == "__main__":
    main()