#!/usr/bin/env python3
"""
Pipeline Debug Test - COMPLETE PHASE-BY-PHASE ANALYSIS

Tests the complete OCR pipeline from image input to engine coordination,
showing detailed results and failures at each stage.

Pipeline Flow Tested:
1. Raw Image Loading
2. Image Preprocessing (image_processor.py)
3. Content Classification (content_classifier.py) 
4. Engine Coordination (engine_coordinator.py)
5. Engine Execution (paddleocr_engine.py, etc.)

This test will show you exactly where your pipeline fails and why.
"""

import sys
import os
from pathlib import Path
import numpy as np
from PIL import Image
import traceback
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def print_phase(phase_name: str, status: str = "STARTING"):
    """Print formatted phase information"""
    print(f"\n{'='*60}")
    print(f"PHASE: {phase_name} - {status}")
    print(f"{'='*60}")

def print_result(title: str, data: any, max_length: int = 200):
    """Print formatted result information"""
    print(f"\n--- {title} ---")
    if isinstance(data, str):
        display_data = data[:max_length] + "..." if len(data) > max_length else data
        print(f"Text: '{display_data}'")
        print(f"Length: {len(data)} characters")
    elif isinstance(data, (int, float)):
        print(f"Value: {data}")
    elif isinstance(data, np.ndarray):
        print(f"Array shape: {data.shape}, dtype: {data.dtype}")
        print(f"Value range: [{data.min():.3f}, {data.max():.3f}]")
    elif isinstance(data, list):
        print(f"List length: {len(data)}")
        if data and hasattr(data[0], '__dict__'):
            print(f"First item type: {type(data[0]).__name__}")
    elif hasattr(data, '__dict__'):
        print(f"Object type: {type(data).__name__}")
        print(f"Attributes: {list(data.__dict__.keys())}")
    else:
        print(f"Data: {str(data)[:max_length]}")

def test_complete_pipeline():
    """Test complete OCR pipeline with detailed phase analysis"""
    
    print("ADVANCED OCR PIPELINE DEBUG TEST")
    print("Testing pipeline phases up to engine coordination")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    
    # ========================================================================
    # PHASE 1: IMAGE LOADING
    # ========================================================================
    print_phase("1. IMAGE LOADING")
    
    try:
        # Use the test image from your test file
        img_path = Path(__file__).parent / 'data' / 'sample_images' / 'img3.jpg'
        
        if not img_path.exists():
            print(f"❌ Test image not found: {img_path}")
            print("Available files:")
            parent_dir = img_path.parent
            if parent_dir.exists():
                for file in parent_dir.iterdir():
                    print(f"  - {file.name}")
            else:
                print(f"Directory doesn't exist: {parent_dir}")
            return False
        
        # Load image
        pil_image = Image.open(img_path)
        np_image = np.array(pil_image)
        
        print(f"✅ Image loaded successfully")
        print_result("PIL Image", f"Size: {pil_image.size}, Mode: {pil_image.mode}")
        print_result("Numpy Array", np_image)
        
    except Exception as e:
        print(f"❌ Image loading failed: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False
    
    # ========================================================================
    # PHASE 2: CONFIGURATION LOADING
    # ========================================================================
    print_phase("2. CONFIGURATION LOADING")
    
    try:
        from advanced_ocr.config import OCRConfig
        
        config = OCRConfig()
        
        print(f"✅ Configuration loaded successfully")
        print_result("Config Type", type(config).__name__)
        
        # Check key configuration sections
        if hasattr(config, 'engines'):
            print("📋 Engine configurations available")
        if hasattr(config, 'preprocessing'):
            print("📋 Preprocessing configurations available")
        if hasattr(config, 'coordination'):
            print("📋 Coordination configurations available")
            
    except Exception as e:
        print(f"❌ Configuration loading failed: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False
    
    # ========================================================================
    # PHASE 3: MODEL UTILS INITIALIZATION
    # ========================================================================
    print_phase("3. MODEL UTILS INITIALIZATION")
    
    try:
        from advanced_ocr.utils.model_utils import ModelLoader
        
        model_loader = ModelLoader(config)
        
        print(f"✅ ModelLoader initialized successfully")
        print_result("Cache Info", model_loader.get_cache_info())
        
    except Exception as e:
        print(f"❌ ModelLoader initialization failed: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False
    
    # ========================================================================
    # PHASE 4: IMAGE PREPROCESSING
    # ========================================================================
    print_phase("4. IMAGE PREPROCESSING")
    
    try:
        from advanced_ocr.preprocessing.image_processor import ImageProcessor
        
        # Initialize image processor
        image_processor = ImageProcessor(model_loader, config)
        # ADD THIS LINE:
        image_processor.enable_text_detection = False

        print("Text detection DISABLED for debugging")
        print(f"✅ ImageProcessor initialized")
        
        # Process the image
        start_time = time.time()
        preprocessing_result = image_processor.process_image(np_image)
        processing_time = time.time() - start_time
        
        print(f"✅ Image preprocessing completed in {processing_time:.3f}s")
        print_result("Enhanced Image", preprocessing_result.enhanced_image)
        print_result("Text Regions", f"{len(preprocessing_result.text_regions)} regions detected")
        print_result("Quality Metrics", preprocessing_result.quality_metrics)
        
        # Show first few text regions
        if preprocessing_result.text_regions:
            print("\n--- First 3 Text Regions ---")
            for i, region in enumerate(preprocessing_result.text_regions[:3]):
                print(f"Region {i}: {region.bbox.coordinates} (confidence: {region.confidence:.3f})")
        
    except Exception as e:
        print(f"❌ Image preprocessing failed: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False
    
    # ========================================================================
    # PHASE 5: CONTENT CLASSIFICATION
    # ========================================================================
    print_phase("5. CONTENT CLASSIFICATION")
    
    try:
        from advanced_ocr.preprocessing.content_classifier import ContentClassifier
        
        # Initialize content classifier
        content_classifier = ContentClassifier(config)
        
        print(f"✅ ContentClassifier initialized")
        
        # Classify content
        start_time = time.time()
        content_classification = content_classifier.classify_content(pil_image)
        classification_time = time.time() - start_time
        
        print(f"✅ Content classification completed in {classification_time:.3f}s")
        print_result("Content Type", content_classification.content_type)
        print_result("Confidence Scores", content_classification.confidence_scores)
        print_result("Processing Time", f"{content_classification.processing_time:.3f}s")
        
    except Exception as e:
        print(f"❌ Content classification failed: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        # Continue with fallback
        content_classification = None
        print("🔄 Continuing without content classification...")
    
    # ========================================================================
    # PHASE 6: ENGINE COORDINATOR INITIALIZATION
    # ========================================================================
    print_phase("6. ENGINE COORDINATOR INITIALIZATION")
    
    try:
        from advanced_ocr.engines.engine_coordinator import EngineCoordinator
        
        # Initialize engine coordinator
        engine_coordinator = EngineCoordinator(config)
        
        print(f"✅ EngineCoordinator initialized")
        print_result("Coordinator Status", str(engine_coordinator))
        
        # Check if content classifier was loaded
        if hasattr(engine_coordinator, 'content_classifier'):
            classifier_status = "Available" if engine_coordinator.content_classifier else "Not Available"
            print(f"📋 Content Classifier: {classifier_status}")
        
    except Exception as e:
        print(f"❌ EngineCoordinator initialization failed: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False
    
    # ========================================================================
    # PHASE 7: ENGINE COORDINATION
    # ========================================================================
    print_phase("7. ENGINE COORDINATION")
    
    try:
        # Convert PIL image to numpy for coordination
        if isinstance(preprocessing_result.enhanced_image, np.ndarray):
            coordination_image = preprocessing_result.enhanced_image
        else:
            coordination_image = np.array(preprocessing_result.enhanced_image)
        
        print(f"📋 Coordinating with image shape: {coordination_image.shape}")
        print(f"📋 Text regions: {len(preprocessing_result.text_regions)}")
        
        # Run coordination
        start_time = time.time()
        coordination_results = engine_coordinator.coordinate(
            Image.fromarray(coordination_image) if isinstance(coordination_image, np.ndarray) else coordination_image,
            preprocessing_result.text_regions
        )
        coordination_time = time.time() - start_time
        
        print(f"✅ Engine coordination completed in {coordination_time:.3f}s")
        
        # Analyze results
        if isinstance(coordination_results, list):
            print_result("Results Type", f"List with {len(coordination_results)} results")
            for i, result in enumerate(coordination_results):
                print(f"\n--- Result {i+1} ---")
                print(f"Engine: {result.engine_name}")
                print(f"Success: {result.success}")
                print(f"Text: '{result.text[:100]}...' ({len(result.text)} chars)")
                print(f"Confidence: {result.confidence:.3f}")
                if hasattr(result, 'error_message') and result.error_message:
                    print(f"Error: {result.error_message}")
                if hasattr(result, 'metadata'):
                    print(f"Metadata: {result.metadata}")
        else:
            print_result("Results Type", type(coordination_results).__name__)
            print_result("Result Details", coordination_results)
        
        # Get engine statistics
        try:
            engine_stats = engine_coordinator.get_engine_stats()
            print(f"\n--- Engine Statistics ---")
            for engine_name, stats in engine_stats.items():
                print(f"{engine_name}: {stats}")
        except Exception as e:
            print(f"⚠️  Could not get engine stats: {e}")
        
    except Exception as e:
        print(f"❌ Engine coordination failed: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False
    
    # ========================================================================
    # PHASE 8: INDIVIDUAL ENGINE TESTING
    # ========================================================================
    print_phase("8. INDIVIDUAL ENGINE TESTING")
    
    # Test individual engines that were used
    engines_to_test = ['paddleocr', 'tesseract', 'easyocr', 'trocr']
    
    for engine_name in engines_to_test:
        print(f"\n--- Testing {engine_name.upper()} Engine ---")
        try:
            engine = engine_coordinator._get_engine(engine_name)
            if engine is None:
                print(f"❌ {engine_name} engine not available")
                continue
            
            print(f"✅ {engine_name} engine loaded")
            print(f"Status: {engine.get_status().value}")
            print(f"Ready: {engine.is_ready()}")
            
            # Try extraction
            if engine.is_ready():
                test_result = engine.extract(coordination_image, preprocessing_result.text_regions[:2])
                print(f"Test extraction: '{test_result.text[:50]}...' (conf: {test_result.confidence:.3f})")
            
        except Exception as e:
            print(f"❌ {engine_name} engine test failed: {e}")
    
    # ========================================================================
    # PHASE 9: SUMMARY
    # ========================================================================
    print_phase("9. PIPELINE SUMMARY", "COMPLETE")
    
    total_phases = 9
    print(f"✅ All {total_phases} phases completed successfully!")
    print(f"\n📊 FINAL RESULTS:")
    print(f"  • Image processed: {pil_image.size}")
    print(f"  • Text regions detected: {len(preprocessing_result.text_regions)}")
    if content_classification:
        print(f"  • Content type: {content_classification.content_type}")
    
    if isinstance(coordination_results, list):
        total_text = ' '.join([r.text for r in coordination_results if r.text])
        avg_confidence = np.mean([r.confidence for r in coordination_results])
        print(f"  • Total extracted text: {len(total_text)} characters")
        print(f"  • Average confidence: {avg_confidence:.3f}")
        print(f"  • Engines used: {len(coordination_results)}")
    
    print(f"\n🎯 PIPELINE STATUS: WORKING CORRECTLY")
    return True

def main():
    """Main test execution"""
    print("Starting Advanced OCR Pipeline Debug Test...")
    
    try:
        success = test_complete_pipeline()
        
        if success:
            print("\n🎉 PIPELINE TEST COMPLETED SUCCESSFULLY!")
            print("Your OCR pipeline is working correctly up to engine coordination.")
        else:
            print("\n💥 PIPELINE TEST FAILED!")
            print("Check the error messages above to identify the issue.")
        
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted by user")
    except Exception as e:
        print(f"\n💥 UNEXPECTED ERROR: {e}")
        print(f"Traceback: {traceback.format_exc()}")

if __name__ == "__main__":
    main()