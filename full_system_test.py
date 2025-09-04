# full_system_test.py - Test your COMPLETE advanced OCR system

import sys
import os
from pathlib import Path
import time
import json
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_full_advanced_system(image_path):
    """Test using your complete advanced OCR system"""
    print(f"Testing FULL Advanced OCR System on: {Path(image_path).name}")
    print("=" * 70)
    
    try:
        # Import your advanced components
        from src.core.engine_manager import OCREngineManager
        from src.preprocessing.adaptive_processor import AdaptivePreprocessor
        from src.postprocessing.postprocessing_pipeline import PostprocessingPipeline
        from src.utils.config import ConfigManager
        
        # Initialize configuration
        config_manager = ConfigManager()
        config = config_manager.get_config()
        
        # Initialize your advanced components
        print("Initializing advanced OCR components...")
        
        # 1. Adaptive Preprocessor (uses your smart preprocessing)
        preprocessor = AdaptivePreprocessor(config.get('preprocessing', {}))
        
        # 2. Engine Manager (coordinates multiple engines)
        engine_manager = OCREngineManager(config.get('engines', {}))
        
        # 3. Post-processing Pipeline (your advanced text correction)
        postprocessor = PostprocessingPipeline(config.get('postprocessing', {}))
        
        # Load and analyze image
        image = Image.open(image_path)
        print(f"Image loaded: {image.size} pixels, {image.mode} mode")
        
        # STEP 1: Advanced Preprocessing
        print("\n1. Running Advanced Preprocessing...")
        start_time = time.time()
        preprocessed_results = preprocessor.process_image(image)
        preprocess_time = time.time() - start_time
        print(f"   Preprocessing completed in {preprocess_time:.2f}s")
        print(f"   Strategy used: {preprocessed_results.get('strategy', 'unknown')}")
        
        # STEP 2: Multi-Engine OCR Processing
        print("\n2. Running Multi-Engine OCR...")
        start_time = time.time()
        
        # Process with multiple engines through your engine manager
        ocr_results = engine_manager.process_image_multi_engine(
            preprocessed_results.get('processed_image', image)
        )
        
        ocr_time = time.time() - start_time
        print(f"   OCR processing completed in {ocr_time:.2f}s")
        print(f"   Engines used: {[result.engine_name for result in ocr_results]}")
        
        # STEP 3: Advanced Post-processing
        print("\n3. Running Advanced Post-processing...")
        start_time = time.time()
        
        final_result = postprocessor.process_results(ocr_results)
        
        postprocess_time = time.time() - start_time
        print(f"   Post-processing completed in {postprocess_time:.2f}s")
        
        # STEP 4: Display Results
        print("\n" + "="*70)
        print("ADVANCED OCR SYSTEM RESULTS")
        print("="*70)
        
        print(f"Overall Confidence: {final_result.confidence_score:.1f}%")
        print(f"Total Processing Time: {preprocess_time + ocr_time + postprocess_time:.2f}s")
        print(f"Text Quality Score: {final_result.get('quality_score', 'N/A')}")
        
        print(f"\nExtracted Text:")
        print("-" * 50)
        print(final_result.full_text)
        print("-" * 50)
        
        # Show advanced features
        if hasattr(final_result, 'document_structure'):
            print(f"\nDocument Structure Detected:")
            print(f"  Lines: {len(final_result.document_structure.lines)}")
            print(f"  Paragraphs: {len(final_result.document_structure.paragraphs)}")
        
        if hasattr(final_result, 'text_regions'):
            print(f"  Text Regions: {len(final_result.text_regions)}")
        
        # Show correction statistics
        corrections = final_result.get('corrections_applied', [])
        if corrections:
            print(f"\nText Corrections Applied: {len(corrections)}")
            for correction in corrections[:5]:  # Show first 5
                print(f"  '{correction.get('original', '')}' -> '{correction.get('corrected', '')}'")
        
        return final_result
        
    except ImportError as e:
        print(f"❌ Missing component: {e}")
        print("This indicates some advanced components need to be properly set up.")
        return None
    except Exception as e:
        print(f"❌ Error in advanced system: {e}")
        import traceback
        traceback.print_exc()
        return None

def compare_basic_vs_advanced(image_path):
    """Compare basic Tesseract vs your advanced system"""
    print("COMPARISON: Basic Tesseract vs Advanced OCR System")
    print("="*70)
    
    # Basic Tesseract test
    print("\n📄 BASIC TESSERACT:")
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = r'C:\Users\adbm\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
    
    start_time = time.time()
    basic_result = pytesseract.image_to_string(Image.open(image_path))
    basic_time = time.time() - start_time
    
    print(f"Time: {basic_time:.2f}s")
    print(f"Text: {basic_result.strip()[:100]}...")
    
    # Advanced system test
    print(f"\n🚀 ADVANCED OCR SYSTEM:")
    advanced_result = test_full_advanced_system(image_path)
    
    if advanced_result:
        print(f"\n📊 COMPARISON SUMMARY:")
        print(f"Basic Tesseract: {len(basic_result.split())} words")
        print(f"Advanced System: {len(advanced_result.full_text.split())} words")
        print(f"Confidence improvement: {advanced_result.confidence_score:.1f}%")

def test_with_trocr_for_mixed_content(image_path):
    """Specifically test TrOCR for mixed handwritten/printed content"""
    print(f"\n🤖 Testing TrOCR for Mixed Content")
    print("-" * 50)
    
    try:
        from src.engines.trocr_engine import TrOCREngine
        
        # Test both printed and handwritten TrOCR models
        models_to_test = [
            "microsoft/trocr-base-printed",
            "microsoft/trocr-base-handwritten",
            "microsoft/trocr-large-printed"
        ]
        
        for model_name in models_to_test:
            try:
                print(f"\nTesting {model_name}...")
                
                engine = TrOCREngine({
                    'model_name': model_name,
                    'device': 'cpu'  # Use CPU for compatibility
                })
                
                if engine.initialize():
                    result = engine.process_image(Image.open(image_path))
                    print(f"✅ Success: {result.full_text[:100]}...")
                else:
                    print("❌ Failed to initialize")
                    
            except Exception as e:
                print(f"❌ Error: {e}")
                
    except ImportError:
        print("❌ TrOCR engine not available")

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python full_system_test.py <image_path>")
        print("  python full_system_test.py --compare <image_path>")
        print("  python full_system_test.py --trocr <image_path>")
        return
    
    if sys.argv[1] == "--compare":
        compare_basic_vs_advanced(sys.argv[2])
    elif sys.argv[1] == "--trocr":
        test_with_trocr_for_mixed_content(sys.argv[2])
    else:
        test_full_advanced_system(sys.argv[1])

if __name__ == "__main__":
    main()