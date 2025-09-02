import time
import logging
from pathlib import Path
from typing import Dict, Any

# Import all engine classes
from src.engines.easyocr_engine import EasyOCREngine
from src.engines.tesseract_engine import TesseractEngine
from src.engines.paddle_engine import PaddleOCREngine
from src.engines.trocr_engine import TrOCREngine

# Import core utilities
from src.utils.image_utils import get_image_quality
from src.utils.text_utils import (
    analyze_and_consolidate, 
    get_engine_comparison, 
    correct_ocr_errors,
    extract_structured_data,
    perform_text_analysis
)

logger = logging.getLogger(__name__)

class OCRProcessor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.enabled_engines = self.config.get('engines', [])
        self.engines = {}

        self._initialize_engines()

    def _initialize_engines(self):
        """Initializes all enabled OCR engines."""
        if 'easyocr' in self.enabled_engines:
            self.engines['easyocr'] = EasyOCREngine()
        if 'tesseract' in self.enabled_engines:
            self.engines['tesseract'] = TesseractEngine()
        if 'paddle' in self.enabled_engines:
            self.engines['paddle'] = PaddleOCREngine()
        if 'trocr' in self.enabled_engines:
            self.engines['trocr'] = TrOCREngine()
        
        # Log which engines were successfully initialized
        for name in self.engines:
            logger.info(f"{name.capitalize()} engine initialized successfully.")

    def process_image(self, image_path: str, verbose: bool = False) -> Dict[str, Any]:
        """
        Main image processing function - orchestrates all OCR engines
        """
        start_time = time.time()
        
        if verbose:
            print(f"\n🖼️ Processing: {Path(image_path).name}")
            print("=" * 60)
        
        all_results = []
        engine_performance = {}
        text_by_engine = {}

        try:
            # Analyze image to determine text type (printed vs. handwritten)
            image_quality = get_image_quality(image_path)
            
            # Smartly route to specialized engines based on image quality/type
            if 'trocr' in self.engines and image_quality['is_handwritten']:
                # Prioritize TrOCR for handwritten text
                engines_to_run = ['trocr', 'easyocr', 'tesseract']
            else:
                # Prioritize other engines for printed text
                engines_to_run = ['easyocr', 'paddle', 'tesseract']

            engines_to_run = [e for e in engines_to_run if e in self.engines]

            for engine_name in engines_to_run:
                engine = self.engines[engine_name]
                if verbose:
                    print(f"\n🚀 Running {engine_name.upper()}...")
                
                engine_start = time.time()
                try:
                    # Each engine wrapper handles its own specific preprocessing
                    results = engine.extract_text(image_path)
                    
                    engine_time = time.time() - engine_start
                    engine_performance[engine_name] = {
                        'processing_time': engine_time,
                        'regions_found': len(results),
                        'success': True
                    }
                    
                    combined_text = ' '.join([res['text'] for res in results])
                    text_by_engine[engine_name] = combined_text
                    
                    for result in results:
                        result['engine'] = engine_name.upper()
                    
                    all_results.extend(results)
                    
                    if verbose:
                        print(f"   ✅ Found {len(results)} text regions in {engine_time:.2f}s")
                        
                except Exception as e:
                    engine_performance[engine_name] = {
                        'processing_time': time.time() - engine_start,
                        'regions_found': 0,
                        'success': False,
                        'error': str(e)
                    }
                    
                    if verbose:
                        print(f"   ❌ Failed: {str(e)}")
                    
                    logger.warning(f"{engine_name} failed: {e}")
            
            final_results = analyze_and_consolidate(all_results, text_by_engine, engine_performance)
            
            final_results['processing_metadata'] = {
                'image_path': image_path,
                'total_processing_time': time.time() - start_time,
                'engines_used': list(engines_to_run),
                'engine_performance': engine_performance,
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
            }
            
            return final_results
            
        except Exception as e:
            error_result = {
                'error': str(e),
                'processing_metadata': {
                    'image_path': image_path,
                    'total_processing_time': time.time() - start_time,
                    'engines_used': list(self.engines.keys()),
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
                }
            }
            return error_result