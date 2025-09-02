# src/core/ocr_engine.py - The Main OCR Processing Engine

import os
import cv2
import numpy as np
from PIL import Image
import torch
from typing import List, Dict, Any, Optional, Tuple
import logging
from pathlib import Path
import time
from datetime import datetime
import re

class AdvancedOCREngine:
    """
    Advanced Multi-Engine OCR System
    Combines multiple OCR engines for maximum accuracy on both printed and handwritten text
    """
    
    def __init__(self, config: Dict = None, use_gpu: bool = True, engines: List[str] = None, verbose: bool = False):
        """Initialize the Advanced OCR Engine"""
        self.config = config or {}
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.verbose = verbose
        self.device = "cuda" if self.use_gpu else "cpu"
        self.logger = logging.getLogger(__name__)
        
        # Default engines if none specified
        self.enabled_engines = engines or ['paddle', 'trocr', 'easyocr']
        
        # Initialize engine instances
        self.engines = {}
        self._initialize_engines()
    
    def _initialize_engines(self):
        """Initialize all enabled OCR engines"""
        if self.verbose:
            print(f"🚀 Initializing OCR engines on {self.device.upper()}...")
        
        # Initialize PaddleOCR
        if 'paddle' in self.enabled_engines:
            try:
                from src.engines.paddle_engine import PaddleEngine
                self.engines['paddle'] = PaddleEngine(
                    use_gpu=self.use_gpu,
                    config=self.config.get('engines', {}).get('paddle_ocr', {})
                )
                if self.verbose:
                    print("✅ PaddleOCR initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize PaddleOCR: {e}")
                if self.verbose:
                    print(f"⚠️ PaddleOCR initialization failed: {e}")
        
        # Initialize TrOCR (Handwritten Text Specialist)
        if 'trocr' in self.enabled_engines:
            try:
                from src.engines.trocr_engine import TrOCREngine
                self.engines['trocr'] = TrOCREngine(
                    use_gpu=self.use_gpu,
                    config=self.config.get('engines', {}).get('trocr', {})
                )
                if self.verbose:
                    print("✅ TrOCR (Handwritten Specialist) initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize TrOCR: {e}")
                if self.verbose:
                    print(f"⚠️ TrOCR initialization failed: {e}")
        
        # Initialize EasyOCR
        if 'easyocr' in self.enabled_engines:
            try:
                from src.engines.easyocr_engine import EasyOCREngine
                self.engines['easyocr'] = EasyOCREngine(
                    use_gpu=self.use_gpu,
                    config=self.config.get('engines', {}).get('easyocr', {})
                )
                if self.verbose:
                    print("✅ EasyOCR initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize EasyOCR: {e}")
                if self.verbose:
                    print(f"⚠️ EasyOCR initialization failed: {e}")
        
        # Initialize Tesseract (if enabled)
        if 'tesseract' in self.enabled_engines:
            try:
                from src.engines.tesseract_engine import TesseractEngine
                self.engines['tesseract'] = TesseractEngine(
                    config=self.config.get('engines', {}).get('tesseract', {})
                )
                if self.verbose:
                    print("✅ Tesseract OCR initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize Tesseract: {e}")
                if self.verbose:
                    print(f"⚠️ Tesseract initialization failed: {e}")
        
        if not self.engines:
            raise RuntimeError("No OCR engines could be initialized!")
        
        if self.verbose:
            print(f"🎯 Successfully initialized {len(self.engines)} OCR engines")
    
    def preprocess_image(self, image_path: str) -> Tuple[np.ndarray, Image.Image, Dict]:
        """
        Advanced image preprocessing for optimal OCR results
        Returns processed OpenCV image, PIL image, and metadata
        """
        try:
            # Load image
            cv_image = cv2.imread(image_path)
            pil_image = Image.open(image_path).convert('RGB')
            
            if cv_image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Get image metadata
            height, width = cv_image.shape[:2]
            file_size = os.path.getsize(image_path)
            
            metadata = {
                'original_size': (width, height),
                'file_size_mb': file_size / (1024 * 1024),
                'aspect_ratio': width / height,
                'preprocessing_applied': []
            }
            
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            
            # Analyze image quality
            blur_value = cv2.Laplacian(gray, cv2.CV_64F).var()
            brightness = np.mean(gray)
            contrast = np.std(gray)
            
            metadata.update({
                'blur_score': blur_value,
                'brightness': brightness,
                'contrast': contrast,
                'is_blurry': blur_value < 100,
                'is_dark': brightness < 80,
                'is_low_contrast': contrast < 50
            })
            
            # Apply preprocessing based on image analysis
            processed = gray.copy()
            
            # Enhance contrast if needed
            if contrast < 50:
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                processed = clahe.apply(processed)
                metadata['preprocessing_applied'].append('contrast_enhancement')
            
            # Brightness adjustment if needed
            if brightness < 80:
                processed = cv2.convertScaleAbs(processed, alpha=1.2, beta=30)
                metadata['preprocessing_applied'].append('brightness_adjustment')
            elif brightness > 200:
                processed = cv2.convertScaleAbs(processed, alpha=0.8, beta=-20)
                metadata['preprocessing_applied'].append('brightness_reduction')
            
            # Noise reduction for blurry images
            if blur_value < 100:
                processed = cv2.medianBlur(processed, 3)
                metadata['preprocessing_applied'].append('noise_reduction')
            
            # Resize if image is too large (optional)
            max_dimension = self.config.get('preprocessing', {}).get('max_dimension', 2048)
            if max(width, height) > max_dimension:
                scale = max_dimension / max(width, height)
                new_width = int(width * scale)
                new_height = int(height * scale)
                processed = cv2.resize(processed, (new_width, new_height), interpolation=cv2.INTER_AREA)
                pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                metadata['preprocessing_applied'].append(f'resize_to_{new_width}x{new_height}')
            
            # Convert back to BGR for engines that need it
            if len(processed.shape) == 2:  # If grayscale
                processed_bgr = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)
            else:
                processed_bgr = processed
            
            return processed_bgr, pil_image, metadata
            
        except Exception as e:
            raise ValueError(f"Image preprocessing failed: {str(e)}")
    
    def process_image(self, image_path: str) -> Dict[str, Any]:
        """
        Main image processing function - orchestrates all OCR engines
        """
        start_time = time.time()
        
        if self.verbose:
            print(f"\n🖼️ Processing: {Path(image_path).name}")
            print("=" * 60)
        
        try:
            # Preprocess image
            cv_image, pil_image, metadata = self.preprocess_image(image_path)
            
            if self.verbose:
                print(f"📐 Image: {metadata['original_size'][0]}x{metadata['original_size'][1]} pixels")
                print(f"📊 Quality: Blur={metadata['blur_score']:.0f}, Brightness={metadata['brightness']:.0f}, Contrast={metadata['contrast']:.0f}")
                if metadata['preprocessing_applied']:
                    print(f"🔧 Applied: {', '.join(metadata['preprocessing_applied'])}")
            
            # Run all enabled engines
            all_results = []
            engine_performance = {}
            
            for engine_name, engine in self.engines.items():
                if self.verbose:
                    print(f"\n🚀 Running {engine_name.upper()}...")
                
                engine_start = time.time()
                try:
                    # Different engines need different image formats
                    if engine_name == 'trocr':
                        results = engine.extract_text(pil_image)
                    else:
                        results = engine.extract_text(cv_image)
                    
                    engine_time = time.time() - engine_start
                    engine_performance[engine_name] = {
                        'processing_time': engine_time,
                        'regions_found': len(results),
                        'success': True
                    }
                    
                    # Add engine identifier to each result
                    for result in results:
                        result['engine'] = engine_name.upper()
                    
                    all_results.extend(results)
                    
                    if self.verbose:
                        print(f"   ✅ Found {len(results)} text regions in {engine_time:.2f}s")
                        
                except Exception as e:
                    engine_performance[engine_name] = {
                        'processing_time': time.time() - engine_start,
                        'regions_found': 0,
                        'success': False,
                        'error': str(e)
                    }
                    
                    if self.verbose:
                        print(f"   ❌ Failed: {str(e)}")
                    
                    self.logger.warning(f"{engine_name} failed: {e}")
            
            # Analyze and merge results
            final_results = self._analyze_and_merge_results(all_results, metadata, engine_performance)
            
            # Add processing metadata
            final_results['processing_metadata'] = {
                'image_path': image_path,
                'total_processing_time': time.time() - start_time,
                'engines_used': list(self.engines.keys()),
                'image_metadata': metadata,
                'engine_performance': engine_performance,
                'timestamp': datetime.now().isoformat()
            }
            
            if self.verbose:
                total_time = time.time() - start_time
                print(f"\n🎯 Processing complete in {total_time:.2f} seconds")
                print("=" * 60)
            
            return final_results
            
        except Exception as e:
            error_result = {
                'error': str(e),
                'processing_metadata': {
                    'image_path': image_path,
                    'total_processing_time': time.time() - start_time,
                    'engines_used': list(self.engines.keys()),
                    'timestamp': datetime.now().isoformat()
                }
            }
            return error_result
    
    def _analyze_and_merge_results(self, all_results: List[Dict], image_metadata: Dict, engine_performance: Dict) -> Dict:
        """Intelligent analysis and merging of results from all engines"""
        
        if not all_results:
            return {
                'summary': {'message': 'No text was detected in the image.'},
                'detailed_analysis': {},
                'engine_comparison': engine_performance,
                'full_results': {'all_text': [], 'by_engine': {}},
                'metadata': {
                    'total_text_regions': 0,
                    'engines_used': list(self.engines.keys())
                }
            }
        
        # Group results by engine
        engine_results = {}
        for result in all_results:
            engine = result.get('engine', 'unknown')
            if engine not in engine_results:
                engine_results[engine] = []
            engine_results[engine].append(result)
        
        # Sort all results by confidence
        all_results.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        
        # Create intelligent summary
        summary = self._create_intelligent_summary(all_results, engine_results, image_metadata)
        
        # Detailed content analysis
        detailed_analysis = self._analyze_text_content(all_results)
        
        # Engine comparison
        engine_comparison = self._compare_engine_performance(engine_results, engine_performance)
        
        return {
            'summary': summary,
            'detailed_analysis': detailed_analysis,
            'engine_comparison': engine_comparison,
            'full_results': {
                'all_text': all_results,
                'by_engine': engine_results
            },
            'metadata': {
                'total_text_regions': len(all_results),
                'engines_used': list(self.engines.keys()),
                'image_quality_score': self._calculate_image_quality_score(image_metadata)
            }
        }
    
    def _create_intelligent_summary(self, results: List[Dict], engine_results: Dict, image_metadata: Dict) -> Dict:
        """Create an intelligent summary with AI-like analysis"""
        
        if not results:
            return {'message': 'No text was detected in the image.'}
        
        # Get the highest confidence text
        primary_text = self._get_best_text_extraction(results, engine_results)
        
        # Analyze all text combined
        all_text = ' '.join([r['text'] for r in results if r['text'].strip()])
        
        # Text characteristics analysis
        text_characteristics = self._analyze_text_characteristics(all_text)
        
        # Confidence analysis
        confidence_analysis = self._analyze_confidence_scores(results, engine_results)
        
        # Generate AI recommendations
        recommendations = self._generate_ai_recommendations(results, engine_results, image_metadata, confidence_analysis)
        
        return {
            'primary_text': primary_text,
            'text_characteristics': text_characteristics,
            'confidence_analysis': confidence_analysis,
            'recommendations': recommendations
        }
    
    def _get_best_text_extraction(self, results: List[Dict], engine_results: Dict) -> str:
        """Intelligently select the best text extraction"""
        
        if not results:
            return ""
        
        # Strategy 1: If engines agree, use the agreed text
        if len(engine_results) > 1:
            # Get best result from each engine
            engine_best = {}
            for engine, engine_texts in engine_results.items():
                if engine_texts:
                    best = max(engine_texts, key=lambda x: x.get('confidence', 0))
                    engine_best[engine] = best['text'].strip().lower()
            
            # Check for agreement (similar text)
            unique_texts = list(set(engine_best.values()))
            if len(unique_texts) == 1:
                # Perfect agreement
                return max(results, key=lambda x: x.get('confidence', 0))['text']
        
        # Strategy 2: Use highest confidence result
        best_result = max(results, key=lambda x: x.get('confidence', 0))
        
        # Strategy 3: If TrOCR has good confidence, prefer it for handwritten text
        trocr_results = engine_results.get('TROCR', [])
        if trocr_results:
            best_trocr = max(trocr_results, key=lambda x: x.get('confidence', 0))
            if best_trocr.get('confidence', 0) > 0.7:
                return best_trocr['text']
        
        return best_result['text']
    
    def _analyze_text_characteristics(self, text: str) -> Dict:
        """Analyze characteristics of the extracted text"""
        if not text.strip():
            return {}
        
        # Basic statistics
        word_count = len(text.split())
        char_count = len(text)
        unique_words = len(set(text.lower().split()))
        
        # Content analysis
        has_numbers = bool(re.search(r'\d', text))
        has_special_chars = bool(re.search(r'[!@#$%^&*(),.?":{}|<>]', text))
        has_uppercase = bool(re.search(r'[A-Z]', text))
        has_lowercase = bool(re.search(r'[a-z]', text))
        
        # Text type classification
        text_type = self._classify_text_type(text)
        
        # Language detection (simple)
        estimated_language = 'English'  # Could be enhanced with proper language detection
        
        return {
            'word_count': word_count,
            'character_count': char_count,
            'unique_words': unique_words,
            'contains_numbers': has_numbers,
            'contains_special_characters': has_special_chars,
            'has_mixed_case': has_uppercase and has_lowercase,
            'text_type': text_type,
            'estimated_language': estimated_language,
            'avg_word_length': char_count / max(word_count, 1)
        }
    
    def _classify_text_type(self, text: str) -> str:
        """Classify the type of text using pattern matching"""
        text_lower = text.lower()
        
        # Financial documents
        if re.search(r'\b(invoice|receipt|bill|total|amount|price|\$|€|£|\d+\.\d{2})\b', text_lower):
            return 'Financial Document'
        
        # Forms and personal info
        elif re.search(r'\b(name|address|phone|email|contact|form|date of birth|ssn)\b', text_lower):
            return 'Form/Personal Information'
        
        # Academic/technical
        elif re.search(r'\b(chapter|section|page|figure|table|\d+\.\d+|abstract|conclusion)\b', text_lower):
            return 'Academic/Technical Document'
        
        # Correspondence
        elif re.search(r'\b(dear|sincerely|regards|letter|email|subject|from|to)\b', text_lower):
            return 'Correspondence'
        
        # License plates, signs, labels
        elif len(text.split()) <= 5 and re.search(r'[A-Z0-9]', text):
            return 'Sign/Label/ID'
        
        # Handwritten notes
        elif any(engine == 'TROCR' for engine in [r.get('engine') for r in []] if r.get('confidence', 0) > 0.8):
            return 'Handwritten Text'
        
        # Default
        else:
            return 'General Text'
    
    def _analyze_confidence_scores(self, results: List[Dict], engine_results: Dict) -> Dict:
        """Analyze confidence scores across all engines"""
        
        if not results:
            return {}
        
        confidences = [r.get('confidence', 0) for r in results]
        
        # Calculate statistics
        highest_confidence = max(confidences)
        lowest_confidence = min(confidences)
        average_confidence = sum(confidences) / len(confidences)
        
        # Engine agreement analysis
        engine_agreement = self._check_engine_agreement(engine_results)
        
        # Quality assessment
        quality_score = self._calculate_text_quality_score(results)
        
        return {
            'highest_confidence': highest_confidence,
            'lowest_confidence': lowest_confidence,
            'average_confidence': average_confidence,
            'confidence_range': highest_confidence - lowest_confidence,
            'engine_agreement': engine_agreement,
            'quality_score': quality_score
        }
    
    def _check_engine_agreement(self, engine_results: Dict) -> Dict:
        """Check agreement between different engines"""
        
        if len(engine_results) < 2:
            return {'engines_agree': True, 'agreement_score': 1.0}
        
        # Get best text from each engine
        engine_texts = {}
        for engine, results in engine_results.items():
            if results:
                best = max(results, key=lambda x: x.get('confidence', 0))
                engine_texts[engine] = best['text'].lower().strip()
        
        # Calculate agreement score
        texts = list(engine_texts.values())
        unique_texts = len(set(texts))
        total_engines = len(texts)
        
        agreement_score = 1.0 - (unique_texts - 1) / max(total_engines - 1, 1)
        engines_agree = agreement_score > 0.7
        
        return {
            'engines_agree': engines_agree,
            'agreement_score': agreement_score,
            'conflicting_results': unique_texts > 1,
            'engine_texts': engine_texts
        }
    
    def _calculate_text_quality_score(self, results: List[Dict]) -> float:
        """Calculate overall text quality score"""
        
        if not results:
            return 0.0
        
        # Factors that indicate good quality
        avg_confidence = sum([r.get('confidence', 0) for r in results]) / len(results)
        
        # Text consistency (similar results indicate good quality)
        texts = [r['text'].lower().strip() for r in results]
        unique_texts = len(set(texts))
        consistency_score = 1.0 - (unique_texts - 1) / max(len(results) - 1, 1)
        
        # Length factor (very short extractions might be noise)
        avg_text_length = sum([len(r['text']) for r in results]) / len(results)
        length_factor = min(avg_text_length / 10, 1.0)  # Cap at 1.0
        
        # Combined score
        quality_score = (avg_confidence * 0.5 + consistency_score * 0.3 + length_factor * 0.2)
        
        return min(quality_score, 1.0)
    
    def _generate_ai_recommendations(self, results: List[Dict], engine_results: Dict, image_metadata: Dict, confidence_analysis: Dict) -> List[str]:
        """Generate AI-powered recommendations"""
        
        recommendations = []
        
        # Image quality recommendations
        if image_metadata.get('is_blurry'):
            recommendations.append("Image appears blurry. Consider using a sharper, higher resolution image for better results.")
        
        if image_metadata.get('is_dark'):
            recommendations.append("Image is quite dark. Better lighting or brightness adjustment could improve accuracy.")
        
        if image_metadata.get('is_low_contrast'):
            recommendations.append("Low contrast detected. Enhancing contrast could improve text recognition.")
        
        # Confidence-based recommendations
        avg_confidence = confidence_analysis.get('average_confidence', 0)
        if avg_confidence < 0.5:
            recommendations.append("Low confidence scores detected. Consider preprocessing the image or using a clearer source.")
        elif avg_confidence > 0.9:
            recommendations.append("Excellent text recognition achieved! The image quality is optimal for OCR.")
        
        # Engine-specific recommendations
        if 'TROCR' in engine_results and engine_results['TROCR']:
            trocr_avg = sum([r.get('confidence', 0) for r in engine_results['TROCR']]) / len(engine_results['TROCR'])
            if trocr_avg > 0.8:
                recommendations.append("TrOCR performed excellently - this text appears to be handwritten.")
        
        # Engine agreement recommendations
        if not confidence_analysis.get('engine_agreement', {}).get('engines_agree', True):
            recommendations.append("Engines produced different results. Manual verification recommended for critical applications.")
        
        # Text type specific recommendations
        if any('Financial' in r.get('text', '') for r in results):
            recommendations.append("Financial document detected. Consider using specialized financial OCR for critical data.")
        
        # Default positive message
        if not recommendations:
            recommendations.append("Text extraction completed successfully with good confidence across all engines.")
        
        return recommendations
    
    def _analyze_text_content(self, results: List[Dict]) -> Dict:
        """Detailed analysis of extracted text content"""
        
        if not results:
            return {}
        
        all_text = ' '.join([r['text'] for r in results if r['text'].strip()])
        
        # Pattern detection
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', all_text)
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', all_text)
        phones = re.findall(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', all_text)
        dates = re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', all_text)
        
        return {
            'total_characters': len(all_text),
            'total_words': len(all_text.split()),
            'unique_words': len(set(all_text.lower().split())),
            'contains_urls': len(urls) > 0,
            'contains_emails': len(emails) > 0,
            'contains_phones': len(phones) > 0,
            'contains_dates': len(dates) > 0,
            'extracted_urls': urls,
            'extracted_emails': emails,
            'extracted_phones': phones,
            'extracted_dates': dates
        }
    
    def _compare_engine_performance(self, engine_results: Dict, engine_performance: Dict) -> Dict:
        """Compare performance of different engines"""
        
        comparison = {}
        
        for engine_name in self.engines.keys():
            results = engine_results.get(engine_name.upper(), [])
            performance = engine_performance.get(engine_name, {})
            
            if results:
                avg_confidence = sum([r.get('confidence', 0) for r in results]) / len(results)
                total_chars = sum([len(r['text']) for r in results])
                best_result = max(results, key=lambda x: x.get('confidence', 0))
                
                comparison[engine_name.upper()] = {
                    'text_regions_found': len(results),
                    'average_confidence': round(avg_confidence, 3),
                    'total_characters_extracted': total_chars,
                    'processing_time': performance.get('processing_time', 0),
                    'best_result': best_result['text'][:100] + '...' if len(best_result['text']) > 100 else best_result['text'],
                    'success': performance.get('success', False)
                }
            else:
                comparison[engine_name.upper()] = {
                    'text_regions_found': 0,
                    'average_confidence': 0.0,
                    'total_characters_extracted': 0,
                    'processing_time': performance.get('processing_time', 0),
                    'best_result': '',
                    'success': performance.get('success', False),
                    'error': performance.get('error', 'No text found')
                }
        
        return comparison
    
    def _calculate_image_quality_score(self, metadata: Dict) -> float:
        """Calculate overall image quality score"""
        
        score = 1.0
        
        # Penalize for blur
        if metadata.get('is_blurry', False):
            score -= 0.3
        
        # Penalize for poor lighting
        if metadata.get('is_dark', False) or metadata.get('brightness', 128) > 220:
            score -= 0.2
        
        # Penalize for low contrast
        if metadata.get('is_low_contrast', False):
            score -= 0.2
        
        # File size factor (very small files might be low quality)
        file_size_mb = metadata.get('file_size_mb', 1)
        if file_size_mb < 0.1:
            score -= 0.1
        
        return max(score, 0.0)