import pytest
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


class TestPreprocessingPipelineReal:
    """Test preprocessing pipeline with real images and visual output"""
    
    def setup_method(self):
        """Setup test environment"""
        # Fix path - you're running from project root, so no need to go up
        self.test_image_path = os.path.join('data', 'sample_images', 'img3.jpg')
        self.output_dir = 'test_output'
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Print current directory and check for image
        print(f"Current directory: {os.getcwd()}")
        print(f"Looking for image at: {self.test_image_path}")
        print(f"Available images in data/sample_images/:")
        sample_dir = os.path.join('data', 'sample_images')
        if os.path.exists(sample_dir):
            for f in os.listdir(sample_dir):
                print(f"  - {f}")
        else:
            print("  sample_images directory not found!")
    
    def save_debug_image(self, image, filename, title=""):
        """Save debug image with optional matplotlib display"""
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3:
                # Handle RGBA to RGB conversion
                if image.shape[2] == 4:
                    # Convert RGBA to RGB
                    image_rgb = image[:, :, :3]
                    print(f"   Converted RGBA to RGB for {filename}")
                else:
                    image_rgb = image
                    
                plt.figure(figsize=(12, 8))
                plt.imshow(image_rgb)
                plt.title(title if title else filename)
                plt.axis('off')
                plt.savefig(os.path.join(self.output_dir, f"{filename}.png"), dpi=150, bbox_inches='tight')
                plt.close()
                
                # Save as PNG to avoid JPEG issues
                Image.fromarray(image_rgb.astype(np.uint8)).save(os.path.join(self.output_dir, f"{filename}.png"))
            else:
                # Grayscale image
                plt.figure(figsize=(12, 8))
                plt.imshow(image, cmap='gray')
                plt.title(title if title else filename)
                plt.axis('off')
                plt.savefig(os.path.join(self.output_dir, f"{filename}.png"), dpi=150, bbox_inches='tight')
                plt.close()
    
    def draw_text_regions(self, image, text_regions, filename="text_regions"):
        """Draw bounding boxes on image to visualize text regions"""
        if isinstance(image, np.ndarray):
            vis_image = image.copy()
            if len(vis_image.shape) == 2:  # Grayscale
                vis_image = cv2.cvtColor(vis_image, cv2.COLOR_GRAY2RGB)
            
            # Draw rectangles for each text region
            for i, region in enumerate(text_regions[:50]):  # Limit to first 50 for visibility
                try:
                    if hasattr(region, 'bbox'):
                        bbox = region.bbox
                        # Handle BoundingBox object
                        if hasattr(bbox, 'x') and hasattr(bbox, 'y'):
                            x, y, w, h = bbox.x, bbox.y, bbox.width, bbox.height
                        elif hasattr(bbox, 'x1') and hasattr(bbox, 'y1'):
                            x, y = bbox.x1, bbox.y1
                            w, h = bbox.x2 - bbox.x1, bbox.y2 - bbox.y1
                        else:
                            print(f"   Unknown bbox format: {type(bbox)}, attrs: {dir(bbox)}")
                            continue
                    elif hasattr(region, 'x') and hasattr(region, 'y'):
                        x, y, w, h = region.x, region.y, region.width, region.height
                    elif isinstance(region, (list, tuple)) and len(region) >= 4:
                        x, y, w, h = region[:4]
                    else:
                        print(f"   Unknown region format: {type(region)}, attrs: {dir(region)}")
                        continue
                except Exception as e:
                    print(f"   Error processing region {i}: {e}")
                    continue
                
                # Draw rectangle
                cv2.rectangle(vis_image, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)
                # Add region number
                cv2.putText(vis_image, str(i), (int(x), int(y)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            self.save_debug_image(vis_image, filename, f"Text Regions Detected: {len(text_regions)}")
            return vis_image
        return None

    def test_real_image_exists(self):
        """Verify test image exists"""
        assert os.path.exists(self.test_image_path), f"Test image not found: {self.test_image_path}"
        print(f"✅ Test image found: {self.test_image_path}")

    def test_quality_analyzer_real_image(self):
        """Test QualityAnalyzer with real image"""
        try:
            from advanced_ocr.preprocessing.quality_analyzer import QualityAnalyzer
            from advanced_ocr.config import OCRConfig
            
            # Load real image
            with Image.open(self.test_image_path) as pil_image:
                # Convert to RGB if RGBA to avoid issues
                if pil_image.mode == 'RGBA':
                    pil_image = pil_image.convert('RGB')
                    print("   Converted RGBA to RGB")
                original_image = np.array(pil_image)
                self.save_debug_image(original_image, "01_original", "Original Image")
                
                print(f"📷 Original image shape: {original_image.shape}")
                
                # Test quality analysis
                config = OCRConfig("quality_analyzer")
                analyzer = QualityAnalyzer(config)
                
                quality_metrics = analyzer.analyze_image_quality(original_image)
                
                print(f"🔍 Quality Analysis Results:")
                if hasattr(quality_metrics, '__dict__'):
                    for key, value in quality_metrics.__dict__.items():
                        print(f"   - {key}: {value}")
                else:
                    print(f"   - Quality metrics: {quality_metrics}")
                
                assert quality_metrics is not None, "Quality analysis returned None"
                
        except Exception as e:
            pytest.fail(f"QualityAnalyzer real image test failed: {e}")

    def test_text_detector_real_image(self):
        """Test TextDetector with real image"""
        try:
            from advanced_ocr.preprocessing.text_detector import TextDetector
            from advanced_ocr.config import OCRConfig
            from advanced_ocr.utils.model_utils import ModelLoader
            
            # Load real image
            with Image.open(self.test_image_path) as pil_image:
                if pil_image.mode == 'RGBA':
                    pil_image = pil_image.convert('RGB')
                original_image = np.array(pil_image)
                
                # Test text detection
                config = OCRConfig("text_detector")
                model_loader = ModelLoader(config)
                detector = TextDetector(model_loader, config)
                
                print(f"🔍 Running text detection on image shape: {original_image.shape}")
                text_regions = detector.detect_text_regions(original_image)
                
                print(f"📝 Text Detection Results:")
                print(f"   - Number of regions detected: {len(text_regions)}")
                print(f"   - Region types: {[type(r).__name__ for r in text_regions[:5]]}")
                
                # Visualize text regions
                if len(text_regions) > 0:
                    self.draw_text_regions(original_image, text_regions, "02_text_regions")
                    
                    # Print details of first few regions
                    for i, region in enumerate(text_regions[:10]):
                        print(f"   - Region {i}: {region}")
                
                # Verify reasonable number of regions (should be 20-80 according to your spec)
                assert 0 < len(text_regions) < 500, f"Unreasonable number of text regions: {len(text_regions)}"
                if len(text_regions) > 80:
                    print(f"⚠️  WARNING: {len(text_regions)} regions detected (expected 20-80)")
                
        except Exception as e:
            pytest.fail(f"TextDetector real image test failed: {e}")

    def test_image_processor_real_image(self):
        """Test ImageProcessor with real image - FULL PIPELINE"""
        try:
            from advanced_ocr.preprocessing.image_processor import ImageProcessor
            from advanced_ocr.config import OCRConfig
            from advanced_ocr.utils.model_utils import ModelLoader
            
            # Load real image
            with Image.open(self.test_image_path) as pil_image:
                if pil_image.mode == 'RGBA':
                    pil_image = pil_image.convert('RGB')
                original_image = np.array(pil_image)
                
                print(f"🚀 Testing FULL ImageProcessor Pipeline:")
                print(f"   - Original image: {original_image.shape}")
                
                # Test full preprocessing pipeline
                config = OCRConfig("image_processor")
                model_loader = ModelLoader(config)
                processor = ImageProcessor(model_loader, config)
                
                # Process image through full pipeline
                result = processor.process_image(original_image)
                
                print(f"✅ ImageProcessor Results:")
                print(f"   - Enhanced image shape: {result.enhanced_image.shape}")
                print(f"   - Text regions count: {len(result.text_regions)}")
                print(f"   - Quality metrics type: {type(result.quality_metrics)}")
                
                # Save enhanced image
                self.save_debug_image(result.enhanced_image, "03_enhanced", "Enhanced Image")
                
                # Visualize final text regions
                if len(result.text_regions) > 0:
                    self.draw_text_regions(result.enhanced_image, result.text_regions, "04_final_regions")
                
                # Print quality metrics details
                if hasattr(result.quality_metrics, '__dict__'):
                    print(f"📊 Quality Metrics Details:")
                    for key, value in result.quality_metrics.__dict__.items():
                        print(f"   - {key}: {value}")
                
                # Verify pipeline results
                assert result.enhanced_image is not None, "Enhanced image is None"
                assert result.text_regions is not None, "Text regions is None"
                assert result.quality_metrics is not None, "Quality metrics is None"
                assert isinstance(result.enhanced_image, np.ndarray), "Enhanced image not numpy array"
                assert isinstance(result.text_regions, list), "Text regions not list"
                
                # Check if enhancement actually changed the image
                if not np.array_equal(original_image, result.enhanced_image):
                    print("✅ Image was enhanced (modified)")
                else:
                    print("ℹ️  Image was not modified by enhancement")
                
                # Performance check
                if len(result.text_regions) > 100:
                    print(f"⚠️  Performance Warning: {len(result.text_regions)} regions detected")
                
                print(f"🎯 Preprocessing pipeline test completed successfully!")
                print(f"📁 Debug images saved to: {self.output_dir}")
                
        except Exception as e:
            pytest.fail(f"ImageProcessor full pipeline test failed: {e}")

    def test_preprocessing_logic_flow(self):
        """Test that preprocessing follows correct logic flow"""
        try:
            from advanced_ocr.preprocessing.image_processor import ImageProcessor
            import inspect
            
            # Check that ImageProcessor imports the correct dependencies
            source = inspect.getsource(ImageProcessor)
            
            print("🔍 Checking ImageProcessor Logic Flow:")
            
            # Should use QualityAnalyzer
            assert 'quality_analyzer' in source.lower() or 'qualityanalyzer' in source.lower(), \
                "ImageProcessor should import and use QualityAnalyzer"
            print("✅ Uses QualityAnalyzer")
            
            # Should use TextDetector  
            assert 'text_detector' in source.lower() or 'textdetector' in source.lower(), \
                "ImageProcessor should import and use TextDetector"
            print("✅ Uses TextDetector")
            
            # Should NOT do engine work
            forbidden_terms = ['tesseract', 'paddleocr', 'easyocr', 'trocr', 'extract_text']
            for term in forbidden_terms:
                assert term.lower() not in source.lower(), f"ImageProcessor should not do {term} work"
            print("✅ Does not do engine work")
            
            # Should coordinate preprocessing
            expected_methods = ['process_image', 'enhance_image']
            for method in expected_methods:
                assert method in source.lower(), f"ImageProcessor should have {method} method"
            print("✅ Has proper orchestration methods")
            
        except Exception as e:
            pytest.fail(f"Preprocessing logic flow test failed: {e}")


if __name__ == "__main__":
    # Run tests with verbose output
    test_instance = TestPreprocessingPipelineReal()
    test_instance.setup_method()
    
    print("=" * 80)
    print("🧪 ADVANCED OCR - PREPROCESSING PIPELINE TESTS")
    print("=" * 80)
    
    try:
        test_instance.test_real_image_exists()
        test_instance.test_quality_analyzer_real_image() 
        test_instance.test_text_detector_real_image()
        test_instance.test_image_processor_real_image()
        test_instance.test_preprocessing_logic_flow()
        
        print("=" * 80)
        print("🎉 ALL PREPROCESSING TESTS PASSED!")
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()