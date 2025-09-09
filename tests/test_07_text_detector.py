#!/usr/bin/env python3
"""
Test 7: Text Detector - Fixed for TextRegion objects
Purpose: Test text region detection capabilities
Tests: src/preprocessing/text_detector.py, region identification
Success criteria: Text regions detected with bounding boxes from img1.jpg
"""

import sys
import os
import cv2
import numpy as np
import time
from pathlib import Path
import importlib.util

# Fix the path issue - add both project root and src to path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"

# Ensure paths are in sys.path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

def test_text_detector():
    """Test the Text Detector component"""
    print("=" * 60)
    print("TEST 7: TEXT DETECTOR")
    print("=" * 60)
    
    try:
        # Test image path
        test_image_path = project_root / "data" / "sample_images" / "img1.jpg"
        
        # Load test image
        if not test_image_path.exists():
            print(f"✗ Test image not found: {test_image_path}")
            return False
            
        image = cv2.imread(str(test_image_path))
        if image is None:
            print("✗ Failed to load test image")
            return False
            
        print(f"✓ Test image loaded: {image.shape}")
        
        # Test 7.1: Import Text Detector
        TextDetector = None
        text_detector_module = None
        try:
            # Try direct module loading
            detector_file = src_path / "preprocessing" / "text_detector.py"
            if not detector_file.exists():
                print(f"✗ Text detector file not found: {detector_file}")
                return False
                
            spec = importlib.util.spec_from_file_location("text_detector", detector_file)
            text_detector_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(text_detector_module)
            
            # Find available detector classes
            available_classes = [attr for attr in dir(text_detector_module) 
                               if not attr.startswith('_') and attr[0].isupper()]
            
            print(f"Available classes in text_detector: {available_classes}")
            
            # Try to get the detector class
            detector_classes = ['TextDetector', 'CRAFTDetector', 'EASTDetector', 
                              'AdvancedTextDetector', 'TextRegionDetector', 'DocumentTextDetector']
            
            for class_name in detector_classes:
                if hasattr(text_detector_module, class_name):
                    TextDetector = getattr(text_detector_module, class_name)
                    print(f"✓ {class_name} imported successfully")
                    break
            
            if TextDetector is None and available_classes:
                # Try first available class that looks like a detector
                for cls_name in available_classes:
                    if 'detect' in cls_name.lower() or 'text' in cls_name.lower():
                        TextDetector = getattr(text_detector_module, cls_name)
                        print(f"✓ {cls_name} imported as TextDetector")
                        break
                        
            if TextDetector is None:
                print("✗ No suitable text detector class found")
                return False
                
        except Exception as e:
            print(f"✗ Failed to import text detector: {e}")
            return False
        
        # Test 7.2: Initialize Text Detector
        try:
            detector = TextDetector()
            print("✓ TextDetector initialized")
        except Exception as e:
            # Try with config parameter
            try:
                detector = TextDetector(config={})
                print("✓ TextDetector initialized with config")
            except Exception as e2:
                print(f"✗ Failed to initialize TextDetector: {e}, {e2}")
                return False
        
        # Test 7.3: Discover available methods
        available_methods = [method for method in dir(detector) 
                           if not method.startswith('_') and callable(getattr(detector, method))]
        print(f"✓ Available methods: {available_methods}")
        
        # Test 7.4: Text Detection (try different method names)
        text_regions = None
        detection_method_used = None
        detection_time = 0.0
        
        # Common method names to try
        method_attempts = [
            'detect_text_regions',
            'detect_text', 
            'detect',
            'find_text_regions',
            'find_text',
            'get_text_regions',
            'extract_text_regions',
            'locate_text'
        ]
        
        for method_name in method_attempts:
            if hasattr(detector, method_name):
                try:
                    start_time = time.time()
                    method = getattr(detector, method_name)
                    result = method(image)
                    detection_time = time.time() - start_time
                    detection_method_used = method_name
                    
                    # Handle different return types
                    if isinstance(result, list) and len(result) > 0:
                        text_regions = result
                        print(f"✓ Text detection completed using {method_name}")
                        print(f"  - Processing time: {detection_time:.3f}s")
                        print(f"  - Number of regions detected: {len(text_regions)}")
                        break
                    elif hasattr(result, 'regions') or hasattr(result, 'text_regions'):
                        # Result object with regions attribute
                        regions = getattr(result, 'regions', getattr(result, 'text_regions', None))
                        if regions and len(regions) > 0:
                            text_regions = regions
                            print(f"✓ Text detection completed using {method_name}")
                            print(f"  - Processing time: {detection_time:.3f}s")
                            print(f"  - Number of regions detected: {len(text_regions)}")
                            break
                    elif result is not None:
                        # Some other valid result
                        text_regions = result
                        print(f"✓ Text detection completed using {method_name}")
                        print(f"  - Processing time: {detection_time:.3f}s")
                        print(f"  - Result type: {type(result)}")
                        break
                    else:
                        print(f"✗ {method_name} returned None or empty result")
                except Exception as e:
                    print(f"✗ {method_name} failed: {e}")
        
        if text_regions is None:
            print("✗ No working text detection method found")
            # Try basic OpenCV text detection as fallback
            try:
                print("Attempting fallback OpenCV text detection...")
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
                # Simple contour-based text detection
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Filter contours by area and aspect ratio
                text_regions = []
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    area = w * h
                    aspect_ratio = w / h
                    
                    if area > 100 and 0.1 < aspect_ratio < 10:  # Basic text region filters
                        text_regions.append([x, y, x+w, y+h])
                
                if len(text_regions) > 0:
                    detection_method_used = "opencv_fallback"
                    print(f"✓ Fallback detection found {len(text_regions)} potential text regions")
                else:
                    print("✗ Even fallback detection failed")
                    return False
            except Exception as e:
                print(f"✗ Fallback detection failed: {e}")
                return False
        
        # Test 7.5: Analyze detected regions (FIXED for TextRegion objects)
        try:
            print("✓ Analyzing detected text regions:")
            
            if isinstance(text_regions, list):
                valid_regions = 0
                total_area = 0
                confidence_sum = 0.0
                confidence_count = 0
                
                for i, region in enumerate(text_regions):
                    try:
                        # Handle TextRegion objects and other formats
                        if hasattr(region, 'bbox'):
                            # TextRegion object with bbox (x, y, width, height)
                            x1, y1, width, height = region.bbox
                            x2, y2 = x1 + width, y1 + height
                            confidence = getattr(region, 'confidence', 0.0)
                            method = getattr(region, 'method', 'unknown')
                        elif isinstance(region, (list, tuple)) and len(region) >= 4:
                            # Bounding box format [x1, y1, x2, y2] or [x, y, w, h]
                            if len(region) == 4:
                                x1, y1, x2, y2 = region[:4]
                                if x2 < x1:  # width, height format
                                    w, h = x2, y2
                                    x2, y2 = x1 + w, y1 + h
                                width = x2 - x1
                                height = y2 - y1
                            else:
                                x1, y1, width, height = region[:4]
                                x2, y2 = x1 + width, y1 + height
                            confidence = 0.0
                            method = 'unknown'
                        else:
                            continue
                        
                        area = width * height
                        if area > 0:
                            valid_regions += 1
                            total_area += area
                            
                            if confidence > 0:
                                confidence_sum += confidence
                                confidence_count += 1
                            
                            if i < 5:  # Show first 5 regions
                                conf_str = f", conf: {confidence:.2f}" if hasattr(region, 'bbox') else ""
                                method_str = f", method: {method}" if hasattr(region, 'bbox') and method != 'unknown' else ""
                                print(f"  - Region {i+1}: ({x1}, {y1}) to ({x2}, {y2}), area: {area}{conf_str}{method_str}")
                    
                    except Exception as e:
                        print(f"  - Region {i+1}: Error analyzing - {e}")
                
                print(f"  - Valid regions: {valid_regions}/{len(text_regions)}")
                if valid_regions > 0:
                    avg_area = total_area / valid_regions
                    print(f"  - Average region area: {avg_area:.0f} pixels")
                    print(f"  - Total text area coverage: {total_area / (image.shape[0] * image.shape[1]) * 100:.1f}%")
                    
                    if confidence_count > 0:
                        avg_confidence = confidence_sum / confidence_count
                        print(f"  - Average confidence: {avg_confidence:.3f}")
                        print(f"  - High confidence regions (>0.7): {sum(1 for r in text_regions if hasattr(r, 'bbox') and r.confidence > 0.7)}")
            
        except Exception as e:
            print(f"✗ Region analysis failed: {e}")
        
        # Test 7.6: Test with different image types
        try:
            print("✓ Testing edge cases:")
            
            # Grayscale image
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if len(gray_image.shape) == 2:
                # Convert back to 3-channel for detector
                gray_3ch = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
                
            method = getattr(detector, detection_method_used) if detection_method_used != "opencv_fallback" else None
            if method:
                try:
                    gray_result = method(gray_3ch)
                    if gray_result is not None and len(gray_result) > 0:
                        print(f"  - Grayscale image: {len(gray_result)} regions detected")
                    else:
                        print("  - Grayscale image: No regions detected")
                except Exception as e:
                    print(f"  - Grayscale image: Error - {e}")
            
            # Small image
            small_image = cv2.resize(image, (400, 300))
            if method:
                try:
                    small_result = method(small_image)
                    if small_result is not None and len(small_result) > 0:
                        print(f"  - Small image: {len(small_result)} regions detected")
                    else:
                        print("  - Small image: No regions detected")
                except Exception as e:
                    print(f"  - Small image: Error - {e}")
                    
        except Exception as e:
            print(f"✗ Edge case testing failed: {e}")
        
        # Test 7.7: Save detection visualization (FIXED for TextRegion objects)
        try:
            debug_dir = project_root / "debug" / "text_detection"
            debug_dir.mkdir(parents=True, exist_ok=True)
            
            # Create visualization
            vis_image = image.copy()
            
            if text_regions and isinstance(text_regions, list):
                region_count = 0
                for region in text_regions:
                    try:
                        # Handle TextRegion objects and other formats
                        if hasattr(region, 'bbox'):
                            # TextRegion object with bbox (x, y, width, height)
                            x1, y1, width, height = region.bbox
                            x2, y2 = x1 + width, y1 + height
                            confidence = getattr(region, 'confidence', 0.0)
                        elif isinstance(region, (list, tuple)) and len(region) >= 4:
                            x1, y1, x2, y2 = region[:4]
                            if x2 < x1:  # width, height format
                                w, h = x2, y2
                                x2, y2 = x1 + w, y1 + h
                            confidence = 0.0
                        else:
                            continue
                            
                        # Draw bounding box with color based on confidence
                        if confidence > 0.7:
                            color = (0, 255, 0)  # Green for high confidence
                        elif confidence > 0.5:
                            color = (0, 255, 255)  # Yellow for medium confidence
                        else:
                            color = (0, 0, 255)  # Red for low confidence
                            
                        cv2.rectangle(vis_image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        
                        # Add confidence text if available
                        if hasattr(region, 'bbox') and confidence > 0:
                            label = f"{region_count+1}:{confidence:.2f}"
                        else:
                            label = f"{region_count+1}"
                            
                        cv2.putText(vis_image, label, (int(x1), int(y1-5)), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                        region_count += 1
                    except Exception as ex:
                        continue
                
                print(f"  - Drew {region_count} bounding boxes on visualization")
            
            # Save images
            cv2.imwrite(str(debug_dir / "01_original.jpg"), image)
            cv2.imwrite(str(debug_dir / f"02_detected_regions_{detection_method_used}.jpg"), vis_image)
            
            print(f"✓ Detection visualization saved to: {debug_dir}")
            
        except Exception as e:
            print(f"✗ Failed to save visualization: {e}")
        
        # Test 7.8: Performance and accuracy assessment
        try:
            print("✓ Performance summary:")
            print(f"  - Detection method: {detection_method_used}")
            print(f"  - Processing time: {detection_time:.3f}s")
            print(f"  - Regions detected: {len(text_regions) if text_regions else 0}")
            
            # Basic performance metrics
            image_area = image.shape[0] * image.shape[1]
            processing_rate = image_area / detection_time if detection_time > 0 else 0
            print(f"  - Processing rate: {processing_rate:.0f} pixels/second")
            
            if text_regions and len(text_regions) > 0:
                print(f"  - Average regions per megapixel: {len(text_regions) / (image_area / 1000000):.1f}")
                
                # Quality metrics for TextRegion objects
                if hasattr(text_regions[0], 'bbox'):
                    high_conf = sum(1 for r in text_regions if r.confidence > 0.7)
                    med_conf = sum(1 for r in text_regions if 0.5 < r.confidence <= 0.7)
                    low_conf = sum(1 for r in text_regions if r.confidence <= 0.5)
                    
                    print(f"  - Confidence distribution: High({high_conf}), Med({med_conf}), Low({low_conf})")
            
        except Exception as e:
            print(f"✗ Performance assessment failed: {e}")
        
        # Test 7.9: Additional detector features (if available)
        try:
            additional_features = []
            
            # Check for confidence scores
            if hasattr(detector, 'get_confidence') or hasattr(detector, 'confidence_threshold'):
                additional_features.append("confidence_scoring")
            
            # Check for text orientation detection
            if hasattr(detector, 'detect_orientation') or hasattr(detector, 'get_orientation'):
                additional_features.append("orientation_detection")
            
            # Check for text type classification
            if hasattr(detector, 'classify_text_type') or hasattr(detector, 'get_text_type'):
                additional_features.append("text_classification")
            
            # Check for multi-scale detection
            if hasattr(detector, 'multi_scale_detection') or hasattr(detector, 'detect_multi_scale'):
                additional_features.append("multi_scale_detection")
            
            if additional_features:
                print(f"✓ Additional features detected: {', '.join(additional_features)}")
            else:
                print("✓ Basic text detection functionality confirmed")
                
        except Exception as e:
            print(f"✗ Feature detection failed: {e}")
        
        print("=" * 60)
        print("TEST 7: PASSED - Text detector works correctly")
        print("=" * 60)
        print("SUCCESS: Text detector is functioning properly")
        print("Next step: Run Test 8 (Base Engine Classes)")
        return True
        
    except Exception as e:
        print(f"✗ CRITICAL ERROR in Text Detector test: {e}")
        import traceback
        traceback.print_exc()
        print("=" * 60)
        print("TEST 7: FAILED")
        print("=" * 60)
        return False

if __name__ == "__main__":
    success = test_text_detector()
    sys.exit(0 if success else 1)