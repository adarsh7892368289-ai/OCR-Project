#!/usr/bin/env python3
"""
Enhanced Pipeline Test - Saves line-by-line results and visualizations
"""

import os
import sys
import json
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import traceback
import re
# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

class EnhancedPipelineTest:
    def __init__(self):
        self.test_image_path = "data/sample_images/img3.jpg"
        self.output_dir = Path("debug/enhanced_pipeline_test")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Clear previous results
        for file in list(self.output_dir.glob("*")):
            try:
                if file.is_file():
                    file.unlink()
            except:
                pass
    
    def log_step(self, step_name, success, data=None, error=None):
        status = "✅" if success else "❌"
        print(f"{status} {step_name}")
        if error:
            print(f"   Error: {error}")
        if success and data is not None:
            if hasattr(data, 'shape'):
                print(f"   Result: {type(data).__name__} with shape {data.shape}")
            elif isinstance(data, list):
                print(f"   Result: {type(data).__name__} with {len(data)} items")
            elif hasattr(data, '__dict__'):
                print(f"   Result: {type(data).__name__} object")
                if hasattr(data, 'enhanced_image'):
                    print(f"   Contains enhanced_image with shape: {data.enhanced_image.shape}")
            else:
                print(f"   Result: {type(data).__name__}")
        print()
    
    def save_result(self, filename, data):
        try:
            output_file = self.output_dir / filename
            if isinstance(data, np.ndarray):
                cv2.imwrite(str(output_file), data)
            elif isinstance(data, (dict, list)):
                with open(output_file, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
            else:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(str(data))
            print(f"   💾 Saved: {filename}")
        except Exception as e:
            print(f"   ⚠️  Save failed: {e}")
    
    def extract_text_from_ocr_result(self, result, engine_name):
        """Extract text with proper receipt line reconstruction based on spatial analysis"""
        text_content = ""
        result_details = {}
        line_analysis = []
        
        try:
            if isinstance(result, list):
                # Handle list of OCR results with advanced spatial grouping
                text_items = []
                
                for i, item in enumerate(result):
                    item_text = ""
                    bbox = None
                    confidence = 0.0
                    
                    # Extract text, bbox, and confidence from different result formats
                    if hasattr(item, 'text') and hasattr(item, 'bbox'):
                        item_text = item.text
                        bbox = item.bbox
                        confidence = getattr(item, 'confidence', 0.0)
                    elif isinstance(item, dict):
                        item_text = item.get('text', '')
                        bbox = item.get('bbox')
                        confidence = item.get('confidence', 0.0)
                    elif isinstance(item, (list, tuple)) and len(item) >= 2:
                        bbox = item[0] if len(item) > 0 else None
                        item_text = str(item[1]) if len(item) > 1 else ""
                        confidence = item[2] if len(item) > 2 else 0.0
                    elif isinstance(item, str):
                        item_text = item
                    
                    if item_text and item_text.strip():
                        # Parse bounding box to get coordinates
                        try:
                            if bbox and isinstance(bbox, (list, tuple)):
                                if len(bbox) == 4:  # (x, y, w, h)
                                    x, y, w, h = bbox
                                    center_y = y + h/2
                                    center_x = x + w/2
                                    left_x = x
                                    right_x = x + w
                                    top_y = y
                                    bottom_y = y + h
                                    text_height = h
                                    text_width = w
                                elif len(bbox) == 8:  # 4 corner points [x1,y1,x2,y2,x3,y3,x4,y4]
                                    points = np.array(bbox).reshape(4, 2)
                                    left_x = points[:, 0].min()
                                    right_x = points[:, 0].max()
                                    top_y = points[:, 1].min()
                                    bottom_y = points[:, 1].max()
                                    center_y = (top_y + bottom_y) / 2
                                    center_x = (left_x + right_x) / 2
                                    text_height = bottom_y - top_y
                                    text_width = right_x - left_x
                                elif len(bbox) >= 8:  # PaddleOCR format: nested points
                                    # Flatten if nested
                                    flat_bbox = []
                                    for point in bbox:
                                        if isinstance(point, (list, tuple)):
                                            flat_bbox.extend(point)
                                        else:
                                            flat_bbox.append(point)
                                    
                                    if len(flat_bbox) >= 8:
                                        points = np.array(flat_bbox[:8]).reshape(4, 2)
                                        left_x = points[:, 0].min()
                                        right_x = points[:, 0].max()
                                        top_y = points[:, 1].min()
                                        bottom_y = points[:, 1].max()
                                        center_y = (top_y + bottom_y) / 2
                                        center_x = (left_x + right_x) / 2
                                        text_height = bottom_y - top_y
                                        text_width = right_x - left_x
                                    else:
                                        # Fallback
                                        center_y = i * 20
                                        center_x = 0
                                        left_x = 0
                                        right_x = 100
                                        top_y = i * 20
                                        bottom_y = (i + 1) * 20
                                        text_height = 20
                                        text_width = 100
                                else:
                                    # Fallback for unknown format
                                    center_y = i * 20
                                    center_x = 0
                                    left_x = 0
                                    right_x = 100
                                    top_y = i * 20
                                    bottom_y = (i + 1) * 20
                                    text_height = 20
                                    text_width = 100
                            else:
                                # No bbox available
                                center_y = i * 20
                                center_x = 0
                                left_x = 0
                                right_x = 100
                                top_y = i * 20
                                bottom_y = (i + 1) * 20
                                text_height = 20
                                text_width = 100
                            
                            text_items.append({
                                'index': i,
                                'text': item_text.strip(),
                                'bbox': bbox,
                                'confidence': confidence,
                                'center_y': center_y,
                                'center_x': center_x,
                                'left_x': left_x,
                                'right_x': right_x,
                                'top_y': top_y,
                                'bottom_y': bottom_y,
                                'text_height': text_height,
                                'text_width': text_width,
                                'is_numeric': item_text.strip().replace('.', '').replace('$', '').replace(',', '').isdigit(),
                                'is_price': bool(re.match(r'^\$?\d+\.\d{2}$', item_text.strip())),
                                'word_count': len(item_text.strip().split())
                            })
                            
                        except Exception as e:
                            print(f"   ⚠️  Bbox parsing failed for item {i}: {e}")
                            # Add with default positioning
                            text_items.append({
                                'index': i,
                                'text': item_text.strip(),
                                'bbox': bbox,
                                'confidence': confidence,
                                'center_y': i * 20,
                                'center_x': 0,
                                'left_x': 0,
                                'right_x': 100,
                                'top_y': i * 20,
                                'bottom_y': (i + 1) * 20,
                                'text_height': 20,
                                'text_width': 100,
                                'is_numeric': False,
                                'is_price': False,
                                'word_count': len(item_text.strip().split())
                            })
                
                # Advanced line grouping algorithm
                if text_items:
                    print(f"   📊 Processing {len(text_items)} text items for line grouping")
                    
                    # Sort by Y coordinate first
                    text_items.sort(key=lambda x: (x['center_y'], x['center_x']))
                    
                    # Calculate dynamic line threshold based on text heights
                    text_heights = [item['text_height'] for item in text_items if item['text_height'] > 5]
                    if text_heights:
                        avg_height = np.mean(text_heights)
                        median_height = np.median(text_heights)
                        # Use smaller of average or median, but at least 8 pixels
                        line_threshold = max(8, min(avg_height * 0.4, median_height * 0.5))
                    else:
                        line_threshold = 10
                    
                    print(f"   📏 Calculated line threshold: {line_threshold:.1f}px (avg height: {np.mean(text_heights) if text_heights else 'N/A'})")
                    
                    # Group items into lines using clustering approach
                    grouped_lines = []
                    remaining_items = text_items.copy()
                    
                    while remaining_items:
                        # Start new line with topmost remaining item
                        anchor_item = remaining_items[0]
                        current_line = [anchor_item]
                        remaining_items.remove(anchor_item)
                        
                        # Find all items that belong to this line
                        items_to_remove = []
                        for item in remaining_items:
                            # Check if item's Y-coordinate overlaps with current line
                            line_top = min(x['top_y'] for x in current_line)
                            line_bottom = max(x['bottom_y'] for x in current_line)
                            line_center_y = (line_top + line_bottom) / 2
                            
                            # Item belongs to line if its vertical center is within the line bounds
                            # or if there's significant vertical overlap
                            item_center_y = item['center_y']
                            y_distance = abs(item_center_y - line_center_y)
                            
                            # Check for vertical overlap
                            item_top = item['top_y']
                            item_bottom = item['bottom_y']
                            
                            vertical_overlap = max(0, min(line_bottom, item_bottom) - max(line_top, item_top))
                            overlap_ratio = vertical_overlap / min(item['text_height'], line_bottom - line_top) if min(item['text_height'], line_bottom - line_top) > 0 else 0
                            
                            # Add to line if close vertically or has significant overlap
                            if y_distance <= line_threshold or overlap_ratio > 0.3:
                                current_line.append(item)
                                items_to_remove.append(item)
                        
                        # Remove items that were added to current line
                        for item in items_to_remove:
                            remaining_items.remove(item)
                        
                        # Sort current line by X coordinate
                        current_line.sort(key=lambda x: x['center_x'])
                        grouped_lines.append(current_line)
                    
                    print(f"   📋 Grouped into {len(grouped_lines)} lines")
                    
                    # Process each line to create readable text
                    line_texts = []
                    for line_idx, line_items in enumerate(grouped_lines):
                        # Create properly spaced text for the line
                        line_text = self.reconstruct_line_text(line_items)
                        
                        # Calculate line statistics
                        line_confidences = [item['confidence'] for item in line_items if isinstance(item['confidence'], (int, float))]
                        avg_confidence = np.mean(line_confidences) if line_confidences else 'unknown'
                        
                        # Calculate line bounding box
                        line_bbox = [
                            min(item['left_x'] for item in line_items),
                            min(item['top_y'] for item in line_items),
                            max(item['right_x'] for item in line_items) - min(item['left_x'] for item in line_items),
                            max(item['bottom_y'] for item in line_items) - min(item['top_y'] for item in line_items)
                        ]
                        
                        line_analysis.append({
                            'line_number': line_idx + 1,
                            'text': line_text,
                            'confidence': avg_confidence,
                            'bbox': line_bbox,
                            'char_count': len(line_text),
                            'word_count': len(line_text.split()),
                            'items_in_line': len(line_items),
                            'y_position': np.mean([item['center_y'] for item in line_items]),
                            'item_details': line_items
                        })
                        
                        line_texts.append(line_text)
                    
                    # Join lines with newlines
                    text_content = '\n'.join(line_texts)
                    
                    result_details = {
                        'engine': engine_name,
                        'total_detections': len(result),
                        'valid_text_items': len(text_items),
                        'grouped_lines': len(grouped_lines),
                        'combined_text': text_content,
                        'line_analysis': line_analysis,
                        'statistics': {
                            'total_lines': len(line_analysis),
                            'total_characters': sum(line['char_count'] for line in line_analysis),
                            'total_words': sum(line['word_count'] for line in line_analysis),
                            'avg_confidence': np.mean([line['confidence'] for line in line_analysis if isinstance(line['confidence'], (int, float))]) if line_analysis else 0,
                            'line_threshold_used': line_threshold,
                            'avg_text_height': np.mean(text_heights) if text_heights else 0
                        }
                    }
            
            # Handle other result formats (string, object, etc.)
            elif hasattr(result, 'text'):
                text_content = result.text
                lines = text_content.split('\n')
                for i, line in enumerate(lines):
                    if line.strip():
                        line_analysis.append({
                            'line_number': i + 1,
                            'text': line.strip(),
                            'confidence': getattr(result, 'confidence', 'unknown'),
                            'bbox': getattr(result, 'bbox', None),
                            'char_count': len(line.strip()),
                            'word_count': len(line.strip().split())
                        })
                
                result_details = {
                    'engine': engine_name,
                    'text': text_content,
                    'confidence': getattr(result, 'confidence', None),
                    'bbox': getattr(result, 'bbox', None),
                    'line_analysis': line_analysis,
                    'statistics': {
                        'total_lines': len(line_analysis),
                        'total_characters': sum(line['char_count'] for line in line_analysis),
                        'total_words': sum(line['word_count'] for line in line_analysis)
                    }
                }
                
            elif isinstance(result, str):
                text_content = result
                lines = text_content.split('\n')
                for i, line in enumerate(lines):
                    if line.strip():
                        line_analysis.append({
                            'line_number': i + 1,
                            'text': line.strip(),
                            'confidence': 'unknown',
                            'bbox': None,
                            'char_count': len(line.strip()),
                            'word_count': len(line.strip().split())
                        })
                        
                result_details = {
                    'engine': engine_name,
                    'text': text_content,
                    'line_analysis': line_analysis,
                    'statistics': {
                        'total_lines': len(line_analysis),
                        'total_characters': sum(line['char_count'] for line in line_analysis),
                        'total_words': sum(line['word_count'] for line in line_analysis)
                    }
                }
                
            elif isinstance(result, dict):
                text_content = result.get('text', '')
                result_details = result.copy()
                result_details['engine'] = engine_name
                
        except Exception as e:
            print(f"   ⚠️  Text extraction error: {e}")
            import traceback
            traceback.print_exc()
            result_details = {
                'engine': engine_name,
                'error': str(e),
                'raw_result_type': str(type(result))
            }
        
        return text_content, result_details

    def reconstruct_line_text(self, line_items):
        """Reconstruct line text with proper spacing for receipt format"""
        if not line_items:
            return ""
        
        if len(line_items) == 1:
            return line_items[0]['text']
        
        # Sort by X coordinate
        sorted_items = sorted(line_items, key=lambda x: x['center_x'])
        
        # Analyze the line to determine if it's a receipt item line (product + price)
        prices = [item for item in sorted_items if item.get('is_price', False) or 
                (item.get('is_numeric', False) and '.' in item['text'])]
        
        # If we have what looks like prices, try to format as receipt line
        if prices:
            # Get the rightmost price
            rightmost_price = max(prices, key=lambda x: x['center_x'])
            
            # Get all non-price items to the left of the rightmost price
            description_items = [item for item in sorted_items 
                            if item['center_x'] < rightmost_price['center_x'] and 
                            not item.get('is_price', False)]
            
            if description_items:
                # Build description part
                description_parts = []
                for i, item in enumerate(description_items):
                    if i > 0:
                        prev_item = description_items[i-1]
                        gap = item['left_x'] - prev_item['right_x']
                        if gap > 20:  # Significant gap
                            description_parts.append("  ")
                        else:
                            description_parts.append(" ")
                    description_parts.append(item['text'])
                
                description = ''.join(description_parts).strip()
                
                # Calculate spacing to price
                if description_items:
                    last_desc_item = max(description_items, key=lambda x: x['right_x'])
                    price_gap = rightmost_price['left_x'] - last_desc_item['right_x']
                    
                    # Determine spacing based on gap size
                    if price_gap > 100:
                        spacing = "    "  # Large gap - multiple spaces
                    elif price_gap > 50:
                        spacing = "   "   # Medium gap
                    elif price_gap > 20:
                        spacing = "  "    # Small gap
                    else:
                        spacing = " "     # Minimal gap
                else:
                    spacing = "  "
                
                return f"{description}{spacing}{rightmost_price['text']}"
        
        # Fallback: simple spacing based on gaps
        result_parts = []
        for i, item in enumerate(sorted_items):
            if i > 0:
                prev_item = sorted_items[i-1]
                gap = item['left_x'] - prev_item['right_x']
                if gap > 50:
                    result_parts.append("   ")  # Large gap
                elif gap > 20:
                    result_parts.append("  ")   # Medium gap
                else:
                    result_parts.append(" ")    # Small gap
            result_parts.append(item['text'])
        
        return ''.join(result_parts)
    
    def save_line_by_line_text(self, result_details, engine_name):
        """Save readable line-by-line text file"""
        try:
            if 'line_analysis' not in result_details:
                return
            
            filename = f"lines_{engine_name}.txt"
            output_file = self.output_dir / filename
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"=== {engine_name.upper()} OCR RESULTS - LINE BY LINE ===\n")
                f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                if 'statistics' in result_details:
                    stats = result_details['statistics']
                    f.write("STATISTICS:\n")
                    f.write(f"  Total Lines: {stats.get('total_lines', 0)}\n")
                    f.write(f"  Total Characters: {stats.get('total_characters', 0)}\n")
                    f.write(f"  Total Words: {stats.get('total_words', 0)}\n")
                    if 'avg_confidence' in stats:
                        f.write(f"  Average Confidence: {stats['avg_confidence']:.3f}\n")
                    f.write("\n")
                
                f.write("DETECTED LINES:\n")
                f.write("-" * 50 + "\n")
                
                for line_info in result_details['line_analysis']:
                    line_num = line_info['line_number']
                    text = line_info['text']
                    confidence = line_info.get('confidence', 'unknown')
                    char_count = line_info.get('char_count', 0)
                    word_count = line_info.get('word_count', 0)
                    
                    f.write(f"Line {line_num:3d}: {text}\n")
                    if isinstance(confidence, (int, float)):
                        f.write(f"           Confidence: {confidence:.3f}, Chars: {char_count}, Words: {word_count}\n")
                    else:
                        f.write(f"           Confidence: {confidence}, Chars: {char_count}, Words: {word_count}\n")
                    f.write("\n")
                
                # Full combined text
                if 'combined_text' in result_details:
                    f.write("\n" + "=" * 50 + "\n")
                    f.write("COMBINED TEXT:\n")
                    f.write("=" * 50 + "\n")
                    f.write(result_details['combined_text'])
            
            print(f"   📝 Saved line-by-line: {filename}")
            
        except Exception as e:
            print(f"   ⚠️  Failed to save line-by-line text: {e}")
    
    def test_your_pipeline(self):
        print("🔍 Testing Your Enhanced Pipeline with Line Analysis")
        print("=" * 60)
        
        # Step 1: Load Image
        try:
            image = cv2.imread(self.test_image_path)
            if image is None:
                raise ValueError("Failed to load image")
            self.log_step("1. Image Loading", True, image)
            self.save_result("01_original_image.jpg", image)
        except Exception as e:
            self.log_step("1. Image Loading", False, error=str(e))
            return
        
        # Step 2: Load Configuration
        try:
            from src.utils.config import load_config
            config = load_config()
            self.log_step("2. Configuration Loading", True, config)
            
            config_info = {
                'type': str(type(config)),
                'attributes': [attr for attr in dir(config) if not attr.startswith('_')]
            }
            self.save_result("02_config_info.json", config_info)
            
        except Exception as e:
            self.log_step("2. Configuration Loading", False, error=str(e))
            config = None
        
        # Step 3: Image Processing
        processed_image = image
        try:
            from src.utils import image_utils
            available_methods = [method for method in dir(image_utils) if not method.startswith('_')]
            print(f"   Available image_utils methods: {available_methods}")
            
            for method_name in ['preprocess', 'enhance_image', 'process_image', 'convert_to_grayscale']:
                if hasattr(image_utils, method_name):
                    try:
                        method = getattr(image_utils, method_name)
                        processed_result = method(image)
                        
                        extracted_img = self.get_image_from_result(processed_result)
                        if extracted_img is not None:
                            processed_image = extracted_img
                        
                        print(f"   ✅ Used method: {method_name}")
                        break
                    except Exception as e:
                        print(f"   ⚠️  Method {method_name} failed: {e}")
            
            self.log_step("3. Image Processing", True, processed_image)
            self.save_result("03_processed_image.jpg", processed_image)
            
        except Exception as e:
            self.log_step("3. Image Processing", False, error=str(e))
        
        # Step 4: Quality Analysis
        quality_result = None
        try:
            from src.preprocessing.quality_analyzer import QualityAnalyzer
            analyzer = QualityAnalyzer()
            
            available_methods = [method for method in dir(analyzer) if not method.startswith('_') and callable(getattr(analyzer, method))]
            print(f"   Available analyzer methods: {available_methods}")
            
            for method_name in ['analyze_image', 'analyze_quality', 'analyze', 'assess_quality']:
                if hasattr(analyzer, method_name):
                    try:
                        method = getattr(analyzer, method_name)
                        quality_result = method(processed_image)
                        print(f"   ✅ Used method: {method_name}")
                        break
                    except Exception as e:
                        print(f"   ⚠️  Method {method_name} failed: {e}")
            
            if quality_result is not None:
                self.log_step("4. Quality Analysis", True, quality_result)
                self.save_result("04_quality_analysis.json", quality_result)
            else:
                self.log_step("4. Quality Analysis", False, error="No quality result obtained")
            
        except Exception as e:
            self.log_step("4. Quality Analysis", False, error=str(e))
        
        # Step 5: Image Enhancement
        enhancement_result = None
        enhanced_image = processed_image
        
        try:
            from src.preprocessing.image_enhancer import ImageEnhancer
            enhancer = ImageEnhancer()
            
            available_methods = [method for method in dir(enhancer) if not method.startswith('_') and callable(getattr(enhancer, method))]
            print(f"   Available enhancer methods: {available_methods}")
            
            for method_name in ['enhance_image', 'enhance', 'process', 'preprocess']:
                if hasattr(enhancer, method_name):
                    try:
                        method = getattr(enhancer, method_name)
                        enhancement_result = method(processed_image)
                        print(f"   ✅ Used method: {method_name}")
                        break
                    except Exception as e:
                        print(f"   ⚠️  Method {method_name} failed: {e}")
            
            if enhancement_result is not None:
                self.log_step("5. Image Enhancement", True, enhancement_result)
                
                extracted_enhanced = self.get_image_from_result(enhancement_result)
                if extracted_enhanced is not None:
                    enhanced_image = extracted_enhanced
                    self.save_result("05_enhanced_image.jpg", enhanced_image)
                
                if hasattr(enhancement_result, '__dict__'):
                    enhancement_info = {
                        'type': str(type(enhancement_result)),
                        'attributes': [attr for attr in dir(enhancement_result) if not attr.startswith('_')]
                    }
                    self.save_result("05_enhancement_info.json", enhancement_info)
            else:
                self.log_step("5. Image Enhancement", False, error="No enhancement result obtained")
            
        except Exception as e:
            self.log_step("5. Image Enhancement", False, error=str(e))
        
        # Step 6: Text Detection (with fixed config handling)
        try:
            # Create proper config for TextDetector
            text_detection_config = {
                'method': 'auto',
                'confidence_threshold': 0.6,
                'nms_threshold': 0.3,
                'min_region_area': 150
            }
            
            from src.preprocessing.text_detector import TextDetector
            detector = TextDetector(text_detection_config)  # Pass the config properly
            
            available_methods = [method for method in dir(detector) if not method.startswith('_') and callable(getattr(detector, method))]
            print(f"   Available detector methods: {available_methods}")
            
            text_regions = []
            for method_name in ['detect_text_regions', 'detect', 'find_text_regions', 'detect_text']:
                if hasattr(detector, method_name):
                    try:
                        method = getattr(detector, method_name)
                        text_regions = method(enhanced_image)
                        print(f"   ✅ Used method: {method_name}")
                        if text_regions:
                            break
                    except Exception as e:
                        print(f"   ⚠️  Method {method_name} failed: {e}")
            
            if text_regions:
                self.log_step("6. Text Detection", True, text_regions)
                self.save_result("06_text_regions.json", text_regions)
                
                try:
                    detection_image = enhanced_image.copy()
                    boxes_drawn = 0
                    
                    for i, region in enumerate(text_regions):
                        try:
                            if isinstance(region, (list, tuple)) and len(region) >= 4:
                                x, y, w, h = region[:4]
                                cv2.rectangle(detection_image, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)
                                boxes_drawn += 1
                            elif isinstance(region, dict) and 'bbox' in region:
                                bbox = region['bbox']
                                if len(bbox) >= 4:
                                    x, y, w, h = bbox[:4]
                                    cv2.rectangle(detection_image, (int(x), int(y)), (int(x+w), int(y+h)), (0, 255, 0), 2)
                                    boxes_drawn += 1
                        except Exception as e:
                            print(f"   ⚠️  Failed to draw box {i}: {e}")
                    
                    if boxes_drawn > 0:
                        self.save_result("06_detection_visualization.jpg", detection_image)
                        print(f"   📦 Drew {boxes_drawn} bounding boxes")
                        
                except Exception as e:
                    print(f"   ⚠️  Visualization failed: {e}")
            else:
                self.log_step("6. Text Detection", False, error="No text regions detected")
            
        except Exception as e:
            self.log_step("6. Text Detection", False, error=str(e))
        
        # Step 7: Enhanced OCR Testing with Line Analysis
        ocr_results = {}
        
        if not isinstance(enhanced_image, np.ndarray):
            print(f"   ⚠️  Enhanced image is {type(enhanced_image)}, using original image")
            enhanced_image = image
        
        print(f"   📸 Using image with shape: {enhanced_image.shape}")
        
        # Test Tesseract with enhanced analysis
        try:
            from src.engines.tesseract_engine import TesseractEngine
            engine = TesseractEngine()
            
            tesseract_result = None
            for method_name in ['process_image', 'extract_text', 'process']:
                if hasattr(engine, method_name):
                    try:
                        print(f"   🔄 Trying Tesseract method: {method_name}")
                        method = getattr(engine, method_name)
                        tesseract_result = method(enhanced_image)
                        print(f"   ✅ Used method: {method_name}")
                        break
                    except Exception as e:
                        print(f"   ⚠️  Method {method_name} failed: {e}")
            
            if tesseract_result is not None:
                text_content, result_details = self.extract_text_from_ocr_result(tesseract_result, 'tesseract')
                ocr_results['tesseract'] = tesseract_result
                
                self.log_step("7a. Tesseract OCR", True, tesseract_result)
                self.save_result("07a_tesseract_text.txt", text_content)
                self.save_result("07a_tesseract_details.json", result_details)
                self.save_line_by_line_text(result_details, 'tesseract')
                
                # Create line visualization
                vis_image = self.create_line_visualization(enhanced_image, tesseract_result, 'tesseract')
                self.save_result("07a_tesseract_lines_visualization.jpg", vis_image)
                
                if text_content:
                    print(f"   📝 Lines detected: {result_details.get('statistics', {}).get('total_lines', 'unknown')}")
            else:
                self.log_step("7a. Tesseract OCR", False, error="No result returned")
                
        except Exception as e:
            self.log_step("7a. Tesseract OCR", False, error=str(e))
        
        # Test EasyOCR with enhanced analysis
        try:
            from src.engines.easyocr_engine import EasyOCREngine
            engine = EasyOCREngine()
            
            easyocr_result = None
            for method_name in ['process_image', 'extract_text', 'process']:
                if hasattr(engine, method_name):
                    try:
                        method = getattr(engine, method_name)
                        easyocr_result = method(enhanced_image)
                        print(f"   ✅ Used method: {method_name}")
                        break
                    except Exception as e:
                        print(f"   ⚠️  Method {method_name} failed: {e}")
            
            if easyocr_result is not None:
                text_content, result_details = self.extract_text_from_ocr_result(easyocr_result, 'easyocr')
                ocr_results['easyocr'] = easyocr_result
                
                self.log_step("7b. EasyOCR", True, easyocr_result)
                self.save_result("07b_easyocr_text.txt", text_content)
                self.save_result("07b_easyocr_details.json", result_details)
                self.save_line_by_line_text(result_details, 'easyocr')
                
                vis_image = self.create_line_visualization(enhanced_image, easyocr_result, 'easyocr')
                self.save_result("07b_easyocr_lines_visualization.jpg", vis_image)
                
                if text_content:
                    print(f"   📝 Lines detected: {result_details.get('statistics', {}).get('total_lines', 'unknown')}")
            else:
                self.log_step("7b. EasyOCR", False, error="No result returned")
                
        except Exception as e:
            self.log_step("7b. EasyOCR", False, error=str(e))
        
        # Test PaddleOCR with enhanced analysis
        try:
            from src.engines.paddleocr_engine import PaddleOCREngine
            engine = PaddleOCREngine()
            
            paddleocr_result = None
            for method_name in ['process_image', 'extract_text', 'process']:
                if hasattr(engine, method_name):
                    try:
                        method = getattr(engine, method_name)
                        paddleocr_result = method(enhanced_image)
                        print(f"   ✅ Used method: {method_name}")
                        break
                    except Exception as e:
                        print(f"   ⚠️  Method {method_name} failed: {e}")
            
            if paddleocr_result is not None:
                text_content, result_details = self.extract_text_from_ocr_result(paddleocr_result, 'paddleocr')
                ocr_results['paddleocr'] = paddleocr_result
                
                self.log_step("7c. PaddleOCR", True, paddleocr_result)
                self.save_result("07c_paddleocr_text.txt", text_content)
                self.save_result("07c_paddleocr_details.json", result_details)
                self.save_line_by_line_text(result_details, 'paddleocr')
                
                vis_image = self.create_line_visualization(enhanced_image, paddleocr_result, 'paddleocr')
                self.save_result("07c_paddleocr_lines_visualization.jpg", vis_image)
                
                if text_content:
                    print(f"   📝 Lines detected: {result_details.get('statistics', {}).get('total_lines', 'unknown')}")
            else:
                self.log_step("7c. PaddleOCR", False, error="No result returned")
                
        except Exception as e:
            self.log_step("7c. PaddleOCR", False, error=str(e))
        
        # Step 8: Engine Manager Testing
        try:
            from src.core.engine_manager import EngineManager
            manager = EngineManager()
            
            available_methods = [method for method in dir(manager) if not method.startswith('_') and callable(getattr(manager, method))]
            print(f"   Available manager methods: {available_methods}")
            
            # Check engine status
            if hasattr(manager, 'get_available_engines'):
                try:
                    available_engines = manager.get_available_engines()
                    print(f"   📋 Available engines: {available_engines}")
                except:
                    pass
            
            if hasattr(manager, 'get_initialized_engines'):
                try:
                    initialized_engines = manager.get_initialized_engines()
                    print(f"   🔧 Initialized engines: {initialized_engines}")
                except:
                    pass
            
            manager_result = None
            for method_name in ['process_with_best_engine', 'process_image', 'process', 'extract_text']:
                if hasattr(manager, method_name):
                    try:
                        method = getattr(manager, method_name)
                        manager_result = method(enhanced_image)
                        print(f"   ✅ Used method: {method_name}")
                        break
                    except Exception as e:
                        print(f"   ⚠️  Method {method_name} failed: {e}")
            
            if manager_result is not None:
                text_content, result_details = self.extract_text_from_ocr_result(manager_result, 'engine_manager')
                
                self.log_step("8. Engine Manager", True, manager_result)
                self.save_result("08_manager_text.txt", text_content)
                self.save_result("08_manager_details.json", result_details)
                
                if isinstance(manager_result, list) and len(manager_result) > 0:
                    self.save_line_by_line_text(result_details, 'engine_manager')
                    vis_image = self.create_line_visualization(enhanced_image, manager_result, 'engine_manager')
                    self.save_result("08_manager_lines_visualization.jpg", vis_image)
            else:
                self.log_step("8. Engine Manager", False, error="No result returned")
            
        except Exception as e:
            self.log_step("8. Engine Manager", False, error=str(e))
        
        # Final Summary
        print("\n" + "=" * 60)
        print("📊 ENHANCED PIPELINE TEST SUMMARY")
        print("=" * 60)
        
        if ocr_results:
            print(f"✅ OCR Engines Working: {len(ocr_results)}")
            for engine_name, result in ocr_results.items():
                text_content, result_details = self.extract_text_from_ocr_result(result, engine_name)
                
                # Get statistics
                stats = result_details.get('statistics', {})
                total_lines = stats.get('total_lines', 'unknown')
                total_chars = stats.get('total_characters', 'unknown')
                total_words = stats.get('total_words', 'unknown')
                avg_conf = stats.get('avg_confidence', 'unknown')
                
                preview = text_content[:80] + "..." if len(text_content) > 80 else text_content
                
                print(f"   {engine_name:12}: Lines: {total_lines}, Words: {total_words}, Chars: {total_chars}")
                if isinstance(avg_conf, (int, float)):
                    print(f"   {'':12}  Avg Confidence: {avg_conf:.3f}")
                print(f"   {'':12}  Preview: {preview}")
                print()
        else:
            print("❌ No OCR engines produced results")
        
        print(f"\n📁 All outputs saved to: {self.output_dir}")
        
        # List all saved files with categories
        saved_files = list(self.output_dir.glob("*"))
        if saved_files:
            print(f"📄 Files created ({len(saved_files)}):")
            
            # Categorize files
            images = [f for f in saved_files if f.suffix.lower() in ['.jpg', '.png', '.jpeg']]
            texts = [f for f in saved_files if f.name.startswith('lines_')]
            jsons = [f for f in saved_files if f.suffix == '.json']
            others = [f for f in saved_files if f not in images + texts + jsons]
            
            if images:
                print("   📸 Images:")
                for file in sorted(images):
                    print(f"      - {file.name}")
            
            if texts:
                print("   📝 Line-by-Line Text Files:")
                for file in sorted(texts):
                    print(f"      - {file.name}")
            
            if jsons:
                print("   📋 JSON Details:")
                for file in sorted(jsons):
                    print(f"      - {file.name}")
            
            if others:
                print("   📄 Other Files:")
                for file in sorted(others):
                    print(f"      - {file.name}")
        
        print("\n🔍 Check the saved files to analyze your pipeline results!")
        print("💡 Special files to check:")
        print("   - lines_*.txt: Readable line-by-line OCR results")
        print("   - *_lines_visualization.jpg: Visual line detection")
        print("   - *_details.json: Complete analysis with statistics")

if __name__ == "__main__":
    tester = EnhancedPipelineTest()
    tester.test_your_pipeline()