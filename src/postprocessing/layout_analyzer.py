# src/postprocessing/layout_analyzer.py
import re
from typing import List, Dict, Any, Tuple
from spellchecker import SpellChecker
import string
from collections import Counter

class LayoutAnalyzer:
    """Analyze and preserve document layout"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.line_height_threshold = config.get("line_height_threshold", 1.5)
        self.paragraph_gap_threshold = config.get("paragraph_gap_threshold", 2.0)
        
    def analyze_layout(self, results: List[Any], image_shape: Tuple[int, int]) -> Dict[str, Any]:
        """Analyze document layout structure"""
        if not results:
            return {}
            
        # Group results into lines
        lines = self._group_into_lines(results)
        
        # Group lines into paragraphs
        paragraphs = self._group_into_paragraphs(lines)
        
        # Detect columns
        columns = self._detect_columns(paragraphs, image_shape)
        
        return {
            "lines": lines,
            "paragraphs": paragraphs,
            "columns": columns,
            "reading_order": self._determine_reading_order(paragraphs)
        }
        
    def _group_into_lines(self, results: List[Any]) -> List[List[Any]]:
        """Group OCR results into text lines"""
        if not results:
            return []
            
        # Sort by y-coordinate
        sorted_results = sorted(results, key=lambda r: r.bbox[1] if hasattr(r, 'bbox') else 0)
        
        lines = []
        current_line = [sorted_results[0]]
        current_y = sorted_results[0].bbox[1] if hasattr(sorted_results[0], 'bbox') else 0
        
        for result in sorted_results[1:]:
            y_pos = result.bbox[1] if hasattr(result, 'bbox') else 0
            height = result.bbox[3] if hasattr(result, 'bbox') else 20
            
            # Check if this result belongs to the same line
            if abs(y_pos - current_y) <= height * 0.5:  # Same line
                current_line.append(result)
            else:
                # Sort current line by x-coordinate
                current_line.sort(key=lambda r: r.bbox[0] if hasattr(r, 'bbox') else 0)
                lines.append(current_line)
                current_line = [result]
                current_y = y_pos
                
        # Add the last line
        if current_line:
            current_line.sort(key=lambda r: r.bbox[0] if hasattr(r, 'bbox') else 0)
            lines.append(current_line)
            
        return lines
        
    def _group_into_paragraphs(self, lines: List[List[Any]]) -> List[List[List[Any]]]:
        """Group lines into paragraphs based on spacing"""
        if not lines:
            return []
            
        paragraphs = []
        current_paragraph = [lines[0]]
        
        for i in range(1, len(lines)):
            prev_line = lines[i-1]
            curr_line = lines[i]
            
            # Calculate gap between lines
            if prev_line and curr_line:
                prev_bottom = max(r.bbox[1] + r.bbox[3] for r in prev_line if hasattr(r, 'bbox'))
                curr_top = min(r.bbox[1] for r in curr_line if hasattr(r, 'bbox'))
                gap = curr_top - prev_bottom
                
                # Calculate average line height
                avg_height = sum(r.bbox[3] for r in prev_line if hasattr(r, 'bbox')) / len(prev_line)
                
                # If gap is larger than threshold, start new paragraph
                if gap > avg_height * self.paragraph_gap_threshold:
                    paragraphs.append(current_paragraph)
                    current_paragraph = [curr_line]
                else:
                    current_paragraph.append(curr_line)
            else:
                current_paragraph.append(curr_line)
                
        # Add the last paragraph
        if current_paragraph:
            paragraphs.append(current_paragraph)
            
        return paragraphs
        
    def _detect_columns(self, paragraphs: List[List[List[Any]]], image_shape: Tuple[int, int]) -> List[Dict[str, Any]]:
        """Detect column structure in the document"""
        if not paragraphs:
            return []
            
        # Analyze x-coordinates of paragraphs
        paragraph_positions = []
        for paragraph in paragraphs:
            if paragraph and paragraph[0]:  # Check if paragraph has lines
                # Get leftmost and rightmost positions
                left_x = min(min(r.bbox[0] for r in line if hasattr(r, 'bbox')) for line in paragraph)
                right_x = max(max(r.bbox[0] + r.bbox[2] for r in line if hasattr(r, 'bbox')) for line in paragraph)
                paragraph_positions.append((left_x, right_x))
                
        if not paragraph_positions:
            return []
            
        # Cluster paragraphs by x-position
        columns = []
        image_width = image_shape[1]
        
        # Simple column detection based on x-position clustering
        left_positions = [pos[0] for pos in paragraph_positions]
        
        # If there's significant variation in left positions, likely multiple columns
        if len(set(pos // 50 for pos in left_positions)) > 1:  # Group by 50-pixel bins
            # Sort unique left positions
            unique_positions = sorted(set(pos // 50 for pos in left_positions))
            
            for i, pos_group in enumerate(unique_positions):
                column_start = pos_group * 50
                column_end = image_width if i == len(unique_positions) - 1 else (unique_positions[i + 1] * 50)
                
                columns.append({
                    "index": i,
                    "x_start": column_start,
                    "x_end": column_end,
                    "width": column_end - column_start
                })
        else:
            # Single column document
            columns.append({
                "index": 0,
                "x_start": 0,
                "x_end": image_width,
                "width": image_width
            })
            
        return columns
        
    def _determine_reading_order(self, paragraphs: List[List[List[Any]]]) -> List[int]:
        """Determine optimal reading order for paragraphs"""
        if not paragraphs:
            return []
            
        # Create paragraph metadata
        paragraph_info = []
        for i, paragraph in enumerate(paragraphs):
            if paragraph and paragraph[0]:
                # Get paragraph bounding box
                all_results = [result for line in paragraph for result in line]
                if all_results:
                    left = min(r.bbox[0] for r in all_results if hasattr(r, 'bbox'))
                    top = min(r.bbox[1] for r in all_results if hasattr(r, 'bbox'))
                    right = max(r.bbox[0] + r.bbox[2] for r in all_results if hasattr(r, 'bbox'))
                    bottom = max(r.bbox[1] + r.bbox[3] for r in all_results if hasattr(r, 'bbox'))
                    
                    paragraph_info.append({
                        "index": i,
                        "left": left,
                        "top": top,
                        "right": right,
                        "bottom": bottom,
                        "center_x": (left + right) / 2,
                        "center_y": (top + bottom) / 2
                    })
                    
        # Sort by reading order (top to bottom, left to right)
        reading_order = sorted(paragraph_info, key=lambda p: (p["top"], p["left"]))
        
        return [p["index"] for p in reading_order]
