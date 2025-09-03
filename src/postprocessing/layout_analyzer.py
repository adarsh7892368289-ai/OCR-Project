# src/postprocessing/layout_analyzer.py
from typing import List, Dict, Any, Tuple
import numpy as np
from ..core.base_engine import OCRResult

class LayoutAnalyzer:
    """Analyze document layout and structure"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.line_height_threshold = config.get("line_height_threshold", 1.5)
        self.paragraph_gap_threshold = config.get("paragraph_gap_threshold", 2.0)
        
    def analyze_layout(self, results: List[OCRResult], image_shape: Tuple[int, int]) -> Dict[str, Any]:
        """Analyze document layout"""
        if not results:
            return {"lines": [], "paragraphs": [], "columns": 1, "reading_order": []}
            
        try:
            # Group results into lines
            lines = self._group_into_lines(results)
            
            # Group lines into paragraphs
            paragraphs = self._group_into_paragraphs(lines)
            
            # Detect columns
            columns = self._detect_columns(results, image_shape)
            
            # Determine reading order
            reading_order = self._determine_reading_order(results)
            
            return {
                "lines": [self._line_to_dict(line) for line in lines],
                "paragraphs": [self._paragraph_to_dict(para) for para in paragraphs],
                "columns": columns,
                "reading_order": reading_order
            }
        except Exception as e:
            print(f"Layout analysis error: {e}")
            return {"lines": [], "paragraphs": [], "columns": 1, "reading_order": []}
        
    def _group_into_lines(self, results: List[OCRResult]) -> List[List[OCRResult]]:
        """Group results into text lines"""
        if not results:
            return []
            
        # Sort by Y coordinate
        sorted_results = sorted(results, key=lambda r: r.bbox[1])
        
        lines = []
        current_line = [sorted_results[0]]
        
        for result in sorted_results[1:]:
            # Check if result belongs to current line
            current_y = current_line[0].bbox[1]
            current_height = max(r.bbox[3] for r in current_line)
            result_y = result.bbox[1]
            
            # If Y difference is less than line height threshold, add to current line
            if abs(result_y - current_y) < current_height * self.line_height_threshold:
                current_line.append(result)
            else:
                # Sort current line by X coordinate and start new line
                current_line.sort(key=lambda r: r.bbox[0])
                lines.append(current_line)
                current_line = [result]
                
        # Don't forget the last line
        if current_line:
            current_line.sort(key=lambda r: r.bbox[0])
            lines.append(current_line)
            
        return lines
        
    def _group_into_paragraphs(self, lines: List[List[OCRResult]]) -> List[List[List[OCRResult]]]:
        """Group lines into paragraphs"""
        if not lines:
            return []
            
        paragraphs = []
        current_paragraph = [lines[0]]
        
        for i in range(1, len(lines)):
            try:
                # Calculate gap between lines
                prev_line_bottom = max(result.bbox[1] + result.bbox[3] for result in lines[i-1])
                curr_line_top = min(result.bbox[1] for result in lines[i])
                gap = curr_line_top - prev_line_bottom
                
                # Average line height for context
                all_heights = [result.bbox[3] for result in lines[i-1] + lines[i] if result.bbox[3] > 0]
                avg_height = np.mean(all_heights) if all_heights else 20
                
                # If gap is larger than threshold, start new paragraph
                if gap > avg_height * self.paragraph_gap_threshold:
                    paragraphs.append(current_paragraph)
                    current_paragraph = [lines[i]]
                else:
                    current_paragraph.append(lines[i])
            except Exception as e:
                print(f"Paragraph grouping error: {e}")
                current_paragraph.append(lines[i])
                
        # Don't forget the last paragraph
        if current_paragraph:
            paragraphs.append(current_paragraph)
            
        return paragraphs
        
    def _detect_columns(self, results: List[OCRResult], image_shape: Tuple[int, int]) -> int:
        """Detect number of columns"""
        if not results:
            return 1
            
        try:
            # Simple column detection based on X coordinates
            x_positions = [result.bbox[0] for result in results if result.bbox[2] > 0]
            
            if len(x_positions) < 5:  # Too few data points
                return 1
            
            # Use histogram to find column boundaries
            hist, bins = np.histogram(x_positions, bins=min(20, len(x_positions)))
            
            # Find peaks in histogram (potential column starts)
            peaks = []
            for i in range(1, len(hist) - 1):
                if hist[i] > hist[i-1] and hist[i] > hist[i+1] and hist[i] > np.mean(hist):
                    peaks.append(bins[i])
                    
            # Estimate number of columns
            return max(1, len(peaks))
        except Exception as e:
            print(f"Column detection error: {e}")
            return 1
        
    def _determine_reading_order(self, results: List[OCRResult]) -> List[int]:
        """Determine reading order of text regions"""
        try:
            # Simple left-to-right, top-to-bottom order
            sorted_indices = sorted(range(len(results)), 
                                  key=lambda i: (results[i].bbox[1], results[i].bbox[0]))
            return sorted_indices
        except Exception as e:
            print(f"Reading order error: {e}")
            return list(range(len(results)))
        
    def _line_to_dict(self, line: List[OCRResult]) -> Dict[str, Any]:
        """Convert line to dictionary representation"""
        if not line:
            return {"text": "", "bbox": (0, 0, 0, 0), "confidence": 0.0, "word_count": 0}
            
        try:
            text = " ".join(result.text for result in line)
            bbox = self._get_combined_bbox(line)
            confidences = [result.confidence for result in line if result.confidence > 0]
            confidence = np.mean(confidences) if confidences else 0.0
            
            return {
                "text": text,
                "bbox": bbox,
                "confidence": confidence,
                "word_count": len(line)
            }
        except Exception as e:
            print(f"Line conversion error: {e}")
            return {"text": "", "bbox": (0, 0, 0, 0), "confidence": 0.0, "word_count": 0}
        
    def _paragraph_to_dict(self, paragraph: List[List[OCRResult]]) -> Dict[str, Any]:
        """Convert paragraph to dictionary representation"""
        if not paragraph:
            return {"text": "", "bbox": (0, 0, 0, 0), "confidence": 0.0, "line_count": 0, "word_count": 0}
            
        try:
            # Flatten paragraph
            all_results = [result for line in paragraph for result in line]
            text = "\n".join(" ".join(result.text for result in line) for line in paragraph)
            bbox = self._get_combined_bbox(all_results)
            confidences = [result.confidence for result in all_results if result.confidence > 0]
            confidence = np.mean(confidences) if confidences else 0.0
            
            return {
                "text": text,
                "bbox": bbox,
                "confidence": confidence,
                "line_count": len(paragraph),
                "word_count": len(all_results)
            }
        except Exception as e:
            print(f"Paragraph conversion error: {e}")
            return {"text": "", "bbox": (0, 0, 0, 0), "confidence": 0.0, "line_count": 0, "word_count": 0}
        
    def _get_combined_bbox(self, results: List[OCRResult]) -> Tuple[int, int, int, int]:
        """Get combined bounding box for multiple results"""
        if not results:
            return (0, 0, 0, 0)
            
        try:
            valid_results = [r for r in results if r.bbox[2] > 0 and r.bbox[3] > 0]
            if not valid_results:
                return (0, 0, 0, 0)
                
            x_min = min(result.bbox[0] for result in valid_results)
            y_min = min(result.bbox[1] for result in valid_results)
            x_max = max(result.bbox[0] + result.bbox[2] for result in valid_results)
            y_max = max(result.bbox[1] + result.bbox[3] for result in valid_results)
            
            return (x_min, y_min, x_max - x_min, y_max - y_min)
        except Exception as e:
            print(f"Bbox calculation error: {e}")
            return (0, 0, 0, 0)