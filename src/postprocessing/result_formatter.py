
# src/postprocessing/result_formatter.py
from typing import List, Dict, Any
import json
from ..core.base_engine import OCRResult

class ResultFormatter:
    """Format OCR results into different output formats"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.preserve_formatting = config.get("preserve_formatting", True)
        self.include_confidence = config.get("include_confidence", False)
        
    def format_as_json(self, results: List[OCRResult], layout_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """Format results as JSON"""
        try:
            formatted = {
                "words": [],
                "lines": layout_info.get("lines", []) if layout_info else [],
                "paragraphs": layout_info.get("paragraphs", []) if layout_info else []
            }
            
            for result in results:
                word_data = {
                    "text": result.text,
                    "bbox": result.bbox
                }
                
                if self.include_confidence:
                    word_data["confidence"] = result.confidence
                    
                formatted["words"].append(word_data)
                
            return formatted
        except Exception as e:
            print(f"JSON formatting error: {e}")
            return {"words": [], "lines": [], "paragraphs": []}
        
    def format_as_hocr(self, results: List[OCRResult], image_stats: Dict[str, Any]) -> str:
        """Format results as hOCR HTML"""
        try:
            width = image_stats.get("width", 1000)
            height = image_stats.get("height", 1000)
            
            html = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN"
    "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
<html xmlns="http://www.w3.org/1999/xhtml">
<head>
<title></title>
<meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
<meta name='ocr-system' content='Modern OCR System' />
</head>
<body>
<div class='ocr_page' id='page_1' title='bbox 0 0 {width} {height}'>
"""
            
            for i, result in enumerate(results):
                x, y, w, h = result.bbox
                confidence = int(result.confidence * 100) if self.include_confidence else 95
                text = result.text.replace('"', '&quot;').replace('<', '&lt;').replace('>', '&gt;')
                
                html += f"""<span class='ocrx_word' id='word_{i}' title='bbox {x} {y} {x+w} {y+h}; x_wconf {confidence}'>{text}</span> """
                
            html += """
</div>
</body>
</html>"""
            return html
        except Exception as e:
            print(f"hOCR formatting error: {e}")
            return "<html><body>Error formatting hOCR</body></html>"
        
    def format_as_csv(self, results: List[OCRResult]) -> str:
        """Format results as CSV"""
        try:
            lines = ["text,x,y,width,height,confidence\n"]
            
            for result in results:
                x, y, w, h = result.bbox
                text = result.text.replace('"', '""')  # Escape quotes
                line = f'"{text}",{x},{y},{w},{h},{result.confidence:.3f}\n'
                lines.append(line)
                
            return ''.join(lines)
        except Exception as e:
            print(f"CSV formatting error: {e}")
            return "text,x,y,width,height,confidence\n"