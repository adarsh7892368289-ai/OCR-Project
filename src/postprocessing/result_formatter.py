# src/postprocessing/result_formatter.py
import re
from typing import List, Dict, Any, Tuple
from spellchecker import SpellChecker
import string
from collections import Counter
class ResultFormatter:
    """Format final OCR results into various output formats"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.preserve_formatting = config.get("preserve_formatting", True)
        self.include_confidence = config.get("include_confidence", False)
        
    def format_as_text(self, results: List[Any], layout_info: Dict[str, Any] = None) -> str:
        """Format results as plain text"""
        if not results:
            return ""
            
        if layout_info and "paragraphs" in layout_info:
            return self._format_with_layout(layout_info)
        else:
            return self._format_simple(results)
            
    def _format_with_layout(self, layout_info: Dict[str, Any]) -> str:
        """Format text preserving document layout"""
        paragraphs = layout_info.get("paragraphs", [])
        reading_order = layout_info.get("reading_order", list(range(len(paragraphs))))
        
        formatted_text = []
        
        for para_idx in reading_order:
            if para_idx < len(paragraphs):
                paragraph = paragraphs[para_idx]
                paragraph_text = []
                
                for line in paragraph:
                    line_text = " ".join(result.text for result in line if hasattr(result, 'text'))
                    if line_text.strip():
                        paragraph_text.append(line_text.strip())
                        
                if paragraph_text:
                    formatted_text.append("\n".join(paragraph_text))
                    
        return "\n\n".join(formatted_text)
        
    def _format_simple(self, results: List[Any]) -> str:
        """Simple text formatting without layout analysis"""
        return " ".join(result.text for result in results if hasattr(result, 'text') and result.text.strip())
        
    def format_as_json(self, results: List[Any], layout_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """Format results as JSON structure"""
        formatted_results = []
        
        for result in results:
            result_dict = {
                "text": getattr(result, 'text', ''),
                "bbox": getattr(result, 'bbox', (0, 0, 0, 0))
            }
            
            if self.include_confidence and hasattr(result, 'confidence'):
                result_dict["confidence"] = result.confidence
                
            formatted_results.append(result_dict)
            
        output = {
            "results": formatted_results,
            "full_text": self.format_as_text(results, layout_info)
        }
        
        if layout_info:
            output["layout"] = layout_info
            
        return output
        
    def format_as_hocr(self, results: List[Any], image_shape: Tuple[int, int]) -> str:
        """Format results as hOCR HTML"""
        hocr_template = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.0 Transitional//EN"
    "http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd">
<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en" lang="en">
<head>
    <meta http-equiv="Content-Type" content="text/html; charset=utf-8"/>
    <meta name='ocr-system' content='modern-ocr-system'/>
    <meta name='ocr-capabilities' content='ocr_page ocr_carea ocr_par ocr_line ocrx_word'/>
</head>
<body>
    <div class='ocr_page' id='page_1' title='bbox 0 0 {width} {height}'>
{content}
    </div>
</body>
</html>"""
        
        content_lines = []
        for i, result in enumerate(results):
            if hasattr(result, 'bbox') and hasattr(result, 'text'):
                x, y, w, h = result.bbox
                confidence = int(getattr(result, 'confidence', 0.95) * 100)
                
                word_html = f"""        <span class='ocrx_word' id='word_{i+1}_{1}' title='bbox {x} {y} {x+w} {y+h}; x_size 20; x_descenders 4; x_ascenders 8; x_conf {confidence}'>{result.text}</span>"""
                content_lines.append(word_html)
                
        content = "\n".join(content_lines)
        
        return hocr_template.format(
            width=image_shape[1],
            height=image_shape[0],
            content=content
        )