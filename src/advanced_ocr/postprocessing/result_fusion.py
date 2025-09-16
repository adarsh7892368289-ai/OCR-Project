"""
Enhanced Result Formatter with Multi-Format Output Support
Step 5: Advanced Post-processing Implementation

Features:
- Multiple output formats (JSON, XML, HTML, PDF, CSV, Markdown)
- Structured document representation
- Layout-aware formatting
- Quality indicators
- Export capabilities
"""

import json
import xml.etree.ElementTree as ET
from xml.dom import minidom
import csv
import html
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import time
import logging
from enum import Enum
import re
import statistics

try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

from advanced_ocr.core.base_engine import OCRResult, DocumentResult, TextRegion, BoundingBox
from advanced_ocr.config import ConfigManager
from advanced_ocr.utils.logger import OCRLogger
from advanced_ocr.postprocessing.structure_extractor import StructureExtractor, StructureType, DocumentStructure

logger = get_logger(__name__)


class OutputFormat(Enum):
    """Supported output formats"""
    JSON = "json"
    XML = "xml"
    HTML = "html"
    PDF = "pdf"
    CSV = "csv"
    MARKDOWN = "markdown"
    TEXT = "text"
    STRUCTURED_JSON = "structured_json"


class StructureLevel(Enum):
    """Level of structure detail in output"""
    BASIC = "basic"          # Just text and confidence
    DETAILED = "detailed"    # Include regions, bounding boxes
    COMPLETE = "complete"    # Full structure analysis
    LAYOUT_AWARE = "layout_aware"  # Include reading order, layout info


@dataclass
class FormattedDocument:
    """Formatted document result"""
    content: Union[str, Dict, List]
    format_type: OutputFormat
    metadata: Dict[str, Any] = field(default_factory=dict)
    processing_time: float = 0.0
    structure_level: StructureLevel = StructureLevel.BASIC
    quality_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'content': self.content,
            'format_type': self.format_type.value,
            'metadata': self.metadata,
            'processing_time': self.processing_time,
            'structure_level': self.structure_level.value,
            'quality_score': self.quality_score
        }


@dataclass
class FormatConfig:
    """Configuration for specific output format"""
    include_confidence: bool = True
    include_bounding_boxes: bool = False
    include_metadata: bool = True
    pretty_print: bool = True
    max_line_length: int = 80
    custom_styles: Dict[str, Any] = field(default_factory=dict)


class EnhancedResultFormatter:
    """
    Advanced result formatter with multi-format output support
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = ConfigManager(config_path).get_section('result_formatter', {})
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        
        # Default formatting configurations
        self.default_configs = {
            OutputFormat.JSON: FormatConfig(
                include_confidence=True,
                include_bounding_boxes=True,
                include_metadata=True,
                pretty_print=True
            ),
            OutputFormat.XML: FormatConfig(
                include_confidence=True,
                include_bounding_boxes=False,
                include_metadata=True,
                pretty_print=True
            ),
            OutputFormat.HTML: FormatConfig(
                include_confidence=True,
                include_bounding_boxes=False,
                include_metadata=True,
                pretty_print=True,
                custom_styles={
                    'confidence_colors': True,
                    'responsive_layout': True
                }
            ),
            OutputFormat.CSV: FormatConfig(
                include_confidence=True,
                include_bounding_boxes=True,
                include_metadata=False
            ),
            OutputFormat.MARKDOWN: FormatConfig(
                include_confidence=False,
                include_bounding_boxes=False,
                include_metadata=True,
                max_line_length=80
            )
        }
        
        # Initialize structure extractor for layout-aware formatting
        self.structure_extractor = StructureExtractor(self.config.get('structure_extractor', {}))
        
        # Statistics
        self.stats = {
            'documents_formatted': 0,
            'formats_generated': {},
            'avg_processing_time': 0.0
        }
        
        self.logger.info("Enhanced result formatter initialized")
    
    def format_result(
        self,
        ocr_result: OCRResult,
        output_format: OutputFormat,
        structure_level: StructureLevel = StructureLevel.DETAILED,
        custom_config: Optional[FormatConfig] = None
    ) -> FormattedDocument:
        """
        Format OCR result in specified format
        
        Args:
            ocr_result: OCR result to format
            output_format: Desired output format
            structure_level: Level of structure detail
            custom_config: Custom formatting configuration
            
        Returns:
            FormattedDocument with formatted content
        """
        start_time = time.time()
        
        try:
            # Get configuration
            config = custom_config or self.default_configs.get(
                output_format, 
                FormatConfig()
            )
            
            # Extract structure if needed
            document_structure = None
            if structure_level in [StructureLevel.COMPLETE, StructureLevel.LAYOUT_AWARE]:
                document_structure = self.structure_extractor.extract_structure(ocr_result)
            
            # Format according to requested format
            content = self._format_content(
                ocr_result, 
                output_format, 
                structure_level, 
                config,
                document_structure
            )
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(ocr_result, document_structure)
            
            # Generate metadata
            metadata = self._generate_metadata(
                ocr_result, 
                output_format, 
                structure_level,
                document_structure
            )
            
            processing_time = time.time() - start_time
            
            # Update statistics
            self.stats['documents_formatted'] += 1
            format_key = output_format.value
            self.stats['formats_generated'][format_key] = self.stats['formats_generated'].get(format_key, 0) + 1
            
            formatted_doc = FormattedDocument(
                content=content,
                format_type=output_format,
                metadata=metadata,
                processing_time=processing_time,
                structure_level=structure_level,
                quality_score=quality_score
            )
            
            self.logger.info(
                f"Document formatted as {output_format.value} "
                f"({structure_level.value}) in {processing_time:.3f}s"
            )
            
            return formatted_doc
            
        except Exception as e:
            self.logger.error(f"Error formatting result: {e}")
            processing_time = time.time() - start_time
            
            return FormattedDocument(
                content=f"Error formatting document: {str(e)}",
                format_type=output_format,
                processing_time=processing_time,
                quality_score=0.0
            )
    
    def _format_content(
        self,
        ocr_result: OCRResult,
        output_format: OutputFormat,
        structure_level: StructureLevel,
        config: FormatConfig,
        document_structure: Optional[DocumentStructure] = None
    ) -> Union[str, Dict, List]:
        """Format content according to specified format"""
        
        if output_format == OutputFormat.JSON:
            return self._format_json(ocr_result, structure_level, config, document_structure)
        elif output_format == OutputFormat.STRUCTURED_JSON:
            return self._format_structured_json(ocr_result, structure_level, config, document_structure)
        elif output_format == OutputFormat.XML:
            return self._format_xml(ocr_result, structure_level, config, document_structure)
        elif output_format == OutputFormat.HTML:
            return self._format_html(ocr_result, structure_level, config, document_structure)
        elif output_format == OutputFormat.CSV:
            return self._format_csv(ocr_result, structure_level, config)
        elif output_format == OutputFormat.MARKDOWN:
            return self._format_markdown(ocr_result, structure_level, config, document_structure)
        elif output_format == OutputFormat.TEXT:
            return self._format_text(ocr_result, structure_level, config, document_structure)
        else:
            return str(ocr_result.full_text or "")
    
    def _format_json(
        self,
        ocr_result: OCRResult,
        structure_level: StructureLevel,
        config: FormatConfig,
        document_structure: Optional[DocumentStructure] = None
    ) -> Dict[str, Any]:
        """Format as JSON"""
        
        result = {
            "full_text": ocr_result.full_text or "",
            "confidence": ocr_result.overall_confidence
        }
        
        # Add regions if detailed or complete
        if structure_level in [StructureLevel.DETAILED, StructureLevel.COMPLETE, StructureLevel.LAYOUT_AWARE]:
            regions = []
            for region in (ocr_result.regions or []):
                region_data = {
                    "text": region.full_text if hasattr(region, 'full_text') else str(region),
                    "confidence": getattr(region, 'confidence', 1.0)
                }
                
                if config.include_bounding_boxes and hasattr(region, 'bbox'):
                    bbox = region.bbox
                    region_data["bounding_box"] = {
                        "x": bbox.x,
                        "y": bbox.y,
                        "width": bbox.width,
                        "height": bbox.height
                    } if bbox else None
                
                regions.append(region_data)
            
            result["regions"] = regions
        
        # Add structure information if available
        if document_structure and structure_level in [StructureLevel.COMPLETE, StructureLevel.LAYOUT_AWARE]:
            structure_data = {
                "elements": [],
                "reading_order": document_structure.reading_order,
                "confidence": document_structure.confidence
            }
            
            for element in document_structure.elements:
                elem_data = {
                    "type": element.element_type.value if hasattr(element, 'element_type') else "unknown",
                    "content": element.content if hasattr(element, 'content') else str(element),
                    "confidence": getattr(element, 'confidence', 1.0)
                }
                structure_data["elements"].append(elem_data)
            
            result["document_structure"] = structure_data
        
        return result
    
    def _format_structured_json(
        self,
        ocr_result: OCRResult,
        structure_level: StructureLevel,
        config: FormatConfig,
        document_structure: Optional[DocumentStructure] = None
    ) -> Dict[str, Any]:
        """Format as structured JSON with semantic organization"""
        
        result = {
            "document": {
                "full_text": ocr_result.full_text or "",
                "confidence": ocr_result.overall_confidence,
                "processing_timestamp": time.time()
            }
        }
        
        if document_structure:
            # Organize content by structure type
            structured_content = {}
            
            for element in document_structure.elements:
                elem_type = element.element_type.value if hasattr(element, 'element_type') else "paragraph"
                if elem_type not in structured_content:
                    structured_content[elem_type] = []
                
                elem_data = {
                    "content": element.content if hasattr(element, 'content') else str(element),
                    "confidence": getattr(element, 'confidence', 1.0)
                }
                
                if config.include_bounding_boxes and hasattr(element, 'bounding_box'):
                    bbox = element.bounding_box
                    elem_data["position"] = {
                        "x": bbox.x,
                        "y": bbox.y,
                        "width": bbox.width,
                        "height": bbox.height
                    } if bbox else None
                
                structured_content[elem_type].append(elem_data)
            
            result["structured_content"] = structured_content
            result["reading_order"] = document_structure.reading_order
        
        return result
    
    def _format_xml(
        self,
        ocr_result: OCRResult,
        structure_level: StructureLevel,
        config: FormatConfig,
        document_structure: Optional[DocumentStructure] = None
    ) -> str:
        """Format as XML"""
        
        root = ET.Element("document")
        root.set("confidence", str(ocr_result.overall_confidence))
        
        # Full text element
        full_text_elem = ET.SubElement(root, "full_text")
        full_text_elem.text = ocr_result.full_text or ""
        
        # Regions
        if structure_level in [StructureLevel.DETAILED, StructureLevel.COMPLETE, StructureLevel.LAYOUT_AWARE]:
            regions_elem = ET.SubElement(root, "regions")
            
            for i, region in enumerate(ocr_result.regions or []):
                region_elem = ET.SubElement(regions_elem, "region")
                region_elem.set("id", str(i))
                region_elem.set("confidence", str(getattr(region, 'confidence', 1.0)))
                
                text_elem = ET.SubElement(region_elem, "text")
                text_elem.text = region.full_text if hasattr(region, 'full_text') else str(region)
                
                if config.include_bounding_boxes and hasattr(region, 'bbox') and region.bbox:
                    bbox_elem = ET.SubElement(region_elem, "bounding_box")
                    bbox_elem.set("x", str(region.bbox.x))
                    bbox_elem.set("y", str(region.bbox.y))
                    bbox_elem.set("width", str(region.bbox.width))
                    bbox_elem.set("height", str(region.bbox.height))
        
        # Document structure
        if document_structure and structure_level in [StructureLevel.COMPLETE, StructureLevel.LAYOUT_AWARE]:
            structure_elem = ET.SubElement(root, "document_structure")
            structure_elem.set("confidence", str(document_structure.confidence))
            
            elements_elem = ET.SubElement(structure_elem, "elements")
            for i, element in enumerate(document_structure.elements):
                elem_xml = ET.SubElement(elements_elem, "element")
                elem_xml.set("id", str(i))
                elem_xml.set("type", element.element_type.value if hasattr(element, 'element_type') else "unknown")
                elem_xml.set("confidence", str(getattr(element, 'confidence', 1.0)))
                elem_xml.text = element.content if hasattr(element, 'content') else str(element)
        
        # Pretty print if requested
        if config.pretty_print:
            rough_string = ET.tostring(root, 'utf-8')
            reparsed = minidom.parseString(rough_string)
            return reparsed.toprettyxml(indent="  ")
        else:
            return ET.tostring(root, encoding='unicode')
    
    def _format_html(
        self,
        ocr_result: OCRResult,
        structure_level: StructureLevel,
        config: FormatConfig,
        document_structure: Optional[DocumentStructure] = None
    ) -> str:
        """Format as HTML"""
        
        html_content = []
        
        # HTML header
        html_content.append('<!DOCTYPE html>')
        html_content.append('<html lang="en">')
        html_content.append('<head>')
        html_content.append('<meta charset="UTF-8">')
        html_content.append('<meta name="viewport" content="width=device-width, initial-scale=1.0">')
        html_content.append('<title>OCR Result</title>')
        
        # CSS styles
        html_content.append('<style>')
        html_content.append(self._get_html_styles(config))
        html_content.append('</style>')
        html_content.append('</head>')
        html_content.append('<body>')
        
        # Document header
        html_content.append('<div class="document-header">')
        html_content.append('<h1>OCR Document</h1>')
        html_content.append(f'<p class="confidence">Overall Confidence: {ocr_result.overall_confidence:.2%}</p>')
        html_content.append('</div>')
        
        # Main content
        html_content.append('<div class="document-content">')
        
        if document_structure and structure_level in [StructureLevel.COMPLETE, StructureLevel.LAYOUT_AWARE]:
            # Structure-aware formatting
            for element in document_structure.elements:
                elem_type = element.element_type.value if hasattr(element, 'element_type') else "paragraph"
                confidence = getattr(element, 'confidence', 1.0)
                content = html.escape(element.content if hasattr(element, 'content') else str(element))
                
                confidence_class = self._get_confidence_class(confidence)
                
                if elem_type == "heading":
                    html_content.append(f'<h2 class="element heading {confidence_class}">{content}</h2>')
                elif elem_type == "list":
                    html_content.append(f'<ul class="element list {confidence_class}"><li>{content}</li></ul>')
                else:
                    html_content.append(f'<p class="element paragraph {confidence_class}">{content}</p>')
                
                if config.include_confidence:
                    html_content.append(f'<span class="confidence-badge">{confidence:.1%}</span>')
        else:
            # Simple text formatting
            if ocr_result.regions:
                for region in ocr_result.regions:
                    text = region.full_text if hasattr(region, 'full_text') else str(region)
                    confidence = getattr(region, 'confidence', 1.0)
                    confidence_class = self._get_confidence_class(confidence)
                    
                    html_content.append(f'<p class="region {confidence_class}">')
                    html_content.append(html.escape(text))
                    if config.include_confidence:
                        html_content.append(f' <span class="confidence-badge">{confidence:.1%}</span>')
                    html_content.append('</p>')
            else:
                html_content.append(f'<p>{html.escape(ocr_result.full_text or "")}</p>')
        
        html_content.append('</div>')
        html_content.append('</body>')
        html_content.append('</html>')
        
        return '\n'.join(html_content)
    
    def _format_csv(
        self,
        ocr_result: OCRResult,
        structure_level: StructureLevel,
        config: FormatConfig
    ) -> str:
        """Format as CSV"""
        
        import io
        output = io.StringIO()
        
        # Define headers
        headers = ["text", "confidence"]
        if config.include_bounding_boxes:
            headers.extend(["x", "y", "width", "height"])
        
        writer = csv.writer(output)
        writer.writerow(headers)
        
        # Write regions
        if ocr_result.regions:
            for region in ocr_result.regions:
                row = [
                    region.full_text if hasattr(region, 'full_text') else str(region),
                    getattr(region, 'confidence', 1.0)
                ]
                
                if config.include_bounding_boxes:
                    if hasattr(region, 'bbox') and region.bbox:
                        row.extend([region.bbox.x, region.bbox.y, region.bbox.width, region.bbox.height])
                    else:
                        row.extend([0, 0, 0, 0])
                
                writer.writerow(row)
        else:
            # Single row with full text
            row = [ocr_result.full_text or "", ocr_result.overall_confidence]
            if config.include_bounding_boxes:
                row.extend([0, 0, 0, 0])
            writer.writerow(row)
        
        return output.getvalue()
    
    def _format_markdown(
        self,
        ocr_result: OCRResult,
        structure_level: StructureLevel,
        config: FormatConfig,
        document_structure: Optional[DocumentStructure] = None
    ) -> str:
        """Format as Markdown"""
        
        content = []
        
        # Document header
        content.append("# OCR Document")
        content.append("")
        
        if config.include_metadata:
            content.append("## Document Information")
            content.append(f"- **Overall Confidence:** {ocr_result.overall_confidence:.2%}")
            content.append(f"- **Total Regions:** {len(ocr_result.regions) if ocr_result.regions else 0}")
            content.append("")
        
        # Content
        content.append("## Content")
        content.append("")
        
        if document_structure and structure_level in [StructureLevel.COMPLETE, StructureLevel.LAYOUT_AWARE]:
            # Structure-aware formatting
            for element in document_structure.elements:
                elem_type = element.element_type.value if hasattr(element, 'element_type') else "paragraph"
                element_content = element.content if hasattr(element, 'content') else str(element)
                
                if elem_type == "heading":
                    content.append(f"## {element_content}")
                elif elem_type == "list":
                    content.append(f"- {element_content}")
                else:
                    # Wrap long lines
                    wrapped_content = self._wrap_text(element_content, config.max_line_length)
                    content.append(wrapped_content)
                
                content.append("")
        else:
            # Simple formatting
            if ocr_result.full_text:
                wrapped_text = self._wrap_text(ocr_result.full_text, config.max_line_length)
                content.append(wrapped_text)
        
        return "\n".join(content)
    
    def _format_text(
        self,
        ocr_result: OCRResult,
        structure_level: StructureLevel,
        config: FormatConfig,
        document_structure: Optional[DocumentStructure] = None
    ) -> str:
        """Format as plain text"""
        
        if document_structure and structure_level == StructureLevel.LAYOUT_AWARE:
            # Respect reading order
            content_parts = []
            for idx in document_structure.reading_order:
                if idx < len(document_structure.elements):
                    element = document_structure.elements[idx]
                    content_parts.append(element.content if hasattr(element, 'content') else str(element))
            return "\n\n".join(content_parts)
        
        return ocr_result.full_text or ""
    
    def _get_html_styles(self, config: FormatConfig) -> str:
        """Generate CSS styles for HTML output"""
        
        styles = """
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f9f9f9;
        }
        
        .document-header {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        
        .document-content {
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .element {
            margin-bottom: 15px;
            padding: 10px;
            border-radius: 4px;
            position: relative;
        }
        
        .confidence-badge {
            font-size: 0.8em;
            padding: 2px 6px;
            border-radius: 12px;
            margin-left: 8px;
            font-weight: bold;
        }
        
        .confidence-high { background-color: #d4edda; border-left: 4px solid #28a745; }
        .confidence-medium { background-color: #fff3cd; border-left: 4px solid #ffc107; }
        .confidence-low { background-color: #f8d7da; border-left: 4px solid #dc3545; }
        
        .confidence-high .confidence-badge { background-color: #28a745; color: white; }
        .confidence-medium .confidence-badge { background-color: #ffc107; color: black; }
        .confidence-low .confidence-badge { background-color: #dc3545; color: white; }
        """
        
        if config.custom_styles.get('responsive_layout', False):
            styles += """
            @media (max-width: 600px) {
                body { padding: 10px; }
                .document-header, .document-content { padding: 15px; }
            }
            """
        
        return styles
    
    def _get_confidence_class(self, confidence: float) -> str:
        """Get CSS class based on confidence level"""
        if confidence >= 0.8:
            return "confidence-high"
        elif confidence >= 0.5:
            return "confidence-medium"
        else:
            return "confidence-low"
    
    def _wrap_text(self, text: str, max_length: int) -> str:
        """Wrap text to specified length"""
        if not text or max_length <= 0:
            return text
        
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            word_length = len(word)
            if current_length + word_length + 1 <= max_length:
                current_line.append(word)
                current_length += word_length + (1 if current_line else 0)
            else:
                if current_line:
                    lines.append(" ".join(current_line))
                current_line = [word]
                current_length = word_length
        
        if current_line:
            lines.append(" ".join(current_line))
        
        return "\n".join(lines)
    
    def _calculate_quality_score(
        self, 
        ocr_result: OCRResult, 
        document_structure: Optional[DocumentStructure] = None
    ) -> float:
        """Calculate overall quality score for the document"""
        
        scores = []
        
        # Confidence-based score
        scores.append(ocr_result.overall_confidence)
        
        # Structure-based score
        if document_structure:
            structure_confidence = document_structure.confidence
            scores.append(structure_confidence)
            
            # Content organization score
            if document_structure.elements:
                element_scores = [
                    getattr(elem, 'confidence', 1.0) 
                    for elem in document_structure.elements
                ]
                if element_scores:
                    scores.append(statistics.mean(element_scores))
        
        # Text quality heuristics
        if ocr_result.full_text:
            text_quality = self._assess_text_quality_simple(ocr_result.full_text)
            scores.append(text_quality)
        
        return statistics.mean(scores) if scores else 0.0
    
    def _assess_text_quality_simple(self, text: str) -> float:
        """Simple text quality assessment"""
        if not text:
            return 0.0
        
        quality = 1.0
        
        # Check character distribution
        alpha_ratio = len(re.findall(r'[a-zA-Z]', text)) / len(text)
        if alpha_ratio < 0.3:
            quality *= 0.7
        
        # Check for excessive special characters
        special_ratio = len(re.findall(r'[^a-zA-Z0-9\s\-.,!?()"]', text)) / len(text)
        if special_ratio > 0.2:
            quality *= 0.8
        
        # Check word length distribution
        words = text.split()
        if words:
            avg_word_length = statistics.mean(len(word) for word in words)
            if avg_word_length < 2 or avg_word_length > 12:
                quality *= 0.9
        
        return quality
    
    def _generate_metadata(
        self,
        ocr_result: OCRResult,
        output_format: OutputFormat,
        structure_level: StructureLevel,
        document_structure: Optional[DocumentStructure] = None
    ) -> Dict[str, Any]:
        """Generate document metadata"""
        
        metadata = {
            'format': output_format.value,
            'structure_level': structure_level.value,
            'timestamp': time.time(),
            'total_regions': len(ocr_result.regions) if ocr_result.regions else 0,
            'overall_confidence': ocr_result.overall_confidence,
            'text_length': len(ocr_result.full_text) if ocr_result.full_text else 0
        }
        
        if document_structure:
            metadata['structure_elements'] = len(document_structure.elements)
            metadata['structure_confidence'] = document_structure.confidence
            
            # Element type distribution
            element_types = {}
            for element in document_structure.elements:
                elem_type = element.element_type.value if hasattr(element, 'element_type') else "unknown"
                element_types[elem_type] = element_types.get(elem_type, 0) + 1
            metadata['element_types'] = element_types
        
        # Confidence distribution
        if ocr_result.regions:
            confidences = [getattr(r, 'confidence', 1.0) for r in ocr_result.regions]
            if confidences:
                metadata['confidence_stats'] = {
                    'mean': statistics.mean(confidences),
                    'median': statistics.median(confidences),
                    'std': statistics.stdev(confidences) if len(confidences) > 1 else 0,
                    'min': min(confidences),
                    'max': max(confidences)
                }