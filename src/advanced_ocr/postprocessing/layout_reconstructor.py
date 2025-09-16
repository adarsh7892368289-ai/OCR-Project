"""
Advanced OCR System - Layout Reconstruction Module
ONLY JOB: Reconstruct document layout structure
DEPENDENCIES: results.py, image_utils.py, config.py
USED BY: text_processor.py ONLY
"""

import re
import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum

from ..results import OCRResult, TextRegion, Word, Line, Paragraph, Block, Page, BoundingBox
from ..config import OCRConfig
from ..utils.image_utils import CoordinateTransformer


class LayoutType(Enum):
    """Types of document layouts"""
    SINGLE_COLUMN = "single_column"
    MULTI_COLUMN = "multi_column"
    TABLE = "table"
    MIXED = "mixed"
    FORM = "form"


class ReadingOrder(Enum):
    """Reading order strategies"""
    LEFT_TO_RIGHT_TOP_TO_BOTTOM = "ltr_ttb"
    RIGHT_TO_LEFT_TOP_TO_BOTTOM = "rtl_ttb"
    TOP_TO_BOTTOM_LEFT_TO_RIGHT = "ttb_ltr"
    COLUMN_WISE = "column_wise"


@dataclass
class LayoutAnalysis:
    """Analysis of document layout structure"""
    layout_type: LayoutType
    reading_order: ReadingOrder
    column_count: int
    average_line_height: float
    average_word_spacing: float
    text_alignment: str  # left, center, right, justified
    has_tables: bool
    has_forms: bool


@dataclass
class ReconstructionMetrics:
    """Metrics from layout reconstruction process"""
    original_regions: int
    words_created: int
    lines_created: int
    paragraphs_created: int
    blocks_created: int
    confidence: float
    layout_analysis: LayoutAnalysis


class LayoutReconstructor:
    """
    ONLY RESPONSIBILITY: Reconstruct document layout structure
    
    Receives OCRResult from text_processor.py and reconstructs text hierarchy
    (word→line→paragraph→block→page). Preserves original spacing and structure.
    Does NOT perform text cleaning, confidence analysis, or result fusion.
    """
    
    def __init__(self, config: OCRConfig):
        self.config = config
        self.coordinate_transformer = CoordinateTransformer()
        
        # Layout reconstruction parameters
        self.line_height_tolerance = 0.3  # 30% tolerance for line height variations
        self.word_spacing_threshold = 1.5  # Multiple of average character width
        self.paragraph_spacing_threshold = 1.8  # Multiple of average line height
        self.column_gap_threshold = 2.0  # Multiple of average word spacing
        
        # Text alignment detection thresholds
        self.alignment_tolerance = 0.1  # 10% of line width
        
    def reconstruct_layout(self, result: OCRResult) -> Tuple[OCRResult, ReconstructionMetrics]:
        """
        Reconstruct hierarchical layout from flat OCR result
        
        Args:
            result: OCR result with flat text regions
            
        Returns:
            Tuple of (enhanced_result_with_hierarchy, reconstruction_metrics)
        """
        if not result.regions:
            # No regions - create basic structure from text
            enhanced_result = self._create_basic_structure(result)
            metrics = ReconstructionMetrics(
                original_regions=0,
                words_created=len(result.text.split()),
                lines_created=1,
                paragraphs_created=1,
                blocks_created=1,
                confidence=0.5,
                layout_analysis=LayoutAnalysis(
                    layout_type=LayoutType.SINGLE_COLUMN,
                    reading_order=ReadingOrder.LEFT_TO_RIGHT_TOP_TO_BOTTOM,
                    column_count=1,
                    average_line_height=0.0,
                    average_word_spacing=0.0,
                    text_alignment="left",
                    has_tables=False,
                    has_forms=False
                )
            )
            return enhanced_result, metrics
        
        # Perform layout analysis
        layout_analysis = self._analyze_layout(result.regions)
        
        # Build hierarchical structure
        hierarchical_regions = self._build_hierarchy(result.regions, layout_analysis)
        
        # Create enhanced result
        enhanced_result = OCRResult(
            text=result.text,
            confidence=result.confidence,
            regions=hierarchical_regions,
            engine_name=result.engine_name,
            confidence_metrics=result.confidence_metrics
        )
        
        # Calculate metrics
        metrics = self._calculate_metrics(result.regions, hierarchical_regions, layout_analysis)
        
        return enhanced_result, metrics
    
    def _create_basic_structure(self, result: OCRResult) -> OCRResult:
        """Create basic hierarchical structure when no regions are available"""
        if not result.text.strip():
            return result
        
        # Create words from text
        words = []
        word_texts = result.text.split()
        x_pos = 0
        y_pos = 0
        char_width = 10  # Estimated character width
        line_height = 20  # Estimated line height
        
        for i, word_text in enumerate(word_texts):
            # Estimate bounding box
            word_width = len(word_text) * char_width
            bbox = BoundingBox(x_pos, y_pos, word_width, line_height)
            
            word = Word(
                text=word_text,
                bbox=bbox,
                confidence=result.confidence
            )
            words.append(word)
            
            x_pos += word_width + char_width  # Add space
            
            # Simple line wrapping estimation
            if x_pos > 500:  # Estimated page width
                x_pos = 0
                y_pos += line_height
        
        # Create single line containing all words
        if words:
            line_bbox = self._calculate_bounding_box([w.bbox for w in words])
            line = Line(
                text=result.text,
                bbox=line_bbox,
                confidence=result.confidence,
                words=words
            )
            
            # Create single paragraph containing the line
            paragraph = Paragraph(
                text=result.text,
                bbox=line_bbox,
                confidence=result.confidence,
                lines=[line]
            )
            
            # Create single block containing the paragraph
            block = Block(
                text=result.text,
                bbox=line_bbox,
                confidence=result.confidence,
                paragraphs=[paragraph]
            )
            
            # Create page containing the block
            page = Page(
                text=result.text,
                bbox=line_bbox,
                confidence=result.confidence,
                blocks=[block]
            )
            
            return OCRResult(
                text=result.text,
                confidence=result.confidence,
                regions=[page],
                engine_name=result.engine_name,
                confidence_metrics=result.confidence_metrics
            )
        
        return result
    
    def _analyze_layout(self, regions: List[TextRegion]) -> LayoutAnalysis:
        """Analyze the overall layout structure of the document"""
        if not regions:
            return LayoutAnalysis(
                layout_type=LayoutType.SINGLE_COLUMN,
                reading_order=ReadingOrder.LEFT_TO_RIGHT_TOP_TO_BOTTOM,
                column_count=1,
                average_line_height=0.0,
                average_word_spacing=0.0,
                text_alignment="left",
                has_tables=False,
                has_forms=False
            )
        
        # Calculate basic metrics
        line_heights = []
        word_spacings = []
        y_positions = []
        x_positions = []
        
        for region in regions:
            if hasattr(region, 'bbox'):
                line_heights.append(region.bbox.height)
                y_positions.append(region.bbox.y)
                x_positions.append(region.bbox.x)
        
        avg_line_height = np.mean(line_heights) if line_heights else 20
        avg_word_spacing = self._estimate_word_spacing(regions)
        
        # Detect layout type
        layout_type = self._detect_layout_type(regions, x_positions)
        
        # Detect column count
        column_count = self._detect_column_count(regions, x_positions)
        
        # Detect reading order
        reading_order = self._detect_reading_order(regions)
        
        # Detect text alignment
        text_alignment = self._detect_text_alignment(regions)
        
        # Detect special structures
        has_tables = self._detect_tables(regions)
        has_forms = self._detect_forms(regions)
        
        return LayoutAnalysis(
            layout_type=layout_type,
            reading_order=reading_order,
            column_count=column_count,
            average_line_height=avg_line_height,
            average_word_spacing=avg_word_spacing,
            text_alignment=text_alignment,
            has_tables=has_tables,
            has_forms=has_forms
        )
    
    def _estimate_word_spacing(self, regions: List[TextRegion]) -> float:
        """Estimate average word spacing"""
        spacings = []
        
        # Sort regions by y-position then x-position
        sorted_regions = sorted(regions, key=lambda r: (r.bbox.y, r.bbox.x) 
                               if hasattr(r, 'bbox') else (0, 0))
        
        for i in range(len(sorted_regions) - 1):
            current = sorted_regions[i]
            next_region = sorted_regions[i + 1]
            
            if not (hasattr(current, 'bbox') and hasattr(next_region, 'bbox')):
                continue
            
            # Check if regions are on the same line (similar y-positions)
            y_diff = abs(current.bbox.y - next_region.bbox.y)
            if y_diff < current.bbox.height * 0.5:
                # Calculate horizontal spacing
                spacing = next_region.bbox.x - (current.bbox.x + current.bbox.width)
                if spacing > 0:
                    spacings.append(spacing)
        
        return np.median(spacings) if spacings else 10.0
    
    def _detect_layout_type(self, regions: List[TextRegion], x_positions: List[float]) -> LayoutType:
        """Detect the type of document layout"""
        if not x_positions:
            return LayoutType.SINGLE_COLUMN
        
        # Check for multiple distinct x-positions (indicating columns)
        unique_x = set(round(x / 50) * 50 for x in x_positions)  # Group by 50-pixel buckets
        
        if len(unique_x) >= 3:
            return LayoutType.MULTI_COLUMN
        
        # Check for table-like structures (regular grid patterns)
        if self._has_grid_pattern(regions):
            return LayoutType.TABLE
        
        # Check for form-like structures (labels and fields)
        if self._has_form_pattern(regions):
            return LayoutType.FORM
        
        return LayoutType.SINGLE_COLUMN
    
    def _detect_column_count(self, regions: List[TextRegion], x_positions: List[float]) -> int:
        """Detect number of columns in the document"""
        if not x_positions:
            return 1
        
        # Cluster x-positions to find column boundaries
        sorted_x = sorted(set(x_positions))
        column_starts = []
        
        if not sorted_x:
            return 1
        
        column_starts.append(sorted_x[0])
        
        for x in sorted_x[1:]:
            # If there's a significant gap, it's likely a new column
            if x - column_starts[-1] > 100:  # 100 pixel threshold
                column_starts.append(x)
        
        return len(column_starts)
    
    def _detect_reading_order(self, regions: List[TextRegion]) -> ReadingOrder:
        """Detect the reading order of the document"""
        # For now, assume standard left-to-right, top-to-bottom
        # Can be enhanced with language detection and cultural conventions
        return ReadingOrder.LEFT_TO_RIGHT_TOP_TO_BOTTOM
    
    def _detect_text_alignment(self, regions: List[TextRegion]) -> str:
        """Detect predominant text alignment"""
        if not regions:
            return "left"
        
        left_aligned = 0
        center_aligned = 0
        right_aligned = 0
        
        # Get document bounds
        all_x = [r.bbox.x for r in regions if hasattr(r, 'bbox')]
        all_right = [r.bbox.x + r.bbox.width for r in regions if hasattr(r, 'bbox')]
        
        if not all_x or not all_right:
            return "left"
        
        doc_left = min(all_x)
        doc_right = max(all_right)
        doc_center = (doc_left + doc_right) / 2
        doc_width = doc_right - doc_left
        
        tolerance = doc_width * self.alignment_tolerance
        
        for region in regions:
            if not hasattr(region, 'bbox'):
                continue
                
            region_left = region.bbox.x
            region_right = region.bbox.x + region.bbox.width
            region_center = (region_left + region_right) / 2
            
            # Check alignment
            if abs(region_left - doc_left) < tolerance:
                left_aligned += 1
            elif abs(region_center - doc_center) < tolerance:
                center_aligned += 1
            elif abs(region_right - doc_right) < tolerance:
                right_aligned += 1
            else:
                left_aligned += 1  # Default to left if unclear
        
        # Return predominant alignment
        if center_aligned > max(left_aligned, right_aligned):
            return "center"
        elif right_aligned > left_aligned:
            return "right"
        else:
            return "left"
    
    def _detect_tables(self, regions: List[TextRegion]) -> bool:
        """Detect if document contains table structures"""
        return self._has_grid_pattern(regions)
    
    def _detect_forms(self, regions: List[TextRegion]) -> bool:
        """Detect if document contains form structures"""
        return self._has_form_pattern(regions)
    
    def _has_grid_pattern(self, regions: List[TextRegion]) -> bool:
        """Check if regions form a grid pattern (indicating table)"""
        if len(regions) < 6:  # Need at least 2x3 grid
            return False
        
        # Get all y-positions and x-positions
        y_positions = [r.bbox.y for r in regions if hasattr(r, 'bbox')]
        x_positions = [r.bbox.x for r in regions if hasattr(r, 'bbox')]
        
        if not y_positions or not x_positions:
            return False
        
        # Check for regular spacing in both dimensions
        unique_y = sorted(set(round(y / 20) * 20 for y in y_positions))
        unique_x = sorted(set(round(x / 20) * 20 for x in x_positions))
        
        # Look for at least 3 rows and 2 columns with regular spacing
        if len(unique_y) >= 3 and len(unique_x) >= 2:
            # Check if spacings are relatively uniform
            y_spacings = [unique_y[i+1] - unique_y[i] for i in range(len(unique_y)-1)]
            x_spacings = [unique_x[i+1] - unique_x[i] for i in range(len(unique_x)-1)]
            
            y_cv = np.std(y_spacings) / np.mean(y_spacings) if y_spacings else 1.0
            x_cv = np.std(x_spacings) / np.mean(x_spacings) if x_spacings else 1.0
            
            # If coefficient of variation is low, it's likely a grid
            return y_cv < 0.5 and x_cv < 0.5
        
        return False
    
    def _has_form_pattern(self, regions: List[TextRegion]) -> bool:
        """Check if regions form a form pattern (labels and fields)"""
        if len(regions) < 4:
            return False
        
        # Look for patterns like "Label:" followed by space or field
        form_indicators = 0
        
        for region in regions:
            if hasattr(region, 'text'):
                text = region.text.strip()
                # Check for form-like patterns
                if (text.endswith(':') or 
                    text.endswith('_') or 
                    re.match(r'.*\s*:\s*$', text) or
                    re.match(r'.*__+.*', text)):
                    form_indicators += 1
        
        # If more than 30% of regions look like form elements
        return form_indicators / len(regions) > 0.3
    
    def _build_hierarchy(self, regions: List[TextRegion], 
                        layout_analysis: LayoutAnalysis) -> List[TextRegion]:
        """Build hierarchical structure from flat regions"""
        if not regions:
            return []
        
        # Convert all regions to words first
        words = self._extract_or_create_words(regions)
        
        # Group words into lines
        lines = self._group_words_into_lines(words, layout_analysis)
        
        # Group lines into paragraphs
        paragraphs = self._group_lines_into_paragraphs(lines, layout_analysis)
        
        # Group paragraphs into blocks/columns
        blocks = self._group_paragraphs_into_blocks(paragraphs, layout_analysis)
        
        # Create page containing all blocks
        page = self._create_page(blocks)
        
        return [page]
    
    def _extract_or_create_words(self, regions: List[TextRegion]) -> List[Word]:
        """Extract words from regions or create them from text"""
        words = []
        
        for region in regions:
            if isinstance(region, Word):
                words.append(region)
            else:
                # Create words from region text
                if hasattr(region, 'text') and region.text:
                    region_words = self._split_region_into_words(region)
                    words.extend(region_words)
        
        return words
    
    def _split_region_into_words(self, region: TextRegion) -> List[Word]:
        """Split a text region into individual words"""
        if not hasattr(region, 'text') or not region.text:
            return []
        
        words = []
        word_texts = region.text.split()
        
        if not hasattr(region, 'bbox') or not word_texts:
            # Create simple words without positioning
            for word_text in word_texts:
                word = Word(
                    text=word_text,
                    bbox=BoundingBox(0, 0, len(word_text) * 10, 20),
                    confidence=getattr(region, 'confidence', 1.0)
                )
                words.append(word)
            return words
        
        # Estimate word positions within the region
        region_bbox = region.bbox
        total_chars = sum(len(w) for w in word_texts) + len(word_texts) - 1  # Include spaces
        
        x_pos = region_bbox.x
        char_width = region_bbox.width / total_chars if total_chars > 0 else 10
        
        for word_text in word_texts:
            word_width = len(word_text) * char_width
            
            word_bbox = BoundingBox(
                x=x_pos,
                y=region_bbox.y,
                width=word_width,
                height=region_bbox.height
            )
            
            word = Word(
                text=word_text,
                bbox=word_bbox,
                confidence=getattr(region, 'confidence', 1.0)
            )
            words.append(word)
            
            x_pos += word_width + char_width  # Add space
        
        return words
    
    def _group_words_into_lines(self, words: List[Word], 
                               layout_analysis: LayoutAnalysis) -> List[Line]:
        """Group words into lines based on vertical positioning"""
        if not words:
            return []
        
        # Sort words by y-position then x-position
        sorted_words = sorted(words, key=lambda w: (w.bbox.y, w.bbox.x))
        
        lines = []
        current_line_words = []
        current_y = sorted_words[0].bbox.y
        line_height_tolerance = layout_analysis.average_line_height * self.line_height_tolerance
        
        for word in sorted_words:
            # Check if word belongs to current line
            y_diff = abs(word.bbox.y - current_y)
            
            if y_diff <= line_height_tolerance and current_line_words:
                # Add to current line
                current_line_words.append(word)
            else:
                # Start new line
                if current_line_words:
                    line = self._create_line_from_words(current_line_words)
                    lines.append(line)
                
                current_line_words = [word]
                current_y = word.bbox.y
        
        # Add the last line
        if current_line_words:
            line = self._create_line_from_words(current_line_words)
            lines.append(line)
        
        return lines
    
    def _create_line_from_words(self, words: List[Word]) -> Line:
        """Create a line from a list of words"""
        if not words:
            return Line(text="", bbox=BoundingBox(0, 0, 0, 0), confidence=1.0, words=[])
        
        # Sort words by x-position
        sorted_words = sorted(words, key=lambda w: w.bbox.x)
        
        # Combine text with proper spacing
        line_text = " ".join(word.text for word in sorted_words)
        
        # Calculate line bounding box
        line_bbox = self._calculate_bounding_box([w.bbox for w in sorted_words])
        
        # Calculate average confidence
        avg_confidence = np.mean([w.confidence for w in sorted_words])
        
        return Line(
            text=line_text,
            bbox=line_bbox,
            confidence=avg_confidence,
            words=sorted_words
        )
    
    def _group_lines_into_paragraphs(self, lines: List[Line], 
                                   layout_analysis: LayoutAnalysis) -> List[Paragraph]:
        """Group lines into paragraphs based on spacing and alignment"""
        if not lines:
            return []
        
        paragraphs = []
        current_paragraph_lines = []
        
        spacing_threshold = layout_analysis.average_line_height * self.paragraph_spacing_threshold
        
        for i, line in enumerate(lines):
            current_paragraph_lines.append(line)
            
            # Check if this is the end of a paragraph
            is_paragraph_end = False
            
            if i < len(lines) - 1:
                next_line = lines[i + 1]
                
                # Check vertical spacing
                vertical_gap = next_line.bbox.y - (line.bbox.y + line.bbox.height)
                if vertical_gap > spacing_threshold:
                    is_paragraph_end = True
                
                # Check for significant indentation change
                indent_diff = abs(next_line.bbox.x - line.bbox.x)
                if indent_diff > layout_analysis.average_word_spacing * 3:
                    is_paragraph_end = True
            else:
                # Last line
                is_paragraph_end = True
            
            if is_paragraph_end and current_paragraph_lines:
                paragraph = self._create_paragraph_from_lines(current_paragraph_lines)
                paragraphs.append(paragraph)
                current_paragraph_lines = []
        
        return paragraphs
    
    def _create_paragraph_from_lines(self, lines: List[Line]) -> Paragraph:
        """Create a paragraph from a list of lines"""
        if not lines:
            return Paragraph(text="", bbox=BoundingBox(0, 0, 0, 0), confidence=1.0, lines=[])
        
        # Combine text with line breaks
        paragraph_text = "\n".join(line.text for line in lines)
        
        # Calculate paragraph bounding box
        paragraph_bbox = self._calculate_bounding_box([line.bbox for line in lines])
        
        # Calculate average confidence
        avg_confidence = np.mean([line.confidence for line in lines])
        
        return Paragraph(
            text=paragraph_text,
            bbox=paragraph_bbox,
            confidence=avg_confidence,
            lines=lines
        )
    
    def _group_paragraphs_into_blocks(self, paragraphs: List[Paragraph], 
                                    layout_analysis: LayoutAnalysis) -> List[Block]:
        """Group paragraphs into blocks/columns"""
        if not paragraphs:
            return []
        
        if layout_analysis.column_count <= 1:
            # Single column - all paragraphs in one block
            return [self._create_block_from_paragraphs(paragraphs)]
        
        # Multi-column layout - group by x-position
        column_groups = self._group_by_columns(paragraphs, layout_analysis.column_count)
        
        blocks = []
        for column_paragraphs in column_groups:
            if column_paragraphs:
                block = self._create_block_from_paragraphs(column_paragraphs)
                blocks.append(block)
        
        return blocks
    
    def _group_by_columns(self, paragraphs: List[Paragraph], 
                         column_count: int) -> List[List[Paragraph]]:
        """Group paragraphs by columns"""
        if column_count <= 1:
            return [paragraphs]
        
        # Sort paragraphs by x-position
        sorted_paragraphs = sorted(paragraphs, key=lambda p: p.bbox.x)
        
        # Divide into columns
        column_groups = [[] for _ in range(column_count)]
        
        # Simple division - can be enhanced with clustering
        for i, paragraph in enumerate(sorted_paragraphs):
            column_index = i % column_count
            column_groups[column_index].append(paragraph)
        
        return column_groups
    
    def _create_block_from_paragraphs(self, paragraphs: List[Paragraph]) -> Block:
        """Create a block from a list of paragraphs"""
        if not paragraphs:
            return Block(text="", bbox=BoundingBox(0, 0, 0, 0), confidence=1.0, paragraphs=[])
        
        # Sort paragraphs by reading order (top to bottom)
        sorted_paragraphs = sorted(paragraphs, key=lambda p: p.bbox.y)
        
        # Combine text with paragraph breaks
        block_text = "\n\n".join(paragraph.text for paragraph in sorted_paragraphs)
        
        # Calculate block bounding box
        block_bbox = self._calculate_bounding_box([p.bbox for p in sorted_paragraphs])
        
        # Calculate average confidence
        avg_confidence = np.mean([p.confidence for p in sorted_paragraphs])
        
        return Block(
            text=block_text,
            bbox=block_bbox,
            confidence=avg_confidence,
            paragraphs=sorted_paragraphs
        )
    
    def _create_page(self, blocks: List[Block]) -> Page:
        """Create a page from a list of blocks"""
        if not blocks:
            return Page(text="", bbox=BoundingBox(0, 0, 0, 0), confidence=1.0, blocks=[])
        
        # Sort blocks by reading order
        sorted_blocks = self._sort_blocks_by_reading_order(blocks)
        
        # Combine text with block separators
        page_text = "\n\n".join(block.text for block in sorted_blocks)
        
        # Calculate page bounding box
        page_bbox = self._calculate_bounding_box([b.bbox for b in sorted_blocks])
        
        # Calculate average confidence
        avg_confidence = np.mean([b.confidence for b in sorted_blocks])
        
        return Page(
            text=page_text,
            bbox=page_bbox,
            confidence=avg_confidence,
            blocks=sorted_blocks
        )
    
    def _sort_blocks_by_reading_order(self, blocks: List[Block]) -> List[Block]:
        """Sort blocks according to reading order"""
        # For multi-column layouts, sort by column then by vertical position
        # For single column, sort by vertical position
        
        # Simple top-to-bottom, left-to-right sorting
        return sorted(blocks, key=lambda b: (b.bbox.y, b.bbox.x))
    
    def _calculate_bounding_box(self, bboxes: List[BoundingBox]) -> BoundingBox:
        """Calculate encompassing bounding box from list of bounding boxes"""
        if not bboxes:
            return BoundingBox(0, 0, 0, 0)
        
        min_x = min(bbox.x for bbox in bboxes)
        min_y = min(bbox.y for bbox in bboxes)
        max_x = max(bbox.x + bbox.width for bbox in bboxes)
        max_y = max(bbox.y + bbox.height for bbox in bboxes)
        
        return BoundingBox(
            x=min_x,
            y=min_y,
            width=max_x - min_x,
            height=max_y - min_y
        )
    
    def _calculate_metrics(self, original_regions: List[TextRegion], 
                          hierarchical_regions: List[TextRegion], 
                          layout_analysis: LayoutAnalysis) -> ReconstructionMetrics:
        """Calculate metrics from the reconstruction process"""
        # Count elements at each level
        words_count = 0
        lines_count = 0
        paragraphs_count = 0
        blocks_count = 0
        
        for page in hierarchical_regions:
            if isinstance(page, Page):
                for block in page.blocks:
                    blocks_count += 1
                    for paragraph in block.paragraphs:
                        paragraphs_count += 1
                        for line in paragraph.lines:
                            lines_count += 1
                            words_count += len(line.words)
        
        # Calculate reconstruction confidence based on structure completeness
        confidence = self._calculate_reconstruction_confidence(
            original_regions, hierarchical_regions, layout_analysis
        )
        
        return ReconstructionMetrics(
            original_regions=len(original_regions),
            words_created=words_count,
            lines_created=lines_count,
            paragraphs_created=paragraphs_count,
            blocks_created=blocks_count,
            confidence=confidence,
            layout_analysis=layout_analysis
        )
    
    def _calculate_reconstruction_confidence(self, original_regions: List[TextRegion], 
                                           hierarchical_regions: List[TextRegion], 
                                           layout_analysis: LayoutAnalysis) -> float:
        """Calculate confidence in the reconstruction process"""
        confidence_factors = []
        
        # Factor 1: Coverage - how much of original content is preserved
        original_text_length = sum(len(getattr(r, 'text', '')) for r in original_regions)
        hierarchical_text_length = sum(len(getattr(r, 'text', '')) for r in hierarchical_regions)
        
        if original_text_length > 0:
            coverage_ratio = min(1.0, hierarchical_text_length / original_text_length)
            confidence_factors.append(coverage_ratio)
        else:
            confidence_factors.append(1.0)
        
        # Factor 2: Structure completeness
        has_words = any(isinstance(r, Page) and r.blocks for r in hierarchical_regions)
        has_lines = any(isinstance(r, Page) and any(b.paragraphs for b in r.blocks) 
                       for r in hierarchical_regions if isinstance(r, Page))
        has_paragraphs = any(isinstance(r, Page) and any(any(p.lines for p in b.paragraphs) 
                            for b in r.blocks) for r in hierarchical_regions if isinstance(r, Page))
        
        structure_completeness = (has_words + has_lines + has_paragraphs) / 3.0
        confidence_factors.append(structure_completeness)
        
        # Factor 3: Layout analysis confidence
        layout_confidence = 1.0
        if layout_analysis.layout_type == LayoutType.MIXED:
            layout_confidence = 0.7  # Mixed layouts are more uncertain
        elif layout_analysis.layout_type == LayoutType.SINGLE_COLUMN:
            layout_confidence = 0.9  # Simple layouts are more reliable
        else:
            layout_confidence = 0.8  # Multi-column and tables are moderately reliable
        
        confidence_factors.append(layout_confidence)
        
        # Factor 4: Region count consistency
        if len(original_regions) > 0:
            # Penalize if we have too few or too many hierarchical elements
            expected_hierarchy_ratio = 2.0  # Expect some hierarchy expansion
            actual_ratio = len(hierarchical_regions) / len(original_regions)
            ratio_confidence = 1.0 - min(1.0, abs(actual_ratio - expected_hierarchy_ratio) / 2.0)
            confidence_factors.append(ratio_confidence)
        else:
            confidence_factors.append(0.5)
        
        # Calculate weighted average
        return np.mean(confidence_factors)
    
    def get_layout_summary(self, metrics: ReconstructionMetrics) -> Dict[str, any]:
        """
        Get a summary of the layout reconstruction for debugging/analysis
        
        Args:
            metrics: Metrics from reconstruction process
            
        Returns:
            Dictionary containing layout summary
        """
        return {
            'layout_type': metrics.layout_analysis.layout_type.value,
            'reading_order': metrics.layout_analysis.reading_order.value,
            'column_count': metrics.layout_analysis.column_count,
            'text_alignment': metrics.layout_analysis.text_alignment,
            'has_tables': metrics.layout_analysis.has_tables,
            'has_forms': metrics.layout_analysis.has_forms,
            'structure_stats': {
                'original_regions': metrics.original_regions,
                'words_created': metrics.words_created,
                'lines_created': metrics.lines_created,
                'paragraphs_created': metrics.paragraphs_created,
                'blocks_created': metrics.blocks_created
            },
            'reconstruction_confidence': metrics.confidence,
            'average_line_height': metrics.layout_analysis.average_line_height,
            'average_word_spacing': metrics.layout_analysis.average_word_spacing
        }
    
    def validate_hierarchy(self, hierarchical_regions: List[TextRegion]) -> Dict[str, bool]:
        """
        Validate the hierarchical structure for consistency
        
        Args:
            hierarchical_regions: List of hierarchical text regions
            
        Returns:
            Dictionary of validation results
        """
        validation_results = {
            'has_pages': False,
            'has_blocks': False,
            'has_paragraphs': False,
            'has_lines': False,
            'has_words': False,
            'hierarchy_consistent': True,
            'bounding_boxes_valid': True,
            'text_consistency': True
        }
        
        for region in hierarchical_regions:
            if isinstance(region, Page):
                validation_results['has_pages'] = True
                
                for block in region.blocks:
                    validation_results['has_blocks'] = True
                    
                    # Check if block text matches combined paragraph text
                    combined_paragraph_text = "\n\n".join(p.text for p in block.paragraphs)
                    if block.text.strip() != combined_paragraph_text.strip():
                        validation_results['text_consistency'] = False
                    
                    for paragraph in block.paragraphs:
                        validation_results['has_paragraphs'] = True
                        
                        # Check if paragraph text matches combined line text
                        combined_line_text = "\n".join(l.text for l in paragraph.lines)
                        if paragraph.text.strip() != combined_line_text.strip():
                            validation_results['text_consistency'] = False
                        
                        for line in paragraph.lines:
                            validation_results['has_lines'] = True
                            
                            # Check if line text matches combined word text
                            combined_word_text = " ".join(w.text for w in line.words)
                            if line.text.strip() != combined_word_text.strip():
                                validation_results['text_consistency'] = False
                            
                            for word in line.words:
                                validation_results['has_words'] = True
                                
                                # Check bounding box validity
                                if (word.bbox.width <= 0 or word.bbox.height <= 0):
                                    validation_results['bounding_boxes_valid'] = False
        
        return validation_results