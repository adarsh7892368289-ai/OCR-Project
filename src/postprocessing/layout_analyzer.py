"""
Enhanced Layout Analyzer with Document Structure Understanding
Step 5: Advanced Post-processing Implementation

Features:
- Document structure detection
- Reading order optimization
- Layout classification
- Table detection and extraction
- Multi-column support
- Performance monitoring
"""

import cv2
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from pathlib import Path
import time
import json
from collections import defaultdict, Counter
from enum import Enum
import math
import statistics
import re

try:
    from sklearn.cluster import DBSCAN, KMeans
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from ..core.base_engine import OCRResult, TextRegion, BoundingBox
from ..utils.config import ConfigManager
from ..utils.logger import get_logger
from ..utils.image_utils import ImageUtils

logger = get_logger(__name__)


class LayoutType(Enum):
    """Document layout types"""
    SINGLE_COLUMN = "single_column"
    MULTI_COLUMN = "multi_column"
    TABLE = "table"
    FORM = "form"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class TextBlockType(Enum):
    """Text block types"""
    TITLE = "title"
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    TABLE_CELL = "table_cell"
    CAPTION = "caption"
    FOOTER = "footer"
    HEADER = "header"
    UNKNOWN = "unknown"


@dataclass
class TextBlock:
    """Enhanced text block with layout information"""
    regions: List[TextRegion]
    bbox: BoundingBox
    block_type: TextBlockType
    text: str = ""
    confidence: float = 0.0
    reading_order: int = 0
    column_index: int = 0
    row_index: int = 0
    font_size: Optional[float] = None
    is_bold: bool = False
    is_italic: bool = False
    alignment: str = "left"  # left, center, right, justify
    
    def __post_init__(self):
        if not self.text:
            self.text = ' '.join(region.text for region in self.regions)
        if self.confidence == 0.0:
            self.confidence = (
                sum(region.confidence for region in self.regions) / len(self.regions)
                if self.regions else 0.0
            )


@dataclass
class TableStructure:
    """Table structure information"""
    rows: int
    columns: int
    cells: List[List[Optional[TextBlock]]]
    bbox: BoundingBox
    confidence: float = 0.0
    has_header: bool = False
    
    def get_cell_text(self, row: int, col: int) -> str:
        """Get text content of a specific cell"""
        if 0 <= row < len(self.cells) and 0 <= col < len(self.cells[row]):
            cell = self.cells[row][col]
            return cell.text if cell else ""
        return ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'rows': self.rows,
            'columns': self.columns,
            'bbox': self.bbox.__dict__,
            'confidence': self.confidence,
            'has_header': self.has_header,
            'cell_data': [
                [cell.text if cell else "" for cell in row]
                for row in self.cells
            ]
        }


@dataclass
class LayoutAnalysis:
    """Complete layout analysis result"""
    layout_type: LayoutType
    text_blocks: List[TextBlock]
    tables: List[TableStructure]
    reading_order: List[int]
    column_count: int = 1
    confidence: float = 0.0
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'layout_type': self.layout_type.value,
            'block_count': len(self.text_blocks),
            'table_count': len(self.tables),
            'column_count': self.column_count,
            'confidence': self.confidence,
            'processing_time': self.processing_time,
            'metadata': self.metadata,
            'tables': [table.to_dict() for table in self.tables]
        }


class EnhancedLayoutAnalyzer:
    """
    Advanced layout analyzer with document structure understanding
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = ConfigManager(config_path).get_section('layout_analyzer', {})
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        
        # Configuration
        self.enable_table_detection = self.config.get('enable_table_detection', True)
        self.enable_column_detection = self.config.get('enable_column_detection', True)
        self.enable_block_classification = self.config.get('enable_block_classification', True)
        self.enable_reading_order = self.config.get('enable_reading_order', True)
        
        # Detection parameters
        self.min_table_cells = self.config.get('min_table_cells', 4)
        self.column_gap_threshold = self.config.get('column_gap_threshold', 50)
        self.line_height_threshold = self.config.get('line_height_threshold', 1.5)
        self.clustering_eps = self.config.get('clustering_eps', 20)
        self.min_block_size = self.config.get('min_block_size', 3)
        
        # Initialize components
        self.image_utils = ImageUtils()
        
        # Statistics
        self.stats = defaultdict(int)
        self.processing_history = []
        
        self.logger.info("Enhanced layout analyzer initialized")
    
    def analyze_layout(
        self,
        ocr_result: OCRResult,
        image: Optional[np.ndarray] = None
    ) -> LayoutAnalysis:
        """
        Analyze document layout and structure
        
        Args:
            ocr_result: OCR result with text regions
            image: Original image for additional analysis
            
        Returns:
            LayoutAnalysis with document structure information
        """
        start_time = time.time()
        
        if not ocr_result.regions:
            return LayoutAnalysis(
                layout_type=LayoutType.UNKNOWN,
                text_blocks=[],
                tables=[],
                reading_order=[],
                processing_time=time.time() - start_time
            )
        
        try:
            # Step 1: Group regions into text blocks
            text_blocks = self._group_regions_into_blocks(ocr_result.regions)
            
            # Step 2: Classify block types
            if self.enable_block_classification:
                text_blocks = self._classify_text_blocks(text_blocks)
            
            # Step 3: Detect tables
            tables = []
            if self.enable_table_detection:
                tables = self._detect_tables(text_blocks, image)
            
            # Step 4: Detect column layout
            column_count = 1
            if self.enable_column_detection:
                column_count = self._detect_columns(text_blocks)
                text_blocks = self._assign_column_indices(text_blocks, column_count)
            
            # Step 5: Determine reading order
            reading_order = list(range(len(text_blocks)))
            if self.enable_reading_order:
                reading_order = self._calculate_reading_order(text_blocks, column_count)
            
            # Step 6: Classify overall layout type
            layout_type = self._classify_layout_type(text_blocks, tables, column_count)
            
            # Calculate confidence
            confidence = self._calculate_layout_confidence(text_blocks, tables)
            
            processing_time = time.time() - start_time
            
            # Update statistics
            self.stats['layouts_analyzed'] += 1
            self.stats[f'layout_type_{layout_type.value}'] += 1
            self.stats['tables_detected'] += len(tables)
            self.processing_history.append({
                'timestamp': time.time(),
                'processing_time': processing_time,
                'block_count': len(text_blocks),
                'table_count': len(tables),
                'column_count': column_count
            })
            
            analysis = LayoutAnalysis(
                layout_type=layout_type,
                text_blocks=text_blocks,
                tables=tables,
                reading_order=reading_order,
                column_count=column_count,
                confidence=confidence,
                processing_time=processing_time,
                metadata={
                    'original_regions': len(ocr_result.regions),
                    'blocks_created': len(text_blocks),
                    'clustering_method': 'dbscan' if SKLEARN_AVAILABLE else 'distance_based'
                }
            )
            
            self.logger.info(
                f"Layout analysis completed: {layout_type.value}, "
                f"{len(text_blocks)} blocks, {len(tables)} tables"
            )
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error in layout analysis: {e}")
            return LayoutAnalysis(
                layout_type=LayoutType.UNKNOWN,
                text_blocks=[],
                tables=[],
                reading_order=[],
                processing_time=time.time() - start_time
            )
    
    def _group_regions_into_blocks(self, regions: List[TextRegion]) -> List[TextBlock]:
        """Group text regions into coherent blocks"""
        if not regions:
            return []
        
        # Extract features for clustering
        features = []
        for region in regions:
            if region.bbox:
                features.append([
                    region.bbox.x1,
                    region.bbox.y1,
                    region.bbox.x2 - region.bbox.x1,  # width
                    region.bbox.y2 - region.bbox.y1   # height
                ])
            else:
                features.append([0, 0, 0, 0])
        
        features = np.array(features)
        
        if SKLEARN_AVAILABLE and len(features) > 1:
            # Use DBSCAN clustering for better grouping
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(features)
            
            clustering = DBSCAN(
                eps=self.clustering_eps / 100.0,  # Normalize for scaled features
                min_samples=1
            )
            cluster_labels = clustering.fit_predict(scaled_features)
        else:
            # Simple distance-based grouping
            cluster_labels = self._distance_based_clustering(features)
        
        # Group regions by cluster
        clusters = defaultdict(list)
        for i, label in enumerate(cluster_labels):
            clusters[label].append(regions[i])
        
        # Create text blocks
        text_blocks = []
        for cluster_regions in clusters.values():
            if len(cluster_regions) == 0:
                continue
            
            # Calculate bounding box for the block
            min_x1 = min(r.bbox.x1 for r in cluster_regions if r.bbox)
            min_y1 = min(r.bbox.y1 for r in cluster_regions if r.bbox)
            max_x2 = max(r.bbox.x2 for r in cluster_regions if r.bbox)
            max_y2 = max(r.bbox.y2 for r in cluster_regions if r.bbox)
            
            block_bbox = BoundingBox(min_x1, min_y1, max_x2, max_y2)
            
            # Sort regions within block by reading order
            sorted_regions = sorted(
                cluster_regions,
                key=lambda r: (r.bbox.y1 if r.bbox else 0, r.bbox.x1 if r.bbox else 0)
            )
            
            text_block = TextBlock(
                regions=sorted_regions,
                bbox=block_bbox,
                block_type=TextBlockType.UNKNOWN
            )
            
            text_blocks.append(text_block)
        
        return text_blocks
    
    def _distance_based_clustering(self, features: np.ndarray) -> List[int]:
        """Simple distance-based clustering fallback"""
        if len(features) == 0:
            return []
        
        clusters = []
        cluster_id = 0
        
        for i, feature in enumerate(features):
            assigned = False
            
            # Check if this feature belongs to an existing cluster
            for j in range(i):
                if clusters[j] == -1:  # Skip noise points
                    continue
                
                # Calculate distance
                other_feature = features[j]
                distance = np.sqrt(np.sum((feature - other_feature) ** 2))
                
                if distance < self.clustering_eps:
                    clusters.append(clusters[j])
                    assigned = True
                    break
            
            if not assigned:
                clusters.append(cluster_id)
                cluster_id += 1
        
        return clusters
    
    def _classify_text_blocks(self, text_blocks: List[TextBlock]) -> List[TextBlock]:
        """Classify text blocks by their type"""
        for block in text_blocks:
            block.block_type = self._determine_block_type(block)
        
        return text_blocks
    
    def _determine_block_type(self, block: TextBlock) -> TextBlockType:
        """Determine the type of a text block"""
        text = block.text.strip()
        
        if not text:
            return TextBlockType.UNKNOWN
        
        # Title detection (short, often centered, larger font implied by position)
        if (len(text) < 100 and 
            len(text.split()) < 10 and
            block.bbox.y1 < 200):  # Assuming top of page
            return TextBlockType.TITLE
        
        # Heading detection (short lines, often bold/larger)
        if (len(text) < 200 and 
            len(text.split()) < 20 and
            not text.endswith(('.', '!', '?'))):
            return TextBlockType.HEADING
        
        # List item detection
        if (text.startswith(('•', '-', '*', '1.', '2.', '3.', '4.', '5.')) or
            re.match(r'^\d+[\.\)]\s', text) or
            re.match(r'^[a-zA-Z][\.\)]\s', text)):
            return TextBlockType.LIST_ITEM
        
        # Table cell detection (short, structured content)
        if (len(text) < 50 and 
            (re.search(r'\d+', text) or len(text.split()) < 5)):
            return TextBlockType.TABLE_CELL
        
        # Footer detection (bottom of page)
        if block.bbox.y1 > 800:  # Assuming page height > 800
            return TextBlockType.FOOTER
        
        # Header detection (top of page)
        if block.bbox.y1 < 100:
            return TextBlockType.HEADER
        
        # Default to paragraph
        return TextBlockType.PARAGRAPH
    
    def _detect_tables(
        self, 
        text_blocks: List[TextBlock], 
        image: Optional[np.ndarray] = None
    ) -> List[TableStructure]:
        """Detect table structures in the document"""
        tables = []
        
        # Find potential table regions
        table_candidates = self._find_table_candidates(text_blocks)
        
        for candidate_blocks in table_candidates:
            table = self._analyze_table_structure(candidate_blocks, image)
            if table:
                tables.append(table)
        
        return tables
    
    def _find_table_candidates(self, text_blocks: List[TextBlock]) -> List[List[TextBlock]]:
        """Find groups of blocks that might form tables"""
        # Group blocks that are likely table cells
        table_blocks = [
            block for block in text_blocks 
            if block.block_type == TextBlockType.TABLE_CELL or
            (len(block.text.split()) < 10 and len(block.regions) == 1)
        ]
        
        if len(table_blocks) < self.min_table_cells:
            return []
        
        # Group nearby table blocks
        candidates = []
        
        if SKLEARN_AVAILABLE:
            # Use clustering to group table blocks
            positions = np.array([
                [block.bbox.x1, block.bbox.y1] for block in table_blocks
            ])
            
            clustering = DBSCAN(eps=50, min_samples=2)
            cluster_labels = clustering.fit_predict(positions)
            
            clusters = defaultdict(list)
            for i, label in enumerate(cluster_labels):
                if label != -1:  # Ignore noise
                    clusters[label].append(table_blocks[i])
            
            for cluster_blocks in clusters.values():
                if len(cluster_blocks) >= self.min_table_cells:
                    candidates.append(cluster_blocks)
        else:
            # Simple grid-based grouping
            if len(table_blocks) >= self.min_table_cells:
                candidates.append(table_blocks)
        
        return candidates
    
    def _analyze_table_structure(
        self, 
        blocks: List[TextBlock], 
        image: Optional[np.ndarray] = None
    ) -> Optional[TableStructure]:
        """Analyze the structure of a table candidate"""
        if len(blocks) < self.min_table_cells:
            return None
        
        # Sort blocks by position
        sorted_blocks = sorted(
            blocks,
            key=lambda b: (round(b.bbox.y1 / 20) * 20, b.bbox.x1)  # Group by approximate row
        )
        
        # Detect rows and columns
        rows = self._detect_table_rows(sorted_blocks)
        columns = self._detect_table_columns(rows)
        
        if len(rows) < 2 or columns < 2:
            return None
        
        # Create cell matrix
        cells = [[None for _ in range(columns)] for _ in range(len(rows))]
        
        for row_idx, row_blocks in enumerate(rows):
            for block in row_blocks:
                col_idx = self._determine_column_index(block, columns)
                if 0 <= col_idx < columns:
                    cells[row_idx][col_idx] = block
        
        # Calculate table bounding box
        min_x1 = min(block.bbox.x1 for block in blocks)
        min_y1 = min(block.bbox.y1 for block in blocks)
        max_x2 = max(block.bbox.x2 for block in blocks)
        max_y2 = max(block.bbox.y2 for block in blocks)
        table_bbox = BoundingBox(min_x1, min_y1, max_x2, max_y2)
        
        # Calculate confidence
        filled_cells = sum(1 for row in cells for cell in row if cell is not None)
        total_cells = len(rows) * columns
        confidence = filled_cells / total_cells if total_cells > 0 else 0.0
        
        # Detect header
        has_header = self._detect_table_header(cells)
        
        return TableStructure(
            rows=len(rows),
            columns=columns,
            cells=cells,
            bbox=table_bbox,
            confidence=confidence,
            has_header=has_header
        )
    
    def _detect_table_rows(self, sorted_blocks: List[TextBlock]) -> List[List[TextBlock]]:
        """Group blocks into table rows"""
        if not sorted_blocks:
            return []
        
        rows = []
        current_row = [sorted_blocks[0]]
        current_y = sorted_blocks[0].bbox.y1
        
        for block in sorted_blocks[1:]:
            # If the block is on roughly the same line, add to current row
            if abs(block.bbox.y1 - current_y) < self.line_height_threshold * 10:
                current_row.append(block)
            else:
                # Start a new row
                if current_row:
                    rows.append(sorted(current_row, key=lambda b: b.bbox.x1))
                current_row = [block]
                current_y = block.bbox.y1
        
        # Add the last row
        if current_row:
            rows.append(sorted(current_row, key=lambda b: b.bbox.x1))
        
        return rows
    
    def _detect_table_columns(self, rows: List[List[TextBlock]]) -> int:
        """Detect the number of columns in a table"""
        if not rows:
            return 0
        
        # Find the maximum number of blocks in any row
        max_columns = max(len(row) for row in rows)
        
        # Verify column consistency
        column_positions = []
        for row in rows:
            if len(row) == max_columns:
                positions = [block.bbox.x1 for block in row]
                column_positions.append(positions)
        
        if not column_positions:
            return max_columns
        
        # Calculate average column positions
        avg_positions = []
        for col_idx in range(max_columns):
            positions = [pos[col_idx] for pos in column_positions if col_idx < len(pos)]
            if positions:
                avg_positions.append(sum(positions) / len(positions))
        
        return len(avg_positions)
    
    def _determine_column_index(self, block: TextBlock, total_columns: int) -> int:
        """Determine which column a block belongs to"""
        # Simple approach: divide the width into equal columns
        # In a real implementation, you'd use the detected column positions
        block_center = (block.bbox.x1 + block.bbox.x2) / 2
        
        # This is a simplified approach - you'd want to use actual column boundaries
        # For now, assume equal column widths
        return min(int(block_center / (1000 / total_columns)), total_columns - 1)
    
    def _detect_table_header(self, cells: List[List[Optional[TextBlock]]]) -> bool:
        """Detect if the table has a header row"""
        if not cells or len(cells) < 2:
            return False
        
        first_row = cells[0]
        second_row = cells[1]
        
        # Check if first row has different characteristics
        first_row_texts = [cell.text if cell else "" for cell in first_row]
        second_row_texts = [cell.text if cell else "" for cell in second_row]
        
        # Simple heuristics for header detection
        first_has_numbers = any(re.search(r'\d+', text) for text in first_row_texts)
        second_has_numbers = any(re.search(r'\d+', text) for text in second_row_texts)
        
        # Headers typically have fewer numbers than data rows
        return not first_has_numbers and second_has_numbers
    
    def _detect_columns(self, text_blocks: List[TextBlock]) -> int:
        """Detect the number of columns in the document"""
        if not text_blocks:
            return 1
        
        # Collect x-positions of all blocks
        x_positions = []
        for block in text_blocks:
            x_positions.append(block.bbox.x1)
            x_positions.append(block.bbox.x2)
        
        if len(x_positions) < 4:
            return 1
        
        # Find gaps that might indicate column boundaries
        x_positions = sorted(set(x_positions))
        gaps = []
        
        for i in range(1, len(x_positions)):
            gap = x_positions[i] - x_positions[i-1]
            if gap > self.column_gap_threshold:
                gaps.append((x_positions[i-1], x_positions[i], gap))
        
        if not gaps:
            return 1
        
        # Simple heuristic: number of major gaps + 1
        major_gaps = [gap for gap in gaps if gap[2] > self.column_gap_threshold * 2]
        return min(len(major_gaps) + 1, 4)  # Cap at 4 columns
    
    def _assign_column_indices(
        self, 
        text_blocks: List[TextBlock], 
        column_count: int
    ) -> List[TextBlock]:
        """Assign column indices to text blocks"""
        if column_count <= 1:
            for block in text_blocks:
                block.column_index = 0
            return text_blocks
        
        # Find column boundaries
        all_x_positions = []
        for block in text_blocks:
            all_x_positions.extend([block.bbox.x1, block.bbox.x2])
        
        min_x = min(all_x_positions)
        max_x = max(all_x_positions)
        column_width = (max_x - min_x) / column_count
        
        for block in text_blocks:
            block_center = (block.bbox.x1 + block.bbox.x2) / 2
            column_index = int((block_center - min_x) / column_width)
            block.column_index = min(column_index, column_count - 1)
        
        return text_blocks
    
    def _calculate_reading_order(
        self, 
        text_blocks: List[TextBlock], 
        column_count: int
    ) -> List[int]:
        """Calculate optimal reading order for text blocks"""
        if column_count <= 1:
            # Single column: sort by y-position then x-position
            sorted_indices = sorted(
                range(len(text_blocks)),
                key=lambda i: (text_blocks[i].bbox.y1, text_blocks[i].bbox.x1)
            )
        else:
            # Multi-column: sort by column, then by y-position within column
            sorted_indices = sorted(
                range(len(text_blocks)),
                key=lambda i: (
                    text_blocks[i].column_index,
                    text_blocks[i].bbox.y1,
                    text_blocks[i].bbox.x1
                )
            )
        
        # Update reading order in blocks
        for order, index in enumerate(sorted_indices):
            text_blocks[index].reading_order = order
        
        return sorted_indices
    
    def _classify_layout_type(
        self, 
        text_blocks: List[TextBlock], 
        tables: List[TableStructure], 
        column_count: int
    ) -> LayoutType:
        """Classify the overall layout type of the document"""
        if not text_blocks:
            return LayoutType.UNKNOWN
        
        # Check for tables
        if len(tables) > 0:
            table_area = sum(
                (table.bbox.x2 - table.bbox.x1) * (table.bbox.y2 - table.bbox.y1)
                for table in tables
            )
            total_area = sum(
                (block.bbox.x2 - block.bbox.x1) * (block.bbox.y2 - block.bbox.y1)
                for block in text_blocks
            )
            
            if table_area > total_area * 0.5:
                return LayoutType.TABLE
        
        # Check for form-like structure
        form_indicators = sum(
            1 for block in text_blocks
            if len(block.text.split()) < 5 and ':' in block.text
        )
        
        if form_indicators > len(text_blocks) * 0.3:
            return LayoutType.FORM
        
        # Check column count
        if column_count > 1:
            return LayoutType.MULTI_COLUMN
        
        # Check for mixed content
        block_types = {block.block_type for block in text_blocks}
        if len(block_types) > 3:
            return LayoutType.MIXED
        
        return LayoutType.SINGLE_COLUMN
    
    def _calculate_layout_confidence(
        self, 
        text_blocks: List[TextBlock], 
        tables: List[TableStructure]
    ) -> float:
        """Calculate overall confidence in layout analysis"""
        if not text_blocks:
            return 0.0
        
        # Base confidence on text block confidences
        block_confidence = sum(block.confidence for block in text_blocks) / len(text_blocks)
        
        # Boost confidence for detected tables
        table_confidence = (
            sum(table.confidence for table in tables) / len(tables)
            if tables else 0.0
        )
        
        # Combine confidences
        overall_confidence = 0.7 * block_confidence + 0.3 * table_confidence
        
        # Adjust based on block classification success
        classified_blocks = sum(
            1 for block in text_blocks
            if block.block_type != TextBlockType.UNKNOWN
        )
        classification_ratio = classified_blocks / len(text_blocks)
        
        overall_confidence *= (0.5 + 0.5 * classification_ratio)
        
        return round(min(1.0, max(0.0, overall_confidence)), 3)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get layout analysis statistics"""
        stats = dict(self.stats)
        
        if self.processing_history:
            recent_history = self.processing_history[-100:]
            stats['performance'] = {
                'avg_processing_time': statistics.mean(
                    [h['processing_time'] for h in recent_history]
                ),
                'avg_blocks_per_document': statistics.mean(
                    [h['block_count'] for h in recent_history]
                ),
                'avg_tables_per_document': statistics.mean(
                    [h['table_count'] for h in recent_history]
                ),
                'samples': len(recent_history)
            }
        
        stats['configuration'] = {
            'enable_table_detection': self.enable_table_detection,
            'enable_column_detection': self.enable_column_detection,
            'enable_block_classification': self.enable_block_classification,
            'min_table_cells': self.min_table_cells,
            'column_gap_threshold': self.column_gap_threshold
        }
        
        return stats
    
    def export_layout_data(
        self, 
        analysis: LayoutAnalysis, 
        format_type: str = 'json'
    ) -> Union[str, Dict[str, Any]]:
        """
        Export layout analysis data in various formats
        
        Args:
            analysis: Layout analysis result
            format_type: Export format ('json', 'xml', 'html')
            
        Returns:
            Formatted layout data
        """
        if format_type == 'json':
            return analysis.to_dict()
        
        elif format_type == 'html':
            return self._export_to_html(analysis)
        
        elif format_type == 'xml':
            return self._export_to_xml(analysis)
        
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    
    def _export_to_html(self, analysis: LayoutAnalysis) -> str:
        """Export layout analysis to HTML format"""
        html_parts = ['<div class="document-layout">']
        
        for block in analysis.text_blocks:
            css_class = f"text-block {block.block_type.value}"
            html_parts.append(f'<div class="{css_class}" data-column="{block.column_index}">')
            html_parts.append(f'<p>{block.text}</p>')
            html_parts.append('</div>')
        
        for table in analysis.tables:
            html_parts.append('<table class="detected-table">')
            for row in table.cells:
                html_parts.append('<tr>')
                for cell in row:
                    cell_text = cell.text if cell else ""
                    html_parts.append(f'<td>{cell_text}</td>')
                html_parts.append('</tr>')
            html_parts.append('</table>')
        
        html_parts.append('</div>')
        
        return '\n'.join(html_parts)
    
    def _export_to_xml(self, analysis: LayoutAnalysis) -> str:
        """Export layout analysis to XML format"""
        xml_parts = ['<?xml version="1.0" encoding="UTF-8"?>']
        xml_parts.append('<document>')
        xml_parts.append(f'<layout type="{analysis.layout_type.value}" confidence="{analysis.confidence}">')
        
        for i, block in enumerate(analysis.text_blocks):
            xml_parts.append(f'<text-block id="{i}" type="{block.block_type.value}" column="{block.column_index}">')
            xml_parts.append(f'<text>{block.text}</text>')
            xml_parts.append(f'<confidence>{block.confidence}</confidence>')
            xml_parts.append('</text-block>')
        
        for i, table in enumerate(analysis.tables):
            xml_parts.append(f'<table id="{i}" rows="{table.rows}" columns="{table.columns}">')
            for row_idx, row in enumerate(table.cells):
                xml_parts.append(f'<row index="{row_idx}">')
                for col_idx, cell in enumerate(row):
                    cell_text = cell.text if cell else ""
                    xml_parts.append(f'<cell index="{col_idx}">{cell_text}</cell>')
                xml_parts.append('</row>')
            xml_parts.append('</table>')
        
        xml_parts.append('</layout>')
        xml_parts.append('</document>')
        
        return '\n'.join(xml_parts)