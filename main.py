# main.py - Main Application Entry Point

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Optional, Dict, Any
import logging
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.panel import Panel

# Add src to Python path
sys.path.append(str(Path(__file__).parent / "src"))

from src.core.ocr_engine import AdvancedOCREngine
from src.utils.logger import setup_logger
from src.utils.config import load_config

console = Console()

def setup_directories():
    """Ensure all required directories exist"""
    directories = [
        "data/input/test_images",
        "data/input/sample_images", 
        "data/output/json_results",
        "data/output/text_files",
        "data/output/annotated_images",
        "data/models",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

def validate_image_path(image_path: str) -> bool:
    """Validate if image path exists and is valid format"""
    if not os.path.exists(image_path):
        console.print(f"[red]Error: Image file not found: {image_path}[/red]")
        return False
    
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
    file_extension = Path(image_path).suffix.lower()
    
    if file_extension not in valid_extensions:
        console.print(f"[red]Error: Unsupported image format: {file_extension}[/red]")
        console.print(f"[yellow]Supported formats: {', '.join(valid_extensions)}[/yellow]")
        return False
    
    return True

def display_results(results: Dict[Any, Any], detailed: bool = True):
    """Display OCR results in a beautiful format using Rich"""

    if 'error' in results:
        console.print(Panel(f"[red]Error: {results['error']}[/red]", title="OCR Error"))
        return

    # Main results panel
    summary = results.get('summary', {})

    if 'primary_text' in summary and summary['primary_text'].strip():
        # Primary text panel
        console.print(Panel(
            f"[bold green]{summary['primary_text']}[/bold green]",
            title="🎯 Extracted Text",
            border_style="green"
        ))
    else:
        console.print(Panel(
            "[yellow]No text was detected in the image[/yellow]",
            title="🎯 Extraction Result",
            border_style="yellow"
        ))
        return

    # ------------------ Corrected Detailed Engine Results Section ------------------
    if detailed and 'text_by_engine' in results:
        console.print("\n" + "╭" + "─" * 79 + " 📊 Detailed Engine Results " + "─" * 79 + "╮")
        for engine_name, extracted_text in results['text_by_engine'].items():
            if extracted_text:
                panel_content = f"[bold cyan]Engine:[/bold cyan] {engine_name.upper()}\n[bold]Extracted Text:[/bold]\n{extracted_text.strip()}"
                console.print(Panel(panel_content, border_style="dim"))
            else:
                panel_content = f"[bold cyan]Engine:[/bold cyan] {engine_name.upper()}\n[yellow]No text was extracted by this engine.[/yellow]"
                console.print(Panel(panel_content, border_style="dim"))
        console.print("╰" + "─" * 190 + "╯")
    # --------------------------------------------------------------------------------

    # Text analysis table
    if detailed and 'text_characteristics' in summary:
        chars = summary['text_characteristics']

        analysis_table = Table(title="📊 Text Analysis", show_header=True, header_style="bold magenta")
        analysis_table.add_column("Property", style="cyan")
        analysis_table.add_column("Value", style="white")

        analysis_table.add_row("Text Type", chars.get('text_type', 'Unknown'))
        analysis_table.add_row("Word Count", str(chars.get('word_count', 0)))
        analysis_table.add_row("Contains Numbers", "✓" if chars.get('contains_numbers') else "✗")
        analysis_table.add_row("Special Characters", "✓" if chars.get('contains_special_characters') else "✗")
        analysis_table.add_row("Language", chars.get('estimated_language', 'Unknown'))

        console.print(analysis_table)

    # Confidence analysis
    if 'confidence_analysis' in summary:
        conf = summary['confidence_analysis']
        confidence_table = Table(title="🎯 Confidence Analysis", show_header=True, header_style="bold blue")
        confidence_table.add_column("Metric", style="cyan")
        confidence_table.add_column("Value", style="white")

        confidence_table.add_row(
            "Highest Confidence",
            f"{conf.get('highest_confidence', 0):.1%}"
        )
        confidence_table.add_row(
            "Average Confidence",
            f"{conf.get('average_confidence', 0):.1%}"
        )

        agreement = conf.get('engine_agreement', {})
        agreement_status = "✓ High" if agreement.get('engines_agree', False) else "⚠ Mixed"
        confidence_table.add_row("Engine Agreement", agreement_status)

        console.print(confidence_table)

    # Engine performance comparison
    if detailed and 'engine_comparison' in results:
        engine_table = Table(title="⚙️ Engine Performance", show_header=True, header_style="bold yellow")
        engine_table.add_column("Engine", style="cyan")
        engine_table.add_column("Regions Found", justify="right")
        engine_table.add_column("Avg Confidence", justify="right")
        engine_table.add_column("Characters", justify="right")

        for engine, stats in results['engine_comparison'].items():
            engine_table.add_row(
                engine,
                str(stats.get('text_regions_found', 0)),
                f"{stats.get('average_confidence', 0):.1%}",
                str(stats.get('total_characters_extracted', 0))
            )

        console.print(engine_table)

    # Recommendations
    if 'recommendations' in summary:
        recommendations_text = "\n".join([f"• {rec}" for rec in summary['recommendations']])
        console.print(Panel(
            recommendations_text,
            title="💡 AI Recommendations",
            border_style="blue"
        ))
def save_results(results: Dict[Any, Any], image_path: str, output_dir: str = "data/output"):
    """Save results to various formats"""
    image_name = Path(image_path).stem
    timestamp = Path(image_path).stat().st_mtime
    
    # Save JSON
    json_path = Path(output_dir) / "json_results" / f"{image_name}_ocr_results.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    # Save plain text
    if 'summary' in results and 'primary_text' in results['summary']:
        text_path = Path(output_dir) / "text_files" / f"{image_name}_extracted_text.txt"
        text_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(results['summary']['primary_text'])
    
    console.print(f"[green]Results saved to:[/green]")
    console.print(f"  JSON: {json_path}")
    if 'summary' in results and 'primary_text' in results['summary']:
        console.print(f"  Text: {text_path}")

def main():
    parser = argparse.ArgumentParser(
        description="🤖 Advanced AI-Powered OCR System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --image data/input/sample.jpg
  python main.py --image test.png --simple --save
  python main.py --image document.pdf --engines paddle,trocr
        """
    )
    
    parser.add_argument(
        '--image', '-i', 
        required=True,
        help='Path to input image file'
    )
    
    parser.add_argument(
        '--engines',
        default='paddle,trocr,easyocr',
        help='OCR engines to use (comma-separated): paddle,trocr,easyocr,tesseract'
    )
    
    parser.add_argument(
        '--config', '-c',
        default='config/development.yaml',
        help='Configuration file path'
    )
    
    parser.add_argument(
        '--simple', '-s',
        action='store_true',
        help='Show simplified output'
    )
    
    parser.add_argument(
        '--save',
        action='store_true',
        help='Save results to output directory'
    )
    
    parser.add_argument(
        '--no-gpu',
        action='store_true',
        help='Disable GPU acceleration'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup
    setup_directories()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logger(level=log_level)
    
    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        console.print(f"[yellow]Config file not found: {args.config}. Using defaults.[/yellow]")
        config = {}
    
    # Validate image
    if not validate_image_path(args.image):
        return 1
    
    # Initialize OCR system
    console.print(Panel(
        "🚀 Initializing Advanced AI OCR System...",
        title="OCR Engine Startup"
    ))
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            init_task = progress.add_task("Loading OCR engines...", total=None)
            
            ocr_engine = AdvancedOCREngine(
                config=config,
                use_gpu=not args.no_gpu,
                engines=args.engines.split(','),
                verbose=args.verbose
            )
            
            progress.update(init_task, completed=True)
        
        console.print("[green]✅ OCR system initialized successfully![/green]\n")
        
    except Exception as e:
        console.print(f"[red]❌ Failed to initialize OCR system: {str(e)}[/red]")
        return 1
    
    # Process image
    console.print(f"🖼️ Processing image: [bold]{Path(args.image).name}[/bold]")
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            process_task = progress.add_task("Extracting text with AI...", total=None)
            
            results = ocr_engine.process_image(args.image)
            
            progress.update(process_task, completed=True)
        
    except Exception as e:
        console.print(f"[red]❌ Processing failed: {str(e)}[/red]")
        return 1
    
    # Display results
    console.print("\n")
    display_results(results, detailed=not args.simple)
    
    # Save results if requested
    if args.save:
        console.print("\n")
        save_results(results, args.image)
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user.[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]Unexpected error: {str(e)}[/red]")
        sys.exit(1)