import sys
import json
import logging
from typing import Dict, Any
from pathlib import Path

# Import the OCRProcessor from the new dedicated module
from src.core.ocr_processor import OCRProcessor

def setup_logging(config: Dict[str, Any]):
    """Configures the logging for the application."""
    log_level = config.get("logging", {}).get("level", "INFO").upper()
    logging.basicConfig(level=log_level,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout)])

def main():
    """Main function to run the OCR processing."""
    
    # Define the application configuration
    config = {
        "logging": {
            "level": "INFO"
        },
        "ocr_engines": {
            "tesseract": {"path": "/usr/bin/tesseract"},
            "google_vision": {"api_key": "YOUR_API_KEY"}  # Replace with a real key for live results
        }
    }
    
    setup_logging(config)
    logger = logging.getLogger(__name__)

    if len(sys.argv) < 2:
        logger.error("Usage: python main.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    try:
        # Initialize the OCRProcessor with the configuration
        processor = OCRProcessor(config)
        
        # Process the image
        result = processor.process_image(image_path)
        
        # Print the results in a readable format
        print("\n--- OCR Processing Report ---")
        print(json.dumps(result, indent=4))
        
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)

if __name__ == "__main__":
    main()
