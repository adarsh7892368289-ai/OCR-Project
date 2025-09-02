# src/api/main.py

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import io
from PIL import Image
import logging
from typing import Optional, List, Dict, Any
import time
import os
from pathlib import Path

# Import our OCR system components
from ..core.engine_manager import EngineManager
from ..preprocessing.image_enhancer import ImageEnhancer
from ..preprocessing.skew_corrector import SkewCorrector
from ..preprocessing.text_detector import TextDetector
from ..postprocessing.text_corrector import TextCorrector
from ..postprocessing.confidence_filter import ConfidenceFilter
from ..postprocessing.layout_analyzer import LayoutAnalyzer
from ..postprocessing.result_formatter import ResultFormatter
from ..utils.config import Config
from ..utils.logger import setup_logger

class ModernOCRSystem:
    """Main OCR System orchestrating all components"""
    
    def __init__(self, config_path: Optional[str] = None):
        # Load configuration
        self.config = Config(config_path)
        self.logger = setup_logger("OCRSystem", self.config.get("log_level", "INFO"))
        
        # Initialize components
        self.engine_manager = EngineManager(self.config.get("engine_manager", {}))
        self.image_enhancer = ImageEnhancer(self.config.get("preprocessing", {}))
        self.skew_corrector = SkewCorrector(self.config.get("preprocessing", {}))
        self.text_detector = TextDetector(self.config.get("preprocessing", {}))
        self.text_corrector = TextCorrector(self.config.get("postprocessing", {}))
        self.confidence_filter = ConfidenceFilter(self.config.get("postprocessing", {}))
        self.layout_analyzer = LayoutAnalyzer(self.config.get("postprocessing", {}))
        self.result_formatter = ResultFormatter(self.config.get("output", {}))
        
        # Initialize engines
        self.initialize_system()
        
    def initialize_system(self):
        """Initialize the OCR system"""
        self.logger.info("Initializing Modern OCR System...")
        
        # Engine configurations
        engine_configs = {
            "tesseract": self.config.get("engines.tesseract", {
                "psm": 6,
                "lang": "eng"
            }),
            "easyocr": self.config.get("engines.easyocr", {
                "languages": ["en"],
                "gpu": True
            }),
            "trocr": self.config.get("engines.trocr", {
                "model_name": "microsoft/trocr-base-handwritten",
                "device": "cuda"
            })
        }
        
        # Initialize engines
        init_results = self.engine_manager.initialize_engines(engine_configs)
        
        for engine_name, success in init_results.items():
            if success:
                self.logger.info(f"✓ {engine_name} initialized successfully")
            else:
                self.logger.warning(f"✗ {engine_name} initialization failed")
                
        if not any(init_results.values()):
            raise RuntimeError("No OCR engines could be initialized")
            
        self.logger.info("System initialization complete")
        
    def process_image(self, image: np.ndarray, **options) -> Dict[str, Any]:
        """Process image through complete OCR pipeline"""
        start_time = time.time()
        
        try:
            # Step 1: Preprocessing
            self.logger.info("Starting preprocessing...")
            processed_image = self._preprocess_image(image, **options)
            
            # Step 2: OCR Processing
            self.logger.info("Running OCR engines...")
            ocr_result = self.engine_manager.process_image(processed_image, **options)
            
            # Step 3: Postprocessing
            self.logger.info("Applying postprocessing...")
            final_result = self._postprocess_results(ocr_result, image.shape[:2], **options)
            
            # Step 4: Format output
            self.logger.info("Formatting output...")
            formatted_result = self._format_output(final_result, **options)
            
            total_time = time.time() - start_time
            self.logger.info(f"Processing completed in {total_time:.2f}s")
            
            return {
                "success": True,
                "processing_time": total_time,
                "engine_info": self.engine_manager.get_engine_info(),
                **formatted_result
            }
            
        except Exception as e:
            self.logger.error(f"Processing failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time
            }
            
    def _preprocess_image(self, image: np.ndarray, **options) -> np.ndarray:
        """Apply preprocessing pipeline"""
        processed = image.copy()
        
        # Image enhancement
        if options.get("enhance", True):
            processed = self.image_enhancer.enhance_image(processed)
            
        # Skew correction
        if options.get("correct_skew", True):
            processed = self.skew_corrector.correct_skew(processed)
            
        return processed
        
    def _postprocess_results(self, ocr_result, image_shape: tuple, **options) -> Dict[str, Any]:
        """Apply postprocessing pipeline"""
        # Filter low-confidence results
        if options.get("filter_confidence", True):
            filtered_results = self.confidence_filter.filter_results(ocr_result.results)
        else:
            filtered_results = ocr_result.results
            
        # Text correction
        if options.get("correct_text", True):
            corrected_text = self.text_corrector.correct_text(ocr_result.full_text)
        else:
            corrected_text = ocr_result.full_text
            
        # Layout analysis
        if options.get("analyze_layout", True):
            layout_info = self.layout_analyzer.analyze_layout(filtered_results, image_shape)
        else:
            layout_info = {}
            
        return {
            "original_result": ocr_result,
            "filtered_results": filtered_results,
            "corrected_text": corrected_text,
            "layout_info": layout_info
        }
        
    def _format_output(self, processed_results: Dict[str, Any], **options) -> Dict[str, Any]:
        """Format final output"""
        output_format = options.get("output_format", "json")
        
        filtered_results = processed_results["filtered_results"]
        layout_info = processed_results["layout_info"]
        
        formatted_output = {
            "text": processed_results["corrected_text"],
            "confidence": processed_results["original_result"].confidence_score,
            "word_count": len(processed_results["corrected_text"].split()),
            "line_count": processed_results["corrected_text"].count('\n') + 1
        }
        
        if output_format == "detailed":
            formatted_output.update({
                "raw_text": processed_results["original_result"].full_text,
                "results": self.result_formatter.format_as_json(filtered_results, layout_info),
                "layout_analysis": layout_info
            })
        elif output_format == "hocr":
            formatted_output["hocr"] = self.result_formatter.format_as_hocr(
                filtered_results, processed_results["original_result"].image_stats
            )
            
        return formatted_output

# FastAPI Application
app = FastAPI(
    title="Modern OCR System",
    description="Advanced OCR system with multi-engine support",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global OCR system instance
ocr_system = None

@app.on_event("startup")
async def startup_event():
    """Initialize OCR system on startup"""
    global ocr_system
    try:
        ocr_system = ModernOCRSystem()
    except Exception as e:
        logging.error(f"Failed to initialize OCR system: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global ocr_system
    if ocr_system:
        ocr_system.engine_manager.cleanup()

@app.get("/")
async def root():
    """Root endpoint with system information"""
    return {
        "message": "Modern OCR System",
        "version": "1.0.0",
        "engines": ocr_system.engine_manager.get_engine_info() if ocr_system else {}
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if ocr_system is None:
        raise HTTPException(status_code=503, detail="OCR system not initialized")
        
    return {
        "status": "healthy",
        "engines": ocr_system.engine_manager.get_engine_info()
    }

@app.post("/ocr/process")
async def process_ocr(
    file: UploadFile = File(...),
    enhance: bool = Form(True),
    correct_skew: bool = Form(True),
    filter_confidence: bool = Form(True),
    correct_text: bool = Form(True),
    analyze_layout: bool = Form(True),
    output_format: str = Form("json"),
    force_engines: Optional[str] = Form(None)
):
    """Process uploaded image with OCR"""
    if ocr_system is None:
        raise HTTPException(status_code=503, detail="OCR system not initialized")
        
    try:
        # Read uploaded file
        contents = await file.read()
        
        # Convert to OpenCV image
        image = Image.open(io.BytesIO(contents))
        image_array = np.array(image)
        
        # Convert RGB to BGR if needed
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            
        # Prepare options
        options = {
            "enhance": enhance,
            "correct_skew": correct_skew,
            "filter_confidence": filter_confidence,
            "correct_text": correct_text,
            "analyze_layout": analyze_layout,
            "output_format": output_format
        }
        
        if force_engines:
            options["force_engines"] = force_engines.split(",")
            
        # Process image
        result = ocr_system.process_image(image_array, **options)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ocr/batch")
async def process_batch(
    files: List[UploadFile] = File(...),
    enhance: bool = Form(True),
    correct_skew: bool = Form(True),
    filter_confidence: bool = Form(True),
    correct_text: bool = Form(True),
    analyze_layout: bool = Form(True),
    output_format: str = Form("json")
):
    """Process multiple images in batch"""
    if ocr_system is None:
        raise HTTPException(status_code=503, detail="OCR system not initialized")
        
    results = []
    
    for file in files:
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
            image_array = np.array(image)
            
            if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                
            options = {
                "enhance": enhance,
                "correct_skew": correct_skew,
                "filter_confidence": filter_confidence,
                "correct_text": correct_text,
                "analyze_layout": analyze_layout,
                "output_format": output_format
            }
            
            result = ocr_system.process_image(image_array, **options)
            result["filename"] = file.filename
            results.append(result)
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })
            
    return JSONResponse(content={"results": results})

@app.get("/demo")
async def demo_page():
    """Demo page for testing OCR"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Modern OCR System Demo</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 800px; margin: 0 auto; }
            .upload-area { border: 2px dashed #ccc; padding: 40px; text-align: center; margin: 20px 0; }
            .options { margin: 20px 0; }
            .option { margin: 10px 0; }
            .result { margin: 20px 0; padding: 20px; background: #f5f5f5; border-radius: 5px; }
            .error { background: #ffebee; border: 1px solid #f44336; }
            .success { background: #e8f5e8; border: 1px solid #4caf50; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Modern OCR System Demo</h1>
            <form id="ocrForm" enctype="multipart/form-data">
                <div class="upload-area">
                    <input type="file" id="imageFile" name="file" accept="image/*" required>
                    <p>Choose an image file to process</p>
                </div>
                
                <div class="options">
                    <h3>Processing Options</h3>
                    <div class="option">
                        <label><input type="checkbox" name="enhance" checked> Image Enhancement</label>
                    </div>
                    <div class="option">
                        <label><input type="checkbox" name="correct_skew" checked> Skew Correction</label>
                    </div>
                    <div class="option">
                        <label><input type="checkbox" name="filter_confidence" checked> Confidence Filtering</label>
                    </div>
                    <div class="option">
                        <label><input type="checkbox" name="correct_text" checked> Text Correction</label>
                    </div>
                    <div class="option">
                        <label><input type="checkbox" name="analyze_layout" checked> Layout Analysis</label>
                    </div>
                    <div class="option">
                        <label>Output Format: 
                            <select name="output_format">
                                <option value="json">JSON</option>
                                <option value="detailed">Detailed</option>
                                <option value="hocr">hOCR</option>
                            </select>
                        </label>
                    </div>
                </div>
                
                <button type="submit">Process Image</button>
            </form>
            
            <div id="result" class="result" style="display: none;"></div>
        </div>
        
        <script>
            document.getElementById('ocrForm').onsubmit = async function(e) {
                e.preventDefault();
                
                const formData = new FormData(this);
                const resultDiv = document.getElementById('result');
                
                resultDiv.style.display = 'block';
                resultDiv.innerHTML = 'Processing...';
                resultDiv.className = 'result';
                
                try {
                    const response = await fetch('/ocr/process', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    
                    if (result.success) {
                        resultDiv.className = 'result success';
                        resultDiv.innerHTML = `
                            <h3>OCR Results</h3>
                            <p><strong>Text:</strong></p>
                            <pre>${result.text}</pre>
                            <p><strong>Confidence:</strong> ${(result.confidence * 100).toFixed(1)}%</p>
                            <p><strong>Processing Time:</strong> ${result.processing_time.toFixed(2)}s</p>
                            <p><strong>Word Count:</strong> ${result.word_count}</p>
                        `;
                    } else {
                        resultDiv.className = 'result error';
                        resultDiv.innerHTML = `<p><strong>Error:</strong> ${result.error}</p>`;
                    }
                } catch (error) {
                    resultDiv.className = 'result error';
                    resultDiv.innerHTML = `<p><strong>Error:</strong> ${error.message}</p>`;
                }
            };
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)