from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
import cv2
import numpy as np
import io
from PIL import Image
import logging
import time
from pathlib import Path

# --- FIXED: Import the correct core system components ---
# Import only the final, working classes
from ..core.engine_manager import OCREngineManager
from ..core.base_engine import (
    OCRResult, DocumentResult, DocumentStructure, 
    TextRegion, BoundingBox, TextType
)

# Import preprocessing components
from ..preprocessing.text_detector import AdvancedTextDetector

# Import postprocessing components
from ..postprocessing.postprocessing_pipeline import PostProcessingPipeline, PipelineResult

# Import utilities - Fixed: Use Config instead of ConfigManager
from ..utils.config import Config
from ..utils.logger import setup_logger

# Configure logging FIRST
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FIXED: Import all 4 engines with proper error handling - NO MORE NONE TYPES
AVAILABLE_ENGINES = {}

try:
    from ..engines.tesseract_engine import TesseractEngine
    AVAILABLE_ENGINES['tesseract'] = TesseractEngine
    logger.info("TesseractEngine imported successfully")
except ImportError as e:
    logger.warning(f"Failed to import TesseractEngine: {e}")

try:
    from ..engines.easyocr_engine import EasyOCREngine
    AVAILABLE_ENGINES['easyocr'] = EasyOCREngine
    logger.info("EasyOCREngine imported successfully")
except ImportError as e:
    logger.warning(f"Failed to import EasyOCREngine: {e}")

try:
    from ..engines.paddle_engine import PaddleOCREngine
    AVAILABLE_ENGINES['paddleocr'] = PaddleOCREngine
    logger.info("PaddleOCREngine imported successfully")
except ImportError:
    try:
        from ..engines.paddle_engine import PaddleOCREngine
        AVAILABLE_ENGINES['paddleocr'] = PaddleOCREngine
        logger.info("PaddleOCREngine imported from paddle_engine")
    except ImportError as e:
        logger.warning(f"Failed to import PaddleOCREngine: {e}")

try:
    from ..engines.trocr_engine import TrOCREngine
    AVAILABLE_ENGINES['trocr'] = TrOCREngine
    logger.info("TrOCREngine imported successfully")
except ImportError as e:
    logger.warning(f"Failed to import TrOCREngine: {e}")

# FastAPI app
app = FastAPI(
    title="Advanced OCR System API",
    description="Unified OCR API with multi-engine support (Tesseract, EasyOCR, PaddleOCR, TrOCR), adaptive preprocessing, and comprehensive features",
    version="2.1.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

## Pydantic models for API
class OCRRequest(BaseModel):
    engines: List[str] = Field(default_factory=lambda: list(AVAILABLE_ENGINES.keys()), description="OCR engines to use")
    parallel_processing: bool = Field(default=True, description="Enable parallel processing")
    confidence_threshold: float = Field(default=0.7, description="Minimum confidence threshold")
    output_format: str = Field(default="json", description="Output format")
    text_type: str = Field(default="auto", description="Text type: auto, printed, handwritten, mixed")

class OCRResponse(BaseModel):
    success: bool
    text: str
    confidence: float
    processing_time: float
    engines_used: List[str]
    metadata: Dict[str, Any]
    warnings: List[str] = Field(default_factory=list)
    word_count: Optional[int] = None
    line_count: Optional[int] = None
    layout_analysis: Optional[Dict[str, Any]] = None
    formatted_results: Optional[Dict[str, Any]] = None
    engine_results: Optional[Dict[str, Any]] = None

class StatsResponse(BaseModel):
    ocr_statistics: Dict[str, Any]
    preprocessing_statistics: Dict[str, Any]
    system_info: Dict[str, Any]

class UnifiedOCRSystem:
    """Unified OCR System using all 4 engines with advanced pipeline"""
    
    def __init__(self, config_path: Optional[str] = "data/configs/working_config.yaml"):
        self.config = Config(config_path)
        self.logger = setup_logger("UnifiedOCRSystem", self.config.get("log_level", "INFO"))
        
        # Initialize the core pipeline components directly
        self.text_detector = AdvancedTextDetector(config=self.config.get("text_detection", {}))
        self.engine_manager = OCREngineManager(self.config)
        self.post_processor = PostProcessingPipeline(config_path)

        # CRITICAL: Track what engines are actually registered
        self.registered_engines: Dict[str, str] = {}  # engine_name -> engine_class_name
        
        # Register engines and initialize
        self._register_engines()
        self.engine_manager.initialize_engines()

    def _register_engines(self) -> None:
        """Register all available OCR engines with proper error handling"""
        self.logger.info(f"Attempting to register {len(AVAILABLE_ENGINES)} available engines")
        
        for engine_name, engine_class in AVAILABLE_ENGINES.items():
            try:
                config_key = f"engines.{engine_name}"
                engine_config = self.config.get(config_key, {})
                engine_instance = engine_class(engine_config)
                self.engine_manager.register_engine(engine_instance)
                self.registered_engines[engine_name] = engine_class.__name__
                self.logger.info(f"Successfully registered {engine_name} engine")
            except Exception as e:
                self.logger.warning(f"Failed to register {engine_name} engine: {e}")
        
        if not self.registered_engines:
            raise RuntimeError("No OCR engines could be registered!")
        
        self.logger.info(f"Successfully registered engines: {list(self.registered_engines.keys())}")

    def get_text_type_from_string(self, text_type: str) -> TextType:
        """Convert string to TextType enum"""
        mapping = {
            "auto": TextType.UNKNOWN,
            "printed": TextType.PRINTED,
            "handwritten": TextType.HANDWRITTEN,
            "mixed": TextType.MIXED
        }
        return mapping.get(text_type.lower(), TextType.UNKNOWN)

    def get_available_engines(self) -> List[str]:
        """Get list of actually available engines - NO MORE ATTRIBUTE ERRORS"""
        # Primary: Use our tracked registered engines
        if self.registered_engines:
            return list(self.registered_engines.keys())
        
        # Fallback 1: Try engine_manager's engines dict
        if hasattr(self.engine_manager, 'engines') and isinstance(self.engine_manager.engines, dict):
            return list(self.engine_manager.engines.keys())
        
        # Fallback 2: Try initialized_engines dict
        if hasattr(self.engine_manager, 'initialized_engines') and isinstance(self.engine_manager.initialized_engines, dict):
            return list(self.engine_manager.initialized_engines.keys())
        
        # Final fallback: Return engines that were imported successfully
        return list(AVAILABLE_ENGINES.keys())

    def filter_valid_engines(self, requested_engines: List[str]) -> List[str]:
        """Filter and validate requested engines - GUARANTEES NO NONE VALUES"""
        available = self.get_available_engines()
        
        # Filter out None, empty strings, and unavailable engines
        valid_engines = []
        for engine in requested_engines:
            if engine and isinstance(engine, str) and engine.strip().lower() in available:
                valid_engines.append(engine.strip().lower())
        
        # If no valid engines, use the first available one
        if not valid_engines and available:
            valid_engines = [available[0]]
        
        return valid_engines

    def process_image(self, image: np.ndarray, request: OCRRequest) -> Dict[str, Any]:
        """Runs the complete OCR pipeline with bulletproof error handling"""
        start_time = time.time()
        
        # Step 1: Text Detection with safe error handling
        try:
            detection_result = self.text_detector.detect_text_regions_dict(image)
            regions = detection_result.get('regions', []) if isinstance(detection_result, dict) else []
        except Exception as e:
            self.logger.warning(f"Text detection failed: {e}, proceeding with full image")
            regions = []

        # Step 2: Prepare regions data
        if not regions:
            self.logger.warning("No text regions detected, falling back to full image processing.")
            regions_data = [{
                'image': image, 
                'metadata': {
                    'bbox': {'x': 0, 'y': 0, 'width': image.shape[1], 'height': image.shape[0]}, 
                    'confidence': 0.5
                }
            }]
        else:
            regions_data = self._prepare_regions_safely(image, regions)

        image_stats = {'height': image.shape[0], 'width': image.shape[1]}
        text_type = self.get_text_type_from_string(request.text_type)

        # Step 3: Engine Processing - BULLETPROOF
        requested_engines = self.filter_valid_engines(request.engines)
        self.logger.info(f"Using validated engines: {requested_engines}")
        
        if len(requested_engines) > 1:
            document_result, engines_used, engine_results = self._process_multi_engine(
                regions_data, image_stats, requested_engines, text_type
            )
        else:
            document_result, engines_used, engine_results = self._process_single_engine(
                regions_data, image_stats, requested_engines[0], text_type
            )

        # Step 4: Post-processing with safe fallback
        post_processed_result = self._safe_post_process(document_result)
        
        # Step 5: Final result aggregation
        total_time = time.time() - start_time
        return self._build_final_result(
            post_processed_result, document_result, engines_used, 
            engine_results, total_time, request, text_type
        )

    def _prepare_regions_safely(self, image: np.ndarray, regions: List[Dict]) -> List[Dict[str, Any]]:
        """Safely prepare region data with bounds checking"""
        regions_data = []
        
        for r in regions:
            try:
                bbox = r.get('bbox', [0, 0, image.shape[1], image.shape[0]])
                x, y, w, h = bbox[:4]  # Ensure we have at least 4 values
                
                # Clamp to image bounds
                x = max(0, min(int(x), image.shape[1] - 1))
                y = max(0, min(int(y), image.shape[0] - 1))
                w = max(1, min(int(w), image.shape[1] - x))
                h = max(1, min(int(h), image.shape[0] - y))
                
                region_img = image[y:y+h, x:x+w]
                if region_img.size > 0:
                    regions_data.append({
                        'image': region_img, 
                        'metadata': {
                            'bbox': {'x': x, 'y': y, 'width': w, 'height': h}, 
                            'confidence': r.get('confidence', 0.5)
                        }
                    })
            except Exception as e:
                self.logger.warning(f"Failed to process region: {e}")
                continue
        
        # Fallback if no valid regions
        if not regions_data:
            regions_data = [{
                'image': image, 
                'metadata': {
                    'bbox': {'x': 0, 'y': 0, 'width': image.shape[1], 'height': image.shape[0]}, 
                    'confidence': 0.5
                }
            }]
        
        return regions_data

    def _process_multi_engine(self, regions_data: List[Dict], image_stats: Dict, 
                            engines: List[str], text_type: TextType) -> tuple[DocumentResult, List[str], Dict[str, Dict]]:
        """Process with multiple engines safely"""
        try:
            multi_results = self.engine_manager.process_regions_multi_engine(
                regions_data=regions_data,
                full_image_stats=image_stats,
                engine_names=engines,  # Already validated - no None values
                text_type=text_type
            )
            
            # Aggregate results
            best_result = None
            best_confidence = 0.0
            engine_results = {}
            
            for engine_name, result in multi_results.items():
                engine_results[engine_name] = {
                    "text": result.full_text,
                    "confidence": result.confidence_score,
                    "processing_time": result.processing_time,
                    "region_count": len(result.text_regions)
                }
                
                if result.confidence_score > best_confidence:
                    best_confidence = result.confidence_score
                    best_result = result
            
            return best_result or next(iter(multi_results.values())), engines, engine_results
            
        except Exception as e:
            self.logger.error(f"Multi-engine processing failed: {e}, falling back to single engine")
            return self._process_single_engine(regions_data, image_stats, engines[0], text_type)

    def _process_single_engine(self, regions_data: List[Dict], image_stats: Dict, 
                             engine_name: str, text_type: TextType) -> tuple[DocumentResult, List[str], Dict[str, Dict]]:
        """Process with single engine safely"""
        try:
            document_result = self.engine_manager.process_regions(
                regions_data=regions_data,
                full_image_stats=image_stats,
                text_type=text_type,
                engine_name=engine_name
            )
            
            engine_results = {
                engine_name: {
                    "text": document_result.text,
                    "confidence": document_result.confidence_score,
                    "processing_time": document_result.processing_time,
                    "region_count": len(document_result.text_regions)
                }
            }
            
            return document_result, [engine_name], engine_results
            
        except Exception as e:
            self.logger.error(f"Engine processing failed: {e}")
            # Create fallback result
            fallback_result = DocumentResult(
                full_text="Processing failed",
                confidence_score=0.0,
                processing_time=0.0,
                text_regions=[],
                image_stats=image_stats,
                engine_name=engine_name,
                document_structure=DocumentStructure()
            )
            
            return fallback_result, [engine_name], {
                engine_name: {
                    "text": "Processing failed", 
                    "confidence": 0.0, 
                    "processing_time": 0.0, 
                    "region_count": 0
                }
            }

    def _safe_post_process(self, document_result: DocumentResult) -> PipelineResult:
        """Safely post-process with fallback"""
        try:
            return self.post_processor.process(document_result)
        except Exception as e:
            self.logger.error(f"Post-processing failed: {e}")
            # Create basic fallback
            return PipelineResult(
                success=True,
                original_document_result=document_result,
                formatted_results={"json": {"content": document_result.full_text}},
                pipeline_stats={},
                layout_analysis=None,
                error_message=""
            )

    def _build_final_result(self, post_processed_result: PipelineResult, document_result: DocumentResult,
                          engines_used: List[str], engine_results: Dict[str, Dict], 
                          total_time: float, request: OCRRequest, text_type: TextType) -> Dict[str, Any]:
        """Build final result with safe attribute access"""
        
        # Safe text extraction
        formatted_results = getattr(post_processed_result, 'formatted_results', {})
        if isinstance(formatted_results, dict) and request.output_format in formatted_results:
            format_data = formatted_results.get(request.output_format, {})
            final_text = format_data.get('content', document_result.full_text) if isinstance(format_data, dict) else str(format_data)
        else:
            final_text = document_result.full_text

        # Safe layout analysis
        layout_analysis_data = None
        if hasattr(post_processed_result, 'layout_analysis') and post_processed_result.layout_analysis:
            try:
                layout_analysis_data = vars(post_processed_result.layout_analysis)
            except Exception:
                layout_analysis_data = None

        # Safe engine performance - FIXED THE KEY ISSUE
        engine_performance = {}
        try:
            all_performance = self.engine_manager.get_engine_performance()
            if isinstance(all_performance, dict):
                engine_performance = {name: all_performance.get(name, {}) for name in engines_used}
        except Exception:
            pass

        return {
            "success": post_processed_result.success,
            "text": final_text or "",
            "confidence": document_result.confidence_score,
            "processing_time": total_time,
            "engines_used": engines_used,
            "metadata": {
                "original_stats": document_result.image_stats,
                "pipeline_stats": getattr(post_processed_result, 'pipeline_stats', {}),
                "engine_performance": engine_performance,
                "text_type_detected": text_type.value if hasattr(text_type, 'value') else "unknown"
            },
            "warnings": [post_processed_result.error_message] if not post_processed_result.success and hasattr(post_processed_result, 'error_message') and post_processed_result.error_message else [],
            "word_count": len(final_text.split()) if final_text else 0,
            "line_count": final_text.count('\n') + 1 if final_text else 0,
            "layout_analysis": layout_analysis_data,
            "formatted_results": formatted_results,
            "engine_results": engine_results
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics with bulletproof error handling"""
        engine_performance = {}
        try:
            engine_performance = self.engine_manager.get_engine_performance()
            if not isinstance(engine_performance, dict):
                engine_performance = {}
        except Exception:
            pass
        
        detection_stats = {}
        try:
            detection_stats = self.text_detector.get_detection_stats()
            if not isinstance(detection_stats, dict):
                detection_stats = {}
        except Exception:
            pass
        
        available_engines = self.get_available_engines()
        
        return {
            "ocr_statistics": engine_performance,
            "preprocessing_statistics": detection_stats,
            "system_info": {
                "available_engines": available_engines,
                "initialized_engines": available_engines,  # Same as available since we track properly
                "api_version": "2.1.0",
                "total_engines": len(available_engines)
            }
        }
        
    def shutdown(self):
        """Cleanup resources"""
        try:
            self.engine_manager.cleanup_engines()
        except Exception as e:
            self.logger.warning(f"Cleanup warning: {e}")

# Global system instance
ocr_system: Optional[UnifiedOCRSystem] = None

def image_from_upload(upload: UploadFile) -> np.ndarray:
    """Convert uploaded file to OpenCV image"""
    try:
        image_data = upload.file.read()
        pil_image = Image.open(io.BytesIO(image_data))
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        cv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        return cv_image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Initialize the unified OCR system on startup"""
    global ocr_system
    try:
        ocr_system = UnifiedOCRSystem()
        logger.info("Unified OCR API started successfully with all available engines")
    except Exception as e:
        logger.error(f"Failed to initialize OCR system: {e}")
        raise

@app.on_event("shutdown") 
async def shutdown_event():
    """Cleanup on shutdown"""
    global ocr_system
    if ocr_system:
        ocr_system.shutdown()
        logger.info("OCR API shutdown complete")

@app.get("/")
async def root():
    """Root endpoint with comprehensive API information"""
    engines_stats = ocr_system.get_statistics() if ocr_system else {"system_info": {"available_engines": []}}
    return {
        "message": "Advanced OCR System API - All 4 Engines Support",
        "version": "2.1.0",
        "features": [
            "Multi-engine OCR support (Tesseract, EasyOCR, PaddleOCR, TrOCR)",
            "Adaptive preprocessing pipeline", 
            "Parallel processing",
            "Text correction",
            "Layout analysis",
            "Performance monitoring",
            "Mixed text type support (printed + handwritten)"
        ],
        "endpoints": {
            "POST /ocr": "Enhanced OCR processing with all engines",
            "GET /stats": "System statistics",
            "GET /engines": "Available engines info",
            "GET /demo": "Interactive demo page"
        },
        "engines": engines_stats.get('system_info', {})
    }

@app.get("/engines")
async def get_engines_info():
    """Get information about available engines"""
    if not ocr_system:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "message": "OCR system not initialized"}
        )
    
    stats = ocr_system.get_statistics()
    
    return {
        "available_engines": stats['system_info'].get('available_engines', []),
        "initialized_engines": stats['system_info'].get('initialized_engines', []),
        "total_engines": stats['system_info'].get('total_engines', 0),
        "engine_performance": stats.get('ocr_statistics', {}),
        "recommendations": {
            "tesseract": "Best for printed text, fast processing",
            "easyocr": "Good balance, supports many languages", 
            "paddleocr": "Excellent for mixed content, good accuracy",
            "trocr": "Best for handwritten text, transformer-based"
        }
    }

@app.get("/health")
async def health_check():
    """Enhanced health check endpoint"""
    if not ocr_system:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "message": "OCR system not initialized"}
        )
    
    stats = ocr_system.get_statistics()
    
    system_status = {
        "status": "healthy",
        "timestamp": time.time(),
        "available_engines": stats['system_info'].get('available_engines', []),
        "initialized_engines": stats['system_info'].get('initialized_engines', []),
        "engine_count": stats['system_info'].get('total_engines', 0)
    }
    
    return system_status

@app.post("/ocr", response_model=OCRResponse)
async def unified_ocr_process(
    file: UploadFile = File(...),
    engines: str = Form(default_factory=lambda: ",".join(AVAILABLE_ENGINES.keys())),
    parallel_processing: bool = Form(True),
    confidence_threshold: float = Form(0.7),
    output_format: str = Form("json"),
    text_type: str = Form("auto")
):
    """Unified OCR processing endpoint using all 4 engines with advanced pipeline"""
    if not ocr_system:
        raise HTTPException(status_code=500, detail="OCR system not initialized")
    
    try:
        image = image_from_upload(file)
        
        # Parse engines list safely
        engine_list = []
        if engines:
            engine_list = [e.strip().lower() for e in engines.split(",") if e.strip()]
        
        if not engine_list:
            engine_list = list(AVAILABLE_ENGINES.keys())
        
        request = OCRRequest(
            engines=engine_list,
            parallel_processing=parallel_processing,
            confidence_threshold=confidence_threshold,
            output_format=output_format,
            text_type=text_type
        )
        
        result = ocr_system.process_image(image, request)
        
        if result['success']:
            return OCRResponse(**result)
        else:
            raise HTTPException(status_code=500, detail=result.get('text', 'OCR processing failed.'))
    except Exception as e:
        logger.error(f"API endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")

@app.get("/stats", response_model=StatsResponse)
async def get_system_stats():
    """Get comprehensive system statistics"""
    if not ocr_system:
        raise HTTPException(status_code=500, detail="OCR system not initialized")
    
    stats = ocr_system.get_statistics()
    return StatsResponse(
        ocr_statistics=stats['ocr_statistics'],
        preprocessing_statistics=stats['preprocessing_statistics'],
        system_info=stats['system_info']
    )

@app.get("/demo")
async def demo_page():
    """Enhanced interactive demo page with all 4 engines"""
    available_engines_str = ",".join(AVAILABLE_ENGINES.keys())
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Advanced OCR System Demo - All 4 Engines</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            .header {{ text-align: center; margin-bottom: 30px; }}
            .tabs {{ display: flex; border-bottom: 2px solid #e0e0e0; margin-bottom: 20px; }}
            .tab {{ padding: 10px 20px; cursor: pointer; border: none; background: none; font-size: 16px; }}
            .tab.active {{ background: #007bff; color: white; }}
            .tab-content {{ display: none; }}
            .tab-content.active {{ display: block; }}
            .upload-area {{ border: 2px dashed #ddd; padding: 40px; text-align: center; margin-bottom: 20px; }}
            .options-grid {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; margin-bottom: 20px; }}
            .option-group {{ padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
            .checkbox-group {{ display: flex; flex-direction: column; gap: 10px; }}
            .submit-btn {{ background: #007bff; color: white; padding: 12px 24px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px; }}
            .result {{ margin-top: 20px; padding: 15px; background: #f8f9fa; border-radius: 5px; white-space: pre-wrap; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Advanced OCR System Demo</h1>
                <p>Upload images and extract text using available OCR engines</p>
                <p><strong>Available Engines: {available_engines_str}</strong></p>
            </div>
            
            <div class="content">
                <div class="tabs">
                    <button class="tab active" onclick="switchTab('single')">OCR Processing</button>
                    <button class="tab" onclick="switchTab('engines')">Engines Info</button>
                    <button class="tab" onclick="switchTab('stats')">Statistics</button>
                </div>
                
                <div id="single-tab" class="tab-content active">
                    <form id="ocrForm" enctype="multipart/form-data">
                        <div class="upload-area">
                            <input type="file" id="imageFile" name="file" accept="image/*" required>
                            <p><strong>Choose an image file to process</strong></p>
                            <small>Supported formats: JPG, PNG, TIFF, BMP</small>
                        </div>
                        
                        <div class="options-grid">
                            <div class="option-group">
                                <h3>OCR Engines (Select Multiple)</h3>
                                <div class="checkbox-group" id="engineCheckboxes">
                                    <!-- Dynamic engine checkboxes will be populated here -->
                                </div>
                            </div>
                            
                            <div class="option-group">
                                <h3>Text Type</h3>
                                <select name="text_type">
                                    <option value="auto" selected>Auto Detect</option>
                                    <option value="printed">Printed Text</option>
                                    <option value="handwritten">Handwritten Text</option>
                                    <option value="mixed">Mixed (Printed + Handwritten)</option>
                                </select>
                            </div>
                            
                            <div class="option-group">
                                <h3>Output Options</h3>
                                <label>Format: 
                                    <select name="output_format">
                                        <option value="json" selected>JSON</option>
                                        <option value="detailed">Detailed</option>
                                        <option value="markdown">Markdown</option>
                                    </select>
                                </label>
                                <br><br>
                                <label>Confidence Threshold: 
                                    <input type="range" name="confidence_threshold" min="0" max="1" step="0.1" value="0.7" oninput="this.nextElementSibling.innerHTML=this.value">
                                    <span>0.7</span>
                                </label>
                            </div>
                        </div>
                        
                        <button type="submit" class="submit-btn">Process with Selected Engines</button>
                    </form>
                    
                    <div id="result" class="result" style="display: none;"></div>
                </div>
                
                <div id="engines-tab" class="tab-content">
                    <button onclick="loadEngines()" class="submit-btn">Load Engines Info</button>
                    <div id="enginesResult" class="result" style="display: none;"></div>
                </div>
                
                <div id="stats-tab" class="tab-content">
                    <button onclick="loadStats()" class="submit-btn">Load Statistics</button>
                    <div id="statsResult" class="result" style="display: none;"></div>
                </div>
            </div>
        </div>
        
        <script>
            // Initialize engine checkboxes dynamically
            const availableEngines = '{available_engines_str}'.split(',');
            const checkboxContainer = document.getElementById('engineCheckboxes');
            
            const engineDescriptions = {{
                'tesseract': 'Fast, printed text',
                'easyocr': 'Balanced performance',
                'paddleocr': 'Mixed content',
                'trocr': 'Handwritten text'
            }};
            
            availableEngines.forEach(engine => {{
                if (engine.trim()) {{
                    const label = document.createElement('label');
                    const checkbox = document.createElement('input');
                    checkbox.type = 'checkbox';
                    checkbox.name = 'engine';
                    checkbox.value = engine.trim();
                    checkbox.checked = true;
                    
                    const description = engineDescriptions[engine.trim()] || 'OCR Engine';
                    label.appendChild(checkbox);
                    label.appendChild(document.createTextNode(` ${{engine.trim()}} (${{description}})`));
                    
                    checkboxContainer.appendChild(label);
                }}
            }});

            function switchTab(tabName) {{
                document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
                event.target.classList.add('active');
                document.getElementById(tabName + '-tab').classList.add('active');
            }}

            document.getElementById('ocrForm').addEventListener('submit', async function(e) {{
                e.preventDefault();
                const formData = new FormData();
                
                // Add file
                const fileInput = document.getElementById('imageFile');
                if (!fileInput.files[0]) {{
                    alert('Please select an image file');
                    return;
                }}
                formData.append('file', fileInput.files[0]);
                
                // Add selected engines
                const selectedEngines = Array.from(document.querySelectorAll('input[name="engine"]:checked'))
                    .map(cb => cb.value);
                
                if (selectedEngines.length === 0) {{
                    alert('Please select at least one engine');
                    return;
                }}
                
                formData.append('engines', selectedEngines.join(','));
                
                // Add other options
                formData.append('text_type', document.querySelector('[name="text_type"]').value);
                formData.append('output_format', document.querySelector('[name="output_format"]').value);
                formData.append('confidence_threshold', document.querySelector('[name="confidence_threshold"]').value);
                formData.append('parallel_processing', 'true');
                
                const resultDiv = document.getElementById('result');
                resultDiv.style.display = 'block';
                resultDiv.innerHTML = 'Processing with engines: ' + selectedEngines.join(', ') + '...';
                
                try {{
                    const response = await fetch('/ocr', {{
                        method: 'POST',
                        body: formData
                    }});
                    
                    const data = await response.json();
                    if (response.ok) {{
                        resultDiv.innerHTML = JSON.stringify(data, null, 2);
                    }} else {{
                        resultDiv.innerHTML = 'Error: ' + (data.detail || 'Unknown error');
                    }}
                }} catch (error) {{
                    resultDiv.innerHTML = 'Error: ' + error.message;
                }}
            }});

            async function loadStats() {{
                const statsDiv = document.getElementById('statsResult');
                statsDiv.style.display = 'block';
                statsDiv.innerHTML = 'Loading...';
                
                try {{
                    const response = await fetch('/stats');
                    const data = await response.json();
                    if (response.ok) {{
                        statsDiv.innerHTML = JSON.stringify(data, null, 2);
                    }} else {{
                        statsDiv.innerHTML = 'Error: ' + (data.detail || 'Unknown error');
                    }}
                }} catch (error) {{
                    statsDiv.innerHTML = 'Error: ' + error.message;
                }}
            }}

            async function loadEngines() {{
                const enginesDiv = document.getElementById('enginesResult');
                enginesDiv.style.display = 'block';
                enginesDiv.innerHTML = 'Loading engines info...';
                
                try {{
                    const response = await fetch('/engines');
                    const data = await response.json();
                    if (response.ok) {{
                        enginesDiv.innerHTML = JSON.stringify(data, null, 2);
                    }} else {{
                        enginesDiv.innerHTML = 'Error: ' + (data.detail || 'Unknown error');
                    }}
                }} catch (error) {{
                    enginesDiv.innerHTML = 'Error: ' + error.message;
                }}
            }}
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)