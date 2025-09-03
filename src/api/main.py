# src/api/main.py - Unified OCR API with Full Feature Set

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, BackgroundTasks
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

# Import core system components
from ..core.engine_manager import EngineManager
from ..core.enhanced_engine_manager import (
    EnhancedEngineManager, EnhancedProcessingOptions, 
    quick_ocr_with_preprocessing, batch_ocr_with_preprocessing
)

# Import preprocessing components
from ..preprocessing.image_enhancer import ImageEnhancer
from ..preprocessing.skew_corrector import SkewCorrector
from ..preprocessing.text_detector import TextDetector
from ..preprocessing.adaptive_processor import ProcessingLevel as PrepLevel, PipelineStrategy

# Import postprocessing components
from ..postprocessing.text_corrector import TextCorrector
from ..postprocessing.confidence_filter import ConfidenceFilter
from ..postprocessing.layout_analyzer import LayoutAnalyzer
from ..postprocessing.result_formatter import ResultFormatter

# Import utilities
from ..utils.config import Config
from ..utils.logger import setup_logger

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Advanced OCR System API",
    description="Unified OCR API with multi-engine support, adaptive preprocessing, and comprehensive features",
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

# Pydantic models for API
class OCRRequest(BaseModel):
    engines: List[str] = Field(default=["paddleocr", "easyocr"], description="OCR engines to use")
    enable_preprocessing: bool = Field(default=True, description="Enable adaptive preprocessing")
    preprocessing_level: str = Field(default="balanced", description="Preprocessing intensity")
    preprocessing_strategy: str = Field(default="content_aware", description="Preprocessing strategy")
    preprocess_per_engine: bool = Field(default=False, description="Different preprocessing per engine")
    parallel_processing: bool = Field(default=True, description="Enable parallel processing")
    confidence_threshold: float = Field(default=0.7, description="Minimum confidence threshold")
    validate_preprocessing: bool = Field(default=True, description="Validate preprocessing results")
    fallback_to_original: bool = Field(default=True, description="Fallback to original if quality degrades")
    # Legacy options for backward compatibility
    enhance: bool = Field(default=True, description="Image enhancement")
    correct_skew: bool = Field(default=True, description="Skew correction")
    filter_confidence: bool = Field(default=True, description="Confidence filtering")
    correct_text: bool = Field(default=True, description="Text correction")
    analyze_layout: bool = Field(default=True, description="Layout analysis")
    output_format: str = Field(default="json", description="Output format")

class OCRResponse(BaseModel):
    success: bool
    text: str
    confidence: float
    processing_time: float
    engines_used: List[str]
    preprocessing_info: Dict[str, Any]
    metadata: Dict[str, Any]
    warnings: List[str] = []
    # Additional fields for detailed output
    word_count: Optional[int] = None
    line_count: Optional[int] = None
    raw_text: Optional[str] = None
    layout_analysis: Optional[Dict[str, Any]] = None
    hocr: Optional[str] = None

class BatchOCRResponse(BaseModel):
    success: bool
    results: List[OCRResponse]
    total_processing_time: float
    batch_statistics: Dict[str, Any]

class StatsResponse(BaseModel):
    ocr_statistics: Dict[str, Any]
    preprocessing_statistics: Dict[str, Any]
    system_info: Dict[str, Any]

class UnifiedOCRSystem:
    """Unified OCR System combining both legacy and enhanced functionality"""
    
    def __init__(self, config_path: Optional[str] = None):
        # Load configuration
        self.config = Config(config_path)
        self.logger = setup_logger("UnifiedOCRSystem", self.config.get("log_level", "INFO"))
        
        # Initialize both managers
        self.enhanced_manager = None
        self.legacy_manager = None
        
        # Initialize processing components
        self.image_enhancer = ImageEnhancer(self.config.get("preprocessing", {}))
        self.skew_corrector = SkewCorrector(self.config.get("preprocessing", {}))
        self.text_detector = TextDetector(self.config.get("preprocessing", {}))
        self.text_corrector = TextCorrector(self.config.get("postprocessing", {}))
        self.confidence_filter = ConfidenceFilter(self.config.get("postprocessing", {}))
        self.layout_analyzer = LayoutAnalyzer(self.config.get("postprocessing", {}))
        self.result_formatter = ResultFormatter(self.config.get("output", {}))
        
        self.initialize_system()
        
    def initialize_system(self):
        """Initialize both enhanced and legacy systems"""
        self.logger.info("Initializing Unified OCR System...")
        
        try:
            # Try to initialize enhanced manager first
            self.enhanced_manager = EnhancedEngineManager()
            self.logger.info("✓ Enhanced manager initialized successfully")
        except Exception as e:
            self.logger.warning(f"Enhanced manager initialization failed: {e}")
            
        try:
            # Initialize legacy manager as fallback
            self.legacy_manager = EngineManager(self.config.get("engine_manager", {}))
            
            # Engine configurations for legacy manager
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
            
            init_results = self.legacy_manager.initialize_engines(engine_configs)
            successful_engines = [name for name, success in init_results.items() if success]
            
            if successful_engines:
                self.logger.info(f"✓ Legacy manager initialized with engines: {successful_engines}")
            else:
                self.logger.warning("✗ Legacy manager initialization failed")
                
        except Exception as e:
            self.logger.warning(f"Legacy manager initialization failed: {e}")
            
        if not self.enhanced_manager and not self.legacy_manager:
            raise RuntimeError("No OCR managers could be initialized")
            
        self.logger.info("System initialization complete")
        
    def process_image_enhanced(self, image: np.ndarray, options: EnhancedProcessingOptions) -> Dict[str, Any]:
        """Process image using enhanced manager"""
        if not self.enhanced_manager:
            raise RuntimeError("Enhanced manager not available")
            
        return self.enhanced_manager.process_image(image, options)
        
    def process_image_legacy(self, image: np.ndarray, **options) -> Dict[str, Any]:
        """Process image using legacy pipeline"""
        if not self.legacy_manager:
            raise RuntimeError("Legacy manager not available")
            
        start_time = time.time()
        
        try:
            # Step 1: Preprocessing
            processed_image = self._preprocess_image_legacy(image, **options)
            
            # Step 2: OCR Processing
            ocr_result = self.legacy_manager.process_image(processed_image, **options)
            
            # Step 3: Postprocessing
            final_result = self._postprocess_results_legacy(ocr_result, image.shape[:2], **options)
            
            # Step 4: Format output
            formatted_result = self._format_output_legacy(final_result, **options)
            
            total_time = time.time() - start_time
            
            return {
                "success": True,
                "processing_time": total_time,
                "engine_info": self.legacy_manager.get_engine_info(),
                **formatted_result
            }
            
        except Exception as e:
            self.logger.error(f"Legacy processing failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time
            }
            
    def _preprocess_image_legacy(self, image: np.ndarray, **options) -> np.ndarray:
        """Apply legacy preprocessing pipeline"""
        processed = image.copy()
        
        if options.get("enhance", True):
            processed = self.image_enhancer.enhance_image(processed)
            
        if options.get("correct_skew", True):
            processed = self.skew_corrector.correct_skew(processed)
            
        return processed
        
    def _postprocess_results_legacy(self, ocr_result, image_shape: tuple, **options) -> Dict[str, Any]:
        """Apply legacy postprocessing pipeline"""
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
        
    def _format_output_legacy(self, processed_results: Dict[str, Any], **options) -> Dict[str, Any]:
        """Format legacy output"""
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
        
    def get_statistics(self):
        """Get system statistics"""
        stats = {}
        
        if self.enhanced_manager:
            stats["enhanced"] = self.enhanced_manager.get_statistics()
            
        if self.legacy_manager:
            stats["legacy"] = self.legacy_manager.get_engine_info()
            
        return stats
        
    def shutdown(self):
        """Cleanup resources"""
        if self.enhanced_manager:
            self.enhanced_manager.shutdown()
            
        if self.legacy_manager:
            self.legacy_manager.cleanup()

# Global system instance
ocr_system = None

def create_processing_options(request: OCRRequest) -> EnhancedProcessingOptions:
    """Create processing options from request"""
    
    # Map string values to enums
    prep_level_map = {
        "minimal": PrepLevel.MINIMAL,
        "light": PrepLevel.LIGHT,
        "balanced": PrepLevel.BALANCED,
        "intensive": PrepLevel.INTENSIVE,
        "maximum": PrepLevel.MAXIMUM
    }
    
    strategy_map = {
        "speed_optimized": PipelineStrategy.SPEED_OPTIMIZED,
        "quality_optimized": PipelineStrategy.QUALITY_OPTIMIZED,
        "content_aware": PipelineStrategy.CONTENT_AWARE,
        "custom": PipelineStrategy.CUSTOM
    }
    
    return EnhancedProcessingOptions(
        engines=request.engines,
        parallel_processing=request.parallel_processing,
        confidence_threshold=request.confidence_threshold,
        enable_preprocessing=request.enable_preprocessing,
        preprocessing_level=prep_level_map.get(request.preprocessing_level, PrepLevel.BALANCED),
        preprocessing_strategy=strategy_map.get(request.preprocessing_strategy, PipelineStrategy.CONTENT_AWARE),
        preprocess_per_engine=request.preprocess_per_engine,
        validate_preprocessing=request.validate_preprocessing,
        fallback_to_original=request.fallback_to_original
    )

def image_from_upload(upload: UploadFile) -> np.ndarray:
    """Convert uploaded file to OpenCV image"""
    try:
        # Read image data
        image_data = upload.file.read()
        
        # Convert to PIL Image
        pil_image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if needed
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Convert to OpenCV format
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
        logger.info("Unified OCR API started successfully")
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
    return {
        "message": "Advanced OCR System API",
        "version": "2.1.0",
        "features": [
            "Multi-engine OCR support",
            "Adaptive preprocessing pipeline", 
            "Legacy compatibility mode",
            "Quality-aware processing",
            "Parallel processing",
            "Batch operations",
            "Performance monitoring",
            "Layout analysis",
            "Text correction"
        ],
        "endpoints": {
            "POST /ocr": "Enhanced OCR processing",
            "POST /ocr/process": "Legacy OCR processing",
            "POST /ocr/batch": "Batch processing",
            "GET /stats": "System statistics",
            "POST /reset-stats": "Reset statistics",
            "GET /health": "Health check",
            "GET /demo": "Interactive demo page"
        },
        "engines": ocr_system.get_statistics() if ocr_system else {}
    }

@app.get("/health")
async def health_check():
    """Enhanced health check endpoint"""
    if not ocr_system:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "message": "OCR system not initialized"}
        )
    
    system_status = {
        "status": "healthy",
        "timestamp": time.time(),
        "enhanced_manager": bool(ocr_system.enhanced_manager),
        "legacy_manager": bool(ocr_system.legacy_manager)
    }
    
    return system_status

@app.post("/ocr", response_model=OCRResponse)
async def enhanced_ocr(
    file: UploadFile = File(...),
    engines: str = "paddleocr,easyocr",
    enable_preprocessing: bool = True,
    preprocessing_level: str = "balanced",
    preprocessing_strategy: str = "content_aware",
    preprocess_per_engine: bool = False,
    parallel_processing: bool = True,
    confidence_threshold: float = 0.7,
    validate_preprocessing: bool = True,
    fallback_to_original: bool = True,
    output_format: str = "json"
):
    """
    Enhanced OCR processing with adaptive preprocessing
    """
    if not ocr_system:
        raise HTTPException(status_code=500, detail="OCR system not initialized")
    
    try:
        # Convert image
        image = image_from_upload(file)
        
        # Try enhanced processing first
        if ocr_system.enhanced_manager:
            # Create request object
            request = OCRRequest(
                engines=engines.split(","),
                enable_preprocessing=enable_preprocessing,
                preprocessing_level=preprocessing_level,
                preprocessing_strategy=preprocessing_strategy,
                preprocess_per_engine=preprocess_per_engine,
                parallel_processing=parallel_processing,
                confidence_threshold=confidence_threshold,
                validate_preprocessing=validate_preprocessing,
                fallback_to_original=fallback_to_original,
                output_format=output_format
            )
            
            options = create_processing_options(request)
            results = ocr_system.process_image_enhanced(image, options)
            
            # Format response
            if results["success"] and results.get("combined_result"):
                response_data = {
                    "success": True,
                    "text": results["combined_result"].text,
                    "confidence": results["combined_result"].confidence,
                    "processing_time": results["processing_time"],
                    "engines_used": results["engines_used"],
                    "preprocessing_info": results.get("preprocessing_results", {}),
                    "metadata": results["metadata"],
                    "warnings": results.get("warnings", []),
                    "word_count": len(results["combined_result"].text.split()),
                    "line_count": results["combined_result"].text.count('\n') + 1
                }
                
                if output_format == "detailed":
                    response_data.update({
                        "raw_text": results.get("raw_text", ""),
                        "layout_analysis": results.get("layout_info", {})
                    })
                    
                return OCRResponse(**response_data)
            else:
                return OCRResponse(
                    success=False,
                    text="",
                    confidence=0.0,
                    processing_time=results["processing_time"],
                    engines_used=results.get("engines_used", []),
                    preprocessing_info=results.get("preprocessing_results", {}),
                    metadata=results.get("metadata", {}),
                    warnings=results.get("warnings", ["Enhanced OCR processing failed"])
                )
        
        # Fallback to legacy processing
        elif ocr_system.legacy_manager:
            options = {
                "enhance": True,
                "correct_skew": True,
                "filter_confidence": True,
                "correct_text": True,
                "analyze_layout": True,
                "output_format": output_format
            }
            
            results = ocr_system.process_image_legacy(image, **options)
            
            return OCRResponse(
                success=results["success"],
                text=results.get("text", ""),
                confidence=results.get("confidence", 0.0),
                processing_time=results["processing_time"],
                engines_used=[],
                preprocessing_info={},
                metadata=results.get("engine_info", {}),
                warnings=["Using legacy processing mode"],
                word_count=results.get("word_count"),
                line_count=results.get("line_count"),
                raw_text=results.get("raw_text"),
                layout_analysis=results.get("layout_analysis")
            )
        else:
            raise HTTPException(status_code=503, detail="No OCR managers available")
            
    except Exception as e:
        logger.error(f"OCR processing error: {e}")
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")

@app.post("/ocr/process")
async def legacy_ocr_process(
    file: UploadFile = File(...),
    enhance: bool = Form(True),
    correct_skew: bool = Form(True),
    filter_confidence: bool = Form(True),
    correct_text: bool = Form(True),
    analyze_layout: bool = Form(True),
    output_format: str = Form("json"),
    force_engines: Optional[str] = Form(None)
):
    """Legacy OCR processing endpoint for backward compatibility"""
    if not ocr_system or not ocr_system.legacy_manager:
        raise HTTPException(status_code=503, detail="Legacy OCR system not available")
        
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
        result = ocr_system.process_image_legacy(image_array, **options)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ocr/batch", response_model=BatchOCRResponse)
async def batch_ocr(
    files: List[UploadFile] = File(...),
    engines: str = "paddleocr,easyocr",
    enable_preprocessing: bool = True,
    preprocessing_level: str = "balanced",
    parallel_processing: bool = True
):
    """
    Perform batch OCR on multiple images
    """
    if not ocr_system:
        raise HTTPException(status_code=500, detail="OCR system not initialized")
    
    try:
        start_time = time.time()
        
        # Process all images
        batch_results = []
        successful_count = 0
        
        for i, file in enumerate(files):
            try:
                image = image_from_upload(file)
                
                # Use enhanced processing if available
                if ocr_system.enhanced_manager:
                    request = OCRRequest(
                        engines=engines.split(","),
                        enable_preprocessing=enable_preprocessing,
                        preprocessing_level=preprocessing_level,
                        parallel_processing=parallel_processing
                    )
                    
                    options = create_processing_options(request)
                    result = ocr_system.process_image_enhanced(image, options)
                    
                    if result["success"] and result.get("combined_result"):
                        ocr_response = OCRResponse(
                            success=True,
                            text=result["combined_result"].text,
                            confidence=result["combined_result"].confidence,
                            processing_time=result["processing_time"],
                            engines_used=result["engines_used"],
                            preprocessing_info=result.get("preprocessing_results", {}),
                            metadata=result["metadata"],
                            warnings=result.get("warnings", [])
                        )
                        successful_count += 1
                    else:
                        ocr_response = OCRResponse(
                            success=False,
                            text="",
                            confidence=0.0,
                            processing_time=result["processing_time"],
                            engines_used=result.get("engines_used", []),
                            preprocessing_info=result.get("preprocessing_results", {}),
                            metadata=result.get("metadata", {}),
                            warnings=result.get("warnings", [f"Processing failed for image {i+1}"])
                        )
                        
                # Fallback to legacy processing
                elif ocr_system.legacy_manager:
                    options = {
                        "enhance": enable_preprocessing,
                        "correct_skew": enable_preprocessing,
                        "filter_confidence": True,
                        "correct_text": True,
                        "analyze_layout": True,
                        "output_format": "json"
                    }
                    
                    result = ocr_system.process_image_legacy(image, **options)
                    
                    ocr_response = OCRResponse(
                        success=result["success"],
                        text=result.get("text", ""),
                        confidence=result.get("confidence", 0.0),
                        processing_time=result["processing_time"],
                        engines_used=[],
                        preprocessing_info={},
                        metadata=result.get("engine_info", {}),
                        warnings=["Using legacy processing mode"]
                    )
                    
                    if result["success"]:
                        successful_count += 1
                        
                batch_results.append(ocr_response)
                
            except Exception as e:
                # Handle individual image processing errors
                batch_results.append(OCRResponse(
                    success=False,
                    text="",
                    confidence=0.0,
                    processing_time=0.0,
                    engines_used=[],
                    preprocessing_info={},
                    metadata={},
                    warnings=[f"Error processing image {i+1}: {str(e)}"]
                ))
        
        total_time = time.time() - start_time
        
        # Create batch response
        response = BatchOCRResponse(
            success=successful_count > 0,
            results=batch_results,
            total_processing_time=total_time,
            batch_statistics={
                "total_images": len(files),
                "successful_images": successful_count,
                "success_rate": successful_count / len(files),
                "average_processing_time": total_time / len(files)
            }
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Batch OCR error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch OCR failed: {str(e)}")

@app.get("/stats", response_model=StatsResponse)
async def get_statistics():
    """Get comprehensive system statistics"""
    if not ocr_system:
        raise HTTPException(status_code=500, detail="OCR system not initialized")
    
    try:
        stats = ocr_system.get_statistics()
        
        ocr_stats = {}
        preprocessing_stats = {}
        
        if ocr_system.enhanced_manager:
            ocr_stats = ocr_system.enhanced_manager.get_statistics()
            preprocessing_stats = ocr_system.enhanced_manager.get_preprocessing_statistics()
            
        system_info = {
            "available_engines": [],
            "enhanced_mode": bool(ocr_system.enhanced_manager),
            "legacy_mode": bool(ocr_system.legacy_manager),
            "api_version": "2.1.0"
        }
        
        if ocr_system.enhanced_manager:
            system_info["available_engines"] = list(ocr_system.enhanced_manager.engines.keys())
        elif ocr_system.legacy_manager:
            system_info["available_engines"] = list(ocr_system.legacy_manager.engines.keys())
        
        return StatsResponse(
            ocr_statistics=ocr_stats,
            preprocessing_statistics=preprocessing_stats,
            system_info=system_info
        )
        
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")

@app.post("/reset-stats")
async def reset_statistics():
    """Reset all statistics"""
    if not ocr_system:
        raise HTTPException(status_code=500, detail="OCR system not initialized")
    
    try:
        if ocr_system.enhanced_manager:
            ocr_system.enhanced_manager.reset_statistics()
            
        return {"message": "Statistics reset successfully"}
        
    except Exception as e:
        logger.error(f"Reset stats error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reset statistics: {str(e)}")

@app.get("/demo")
async def demo_page():
    """Enhanced interactive demo page"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Advanced OCR System Demo</title>
        <style>
            body { 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                margin: 0; 
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }
            .container { 
                max-width: 1000px; 
                margin: 0 auto; 
                background: white;
                border-radius: 10px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.3);
                overflow: hidden;
            }
            .header {
                background: linear-gradient(45deg, #667eea, #764ba2);
                color: white;
                padding: 30px;
                text-align: center;
            }
            .content { padding: 30px; }
            .upload-area { 
                border: 3px dashed #667eea; 
                padding: 40px; 
                text-align: center; 
                margin: 20px 0;
                border-radius: 10px;
                background: #f8f9ff;
                transition: all 0.3s ease;
            }
            .upload-area:hover {
                border-color: #764ba2;
                background: #f0f2ff;
            }
            .options-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }
            .option-group {
                background: #f8f9ff;
                padding: 20px;
                border-radius: 10px;
                border: 1px solid #e1e5ff;
            }
            .option-group h3 {
                margin-top: 0;
                color: #667eea;
            }
            .option { 
                margin: 15px 0; 
                display: flex;
                align-items: center;
            }
            .option input[type="checkbox"] {
                margin-right: 10px;
                transform: scale(1.2);
            }
            .option select, .option input[type="text"] {
                padding: 8px;
                border: 2px solid #e1e5ff;
                border-radius: 5px;
                margin-left: 10px;
            }
            .submit-btn {
                background: linear-gradient(45deg, #667eea, #764ba2);
                color: white;
                padding: 15px 30px;
                border: none;
                border-radius: 25px;
                font-size: 16px;
                cursor: pointer;
                transition: all 0.3s ease;
                width: 100%;
                margin: 20px 0;
            }
            .submit-btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
            }
            .result { 
                margin: 20px 0; 
                padding: 25px; 
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            .error { 
                background: #ffebee; 
                border-left: 5px solid #f44336; 
                color: #c62828;
            }
            .success { 
                background: #e8f5e8; 
                border-left: 5px solid #4caf50;
                color: #2e7d32;
            }
            .processing {
                background: #fff3e0;
                border-left: 5px solid #ff9800;
                color: #ef6c00;
            }
            .tabs {
                display: flex;
                background: #f5f5f5;
                border-radius: 10px;
                margin-bottom: 20px;
                overflow: hidden;
            }
            .tab {
                flex: 1;
                padding: 15px;
                text-align: center;
                cursor: pointer;
                background: #f5f5f5;
                border: none;
                transition: all 0.3s ease;
            }
            .tab.active {
                background: #667eea;
                color: white;
            }
            .tab-content {
                display: none;
            }
            .tab-content.active {
                display: block;
            }
            pre {
                background: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
                overflow-x: auto;
                white-space: pre-wrap;
                word-wrap: break-word;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Advanced OCR System Demo</h1>
                <p>Upload images and extract text using multiple OCR engines with intelligent preprocessing</p>
            </div>
            
            <div class="content">
                <div class="tabs">
                    <button class="tab active" onclick="switchTab('single')">Single Image</button>
                    <button class="tab" onclick="switchTab('batch')">Batch Processing</button>
                    <button class="tab" onclick="switchTab('stats')">Statistics</button>
                </div>
                
                <!-- Single Image Tab -->
                <div id="single-tab" class="tab-content active">
                    <form id="ocrForm" enctype="multipart/form-data">
                        <div class="upload-area">
                            <input type="file" id="imageFile" name="file" accept="image/*" required>
                            <p><strong>Choose an image file to process</strong></p>
                            <small>Supported formats: JPG, PNG, TIFF, BMP</small>
                        </div>
                        
                        <div class="options-grid">
                            <div class="option-group">
                                <h3>OCR Engines</h3>
                                <div class="option">
                                    <input type="text" name="engines" value="paddleocr,easyocr" placeholder="comma,separated,engines">
                                </div>
                            </div>
                            
                            <div class="option-group">
                                <h3>Preprocessing Options</h3>
                                <div class="option">
                                    <label><input type="checkbox" name="enable_preprocessing" checked> Enable Adaptive Preprocessing</label>
                                </div>
                                <div class="option">
                                    <label>Intensity: 
                                        <select name="preprocessing_level">
                                            <option value="minimal">Minimal</option>
                                            <option value="light">Light</option>
                                            <option value="balanced" selected>Balanced</option>
                                            <option value="intensive">Intensive</option>
                                            <option value="maximum">Maximum</option>
                                        </select>
                                    </label>
                                </div>
                                <div class="option">
                                    <label>Strategy: 
                                        <select name="preprocessing_strategy">
                                            <option value="speed_optimized">Speed Optimized</option>
                                            <option value="quality_optimized">Quality Optimized</option>
                                            <option value="content_aware" selected>Content Aware</option>
                                        </select>
                                    </label>
                                </div>
                            </div>
                            
                            <div class="option-group">
                                <h3>Processing Options</h3>
                                <div class="option">
                                    <label><input type="checkbox" name="parallel_processing" checked> Parallel Processing</label>
                                </div>
                                <div class="option">
                                    <label><input type="checkbox" name="validate_preprocessing" checked> Validate Preprocessing</label>
                                </div>
                                <div class="option">
                                    <label>Confidence Threshold: 
                                        <input type="range" name="confidence_threshold" min="0" max="1" step="0.1" value="0.7" oninput="this.nextElementSibling.innerHTML=this.value">
                                        <span>0.7</span>
                                    </label>
                                </div>
                            </div>
                            
                            <div class="option-group">
                                <h3>Output Options</h3>
                                <div class="option">
                                    <label>Format: 
                                        <select name="output_format">
                                            <option value="json" selected>JSON</option>
                                            <option value="detailed">Detailed</option>
                                        </select>
                                    </label>
                                </div>
                            </div>
                        </div>
                        
                        <button type="submit" class="submit-btn">Process Image</button>
                    </form>
                    
                    <div id="result" class="result" style="display: none;"></div>
                </div>
                
                <!-- Batch Processing Tab -->
                <div id="batch-tab" class="tab-content">
                    <form id="batchForm" enctype="multipart/form-data">
                        <div class="upload-area">
                            <input type="file" id="batchFiles" name="files" accept="image/*" multiple required>
                            <p><strong>Choose multiple images for batch processing</strong></p>
                            <small>Hold Ctrl/Cmd to select multiple files</small>
                        </div>
                        
                        <div class="options-grid">
                            <div class="option-group">
                                <h3>Batch Options</h3>
                                <div class="option">
                                    <input type="text" name="engines" value="paddleocr,easyocr" placeholder="OCR Engines">
                                </div>
                                <div class="option">
                                    <label><input type="checkbox" name="enable_preprocessing" checked> Enable Preprocessing</label>
                                </div>
                                <div class="option">
                                    <label>Level: 
                                        <select name="preprocessing_level">
                                            <option value="balanced" selected>Balanced</option>
                                            <option value="light">Light</option>
                                            <option value="intensive">Intensive</option>
                                        </select>
                                    </label>
                                </div>
                            </div>
                        </div>
                        
                        <button type="submit" class="submit-btn">Process Batch</button>
                    </form>
                    
                    <div id="batchResult" class="result" style="display: none;"></div>
                </div>
                
                <!-- Statistics Tab -->
                <div id="stats-tab" class="tab-content">
                    <button onclick="loadStats()" class="submit-btn">Load Statistics</button>
                    <div id="statsResult" class="result" style="display: none;"></div>
                </div>
            </div>
        </div>
        
        <script>
            function switchTab(tabName) {
                // Hide all tabs
                document.querySelectorAll('.tab-content').forEach(tab => {
                    tab.classList.remove('active');
                });
                document.querySelectorAll('.tab').forEach(tab => {
                    tab.classList.remove('active');
                });
                
                // Show selected tab
                document.getElementById(tabName + '-tab').classList.add('active');
                event.target.classList.add('active');
            }
            
            // Single image processing
            document.getElementById('ocrForm').onsubmit = async function(e) {
                e.preventDefault();
                
                const formData = new FormData(this);
                const resultDiv = document.getElementById('result');
                
                resultDiv.style.display = 'block';
                resultDiv.innerHTML = '<h3>Processing...</h3><p>Please wait while we analyze your image...</p>';
                resultDiv.className = 'result processing';
                
                try {
                    const response = await fetch('/ocr', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    
                    if (result.success) {
                        resultDiv.className = 'result success';
                        resultDiv.innerHTML = `
                            <h3>OCR Results</h3>
                            <div style="margin: 15px 0;">
                                <strong>Extracted Text:</strong>
                                <pre>${result.text}</pre>
                            </div>
                            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0;">
                                <div><strong>Confidence:</strong> ${(result.confidence * 100).toFixed(1)}%</div>
                                <div><strong>Processing Time:</strong> ${result.processing_time.toFixed(2)}s</div>
                                <div><strong>Word Count:</strong> ${result.word_count || 0}</div>
                                <div><strong>Line Count:</strong> ${result.line_count || 0}</div>
                                <div><strong>Engines Used:</strong> ${result.engines_used.join(', ')}</div>
                            </div>
                            ${result.warnings.length > 0 ? `<div><strong>Warnings:</strong><ul>${result.warnings.map(w => '<li>' + w + '</li>').join('')}</ul></div>` : ''}
                            ${result.preprocessing_info ? `<details><summary><strong>Preprocessing Info</strong></summary><pre>${JSON.stringify(result.preprocessing_info, null, 2)}</pre></details>` : ''}
                        `;
                    } else {
                        resultDiv.className = 'result error';
                        resultDiv.innerHTML = `
                            <h3>Processing Failed</h3>
                            <p><strong>Error:</strong> ${result.error || 'Unknown error occurred'}</p>
                            ${result.warnings.length > 0 ? `<div><strong>Warnings:</strong><ul>${result.warnings.map(w => '<li>' + w + '</li>').join('')}</ul></div>` : ''}
                        `;
                    }
                } catch (error) {
                    resultDiv.className = 'result error';
                    resultDiv.innerHTML = `<h3>Request Failed</h3><p><strong>Error:</strong> ${error.message}</p>`;
                }
            };
            
            // Batch processing
            document.getElementById('batchForm').onsubmit = async function(e) {
                e.preventDefault();
                
                const formData = new FormData(this);
                const resultDiv = document.getElementById('batchResult');
                
                resultDiv.style.display = 'block';
                resultDiv.innerHTML = '<h3>Processing Batch...</h3><p>Processing multiple images, please wait...</p>';
                resultDiv.className = 'result processing';
                
                try {
                    const response = await fetch('/ocr/batch', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    
                    if (result.success) {
                        let resultsHtml = `
                            <h3>Batch Processing Complete</h3>
                            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0;">
                                <div><strong>Total Images:</strong> ${result.batch_statistics.total_images}</div>
                                <div><strong>Successful:</strong> ${result.batch_statistics.successful_images}</div>
                                <div><strong>Success Rate:</strong> ${(result.batch_statistics.success_rate * 100).toFixed(1)}%</div>
                                <div><strong>Total Time:</strong> ${result.total_processing_time.toFixed(2)}s</div>
                                <div><strong>Avg Time:</strong> ${result.batch_statistics.average_processing_time.toFixed(2)}s</div>
                            </div>
                            <h4>Individual Results:</h4>
                        `;
                        
                        result.results.forEach((res, idx) => {
                            resultsHtml += `
                                <details style="margin: 10px 0; padding: 10px; border: 1px solid #ddd; border-radius: 5px;">
                                    <summary><strong>Image ${idx + 1}</strong> - ${res.success ? 'Success' : 'Failed'} (${(res.confidence * 100).toFixed(1)}%)</summary>
                                    <div style="margin-top: 10px;">
                                        ${res.success ? `<pre>${res.text}</pre>` : `<p style="color: #f44336;">Processing failed</p>`}
                                        <small>Processing time: ${res.processing_time.toFixed(2)}s</small>
                                    </div>
                                </details>
                            `;
                        });
                        
                        resultDiv.className = 'result success';
                        resultDiv.innerHTML = resultsHtml;
                    } else {
                        resultDiv.className = 'result error';
                        resultDiv.innerHTML = `<h3>Batch Processing Failed</h3><p><strong>Error:</strong> Batch processing encountered errors</p>`;
                    }
                } catch (error) {
                    resultDiv.className = 'result error';
                    resultDiv.innerHTML = `<h3>Request Failed</h3><p><strong>Error:</strong> ${error.message}</p>`;
                }
            };
            
            // Load statistics
            async function loadStats() {
                const resultDiv = document.getElementById('statsResult');
                
                resultDiv.style.display = 'block';
                resultDiv.innerHTML = '<h3>Loading Statistics...</h3>';
                resultDiv.className = 'result processing';
                
                try {
                    const response = await fetch('/stats');
                    const result = await response.json();
                    
                    resultDiv.className = 'result success';
                    resultDiv.innerHTML = `
                        <h3>System Statistics</h3>
                        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px;">
                            <div>
                                <h4>System Info</h4>
                                <pre>${JSON.stringify(result.system_info, null, 2)}</pre>
                            </div>
                            <div>
                                <h4>OCR Statistics</h4>
                                <pre>${JSON.stringify(result.ocr_statistics, null, 2)}</pre>
                            </div>
                            <div>
                                <h4>Preprocessing Statistics</h4>
                                <pre>${JSON.stringify(result.preprocessing_statistics, null, 2)}</pre>
                            </div>
                        </div>
                        <button onclick="resetStats()" style="margin-top: 20px; padding: 10px 20px; background: #f44336; color: white; border: none; border-radius: 5px; cursor: pointer;">Reset Statistics</button>
                    `;
                } catch (error) {
                    resultDiv.className = 'result error';
                    resultDiv.innerHTML = `<h3>Failed to Load Statistics</h3><p><strong>Error:</strong> ${error.message}</p>`;
                }
            }
            
            // Reset statistics
            async function resetStats() {
                try {
                    await fetch('/reset-stats', { method: 'POST' });
                    alert('Statistics reset successfully');
                    loadStats();
                } catch (error) {
                    alert('Failed to reset statistics: ' + error.message);
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)