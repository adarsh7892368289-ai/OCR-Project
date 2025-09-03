# src/api/enhanced_main.py - Updated FastAPI with Step 4 Integration

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
import cv2
import numpy as np
import io
from PIL import Image
import logging
import time
from pathlib import Path

# Import enhanced components
from ..core.enhanced_engine_manager import (
    EnhancedEngineManager, EnhancedProcessingOptions, 
    quick_ocr_with_preprocessing, batch_ocr_with_preprocessing
)
from ..preprocessing.adaptive_processor import ProcessingLevel as PrepLevel, PipelineStrategy

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="Enhanced OCR API with Adaptive Preprocessing",
    description="Advanced OCR API with intelligent preprocessing pipeline",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global manager instance
manager = None

# Pydantic models
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

class OCRResponse(BaseModel):
    success: bool
    text: str
    confidence: float
    processing_time: float
    engines_used: List[str]
    preprocessing_info: Dict[str, Any]
    metadata: Dict[str, Any]
    warnings: List[str]

class BatchOCRResponse(BaseModel):
    success: bool
    results: List[OCRResponse]
    total_processing_time: float
    batch_statistics: Dict[str, Any]

class StatsResponse(BaseModel):
    ocr_statistics: Dict[str, Any]
    preprocessing_statistics: Dict[str, Any]
    system_info: Dict[str, Any]

@app.on_event("startup")
async def startup_event():
    """Initialize the enhanced engine manager on startup"""
    global manager
    try:
        manager = EnhancedEngineManager()
        logger.info("Enhanced OCR API started successfully")
    except Exception as e:
        logger.error(f"Failed to initialize OCR manager: {e}")
        raise

@app.on_event("shutdown") 
async def shutdown_event():
    """Cleanup on shutdown"""
    global manager
    if manager:
        manager.shutdown()
        logger.info("OCR API shutdown complete")

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

@app.post("/ocr", response_model=OCRResponse)
async def perform_ocr(
    file: UploadFile = File(...),
    engines: str = "paddleocr,easyocr",
    enable_preprocessing: bool = True,
    preprocessing_level: str = "balanced",
    preprocessing_strategy: str = "content_aware",
    preprocess_per_engine: bool = False,
    parallel_processing: bool = True,
    confidence_threshold: float = 0.7,
    validate_preprocessing: bool = True,
    fallback_to_original: bool = True
):
    """
    Perform OCR on uploaded image with adaptive preprocessing
    """
    if not manager:
        raise HTTPException(status_code=500, detail="OCR manager not initialized")
    
    try:
        # Convert image
        image = image_from_upload(file)
        
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
            fallback_to_original=fallback_to_original
        )
        
        # Create processing options
        options = create_processing_options(request)
        
        # Process image
        results = manager.process_image(image, options)
        
        # Format response
        if results["success"] and results["combined_result"]:
            response = OCRResponse(
                success=True,
                text=results["combined_result"].text,
                confidence=results["combined_result"].confidence,
                processing_time=results["processing_time"],
                engines_used=results["engines_used"],
                preprocessing_info=results.get("preprocessing_results", {}),
                metadata=results["metadata"],
                warnings=results.get("warnings", [])
            )
        else:
            response = OCRResponse(
                success=False,
                text="",
                confidence=0.0,
                processing_time=results["processing_time"],
                engines_used=results["engines_used"],
                preprocessing_info=results.get("preprocessing_results", {}),
                metadata=results["metadata"],
                warnings=results.get("warnings", ["OCR processing failed"])
            )
        
        return response
        
    except Exception as e:
        logger.error(f"OCR processing error: {e}")
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")

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
    if not manager:
        raise HTTPException(status_code=500, detail="OCR manager not initialized")
    
    try:
        start_time = time.time()
        
        # Convert all images
        images = []
        for file in files:
            image = image_from_upload(file)
            images.append(image)
        
        # Create request object
        request = OCRRequest(
            engines=engines.split(","),
            enable_preprocessing=enable_preprocessing,
            preprocessing_level=preprocessing_level,
            parallel_processing=parallel_processing
        )
        
        options = create_processing_options(request)
        
        # Process all images
        batch_results = []
        successful_count = 0
        
        for i, image in enumerate(images):
            result = manager.process_image(image, options)
            
            if result["success"] and result["combined_result"]:
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
                    engines_used=result["engines_used"],
                    preprocessing_info=result.get("preprocessing_results", {}),
                    metadata=result["metadata"],
                    warnings=result.get("warnings", ["Processing failed"])
                )
            
            batch_results.append(ocr_response)
        
        total_time = time.time() - start_time
        
        # Create batch response
        response = BatchOCRResponse(
            success=successful_count > 0,
            results=batch_results,
            total_processing_time=total_time,
            batch_statistics={
                "total_images": len(images),
                "successful_images": successful_count,
                "success_rate": successful_count / len(images),
                "average_processing_time": total_time / len(images)
            }
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Batch OCR error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch OCR failed: {str(e)}")

@app.get("/stats", response_model=StatsResponse)
async def get_statistics():
    """Get system and processing statistics"""
    if not manager:
        raise HTTPException(status_code=500, detail="OCR manager not initialized")
    
    try:
        ocr_stats = manager.get_statistics()
        preprocessing_stats = manager.get_preprocessing_statistics()
        
        system_info = {
            "available_engines": list(manager.engines.keys()),
            "preprocessing_enabled": True,
            "api_version": "2.0.0"
        }
        
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
    if not manager:
        raise HTTPException(status_code=500, detail="OCR manager not initialized")
    
    try:
        manager.reset_statistics()
        return {"message": "Statistics reset successfully"}
        
    except Exception as e:
        logger.error(f"Reset stats error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reset statistics: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if not manager:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "message": "OCR manager not initialized"}
        )
    
    return {"status": "healthy", "timestamp": time.time()}

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Enhanced OCR API with Adaptive Preprocessing",
        "version": "2.0.0",
        "features": [
            "Multi-engine OCR support",
            "Adaptive preprocessing pipeline", 
            "Quality-aware processing",
            "Parallel processing",
            "Batch operations",
            "Performance monitoring"
        ],
        "endpoints": {
            "POST /ocr": "Single image OCR",
            "POST /ocr/batch": "Batch image OCR", 
            "GET /stats": "System statistics",
            "POST /reset-stats": "Reset statistics",
            "GET /health": "Health check"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)