from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import pandas as pd
import numpy as np
import json
from datetime import datetime
import asyncio
from pathlib import Path

from models.training import TrainingManager
from models.prediction import PredictionManager
from utils.file_handler import FileHandler
from utils.data_processor import DataProcessor
from utils.reset_utility import ResetUtility

app = FastAPI(title="Exoplanet AI Backend", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize managers
training_manager = TrainingManager()
prediction_manager = PredictionManager()
file_handler = FileHandler()
data_processor = DataProcessor()

# Pydantic models
class DatasetUploadResponse(BaseModel):
    mission: str
    filename: str
    rows: int
    columns: int
    message: str

class TrainingRequest(BaseModel):
    mission: str
    dataset_filename: str
    models: List[str]  # ["xgboost", "cnn1d", "resnet1d", "transformer"]
    test_size: float = 0.2

class TrainingResponse(BaseModel):
    training_id: str
    mission: str
    models: List[str]
    status: str
    message: str

class PredictionRequest(BaseModel):
    mission: str
    dataset_filename: str
    model_name: str
    data: List[Dict[str, Any]]
    true_labels: Optional[List[int]] = None

class PredictionResponse(BaseModel):
    predictions: List[float]
    labels: List[str]
    confidence: List[float]
    metrics: Optional[Dict[str, Any]] = None
    visualizations: Optional[Dict[str, str]] = None

class BatchPredictionRequest(BaseModel):
    mission: str
    dataset_filename: str
    model_name: str
    true_labels_column: Optional[str] = None

class BatchPredictionResponse(BaseModel):
    predictions: List[float]
    labels: List[str]
    confidence: List[float]
    total_samples: int
    confirmed_count: int
    false_positive_count: int
    average_confidence: float
    metrics: Optional[Dict[str, Any]] = None
    visualizations: Optional[Dict[str, str]] = None

class ModelStatus(BaseModel):
    training_id: str
    mission: str
    models: List[str]
    status: str
    progress: float
    metrics: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# API Routes
@app.get("/")
async def root():
    return {"message": "Exoplanet AI Backend API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/api/datasets/upload", response_model=DatasetUploadResponse)
async def upload_dataset(
    mission: str = Form(...),
    file: UploadFile = File(...)
):
    """Upload dataset for a specific mission (kepler, k2, tess)"""
    try:
        # Validate mission
        if mission.lower() not in ["kepler", "k2", "tess"]:
            raise HTTPException(status_code=400, detail="Invalid mission. Must be kepler, k2, or tess")
        
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        if not file.filename.lower().endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are supported")
        
        # Save file
        filename = await file_handler.save_uploaded_file(file, mission)
        
        # Process and validate dataset
        dataset_info = await data_processor.process_dataset(filename, mission)
        
        return DatasetUploadResponse(
            mission=mission,
            filename=filename,
            rows=dataset_info["rows"],
            columns=dataset_info["columns"],
            message=f"Dataset uploaded successfully for {mission} mission"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.get("/api/datasets")
async def list_datasets():
    """List all uploaded datasets"""
    try:
        datasets = await file_handler.list_datasets()
        return {"datasets": datasets}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/training/start", response_model=TrainingResponse)
async def start_training(
    request: TrainingRequest,
    background_tasks: BackgroundTasks
):
    """Start training models on a dataset"""
    try:
        # Validate request
        if request.mission.lower() not in ["kepler", "k2", "tess"]:
            raise HTTPException(status_code=400, detail="Invalid mission")
        
        valid_models = ["xgboost", "cnn1d", "resnet1d", "transformer"]
        if not all(model in valid_models for model in request.models):
            raise HTTPException(status_code=400, detail="Invalid model type")
        
        # Start training in background
        training_id = await training_manager.start_training(
            mission=request.mission,
            dataset_filename=request.dataset_filename,
            models=request.models,
            test_size=request.test_size,
            background_tasks=background_tasks
        )
        
        return TrainingResponse(
            training_id=training_id,
            mission=request.mission,
            models=request.models,
            status="started",
            message="Training started successfully"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/training/status/{training_id}", response_model=ModelStatus)
async def get_training_status(training_id: str):
    """Get training status for a specific training job"""
    try:
        status = await training_manager.get_training_status(training_id)
        return ModelStatus(**status)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/training/history")
async def get_training_history():
    """Get all training history"""
    try:
        history = await training_manager.get_training_history()
        return {"training_history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/prediction/predict", response_model=PredictionResponse)
async def make_prediction(request: PredictionRequest):
    """Make predictions using a trained model with optional evaluation"""
    try:
        result = await prediction_manager.predict_with_evaluation(
            mission=request.mission,
            dataset_filename=request.dataset_filename,
            model_name=request.model_name,
            data=request.data,
            true_labels=request.true_labels
        )
        
        return PredictionResponse(
            predictions=result["predictions"],
            labels=result["labels"],
            confidence=result["confidence"],
            metrics=result.get("metrics"),
            visualizations=result.get("visualizations")
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/prediction/batch", response_model=BatchPredictionResponse)
async def batch_predict(
    mission: str = Form(...),
    dataset_filename: str = Form(...),
    model_name: str = Form(...),
    file: UploadFile = File(...),
    true_labels_column: Optional[str] = Form(None)
):
    """Make batch predictions from a CSV file"""
    temp_file_path = None
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        if not file.filename.lower().endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are supported")
        
        # Validate mission
        if mission.lower() not in ["kepler", "k2", "tess"]:
            raise HTTPException(status_code=400, detail="Invalid mission. Must be kepler, k2, or tess")
        
        # Save uploaded file temporarily
        temp_file_path = f"temp_{file.filename}"
        with open(temp_file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        print(f"Batch prediction request: mission={mission}, dataset={dataset_filename}, model={model_name}")
        print(f"Temporary file saved: {temp_file_path}")
        
        # Run batch prediction
        result = await prediction_manager.predict_batch(
            mission=mission,
            dataset_filename=dataset_filename,
            model_name=model_name,
            csv_file_path=temp_file_path,
            true_labels_column=true_labels_column
        )
        
        print(f"Batch prediction completed successfully: {len(result['predictions'])} samples")
        
        return BatchPredictionResponse(
            predictions=result["predictions"],
            labels=result["labels"],
            confidence=result["confidence"],
            total_samples=result["total_samples"],
            confirmed_count=result["confirmed_count"],
            false_positive_count=result["false_positive_count"],
            average_confidence=result["average_confidence"],
            metrics=result.get("metrics"),
            visualizations=result.get("visualizations")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Batch prediction error: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")
    finally:
        # Clean up temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                print(f"Cleaned up temporary file: {temp_file_path}")
            except Exception as cleanup_error:
                print(f"Error cleaning up temporary file: {cleanup_error}")

@app.get("/api/models/available")
async def get_available_models():
    """Get list of available trained models"""
    try:
        models = await prediction_manager.get_available_models()
        print(f"Available models: {models}")
        return {"models": models}
    except Exception as e:
        print(f"Error getting available models: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/models/{model_id}/metrics")
async def get_model_metrics(model_id: str):
    """Get metrics for a specific trained model"""
    try:
        metrics = await prediction_manager.get_model_metrics(model_id)
        return {"metrics": metrics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Reset endpoints
@app.get("/api/reset/summary")
async def get_reset_summary():
    """Get summary of current data before reset"""
    try:
        reset_util = ResetUtility()
        summary = reset_util.get_data_summary()
        return {
            "status": "success",
            "data": summary
        }
    except Exception as e:
        print(f"Error getting reset summary: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get reset summary: {str(e)}")


@app.post("/api/reset/all")
async def reset_all_data():
    """Reset all training and prediction data"""
    try:
        print("Starting data reset...")
        reset_util = ResetUtility()
        results = reset_util.reset_all_data()
        
        print(f"Reset completed: {results}")
        
        return {
            "status": "success",
            "message": "All data reset successfully",
            "data": results
        }
    except Exception as e:
        print(f"Error during reset: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Reset failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
