"""
FastAPI application for Telco Churn prediction model serving.

Provides REST API endpoints for:
- Health checks
- Single and batch predictions
- Model information
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
from datetime import datetime
from typing import List
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import get_config
from src.api.schemas import (
    ChurnPredictionRequest,
    ChurnPredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    ErrorResponse
)
from src.api.model_loader import get_model_loader
from src.api.observability import TimedRoute, log_prediction, get_metrics
from src.data.preprocessing import preprocess_data
from src.features.feature_engineering import engineer_features
from src.logging_utils import setup_logger

logger = setup_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Telco Churn Prediction API",
    description="Production ML API for predicting customer churn in telecom industry",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add custom route class for latency measurement
app.router.route_class = TimedRoute

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle all uncaught exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc),
            timestamp=datetime.now()
        ).dict()
    )


# Startup event - load model
@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    logger.info("=" * 60)
    logger.info("STARTING TELCO CHURN API")
    logger.info("=" * 60)
    
    try:
        model_loader = get_model_loader()
        config = get_config()
        stage = config.get('api.model_stage', 'Production')
        
        model, pipeline, metadata = model_loader.load_model(stage=stage)
        
        logger.info(f"✓ API started successfully")
        logger.info(f"  Model version: {metadata.get('version')}")
        logger.info(f"  Model stage: {metadata.get('stage')}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Failed to load model on startup: {str(e)}")
        logger.warning("API started but model not loaded - /predict will fail")


# Health check endpoint
@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health check endpoint"
)
async def health_check():
    """
    Check if the API and model are healthy.
    
    Returns health status, model loading status, and model metadata.
    """
    model_loader = get_model_loader()
    model_info = model_loader.get_model_info()
    
    is_healthy = model_info.get('loaded', False)
    
    return HealthResponse(
        status="healthy" if is_healthy else "unhealthy",
        model_loaded=model_info.get('loaded', False),
        model_version=model_info.get('version'),
        model_stage=model_info.get('stage'),
        timestamp=datetime.now()
    )


# Model info endpoint
@app.get(
    "/model/info",
    response_model=ModelInfoResponse,
    tags=["Model"],
    summary="Get model information"
)
async def get_model_info():
    """
    Get detailed information about the currently loaded model.
    
    Returns model version, stage, metrics, and parameters.
    """
    model_loader = get_model_loader()
    model_info = model_loader.get_model_info()
    
    if not model_info.get('loaded'):
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    metadata = model_info.get('metadata', {})
    
    return ModelInfoResponse(
        model_name=metadata.get('name', 'unknown'),
        version=str(metadata.get('version', 'unknown')),
        stage=metadata.get('stage', 'unknown'),
        created_at=metadata.get('created_at'),
        last_updated=metadata.get('last_updated'),
        metrics=metadata.get('metrics', {}),
        params=metadata.get('params', {}),
        description=metadata.get('description')
    )


# Single prediction endpoint
@app.post(
    "/predict",
    response_model=ChurnPredictionResponse,
    tags=["Prediction"],
    summary="Predict churn for a single customer"
)
async def predict_churn(request: Request, customer: ChurnPredictionRequest):
    """
    Predict churn probability for a single customer.
    
    Accepts customer features and returns prediction, probability, and risk tier.
    """
    try:
        # Get model loader
        model_loader = get_model_loader()
        
        if not model_loader.is_loaded():
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Convert request to dataframe
        customer_dict = customer.dict(by_alias=True)
        df = pd.DataFrame([customer_dict])
        
        # Preprocess features
        X_processed, _, _ = preprocess_data(
            df,
            fitted_pipeline=model_loader.preprocessing_pipeline
        )
        
        # Engineer features
        X_engineered = engineer_features(X_processed)
        
        # Get prediction
        model = model_loader.model
        prediction = model.predict(X_engineered)[0]
        probability = model.predict_proba(X_engineered)[0, 1]
        
        # Determine risk tier
        if probability > 0.7:
            risk_tier = 'high'
        elif probability > 0.3:
            risk_tier = 'medium'
        else:
            risk_tier = 'low'
        
        # Get model info
        model_info = model_loader.get_model_info()
        
        # Log prediction (sanitized)
        correlation_id = getattr(request.state, 'correlation_id', 'unknown')
        log_prediction(customer_dict, str(prediction), probability, correlation_id)
        
        # Record metrics
        metrics = get_metrics()
        metrics.record_prediction(probability)
        
        # Return response
        return ChurnPredictionResponse(
            prediction='Yes' if prediction == 1 else 'No',
            churn_probability=float(probability),
            risk_tier=risk_tier,
            model_version=str(model_info.get('version', 'unknown')),
            model_stage=model_info.get('stage', 'unknown'),
            prediction_timestamp=datetime.now()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# Batch prediction endpoint
@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    tags=["Prediction"],
    summary="Predict churn for multiple customers"
)
async def predict_batch(request: Request, batch_request: BatchPredictionRequest):
    """
    Predict churn probability for multiple customers.
    
    Accepts a list of customers and returns predictions for all.
    Limited to 1000 customers per request.
    """
    try:
        # Get model loader
        model_loader = get_model_loader()
        
        if not model_loader.is_loaded():
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Convert requests to dataframe
        customers_data = [c.dict(by_alias=True) for c in batch_request.customers]
        df = pd.DataFrame(customers_data)
        
        # Preprocess features
        X_processed, _, _ = preprocess_data(
            df,
            fitted_pipeline=model_loader.preprocessing_pipeline
        )
        
        # Engineer features
        X_engineered = engineer_features(X_processed)
        
        # Get predictions
        model = model_loader.model
        predictions = model.predict(X_engineered)
        probabilities = model.predict_proba(X_engineered)[:, 1]
        
        # Get model info
        model_info = model_loader.get_model_info()
        
        # Build response
        prediction_responses = []
        high_risk_count = 0
        medium_risk_count = 0
        low_risk_count = 0
        
        for pred, prob in zip(predictions, probabilities):
            # Determine risk tier
            if prob > 0.7:
                risk_tier = 'high'
                high_risk_count += 1
            elif prob > 0.3:
                risk_tier = 'medium'
                medium_risk_count += 1
            else:
                risk_tier = 'low'
                low_risk_count += 1
            
            prediction_responses.append(
                ChurnPredictionResponse(
                    prediction='Yes' if pred == 1 else 'No',
                    churn_probability=float(prob),
                    risk_tier=risk_tier,
                    model_version=str(model_info.get('version', 'unknown')),
                    model_stage=model_info.get('stage', 'unknown'),
                    prediction_timestamp=datetime.now()
                )
            )
        
        # Log batch prediction
        correlation_id = getattr(request.state, 'correlation_id', 'unknown')
        logger.info(
            f"Batch prediction: {len(predictions)} customers",
            extra={'correlation_id': correlation_id}
        )
        
        # Record metrics
        metrics = get_metrics()
        for prob in probabilities:
            metrics.record_prediction(prob)
        
        return BatchPredictionResponse(
            predictions=prediction_responses,
            total_count=len(predictions),
            high_risk_count=high_risk_count,
            medium_risk_count=medium_risk_count,
            low_risk_count=low_risk_count
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


# Metrics endpoint (for monitoring)
@app.get(
    "/metrics",
    tags=["Monitoring"],
    summary="Get API metrics"
)
async def get_api_metrics():
    """
    Get API usage metrics.
    
    Returns request counts, error rates, latency, and prediction statistics.
    """
    metrics = get_metrics()
    return metrics.get_metrics()


# Root endpoint
@app.get("/", tags=["Info"])
async def root():
    """Root endpoint with API information."""
    return {
        "service": "Telco Churn Prediction API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }


if __name__ == "__main__":
    import uvicorn
    
    config = get_config()
    
    uvicorn.run(
        "src.api.app:app",
        host=config.get('api.host', '0.0.0.0'),
        port=config.get('api.port', 8080),
        reload=config.get('api.reload', False),
        log_level=config.get('api.log_level', 'info'),
    )
