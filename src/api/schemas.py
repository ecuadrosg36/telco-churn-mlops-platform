"""
Pydantic schemas for FastAPI request/response validation.

Defines data models for API endpoints with validation rules.
"""

from pydantic import BaseModel, Field, constr, confloat, conint
from typing import Optional, List, Literal, Dict, Any
from datetime import datetime


class ChurnPredictionRequest(BaseModel):
    """Request schema for churn prediction."""

    # Customer information
    gender: Literal["Male", "Female"] = Field(..., description="Customer gender")
    senior_citizen: Literal[0, 1] = Field(
        ...,
        alias="SeniorCitizen",
        description="Whether customer is senior (0=No, 1=Yes)",
    )
    partner: Literal["Yes", "No"] = Field(
        ..., alias="Partner", description="Whether customer has a partner"
    )
    dependents: Literal["Yes", "No"] = Field(
        ..., alias="Dependents", description="Whether customer has dependents"
    )

    # Account information
    tenure: conint(ge=0, le=100) = Field(
        ..., description="Number of months as customer (0-100)"
    )

    # Service subscriptions
    phone_service: Literal["Yes", "No"] = Field(
        ..., alias="PhoneService", description="Phone service subscription"
    )
    multiple_lines: Literal["Yes", "No", "No phone service"] = Field(
        ..., alias="MultipleLines"
    )
    internet_service: Literal["DSL", "Fiber optic", "No"] = Field(
        ..., alias="InternetService"
    )
    online_security: Literal["Yes", "No", "No internet service"] = Field(
        ..., alias="OnlineSecurity"
    )
    online_backup: Literal["Yes", "No", "No internet service"] = Field(
        ..., alias="OnlineBackup"
    )
    device_protection: Literal["Yes", "No", "No internet service"] = Field(
        ..., alias="DeviceProtection"
    )
    tech_support: Literal["Yes", "No", "No internet service"] = Field(
        ..., alias="TechSupport"
    )
    streaming_tv: Literal["Yes", "No", "No internet service"] = Field(
        ..., alias="StreamingTV"
    )
    streaming_movies: Literal["Yes", "No", "No internet service"] = Field(
        ..., alias="StreamingMovies"
    )

    # Billing information
    contract: Literal["Month-to-month", "One year", "Two year"] = Field(
        ..., alias="Contract"
    )
    paperless_billing: Literal["Yes", "No"] = Field(..., alias="PaperlessBilling")
    payment_method: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ] = Field(..., alias="PaymentMethod")

    # Charges
    monthly_charges: confloat(ge=0.0) = Field(
        ..., alias="MonthlyCharges", description="Monthly charge amount"
    )
    total_charges: confloat(ge=0.0) = Field(
        ..., alias="TotalCharges", description="Total charges to date"
    )

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 12,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "DSL",
                "OnlineSecurity": "Yes",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "Yes",
                "StreamingTV": "No",
                "StreamingMovies": "No",
                "Contract": "One year",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 65.6,
                "TotalCharges": 787.2,
            }
        }


class ChurnPredictionResponse(BaseModel):
    """Response schema for churn prediction."""

    prediction: Literal["Yes", "No"] = Field(
        ..., description="Churn prediction (Yes/No)"
    )
    churn_probability: float = Field(
        ..., ge=0.0, le=1.0, description="Probability of churn (0-1)"
    )
    risk_tier: Literal["low", "medium", "high"] = Field(
        ..., description="Risk tier based on probability"
    )
    model_version: str = Field(..., description="Model version used for prediction")
    model_stage: str = Field(..., description="Model stage (Staging/Production)")
    prediction_timestamp: datetime = Field(..., description="Timestamp of prediction")

    class Config:
        json_schema_extra = {
            "example": {
                "prediction": "No",
                "churn_probability": 0.23,
                "risk_tier": "low",
                "model_version": "3",
                "model_stage": "Production",
                "prediction_timestamp": "2024-01-15T10:30:00",
            }
        }


class BatchPredictionRequest(BaseModel):
    """Request schema for batch predictions."""

    customers: List[ChurnPredictionRequest] = Field(..., min_length=1, max_length=1000)


class BatchPredictionResponse(BaseModel):
    """Response schema for batch predictions."""

    predictions: List[ChurnPredictionResponse]
    total_count: int
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int


class HealthResponse(BaseModel):
    """Response schema for health check."""

    status: Literal["healthy", "unhealthy"] = Field(
        ..., description="Service health status"
    )
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_version: Optional[str] = Field(None, description="Loaded model version")
    model_stage: Optional[str] = Field(None, description="Loaded model stage")
    timestamp: datetime = Field(..., description="Health check timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "model_version": "3",
                "model_stage": "Production",
                "timestamp": "2024-01-15T10:30:00",
            }
        }


class ModelInfoResponse(BaseModel):
    """Response schema for model information."""

    model_name: str
    version: str
    stage: str
    created_at: Optional[float] = None
    last_updated: Optional[float] = None
    metrics: Dict[str, float] = Field(default_factory=dict)
    params: Dict[str, Any] = Field(default_factory=dict)
    description: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "model_name": "telco-churn-model",
                "version": "3",
                "stage": "Production",
                "metrics": {
                    "val_accuracy": 0.7985,
                    "val_f1": 0.5612,
                    "val_auc": 0.8432,
                },
                "params": {
                    "n_estimators": "200",
                    "max_depth": "6",
                    "learning_rate": "0.1",
                },
            }
        }


class ErrorResponse(BaseModel):
    """Response schema for errors."""

    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: datetime = Field(..., description="Error timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "error": "Prediction failed",
                "detail": "Invalid input format",
                "timestamp": "2024-01-15T10:30:00",
            }
        }
