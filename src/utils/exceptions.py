"""
Custom Exception Classes for Telco Churn MLOps Platform.
"""


class MLOpsError(Exception):
    """Base exception for MLOps platform."""
    
    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ConfigurationError(MLOpsError):
    """Configuration loading or validation error."""
    pass


class DataError(MLOpsError):
    """Data loading, validation, or processing error."""
    pass


class DataValidationError(DataError):
    """Data validation failure."""
    pass


class FeatureEngineeringError(MLOpsError):
    """Feature engineering error."""
    pass


class ModelError(MLOpsError):
    """Model training or inference error."""
    pass


class ModelTrainingError(ModelError):
    """Model training failure."""
    pass


class ModelPredictionError(ModelError):
    """Model prediction failure."""
    pass


class MLflowError(MLOpsError):
    """MLflow tracking or registry error."""
    pass


class APIError(MLOpsError):
    """API-related error."""
    pass


class MonitoringError(MLOpsError):
    """Monitoring or drift detection error."""
    pass
