"""
API observability layer for request/response logging and metrics.

Provides middleware for latency measurement, request logging, and monitoring.
"""

import time
import sys
from pathlib import Path
from typing import Callable
from datetime import datetime
from fastapi import Request, Response
from fastapi.routing import APIRoute

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.logging_utils import setup_logger, get_correlation_id

logger = setup_logger(__name__)


class TimedRoute(APIRoute):
    """Custom APIRoute that adds latency measurement."""

    def get_route_handler(self) -> Callable:
        original_route_handler = super().get_route_handler()

        async def custom_route_handler(request: Request) -> Response:
            # Start timer
            start_time = time.time()

            # Generate correlation ID
            correlation_id = get_correlation_id()
            request.state.correlation_id = correlation_id

            # Log request
            logger.info(
                f"Request started: {request.method} {request.url.path}",
                extra={"correlation_id": correlation_id},
            )

            # Execute request
            response: Response = await original_route_handler(request)

            # Calculate latency
            latency = time.time() - start_time

            # Add headers
            response.headers["X-Correlation-ID"] = correlation_id
            response.headers["X-Process-Time"] = str(latency)

            # Log response
            logger.info(
                f"Request completed: {request.method} {request.url.path} "
                f"[{response.status_code}] in {latency:.3f}s",
                extra={"correlation_id": correlation_id},
            )

            # Log slow requests
            if latency > 1.0:
                logger.warning(
                    f"Slow request detected: {request.url.path} took {latency:.3f}s",
                    extra={"correlation_id": correlation_id},
                )

            return response

        return custom_route_handler


def log_prediction(
    request_data: dict, prediction: str, probability: float, correlation_id: str
) -> None:
    """
    Log prediction for audit trail (sanitized).

    Args:
        request_data: Request data (sanitized)
        prediction: Prediction result
        probability: Prediction probability
        correlation_id: Request correlation ID
    """
    # Sanitize - remove any PII if present
    sanitized_data = {
        "tenure": request_data.get("tenure"),
        "monthly_charges": request_data.get("MonthlyCharges"),
        "contract": request_data.get("Contract"),
        # Don't log full request to avoid PII
    }

    logger.info(
        f"Prediction made: {prediction} (prob: {probability:.4f})",
        extra={
            "correlation_id": correlation_id,
            "prediction": prediction,
            "probability": probability,
            "sample_features": sanitized_data,
        },
    )


class RequestMetrics:
    """Simple in-memory metrics collector."""

    def __init__(self):
        self.request_count = 0
        self.error_count = 0
        self.total_latency = 0.0
        self.prediction_count = 0
        self.high_risk_count = 0

    def record_request(self, latency: float, status_code: int):
        """Record a request."""
        self.request_count += 1
        self.total_latency += latency

        if status_code >= 400:
            self.error_count += 1

    def record_prediction(self, probability: float):
        """Record a prediction."""
        self.prediction_count += 1

        if probability > 0.7:
            self.high_risk_count += 1

    def get_metrics(self) -> dict:
        """Get current metrics."""
        avg_latency = (
            self.total_latency / self.request_count if self.request_count > 0 else 0
        )

        error_rate = (
            self.error_count / self.request_count if self.request_count > 0 else 0
        )

        high_risk_rate = (
            self.high_risk_count / self.prediction_count
            if self.prediction_count > 0
            else 0
        )

        return {
            "total_requests": self.request_count,
            "total_errors": self.error_count,
            "error_rate": error_rate,
            "avg_latency_seconds": avg_latency,
            "total_predictions": self.prediction_count,
            "high_risk_predictions": self.high_risk_count,
            "high_risk_rate": high_risk_rate,
        }

    def reset(self):
        """Reset all metrics."""
        self.__init__()


# Global metrics instance
metrics = RequestMetrics()


def get_metrics() -> RequestMetrics:
    """Get metrics instance."""
    return metrics
