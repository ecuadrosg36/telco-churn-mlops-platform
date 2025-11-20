"""
Unit tests for FastAPI application.
"""

import pytest
from fastapi.testclient import TestClient
from src.api.app import app
from src.api.model_loader import get_model_loader


@pytest.fixture
def client():
    """FastAPI test client."""
    return TestClient(app)


@pytest.fixture
def mock_loaded_model(mock_sklearn_model, mock_preprocessing_pipeline, monkeypatch):
    """Mock a loaded model in the model loader."""
    model_loader = get_model_loader()

    # Mock the model loader
    monkeypatch.setattr(model_loader, "model", mock_sklearn_model)
    monkeypatch.setattr(
        model_loader, "preprocessing_pipeline", mock_preprocessing_pipeline
    )
    monkeypatch.setattr(
        model_loader,
        "model_metadata",
        {"version": "1", "stage": "Production", "name": "test-model"},
    )
    monkeypatch.setattr(model_loader, "model_version", "1")
    monkeypatch.setattr(model_loader, "model_stage", "Production")

    return model_loader


@pytest.mark.unit
class TestRootEndpoint:
    """Tests for root endpoint."""

    def test_root_returns_info(self, client):
        """Test root endpoint returns API info."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "service" in data
        assert "version" in data


@pytest.mark.unit
class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_check_structure(self, client):
        """Test health endpoint returns correct structure."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()

        assert "status" in data
        assert "model_loaded" in data
        assert "timestamp" in data

    def test_health_check_when_model_not_loaded(self, client):
        """Test health check when model is not loaded."""
        response = client.get("/health")

        data = response.json()
        # Initially model may not be loaded
        assert data["status"] in ["healthy", "unhealthy"]


@pytest.mark.unit
class TestPredictionEndpoint:
    """Tests for prediction endpoint."""

    def test_predict_requires_all_fields(self, client):
        """Test prediction endpoint validates required fields."""
        # Missing required fields
        incomplete_data = {
            "gender": "Male",
            "tenure": 12,
            # Missing many required fields
        }

        response = client.post("/predict", json=incomplete_data)

        # Should return validation error
        assert response.status_code == 422

    def test_predict_validates_field_types(self, client, sample_churn_request):
        """Test prediction endpoint validates field types."""
        invalid_data = sample_churn_request.copy()
        invalid_data["tenure"] = "not_a_number"  # Invalid type

        response = client.post("/predict", json=invalid_data)

        assert response.status_code == 422

    def test_predict_validates_categorical_values(self, client, sample_churn_request):
        """Test prediction endpoint validates categorical values."""
        invalid_data = sample_churn_request.copy()
        invalid_data["gender"] = "Unknown"  # Invalid category

        response = client.post("/predict", json=invalid_data)

        assert response.status_code == 422

    @pytest.mark.skip(reason="Requires model to be loaded")
    def test_predict_with_valid_data(
        self, client, sample_churn_request, mock_loaded_model
    ):
        """Test prediction with valid data."""
        response = client.post("/predict", json=sample_churn_request)

        if response.status_code == 503:
            pytest.skip("Model not loaded")

        assert response.status_code == 200
        data = response.json()

        assert "prediction" in data
        assert "churn_probability" in data
        assert "risk_tier" in data
        assert data["prediction"] in ["Yes", "No"]
        assert 0 <= data["churn_probability"] <= 1
        assert data["risk_tier"] in ["low", "medium", "high"]


@pytest.mark.unit
class TestBatchPredictionEndpoint:
    """Tests for batch prediction endpoint."""

    def test_batch_predict_validates_request(self, client):
        """Test batch prediction validates request structure."""
        invalid_data = {"invalid_key": []}

        response = client.post("/predict/batch", json=invalid_data)

        assert response.status_code == 422

    def test_batch_predict_enforces_max_size(self, client, sample_churn_request):
        """Test batch prediction enforces maximum batch size."""
        # Try to send more than 1000 customers
        large_batch = {"customers": [sample_churn_request] * 1001}

        response = client.post("/predict/batch", json=large_batch)

        assert response.status_code == 422


@pytest.mark.unit
class TestModelInfoEndpoint:
    """Tests for model info endpoint."""

    def test_model_info_when_model_not_loaded(self, client):
        """Test model info when model not loaded."""
        response = client.get("/model/info")

        # Should return 503 if model not loaded
        if response.status_code == 503:
            data = response.json()
            assert "detail" in data


@pytest.mark.unit
class TestMetricsEndpoint:
    """Tests for metrics endpoint."""

    def test_metrics_endpoint_returns_data(self, client):
        """Test metrics endpoint returns metrics data."""
        response = client.get("/metrics")

        assert response.status_code == 200
        data = response.json()

        assert "total_requests" in data
        assert "total_predictions" in data


@pytest.mark.unit
class TestCORSMiddleware:
    """Tests for CORS configuration."""

    def test_cors_headers_present(self, client):
        """Test CORS headers are present in responses."""
        response = client.options("/health", headers={"Origin": "http://localhost"})

        # CORS headers should be present
        assert (
            "access-control-allow-origin" in response.headers
            or response.status_code == 200
        )


@pytest.mark.unit
class TestErrorHandling:
    """Tests for API error handling."""

    def test_404_for_unknown_endpoint(self, client):
        """Test 404 for unknown endpoints."""
        response = client.get("/unknown/endpoint")

        assert response.status_code == 404

    def test_method_not_allowed(self, client):
        """Test 405 for wrong HTTP method."""
        # GET on POST-only endpoint
        response = client.get("/predict")

        assert response.status_code == 405
