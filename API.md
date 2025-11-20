# API Documentation

Complete reference for the Telco Churn Prediction API.

## Base URL

- **Local**: `http://localhost:8080`
- **Cloud**: `https://your-service.run.app`

## Authentication

Currently public. For production, implement OAuth2 or API key authentication.

## Endpoints

### 1. Health Check

Check API and model health status.

**Endpoint**: `GET /health`

**Response**: `200 OK`

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "3",
  "model_stage": "Production",
  "timestamp": "2024-01-15T10:30:00"
}
```

### 2. Single Prediction

Predict churn for a single customer.

**Endpoint**: `POST /predict`

**Request Body**:
```json
{
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
  "TotalCharges": 787.2
}
```

**Response**: `200 OK`

```json
{
  "prediction": "No",
  "churn_probability": 0.23,
  "risk_tier": "low",
  "model_version": "3",
  "model_stage": "Production",
  "prediction_timestamp": "2024-01-15T10:30:00"
}
```

**Risk Tiers**:
- `low`: probability < 0.3
- `medium`: 0.3 ≤ probability < 0.7
- `high`: probability ≥ 0.7

### 3. Batch Prediction

Predict churn for multiple customers (max 1000).

**Endpoint**: `POST /predict/batch`

**Request Body**:
```json
{
  "customers": [
    {
      "gender": "Female",
      "SeniorCitizen": 0,
      // ... all required fields
    },
    {
      "gender": "Male",
      "SeniorCitizen": 1,
      // ... all required fields
    }
  ]
}
```

**Response**: `200 OK`

```json
{
  "predictions": [
    {
      "prediction": "No",
      "churn_probability": 0.23,
      "risk_tier": "low",
      "model_version": "3",
      "model_stage": "Production",
      "prediction_timestamp": "2024-01-15T10:30:00"
    },
    // ... more predictions
  ],
  "total_count": 2,
  "high_risk_count": 0,
  "medium_risk_count": 0,
  "low_risk_count": 2
}
```

### 4. Model Information

Get details about the currently loaded model.

**Endpoint**: `GET /model/info`

**Response**: `200 OK`

```json
{
  "model_name": "telco-churn-model",
  "version": "3",
  "stage": "Production",
  "created_at": 1705315800,
  "last_updated": 1705315800,
  "metrics": {
    "val_accuracy": 0.7985,
    "val_f1": 0.5612,
    "val_auc": 0.8432
  },
  "params": {
    "n_estimators": "200",
    "max_depth": "6",
    "learning_rate": "0.1"
  },
  "description": null
}
```

### 5. API Metrics

Get API usage statistics.

**Endpoint**: `GET /metrics`

**Response**: `200 OK`

```json
{
  "total_requests": 1523,
  "total_errors": 12,
  "error_rate": 0.0079,
  "avg_latency_seconds": 0.145,
  "total_predictions": 1511,
  "high_risk_predictions": 189,
  "high_risk_rate": 0.125
}
```

## Request Validation

All prediction endpoints validate input using Pydantic schemas.

### Required Fields

| Field | Type | Values |
|-------|------|--------|
| gender | string | "Male", "Female" |
| SeniorCitizen | integer | 0, 1 |
| Partner | string | "Yes", "No" |
| Dependents | string | "Yes", "No" |
| tenure | integer | 0-100 |
| PhoneService | string | "Yes", "No" |
| MultipleLines | string | "Yes", "No", "No phone service" |
| InternetService | string | "DSL", "Fiber optic", "No" |
| OnlineSecurity | string | "Yes", "No", "No internet service" |
| OnlineBackup | string | "Yes", "No", "No internet service" |
| DeviceProtection | string | "Yes", "No", "No internet service" |
| TechSupport | string | "Yes", "No", "No internet service" |
| StreamingTV | string | "Yes", "No", "No internet service" |
| StreamingMovies | string | "Yes", "No", "No internet service" |
| Contract | string | "Month-to-month", "One year", "Two year" |
| PaperlessBilling | string | "Yes", "No" |
| PaymentMethod | string | "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)" |
| MonthlyCharges | float | ≥ 0 |
| TotalCharges | float | ≥ 0 |

### Error Responses

**Validation Error**: `422 Unprocessable Entity`

```json
{
  "detail": [
    {
      "loc": ["body", "tenure"],
      "msg": "ensure this value is less than or equal to 100",
      "type": "value_error.number.not_le"
    }
  ]
}
```

**Model Not Loaded**: `503 Service Unavailable`

```json
{
  "detail": "Model not loaded"
}
```

**Internal Error**: `500 Internal Server Error`

```json
{
  "error": "Prediction failed",
  "detail": "Error details here",
  "timestamp": "2024-01-15T10:30:00"
}
```

## Rate Limiting

No rate limiting currently implemented. For production:
- Implement token bucket algorithm
- Limit: 100 requests/minute per IP
- Burst: 20 requests

## Examples

### Python

```python
import requests

# Single prediction
url = "http://localhost:8080/predict"
customer = {
    "gender": "Female",
    "SeniorCitizen": 0,
    # ... other fields
}

response = requests.post(url, json=customer)
result = response.json()

print(f"Churn Prediction: {result['prediction']}")
print(f"Probability: {result['churn_probability']:.2%}")
print(f"Risk Tier: {result['risk_tier']}")
```

### cURL

```bash
# Health check
curl -X GET http://localhost:8080/health

# Single prediction
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "MonthlyCharges": 65.6,
    "TotalCharges": 787.2,
    ...
  }'

# Model info
curl -X GET http://localhost:8080/model/info
```

### JavaScript (Fetch)

```javascript
// Single prediction
const customer = {
  gender: "Female",
  SeniorCitizen: 0,
  // ... other fields
};

fetch('http://localhost:8080/predict', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify(customer),
})
  .then(response => response.json())
  .then(data => console.log(data));
```

## Interactive Documentation

FastAPI provides automatic interactive documentation:

- **Swagger UI**: `http://localhost:8080/docs`
- **ReDoc**: `http://localhost:8080/redoc`

## Headers

### Response Headers

All responses include:
- `X-Correlation-ID`: Unique request identifier for tracing
- `X-Process-Time`: Request processing time in seconds
- `Content-Type`: application/json

## Monitoring

All requests are logged with:
- Correlation ID
- Request method and path
- Status code
- Latency
- Anonymized request data (no PII)

View logs:
```bash
# Docker
docker-compose logs -f api

# Cloud Run
gcloud logging read "resource.type=cloud_run_revision"
```
