# System Architecture

This document describes the architecture and design decisions for the Telco Churn MLOps Platform.

## Overview

The platform follows a microservices architecture with separation of concerns across training, inference, monitoring, and infrastructure layers.

## Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                          Data Layer                                   │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌─────────────┐    ┌──────────────┐    ┌──────────────┐          │
│  │   Kaggle    │───▶│ Data         │───▶│  Validators  │          │
│  │   Dataset   │    │ Downloader   │    │  (Schema,    │          │
│  └─────────────┘    └──────────────┘    │   Nulls)     │          │
│                                          └──────────────┘          │
│                              │                                       │
│                              ▼                                       │
│                     ┌──────────────────┐                            │
│                     │  GCS Storage     │                            │
│                     │  (Raw/Processed/ │                            │
│                     │   Gold layers)   │                            │
│                     └──────────────────┘                            │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│                      ML Training Pipeline                             │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Data → Preprocessing → Feature Engineering → Model Training         │
│           │                     │                      │              │
│           ▼                     ▼                      ▼              │
│     ┌──────────────┐    ┌────────────┐      ┌─────────────────┐    │
│     │  Cleaning    │    │  6 Domain  │      │  RandomForest   │    │
│     │  Encoding    │    │  Features: │      │  XGBoost        │    │
│     │  Scaling     │    │  - Tenure  │      │                 │    │
│     └──────────────┘    │  - Charges │      └─────────────────┘    │
│                         │  - Services│               │              │
│                         └────────────┘               ▼              │
│                                              ┌─────────────────┐    │
│                                              │  MLflow         │    │
│                                              │  - Experiments  │    │
│                                              │  - Registry     │    │
│                                              │  - Artifacts    │    │
│                                              └─────────────────┘    │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│                      Inference Layer                                  │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    FastAPI Application                       │   │
│  ├─────────────────────────────────────────────────────────────┤   │
│  │                                                              │   │
│  │  ┌──────────┐  ┌──────────┐  ┌───────────┐  ┌──────────┐  │   │
│  │  │ /health  │  │ /predict │  │ /predict/ │  │ /model/  │  │   │
│  │  │          │  │          │  │  batch    │  │  info    │  │   │
│  │  └──────────┘  └──────────┘  └───────────┘  └──────────┘  │   │
│  │                                                              │   │
│  │  ┌──────────────────────────────────────────────────────┐  │   │
│  │  │           Model Loader Singleton                     │  │   │
│  │  │  - Cache production model                            │  │   │
│  │  │  - Load from MLflow Registry                         │  │   │
│  │  │  - Preprocessing pipeline                            │  │   │
│  │  └──────────────────────────────────────────────────────┘  │   │
│  │                                                              │   │
│  │  ┌──────────────────────────────────────────────────────┐  │   │
│  │  │           Observability Middleware                   │  │   │
│  │  │  - Request/response logging                          │  │   │
│  │  │  - Latency measurement                               │  │   │
│  │  │  - Correlation IDs                                   │  │   │
│  │  └──────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                       │
│                              ▼                                       │
│                     ┌──────────────────┐                            │
│                     │   Cloud Run      │                            │
│                     │  (Auto-scaling)  │                            │
│                     └──────────────────┘                            │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│                      Monitoring Layer                                 │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌──────────────────┐    ┌──────────────────┐    ┌───────────────┐ │
│  │  Data Drift      │    │  Target Drift    │    │ Data Quality  │ │
│  │  Detection       │    │  Detection       │    │ Detection     │ │
│  │  (Evidently)     │    │  (Evidently)     │    │ (Evidently)   │ │
│  └──────────────────┘    └──────────────────┘    └───────────────┘ │
│           │                       │                      │           │
│           └───────────────────────┴──────────────────────┘           │
│                              │                                       │
│                              ▼                                       │
│                     ┌──────────────────┐                            │
│                     │  HTML Reports    │                            │
│                     │  (monitoring/)   │                            │
│                     └──────────────────┘                            │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│                    Infrastructure Layer (GCP)                         │
├──────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐ │
│  │ GCS Buckets │  │ Cloud Run   │  │ IAM         │  │ Logging    │ │
│  │ (Artifacts) │  │ (API)       │  │ (Service    │  │ (Cloud     │ │
│  │             │  │             │  │  Accounts)  │  │  Logging)  │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └────────────┘ │
│                                                                       │
│  Managed by: Terraform Modules                                       │
└──────────────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. Configuration Management

**Location**: `src/config/`

- **Singleton Pattern**: ConfigManager ensures single source of truth
- **Environment-specific**: DEV/PROD configs with YAML merging
- **Validation**: Schema validation at load time

### 2. Data Pipeline

**Location**: `src/data/`

#### Data Download (`download_data.py`)
- Kaggle API integration
- Credential validation
- Automatic retry logic

#### Preprocessing (`preprocessing.py`)
- **DataCleaner**: Handle missing values, type conversion
- **CategoricalEncoder**: Label encoding with unseen category handling
- **FeatureScaler**: StandardScaler for numerical features
- **Pipeline**: Scikit-learn compatible transformers

#### Validation (`validators.py`)
- Schema validation
- Null checks
- Range validation
- Categorical value validation

### 3. Feature Engineering

**Location**: `src/features/`

Six engineered features:
1. **Tenure Bins**: Customer lifetime categorization
2. **Charges Ratio**: TotalCharges/MonthlyCharges (loyalty proxy)
3. **Service Count**: Number of subscribed services
4. **Long-term Indicator**: Binary flag for long-term customers
5. **Monthly Charges Tier**: Price bucketing
6. **Contract Value**: Long-term contract indicator

### 4. Model Training

**Location**: `src/training/`

**Pipeline Steps**:
1. Load and validate data
2. Train/val/test split (70/15/15)
3. Fit preprocessing pipeline
4. Engineer features
5. Train models (RandomForest + XGBoost)
6. Log to MLflow (params, metrics, artifacts)
7. Register to Model Registry

**MLflow Integration**:
- Experiment tracking
- Model registry with stages (Staging/Production)
- Artifact storage (model + pipeline)

### 5. Inference

**Location**: `src/inference/`

#### Model Registry (`model_registry.py`)
- Load by stage (Production/Staging)
- Load by version
- Promote models between stages
- Metadata retrieval

#### Batch Prediction (`batch_predict.py`)
- Load production model
- Process features
- Generate predictions with probabilities
- Risk tier classification (low/medium/high)
- Save with metadata

### 6. API Serving

**Location**: `src/api/`

#### Endpoints
- `GET /health`: Health check with model status
- `POST /predict`: Single customer prediction
- `POST /predict/batch`: Bulk predictions (max 1000)
- `GET /model/info`: Model metadata
- `GET /metrics`: API usage statistics

#### Design Patterns
- **Singleton**: Model loader with caching
- **Middleware**: Custom timed routes for latency
- **Validation**: Pydantic schemas with field constraints
- **Observability**: Structured logging with correlation IDs

### 7. Monitoring

**Location**: `src/monitoring/`

**Evidently AI Integration**:
- Data drift detection
- Target drift detection
- Data quality reports
- Reference dataset comparison
- HTML dashboard generation

### 8. Testing

**Location**: `tests/`

**Coverage**:
- Unit tests (>70% coverage)
- Integration tests (API)
- Fixtures for sample data
- Mock models and pipelines

## Design Decisions

### Why MLflow?
- Industry standard for ML lifecycle
- Built-in model registry
- Artifact storage
- Easy integration with cloud platforms

### Why FastAPI?
- Automatic OpenAPI documentation
- Pydantic validation
- Async support
- High performance

### Why Terraform?
- Infrastructure as code
- Version control for infrastructure
- Multi-cloud support
- Reusable modules

### Why Evidently AI?
- Specialized for ML monitoring
- Comprehensive drift detection
- Visual reports
- Open source

## Security Considerations

### Production Checklist
- [ ] Remove `allUsers` access from Cloud Run
- [ ] Implement API authentication (OAuth2, API keys)
- [ ] Use Secret Manager for credentials
- [ ] Enable VPC for private networking
- [ ] Implement rate limiting
- [ ] Enable audit logging
- [ ] Use least privilege IAM roles
- [ ] Encrypt data at rest and in transit

## Scalability

### Current Configuration
- **API**: 0-10 instances (autoscaling)
- **Memory**: 2Gi per instance
- **CPU**: 2 cores per instance

### Scaling Strategies
1. **Vertical**: Increase memory/CPU per instance
2. **Horizontal**: Increase max instances
3. **Caching**: Add Redis for model caching
4. **CDN**: Use Cloud CDN for static content
5. **Load Balancing**: Cloud Load Balancer

## Monitoring & Observability

### Logs
- **Structured JSON** format
- **Correlation IDs** for request tracing
- **Log levels**: DEBUG, INFO, WARNING, ERROR
- **Rotation**: 10MB max per file, 5 backups

### Metrics
- Request count
- Error rate
- Latency (p50, p95, p99)
- Prediction distribution
- Model version

### Alerts (Future)
- High error rate (>5%)
- Slow requests (>2s)
- Data drift detected
- Model accuracy degradation

## Future Enhancements

1. **Real-time Streaming**: Kafka/Pub-Sub for real-time predictions
2. **A/B Testing**: Shadow deployments, traffic splitting
3. **Model Explainability**: SHAP values, feature importance API
4. **Automated Retraining**: Trigger on drift detection
5. **Multi-model Serving**: Ensemble predictions
6. **Feature Store**: Centralized feature management
7. **Dashboard**: Streamlit/Grafana for monitoring
8. **Authentication**: OAuth2/JWT for API security
