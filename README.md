# Telco Customer Churn MLOps Platform

A **production-ready MLOps platform** for predicting customer churn in the telecommunications industry. Built with best practices for ML model development, deployment, monitoring, and infrastructure automation.

[![CI/CD](https://github.com/yourusername/telco-churn-mlops-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/telco-churn-mlops-platform/actions)
[![Coverage](https://codecov.io/gh/yourusername/telco-churn-mlops-platform/branch/main/graph/badge.svg)](https://codecov.io/gh/yourusername/telco-churn-mlops-platform)

## 🎯 Features

### Core ML Pipeline
- **Automated data pipeline** with Kaggle API integration
- **Feature engineering** with 6 domain-specific features
- **Model training** with RandomForest and XGBoost
- **MLflow tracking** for experiments, metrics, and model registry
- **Data validation** with schema and quality checks

### Production API
- **FastAPI service** with Pydantic validation
- **5 REST endpoints**: health, predict, batch predict, model info, metrics
- **Observability**: request logging, latency tracking, correlation IDs
- **Model registry integration** with stage management (Staging/Production)

### Monitoring & Quality
- **Evidently AI** for drift detection (data, target, quality)
- **Comprehensive testing** with >70% code coverage
- **Production logging** with JSON formatting and rotation

### DevOps & Infrastructure
- **Docker containers** for API and training
- **Docker Compose** for local development with MLflow
- **GitHub Actions** CI/CD with automated testing, building, and deployment
- **Terraform modules** for GCP infrastructure (GCS, Cloud Run, IAM)

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Docker & Docker Compose
- Kaggle API credentials
- GCP account (for cloud deployment)

### 1. Clone & Setup

```bash
git clone https://github.com/yourusername/telco-churn-mlops-platform.git
cd telco-churn-mlops-platform

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your Kaggle credentials
```

### 3. Download Data

```bash
python src/data/download_data.py
```

### 4. Train Model

```bash
python src/training/train_model.py --model-type both
```

### 5. Start API (Local)

```bash
# Option 1: Python
uvicorn src.api.app:app --reload

# Option 2: Docker Compose
docker-compose up

# Access API docs: http://localhost:8080/docs
# Access MLflow: http://localhost:5000
```

## 📖 Documentation

- **[Architecture Guide](ARCHITECTURE.md)** - System design and components
- **[API Documentation](API.md)** - Endpoint reference
- **[Docker Guide](DOCKER.md)** - Container usage
- **[Terraform Guide](infra/terraform/README.md)** - Infrastructure deployment

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Data Pipeline                            │
├─────────────────────────────────────────────────────────────┤
│  Kaggle → Validation → Preprocessing → Feature Engineering │
│                           ↓                                  │
│                      MLflow Tracking                         │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   Model Training                             │
├─────────────────────────────────────────────────────────────┤
│  RandomForest / XGBoost → MLflow Registry → Staging/Prod   │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   Inference & Serving                        │
├─────────────────────────────────────────────────────────────┤
│  FastAPI → Model Loader → Predictions → Observability       │
│             ↓                                                │
│      Cloud Run (GCP)                                         │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    Monitoring                                │
├─────────────────────────────────────────────────────────────┤
│  Evidently → Drift Reports → Alerts                         │
└─────────────────────────────────────────────────────────────┘
```

## 🔧 Usage Examples

### Make Prediction

```python
import requests

url = "http://localhost:8080/predict"
customer_data = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "tenure": 12,
    "MonthlyCharges": 65.6,
    # ... other features
}

response = requests.post(url, json=customer_data)
print(response.json())
# {
#   "prediction": "No",
#   "churn_probability": 0.23,
#   "risk_tier": "low",
#   "model_version": "3"
# }
```

### Batch Predictions

```bash
python src/inference/batch_predict.py \
    --data-path data/gold/customers.csv \
    --model-stage Production
```

### Generate Drift Reports

```bash
python src/monitoring/drift_detection.py \
    --current-data data/predictions/churn_predictions_20240115.csv
```

## 🧪 Testing

```bash
# Run all tests
pytest

# With coverage
pytest --cov=src --cov-report=html

# Specific test module
pytest tests/test_api.py -v
```

## 🐳 Docker Deployment

```bash
# Build images
docker-compose build

# Start services
docker-compose up -d

# Run training
docker-compose --profile training up training

# View logs
docker-compose logs -f api
```

## ☁️ Cloud Deployment (GCP)

```bash
cd infra/terraform

# Initialize
terraform init

# Plan
terraform plan -var-file="terraform.tfvars"

# Deploy
terraform apply

# Get API URL
terraform output api_url
```

## 📊 Project Structure

```
telco-churn-mlops-platform/
├── src/
│   ├── api/              # FastAPI application
│   ├── config/           # Configuration management
│   ├── data/             # Data pipeline & validation
│   ├── features/         # Feature engineering
│   ├── inference/        # Model serving & batch prediction
│   ├── monitoring/       # Drift detection
│   ├── training/         # Model training
│   └── logging_utils/    # Structured logging
├── tests/                # Unit & integration tests
├── configs/              # YAML configurations
├── infra/
│   └── terraform/        # Infrastructure as code
├── .github/
│   └── workflows/        # CI/CD pipelines
├── Dockerfile.api        # API container
├── Dockerfile.training   # Training container
├── docker-compose.yml    # Local dev environment
└── requirements.txt      # Python dependencies
```

## 🔑 Key Technologies

- **ML**: scikit-learn, XGBoost, MLflow
- **API**: FastAPI, Pydantic, Uvicorn
- **Monitoring**: Evidently AI
- **Testing**: Pytest, coverage
- **Containerization**: Docker, Docker Compose
- **CI/CD**: GitHub Actions
- **Infrastructure**: Terraform, Google Cloud Platform
- **Data**: Pandas, NumPy

## 📈 Model Performance

| Model         | Accuracy | F1 Score | AUC   |
|---------------|----------|----------|-------|
| RandomForest  | 79.2%    | 55.1%    | 83.4% |
| XGBoost       | 79.9%    | 56.1%    | 84.3% |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- **Dataset**: [IBM Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **MLflow**: Experiment tracking and model registry
- **Evidently AI**: ML monitoring and drift detection
- **FastAPI**: Modern web framework for APIs

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**Built with ❤️ for production ML deployments**
