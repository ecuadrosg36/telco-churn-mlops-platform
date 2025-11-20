# Docker Commands Reference

This document provides common Docker commands for the Telco Churn MLOps Platform.

## Build Images

### Build API Image
```bash
docker build -f Dockerfile.api -t telco-churn-api:latest .
```

### Build Training Image
```bash
docker build -f Dockerfile.training -t telco-churn-training:latest .
```

### Build All Images with Docker Compose
```bash
docker-compose build
```

## Run Containers

### Start All Services (MLflow + API)
```bash
docker-compose up -d
```

### Start Specific Service
```bash
# MLflow only
docker-compose up -d mlflow

# API only
docker-compose up -d api
```

### Run Training Job
```bash
docker-compose --profile training up training
```

### View Logs
```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f api
docker-compose logs -f mlflow
```

## Development Mode

### Start with Hot Reload
```bash
docker-compose up
```

The API will automatically reload when code changes are detected.

### Access Services
- **API**: http://localhost:8080
- **API Docs**: http://localhost:8080/docs
- **MLflow UI**: http://localhost:5000

## Stop Services

### Stop All Services
```bash
docker-compose down
```

### Stop and Remove Volumes
```bash
docker-compose down -v
```

## Health Checks

### Check Container Health
```bash
docker ps
```

### Check API Health
```bash
curl http://localhost:8080/health
```

### Check MLflow Health
```bash
curl http://localhost:5000/health
```

## Run Standalone Containers

### Run API Container
```bash
docker run -d \
  --name telco-api \
  -p 8080:8080 \
  -v $(pwd)/artifacts:/app/artifacts \
  -v $(pwd)/data:/app/data \
  -e ENVIRONMENT=prod \
  telco-churn-api:latest
```

### Run Training Container
```bash
docker run \
  --name telco-training \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/mlruns:/app/mlruns \
  -v $(pwd)/artifacts:/app/artifacts \
  -e KAGGLE_USERNAME=your_username \
  -e KAGGLE_KEY=your_key \
  telco-churn-training:latest
```

## Debugging

### Execute Commands in Running Container
```bash
# API container
docker exec -it telco-api /bin/bash

# Training container
docker exec -it telco-training /bin/bash
```

### View Container Logs
```bash
docker logs telco-api
docker logs telco-training
```

### Inspect Container
```bash
docker inspect telco-api
```

## Cleanup

### Remove Stopped Containers
```bash
docker container prune
```

### Remove Unused Images
```bash
docker image prune -a
```

### Remove All Project Containers and Volumes
```bash
docker-compose down -v --remove-orphans
```

## Production Deployment

### Tag Images for Registry
```bash
docker tag telco-churn-api:latest gcr.io/your-project/telco-churn-api:1.0.0
docker tag telco-churn-training:latest gcr.io/your-project/telco-churn-training:1.0.0
```

### Push to Container Registry
```bash
docker push gcr.io/your-project/telco-churn-api:1.0.0
docker push gcr.io/your-project/telco-churn-training:1.0.0
```

## Environment Variables

Create a `.env` file in the project root:

```env
KAGGLE_USERNAME=your_username
KAGGLE_KEY=your_api_key
MLFLOW_TRACKING_URI=http://mlflow:5000
ENVIRONMENT=dev
```

Then use it with docker-compose:

```bash
docker-compose --env-file .env up
```

## Troubleshooting

### Port Already in Use
```bash
# Find process using port
lsof -i :8080  # macOS/Linux
netstat -ano | findstr :8080  # Windows

# Use different port
docker-compose up -e PORT=8081
```

### Container Won't Start
```bash
# Check logs
docker-compose logs api

# Rebuild without cache
docker-compose build --no-cache
```

### Volume Permission Issues
```bash
# Fix permissions
sudo chown -R $(whoami):$(whoami) data/ mlruns/ artifacts/
```
