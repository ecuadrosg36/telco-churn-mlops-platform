# Terraform Deployment Guide

This guide explains how to deploy the Telco Churn MLOps platform infrastructure using Terraform.

## Prerequisites

1. **Terraform installed** (>= 1.5)
   ```bash
   terraform --version
   ```

2. **GCP Project**
   - Create a GCP project
   - Enable required APIs:
     ```bash
     gcloud services enable cloudrun.googleapis.com
     gcloud services enable storage.googleapis.com
     gcloud services enable iam.googleapis.com
     ```

3. **Authentication**
   ```bash
   gcloud auth application-default login
   ```

## Quick Start

### 1. Configure Variables

Copy the example tfvars file:
```bash
cd infra/terraform
cp terraform.tfvars.example terraform.tfvars
```

Edit `terraform.tfvars` with your values:
```hcl
project_id            = "your-gcp-project-id"
region                = "us-central1"
environment           = "dev"
artifacts_bucket_name = "telco-churn-artifacts-dev-unique"
```

### 2. Initialize Terraform

```bash
terraform init
```

This downloads required providers and modules.

### 3. Plan Deployment

Preview changes:
```bash
terraform plan
```

Review the planned resources carefully.

### 4. Apply Configuration

Deploy infrastructure:
```bash
terraform apply
```

Type `yes` when prompted.

### 5. View Outputs

After deployment:
```bash
terraform output
```

Example output:
```
api_url                = "https://telco-churn-api-dev-xyz123.run.app"
artifacts_bucket       = "telco-churn-artifacts-dev"
service_account_email  = "telco-churn-api-dev@project.iam.gserviceaccount.com"
```

## Resources Created

Terraform creates the following GCP resources:

### Storage
- **GCS Bucket**: For ML artifacts, models, and data
  - Versioning: Enabled in production
  - Lifecycle: Auto-delete objects after 90 days
  - Uniform access: Enabled

### IAM
- **API Service Account**: Used by Cloud Run service
  - Permissions: Storage read, logging write
- **Training Service Account**: Used by training jobs
  - Permissions: Storage admin, logging write

### Cloud Run
- **API Service**: Containerized FastAPI application
  - Autoscaling: 0-10 instances (configurable)
  - Resources: 2 CPU, 2Gi memory (configurable)
  - Health checks: `/health` endpoint
  - Public access: Enabled (adjust for production)

## Environments

### Development
```bash
terraform workspace new dev
terraform apply -var-file="terraform.tfvars"
```

### Staging
```bash
terraform workspace new staging
terraform apply -var-file="terraform.staging.tfvars"
```

### Production
```bash
terraform workspace new production
terraform apply -var-file="terraform.production.tfvars"
```

## Update Deployment

### Update Cloud Run Image

1. Build and push new image:
   ```bash
   docker build -f Dockerfile.api -t gcr.io/PROJECT_ID/telco-churn-api:v1.1 .
   docker push gcr.io/PROJECT_ID/telco-churn-api:v1.1
   ```

2. Update `terraform.tfvars`:
   ```hcl
   api_image = "gcr.io/PROJECT_ID/telco-churn-api:v1.1"
   ```

3. Apply changes:
   ```bash
   terraform apply
   ```

### Scale Service

Update `terraform.tfvars`:
```hcl
api_min_instances = 2
api_max_instances = 20
```

Apply:
```bash
terraform apply
```

## Remote State (Recommended for Teams)

### Setup GCS Backend

1. Create state bucket:
   ```bash
   gsutil mb gs://your-terraform-state-bucket
   gsutil versioning set on gs://your-terraform-state-bucket
   ```

2. Uncomment backend in `main.tf`:
   ```hcl
   backend "gcs" {
     bucket = "your-terraform-state-bucket"
     prefix = "terraform/state"
   }
   ```

3. Reinitialize:
   ```bash
   terraform init -migrate-state
   ```

## Destroy Infrastructure

**Warning**: This deletes all resources!

```bash
terraform destroy
```

For non-production only (production buckets are protected):
```bash
terraform destroy -var="force_destroy=true"
```

## Troubleshooting

### API Already Exists
```bash
# Import existing resource
terraform import module.cloud_run.google_cloud_run_service.api projects/PROJECT_ID/locations/REGION/services/SERVICE_NAME
```

### Permission Denied
```bash
# Check authentication
gcloud auth application-default login

# Verify project
gcloud config get-value project
```

### State Lock
```bash
# If state is locked, force unlock (use carefully!)
terraform force-unlock LOCK_ID
```

## Monitoring

### View Logs
```bash
# Cloud Run logs
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=telco-churn-api-dev" --limit 50

# View in console
https://console.cloud.google.com/run?project=PROJECT_ID
```

### Check API Health
```bash
API_URL=$(terraform output -raw api_url)
curl $API_URL/health
```

## Cost Estimation

Use Google Cloud Pricing Calculator:
- Cloud Run: ~$0.10/day (low traffic)
- GCS Storage: ~$0.02/GB/month
- Estimated monthly cost: $5-20 (dev), $50-200 (production with traffic)

## Security Best Practices

1. **Use private networks** (VPC for production)
2. **Restrict IAM permissions** (principle of least privilege)
3. **Enable authentication** on Cloud Run (remove `allUsers` access)
4. **Use Secret Manager** for sensitive data
5. **Enable audit logging**
6. **Implement HTTPS only**

## Next Steps

1. Configure CI/CD to deploy via Terraform
2. Set up monitoring and alerting
3. Implement backup strategy
4. Configure domain and SSL
5. Set up VPC and private networking
