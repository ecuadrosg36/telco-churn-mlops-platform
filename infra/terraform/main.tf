# Telco Churn MLOps Platform - GCP Infrastructure
# Main Terraform configuration

terraform {
  required_version = ">= 1.5"
  
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
  
  # Uncomment to use GCS backend for state
  # backend "gcs" {
  #   bucket = "your-terraform-state-bucket"
  #   prefix = "terraform/state"
  # }
}

# Provider configuration
provider "google" {
  project = var.project_id
  region  = var.region
}

# GCS Bucket for ML artifacts and data
module "storage" {
  source = "./modules/storage"
  
  project_id    = var.project_id
  region        = var.region
  bucket_name   = var.artifacts_bucket_name
  environment   = var.environment
  force_destroy = var.environment != "production"
}

# IAM Service Accounts and Roles
module "iam" {
  source = "./modules/iam"
  
  project_id  = var.project_id
  environment = var.environment
}

# Cloud Run Service for API
module "cloud_run" {
  source = "./modules/cloud_run"
  
  project_id         = var.project_id
  region             = var.region
  service_name       = var.api_service_name
  image              = var.api_image
  environment        = var.environment
  service_account    = module.iam.api_service_account_email
  
  # Resource limits
  memory_limit    = var.api_memory_limit
  cpu_limit       = var.api_cpu_limit
  min_instances   = var.api_min_instances
  max_instances   = var.api_max_instances
  
  # Environment variables
  env_vars = {
    ENVIRONMENT            = var.environment
    GCP_PROJECT_ID        = var.project_id
    GCP_REGION            = var.region
    MLFLOW_TRACKING_URI   = var.mlflow_tracking_uri
    ARTIFACTS_BUCKET      = module.storage.bucket_name
  }
  
  depends_on = [module.iam, module.storage]
}

# Outputs
output "api_url" {
  description = "Cloud Run service URL"
  value       = module.cloud_run.service_url
}

output "artifacts_bucket" {
  description = "GCS bucket for artifacts"
  value       = module.storage.bucket_name
}

output "service_account_email" {
  description = "API service account email"
  value       = module.iam.api_service_account_email
}