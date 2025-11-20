# Input variables for Terraform configuration

variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP region for resources"
  type        = string
  default     = "us-central1"
}

variable "environment" {
  description = "Environment name (dev, staging, production)"
  type        = string
  default     = "dev"
  
  validation {
    condition     = contains(["dev", "staging", "production"], var.environment)
    error_message = "Environment must be one of: dev, staging, production."
  }
}

# Storage variables
variable "artifacts_bucket_name" {
  description = "Name for GCS artifacts bucket"
  type        = string
}

# Cloud Run API variables
variable "api_service_name" {
  description = "Name for Cloud Run API service"
  type        = string
  default     = "telco-churn-api"
}

variable "api_image" {
  description = "Docker image for API service"
  type        = string
  default     = "gcr.io/PROJECT_ID/telco-churn-api:latest"
}

variable "api_memory_limit" {
  description = "Memory limit for API service"
  type        = string
  default     = "2Gi"
}

variable "api_cpu_limit" {
  description = "CPU limit for API service"
  type        = string
  default     = "2"
}

variable "api_min_instances" {
  description = "Minimum number of API instances"
  type        = number
  default     = 1
}

variable "api_max_instances" {
  description = "Maximum number of API instances"
  type        = number
  default     = 10
}

# MLflow variables
variable "mlflow_tracking_uri" {
  description = "MLflow tracking URI"
  type        = string
  default     = ""
}

# Tags
variable "labels" {
  description = "Labels to apply to all resources"
  type        = map(string)
  default = {
    project     = "telco-churn"
    managed_by  = "terraform"
  }
}
