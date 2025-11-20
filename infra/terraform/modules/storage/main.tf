# GCS Storage Module - Creates artifact storage bucket

variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP region"
  type        = string
}

variable "bucket_name" {
  description = "GCS bucket name"
  type        = string
}

variable "environment" {
  description = "Environment name"
  type        = string
}

variable "force_destroy" {
  description = "Allow destroying bucket with objects (for non-prod)"
  type        = bool
  default     = false
}

# GCS Bucket for ML artifacts
resource "google_storage_bucket" "artifacts" {
  name          = var.bucket_name
  project       = var.project_id
  location      = var.region
  force_destroy = var.force_destroy
  
  uniform_bucket_level_access = true
  
  versioning {
    enabled = var.environment == "production"
  }
  
  lifecycle_rule {
    condition {
      age = 90
    }
    action {
      type = "Delete"
    }
  }
  
  lifecycle_rule {
    condition {
      age                = 30
      with_state         = "ARCHIVED"
    }
    action {
      type = "Delete"
    }
  }
  
  labels = {
    environment = var.environment
    purpose     = "mlops-artifacts"
    managed_by  = "terraform"
  }
}

# Output
output "bucket_name" {
  description = "Artifacts bucket name"
  value       = google_storage_bucket.artifacts.name
}

output "bucket_url" {
  description = "Artifacts bucket URL"
  value       = google_storage_bucket.artifacts.url
}
