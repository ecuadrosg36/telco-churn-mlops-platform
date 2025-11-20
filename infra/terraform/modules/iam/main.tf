# IAM Module - Service accounts and roles

variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "environment" {
  description = "Environment name"
  type        = string
}

# Service Account for API
resource "google_service_account" "api_service_account" {
  account_id   = "telco-churn-api-${var.environment}"
  display_name = "Telco Churn API Service Account (${var.environment})"
  project      = var.project_id
}

# Service Account for Training
resource "google_service_account" "training_service_account" {
  account_id   = "telco-churn-training-${var.environment}"
  display_name = "Telco Churn Training Service Account (${var.environment})"
  project      = var.project_id
}

# IAM roles for API service account
resource "google_project_iam_member" "api_storage_viewer" {
  project = var.project_id
  role    = "roles/storage.objectViewer"
  member  = "serviceAccount:${google_service_account.api_service_account.email}"
}

resource "google_project_iam_member" "api_logging_writer" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.api_service_account.email}"
}

# IAM roles for training service account
resource "google_project_iam_member" "training_storage_admin" {
  project = var.project_id
  role    = "roles/storage.objectAdmin"
  member  = "serviceAccount:${google_service_account.training_service_account.email}"
}

resource "google_project_iam_member" "training_logging_writer" {
  project = var.project_id
  role    = "roles/logging.logWriter"
  member  = "serviceAccount:${google_service_account.training_service_account.email}"
}

# Outputs
output "api_service_account_email" {
  description = "API service account email"
  value       = google_service_account.api_service_account.email
}

output "training_service_account_email" {
  description = "Training service account email"
  value       = google_service_account.training_service_account.email
}
