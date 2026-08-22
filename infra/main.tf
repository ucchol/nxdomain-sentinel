# Cloud Run service sized to stay inside the always-free tier.
#
# The free allowance is 2M requests, 180,000 vCPU-seconds and 360,000 GiB-seconds
# per month. At 512 MiB and roughly 30 ms per request, 2M requests costs about
# 60,000 vCPU-seconds and 30,000 GiB-seconds, so the binding constraint is the
# request count rather than compute.
#
# min_instance_count stays at 0. Pinning one warm instance would cost roughly
# $65/month and blow the budget entirely; the scheduled ping in
# .github/workflows/keepwarm.yml handles cold starts instead, for free.

terraform {
  required_version = ">= 1.5"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

variable "project_id" { type = string }
variable "region" {
  type    = string
  default = "us-central1"
}
variable "service_name" {
  type    = string
  default = "nxdomain-sentinel"
}
variable "image" {
  type        = string
  description = "Fully qualified Artifact Registry image, set by CI"
}
variable "github_repo" {
  type        = string
  description = "owner/repo, used to scope the deploy identity"
}

resource "google_project_service" "required" {
  for_each = toset([
    "run.googleapis.com",
    "artifactregistry.googleapis.com",
    "iamcredentials.googleapis.com",
  ])
  service            = each.key
  disable_on_destroy = false
}

resource "google_artifact_registry_repository" "images" {
  location      = var.region
  repository_id = var.service_name
  format        = "DOCKER"
  depends_on    = [google_project_service.required]

  # Free tier includes 0.5 GB of storage. Without cleanup, every CI build
  # accumulates and the repository silently starts billing.
  cleanup_policies {
    id     = "keep-recent"
    action = "KEEP"
    most_recent_versions { keep_count = 3 }
  }
  cleanup_policies {
    id     = "drop-old"
    action = "DELETE"
    condition { older_than = "604800s" }
  }
}

resource "google_service_account" "runtime" {
  account_id   = "${var.service_name}-run"
  display_name = "nxdomain-sentinel runtime"
}

resource "google_cloud_run_v2_service" "svc" {
  name                = var.service_name
  location            = var.region
  deletion_protection = false
  ingress             = "INGRESS_TRAFFIC_ALL"

  template {
    service_account = google_service_account.runtime.email
    # Generous concurrency because inference is short and CPU-bound in bursts.
    # Serving more requests per instance directly reduces billed instance time.
    max_instance_request_concurrency = 80
    timeout                          = "30s"

    scaling {
      min_instance_count = 0
      max_instance_count = 4
    }

    containers {
      image = var.image
      ports { container_port = 8080 }

      resources {
        limits = {
          cpu    = "1"
          memory = "512Mi"
        }
        # CPU only during request processing. This is what makes a warm idle
        # instance free rather than billed.
        cpu_idle          = true
        startup_cpu_boost = true
      }

      startup_probe {
        http_get { path = "/healthz" }
        initial_delay_seconds = 2
        period_seconds        = 3
        failure_threshold     = 10
      }
      liveness_probe {
        http_get { path = "/healthz" }
        period_seconds = 30
      }
    }
  }
  depends_on = [google_project_service.required]
}

resource "google_cloud_run_v2_service_iam_member" "public" {
  name     = google_cloud_run_v2_service.svc.name
  location = google_cloud_run_v2_service.svc.location
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# Budget tripwire. The free tier should never be exceeded, so any spend at all
# means something is wrong and should page the owner immediately.
resource "google_billing_budget" "tripwire" {
  count           = var.billing_account == "" ? 0 : 1
  billing_account = var.billing_account
  display_name    = "${var.service_name} tripwire"

  budget_filter { projects = ["projects/${var.project_id}"] }
  amount {
    specified_amount {
      currency_code = "USD"
      units         = "1"
    }
  }
  threshold_rules { threshold_percent = 1.0 }
}

variable "billing_account" {
  type    = string
  default = ""
}

output "service_url" { value = google_cloud_run_v2_service.svc.uri }
output "runtime_service_account" { value = google_service_account.runtime.email }
