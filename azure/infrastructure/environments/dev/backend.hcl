# File: infrastructure/environments/dev/backend.hcl
# Purpose: Development environment backend configuration
resource_group_name  = "etl-project-dev-rg"
storage_account_name = "tfstateu1mvd8"
container_name       = "tfstate"
key                 = "dev/terraform.tfstate"