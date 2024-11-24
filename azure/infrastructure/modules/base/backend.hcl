# File: infrastructure/environments/dev/backend.hcl
# Purpose: Development environment backend configuration

# Azure Storage Backend Configuration
workspaces {
  prefix = "etl-"
}

storage_account_name = "__STORAGE_ACCOUNT_NAME__"  # To be replaced during deployment
container_name      = "tfstate"
key                = "base.tfstate"
resource_group_name = "__RESOURCE_GROUP_NAME__"    # To be replaced during deployment

subscription_id    = "__SUBSCRIPTION_ID__"         # To be replaced during deployment
tenant_id         = "__TENANT_ID__"               # To be replaced during deployment

# Security settings
use_azuread_auth     = true
use_msi              = true                       # Use Managed Service Identity when available