# Azure ETL Infrastructure Deployment Guide

## Prerequisites
- Azure CLI installed and configured
- Terraform >= 1.0.0 installed
- Azure subscription with required permissions
- Access to Azure DevOps or GitHub for source control

## Initial Setup

### 1. Azure Authentication
```bash
az login
az account set --subscription <subscription_id>
```

### 2. Backend Configuration
1. Create a backend.hcl file in the infrastructure directory with:
```hcl
resource_group_name  = "etl-rg-dev"
storage_account_name = "tfstate{random_id}"
container_name       = "tfstate"
key                 = "terraform.tfstate"
```

## Deployment Steps

### 1. Initialize Terraform
```bash
terraform init -backend-config=backend.hcl
```

### 2. Configure Variables
1. Copy `terraform.tfvars.example` to `terraform.tfvars`
2. Update the following variables:
   - project_name
   - environment
   - location
   - databricks_client_secret
   - tags

### 3. Plan Deployment
```bash
terraform plan -out=tfplan
```

### 4. Apply Infrastructure
```bash
terraform apply tfplan
```

## Environment Management

### Development Environment
- Default configuration in terraform.tfvars
- Resource naming pattern: `etl-*-dev`
- Network space: 10.0.0.0/16

### Production Environment (When Ready)
1. Create new terraform.tfvars for production
2. Update environment-specific variables
3. Follow same deployment steps with production backend

## Post-Deployment Steps

1. Configure Data Factory Pipelines
2. Set up Databricks workspace
3. Verify network connectivity
4. Test data lake access
5. Configure monitoring alerts

## Validation Checklist

- [ ] Resource group created
- [ ] Data Lake Storage provisioned
- [ ] Databricks workspace accessible
- [ ] Network security groups configured
- [ ] Key Vault secrets stored
- [ ] Logging enabled

## Troubleshooting

### Common Issues
1. Authentication failures
   - Verify Azure CLI login
   - Check service principal permissions

2. Network connectivity
   - Verify subnet configurations
   - Check NSG rules

3. Storage access
   - Verify managed identity assignments
   - Check storage firewall rules