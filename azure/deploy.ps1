#!/usr/bin/env pwsh

param(
    [Parameter(Mandatory=$true)]
    [ValidateSet('dev', 'prod')]
    [string]$Environment,
    
    [Parameter(Mandatory=$false)]
    [switch]$Plan,
    
    [Parameter(Mandatory=$false)]
    [switch]$Apply,
    
    [Parameter(Mandatory=$false)]
    [switch]$Destroy
)

# Set error action preference
$ErrorActionPreference = "Stop"

# Configuration
$rootDir = $PSScriptRoot
$infraDir = Join-Path $rootDir "infrastructure"
$envDir = Join-Path $infraDir "environments" $Environment

# Ensure Azure CLI is logged in
Write-Host "Checking Azure CLI login status..."
$loginStatus = az account show 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "Not logged in to Azure CLI. Please login..."
    az login
}

# Initialize Terraform
Write-Host "Initializing Terraform..."
Push-Location $infraDir
try {
    terraform init -backend-config="environments/$Environment/backend.hcl"
    if ($LASTEXITCODE -ne 0) { throw "Terraform init failed" }

    # Select workspace
    terraform workspace select $Environment 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Creating new workspace: $Environment"
        terraform workspace new $Environment
    }

    # Load environment variables
    if (Test-Path (Join-Path $rootDir ".env")) {
        Get-Content (Join-Path $rootDir ".env") | ForEach-Object {
            if ($_ -match '^([^=]+)=(.*)$') {
                [Environment]::SetEnvironmentVariable($matches[1], $matches[2])
            }
        }
    }

    # Execute Terraform commands based on parameters
    if ($Plan) {
        Write-Host "Creating Terraform plan..."
        terraform plan -var-file="environments/$Environment/terraform.tfvars" -out="$Environment.tfplan"
    }
    
    if ($Apply) {
        if (Test-Path "$Environment.tfplan") {
            Write-Host "Applying Terraform plan..."
            terraform apply "$Environment.tfplan"
        } else {
            Write-Host "Applying Terraform configuration..."
            terraform apply -var-file="environments/$Environment/terraform.tfvars" -auto-approve
        }
    }
    
    if ($Destroy) {
        Write-Host "WARNING: This will destroy all resources in the $Environment environment!"
        $confirmation = Read-Host "Are you sure you want to proceed? (yes/no)"
        if ($confirmation -eq 'yes') {
            Write-Host "Destroying infrastructure..."
            terraform destroy -var-file="environments/$Environment/terraform.tfvars" -auto-approve
        } else {
            Write-Host "Destroy operation cancelled."
        }
    }
}
finally {
    Pop-Location
}

Write-Host "Deployment script completed successfully."
