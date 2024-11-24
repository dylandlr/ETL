# Azure ETL Infrastructure

## Overview
This project implements a comprehensive ETL infrastructure on Azure, featuring:
- Data Factory pipelines for orchestration
- Databricks for data processing
- Log Analytics for monitoring
- Azure Storage for data management
- Key Vault for secrets management

## Prerequisites
- Azure CLI
- Terraform >= 1.0.0
- PowerShell Core >= 7.0
- Azure Subscription with required permissions
- Git

## Project Structure
```
azure/
├── infrastructure/          # Terraform infrastructure code
│   ├── modules/            # Reusable Terraform modules
│   ├── environments/       # Environment-specific configurations
│   └── main.tf            # Main Terraform configuration
├── src/                    # Source code for ETL processes
├── tests/                  # Test files
├── docs/                   # Documentation
├── monitoring/             # Monitoring configurations
└── deploy.ps1             # Deployment script
```

## Setup Instructions

1. **Environment Setup**
   ```bash
   # Install required Python packages
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

2. **Azure Authentication**
   ```bash
   # Login to Azure
   az login
   ```

3. **Environment Configuration**
   - Copy `.env.example` to `.env`
   - Update the values in `.env` with your Azure configuration
   - Update environment-specific variables in `infrastructure/environments/{env}/terraform.tfvars`

4. **Infrastructure Deployment**
   ```powershell
   # Create deployment plan
   ./deploy.ps1 -Environment dev -Plan

   # Apply infrastructure changes
   ./deploy.ps1 -Environment dev -Apply

   # To destroy infrastructure (if needed)
   ./deploy.ps1 -Environment dev -Destroy
   ```

## Key Components

### Data Factory Pipelines
- Master ETL Pipeline
- Data Validation Pipeline
- Monitoring Integration Pipeline

### Databricks
- Data Transformation Notebooks
- Data Quality Checks
- Error Handling

### Monitoring
- Log Analytics Workspace
- Custom Metrics
- Alerts and Notifications

### Security
- Managed Identities
- Key Vault Integration
- Network Security

## Monitoring and Maintenance

### Logs and Metrics
- Access Log Analytics workspace for:
  - Pipeline execution metrics
  - Data quality metrics
  - Error logs
  - Performance metrics

### Alerts
- Configure alert thresholds in Azure Monitor
- Set up email notifications
- Monitor pipeline health

## Troubleshooting

### Common Issues
1. **Pipeline Failures**
   - Check activity logs in Data Factory
   - Review error messages in Log Analytics

2. **Permission Issues**
   - Verify managed identity permissions
   - Check Key Vault access policies

3. **Performance Issues**
   - Monitor Integration Runtime metrics
   - Check Databricks cluster configurations

## Contributing
1. Create a feature branch
2. Make changes
3. Run tests
4. Submit pull request

## License
Proprietary - All rights reserved