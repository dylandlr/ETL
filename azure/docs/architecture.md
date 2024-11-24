# Azure ETL Infrastructure Architecture

## Overview
This document outlines the architecture of our Azure-based ETL (Extract, Transform, Load) infrastructure, designed for robust data processing and storage across multiple environments.

## Infrastructure Components

### 1. Azure Data Factory (etl-project-dev-adf)
- **Purpose**: Orchestration of data integration and transformation workflows
- **Location**: East US 2
- **Key Components**:
  - System-assigned managed identity for secure authentication
  - Azure Integration Runtime (ir-azure-dev) for data movement
  - Data Lake Linked Service (ls_adls_dev)

### 2. Data Lake Storage Account (datalakeuys5ea)
- **Type**: StorageV2 with hierarchical namespace
- **Replication**: Locally-redundant storage (LRS)
- **Performance Tier**: Standard
- **Data Containers**:
  - raw: For incoming raw data
  - staged: For intermediate processing
  - processed: For transformed data
  - curated: For business-ready data
- **Access Control**: Private container access

### 3. Terraform State Storage Account (tfstateuys5ea)
- **Type**: StorageV2
- **Replication**: Locally-redundant storage (LRS)
- **Performance Tier**: Standard
- **Purpose**: Maintains Terraform state files

## Security Configuration

### Authentication & Authorization
- System-assigned managed identity for Data Factory
- Data Factory has "Storage Blob Data Contributor" role on Data Lake
- Shared access key authentication enabled

### Network Security
- HTTPS-only traffic enforced
- TLS 1.2 minimum version
- Public network access currently enabled
- Pending implementation:
  - Network security groups
  - Private endpoints
  - Azure Private Link

## Infrastructure Management

### Terraform Configuration
- **Provider**: Azure (hashicorp/azurerm ~> 3.85.0)
- **Terraform Version**: >= 1.0.0
- **Backend**: Azure Storage
- **Authentication**: CLI-based
- **Resource Naming**: Randomized string generation for uniqueness

### Environment Management
- Current focus on development (dev) environment
- Modular configuration for easy environment replication
- Environment-specific variable management

## Data Flow
```
[External Sources] → [Raw Container] → [Staged Container] → [Processed Container] → [Curated Container]
                          ↑                    ↑                      ↑                    ↑
                          └────────────────────┴──────────────────────┴────────────────────┘
                                            Azure Data Factory
```

## Future Enhancements
1. Implementation of data pipeline configurations
2. Enhanced security measures:
   - Network rules configuration
   - Private endpoints setup
   - Network security groups
3. Expansion to staging/production environments
4. Comprehensive logging and monitoring
5. Implementation of Azure Private Link

## Known Limitations
- Public network access enabled
- Shared access keys in use
- Limited network restrictions
- Basic error handling

## Best Practices Implemented
- Infrastructure as Code (IaC) approach
- Modular and reusable configurations
- Consistent resource tagging
- Secure authentication methods
- Environment isolation