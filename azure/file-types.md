# Infrastructure Files

## Terraform Files (.tf)
- backend.tf: Backend state configuration
  ```hcl
  terraform {
    backend "azurerm" {
      resource_group_name  = "terraform-state-rg"
      storage_account_name = "tfstate"
      container_name      = "state"
      key                 = "dev.terraform.tfstate"
    }
  }
  ```

- variables.tf: Variable definitions
  ```hcl
  variable "project_name" {
    type        = string
    description = "Name of the project"
  }

  variable "environment" {
    type        = string
    description = "Deployment environment"
    validation {
      condition     = contains(["dev", "staging", "prod"], var.environment)
      error_message = "Environment must be dev, staging, or prod."
    }
  }
  ```

## Pipeline Configuration Files

- azure-pipelines.yml: Azure DevOps pipeline
  ```yaml
  trigger:
    branches:
      include:
        - main
        - releases/*
  
  variables:
    - group: etl-variables
  ```

- .github/workflows/ci.yml: GitHub Actions
  ```yaml
  name: CI
  on:
    push:
      branches: [ main ]
    pull_request:
      branches: [ main ]
  ```

# Data Processing Files

## Databricks Notebooks (.py)
- transform.py: Data transformation logic
  ```python
  from pyspark.sql import SparkSession
  from pyspark.sql.functions import *

  def transform_data(input_path: str, output_path: str):
      spark = SparkSession.builder.getOrCreate()
      df = spark.read.parquet(input_path)
      # transformation logic
  ```

## SQL Files (.sql)
- create_tables.sql: Synapse table definitions
  ```sql
  CREATE TABLE [staging].[source_data]
  (
      id INT NOT NULL,
      data_date DATE NOT NULL,
      value DECIMAL(18,2)
  )
  WITH
  (
      DISTRIBUTION = HASH(id),
      CLUSTERED COLUMNSTORE INDEX
  );
  ```

## Data Factory Definitions (.json)
- pipeline.json: ADF pipeline definition
  ```json
  {
    "name": "MainETLPipeline",
    "properties": {
      "activities": [],
      "parameters": {}
    }
  }
  ```

# Configuration Files

## Python Requirements
- requirements.txt: Production dependencies
  ```txt
  apache-spark==3.2.1
  delta-spark==1.0.0
  great-expectations==0.13.44
  ```

- requirements-dev.txt: Development dependencies
  ```txt
  pytest==6.2.5
  black==22.3.0
  flake8==4.0.1
  ```

## Environment Configuration
- .env.template: Environment variable template
  ```env
  AZURE_SUBSCRIPTION_ID=
  AZURE_TENANT_ID=
  DATABRICKS_HOST=
  DATABRICKS_TOKEN=
  ```

# Documentation Files

## Markdown Documentation
- README.md: Project documentation
  ```markdown
  # ETL Pipeline Project
  
  ## Overview
  This project implements an ETL pipeline using Azure services...
  
  ## Setup Instructions
  1. Clone the repository
  2. Install dependencies...
  ```

# Monitoring Configuration

## Dashboard Definitions
- etl_dashboard.json: Azure Dashboard
  ```json
  {
    "properties": {
      "lenses": {
        "0": {
          "parts": {
            "0": {
              "position": {"x": 0, "y": 0},
              "metadata": {
                "inputs": [],
                "type": "Extension/HubsExtension/PartType/MonitorChartPart"
              }
            }
          }
        }
      }
    }
  }
  ```

## Alert Rules
- alert_rules.json: Monitoring alerts
  ```json
  {
    "location": "global",
    "tags": {},
    "properties": {
      "severity": 2,
      "windowSize": "PT5M",
      "evaluationFrequency": "PT1M",
      "criteria": {}
    }
  }
  ```