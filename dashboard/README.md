# ETL and ML Dashboard

This dashboard provides real-time monitoring of ETL pipelines and machine learning processes using Azure Monitor and MLflow metrics.

## Features

- ETL Pipeline Monitoring
  - Pipeline status distribution
  - Average duration by pipeline
  - Timeline of pipeline runs
  - Success/failure rates

- Machine Learning Metrics
  - Model accuracy over time
  - Prediction counts
  - Experiment status distribution
  - Training/inference performance

- Data Quality Metrics
  - Data volume tracking
  - Data loss monitoring
  - Quality metrics over time

## Setup

1. Install requirements:
```bash
pip install -r requirements.txt
```

2. Set up environment variables in `.env`:
```
AZURE_LOG_ANALYTICS_WORKSPACE_ID=your_workspace_id
AZURE_TENANT_ID=your_tenant_id
AZURE_CLIENT_ID=your_client_id
AZURE_CLIENT_SECRET=your_client_secret
```

3. Run the dashboard:
```bash
streamlit run app.py
```

## Usage

1. Access the dashboard at http://localhost:8501
2. Use the sidebar to adjust the date range for metrics
3. Click the "Refresh Dashboard" button to update metrics
4. Hover over charts for detailed information

## Metrics Tracked

### ETL Metrics
- Pipeline execution status
- Duration of pipeline runs
- Success/failure distribution
- Timeline of executions

### ML Metrics
- Model accuracy
- Prediction counts
- Experiment status
- Training performance

### Data Quality
- Initial vs cleaned data counts
- Data loss percentage
- Quality trends over time

## Troubleshooting

If you encounter issues:

1. Check Azure credentials and permissions
2. Verify MLflow tracking server is accessible
3. Ensure all required environment variables are set
4. Check Azure Monitor query permissions
