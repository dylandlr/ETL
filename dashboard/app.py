import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import mlflow
from azure.monitor.query import LogsQueryClient
from azure.identity import DefaultAzureCredential
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
from data_manager import render_data_management_ui

# Load environment variables
load_dotenv()

# Configure page
st.set_page_config(page_title="ETL & ML Dashboard", layout="wide")

# Azure Monitor setup
credential = DefaultAzureCredential()
logs_client = LogsQueryClient(credential)
workspace_id = os.getenv("AZURE_LOG_ANALYTICS_WORKSPACE_ID")

def get_etl_metrics(days=7):
    """Get ETL pipeline metrics from Azure Monitor"""
    query = """
    AzureDiagnostics
    | where ResourceProvider == "MICROSOFT.DATAFACTORY"
    | where Category == "PipelineRuns"
    | where TimeGenerated > ago(7d)
    | project TimeGenerated, PipelineName, Status, Duration=todouble(Duration)
    """
    
    response = logs_client.query_workspace(
        workspace_id=workspace_id,
        query=query,
        timespan=timedelta(days=days)
    )
    
    records = []
    for table in response.tables:
        for row in table.rows:
            records.append(dict(zip(table.columns, row)))
    
    return pd.DataFrame(records)

def get_mlflow_metrics():
    """Get ML metrics from MLflow"""
    client = mlflow.tracking.MlflowClient()
    experiments = client.search_experiments()
    
    metrics_data = []
    for exp in experiments:
        runs = client.search_runs(exp.experiment_id)
        for run in runs:
            metrics_data.append({
                'experiment': exp.name,
                'run_id': run.info.run_id,
                'status': run.info.status,
                'start_time': datetime.fromtimestamp(run.info.start_time/1000.0),
                **run.data.metrics
            })
    
    return pd.DataFrame(metrics_data)

# Sidebar navigation
st.sidebar.title("ETL & ML Dashboard")
page = st.sidebar.radio("Navigate to", ["Data Management", "Pipeline Monitoring", "ML Metrics"])

if page == "Data Management":
    render_data_management_ui()

elif page == "Pipeline Monitoring":
    st.title("ETL Pipeline Monitoring")
    
    date_range = st.sidebar.slider(
        "Select Date Range (days)",
        min_value=1,
        max_value=30,
        value=7
    )

    # ETL Metrics
    st.header("ETL Pipeline Performance")
    try:
        etl_data = get_etl_metrics(date_range)
        
        # Pipeline Status Distribution
        col1, col2 = st.columns(2)
        
        with col1:
            status_counts = etl_data['Status'].value_counts()
            fig_status = px.pie(
                values=status_counts.values,
                names=status_counts.index,
                title="Pipeline Status Distribution"
            )
            st.plotly_chart(fig_status)
        
        with col2:
            # Average Duration by Pipeline
            avg_duration = etl_data.groupby('PipelineName')['Duration'].mean().reset_index()
            fig_duration = px.bar(
                avg_duration,
                x='PipelineName',
                y='Duration',
                title="Average Pipeline Duration (seconds)"
            )
            st.plotly_chart(fig_duration)
        
        # Timeline of Pipeline Runs
        fig_timeline = px.scatter(
            etl_data,
            x='TimeGenerated',
            y='PipelineName',
            color='Status',
            title="Pipeline Runs Timeline"
        )
        st.plotly_chart(fig_timeline, use_container_width=True)

    except Exception as e:
        st.error(f"Error fetching ETL metrics: {str(e)}")

else:  # ML Metrics page
    st.title("Machine Learning Metrics")
    
    try:
        ml_data = get_mlflow_metrics()
        
        # Model Performance Metrics
        col3, col4 = st.columns(2)
        
        with col3:
            if 'accuracy' in ml_data.columns:
                fig_accuracy = px.line(
                    ml_data,
                    x='start_time',
                    y='accuracy',
                    color='experiment',
                    title="Model Accuracy Over Time"
                )
                st.plotly_chart(fig_accuracy)
        
        with col4:
            if 'prediction_count' in ml_data.columns:
                fig_predictions = px.bar(
                    ml_data,
                    x='start_time',
                    y='prediction_count',
                    color='experiment',
                    title="Predictions Count Over Time"
                )
                st.plotly_chart(fig_predictions)
        
        # Experiment Status
        exp_status = ml_data['status'].value_counts()
        fig_exp_status = px.pie(
            values=exp_status.values,
            names=exp_status.index,
            title="ML Experiment Status Distribution"
        )
        st.plotly_chart(fig_exp_status)

        # Data Quality Metrics
        st.header("Data Quality Metrics")
        quality_metrics = ml_data[['start_time', 'initial_row_count', 'cleaned_row_count']].dropna()
        
        if not quality_metrics.empty:
            fig_quality = go.Figure()
            fig_quality.add_trace(go.Scatter(
                x=quality_metrics['start_time'],
                y=quality_metrics['initial_row_count'],
                name="Initial Row Count"
            ))
            fig_quality.add_trace(go.Scatter(
                x=quality_metrics['start_time'],
                y=quality_metrics['cleaned_row_count'],
                name="Cleaned Row Count"
            ))
            fig_quality.update_layout(title="Data Quality Over Time")
            st.plotly_chart(fig_quality)
            
            # Calculate data loss percentage
            quality_metrics['data_loss_pct'] = (
                (quality_metrics['initial_row_count'] - quality_metrics['cleaned_row_count']) /
                quality_metrics['initial_row_count'] * 100
            )
            
            fig_loss = px.box(
                quality_metrics,
                y='data_loss_pct',
                title="Data Loss Distribution (%)"
            )
            st.plotly_chart(fig_loss)

    except Exception as e:
        st.error(f"Error fetching ML metrics: {str(e)}")

# Refresh button
if st.sidebar.button("Refresh Dashboard"):
    st.experimental_rerun()
