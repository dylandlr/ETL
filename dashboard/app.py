import os
import json
import subprocess
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import mlflow
from mlflow.tracking import MlflowClient
from datetime import datetime
import time
import sys

# Get project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# Import DataLakeManager
from dashboard.data_manager import DataLakeManager

# Set MLflow tracking URI
MLFLOW_TRACKING_URI = "file:" + os.path.join(PROJECT_ROOT, "mlruns")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def fetch_data():
    """Fetch new data into the data lake"""
    try:
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Fetch news data
        status_text.info("🔄 Fetching fresh data into data lake...")
        progress_bar.progress(25)
        result = subprocess.run(
            ['python', os.path.join(PROJECT_ROOT, 'scripts', 'fetch_news_data.py')],
            check=True,
            capture_output=True,
            text=True
        )
        progress_bar.progress(100)
        status_text.success("✅ Data fetched successfully!")
        st.code(result.stdout, language='text')
        return True
    except Exception as e:
        st.error(f"❌ Error fetching data: {str(e)}")
        return False

def run_pipeline():
    """Run the ETL pipeline (without data fetching)"""
    try:
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Train model
        status_text.info("🤖 Training model...")
        progress_bar.progress(50)
        result = subprocess.run(
            ['python', os.path.join(PROJECT_ROOT, 'run_pipeline.py')],
            check=True,
            capture_output=True,
            text=True
        )
        progress_bar.progress(100)
        status_text.success("✅ Pipeline completed successfully!")
        st.code(result.stdout, language='text')
        return True
    except Exception as e:
        st.error(f"❌ Error running pipeline: {str(e)}")
        return False

def plot_confusion_matrix(confusion_matrix_data):
    """Create confusion matrix heatmap"""
    if confusion_matrix_data is None:
        return None
        
    # Convert to numpy array
    matrix = np.array(confusion_matrix_data['matrix'])
    labels = confusion_matrix_data['labels']
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=labels,
        y=labels,
        colorscale='Viridis',
        text=matrix,
        texttemplate="%{text}",
        textfont={"size": 16},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title='Confusion Matrix',
        xaxis_title='Predicted',
        yaxis_title='Actual',
        width=500,
        height=500
    )
    
    return fig

def plot_feature_importance(feature_importance_df):
    """Create feature importance bar chart"""
    if feature_importance_df is None:
        return None
        
    # Get top 10 features
    top_features = feature_importance_df.head(10)
    
    fig = px.bar(
        top_features,
        x='importance',
        y='feature',
        orientation='h',
        title='Top 10 Most Important Features'
    )
    
    fig.update_layout(
        xaxis_title='Importance',
        yaxis_title='Feature',
        height=400
    )
    
    return fig

def plot_metrics_history(runs_df):
    """Plot metrics history"""
    metrics = ['accuracy', 'f1', 'recall', 'cv_mean']
    
    fig = go.Figure()
    for metric in metrics:
        fig.add_trace(go.Scatter(
            x=runs_df['start_time'],
            y=runs_df[metric],
            name=metric.replace('_', ' ').title()
        ))
    
    fig.update_layout(
        title='Model Metrics History',
        xaxis_title='Timestamp',
        yaxis_title='Score',
        height=400
    )
    
    return fig

def plot_data_size_history(runs_df):
    """Plot data size history"""
    # Ensure numeric types for size columns
    size_columns = ['training_size', 'test_size', 'total_size']
    for col in size_columns:
        if col in runs_df.columns:
            runs_df[col] = pd.to_numeric(runs_df[col], errors='coerce')
    
    # Create a melted dataframe for plotting
    plot_df = runs_df[['start_time'] + size_columns].melt(
        id_vars=['start_time'],
        value_vars=size_columns,
        var_name='Dataset Type',
        value_name='Number of Samples'
    )
    
    fig = px.line(
        plot_df,
        x='start_time',
        y='Number of Samples',
        color='Dataset Type',
        title='Dataset Size History',
        labels={'start_time': 'Timestamp'}
    )
    
    fig.update_layout(
        xaxis_title='Timestamp',
        yaxis_title='Number of Samples',
        height=400,
        showlegend=True
    )
    
    return fig

def plot_topic_trends(df):
    """Plot trending topics over time"""
    # Process topics from titles using TF-IDF
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer(max_features=10)
    tfidf_matrix = vectorizer.fit_transform(df['processed_title'])
    
    # Convert to datetime and group by date
    df['date'] = pd.to_datetime(df['published_at'])
    daily_topics = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=vectorizer.get_feature_names_out(),
        index=df['date']
    ).groupby(pd.Grouper(freq='D')).mean()
    
    # Create line plot
    fig = go.Figure()
    for topic in daily_topics.columns:
        fig.add_trace(go.Scatter(
            x=daily_topics.index,
            y=daily_topics[topic],
            name=topic,
            mode='lines+markers'
        ))
    
    fig.update_layout(
        title='Topic Trends Over Time',
        xaxis_title='Date',
        yaxis_title='Topic Importance',
        height=500
    )
    return fig

def plot_sentiment_timeline(df):
    """Plot sentiment distribution over time"""
    df['date'] = pd.to_datetime(df['published_at'])
    daily_sentiment = df.groupby(pd.Grouper(key='date', freq='D')).agg({
        'polarity': ['mean', 'std'],
        'subjectivity': ['mean', 'std']
    }).reset_index()
    
    fig = go.Figure()
    
    # Add polarity line
    fig.add_trace(go.Scatter(
        x=daily_sentiment['date'],
        y=daily_sentiment['polarity']['mean'],
        name='Polarity',
        line=dict(color='blue'),
        mode='lines+markers'
    ))
    
    # Add subjectivity line
    fig.add_trace(go.Scatter(
        x=daily_sentiment['date'],
        y=daily_sentiment['subjectivity']['mean'],
        name='Subjectivity',
        line=dict(color='red'),
        mode='lines+markers'
    ))
    
    fig.update_layout(
        title='Sentiment Timeline',
        xaxis_title='Date',
        yaxis_title='Score',
        height=400
    )
    return fig

def plot_category_evolution(df):
    """Plot category distribution evolution"""
    df['date'] = pd.to_datetime(df['published_at'])
    category_dist = df.groupby([pd.Grouper(key='date', freq='D'), 'category']).size().unstack(fill_value=0)
    category_dist = category_dist.div(category_dist.sum(axis=1), axis=0) * 100
    
    fig = go.Figure()
    for category in category_dist.columns:
        fig.add_trace(go.Scatter(
            x=category_dist.index,
            y=category_dist[category],
            name=category,
            stackgroup='one',
            groupnorm='percent'
        ))
    
    fig.update_layout(
        title='Category Distribution Evolution',
        xaxis_title='Date',
        yaxis_title='Percentage',
        height=400
    )
    return fig

def get_mlflow_runs():
    """Get all MLflow runs and their metrics"""
    client = MlflowClient()
    
    # Get all runs from all experiments
    experiments = client.search_experiments()
    all_runs = []
    
    for experiment in experiments:
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"]
        )
        all_runs.extend(runs)
    
    # Sort all runs by start time
    all_runs.sort(key=lambda x: x.info.start_time, reverse=True)
    
    # Convert to DataFrame
    runs_data = []
    for run in all_runs:
        try:
            run_data = {
                'run_id': run.info.run_id,
                'start_time': pd.to_datetime(run.info.start_time, unit='ms'),
                'status': run.info.status,
                'experiment_id': run.info.experiment_id,
                **run.data.metrics,
                **run.data.params
            }
            
            # Load artifacts
            try:
                local_dir = client.download_artifacts(run.info.run_id, "")
                
                # Load confusion matrix
                confusion_matrix_path = os.path.join(local_dir, "confusion_matrix.json")
                if os.path.exists(confusion_matrix_path):
                    with open(confusion_matrix_path, 'r') as f:
                        run_data['confusion_matrix'] = json.load(f)
                
                # Load classification report
                report_path = os.path.join(local_dir, "classification_report.json")
                if os.path.exists(report_path):
                    with open(report_path, 'r') as f:
                        run_data['classification_report'] = json.load(f)
                
                # Load feature importance
                feature_importance_path = os.path.join(local_dir, "feature_importance.json")
                if os.path.exists(feature_importance_path):
                    with open(feature_importance_path, 'r') as f:
                        run_data['feature_importance'] = pd.DataFrame(json.load(f))
            except Exception as e:
                st.warning(f"Error loading artifacts for run {run.info.run_id}: {str(e)}")
            
            runs_data.append(run_data)
        except Exception as e:
            st.error(f"Error processing run {run.info.run_id}: {str(e)}")
            continue
    
    return pd.DataFrame(runs_data)

def display_sentiment_analysis():
    """Display sentiment analysis with visualizations"""
    # Add data fetching buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("🔄 Fetch New Data"):
            fetch_data()
    with col2:
        if st.button("▶️ Run Pipeline"):
            run_pipeline()
    with col3:
        if st.button("📊 Refresh Visualizations"):
            st.rerun()
    
    try:
        # Initialize DataLakeManager
        data_manager = DataLakeManager()
        container_name = os.getenv('ETL_DATA_LAKE_STORAGE_CONTAINER_NAME', 'raw')
        
        # Load the latest data from Data Lake
        df = data_manager.read_csv_from_datalake(container_name, "news_data/latest/raw_news_data.csv")
        
        if df is None:
            st.warning("No data available in Data Lake. Please fetch new data.")
            return
            
        df['date'] = pd.to_datetime(df['published_at'])
        
        # Display key metrics
        st.subheader("📈 Key Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        # Get historical data size
        try:
            history_files = data_manager.list_files(container_name, "news_data/history")
            total_articles = len(df)
            if history_files:
                st.metric("Total Articles (All Time)", total_articles)
            else:
                st.metric("Total Articles", total_articles)
        except Exception as e:
            st.metric("Total Articles", len(df))
        
        with col2:
            st.metric("Average Sentiment", f"{df['polarity'].mean():.2f}")
        with col3:
            st.metric("Categories", df['category'].nunique())
        with col4:
            st.metric("Date Range", f"{df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}")
        
        # Display data growth
        st.subheader("📈 Data Growth")
        try:
            history_files = data_manager.list_files(container_name, "news_data/history")
            if history_files:
                history_files = sorted(history_files, key=lambda x: x['last_modified'])
                dates = [datetime.fromtimestamp(f['last_modified']).strftime('%Y-%m-%d %H:%M') for f in history_files]
                sizes = [f['size'] for f in history_files]
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=dates, y=sizes, mode='lines+markers', name='Data Size'))
                fig.update_layout(
                    title='Data Growth Over Time',
                    xaxis_title='Date',
                    yaxis_title='File Size (bytes)',
                    showlegend=True
                )
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not load data growth history: {str(e)}")
        
        # Display visualizations
        st.subheader("📊 Topic Trends")
        st.plotly_chart(plot_topic_trends(df), use_container_width=True, key="topic_trends")
        
        st.subheader("😊 Sentiment Timeline")
        st.plotly_chart(plot_sentiment_timeline(df), use_container_width=True, key="sentiment_timeline")
        
        st.subheader("📚 Category Evolution")
        st.plotly_chart(plot_category_evolution(df), use_container_width=True, key="category_evolution")
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")

def display_news_prediction():
    """Display news prediction section"""
    st.write("### 🔮 News Prediction")
    
    # Add control buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Fetch New Data", key="fetch_prediction"):
            fetch_data()
    
    with col2:
        if st.button("▶️ Run Pipeline", key="run_prediction"):
            try:
                if run_pipeline():
                    st.success("✅ Pipeline completed successfully!")
                else:
                    st.error("❌ Pipeline failed to complete")
            except Exception as e:
                st.error(f"❌ Error running prediction: {str(e)}")
    
    with col3:
        if st.button("🔄 Refresh Results", key="refresh_prediction"):
            st.rerun()

    # Add prediction settings
    with st.expander("⚙️ Prediction Settings"):
        col1, col2 = st.columns(2)
        with col1:
            prediction_horizon = st.selectbox(
                "Prediction Horizon",
                ["1 day", "1 week", "1 month"],
                index=1
            )
        with col2:
            prediction_features = st.multiselect(
                "Features to Consider",
                ["Topic Trends", "Sentiment Trends", "Category Distribution", "Source Distribution"],
                default=["Topic Trends", "Sentiment Trends"]
            )
    
    # Display prediction results
    if os.path.exists('analysis/predictions.json'):
        with open('analysis/predictions.json', 'r') as f:
            predictions = json.load(f)
            
        # Display trend predictions
        st.subheader("📈 Predicted News Trends")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🔝 Rising Topics")
            for topic in predictions.get('rising_topics', []):
                st.info(f"📈 {topic['name']}: {topic['growth']}% growth")
        
        with col2:
            st.markdown("### 📉 Declining Topics")
            for topic in predictions.get('declining_topics', []):
                st.warning(f"📉 {topic['name']}: {topic['decline']}% decline")
        
        # Display category predictions
        st.subheader("📊 Category Distribution Forecast")
        if 'category_forecast' in predictions:
            fig = px.pie(
                values=list(predictions['category_forecast'].values()),
                names=list(predictions['category_forecast'].keys()),
                title="Predicted Category Distribution"
            )
            st.plotly_chart(fig)
        
        # Display confidence metrics
        st.subheader("🎯 Model Performance")
        metrics = predictions.get('metrics', {})
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Prediction Accuracy", f"{metrics.get('accuracy', 0):.2f}%")
        with col2:
            st.metric("Mean Absolute Error", f"{metrics.get('mae', 0):.2f}")
        with col3:
            st.metric("R² Score", f"{metrics.get('r2', 0):.2f}")
    else:
        st.info("🔍 No prediction results available. Run the prediction pipeline to see results!")

def display_mlflow_results():
    try:
        # Initialize MLflow
        mlflow.set_tracking_uri("file:./mlruns")
        
        # Get all runs
        runs_df = get_mlflow_runs()
        
        if runs_df.empty:
            st.warning("No model runs available. Please run the prediction pipeline first.")
            return
        
        # Add run selector
        run_ids = runs_df['run_id'].tolist()
        run_times = runs_df['start_time'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist()
        run_options = [f"{time} (ID: {id})" for time, id in zip(run_times, run_ids)]
        
        selected_run = st.selectbox(
            "Select Run",
            options=run_options,
            index=0,
            key="model_performance_run_selector"
        )
        
        # Get selected run index
        selected_run_id = selected_run.split("(ID: ")[1].rstrip(")")
        selected_run_data = runs_df[runs_df['run_id'] == selected_run_id].iloc[0]
        
        # Display run information
        st.header("Run Information")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Accuracy", f"{selected_run_data['accuracy']:.3f}")
            st.metric("F1 Score", f"{selected_run_data['f1']:.3f}")
        
        with col2:
            st.metric("Precision", f"{selected_run_data['precision']:.3f}")
            st.metric("Recall", f"{selected_run_data['recall']:.3f}")
        
        # Display confusion matrix
        st.header("Confusion Matrix")
        confusion_matrix = selected_run_data.get('confusion_matrix')
        if confusion_matrix is not None:
            st.plotly_chart(plot_confusion_matrix(confusion_matrix), key="confusion_matrix")
        else:
            st.warning("No confusion matrix available for this run")
        
        # Display feature importance
        st.header("Feature Importance")
        feature_importance = selected_run_data.get('feature_importance')
        if feature_importance is not None:
            st.plotly_chart(plot_feature_importance(feature_importance), key="feature_importance")
        else:
            st.warning("No feature importance data available for this run")
            
        # Display run parameters
        st.header("Run Parameters")
        params = selected_run_data.get('params', {})
        if params:
            st.json(params)
        else:
            st.info("No parameters recorded for this run")
            
    except Exception as e:
        st.error(f"Error displaying MLflow results: {str(e)}")

def main():
    st.set_page_config(page_title="News Trend Prediction Dashboard", layout="wide")
    
    st.title("News Trend Prediction Dashboard")
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Data Analysis", "Model Performance", "History"])
    
    with tab1:
        st.header("News Data Analysis")
        display_sentiment_analysis()
        display_news_prediction()
    
    with tab2:
        st.header("Model Performance")
        display_mlflow_results()
    
    with tab3:
        st.header("Model History")
        try:
            runs_df = get_mlflow_runs()
            
            if runs_df.empty:
                st.warning("No model history available. Please run the prediction pipeline first.")
                return
            
            # Display metrics history
            st.subheader("Metrics History")
            metrics_plot = plot_metrics_history(runs_df)
            if metrics_plot is not None:
                st.plotly_chart(metrics_plot, key="metrics_history")
            
            # Display dataset size history
            st.subheader("Dataset Size History")
            size_plot = plot_data_size_history(runs_df)
            if size_plot is not None:
                st.plotly_chart(size_plot, key="size_history")
            
            # Display all runs in a table
            st.subheader("All Runs")
            display_cols = ['run_id', 'start_time', 'accuracy', 'f1', 'precision', 'recall', 'data_size']
            st.dataframe(runs_df[display_cols].sort_values('start_time', ascending=False))
        except Exception as e:
            st.error(f"Error displaying model history: {str(e)}")
    
    add_sidebar()

def add_sidebar():
    """Add sidebar with data lake info"""
    # Add data lake info
    st.sidebar.title("Data Lake Info")
    try:
        from dashboard.data_manager import DataLakeManager
        data_manager = DataLakeManager()
        container_name = os.getenv('ETL_DATA_LAKE_STORAGE_CONTAINER_NAME', 'raw')
        raw_files = data_manager.list_files(container_name, "news_data/latest")
        if raw_files:
            last_update = datetime.fromtimestamp(
                max([f.get("last_modified", 0) for f in raw_files])
            ).strftime("%Y-%m-%d %H:%M:%S")
            st.sidebar.info(f"Last data update: {last_update}")
    except Exception as e:
        st.sidebar.warning(f"Unable to fetch data lake info: {str(e)}")

if __name__ == "__main__":
    main()
