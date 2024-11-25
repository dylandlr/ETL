import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import json
import mlflow
import mlflow.sklearn
import time
from datetime import datetime, timedelta
from dashboard.data_manager import DataLakeManager
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

def prepare_trend_data(df):
    """Prepare trend data from raw news data"""
    # Convert date strings to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Group by date
    daily_stats = []
    
    for date, group in df.groupby(df['date'].dt.date):
        # Topic trends (using TF-IDF on titles)
        vectorizer = TfidfVectorizer(max_features=100)
        title_vectors = vectorizer.fit_transform(group['title'])
        top_topics = dict(zip(
            vectorizer.get_feature_names_out(),
            title_vectors.sum(axis=0).A1
        ))
        
        # Category distribution
        category_dist = group['category'].value_counts().to_dict()
        
        # Sentiment distribution
        sentiment_dist = {
            'positive': len(group[group['label'] == 2]),
            'neutral': len(group[group['label'] == 1]),
            'negative': len(group[group['label'] == 0])
        }
        
        # Source distribution
        source_dist = group['source'].value_counts().to_dict()
        
        daily_stats.append({
            'date': date,
            'topics': top_topics,
            'categories': category_dist,
            'sentiment': sentiment_dist,
            'sources': source_dist,
            'total_articles': len(group)
        })
    
    return pd.DataFrame(daily_stats)

def extract_features(trend_data, prediction_horizon=7):
    """Extract features for prediction"""
    features = []
    targets = []
    
    for i in range(len(trend_data) - prediction_horizon):
        current_day = trend_data.iloc[i]
        future_day = trend_data.iloc[i + prediction_horizon]
        
        # Feature vector for current day
        feature_vector = {
            'total_articles': current_day['total_articles'],
            'positive_ratio': current_day['sentiment']['positive'] / current_day['total_articles'],
            'negative_ratio': current_day['sentiment']['negative'] / current_day['total_articles']
        }
        
        # Add top category proportions
        for category, count in current_day['categories'].items():
            feature_vector[f'category_{category}'] = count / current_day['total_articles']
        
        # Add top source proportions
        for source, count in current_day['sources'].items():
            feature_vector[f'source_{source}'] = count / current_day['total_articles']
        
        # Target values (changes in topic frequencies)
        target_vector = {}
        for topic in current_day['topics']:
            current_freq = current_day['topics'].get(topic, 0)
            future_freq = future_day['topics'].get(topic, 0)
            target_vector[topic] = (future_freq - current_freq) / (current_freq + 1e-6)  # Relative change
        
        features.append(feature_vector)
        targets.append(target_vector)
    
    return pd.DataFrame(features), pd.DataFrame(targets)

def train_prediction_model(X, y):
    """Train prediction model"""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    metrics = {
        'mae': mean_absolute_error(y_test, y_pred),
        'r2': r2_score(y_test, y_pred),
        'accuracy': np.mean(np.abs(y_pred - y_test) < 0.1)  # Within 10% error
    }
    
    return model, scaler, metrics

def generate_predictions(model, scaler, latest_features, topic_names):
    """Generate predictions for the next period"""
    # Scale features
    features_scaled = scaler.transform(latest_features)
    
    # Make predictions
    predictions = model.predict(features_scaled)
    
    # Convert predictions to interpretable format
    topic_changes = dict(zip(topic_names, predictions[0]))
    
    # Sort topics by predicted change
    rising_topics = [
        {'name': topic, 'growth': round(change * 100, 1)}
        for topic, change in sorted(topic_changes.items(), key=lambda x: x[1], reverse=True)[:5]
        if change > 0
    ]
    
    declining_topics = [
        {'name': topic, 'decline': round(-change * 100, 1)}
        for topic, change in sorted(topic_changes.items(), key=lambda x: x[1])[:5]
        if change < 0
    ]
    
    return {
        'rising_topics': rising_topics,
        'declining_topics': declining_topics
    }

def main():
    print("Starting news prediction pipeline...")
    
    # Ensure directories exist
    os.makedirs('data', exist_ok=True)
    os.makedirs('analysis', exist_ok=True)
    
    # Fetch latest data
    try:
        data_manager = DataLakeManager()
        data_manager.download_file('raw', 'news_data/latest/raw_news_data.csv', 'data/raw_news_data.csv')
    except Exception as e:
        print(f"Error fetching data: {str(e)}")
        print("Using existing data files...")
    
    # Load raw data
    raw_df = pd.read_csv('data/raw_news_data.csv')
    print(f"\nLoaded {len(raw_df)} records")
    
    # Prepare trend data
    trend_data = prepare_trend_data(raw_df)
    
    # Extract features and targets
    X, y = extract_features(trend_data)
    
    # Start MLflow run
    with mlflow.start_run():
        # Train model
        model, scaler, metrics = train_prediction_model(X, y)
        
        # Log metrics
        mlflow.log_metrics(metrics)
        
        # Generate predictions
        latest_features = X.iloc[-1:].copy()
        predictions = generate_predictions(model, scaler, latest_features, y.columns)
        
        # Add category forecast
        latest_day = trend_data.iloc[-1]
        total = sum(latest_day['categories'].values())
        category_forecast = {
            category: round(count / total * 100, 1)
            for category, count in latest_day['categories'].items()
        }
        predictions['category_forecast'] = category_forecast
        
        # Add metrics
        predictions['metrics'] = metrics
        
        # Save predictions
        with open('analysis/predictions.json', 'w') as f:
            json.dump(predictions, f, indent=2)
        
        print("\nPrediction pipeline completed successfully!")
        print(f"Metrics: {metrics}")

if __name__ == "__main__":
    main()
