import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import mlflow

def train_model(X_train, X_test, y_train, y_test):
    """Train and evaluate Random Forest model"""
    # Initialize model
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    
    # Train model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred, average='weighted'),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted')
    }
    
    # Log with MLflow
    with mlflow.start_run(run_name="random_forest_training"):
        mlflow.log_params({
            'n_estimators': 100,
            'max_depth': 5
        })
        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name, value)
        mlflow.sklearn.log_model(model, "random_forest_model")
    
    return model, metrics

def main():
    # Load prepared data
    data_path = 'data/prepared_news_data.csv'
    df = pd.read_csv(data_path)
    print(f"\nLoaded {len(df)} records from {data_path}")
    
    # Split features and label
    feature_columns = ['category_encoded', 'title_length', 'description_length', 'hour', 'day_of_week']
    X = df[feature_columns]
    y = df['label']
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("\nTraining Random Forest model...")
    # Train and evaluate model
    model, metrics = train_model(X_train, X_test, y_train, y_test)
    
    print("\nModel Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Make predictions on test set
    print("\nMaking predictions on test set...")
    predictions = model.predict(X_test)
    
    print("\nPrediction Distribution:")
    print(pd.Series(predictions).value_counts())
    
    # Feature importance
    print("\nFeature Importance:")
    importance_df = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(importance_df)

if __name__ == "__main__":
    main()
