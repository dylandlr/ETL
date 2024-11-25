import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
import json
import mlflow
import mlflow.sklearn
import time
from dashboard.data_manager import DataLakeManager
from sklearn.feature_extraction.text import TfidfVectorizer

def analyze_predictions(model, X_test_scaled, y_test, raw_df, feature_names):
    """Analyze model predictions with examples"""
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Create analysis DataFrame
    analysis_df = pd.DataFrame({
        'Actual Sentiment': ['Negative' if y == 0 else 'Neutral' if y == 1 else 'Positive' for y in y_test],
        'Predicted Sentiment': ['Negative' if y == 0 else 'Neutral' if y == 1 else 'Positive' for y in y_pred],
        'Title': raw_df.iloc[y_test.index]['title'],
        'Description': raw_df.iloc[y_test.index]['description'],
        'Category': raw_df.iloc[y_test.index]['category'],
        'Polarity': raw_df.iloc[y_test.index]['polarity'],
        'Subjectivity': raw_df.iloc[y_test.index]['subjectivity']
    })
    
    # Add correct/incorrect column
    analysis_df['Correct'] = y_test == y_pred
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Save analysis results
    analysis_dir = 'analysis'
    os.makedirs(analysis_dir, exist_ok=True)
    
    # Save confusion matrix
    cm_dict = {
        'matrix': cm.tolist(),
        'labels': ['Negative', 'Neutral', 'Positive']
    }
    with open(f'{analysis_dir}/confusion_matrix.json', 'w') as f:
        json.dump(cm_dict, f)
    
    # Save classification report
    report = classification_report(y_test, y_pred, 
                                 target_names=['Negative', 'Neutral', 'Positive'],
                                 output_dict=True,
                                 zero_division=0)
    with open(f'{analysis_dir}/classification_report.json', 'w') as f:
        json.dump(report, f)
    
    # Save example predictions
    examples = []
    for sentiment in ['Negative', 'Neutral', 'Positive']:
        sentiment_examples = {
            'sentiment': sentiment,
            'correct': [],
            'incorrect': []
        }
        
        # Correct predictions
        correct = analysis_df[
            (analysis_df['Actual Sentiment'] == sentiment) & 
            (analysis_df['Correct'])
        ].head(2)
        
        for _, row in correct.iterrows():
            sentiment_examples['correct'].append({
                'title': row['Title'],
                'category': row['Category'],
                'polarity': float(row['Polarity']),
                'subjectivity': float(row['Subjectivity'])
            })
        
        # Incorrect predictions
        incorrect = analysis_df[
            (analysis_df['Actual Sentiment'] == sentiment) & 
            (~analysis_df['Correct'])
        ].head(2)
        
        for _, row in incorrect.iterrows():
            sentiment_examples['incorrect'].append({
                'title': row['Title'],
                'category': row['Category'],
                'predicted': row['Predicted Sentiment'],
                'polarity': float(row['Polarity']),
                'subjectivity': float(row['Subjectivity'])
            })
        
        examples.append(sentiment_examples)
    
    with open(f'{analysis_dir}/example_predictions.json', 'w') as f:
        json.dump(examples, f)
    
    print("\nConfusion Matrix:")
    print("Labels: 0=Negative, 1=Neutral, 2=Positive")
    print(cm)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                              target_names=['Negative', 'Neutral', 'Positive'],
                              zero_division=0))
    
    print("\nExample Predictions:")
    for sentiment in ['Negative', 'Neutral', 'Positive']:
        print(f"\n{sentiment} Articles:")
        
        # Correct predictions
        correct = analysis_df[
            (analysis_df['Actual Sentiment'] == sentiment) & 
            (analysis_df['Correct'])
        ].head(2)
        if not correct.empty:
            print("\nCorrectly Classified:")
            for _, row in correct.iterrows():
                print(f"Title: {row['Title']}")
                print(f"Category: {row['Category']}")
                print(f"Polarity: {row['Polarity']:.3f}")
                print(f"Subjectivity: {row['Subjectivity']:.3f}")
                print("---")
        
        # Incorrect predictions
        incorrect = analysis_df[
            (analysis_df['Actual Sentiment'] == sentiment) & 
            (~analysis_df['Correct'])
        ].head(2)
        if not incorrect.empty:
            print("\nIncorrectly Classified:")
            for _, row in incorrect.iterrows():
                print(f"Title: {row['Title']}")
                print(f"Category: {row['Category']}")
                print(f"Predicted as: {row['Predicted Sentiment']}")
                print(f"Polarity: {row['Polarity']:.3f}")
                print(f"Subjectivity: {row['Subjectivity']:.3f}")
                print("---")
    
    return analysis_df

def prepare_data(raw_df):
    """Prepare data for training"""
    # Create features from text
    vectorizer = TfidfVectorizer(max_features=100)
    title_vectors = vectorizer.fit_transform(raw_df['processed_title'])
    
    # Convert to DataFrame
    title_features = pd.DataFrame(
        title_vectors.toarray(),
        columns=[f'title_word_{i}' for i in range(title_vectors.shape[1])]
    )
    
    # Add other features
    features = pd.concat([
        title_features,
        pd.get_dummies(raw_df['category'], prefix='category'),
        raw_df[['polarity', 'subjectivity']]
    ], axis=1)
    
    return features, raw_df['label']

def train_model(X_train, X_test, y_train, y_test, model_params):
    """Train and evaluate the model"""
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Calculate class weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(zip(np.unique(y_train), class_weights))
    print(f"\nClass weights: {class_weight_dict}")
    
    # Update model parameters with class weights
    model_params['class_weight'] = class_weight_dict
    
    # Train model
    model = RandomForestClassifier(**model_params)
    model.fit(X_train_scaled, y_train)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    print(f"\nCross-validation scores: {cv_scores}")
    print(f"Mean CV score: {cv_scores.mean():.2f}")
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }
    
    return model, metrics, scaler, X_test_scaled

def fetch_latest_data():
    """Fetch latest data from Data Lake"""
    data_manager = DataLakeManager()
    
    # Download latest raw and prepared data
    data_manager.download_file('raw', 'news_data/latest/raw_news_data.csv', 'data/raw_news_data.csv')
    data_manager.download_file('processed', 'news_data/latest/prepared_news_data.csv', 'data/prepared_news_data.csv')
    
    print("Successfully fetched latest data from Data Lake")

def main():
    try:
        # Initialize DataLakeManager
        data_manager = DataLakeManager()
        container_name = os.getenv('ETL_DATA_LAKE_STORAGE_CONTAINER_NAME', 'raw')
        
        print("Loading data from Data Lake...")
        df = data_manager.read_csv_from_datalake(container_name, "news_data/latest/raw_news_data.csv")
        
        if df is None:
            print("No data found in Data Lake. Please run fetch_news_data.py first.")
            return
        
        print(f"Loaded {len(df)} articles from Data Lake")
        
        # Prepare data
        print("\nPreparing data...")
        X, y = prepare_data(df)
        feature_columns = X.columns.tolist()
        
        # Split into train and test sets with random state
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=int(time.time()) % 10000, stratify=y
        )
        
        # Add some randomization to model parameters
        model_params = {
            'n_estimators': np.random.randint(150, 250),
            'max_depth': np.random.randint(8, 12),
            'min_samples_split': np.random.randint(4, 7),
            'min_samples_leaf': np.random.randint(2, 4),
            'random_state': int(time.time()) % 10000
        }
        
        print("\nTraining Random Forest model...")
        print(f"Parameters: {model_params}")
        
        # Train and evaluate model
        model, metrics, scaler, X_test_scaled = train_model(X_train, X_test, y_train, y_test, model_params)
        
        print("\nModel Performance Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        # Analyze predictions and save results
        print("\nAnalyzing Predictions...")
        analysis_df = analyze_predictions(model, X_test_scaled, y_test, df, feature_columns)
        
        # Save model and scaler
        model_dir = 'models'
        os.makedirs(model_dir, exist_ok=True)
        
        # Save artifacts
        import joblib
        joblib.dump(model, f'{model_dir}/random_forest_model.joblib')
        joblib.dump(scaler, f'{model_dir}/scaler.joblib')
        print(f"\nModel and scaler saved to {model_dir}/")
        
        # Calculate class weights for logging
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y),
            y=y
        )
        class_weight_dict = dict(zip(np.unique(y), class_weights))
        
        # Log with MLflow
        with mlflow.start_run(run_name="random_forest_training") as run:
            # Log parameters
            params = {
                **model_params,  # Include all model parameters
                'class_weights': str(class_weight_dict),  # Convert dict to string for logging
                'train_size': len(X_train),
                'test_size': len(X_test),
                'total_size': len(X)
            }
            mlflow.log_params(params)
            
            # Log metrics
            metrics['training_size'] = len(X_train)
            metrics['test_size'] = len(X_test)
            metrics['total_size'] = len(df)
            
            for metric_name, value in metrics.items():
                mlflow.log_metric(metric_name, value)
                
            # Save and log feature importance
            importance_df = pd.DataFrame({
                'feature': feature_columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            importance_df.to_json('analysis/feature_importance.json', orient='records')
            mlflow.log_artifact('analysis/feature_importance.json')
            
            print("\nTop 10 Most Important Features:")
            print(importance_df.head(10).to_string())
            
            # Create model signature and input example
            from mlflow.models.signature import infer_signature
            signature = infer_signature(X_test_scaled, y_test)
            input_example = X_test_scaled[0:1]
            
            # Log model with signature and input example
            mlflow.sklearn.log_model(
                model, 
                "random_forest_model",
                signature=signature,
                input_example=input_example
            )
            
            # Log artifacts
            mlflow.log_artifact('analysis/confusion_matrix.json')
            mlflow.log_artifact('analysis/classification_report.json')
            mlflow.log_artifact('analysis/example_predictions.json')
            
            # Upload metrics to Data Lake
            results_dir = 'results'
            os.makedirs(results_dir, exist_ok=True)
            with open(f'{results_dir}/latest_metrics.json', 'w') as f:
                json.dump(metrics, f)
            success, message = data_manager.upload_file(
                container_name,
                f'{results_dir}/latest_metrics.json',
                "news_data/latest/metrics.json"
            )
            print(f"Upload metrics to Data Lake: {message}")
            
            print(f"\nLogged run to MLflow with ID: {run.info.run_id}")
            
    except Exception as e:
        print(f"Error in pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main()
