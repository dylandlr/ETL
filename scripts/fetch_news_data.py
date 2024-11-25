import os
import pandas as pd
import numpy as np
from newsapi import NewsApiClient
from textblob import TextBlob
from dotenv import load_dotenv
import datetime as dt
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from datetime import datetime, timedelta
import sys

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# Import DataLakeManager
from dashboard.data_manager import DataLakeManager

# Load environment variables
load_dotenv('azure/.env')
NEWS_API_KEY = os.getenv('NEWS_API_KEY')

def preprocess_text(text):
    """Basic text preprocessing"""
    # Convert to lowercase and remove special characters
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

def extract_text_features(df):
    """Extract TF-IDF and other text features"""
    # Preprocess text
    df['processed_title'] = df['title'].apply(preprocess_text)
    df['processed_description'] = df['description'].apply(preprocess_text)
    
    # TF-IDF for title and description
    tfidf = TfidfVectorizer(max_features=20)
    
    title_tfidf = tfidf.fit_transform(df['processed_title'])
    desc_tfidf = tfidf.fit_transform(df['processed_description'])
    
    # Convert to DataFrame
    title_features = pd.DataFrame(
        title_tfidf.toarray(),
        columns=[f'title_tfidf_{i}' for i in range(title_tfidf.shape[1])]
    )
    desc_features = pd.DataFrame(
        desc_tfidf.toarray(),
        columns=[f'desc_tfidf_{i}' for i in range(desc_tfidf.shape[1])]
    )
    
    # Combine with original dataframe
    df = pd.concat([df, title_features, desc_features], axis=1)
    
    return df

def fetch_news():
    """Fetch news articles and create labeled dataset"""
    newsapi = NewsApiClient(api_key=NEWS_API_KEY)
    articles = []
    
    # Get news from different categories
    categories = ['business', 'technology', 'science', 'health']
    
    # Fetch articles from the last 7 days to get more data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    
    # Keywords likely to have negative sentiment
    negative_keywords = [
        'crisis', 'disaster', 'failure', 'crash', 'decline',
        'warning', 'threat', 'risk', 'danger', 'problem'
    ]
    
    print("Fetching general category news...")
    for category in categories:
        # Fetch regular category news
        response = newsapi.get_top_headlines(
            language='en',
            country='us',
            category=category,
            page_size=100  # Get maximum articles per category
        )
        
        for article in response['articles']:
            if article['description']:
                articles.append(process_article(article, category))
    
    print("\nFetching news with potentially negative sentiment...")
    # Fetch additional articles with potentially negative sentiment
    for keyword in negative_keywords:
        response = newsapi.get_everything(
            q=keyword,
            language='en',
            from_param=start_date.strftime('%Y-%m-%d'),
            to=end_date.strftime('%Y-%m-%d'),
            sort_by='relevancy',
            page_size=20  # Increased from 5 to 20 articles per keyword
        )
        
        for article in response['articles']:  # Remove the [:5] limit
            if article['description']:
                articles.append(process_article(article, 'general'))
    
    df = pd.DataFrame(articles)
    
    # Print sentiment distribution
    print("\nSentiment Distribution:")
    sentiment_dist = df['label'].map({0: 'Negative', 1: 'Neutral', 2: 'Positive'}).value_counts()
    print(sentiment_dist)
    
    return df

def process_article(article, category):
    """Process a single article and extract features"""
    # Use TextBlob for sentiment analysis
    blob = TextBlob(article['description'])
    sentiment = blob.sentiment
    
    # Convert sentiment to classification: negative (0), neutral (1), positive (2)
    label = 0 if sentiment.polarity < -0.1 else (2 if sentiment.polarity > 0.1 else 1)
    
    return {
        'title': article['title'],
        'description': article['description'],
        'url': article['url'],  # Add URL for deduplication
        'category': category,
        'published_at': article['publishedAt'],
        'polarity': sentiment.polarity,
        'subjectivity': sentiment.subjectivity,
        'label': label
    }

def prepare_features(df):
    """Prepare features for model training"""
    # Extract text features
    df = extract_text_features(df)
    
    # Convert categorical variables
    df['category_encoded'] = pd.Categorical(df['category']).codes
    
    # Create text length features
    df['title_length'] = df['title'].str.len()
    df['description_length'] = df['description'].str.len()
    
    # Create word count features
    df['title_word_count'] = df['processed_title'].str.split().str.len()
    df['description_word_count'] = df['processed_description'].str.split().str.len()
    
    # Convert published_at to datetime features
    df['published_at'] = pd.to_datetime(df['published_at'])
    df['hour'] = df['published_at'].dt.hour
    df['day_of_week'] = df['published_at'].dt.dayofweek
    
    # Select features for model
    feature_columns = (
        ['category_encoded', 'title_length', 'description_length', 
         'title_word_count', 'description_word_count', 'hour', 'day_of_week',
         'polarity', 'subjectivity'] +
        [col for col in df.columns if 'tfidf' in col]
    )
    
    return df[feature_columns + ['label']], feature_columns

def main():
    # Initialize DataLakeManager
    data_manager = DataLakeManager()
    container_name = os.getenv('ETL_DATA_LAKE_STORAGE_CONTAINER_NAME', 'raw')
    
    # Create container if it doesn't exist
    try:
        containers = data_manager.list_containers()
        if container_name not in containers:
            print(f"Creating container '{container_name}'...")
            data_manager.service_client.create_file_system(file_system=container_name)
    except Exception as e:
        print(f"Error checking/creating container: {str(e)}")
    
    # Fetch new data
    print("Fetching news articles...")
    new_df = fetch_news()
    print(f"Fetched {len(new_df)} new articles")
    
    # Load existing data from Data Lake if available
    try:
        existing_df = data_manager.read_csv_from_datalake(container_name, "news_data/latest/raw_news_data.csv")
        if existing_df is not None:
            print(f"Found {len(existing_df)} existing articles in Data Lake")
            
            # Add timestamp to new articles
            new_df['fetch_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Combine existing and new data, dropping duplicates based on title and url
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df = combined_df.drop_duplicates(subset=['title', 'url'], keep='last')
            
            print(f"Total unique articles after combining: {len(combined_df)}")
            df = combined_df
        else:
            print("No existing data found in Data Lake, using newly fetched articles")
            new_df['fetch_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            df = new_df
    except Exception as e:
        print(f"Error reading from Data Lake: {str(e)}")
        print("Using only newly fetched articles")
        new_df['fetch_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        df = new_df
    
    # Prepare features
    print("\nPreparing features...")
    df_prepared, feature_columns = prepare_features(df)
    
    # Save to local directory first
    data_dir = 'data'
    os.makedirs(data_dir, exist_ok=True)
    
    local_raw_path = f'{data_dir}/raw_news_data.csv'
    local_prepared_path = f'{data_dir}/prepared_news_data.csv'
    
    df.to_csv(local_raw_path, index=False)
    df_prepared.to_csv(local_prepared_path, index=False)
    
    # Save a timestamped copy locally
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    history_dir = f'{data_dir}/history'
    os.makedirs(history_dir, exist_ok=True)
    history_path = f'{history_dir}/raw_news_data_{timestamp}.csv'
    df.to_csv(history_path, index=False)
    
    # Upload to Data Lake
    print("\nSyncing with Data Lake...")
    
    # Upload latest version
    success, message = data_manager.upload_file(
        container_name,
        local_raw_path,
        "news_data/latest/raw_news_data.csv"
    )
    print(f"Upload latest raw data: {message}")
    
    success, message = data_manager.upload_file(
        container_name,
        local_prepared_path,
        "news_data/latest/prepared_news_data.csv"
    )
    print(f"Upload latest prepared data: {message}")
    
    # Upload historical version
    success, message = data_manager.upload_file(
        container_name,
        history_path,
        f"news_data/history/raw_news_data_{timestamp}.csv"
    )
    print(f"Upload historical data: {message}")
    
    print("\nData Lake sync completed")
    print(f"Total articles in dataset: {len(df)}")
    print(f"Articles by category:")
    print(df['category'].value_counts())

if __name__ == "__main__":
    main()
