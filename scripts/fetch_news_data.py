import os
import pandas as pd
from newsapi import NewsApiClient
from textblob import TextBlob
from dotenv import load_dotenv
import datetime as dt

# Load environment variables
load_dotenv('azure/.env')
NEWS_API_KEY = os.getenv('NEWS_API_KEY')

def fetch_news():
    """Fetch news articles and create labeled dataset"""
    newsapi = NewsApiClient(api_key=NEWS_API_KEY)
    
    # Get news from different categories
    categories = ['business', 'technology', 'science', 'health']
    articles = []
    
    for category in categories:
        response = newsapi.get_top_headlines(
            language='en',
            country='us',
            category=category
        )
        
        for article in response['articles']:
            if article['description']:
                # Use TextBlob for sentiment analysis to create labels
                sentiment = TextBlob(article['description']).sentiment.polarity
                # Convert sentiment to classification: negative (0), neutral (1), positive (2)
                label = 0 if sentiment < -0.1 else (2 if sentiment > 0.1 else 1)
                
                articles.append({
                    'title': article['title'],
                    'description': article['description'],
                    'category': category,
                    'published_at': article['publishedAt'],
                    'sentiment_score': sentiment,
                    'label': label
                })
    
    return pd.DataFrame(articles)

def prepare_features(df):
    """Prepare features for model training"""
    # Convert categorical variables
    df['category_encoded'] = pd.Categorical(df['category']).codes
    
    # Create text length features
    df['title_length'] = df['title'].str.len()
    df['description_length'] = df['description'].str.len()
    
    # Convert published_at to datetime features
    df['published_at'] = pd.to_datetime(df['published_at'])
    df['hour'] = df['published_at'].dt.hour
    df['day_of_week'] = df['published_at'].dt.dayofweek
    
    # Select features for model
    feature_columns = ['category_encoded', 'title_length', 'description_length', 'hour', 'day_of_week']
    
    return df[feature_columns + ['label']], feature_columns

def main():
    # Fetch and prepare data
    print("Fetching news articles...")
    df = fetch_news()
    print(f"Fetched {len(df)} articles")
    
    # Prepare features
    print("\nPreparing features...")
    df_prepared, feature_columns = prepare_features(df)
    
    # Save raw and prepared data
    data_dir = 'data'
    os.makedirs(data_dir, exist_ok=True)
    
    df.to_csv(f'{data_dir}/raw_news_data.csv', index=False)
    df_prepared.to_csv(f'{data_dir}/prepared_news_data.csv', index=False)
    
    print("\nData saved to 'data' directory")
    print(f"Features used: {feature_columns}")
    print("\nLabel distribution:")
    print(df_prepared['label'].value_counts())

if __name__ == "__main__":
    main()
