from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, isnan, count
import mlflow

def create_spark_session():
    spark = SparkSession.builder \
        .appName("ETL Data Preprocessing") \
        .getOrCreate()
    return spark

def load_data(spark, input_path):
    """Load data from the data lake"""
    return spark.read.format("delta").load(input_path)

def clean_data(df):
    """Basic data cleaning operations"""
    # Get count of null values
    null_counts = df.select([count(when(isnan(c) | col(c).isNull(), c)).alias(c) for c in df.columns])
    
    # Drop rows with null values
    df_cleaned = df.dropna()
    
    # Log metrics
    with mlflow.start_run(run_name="data_preprocessing"):
        mlflow.log_metric("initial_row_count", df.count())
        mlflow.log_metric("cleaned_row_count", df_cleaned.count())
    
    return df_cleaned

def save_processed_data(df, output_path):
    """Save processed data back to the data lake"""
    df.write.format("delta").mode("overwrite").save(output_path)

if __name__ == "__main__":
    # Initialize Spark
    spark = create_spark_session()
    
    # Define paths (these will be passed as parameters in ADF)
    input_path = "/raw/input/your_dataset"
    output_path = "/raw/staging/processed_data"
    
    # Execute preprocessing pipeline
    raw_data = load_data(spark, input_path)
    cleaned_data = clean_data(raw_data)
    save_processed_data(cleaned_data, output_path)
