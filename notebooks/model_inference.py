from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
import mlflow
import mlflow.spark

def create_spark_session():
    spark = SparkSession.builder \
        .appName("ETL Model Inference") \
        .getOrCreate()
    return spark

def prepare_features(df, feature_columns):
    """Prepare feature vector for inference"""
    assembler = VectorAssembler(
        inputCols=feature_columns,
        outputCol="features"
    )
    return assembler.transform(df).select("features")

def load_model(model_path):
    """Load a trained model"""
    try:
        # Try loading from MLflow
        model = mlflow.spark.load_model(model_path)
    except Exception as e:
        # Fallback to loading directly from filesystem
        print(f"Failed to load from MLflow: {str(e)}")
        print("Attempting to load from filesystem...")
        model = mlflow.spark.load_model(f"file://{model_path}")
    return model

def predict(model, data):
    """Make predictions using the loaded model"""
    predictions = model.transform(data)
    return predictions.select("prediction")

def save_predictions(predictions_df, output_path):
    """Save predictions to the data lake"""
    predictions_df.write.format("delta").mode("overwrite").save(output_path)

if __name__ == "__main__":
    # Initialize Spark
    spark = create_spark_session()
    
    # Define paths
    model_path = "/processed/models/random_forest"
    inference_data_path = "/raw/input/inference_data"
    predictions_output_path = "/processed/predictions"
    
    # Load model and data
    model = load_model(model_path)
    inference_data = spark.read.format("delta").load(inference_data_path)
    
    # Define feature columns (same as training)
    feature_columns = ["feature1", "feature2", "feature3"]  # Replace with your actual feature columns
    
    # Prepare data and make predictions
    prepared_data = prepare_features(inference_data, feature_columns)
    predictions = predict(model, prepared_data)
    
    # Log prediction metrics
    with mlflow.start_run(run_name="model_inference"):
        mlflow.log_metric("prediction_count", predictions.count())
    
    # Save predictions
    save_predictions(predictions, predictions_output_path)
