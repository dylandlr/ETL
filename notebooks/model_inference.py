from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.feature import VectorAssembler
import mlflow

def create_spark_session():
    spark = SparkSession.builder \
        .appName("ETL Model Inference") \
        .getOrCreate()
    return spark

def load_model(model_path):
    """Load the trained model"""
    return RandomForestClassificationModel.load(model_path)

def prepare_inference_data(df, feature_columns):
    """Prepare data for inference"""
    assembler = VectorAssembler(
        inputCols=feature_columns,
        outputCol="features"
    )
    return assembler.transform(df)

def make_predictions(model, data):
    """Make predictions using the trained model"""
    predictions = model.transform(data)
    return predictions.select("prediction", *data.columns)

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
    prepared_data = prepare_inference_data(inference_data, feature_columns)
    predictions = make_predictions(model, prepared_data)
    
    # Log prediction metrics
    with mlflow.start_run(run_name="model_inference"):
        mlflow.log_metric("prediction_count", predictions.count())
    
    # Save predictions
    save_predictions(predictions, predictions_output_path)
