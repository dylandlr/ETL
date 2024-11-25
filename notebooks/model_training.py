from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import mlflow
import mlflow.spark
from models import get_model_class, RandomForestModel

def create_spark_session():
    spark = SparkSession.builder \
        .appName("ETL Model Training") \
        .getOrCreate()
    return spark

def prepare_features(df, feature_columns, label_column):
    """Prepare feature vector for training"""
    assembler = VectorAssembler(
        inputCols=feature_columns,
        outputCol="features"
    )
    return assembler.transform(df).select("features", label_column)

def train_model(train_data, test_data, model_type="random_forest", task_type="classification", **model_params):
    """Modular model training function for any ML model"""
    # Get the appropriate model class
    model_class = get_model_class(model_type)
    
    # Initialize and train the model
    model = model_class(task_type=task_type, **model_params)
    trained_model, metrics = model.train(train_data, test_data)
    
    return trained_model, metrics

def save_model(model, path):
    """Save the trained model"""
    model.write().overwrite().save(path)

if __name__ == "__main__":
    # Initialize Spark
    spark = create_spark_session()
    
    # Define paths
    input_path = "/raw/staging/processed_data"
    model_output_path = "/processed/models/random_forest"
    
    # Load processed data
    data = spark.read.format("delta").load(input_path)
    
    # Split data into training and test sets
    train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)
    
    # Define your feature columns and label column
    feature_columns = ["feature1", "feature2", "feature3"]  # Replace with your actual feature columns
    label_column = "label"  # Replace with your actual label column
    
    # Prepare features
    train_data = prepare_features(train_data, feature_columns, label_column)
    test_data = prepare_features(test_data, feature_columns, label_column)
    
    # Train and evaluate model
    model, accuracy = train_model(train_data, test_data, model_type="random_forest", task_type="classification")
    print(f"Model Accuracy: {accuracy}")
    
    # Save model
    save_model(model, model_output_path)
