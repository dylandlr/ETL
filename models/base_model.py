from abc import ABC, abstractmethod
import mlflow
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, RegressionEvaluator

class BaseModel(ABC):
    """Base class for all ML models"""
    
    def __init__(self, model_name, task_type="classification"):
        self.model_name = model_name
        self.task_type = task_type
        self.model = None
    
    @abstractmethod
    def create_model(self):
        """Create and return the specific model implementation"""
        pass
    
    def train(self, train_data, test_data):
        """Train the model and log metrics with MLflow"""
        if not self.model:
            self.model = self.create_model()
        
        # Train model
        trained_model = self.model.fit(train_data)
        
        # Make predictions
        predictions = trained_model.transform(test_data)
        
        # Evaluate based on task type
        metrics = self._evaluate_model(predictions)
        
        # Log with MLflow
        with mlflow.start_run(run_name=f"{self.model_name}_training"):
            for metric_name, value in metrics.items():
                mlflow.log_metric(metric_name, value)
            mlflow.spark.log_model(trained_model, self.model_name)
        
        return trained_model, metrics
    
    def _evaluate_model(self, predictions):
        """Evaluate model based on task type"""
        metrics = {}
        
        if self.task_type == "classification":
            evaluator = MulticlassClassificationEvaluator(
                labelCol="label",
                predictionCol="prediction"
            )
            
            # Calculate classification metrics
            metrics["accuracy"] = evaluator.setMetricName("accuracy").evaluate(predictions)
            metrics["f1"] = evaluator.setMetricName("f1").evaluate(predictions)
            metrics["precision"] = evaluator.setMetricName("weightedPrecision").evaluate(predictions)
            metrics["recall"] = evaluator.setMetricName("weightedRecall").evaluate(predictions)
            
        elif self.task_type == "regression":
            evaluator = RegressionEvaluator(
                labelCol="label",
                predictionCol="prediction"
            )
            
            # Calculate regression metrics
            metrics["rmse"] = evaluator.setMetricName("rmse").evaluate(predictions)
            metrics["mae"] = evaluator.setMetricName("mae").evaluate(predictions)
            metrics["r2"] = evaluator.setMetricName("r2").evaluate(predictions)
        
        return metrics
