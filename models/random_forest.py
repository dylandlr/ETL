from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.regression import RandomForestRegressor
from .base_model import BaseModel

class RandomForestModel(BaseModel):
    """Random Forest implementation"""
    
    def __init__(self, task_type="classification", num_trees=100, max_depth=5):
        super().__init__("random_forest", task_type)
        self.num_trees = num_trees
        self.max_depth = max_depth
    
    def create_model(self):
        """Create either a classifier or regressor based on task_type"""
        if self.task_type == "classification":
            return RandomForestClassifier(
                labelCol="label",
                featuresCol="features",
                numTrees=self.num_trees,
                maxDepth=self.max_depth
            )
        else:
            return RandomForestRegressor(
                labelCol="label",
                featuresCol="features",
                numTrees=self.num_trees,
                maxDepth=self.max_depth
            )
