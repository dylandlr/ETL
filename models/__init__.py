from .base_model import BaseModel
from .random_forest import RandomForestModel
import importlib
import os

# Dictionary mapping model type names to their classes
MODEL_REGISTRY = {
    'random_forest': RandomForestModel,
}

def register_model(model_type, model_class):
    """Register a new model type"""
    MODEL_REGISTRY[model_type] = model_class

def get_model_class(model_type):
    """Get the model class for a given model type"""
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Model type '{model_type}' not found. Available models: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[model_type]

# Auto-discover and register models in the models directory
models_dir = os.path.dirname(__file__)
for filename in os.listdir(models_dir):
    if filename.endswith('.py') and not filename.startswith('__'):
        module_name = filename[:-3]  # Remove .py extension
        try:
            module = importlib.import_module(f".{module_name}", package="models")
            for item_name in dir(module):
                item = getattr(module, item_name)
                if isinstance(item, type) and issubclass(item, BaseModel) and item != BaseModel:
                    model_type = item_name.lower().replace('model', '')
                    register_model(model_type, item)
        except Exception as e:
            print(f"Warning: Could not load model from {filename}: {str(e)}")

__all__ = ['BaseModel', 'MODEL_REGISTRY', 'register_model', 'get_model_class']
