import pytest
from pyspark.sql import SparkSession
import os
from dotenv import load_dotenv

@pytest.fixture(scope="session")
def spark():
    """Create a Spark session for testing."""
    spark = (SparkSession.builder
            .appName("ETL-Tests")
            .config("spark.sql.warehouse.dir", "/tmp/spark-warehouse")
            .config("spark.driver.memory", "2g")
            .config("spark.executor.memory", "2g")
            .getOrCreate())
    yield spark
    spark.stop()

@pytest.fixture(scope="session")
def test_data(spark):
    """Create sample test data."""
    data = [
        ("1", "John", "Doe", "2023-01-01", 100.0),
        ("2", "Jane", "Smith", "2023-01-02", 200.0),
        ("3", "Bob", "Johnson", "2023-01-03", 300.0)
    ]
    columns = ["id", "first_name", "last_name", "date", "amount"]
    return spark.createDataFrame(data, columns)

@pytest.fixture(scope="session")
def env_vars():
    """Load environment variables."""
    load_dotenv()
    return {
        "STORAGE_ACCOUNT_NAME": os.getenv("STORAGE_ACCOUNT_NAME"),
        "STORAGE_CONTAINER_NAME": os.getenv("STORAGE_CONTAINER_NAME"),
        "DATABRICKS_WORKSPACE_NAME": os.getenv("DATABRICKS_WORKSPACE_NAME"),
        "RESOURCE_GROUP_NAME": os.getenv("RESOURCE_GROUP_NAME")
    }

@pytest.fixture(scope="session")
def quality_rules():
    """Define test quality rules."""
    return {
        "null_check": {
            "type": "null_check",
            "columns": ["id", "first_name", "last_name"]
        },
        "range_check": {
            "type": "range_check",
            "column": "amount",
            "min": 0,
            "max": 1000
        },
        "date_format_check": {
            "type": "date_format_check",
            "column": "date",
            "format": "yyyy-MM-dd"
        }
    }