import os
import pytest
import pandas as pd
from pyspark.sql import SparkSession
from azure.storage.filedatalake import DataLakeServiceClient
from azure.identity import DefaultAzureCredential

class TestDataQuality:
    @pytest.fixture(scope="class")
    def spark(self):
        """Create Spark session"""
        try:
            return SparkSession.builder \
                .appName("DataQualityTests") \
                .config("spark.jars.packages", "org.apache.hadoop:hadoop-azure:3.3.1") \
                .getOrCreate()
        except Exception as e:
            pytest.skip(f"Failed to create Spark session: {str(e)}")

    @pytest.fixture(scope="class")
    def datalake_client(self):
        """Create Data Lake client"""
        try:
            account_url = f"https://{os.getenv('STORAGE_ACCOUNT_NAME')}.dfs.core.windows.net"
            credential = DefaultAzureCredential()
            return DataLakeServiceClient(account_url, credential=credential)
        except Exception as e:
            pytest.skip(f"Failed to create Data Lake client: {str(e)}")

    def test_data_completeness(self, spark, datalake_client):
        """Test data completeness in raw and processed layers"""
        containers = ["raw", "processed"]
        
        for container in containers:
            try:
                file_system_client = datalake_client.get_file_system_client(container)
                paths = file_system_client.get_paths()
                
                for path in paths:
                    if path.name.endswith('.parquet'):
                        # Read data
                        df = spark.read.parquet(f"abfss://{container}@{os.getenv('STORAGE_ACCOUNT_NAME')}.dfs.core.windows.net/{path.name}")
                        
                        # Check for null values
                        null_counts = df.select([pd.isnull(c).sum().alias(c) for c in df.columns])
                        high_null_cols = [col for col in df.columns if null_counts.first()[col] > df.count() * 0.1]
                        
                        assert not high_null_cols, f"High null value columns in {container}/{path.name}: {high_null_cols}"
                        
            except Exception as e:
                pytest.fail(f"Failed to check data completeness in {container}: {str(e)}")

    def test_data_consistency(self, spark, datalake_client):
        """Test data consistency between layers"""
        try:
            # Get raw and processed data
            raw_fs = datalake_client.get_file_system_client("raw")
            processed_fs = datalake_client.get_file_system_client("processed")
            
            raw_files = [p.name for p in raw_fs.get_paths() if p.name.endswith('.parquet')]
            processed_files = [p.name for p in processed_fs.get_paths() if p.name.endswith('.parquet')]
            
            # Compare file counts
            assert len(raw_files) > 0, "No files found in raw container"
            assert len(processed_files) > 0, "No files found in processed container"
            
            # Compare record counts for each corresponding file
            for raw_file in raw_files:
                processed_file = raw_file.replace('raw', 'processed')
                if processed_file in processed_files:
                    raw_df = spark.read.parquet(f"abfss://raw@{os.getenv('STORAGE_ACCOUNT_NAME')}.dfs.core.windows.net/{raw_file}")
                    processed_df = spark.read.parquet(f"abfss://processed@{os.getenv('STORAGE_ACCOUNT_NAME')}.dfs.core.windows.net/{processed_file}")
                    
                    raw_count = raw_df.count()
                    processed_count = processed_df.count()
                    
                    assert processed_count >= raw_count * 0.95, \
                        f"Significant data loss in processing: raw={raw_count}, processed={processed_count}"
                    
        except Exception as e:
            pytest.fail(f"Failed to check data consistency: {str(e)}")

    def test_data_freshness(self, datalake_client):
        """Test data freshness"""
        from datetime import datetime, timedelta
        
        containers = ["raw", "processed"]
        max_age_hours = 24  # Maximum allowed age of data
        
        for container in containers:
            try:
                file_system_client = datalake_client.get_file_system_client(container)
                paths = file_system_client.get_paths()
                
                for path in paths:
                    if path.name.endswith('.parquet'):
                        properties = file_system_client.get_file_client(path.name).get_file_properties()
                        last_modified = properties['last_modified']
                        age = datetime.utcnow() - last_modified
                        
                        assert age <= timedelta(hours=max_age_hours), \
                            f"Data in {container}/{path.name} is too old: {age.total_seconds() / 3600:.1f} hours"
                            
            except Exception as e:
                pytest.fail(f"Failed to check data freshness in {container}: {str(e)}")

    def test_schema_validation(self, spark, datalake_client):
        """Test schema validation and evolution"""
        from pyspark.sql.types import StructType
        
        def get_schema_fingerprint(schema: StructType) -> str:
            """Generate a fingerprint for schema comparison"""
            return ",".join(sorted([f"{field.name}:{field.dataType.simpleString()}" for field in schema.fields]))
        
        containers = ["raw", "processed"]
        schema_registry = {}  # Store schema fingerprints
        
        for container in containers:
            try:
                file_system_client = datalake_client.get_file_system_client(container)
                paths = file_system_client.get_paths()
                
                for path in paths:
                    if path.name.endswith('.parquet'):
                        df = spark.read.parquet(f"abfss://{container}@{os.getenv('STORAGE_ACCOUNT_NAME')}.dfs.core.windows.net/{path.name}")
                        current_fingerprint = get_schema_fingerprint(df.schema)
                        
                        # Check if we've seen this type of file before
                        file_type = path.name.split('/')[-1].split('_')[0]  # Assume filename pattern: type_date.parquet
                        if file_type in schema_registry:
                            assert current_fingerprint == schema_registry[file_type], \
                                f"Schema mismatch detected in {container}/{path.name}"
                        else:
                            schema_registry[file_type] = current_fingerprint
                            
            except Exception as e:
                pytest.fail(f"Failed to validate schema in {container}: {str(e)}")

    def test_data_quality_metrics(self, spark, datalake_client):
        """Test data quality metrics"""
        containers = ["raw", "processed"]
        
        for container in containers:
            try:
                file_system_client = datalake_client.get_file_system_client(container)
                paths = file_system_client.get_paths()
                
                for path in paths:
                    if path.name.endswith('.parquet'):
                        df = spark.read.parquet(f"abfss://{container}@{os.getenv('STORAGE_ACCOUNT_NAME')}.dfs.core.windows.net/{path.name}")
                        
                        # Calculate quality metrics
                        total_rows = df.count()
                        metrics = {
                            'total_rows': total_rows,
                            'null_percentages': {},
                            'distinct_counts': {},
                            'min_max_values': {}
                        }
                        
                        for col in df.columns:
                            # Null percentage
                            null_count = df.filter(df[col].isNull()).count()
                            metrics['null_percentages'][col] = (null_count / total_rows) * 100
                            
                            # Distinct count
                            metrics['distinct_counts'][col] = df.select(col).distinct().count()
                            
                            # Min/Max for numeric columns
                            if str(df.schema[col].dataType) in ['IntegerType', 'LongType', 'DoubleType']:
                                min_max = df.agg({col: 'min', col: 'max'}).collect()[0]
                                metrics['min_max_values'][col] = {'min': min_max[0], 'max': min_max[1]}
                        
                        # Assert quality thresholds
                        for col, null_pct in metrics['null_percentages'].items():
                            assert null_pct <= 10, f"High null percentage ({null_pct}%) in {container}/{path.name}, column: {col}"
                        
                        for col, distinct_count in metrics['distinct_counts'].items():
                            if distinct_count == 1:
                                pytest.warn(f"Single value column detected in {container}/{path.name}, column: {col}")
                            
            except Exception as e:
                pytest.fail(f"Failed to calculate quality metrics in {container}: {str(e)}")
