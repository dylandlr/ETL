# Databricks notebook source
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from datetime import datetime

# COMMAND ----------
# Configuration
dbutils.widgets.text("table_path", "", "Table Path")
dbutils.widgets.text("quality_metrics_path", "", "Quality Metrics Path")

table_path = dbutils.widgets.get("table_path")
quality_metrics_path = dbutils.widgets.get("quality_metrics_path")

# COMMAND ----------
def calculate_data_quality_metrics(df):
    try:
        # Get total row count
        total_rows = df.count()
        
        # Calculate null counts for each column
        null_counts = {}
        for column in df.columns:
            null_count = df.filter(col(column).isNull()).count()
            null_counts[column] = null_count
        
        # Calculate distinct value counts
        distinct_counts = {}
        for column in df.columns:
            distinct_count = df.select(column).distinct().count()
            distinct_counts[column] = distinct_count
        
        # Create metrics dataframe
        metrics_data = []
        for column in df.columns:
            metrics_data.append({
                "column_name": column,
                "total_rows": total_rows,
                "null_count": null_counts[column],
                "distinct_count": distinct_counts[column],
                "null_percentage": (null_counts[column] / total_rows) * 100,
                "timestamp": datetime.now()
            })
        
        metrics_df = spark.createDataFrame(metrics_data)
        return metrics_df
    
    except Exception as e:
        print(f"Error calculating data quality metrics: {str(e)}")
        raise

# COMMAND ----------
def write_quality_metrics(metrics_df):
    try:
        metrics_df.write \
            .format("delta") \
            .mode("append") \
            .save(quality_metrics_path)
        print(f"Successfully wrote quality metrics to {quality_metrics_path}")
    except Exception as e:
        print(f"Error writing quality metrics: {str(e)}")
        raise

# COMMAND ----------
def main():
    try:
        # Read the table
        df = spark.read.format("delta").load(table_path)
        
        # Calculate quality metrics
        metrics_df = calculate_data_quality_metrics(df)
        
        # Write metrics
        write_quality_metrics(metrics_df)
        
        print("Data quality analysis completed successfully")
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise

# COMMAND ----------
if __name__ == "__main__":
    main()
