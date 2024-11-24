# Databricks notebook source
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *

# COMMAND ----------
# Configuration
dbutils.widgets.text("input_path", "", "Input Path")
dbutils.widgets.text("output_path", "", "Output Path")

input_path = dbutils.widgets.get("input_path")
output_path = dbutils.widgets.get("output_path")

# COMMAND ----------
# Read data from input path
def read_data():
    try:
        df = spark.read.format("delta").load(input_path)
        print(f"Successfully read data from {input_path}")
        return df
    except Exception as e:
        print(f"Error reading data: {str(e)}")
        raise

# COMMAND ----------
# Transform data
def transform_data(df):
    try:
        # Add processing timestamp
        df_transformed = df.withColumn("processed_timestamp", current_timestamp())
        
        # Add data quality checks
        df_transformed = df_transformed.dropna()
        
        # Add any specific transformations here
        
        print("Successfully transformed data")
        return df_transformed
    except Exception as e:
        print(f"Error transforming data: {str(e)}")
        raise

# COMMAND ----------
# Write data to output path
def write_data(df):
    try:
        df.write \
          .format("delta") \
          .mode("overwrite") \
          .save(output_path)
        print(f"Successfully wrote data to {output_path}")
    except Exception as e:
        print(f"Error writing data: {str(e)}")
        raise

# COMMAND ----------
# Main execution
def main():
    try:
        df = read_data()
        df_transformed = transform_data(df)
        write_data(df_transformed)
        print("Data transformation completed successfully")
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise

# COMMAND ----------
if __name__ == "__main__":
    main()
