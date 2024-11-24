# test_pipeline.py

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_date, year, month, day, when, sum, count, lit
import pytest
import json
import os

def test_end_to_end_pipeline(spark, test_data, quality_rules, env_vars):
    """Test the complete ETL pipeline end-to-end."""
    
    # Step 1: Data Extraction
    try:
        input_df = test_data
        assert input_df.count() > 0, "Failed to extract test data"
        
        # Verify schema
        required_columns = ["id", "first_name", "last_name", "date", "amount"]
        assert all(col in input_df.columns for col in required_columns), "Missing required columns"
    except Exception as e:
        pytest.fail(f"Data extraction failed: {str(e)}")

    # Step 2: Data Transformation
    try:
        transformed_df = input_df.withColumn(
            "date", 
            to_date(col("date"))
        ).withColumn(
            "year", 
            year(col("date"))
        ).withColumn(
            "month", 
            month(col("date"))
        ).withColumn(
            "amount_category",
            when(col("amount") <= 100, "low")
            .when(col("amount") <= 200, "medium")
            .otherwise("high")
        )
        
        assert transformed_df.count() == input_df.count(), "Row count mismatch after transformation"
    except Exception as e:
        pytest.fail(f"Data transformation failed: {str(e)}")

    # Step 3: Data Quality Checks
    try:
        # Null checks
        null_rule = quality_rules["null_check"]
        for column in null_rule["columns"]:
            null_count = transformed_df.filter(col(column).isNull()).count()
            assert null_count == 0, f"Found null values in column {column}"

        # Range check
        range_rule = quality_rules["range_check"]
        range_violations = transformed_df.filter(
            (col(range_rule["column"]) < range_rule["min"]) | 
            (col(range_rule["column"]) > range_rule["max"])
        ).count()
        assert range_violations == 0, "Found values outside acceptable range"

        # Date format check
        date_rule = quality_rules["date_format_check"]
        transformed_df.select(col(date_rule["column"]).cast("date")).collect()
    except Exception as e:
        pytest.fail(f"Data quality checks failed: {str(e)}")

    # Step 4: Aggregations
    try:
        agg_df = transformed_df.groupBy("year", "month").agg(
            count("*").alias("record_count"),
            sum("amount").alias("total_amount")
        )
        
        # Verify aggregations
        result = agg_df.filter(
            (col("year") == 2023) & 
            (col("month") == 1)
        ).first()
        
        assert result is not None, "Aggregation result not found"
        assert result["record_count"] > 0, "No records found in aggregation"
        assert result["total_amount"] > 0, "Total amount is zero"
    except Exception as e:
        pytest.fail(f"Data aggregation failed: {str(e)}")

    # Step 5: Output Validation
    try:
        # Verify final schema
        expected_columns = [
            "id", "first_name", "last_name", "date", "amount",
            "year", "month", "amount_category"
        ]
        assert all(col in transformed_df.columns for col in expected_columns), \
            "Missing columns in final output"
            
        # Verify data integrity
        final_count = transformed_df.count()
        assert final_count == input_df.count(), \
            f"Final row count {final_count} doesn't match input count {input_df.count()}"
    except Exception as e:
        pytest.fail(f"Output validation failed: {str(e)}")

def test_error_handling(spark, test_data):
    """Test error handling in the pipeline."""
    
    # Test invalid date format
    try:
        test_data.withColumn("date", lit("invalid-date")).select(
            to_date(col("date"))
        ).collect()
        pytest.fail("Expected error for invalid date format")
    except Exception as e:
        assert "invalid-date" in str(e).lower(), "Unexpected error message"

    # Test division by zero
    try:
        test_data.withColumn("result", col("amount") / lit(0)).collect()
        pytest.fail("Expected error for division by zero")
    except Exception as e:
        assert "division by zero" in str(e).lower(), "Unexpected error message"

    # Test missing column
    try:
        test_data.select("non_existent_column").collect()
        pytest.fail("Expected error for missing column")
    except Exception as e:
        assert "non_existent_column" in str(e).lower(), "Unexpected error message"