import pytest
from pyspark.sql.functions import col, concat, upper, to_date, year, month, day, when, sum, count, avg, lit
from decimal import Decimal

def test_basic_transformations(spark, test_data):
    """Test basic column transformations."""
    # Apply transformations
    transformed_df = test_data.withColumn(
        "full_name", 
        concat(col("first_name"), lit(" "), col("last_name"))
    ).withColumn(
        "upper_name", 
        upper(col("full_name"))
    )
    
    # Get results
    result = transformed_df.select("full_name", "upper_name").first()
    
    # Assertions
    assert result["full_name"] == "John Doe", "Name concatenation failed"
    assert result["upper_name"] == "JOHN DOE", "Uppercase transformation failed"

def test_date_transformations(spark, test_data):
    """Test date-related transformations."""
    # Apply transformations
    transformed_df = test_data.withColumn(
        "date", 
        to_date(col("date"))
    ).withColumn(
        "year", 
        year(col("date"))
    ).withColumn(
        "month", 
        month(col("date"))
    ).withColumn(
        "day", 
        day(col("date"))
    )
    
    # Get results
    result = transformed_df.select("year", "month", "day").first()
    
    # Assertions
    assert result["year"] == 2023, "Year extraction failed"
    assert result["month"] == 1, "Month extraction failed"
    assert result["day"] == 1, "Day extraction failed"

def test_numerical_transformations(spark, test_data):
    """Test numerical transformations."""
    # Apply transformations
    transformed_df = test_data.withColumn(
        "amount_doubled", 
        col("amount") * 2
    ).withColumn(
        "amount_categorized",
        when(col("amount") <= 100, "low")
        .when(col("amount") <= 200, "medium")
        .otherwise("high")
    )
    
    # Get results
    result = transformed_df.select("amount", "amount_doubled", "amount_categorized").first()
    
    # Assertions
    assert result["amount_doubled"] == 200.0, "Numerical multiplication failed"
    assert result["amount_categorized"] == "low", "Amount categorization failed"

def test_aggregation_transformations(spark, test_data):
    """Test aggregation transformations."""
    # Perform aggregations
    agg_df = test_data.groupBy("date").agg(
        sum("amount").alias("total_amount"),
        count("*").alias("record_count"),
        avg("amount").alias("avg_amount")
    )
    
    # Get results
    result = agg_df.orderBy("date").first()
    
    # Assertions
    assert result["total_amount"] == 100.0, "Sum aggregation failed"
    assert result["record_count"] == 1, "Count aggregation failed"
    assert result["avg_amount"] == 100.0, "Average aggregation failed"

def test_complex_transformations(spark, test_data):
    """Test complex transformations combining multiple operations."""
    # Apply complex transformations
    transformed_df = test_data.withColumn(
        "date", 
        to_date(col("date"))
    ).withColumn(
        "full_name",
        concat(upper(col("first_name")), lit(" "), upper(col("last_name")))
    ).withColumn(
        "amount_category",
        when(col("amount") <= 100, "low")
        .when(col("amount") <= 200, "medium")
        .otherwise("high")
    ).groupBy("amount_category").agg(
        count("*").alias("category_count"),
        sum("amount").alias("category_total")
    )
    
    # Get results for 'low' category
    result = transformed_df.filter(col("amount_category") == "low").first()
    
    # Assertions
    assert result["category_count"] == 1, "Complex transformation count failed"
    assert result["category_total"] == 100.0, "Complex transformation sum failed"