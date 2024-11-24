import pytest
from pyspark.sql.functions import col, lit

def test_null_check(spark, test_data):
    """Test null value checks."""
    # Create test data with nulls
    data_with_nulls = test_data.withColumn("first_name", lit(None))
    
    # Count nulls
    null_count = data_with_nulls.filter(col("first_name").isNull()).count()
    
    # Assertions
    assert null_count > 0, "No null values found when they should exist"
    assert null_count == 3, "Unexpected number of null values"

def test_range_check(spark, test_data, quality_rules):
    """Test numerical range validation."""
    rule = quality_rules["range_check"]
    
    # Test data within range
    valid_count = test_data.filter(
        (col(rule["column"]) >= rule["min"]) & 
        (col(rule["column"]) <= rule["max"])
    ).count()
    
    # Test data outside range
    invalid_data = test_data.withColumn("amount", lit(2000.0))
    invalid_count = invalid_data.filter(
        (col(rule["column"]) < rule["min"]) | 
        (col(rule["column"]) > rule["max"])
    ).count()
    
    # Assertions
    assert valid_count == 3, "Not all valid values were counted"
    assert invalid_count == 3, "Not all invalid values were counted"

def test_date_format_check(spark, test_data, quality_rules):
    """Test date format validation."""
    rule = quality_rules["date_format_check"]
    
    # Test valid dates
    valid_count = test_data.count()
    
    # Test invalid dates
    invalid_data = test_data.withColumn("date", lit("2023/01/01"))
    
    # Assertions
    assert valid_count == 3, "Not all valid dates were counted"
    with pytest.raises(Exception):
        # This should raise an exception due to invalid date format
        invalid_data.select(col("date").cast("date")).collect()

def test_composite_quality_check(spark, test_data, quality_rules):
    """Test multiple quality rules together."""
    # Apply all quality rules
    df = test_data
    
    # Null check
    null_rule = quality_rules["null_check"]
    null_counts = {col: df.filter(col(col).isNull()).count() 
                  for col in null_rule["columns"]}
    
    # Range check
    range_rule = quality_rules["range_check"]
    range_violations = df.filter(
        (col(range_rule["column"]) < range_rule["min"]) | 
        (col(range_rule["column"]) > range_rule["max"])
    ).count()
    
    # Date format check
    date_rule = quality_rules["date_format_check"]
    date_format_violations = 0
    try:
        df.select(col(date_rule["column"]).cast("date")).collect()
    except Exception:
        date_format_violations += 1
    
    # Assertions
    assert all(count == 0 for count in null_counts.values()), "Null values found"
    assert range_violations == 0, "Range violations found"
    assert date_format_violations == 0, "Date format violations found"