# Databricks notebook source
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from datetime import datetime
import json

# COMMAND ----------
# Configuration
dbutils.widgets.text("table_path", "", "Table Path")
dbutils.widgets.text("validation_rules_path", "", "Validation Rules Path")
dbutils.widgets.text("validation_results_path", "", "Validation Results Path")

table_path = dbutils.widgets.get("table_path")
validation_rules_path = dbutils.widgets.get("validation_rules_path")
validation_results_path = dbutils.widgets.get("validation_results_path")

# COMMAND ----------
def load_validation_rules():
    """Load validation rules from JSON file"""
    try:
        rules_df = spark.read.json(validation_rules_path)
        return rules_df.collect()
    except Exception as e:
        print(f"Error loading validation rules: {str(e)}")
        raise

# COMMAND ----------
def apply_validation_rule(df, rule):
    """Apply a single validation rule to the dataframe"""
    try:
        rule_type = rule.rule_type
        column = rule.column
        
        if rule_type == "not_null":
            invalid_count = df.filter(col(column).isNull()).count()
            is_valid = invalid_count == 0
            
        elif rule_type == "unique":
            total_count = df.count()
            distinct_count = df.select(column).distinct().count()
            is_valid = total_count == distinct_count
            invalid_count = total_count - distinct_count
            
        elif rule_type == "range":
            min_value = rule.min_value
            max_value = rule.max_value
            invalid_count = df.filter(
                (col(column) < min_value) | (col(column) > max_value)
            ).count()
            is_valid = invalid_count == 0
            
        elif rule_type == "regex":
            pattern = rule.pattern
            invalid_count = df.filter(
                ~col(column).rlike(pattern)
            ).count()
            is_valid = invalid_count == 0
            
        elif rule_type == "referential_integrity":
            ref_table = spark.read.format("delta").load(rule.reference_table)
            ref_column = rule.reference_column
            invalid_count = df.join(
                ref_table,
                df[column] == ref_table[ref_column],
                "left_anti"
            ).count()
            is_valid = invalid_count == 0
            
        else:
            raise ValueError(f"Unsupported rule type: {rule_type}")
            
        return {
            "rule_id": rule.rule_id,
            "rule_type": rule_type,
            "column": column,
            "is_valid": is_valid,
            "invalid_count": invalid_count,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"Error applying validation rule: {str(e)}")
        raise

# COMMAND ----------
def validate_data(df, rules):
    """Apply all validation rules to the dataframe"""
    try:
        results = []
        for rule in rules:
            result = apply_validation_rule(df, rule)
            results.append(result)
        
        return results
    except Exception as e:
        print(f"Error in data validation: {str(e)}")
        raise

# COMMAND ----------
def save_validation_results(results):
    """Save validation results to Delta table"""
    try:
        results_df = spark.createDataFrame(results)
        
        results_df.write \
            .format("delta") \
            .mode("append") \
            .save(validation_results_path)
            
        print(f"Successfully saved validation results to {validation_results_path}")
        
        # Raise alert for failed validations
        failed_validations = results_df.filter(~col("is_valid"))
        if failed_validations.count() > 0:
            failed_rules = failed_validations.select("rule_id", "rule_type", "column", "invalid_count").collect()
            alert_message = "Data validation failures detected:\n"
            for rule in failed_rules:
                alert_message += f"Rule {rule.rule_id} ({rule.rule_type}) failed for column {rule.column}. Invalid records: {rule.invalid_count}\n"
            raise Exception(alert_message)
            
    except Exception as e:
        print(f"Error saving validation results: {str(e)}")
        raise

# COMMAND ----------
def main():
    try:
        # Read the data
        df = spark.read.format("delta").load(table_path)
        
        # Load validation rules
        rules = load_validation_rules()
        
        # Validate data
        results = validate_data(df, rules)
        
        # Save results
        save_validation_results(results)
        
        print("Data validation completed successfully")
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise

# COMMAND ----------
if __name__ == "__main__":
    main()
