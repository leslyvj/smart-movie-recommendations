"""
Utility functions for Movie Recommendation System
Contains helper functions used across multiple modules
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, sqrt, stddev
import os
import sys
import time
import shutil
from config import SPARK_CONFIG

def create_spark_session(app_name=None):
    """
    Create and configure a Spark session
    
    Args:
        app_name: Optional custom application name
        
    Returns:
        SparkSession object
    """
    print("="*80)
    print("INITIALIZING SPARK SESSION")
    print("="*80)
    
    # Set environment variables for Windows compatibility
    os.environ['SPARK_LOCAL_HOSTNAME'] = 'localhost'
    
    # FIX: Set Python executable paths to use the same Python version
    python_exe = sys.executable
    os.environ['PYSPARK_PYTHON'] = python_exe
    os.environ['PYSPARK_DRIVER_PYTHON'] = python_exe
    
    print(f"Using Python: {python_exe}")
    print(f"Python Version: {sys.version}")
    
    # Build Spark session
    builder = SparkSession.builder
    
    # Apply all configurations
    for key, value in SPARK_CONFIG.items():
        if key == "spark.app.name" and app_name:
            builder = builder.config(key, app_name)
        else:
            builder = builder.config(key, value)
    
    spark = builder.getOrCreate()
    
    # Set log level
    spark.sparkContext.setLogLevel("ERROR")
    
    print(f"✓ Spark Session Created Successfully")
    print(f"  Spark Version: {spark.version}")
    print(f"  Master: {spark.sparkContext.master}")
    print(f"  App Name: {spark.sparkContext.appName}")
    print(f"  Driver Memory: {SPARK_CONFIG['spark.driver.memory']}")
    print("\n")
    
    return spark


def print_header(title, char="="):
    """
    Print a formatted header
    
    Args:
        title: Header title
        char: Character to use for border
    """
    print("\n")
    print(char * 80)
    print(title)
    print(char * 80)
    print("\n")


def print_section(title):
    """
    Print a formatted section header
    
    Args:
        title: Section title
    """
    print("-" * 80)
    print(title)
    print("-" * 80)


def calculate_statistics(df, column_name):
    """
    Calculate basic statistics for a numeric column
    
    Args:
        df: Spark DataFrame
        column_name: Name of the column to analyze
        
    Returns:
        Dictionary with statistics
    """
    # Use built-in stddev instead of manual calculation to avoid nested aggregates
    stats = df.agg(
        avg(column_name).alias("mean"),
        stddev(column_name).alias("std_dev")
    ).collect()[0]
    
    min_val = df.agg({column_name: "min"}).collect()[0][0]
    max_val = df.agg({column_name: "max"}).collect()[0][0]
    
    return {
        "min": min_val if min_val is not None else 0,
        "max": max_val if max_val is not None else 0,
        "mean": stats["mean"] if stats["mean"] is not None else 0,
        "std_dev": stats["std_dev"] if stats["std_dev"] is not None else 0
    }


def save_dataframe(df, filename, format="csv"):
    """
    Save a Spark DataFrame to file (Windows compatible - using Pandas)
    
    Args:
        df: Spark DataFrame
        filename: Output filename
        format: File format (csv, parquet, json)
    """
    output_dir = "./output"
    final_path = os.path.join(output_dir, filename)
    
    try:
        if format == "csv":
            # Convert to Pandas and save (much simpler on Windows)
            pandas_df = df.toPandas()
            
            # Remove old file if exists
            if os.path.exists(final_path):
                os.remove(final_path)
            
            # Save to CSV
            pandas_df.to_csv(final_path, index=False)
            
            print(f"✓ Saved to: {final_path}")
                
        elif format == "parquet":
            df.write.mode("overwrite").parquet(final_path)
            print(f"✓ Saved to: {final_path}")
            
        elif format == "json":
            df.write.mode("overwrite").json(final_path)
            print(f"✓ Saved to: {final_path}")
        
    except Exception as e:
        print(f"✗ Error saving file: {str(e)}")


def load_dataframe(spark, filename):
    """
    Load a DataFrame from file (Windows compatible)
    
    Args:
        spark: SparkSession object
        filename: Filename to load
        
    Returns:
        DataFrame or None if file doesn't exist
    """
    file_path = os.path.join("./output", filename)
    
    try:
        if os.path.exists(file_path) and os.path.isfile(file_path):
            # Single CSV file - load directly
            df = spark.read.csv(file_path, header=True, inferSchema=True)
            return df
        else:
            print(f"⚠ Warning: File not found: {file_path}")
            return None
            
    except Exception as e:
        print(f"✗ Error loading file {filename}: {str(e)}")
        return None


def timer(func):
    """
    Decorator to measure execution time
    
    Args:
        func: Function to time
        
    Returns:
        Wrapped function with timing
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        print(f"⏱ Execution time: {elapsed_time:.2f} seconds")
        return result
    return wrapper


def create_output_directory():
    """Create output directory if it doesn't exist"""
    output_dir = "./output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"✓ Created output directory: {output_dir}")
    return output_dir