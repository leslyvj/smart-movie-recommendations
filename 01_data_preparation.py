"""
Part 1-3: Data Preparation Module
Handles data loading, cleaning, and matrix creation
"""

from pyspark.sql.functions import (
    col, avg, sum as spark_sum, count, first, countDistinct,
    round as spark_round, sqrt
)
from pyspark.sql.types import IntegerType, FloatType
import time

from utils import (
    create_spark_session, print_header, print_section,
    calculate_statistics, save_dataframe, create_output_directory
)
from config import DATA_PATH, DISPLAY_ROWS


def load_and_explore_data(spark):
    """
    Load ratings data and display basic statistics
    
    Args:
        spark: SparkSession object
        
    Returns:
        ratings_df: Raw ratings DataFrame
    """
    print_header("PART 1-2: DATA LOADING AND EXPLORATION")
    
    print_section("LOADING RATINGS DATA")
    
    start_time = time.time()
    
    try:
        ratings_df = spark.read.csv(
            DATA_PATH,
            header=True,
            inferSchema=True
        )
        
        # Drop timestamp and cast types
        ratings_df = ratings_df.drop("timestamp") \
            .withColumn("userId", col("userId").cast(IntegerType())) \
            .withColumn("movieId", col("movieId").cast(IntegerType())) \
            .withColumn("rating", col("rating").cast(FloatType()))
        
        load_time = time.time() - start_time
        print(f"✓ Data loaded successfully in {load_time:.2f} seconds")
        print(f"  File: {DATA_PATH}")
        print(f"  Columns: {', '.join(ratings_df.columns)}")
        print("\n")
        
    except Exception as e:
        print(f"✗ Error loading data: {str(e)}")
        spark.stop()
        exit(1)
    
    # Display basic statistics
    print_section("BASIC DATASET STATISTICS")
    
    total_users = ratings_df.select(countDistinct("userId")).collect()[0][0]
    total_movies = ratings_df.select(countDistinct("movieId")).collect()[0][0]
    total_ratings = ratings_df.count()
    average_rating = ratings_df.select(avg("rating")).collect()[0][0]
    
    print(f"Total Unique Users: {total_users:,}")
    print(f"Total Unique Movies: {total_movies:,}")
    print(f"Total Ratings: {total_ratings:,}")
    print(f"Average Rating: {average_rating:.3f}")
    print(f"Sparsity: {100 * (1 - total_ratings / (total_users * total_movies)):.2f}%")
    print("\n")
    
    # Rating distribution
    print("Rating Distribution:")
    rating_dist = ratings_df.groupBy("rating") \
        .count() \
        .orderBy("rating") \
        .collect()
    
    for row in rating_dist:
        rating_val = row['rating']
        rating_count = row['count']
        percentage = (rating_count / total_ratings) * 100
        bar = "█" * int(percentage / 2)
        print(f"  {rating_val:.1f} ⭐: {rating_count:>8,} ({percentage:>5.2f}%) {bar}")
    
    print("\n")
    
    # Show sample data
    print("Sample Data (First 20 rows):")
    ratings_df.show(DISPLAY_ROWS, truncate=False)
    print("\n")
    
    return ratings_df


def clean_data(ratings_df):
    """
    Clean the ratings data
    
    Args:
        ratings_df: Raw ratings DataFrame
        
    Returns:
        ratings_clean: Cleaned ratings DataFrame
    """
    print_section("DATA CLEANING")
    
    initial_count = ratings_df.count()
    print(f"Before cleaning: {initial_count:,} rows")
    
    # Remove null values
    ratings_clean = ratings_df.filter(
        col("userId").isNotNull() & 
        col("movieId").isNotNull() & 
        col("rating").isNotNull()
    )
    after_nulls = ratings_clean.count()
    print(f"After removing nulls: {after_nulls:,} rows (removed {initial_count - after_nulls:,})")
    
    # Remove duplicates
    ratings_clean = ratings_clean.dropDuplicates(["userId", "movieId"])
    after_duplicates = ratings_clean.count()
    print(f"After removing duplicates: {after_duplicates:,} rows (removed {after_nulls - after_duplicates:,})")
    
    # Filter valid rating range
    ratings_clean = ratings_clean.filter(
        (col("rating") >= 0.5) & (col("rating") <= 5.0)
    )
    final_count = ratings_clean.count()
    print(f"After filtering invalid ratings: {final_count:,} rows (removed {after_duplicates - final_count:,})")
    
    print(f"\n✓ Final cleaned dataset: {final_count:,} rows")
    print(f"  Total rows removed: {initial_count - final_count:,} ({((initial_count - final_count) / initial_count * 100):.2f}%)")
    print("\n")
    
    # Cache for reuse
    ratings_clean.cache()
    ratings_clean.count()  # Trigger caching
    
    return ratings_clean


def create_matrix_representation(ratings_clean):
    """
    Create user-item matrix representation (memory-efficient)
    
    Args:
        ratings_clean: Cleaned ratings DataFrame
        
    Returns:
        Dictionary with matrix information
    """
    print_header("PART 3: USER-ITEM MATRIX ANALYSIS")
    
    print_section("CREATING USER-ITEM MATRIX REPRESENTATION")
    print("Note: Using memory-efficient long-format representation instead of pivot")
    print("\n")
    
    start_time = time.time()
    
    # Get matrix dimensions
    num_users = ratings_clean.select(countDistinct("userId")).collect()[0][0]
    num_movies = ratings_clean.select(countDistinct("movieId")).collect()[0][0]
    final_count = ratings_clean.count()
    total_cells = num_users * num_movies
    non_null_count = final_count
    sparsity = (1 - non_null_count / total_cells) * 100
    
    print(f"✓ Matrix representation created in {time.time() - start_time:.2f} seconds")
    print("\n")
    
    # Display matrix information
    print_section("MATRIX INFORMATION")
    
    print(f"Matrix Dimensions: {num_users:,} users × {num_movies:,} movies")
    print(f"Total Cells: {total_cells:,}")
    print(f"Rated Cells: {non_null_count:,}")
    print(f"Null Cells: {total_cells - non_null_count:,}")
    print(f"Sparsity: {sparsity:.2f}%")
    print(f"Density: {100 - sparsity:.2f}%")
    print("\n")
    
    # Show sample matrix
    print("Sample Matrix (First 5 users, First 10 movies):")
    sample_users = ratings_clean.select("userId").distinct().limit(5).rdd.flatMap(lambda x: x).collect()
    sample_movies = ratings_clean.select("movieId").distinct().limit(10).rdd.flatMap(lambda x: x).collect()
    
    sample_data = ratings_clean.filter(
        col("userId").isin(sample_users) & col("movieId").isin(sample_movies)
    )
    
    sample_pivot = sample_data.groupBy("userId").pivot("movieId", sample_movies).agg(first("rating"))
    sample_pivot.show(truncate=False)
    print("\n")
    
    return {
        "num_users": num_users,
        "num_movies": num_movies,
        "sparsity": sparsity,
        "total_ratings": final_count
    }


def calculate_user_averages(ratings_clean):
    """
    Calculate average ratings per user
    
    Args:
        ratings_clean: Cleaned ratings DataFrame
        
    Returns:
        user_averages_df: DataFrame with user averages
    """
    print_section("CALCULATING USER AVERAGE RATINGS")
    
    start_time = time.time()
    
    user_averages_df = ratings_clean.groupBy("userId") \
        .agg(
            avg("rating").alias("avg_rating"),
            count("rating").alias("num_ratings")
        ) \
        .withColumn("avg_rating", spark_round(col("avg_rating"), 3))
    
    # Cache for performance
    user_averages_df.cache()
    user_avg_count = user_averages_df.count()
    
    avg_time = time.time() - start_time
    
    print(f"✓ User averages calculated in {avg_time:.2f} seconds")
    print(f"  Total users with ratings: {user_avg_count:,}")
    print("\n")
    
    # Display statistics
    print("User Average Rating Statistics:")
    stats = calculate_statistics(user_averages_df, "avg_rating")
    
    print(f"  Minimum Average Rating: {stats['min']:.3f}")
    print(f"  Maximum Average Rating: {stats['max']:.3f}")
    print(f"  Mean of User Averages: {stats['mean']:.3f}")
    print(f"  Standard Deviation: {stats['std_dev']:.3f}")
    print("\n")
    
    # Top and bottom users
    print("Top 5 Users with HIGHEST Average Ratings:")
    user_averages_df.orderBy(col("avg_rating").desc(), col("num_ratings").desc()).show(5, truncate=False)
    
    print("Top 5 Users with LOWEST Average Ratings:")
    user_averages_df.orderBy(col("avg_rating").asc(), col("num_ratings").desc()).show(5, truncate=False)
    
    print("\n")
    
    # Save to file
    save_dataframe(user_averages_df, "user_averages.csv")
    
    return user_averages_df


def main():
    """Main execution function"""
    # Create output directory
    create_output_directory()
    
    # Create Spark session
    spark = create_spark_session("DataPreparation")
    
    try:
        # Step 1: Load and explore data
        ratings_df = load_and_explore_data(spark)
        
        # Step 2: Clean data
        ratings_clean = clean_data(ratings_df)
        
        # Step 3: Create matrix representation
        matrix_info = create_matrix_representation(ratings_clean)
        
        # Step 4: Calculate user averages
        user_averages_df = calculate_user_averages(ratings_clean)
        
        # Save cleaned data
        print_section("SAVING CLEANED DATA")
        save_dataframe(ratings_clean, "ratings_clean.csv")
        
        # Summary
        print_header("DATA PREPARATION COMPLETE - SUMMARY")
        
        print("DataFrames Ready:")
        print("  1. ratings_clean:")
        print(f"     - Rows: {matrix_info['total_ratings']:,}")
        print(f"     - Columns: userId, movieId, rating")
        print(f"     - Cached: Yes ✓")
        
        print("\n  2. user_averages_df:")
        print(f"     - Rows: {matrix_info['num_users']:,} users")
        print(f"     - Columns: userId, avg_rating, num_ratings")
        print(f"     - Cached: Yes ✓")
        
        print("\n  3. Matrix Information:")
        print(f"     - Dimensions: {matrix_info['num_users']:,} × {matrix_info['num_movies']:,}")
        print(f"     - Sparsity: {matrix_info['sparsity']:.2f}%")
        
        print("\n✅ Data preparation completed successfully!")
        print("📁 Output files saved to: ./output/")
        print("\n")
        
    finally:
        # Stop Spark session
        spark.stop()
        print("✓ Spark session stopped\n")


if __name__ == "__main__":
    main()