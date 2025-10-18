"""
Part 5: Rating Prediction Module
Predicts ratings for unseen movies using collaborative filtering
"""

from pyspark.sql.functions import (
    col, count, sum as spark_sum, avg, lit, when,
    abs as spark_abs, round as spark_round
)

from utils import (
    create_spark_session, print_header, print_section,
    calculate_statistics, save_dataframe, create_output_directory,
    load_dataframe  # Add this import
)
from config import (
    TARGET_USER_ID, RATING_PRECISION, DISPLAY_ROWS
)
import time


def load_similarity_data(spark):
    """
    Load similarity calculation results
    
    Args:
        spark: SparkSession object
        
    Returns:
        Tuple of DataFrames
    """
    print_section("LOADING SIMILARITY DATA")
    
    # Use the new load_dataframe function
    ratings_clean = load_dataframe(spark, "ratings_clean.csv")
    user_averages_df = load_dataframe(spark, "user_averages.csv")
    top_similar_users = load_dataframe(spark, "top_similar_users.csv")
    
    if ratings_clean is None or user_averages_df is None or top_similar_users is None:
        print("✗ Error: Required data files not found")
        print("  Please run 02_similarity_calculation.py first")
        spark.stop()
        exit(1)
    
    # Cache for reuse
    ratings_clean.cache()
    user_averages_df.cache()
    top_similar_users.cache()
    
    print(f"✓ Loaded ratings_clean: {ratings_clean.count():,} rows")
    print(f"✓ Loaded user_averages: {user_averages_df.count():,} rows")
    print(f"✓ Loaded top_similar_users: {top_similar_users.count():,} rows")
    print("\n")
    
    return ratings_clean, user_averages_df, top_similar_users


def identify_unseen_movies(ratings_clean, top_similar_users):
    """
    Identify movies not yet rated by target user
    
    Args:
        ratings_clean: Cleaned ratings DataFrame
        top_similar_users: Top similar users DataFrame
        
    Returns:
        DataFrame with unseen movies
    """
    print_section("IDENTIFYING UNSEEN MOVIES")
    
    # Get movies rated by target user
    target_user_movies = ratings_clean \
        .filter(col("userId") == TARGET_USER_ID) \
        .select("movieId") \
        .distinct()
    
    # Get all movies rated by similar users
    similar_users_movies = ratings_clean \
        .join(top_similar_users, ratings_clean.userId == top_similar_users.userId, "inner") \
        .select(ratings_clean.movieId) \
        .distinct()
    
    # Exclude movies already rated by target user
    unseen_movies = similar_users_movies \
        .join(target_user_movies, "movieId", "left_anti")
    
    unseen_count = unseen_movies.count()
    print(f"✓ Found {unseen_count:,} unseen movies (rated by similar users but not by target user)")
    print("\n")
    
    return unseen_movies


def calculate_predictions(ratings_clean, user_averages_df, top_similar_users, unseen_movies):
    """
    Calculate predicted ratings using collaborative filtering formula
    
    Args:
        ratings_clean: Cleaned ratings DataFrame
        user_averages_df: User averages DataFrame
        top_similar_users: Top similar users DataFrame
        unseen_movies: Unseen movies DataFrame
        
    Returns:
        DataFrame with predicted ratings
    """
    print_section("CALCULATING PREDICTED RATINGS")
    print("Formula: p(u,i) = r̄_u + [Σ(sim(u,v) × (r_v,i - r̄_v))] / [Σ|sim(u,v)|]")
    print("\nWhere:")
    print("  p(u,i) = predicted rating for user u on item i")
    print("  r̄_u = average rating by user u")
    print("  sim(u,v) = similarity between users u and v")
    print("  r_v,i = rating by user v on item i")
    print("  r̄_v = average rating by user v")
    print("\n")
    
    start_time = time.time()
    
    # Get target user's average rating
    target_user_avg = user_averages_df.filter(
        col("userId") == TARGET_USER_ID
    ).select("avg_rating").collect()[0][0]
    
    print(f"Target user average rating: {target_user_avg:.3f}")
    print("\n")
    
    # Get ratings from similar users for unseen movies
    candidate_ratings = ratings_clean \
        .join(top_similar_users, ratings_clean.userId == top_similar_users.userId, "inner") \
        .join(unseen_movies, "movieId", "inner") \
        .join(
            user_averages_df.select(col("userId"), col("avg_rating").alias("user_avg")),
            ratings_clean.userId == user_averages_df.userId,
            "inner"
        ) \
        .select(
            ratings_clean.movieId,
            ratings_clean.userId,
            ratings_clean.rating,
            col("similarity"),
            col("user_avg")
        ) \
        .withColumn("rating_deviation", col("rating") - col("user_avg")) \
        .withColumn("weighted_deviation", col("similarity") * col("rating_deviation"))
    
    # Aggregate predictions per movie
    prediction_aggregates = candidate_ratings.groupBy("movieId") \
        .agg(
            spark_sum("weighted_deviation").alias("sum_weighted_dev"),
            spark_sum(spark_abs(col("similarity"))).alias("sum_abs_similarity"),
            count("userId").alias("num_raters"),
            avg("similarity").alias("avg_similarity")
        )
    
    # Calculate final predicted ratings
    predicted_ratings = prediction_aggregates \
        .withColumn("predicted_rating",
            lit(target_user_avg) + (col("sum_weighted_dev") / col("sum_abs_similarity"))
        ) \
        .withColumn("predicted_rating", 
            when(col("predicted_rating") > 5.0, 5.0)
            .when(col("predicted_rating") < 0.5, 0.5)
            .otherwise(col("predicted_rating"))
        ) \
        .withColumn("predicted_rating", spark_round(col("predicted_rating"), RATING_PRECISION)) \
        .withColumn("avg_similarity", spark_round(col("avg_similarity"), 4)) \
        .select("movieId", "predicted_rating", "num_raters", "avg_similarity")
    
    # Cache predictions
    predicted_ratings.cache()
    num_predictions = predicted_ratings.count()
    
    pred_time = time.time() - start_time
    
    print(f"✓ Calculated {num_predictions:,} predicted ratings in {pred_time:.2f} seconds")
    print("\n")
    
    return predicted_ratings, target_user_avg


def display_prediction_statistics(predicted_ratings):
    """
    Display statistics about predicted ratings
    
    Args:
        predicted_ratings: DataFrame with predicted ratings
    """
    print_section("PREDICTION STATISTICS")
    
    stats = calculate_statistics(predicted_ratings, "predicted_rating")
    
    print(f"Minimum Predicted Rating: {stats['min']:.{RATING_PRECISION}f}")
    print(f"Maximum Predicted Rating: {stats['max']:.{RATING_PRECISION}f}")
    print(f"Mean Predicted Rating: {stats['mean']:.3f}")
    print(f"Standard Deviation: {stats['std_dev']:.3f}")
    print("\n")
    
    print("📸 SCREENSHOT 2: SAMPLE PREDICTED RATINGS")
    print("="*80)
    print(f"Top {DISPLAY_ROWS} Highest Predicted Ratings:")
    predicted_ratings.orderBy(col("predicted_rating").desc()).show(DISPLAY_ROWS, truncate=False)
    print("="*80)
    print("\n")
    
    # Show distribution
    print("Distribution of Predicted Ratings:")
    predicted_ratings.groupBy("predicted_rating") \
        .count() \
        .orderBy("predicted_rating", ascending=False) \
        .show(20)
    print("\n")


def main():
    """Main execution function"""
    # Create output directory
    create_output_directory()
    
    # Create Spark session
    spark = create_spark_session("RatingPrediction")
    
    try:
        print_header("PART 5: RATING PREDICTION (COLLABORATIVE FILTERING)")
        
        # Step 1: Load similarity data
        ratings_clean, user_averages_df, top_similar_users = load_similarity_data(spark)
        
        # Step 2: Identify unseen movies
        unseen_movies = identify_unseen_movies(ratings_clean, top_similar_users)
        
        # Step 3: Calculate predictions
        predicted_ratings, target_user_avg = calculate_predictions(
            ratings_clean, 
            user_averages_df, 
            top_similar_users, 
            unseen_movies
        )
        
        # Step 4: Display statistics
        display_prediction_statistics(predicted_ratings)
        
        # Save results
        print_section("SAVING RESULTS")
        save_dataframe(predicted_ratings, "predicted_ratings.csv")
        
        # Summary
        print_header("RATING PREDICTION COMPLETE")
        
        print("✅ Rating prediction completed successfully!")
        print(f"📁 Output file saved to: ./output/predicted_ratings.csv")
        print("\n")
        
    finally:
        # Stop Spark session
        spark.stop()
        print("✓ Spark session stopped\n")


if __name__ == "__main__":
    main()