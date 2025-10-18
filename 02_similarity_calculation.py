"""
Part 4: Similarity Calculation Module
Calculates user-user similarity using Pearson Correlation
"""

from pyspark.sql.functions import (
    col, count, sum as spark_sum, sqrt, lit, when, avg, abs as spark_abs,
    round as spark_round
)

from utils import (
    create_spark_session, print_header, print_section,
    calculate_statistics, save_dataframe, create_output_directory,
    load_dataframe  # Add this import
)
from config import (
    TARGET_USER_ID, MIN_COMMON_MOVIES, TOP_K_SIMILAR_USERS,
    SIMILARITY_PRECISION, DISPLAY_ROWS
)
import time


def load_prepared_data(spark):
    """
    Load data prepared from previous step
    
    Args:
        spark: SparkSession object
        
    Returns:
        Tuple of (ratings_clean, user_averages_df)
    """
    print_section("LOADING PREPARED DATA")
    
    # Use the new load_dataframe function
    ratings_clean = load_dataframe(spark, "ratings_clean.csv")
    user_averages_df = load_dataframe(spark, "user_averages.csv")
    
    if ratings_clean is None or user_averages_df is None:
        print("✗ Error: Required data files not found")
        print("  Please run 01_data_preparation.py first")
        spark.stop()
        exit(1)
    
    # Cache for reuse
    ratings_clean.cache()
    user_averages_df.cache()
    
    print(f"✓ Loaded ratings_clean: {ratings_clean.count():,} rows")
    print(f"✓ Loaded user_averages: {user_averages_df.count():,} rows")
    print("\n")
    
    return ratings_clean, user_averages_df


def get_target_user_info(ratings_clean, user_averages_df):
    """
    Get information about the target user
    
    Args:
        ratings_clean: Cleaned ratings DataFrame
        user_averages_df: User averages DataFrame
        
    Returns:
        Dictionary with target user information
    """
    print_section("TARGET USER ANALYSIS")
    
    # Get ratings for target user
    target_user_ratings = ratings_clean.filter(col("userId") == TARGET_USER_ID)
    target_user_movies = target_user_ratings.select("movieId").distinct()
    num_rated = target_user_ratings.count()
    
    print(f"Target User ID: {TARGET_USER_ID}")
    print(f"Number of movies rated: {num_rated}")
    print("\nTarget user's ratings:")
    target_user_ratings.orderBy("rating", ascending=False).show(10)
    
    # Get target user's average rating
    target_user_avg = user_averages_df.filter(
        col("userId") == TARGET_USER_ID
    ).select("avg_rating").collect()[0][0]
    
    print(f"Target user's average rating: {target_user_avg:.3f}")
    print("\n")
    
    return {
        "target_user_ratings": target_user_ratings,
        "target_user_movies": target_user_movies,
        "target_user_avg": target_user_avg,
        "num_rated": num_rated
    }


def find_common_movies(ratings_clean, target_info):
    """
    Find users with common movies to target user
    
    Args:
        ratings_clean: Cleaned ratings DataFrame
        target_info: Target user information dictionary
        
    Returns:
        DataFrame with common ratings
    """
    print_section("FINDING USERS WITH COMMON MOVIES")
    
    # Join to find all users who rated the same movies as target user
    common_ratings = ratings_clean.alias("target") \
        .join(
            ratings_clean.alias("other"),
            (col("target.movieId") == col("other.movieId")) & 
            (col("other.userId") != TARGET_USER_ID),
            "inner"
        ) \
        .select(
            col("other.userId").alias("other_userId"),
            col("target.movieId").alias("movieId"),
            col("target.rating").alias("target_rating"),
            col("other.rating").alias("other_rating")
        )
    
    # Count common movies per user
    common_movie_counts = common_ratings.groupBy("other_userId") \
        .agg(count("movieId").alias("common_movies")) \
        .filter(col("common_movies") >= MIN_COMMON_MOVIES)
    
    num_candidate_users = common_movie_counts.count()
    print(f"✓ Found {num_candidate_users:,} users with at least {MIN_COMMON_MOVIES} common movies")
    print("\nTop 10 users by number of common movies:")
    common_movie_counts.orderBy(col("common_movies").desc()).show(10)
    print("\n")
    
    return common_ratings, common_movie_counts


def calculate_pearson_correlation(common_ratings, user_averages_df, target_user_avg):
    """
    Calculate Pearson correlation similarity scores
    
    Args:
        common_ratings: DataFrame with common ratings
        user_averages_df: User averages DataFrame
        target_user_avg: Target user's average rating
        
    Returns:
        DataFrame with similarity scores
    """
    print_section("CALCULATING PEARSON CORRELATION")
    print("Formula: sim(u,v) = Σ[(r_u,i - r̄_u)(r_v,i - r̄_v)] / √[Σ(r_u,i - r̄_u)² × Σ(r_v,i - r̄_v)²]")
    print("\n")
    
    start_time = time.time()
    
    # Join with user averages
    ratings_with_avg = common_ratings \
        .join(
            user_averages_df.select(col("userId"), col("avg_rating").alias("target_avg")),
            common_ratings.other_userId == user_averages_df.userId,
            "inner"
        ) \
        .withColumn("target_user_avg", lit(target_user_avg)) \
        .withColumn("target_dev", col("target_rating") - col("target_user_avg")) \
        .withColumn("other_dev", col("other_rating") - col("target_avg"))
    
    # Calculate Pearson correlation components
    pearson_components = ratings_with_avg.groupBy("other_userId") \
        .agg(
            spark_sum(col("target_dev") * col("other_dev")).alias("numerator"),
            spark_sum(col("target_dev") * col("target_dev")).alias("target_sq_sum"),
            spark_sum(col("other_dev") * col("other_dev")).alias("other_sq_sum"),
            count("movieId").alias("common_movies")
        )
    
    # Calculate final similarity scores
    similarity_scores = pearson_components \
        .withColumn("denominator", sqrt(col("target_sq_sum") * col("other_sq_sum"))) \
        .withColumn("similarity", 
            when(col("denominator") == 0, 0)
            .otherwise(col("numerator") / col("denominator"))
        ) \
        .filter(col("common_movies") >= MIN_COMMON_MOVIES) \
        .select("other_userId", "similarity", "common_movies") \
        .withColumnRenamed("other_userId", "userId") \
        .withColumn("similarity", spark_round(col("similarity"), SIMILARITY_PRECISION))
    
    # Cache similarity scores
    similarity_scores.cache()
    similarity_count = similarity_scores.count()
    
    calc_time = time.time() - start_time
    
    print(f"✓ Calculated similarity scores in {calc_time:.2f} seconds")
    print(f"  Total users with similarity scores: {similarity_count:,}")
    print("\n")
    
    return similarity_scores


def display_similarity_statistics(similarity_scores):
    """
    Display statistics about similarity scores
    
    Args:
        similarity_scores: DataFrame with similarity scores
    """
    print_section("SIMILARITY SCORE STATISTICS")
    
    stats = calculate_statistics(similarity_scores, "similarity")
    
    print(f"Minimum Similarity: {stats['min']:.{SIMILARITY_PRECISION}f}")
    print(f"Maximum Similarity: {stats['max']:.{SIMILARITY_PRECISION}f}")
    print(f"Mean Similarity: {stats['mean']:.{SIMILARITY_PRECISION}f}")
    print(f"Standard Deviation: {stats['std_dev']:.{SIMILARITY_PRECISION}f}")
    print("\n")
    
    print("📸 SCREENSHOT 1: USER-USER SIMILARITY MATRIX")
    print("="*80)
    print(f"Top {DISPLAY_ROWS} Most Similar Users to Target User:")
    similarity_scores.orderBy(col("similarity").desc()).show(DISPLAY_ROWS, truncate=False)
    print("="*80)
    print("\n")
    
    print("Bottom 10 Least Similar Users (but still with common movies):")
    similarity_scores.orderBy(col("similarity").asc()).show(10, truncate=False)
    print("\n")


def select_top_similar_users(similarity_scores):
    """
    Select top K similar users for recommendations
    
    Args:
        similarity_scores: DataFrame with similarity scores
        
    Returns:
        DataFrame with top K similar users
    """
    print_section(f"SELECTING TOP {TOP_K_SIMILAR_USERS} SIMILAR USERS")
    
    # Select top K similar users
    top_similar_users = similarity_scores \
        .orderBy(col("similarity").desc()) \
        .limit(TOP_K_SIMILAR_USERS)
    
    top_similar_users.cache()
    top_k_count = top_similar_users.count()
    
    print(f"✓ Selected top {TOP_K_SIMILAR_USERS} similar users for recommendations")
    print(f"  Actual users selected: {top_k_count}")
    print("\nSelected users:")
    top_similar_users.show(20, truncate=False)
    print("\n")
    
    return top_similar_users


def main():
    """Main execution function"""
    # Create output directory
    create_output_directory()
    
    # Create Spark session
    spark = create_spark_session("SimilarityCalculation")
    
    try:
        print_header("PART 4: USER-USER SIMILARITY CALCULATION (PEARSON CORRELATION)")
        
        # Step 1: Load prepared data
        ratings_clean, user_averages_df = load_prepared_data(spark)
        
        # Step 2: Get target user info
        target_info = get_target_user_info(ratings_clean, user_averages_df)
        
        # Step 3: Find common movies
        common_ratings, common_movie_counts = find_common_movies(ratings_clean, target_info)
        
        # Step 4: Calculate Pearson correlation
        similarity_scores = calculate_pearson_correlation(
            common_ratings, 
            user_averages_df, 
            target_info["target_user_avg"]
        )
        
        # Step 5: Display statistics
        display_similarity_statistics(similarity_scores)
        
        # Step 6: Select top similar users
        top_similar_users = select_top_similar_users(similarity_scores)
        
        # Save results
        print_section("SAVING RESULTS")
        save_dataframe(similarity_scores, "similarity_scores.csv")
        save_dataframe(top_similar_users, "top_similar_users.csv")
        
        # Summary
        print_header("SIMILARITY CALCULATION COMPLETE")
        
        print("✅ Similarity calculation completed successfully!")
        print(f"📁 Output files saved to: ./output/")
        print(f"  - similarity_scores.csv")
        print(f"  - top_similar_users.csv")
        print("\n")
        
    finally:
        # Stop Spark session
        spark.stop()
        print("✓ Spark session stopped\n")


if __name__ == "__main__":
    main()