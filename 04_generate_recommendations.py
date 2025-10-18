"""
Part 6: Generate Recommendations Module
Generates final movie recommendations with metadata
"""

from pyspark.sql.functions import col
from pyspark.sql.types import IntegerType

from utils import (
    create_spark_session, print_header, print_section,
    save_dataframe, create_output_directory,
    load_dataframe  # Add this import
)
from config import (
    TARGET_USER_ID, TOP_N_RECOMMENDATIONS, MOVIES_PATH,
    DISPLAY_ROWS
)


def load_prediction_data(spark):
    """
    Load rating prediction results
    
    Args:
        spark: SparkSession object
        
    Returns:
        Tuple of DataFrames
    """
    print_section("LOADING PREDICTION DATA")
    
    # Use the new load_dataframe function
    ratings_clean = load_dataframe(spark, "ratings_clean.csv")
    predicted_ratings = load_dataframe(spark, "predicted_ratings.csv")
    
    if ratings_clean is None or predicted_ratings is None:
        print("✗ Error: Required data files not found")
        print("  Please run 03_rating_prediction.py first")
        spark.stop()
        exit(1)
    
    # Cache for reuse
    ratings_clean.cache()
    predicted_ratings.cache()
    
    print(f"✓ Loaded ratings_clean: {ratings_clean.count():,} rows")
    print(f"✓ Loaded predicted_ratings: {predicted_ratings.count():,} rows")
    print("\n")
    
    return ratings_clean, predicted_ratings


def load_movie_metadata(spark):
    """
    Load movie names and metadata
    
    Args:
        spark: SparkSession object
        
    Returns:
        DataFrame with movie metadata or None
    """
    print_section("LOADING MOVIE METADATA")
    
    try:
        movies_df = spark.read.csv(
            MOVIES_PATH,
            header=True,
            inferSchema=True
        )
        
        movies_df = movies_df.select(
            col("movieId").cast(IntegerType()),
            col("title"),
            col("genres")
        )
        
        movie_count = movies_df.count()
        print(f"✓ Loaded {movie_count:,} movies from {MOVIES_PATH}")
        print("\nSample movies:")
        movies_df.show(5, truncate=False)
        print("\n")
        
        return movies_df
        
    except Exception as e:
        print(f"⚠ Warning: Could not load movies.csv: {str(e)}")
        print("  Continuing with movieId only...")
        print("\n")
        return None


def generate_top_recommendations(predicted_ratings, movies_df):
    """
    Generate top N recommendations
    
    Args:
        predicted_ratings: DataFrame with predicted ratings
        movies_df: DataFrame with movie metadata (can be None)
        
    Returns:
        DataFrame with top recommendations
    """
    print_section(f"GENERATING TOP {TOP_N_RECOMMENDATIONS} RECOMMENDATIONS")
    
    # Get top N recommendations
    top_recommendations = predicted_ratings \
        .orderBy(col("predicted_rating").desc(), col("num_raters").desc()) \
        .limit(TOP_N_RECOMMENDATIONS)
    
    # Join with movie names if available
    if movies_df is not None:
        recommendations_with_metadata = top_recommendations \
            .join(movies_df, "movieId", "left") \
            .select(
                "movieId",
                "title",
                "genres",
                "predicted_rating",
                "num_raters",
                "avg_similarity"
            )
    else:
        recommendations_with_metadata = top_recommendations
    
    return recommendations_with_metadata


def display_recommendations(recommendations_with_metadata, movies_df):
    """
    Display recommendations in formatted output
    
    Args:
        recommendations_with_metadata: DataFrame with recommendations
        movies_df: DataFrame with movie metadata (can be None)
    """
    print("\n")
    print("="*80)
    print("📸 SCREENSHOT 3: FINAL TOP RECOMMENDATIONS")
    print("="*80)
    print(f"\n🎬 TOP {TOP_N_RECOMMENDATIONS} MOVIE RECOMMENDATIONS FOR USER {TARGET_USER_ID}")
    print("="*80)
    print("\n")
    
    recommendations_list = recommendations_with_metadata.collect()
    
    for idx, row in enumerate(recommendations_list, 1):
        print(f"Rank #{idx}")
        print("-" * 60)
        if movies_df is not None and row['title'] is not None:
            print(f"  🎬 Movie: {row['title']}")
            if 'genres' in row and row['genres'] is not None:
                print(f"  🎭 Genres: {row['genres']}")
        print(f"  🆔 Movie ID: {row['movieId']}")
        print(f"  ⭐ Predicted Rating: {row['predicted_rating']:.2f} / 5.00")
        print(f"  👥 Based on {row['num_raters']} similar users")
        print(f"  🔗 Average Similarity Score: {row['avg_similarity']:.4f}")
        print()
    
    print("="*80)
    print("\n")
    
    # Show full table format
    print("Recommendations in Table Format:")
    recommendations_with_metadata.show(truncate=False)
    print("\n")


def show_target_user_favorites(ratings_clean, movies_df):
    """
    Show what the target user has already rated highly
    
    Args:
        ratings_clean: Cleaned ratings DataFrame
        movies_df: DataFrame with movie metadata (can be None)
    """
    print_section("TARGET USER'S FAVORITE MOVIES")
    print("(Movies already rated by the target user)")
    print("\n")
    
    target_favorites = ratings_clean \
        .filter(col("userId") == TARGET_USER_ID) \
        .orderBy(col("rating").desc())
    
    if movies_df is not None:
        target_favorites = target_favorites \
            .join(movies_df, "movieId", "left") \
            .select("movieId", "title", "genres", "rating")
        
        print("Top 10 Highest Rated Movies:")
        target_favorites.show(10, truncate=False)
    else:
        print("Top 10 Highest Rated Movies (Movie IDs only):")
        target_favorites.select("movieId", "rating").show(10, truncate=False)
    
    print("\n")


def main():
    """Main execution function"""
    # Create output directory
    create_output_directory()
    
    # Create Spark session
    spark = create_spark_session("GenerateRecommendations")
    
    try:
        print_header("PART 6: FINAL MOVIE RECOMMENDATIONS")
        
        # Step 1: Load prediction data
        ratings_clean, predicted_ratings = load_prediction_data(spark)
        
        # Step 2: Load movie metadata
        movies_df = load_movie_metadata(spark)
        
        # Step 3: Generate recommendations
        recommendations_with_metadata = generate_top_recommendations(
            predicted_ratings, 
            movies_df
        )
        
        # Step 4: Display recommendations
        display_recommendations(recommendations_with_metadata, movies_df)
        
        # Step 5: Show target user's favorites
        show_target_user_favorites(ratings_clean, movies_df)
        
        # Save results
        print_section("SAVING RESULTS")
        save_dataframe(recommendations_with_metadata, "final_recommendations.csv")
        
        # Summary
        print_header("RECOMMENDATION SYSTEM COMPLETE")
        
        print("✅ All tasks completed successfully!")
        print(f"📁 Output files saved to: ./output/")
        print(f"  - final_recommendations.csv")
        print("\n")
        
        print("📋 PDF REPORT CHECKLIST:")
        print("  ☐ Screenshot 1: User-User Similarity Matrix")
        print("  ☐ Screenshot 2: Sample Predicted Ratings")
        print("  ☐ Screenshot 3: Final Top Recommendations")
        print("  ☐ Include all code with comments")
        print("  ☐ Explain Pearson Correlation formula")
        print("  ☐ Explain Rating Prediction formula")
        print("\n")
        
    finally:
        # Stop Spark session
        spark.stop()
        print("✓ Spark session stopped\n")


if __name__ == "__main__":
    main()